"""
Inventory of all Anton Paar LiteSizer 500 measurements in the LiteSizer data folder, and an
extended DLS size table (reference workbook plus the measurements that are missing from it).

Sources, all read-only (DATA_DIR, scanned recursively):
  *.apkw  Kalliope project files: zip archives of BSON documents, decoded with the minimal BSON
          reader below. Series containers carry no result and are not counted as measurements.
  *.xlsx  Kalliope workbook exports, one sheet per measurement. Units are taken from the unit
          column of every sheet; an unknown unit stops the run.
  *.csv   Kalliope single-measurement summaries (semicolon separated, without units). The start
          time is taken from the file name, which Kalliope writes as <name>_<start>_<workbook>.csv.
Every record is keyed by its local start time (LOCAL_TZ, to the second; .apkw stores UTC).
Records with the same start time are one physical measurement exported several times, or a
recalculation of it; they become one row of the overview that lists every source file.

CSV units: the factor from CSV value to nm (diameter) and to mV (zeta potential) is derived from
the CSV records that also exist in an .apkw or .xlsx export. The run stops if these cross-checks
do not agree on one factor. CSV-only records use the same factor and are flagged; a CSV-only
diameter above INSTRUMENT_MAX_DIAMETER_NM after conversion is left blank and flagged as unresolved.

Polydispersity: .apkw stores the PDI with a sign, all exports show the absolute value. Both are
kept; the PDI criterion uses the exported (absolute) value, as litesizer_visualization.py does.

Extended size table: every row of REFERENCE_XLSX unchanged (including rows above MAX_PDI, which
the downstream scripts filter themselves), plus every DLS measurement that is not in
REFERENCE_XLSX, has a result and has PDI <= MAX_PDI. Temperature, solvent model, quality mode
and name inconsistencies are reported as columns and flags, not used to drop rows (user decision
2026-10-06). Zeta-potential measurements appear in the overview only.
The instrument "solvent" is the solvent model used for viscosity and refractive index; it is not
necessarily the experimental medium (cf. BUFFER in litesizer_zeta_potential.py).

Writes to OUTPUT_DIR (nothing is written beside the raw data):
  dls_measurement_inventory.xlsx             sheets overview, per_day, size_table_extended,
                                             added, source_records, files, exclusions
  dls_size_table_extended.csv
  dls_measurement_inventory_exclusions.csv
Reads/writes no pickle cache. No other script reads these outputs; litesizer_visualization.py
and the other Litesizer scripts still read REFERENCE_XLSX through litesizer_parser.py.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import math
import re
import struct
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import openpyxl
import pandas as pd


# ── Configuration ──────────────────────────────────────────────────────────────
DATA_DIR = Path(r"E:\Daten Promotion Sicherung\Lite Sizer Particle Measurements")
REFERENCE_XLSX_NAME = "Size_repitition_All Sizes.xlsx"        # the DLS table read by Litesizer/*
REFERENCE_ZETA_XLSX_NAME = "Zeta_repitition_allparticles.xlsx"  # read by litesizer_zeta_potential.py
OUTPUT_DIR = Path(
    r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data"
) / "Particle_Characterization" / "DLS_measurement_inventory"
STEM = "dls_measurement_inventory"
LOCAL_TZ = ZoneInfo("Europe/Berlin")  # .apkw start times are UTC; file names and xlsx use local time

# Criterion requested by user, 2026-10-06; the value is the one used in Litesizer/* (E007, still open).
MAX_PDI = 42.0
# decided by user, 2026-10-06: measurements found only in .apkw/.xlsx (no CSV) are added as well.
INCLUDE_NON_CSV = True
# PROPOSED, not approved: upper end of the Litesizer 500 size range (10 µm, manufacturer
# specification). Used only to flag a CSV-only diameter whose unit cannot be confirmed.
INSTRUMENT_MAX_DIAMETER_NM = 10_000.0

SOURCE_PRIORITY = {"reference_xlsx": 0, "apkw": 1, "xlsx": 2, "csv": 3}
APKW_QUALITY_MODE = {0: "Automatic", 1: "Quick"}  # agreement with the exports is checked per measurement
REL_TOL = 1e-4                                     # tolerance for "same value in two exports"
DIAMETER_UNITS_NM = {"nm": 1.0, "µm": 1e3, "μm": 1e3}
ZETA_UNITS_MV = {"mV": 1.0, "V": 1e3}
NOT_PARSED = {".pdf": "Kalliope report (PDF), not parsed",
              ".png": "derived figure, not data",
              ".txt": "derived results text, not data"}
COMPARED_FIELDS = ["dh_nm", "pdi_pct", "D_um2_s", "zeta_mV", "T_C", "transmittance_pct", "quality_mode"]


# ── Minimal BSON reader (.apkw) ────────────────────────────────────────────────
def _bson_doc(b: bytes, i: int = 0):
    size = struct.unpack_from("<i", b, i)[0]
    end, i, doc = i + size, i + 4, {}
    while i < end - 1:
        kind = b[i]
        j = b.index(b"\x00", i + 1)
        key, i = b[i + 1:j].decode("utf-8", "replace"), j + 1
        if kind == 0x01:
            val = struct.unpack_from("<d", b, i)[0]; i += 8
        elif kind == 0x02:
            n = struct.unpack_from("<i", b, i)[0]
            val = b[i + 4:i + 3 + n].decode("utf-8", "replace"); i += 4 + n
        elif kind in (0x03, 0x04):
            val, i = _bson_doc(b, i)
            if kind == 0x04:
                val = list(val.values())
        elif kind == 0x05:
            n = struct.unpack_from("<i", b, i)[0]
            val = b[i + 5:i + 5 + n]; i += 5 + n
        elif kind == 0x07:
            val = b[i:i + 12].hex(); i += 12
        elif kind == 0x08:
            val = bool(b[i]); i += 1
        elif kind == 0x09:
            ms = struct.unpack_from("<q", b, i)[0]; i += 8
            val = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(milliseconds=ms)
        elif kind == 0x0A:
            val = None
        elif kind == 0x10:
            val = struct.unpack_from("<i", b, i)[0]; i += 4
        elif kind in (0x11, 0x12):
            val = struct.unpack_from("<q", b, i)[0]; i += 8
        else:
            raise ValueError(f"Unsupported BSON element type 0x{kind:02x} at byte {i}.")
        doc[key] = val
    return doc, end


# ── Helpers ────────────────────────────────────────────────────────────────────
def _num(value) -> float:
    if value is None or value == "":
        return math.nan
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "nan":
            return math.nan
    return float(value)


def _norm_name(name) -> str:
    return re.sub(r"\s+", " ", str(name or "")).strip()


def _is_recalc(name: str) -> bool:
    return _norm_name(name).lower().startswith("recalculation of")


def name_size_nm(name: str) -> float:
    """Diameter written in a measurement name ("20 nm 1", "1µm 20°C", "0.04µm"); NaN if none."""
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(nm|µm|μm)(?![a-z])", _norm_name(name), re.IGNORECASE)
    if not m:
        return math.nan
    value = float(m[1].replace(",", "."))
    return value * (1e3 if m[2].lower() in ("µm", "μm") else 1.0)


def _record(**kw) -> dict:
    base = dict(source_type="", source_file="", source_part="", kalliope_workbook="", name="",
                mode="", start_local="", is_recalc=False, dh_nm=math.nan, pdi_pct=math.nan,
                pdi_signed_apkw=math.nan, D_um2_s=math.nan, zeta_mV=math.nan,
                conductivity_mS_cm=math.nan, T_C=math.nan, solvent_model="", material="",
                quality_mode="", processed_runs=math.nan, transmittance_pct=math.nan,
                mean_intensity_kcps=math.nan, comment="", csv_raw_value=math.nan, unit_note="")
    base.update(kw)
    return base


# ── Readers ────────────────────────────────────────────────────────────────────
def read_apkw(path: Path) -> tuple[list[dict], int]:
    """All result-carrying measurements of one Kalliope project; also returns the series count."""
    records, n_series = [], 0
    with zipfile.ZipFile(path) as zf:
        for part in sorted(n for n in zf.namelist() if n.startswith("measurements/")):
            doc, _ = _bson_doc(zf.read(part))
            kind = doc["$type"].split(",")[0].rsplit(".", 1)[-1]
            if "Series" in kind:
                n_series += 1
                continue
            ip, meta = doc.get("InputParameter") or {}, doc.get("MetaInfo") or {}
            if not meta.get("StartTime"):
                raise ValueError(f"{path.name}:{part}: no StartTime in MetaInfo.")
            start = dt.datetime.fromisoformat(str(meta["StartTime"]))
            start = start.replace(tzinfo=dt.timezone.utc) if start.tzinfo is None else start
            name = _norm_name(ip.get("Name"))
            rec = _record(source_type="apkw", source_file=path.name, source_part=part,
                          kalliope_workbook=path.name, name=name,
                          start_local=start.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                          is_recalc=_is_recalc(name) or bool(meta.get("Recalculations")),
                          T_C=_num(ip.get("TargetTemperature")) - 273.15,
                          solvent_model=(ip.get("Solvent") or {}).get("Name") or "",
                          material=(ip.get("Material") or {}).get("Name") or "",
                          processed_runs=_num(doc.get("ProcessedRuns")),
                          transmittance_pct=_num(doc.get("Transmission")) * 100.0,
                          comment=_norm_name(ip.get("Comment")))
            if kind.startswith("Dls"):
                run = doc.get("RecentRun")
                rec["mode"] = "size"
                rec["quality_mode"] = APKW_QUALITY_MODE.get(ip.get("QualityMode"), f"code {ip.get('QualityMode')}")
                if isinstance(run, dict):
                    pdi = _num(run.get("PolydispersityIndex"))
                    rec.update(dh_nm=2.0 * _num(run.get("MeanHydroDynamicRadiusSI")) * 1e9,
                               pdi_signed_apkw=pdi, pdi_pct=abs(pdi),
                               D_um2_s=_num(run.get("DiffusionCoefficient")))
            elif kind.startswith("Els"):
                rec.update(mode="zeta", zeta_mV=_num(doc.get("ZetaMean")) * 1e3,
                           conductivity_mS_cm=_num(doc.get("Conductivity")))
            else:
                raise ValueError(f"{path.name}:{part}: unknown measurement type {kind}.")
            records.append(rec)
    return records, n_series


def _sheet_fields(rows) -> tuple[dict, dict]:
    """Column A/B header block and {(subsection, label): (value, unit)} from columns B:D."""
    header, fields, sub = {}, {}, ""
    for row in rows:
        a, b, c, d = (list(row) + [None] * 4)[:4]
        if a is not None and b is not None and c is None and d is None and str(a) in (
                "Workbook name", "Measurement name", "Measurement mode", "Comment"):
            header[str(a)] = b
            continue
        if a == "Comment":
            header["Comment"] = b
        if b is None:
            continue
        if c is None and d is None:
            sub = str(b)
        fields.setdefault((sub, str(b)), (c, d))
        fields.setdefault(("", str(b)), (c, d))
    return header, fields


def read_xlsx(path: Path, source_type: str) -> list[dict]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    records = []
    try:
        for ws in wb.worksheets:
            header, f = _sheet_fields(ws.iter_rows(max_col=4, values_only=True))
            if "Measurement name" not in header:
                continue  # analysis overview sheet
            where = f"{path.name}[{ws.title}]"

            def value(label, sub=""):
                return f.get((sub, label), (None, None))

            mode_txt = str(header.get("Measurement mode") or "").lower()
            start_txt = value("Start time")[0]
            if not start_txt:
                raise ValueError(f"{where}: no 'Start time' field.")
            start = dt.datetime.strptime(str(start_txt).strip(), "%m/%d/%Y %I:%M:%S %p")
            temp, temp_unit = value("Target temperature")
            if temp is not None and temp_unit != "°C":
                raise ValueError(f"{where}: unsupported temperature unit {temp_unit!r}.")
            name = _norm_name(header["Measurement name"])
            rec = _record(source_type=source_type, source_file=path.name, source_part=ws.title,
                          kalliope_workbook=Path(str(header.get("Workbook name") or "").replace("\\", "/")).name,
                          name=name, start_local=start.strftime("%Y-%m-%d %H:%M:%S"),
                          is_recalc=_is_recalc(name), T_C=_num(temp),
                          solvent_model=str(value("Name", "Solvent")[0] or ""),
                          material=str(value("Name", "Material")[0] or ""),
                          processed_runs=_num(value("Processed runs")[0]),
                          transmittance_pct=_num(value("Transmittance")[0]),
                          mean_intensity_kcps=_num(value("Mean intensity")[0]),
                          comment=_norm_name(header.get("Comment")))
            if mode_txt == "particle size":
                dh, dh_unit = value("Hydrodynamic diameter")
                if dh is not None and dh_unit not in DIAMETER_UNITS_NM:
                    raise ValueError(f"{where}: unsupported diameter unit {dh_unit!r}.")
                d_val, d_unit = value("Diffusion coefficient")
                if d_val is not None and d_unit != "µm²/s":
                    raise ValueError(f"{where}: unsupported diffusion unit {d_unit!r}.")
                rec.update(mode="size", quality_mode=str(value("Mode", "Quality")[0] or ""),
                           dh_nm=_num(dh) * DIAMETER_UNITS_NM.get(dh_unit, math.nan) if dh is not None else math.nan,
                           pdi_pct=_num(value("Polydispersity")[0]), D_um2_s=_num(d_val))
            elif mode_txt == "zeta potential":
                z, z_unit = value("Mean zeta potential")
                if z_unit not in ZETA_UNITS_MV:
                    raise ValueError(f"{where}: unsupported zeta unit {z_unit!r}.")
                c_val, c_unit = value("Conductivity")
                if c_val is not None and c_unit != "mS/cm":
                    raise ValueError(f"{where}: unsupported conductivity unit {c_unit!r}.")
                rec.update(mode="zeta", zeta_mV=_num(z) * ZETA_UNITS_MV[z_unit],
                           conductivity_mS_cm=_num(c_val),
                           quality_mode=str(value("Run mode")[0] or ""))
            else:
                raise ValueError(f"{where}: unknown measurement mode {mode_txt!r}.")
            records.append(rec)
    finally:
        wb.close()
    return records


def read_csv(path: Path) -> list[dict]:
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = raw.decode("cp1252"), "cp1252"
    m = re.search(r"_(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})_(.*)\.csv$", path.name)
    if not m:
        raise ValueError(f"{path.name}: no start time of the form _YYYY-MM-DD-hh-mm-ss_ in the file name.")
    start = dt.datetime.strptime(m[1], "%Y-%m-%d-%H-%M-%S").strftime("%Y-%m-%d %H:%M:%S")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = lines[0].rstrip(";").split(";")
    records = []
    for k, line in enumerate(lines[1:], start=1):
        row = dict(zip(head, line.rstrip(";").split(";")))
        name = _norm_name(row.get("Measurement name"))
        mode_txt = row.get("Measurement mode", "").lower()
        rec = _record(source_type="csv", source_file=path.name, source_part=f"row {k} ({encoding})",
                      kalliope_workbook=m[2], name=name, start_local=start, is_recalc=_is_recalc(name),
                      transmittance_pct=_num(row.get("Transmittance")),
                      mean_intensity_kcps=_num(row.get("Mean intensity")))
        if mode_txt == "particle size":
            rec.update(mode="size", csv_raw_value=_num(row.get("Hydrodynamic diameter")),
                       pdi_pct=_num(row.get("Polydispersity")), quality_mode=row.get("Mode", ""))
        elif mode_txt == "zeta potential":
            rec.update(mode="zeta", csv_raw_value=_num(row.get("Mean zeta potential")),
                       conductivity_mS_cm=_num(row.get("Conductivity")))
        else:
            raise ValueError(f"{path.name}: unknown measurement mode {mode_txt!r}.")
        records.append(rec)
    return records


# ── CSV unit factor from cross-checks ──────────────────────────────────────────
def resolve_csv_units(rec: pd.DataFrame, max_diameter_nm: float) -> tuple[pd.DataFrame, dict]:
    """Apply the CSV->nm and CSV->mV factor that the CSV/export pairs agree on."""
    rec = rec.copy()
    factors = {}
    for mode, target in (("size", "dh_nm"), ("zeta", "zeta_mV")):
        csv = rec[(rec.source_type == "csv") & (rec["mode"] == mode)]
        other = rec[(rec.source_type != "csv") & (rec["mode"] == mode) & rec[target].notna()]
        ratios = []
        for idx, r in csv.iterrows():
            same = other[(other.start_local == r.start_local) & (other.is_recalc == r.is_recalc)]
            if r.is_recalc:
                same = same[same.name == r["name"]]
            for _, o in same.iterrows():
                if r.csv_raw_value and not math.isnan(r.csv_raw_value):
                    ratios.append((o[target] / r.csv_raw_value, r.source_file, o.source_file))
        if not ratios:
            raise ValueError(f"No CSV {mode} record has a counterpart in .apkw/.xlsx; the CSV unit cannot be checked.")
        values = np.array([x[0] for x in ratios])
        factor = float(10 ** np.round(np.log10(np.median(values))))
        bad = [x for x in ratios if abs(x[0] / factor - 1) > REL_TOL]
        if bad:
            raise ValueError(f"CSV {mode} unit cross-check disagrees for: {bad}")
        factors[mode] = (factor, len(ratios))
        matched_starts = set(other.start_local)
        for idx, r in csv.iterrows():
            converted = r.csv_raw_value * factor
            confirmed = r.start_local in matched_starts
            note = ("unit confirmed by export of the same measurement" if confirmed
                    else f"unit taken from {len(ratios)} cross-checked CSVs")
            if mode == "size" and not confirmed and converted > max_diameter_nm:
                converted = math.nan
                note = (f"unit unresolved: raw value {r.csv_raw_value} x {factor:g} exceeds "
                        f"INSTRUMENT_MAX_DIAMETER_NM")
            rec.at[idx, target] = converted
            rec.at[idx, "unit_note"] = note
    return rec, factors


# ── Consolidation ──────────────────────────────────────────────────────────────
def _differs(a, b) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return bool(a) and bool(b) and a != b
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) > REL_TOL * max(abs(a), abs(b), 1e-12)


def consolidate(rec: pd.DataFrame, max_pdi: float, include_non_csv: bool):
    """One row per physical measurement (start time) plus the exclusion rows it implies."""
    rec = rec.assign(_prio=rec.source_type.map(SOURCE_PRIORITY))
    rows, excl = [], []
    for k, (start, grp) in enumerate(sorted(rec.groupby("start_local"), key=lambda x: x[0]), start=1):
        mid = f"M{k:03d}"
        modes = set(grp["mode"])
        if len(modes) != 1:
            raise ValueError(f"Start time {start}: records of different modes {modes}.")
        grp = grp.sort_values(["is_recalc", "_prio"])
        primary = grp.iloc[0]
        originals = grp[~grp.is_recalc] if (~grp.is_recalc).any() else grp
        flags = []
        row = {"measurement_id": mid, "start_local": start, "date": start[:10], "time": start[11:],
               "mode": primary["mode"], "name": primary["name"],
               "all_names": " | ".join(dict.fromkeys(grp["name"]))}
        sizes = {s for s in (name_size_nm(n) for n in originals["name"]) if not math.isnan(s)}
        if len(sizes) == 1:
            row["name_size_nm"] = sizes.pop()
        else:
            row["name_size_nm"] = math.nan
            flags.append("names give different sizes" if sizes else "no size in name")
        if len({_norm_name(n).lower() for n in originals["name"]}) > 1:
            flags.append("name differs between exports")
        for field in ["dh_nm", "pdi_pct", "pdi_signed_apkw", "D_um2_s", "zeta_mV", "conductivity_mS_cm",
                      "T_C", "solvent_model", "material", "quality_mode", "processed_runs",
                      "transmittance_pct", "mean_intensity_kcps", "comment"]:
            val = primary[field]
            if (isinstance(val, float) and math.isnan(val)) or val == "":
                for _, o in originals.iterrows():
                    if not ((isinstance(o[field], float) and math.isnan(o[field])) or o[field] == ""):
                        val = o[field]
                        break
            row[field] = val
        for field in COMPARED_FIELDS:
            vals = list(originals[field])
            if any(_differs(a, b) for a in vals for b in vals):
                flags.append(f"{field} differs between exports")
        recalcs = grp[grp.is_recalc & (grp.index != primary.name)]
        if len(recalcs) and primary["mode"] == "size":
            flags.append("recalculated d_H (nm): " + ", ".join(f"{v:.2f}" for v in recalcs.dh_nm))
        notes = [n for n in grp.unit_note if n]
        if any(n.startswith("unit unresolved") for n in notes):
            flags.append("CSV unit unresolved")
        elif any(n.startswith("unit taken") for n in notes) and not (grp.source_type != "csv").any():
            flags.append("CSV only: unit from cross-check")
        if row["mode"] == "size" and isinstance(row["pdi_signed_apkw"], float) and row["pdi_signed_apkw"] < 0:
            flags.append("PDI stored negative in .apkw")
        if row["mode"] == "size" and math.isnan(row["T_C"]):
            flags.append("temperature unknown (CSV only)")
        in_ref = bool((grp.source_type == "reference_xlsx").any())
        has_csv = bool((grp.source_type == "csv").any())
        row.update(in_reference_size_table=in_ref,
                   in_reference_zeta_table=bool((grp.source_file == REFERENCE_ZETA_XLSX_NAME).any()),
                   has_csv=has_csv, n_records=len(grp),
                   sources=" ; ".join(f"{r.source_file} [{r.source_part}]" for _, r in grp.iterrows()))

        if row["mode"] != "size":
            status = "zeta potential, not part of the size table"
        elif in_ref:
            status = "in reference table"
        elif row["pdi_pct"] > max_pdi:
            status, step, param = "not added: PDI", "PDI criterion", f"MAX_PDI={max_pdi:g}"
        elif math.isnan(row["dh_nm"]) or math.isnan(row["pdi_pct"]):
            status, step, param = "not added: no result", "no result", "d_H or PDI missing"
        elif not has_csv and not include_non_csv:
            status, step, param = "not added: no CSV", "source criterion", "INCLUDE_NON_CSV=False"
        else:
            status = "added"
        row["size_table_status"] = status
        row["flags"] = "; ".join(flags)
        rows.append(row)

        for _, r in grp.iterrows():
            src = f"{r.source_file} [{r.source_part}]"
            if r.name != primary.name:
                excl.append(dict(file=src, step="duplicate export", parameter=f"start {start}",
                                 n_before=1, n_after=0,
                                 reason=f"{mid}: same start time as {primary.source_file}"
                                        + (" (recalculation)" if r.is_recalc else "")))
            elif row["mode"] != "size":
                excl.append(dict(file=src, step="mode", parameter="zeta potential", n_before=1, n_after=0,
                                 reason=f"{mid}: zeta potential is not part of the size table (user decision 2026-10-06)"))
            elif status.startswith("not added"):
                val = f"PDI {row['pdi_pct']:.1f} %" if step == "PDI criterion" else param
                excl.append(dict(file=src, step=step, parameter=param, n_before=1, n_after=0,
                                 reason=f"{mid} '{row['name']}': {val}"))
    return pd.DataFrame(rows), pd.DataFrame(excl, columns=["file", "step", "parameter", "n_before", "n_after", "reason"])


# ── Tables ─────────────────────────────────────────────────────────────────────
TABLE_COLUMNS = ["origin", "measurement_id", "start_local", "name", "name_size_nm", "dh_nm", "pdi_pct",
                 "pdi_signed_apkw", "pdi_ok", "D_um2_s", "T_C", "solvent_model", "material", "quality_mode",
                 "processed_runs", "transmittance_pct", "mean_intensity_kcps", "comment", "has_csv",
                 "flags", "sources"]


def size_table(overview: pd.DataFrame, max_pdi: float) -> pd.DataFrame:
    sel = overview[overview.size_table_status.isin(["in reference table", "added"])].copy()
    sel["origin"] = np.where(sel.size_table_status == "added", "added", REFERENCE_XLSX_NAME)
    sel["pdi_ok"] = sel.pdi_pct <= max_pdi
    sel["_added"] = sel.origin == "added"
    return sel.sort_values(["_added", "start_local"])[TABLE_COLUMNS].reset_index(drop=True)


def per_day(overview: pd.DataFrame) -> pd.DataFrame:
    out = []
    for date, g in overview.groupby("date"):
        s, z = g[g["mode"] == "size"], g[g["mode"] == "zeta"]
        out.append({
            "date": date,
            "first_start": g.time.min(), "last_start": g.time.max(),
            "n_size": len(s), "n_zeta": len(z),
            "size_names": ", ".join(dict.fromkeys(s["name"])),
            "zeta_names": ", ".join(dict.fromkeys(z["name"])),
            "T_C": ", ".join(sorted({f"{t:g}" for t in g.T_C if not math.isnan(t)})) or "unknown",
            "solvent_model": ", ".join(sorted({x for x in g.solvent_model if x})) or "unknown",
            "dh_nm_range": (f"{s.dh_nm.min():.1f} – {s.dh_nm.max():.1f}" if s.dh_nm.notna().any() else ""),
            "zeta_mV_range": (f"{z.zeta_mV.min():.1f} – {z.zeta_mV.max():.1f}" if z.zeta_mV.notna().any() else ""),
            "n_in_reference_table": int(s.in_reference_size_table.sum()),
            "n_added": int((s.size_table_status == "added").sum()),
            "n_not_added": int(s.size_table_status.str.startswith("not added").sum()),
        })
    return pd.DataFrame(out)


# ── Self-test with synthetic records ───────────────────────────────────────────
def _selftest() -> None:
    """Synthetic records with a known answer; nothing is written to disk."""
    t1, t2, t3, t4 = "2030-01-01 10:00:00", "2030-01-01 10:05:00", "2030-01-01 10:10:00", "2030-01-01 10:20:00"
    recs = pd.DataFrame([
        _record(source_type="reference_xlsx", source_file="ref", name="50 nm 1", mode="size", start_local=t1,
                dh_nm=50.0, pdi_pct=15.0, T_C=20.0),
        _record(source_type="csv", source_file="a.csv", name="50 nm 1", mode="size", start_local=t1,
                csv_raw_value=0.050, pdi_pct=15.0),
        _record(source_type="apkw", source_file="p", name="Untitled 3", mode="size", start_local=t2,
                dh_nm=200.0, pdi_pct=5.0, pdi_signed_apkw=-5.0, T_C=25.0),
        _record(source_type="csv", source_file="b.csv", name="Untitled 3", mode="size", start_local=t2,
                csv_raw_value=0.200, pdi_pct=5.0),
        _record(source_type="csv", source_file="c.csv", name="Recalculation of Untitled 3", mode="size",
                start_local=t2, is_recalc=True, csv_raw_value=0.201, pdi_pct=5.0),
        _record(source_type="csv", source_file="d.csv", name="500 nm", mode="size", start_local=t3,
                csv_raw_value=0.600, pdi_pct=90.0),
        _record(source_type="csv", source_file="e.csv", name="1 µm", mode="size", start_local=t4,
                csv_raw_value=250.0, pdi_pct=30.0),  # 250 µm: outside the instrument range
        _record(source_type="xlsx", source_file="z", name="20 nm", mode="zeta", start_local=t3[:-2] + "30",
                zeta_mV=-37.0),
        _record(source_type="csv", source_file="z.csv", name="20 nm", mode="zeta", start_local=t3[:-2] + "30",
                csv_raw_value=-0.037),
    ])
    recs, factors = resolve_csv_units(recs, INSTRUMENT_MAX_DIAMETER_NM)
    ov, ex = consolidate(recs, MAX_PDI, True)
    st = dict(zip(ov.start_local, ov.size_table_status))
    checks = {
        "CSV diameter factor": (factors["size"][0], 1000.0),
        "CSV zeta factor": (factors["zeta"][0], 1000.0),
        "unique measurements": (len(ov), 5),
        "reference row kept": (st[t1], "in reference table"),
        "apkw+csv+recalc merged and added": (st[t2], "added"),
        "PDI 90 % not added": (st[t3], "not added: PDI"),
        "250 µm CSV-only unresolved": (st[t4], "not added: no result"),
        "duplicate exports logged": (int((ex.step == "duplicate export").sum()), 4),
        "name size of 'Untitled 3'": (str(ov.loc[ov.start_local == t2, "name_size_nm"].iloc[0]), "nan"),
    }
    failed = {k: v for k, v in checks.items() if v[0] != v[1]}
    for k, (got, want) in checks.items():
        print(f"  self-test {k}: recovered {got}, expected {want}")
    if failed:
        raise AssertionError(f"Self-test failed: {failed}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Self-test on synthetic records:")
    _selftest()

    files = sorted(p for p in DATA_DIR.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No files in {DATA_DIR}.")
    apkw_hash = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.suffix == ".apkw"}
    records, file_rows, excl = [], [], []
    for p in files:
        rel = str(p.relative_to(DATA_DIR))
        if p.name.startswith("~$"):
            status, n = "skipped: Excel lock file (workbook open in Excel)", 0
        elif p.suffix == ".backup" and p.name.endswith(".apkw.backup"):
            orig = p.name[:-len(".backup")]
            same = apkw_hash.get(orig) == hashlib.sha256(p.read_bytes()).hexdigest()
            if same:
                status, n = f"skipped: byte-identical copy of {orig}", 0
            else:
                recs, n_series = read_apkw(p)
                records += recs
                status, n = f"parsed (differs from {orig}; {n_series} series containers)", len(recs)
        elif p.suffix == ".apkw":
            recs, n_series = read_apkw(p)
            records += recs
            status, n = f"parsed ({n_series} series containers without result not counted)", len(recs)
        elif p.suffix == ".xlsx":
            kind = "reference_xlsx" if p.name == REFERENCE_XLSX_NAME else "xlsx"
            recs = read_xlsx(p, kind)
            records += recs
            status, n = "parsed", len(recs)
        elif p.suffix == ".csv":
            recs = read_csv(p)
            records += recs
            status, n = "parsed", len(recs)
        elif p.suffix.lower() in NOT_PARSED:
            status, n = "skipped: " + NOT_PARSED[p.suffix.lower()], 0
        else:
            raise ValueError(f"{rel}: file type {p.suffix!r} is not handled; extend the inventory first.")
        file_rows.append(dict(file=rel, suffix=p.suffix, status=status, n_records=n))
        if n == 0:
            excl.append(dict(file=rel, step="file type", parameter=p.suffix, n_before=1, n_after=0, reason=status))

    rec = pd.DataFrame(records)
    rec, factors = resolve_csv_units(rec, INSTRUMENT_MAX_DIAMETER_NM)
    overview, excl_meas = consolidate(rec, MAX_PDI, INCLUDE_NON_CSV)
    exclusions = pd.concat([pd.DataFrame(excl), excl_meas], ignore_index=True)
    table = size_table(overview, MAX_PDI)
    days = per_day(overview)
    files_df = pd.DataFrame(file_rows)

    # Console summary
    print(f"\nFiles in {DATA_DIR}: {len(files)}; parsed {int((files_df.n_records > 0).sum())}; "
          f"skipped {int((files_df.n_records == 0).sum())}")
    print(f"Source records: {len(rec)} ({rec.source_type.value_counts().to_dict()})")
    for mode, (factor, n) in factors.items():
        print(f"CSV {mode} factor: {factor:g} (from {n} CSV/export pairs)")
    print(f"Unique measurements: {len(overview)} (size {int((overview['mode'] == 'size').sum())}, "
          f"zeta {int((overview['mode'] == 'zeta').sum())})")
    print(overview.size_table_status.value_counts().to_string())
    print(f"Extended size table: {len(table)} rows "
          f"({int((table.origin != 'added').sum())} reference + {int((table.origin == 'added').sum())} added)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec_out = rec.drop(columns=[c for c in rec.columns if c.startswith("_")])
    xlsx_path = OUTPUT_DIR / f"{STEM}.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
        overview.to_excel(xw, sheet_name="overview", index=False)
        days.to_excel(xw, sheet_name="per_day", index=False)
        table.to_excel(xw, sheet_name="size_table_extended", index=False)
        table[table.origin == "added"].to_excel(xw, sheet_name="added", index=False)
        rec_out.sort_values(["start_local", "source_type"]).to_excel(xw, sheet_name="source_records", index=False)
        files_df.to_excel(xw, sheet_name="files", index=False)
        exclusions.to_excel(xw, sheet_name="exclusions", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
    table.to_csv(OUTPUT_DIR / "dls_size_table_extended.csv", index=False, encoding="utf-8-sig")
    exclusions.to_csv(OUTPUT_DIR / f"{STEM}_exclusions.csv", index=False, encoding="utf-8-sig")
    print(f"\nWritten: {xlsx_path}\n         {OUTPUT_DIR / 'dls_size_table_extended.csv'}\n"
          f"         {OUTPUT_DIR / f'{STEM}_exclusions.csv'}")


if __name__ == "__main__":
    main()
