"""
Mapping of laser power and depth for every .rec file under SPT_ROOT, with the lab
convention applied: a recording without a stated laser power (or depth) was made with
the setting of the recording before it. The .rec files are only read, never modified.

Inheritance is iterative within one folder: the recordings of a folder are ordered by
Record Date/Time; a recording that states the quantity in its Comment keeps it, any
other recording takes the effective value of its direct predecessor (which may itself
be inherited). The chain breaks at a recording with a relative statement without a
value (e.g. "etwas tiefer", "Laser erhöht") or with a depth mention without a value;
its successors get no value until a recording states one again. Copies of a recording
in several folders are mapped per folder.
Power is parsed with core.io.POWER_RE (as parse_rec_comment_metadata). Depth: "tiefe"
with a number before or after it (e.g. "tiefe 100", "Tiefe: 80µm", "110 tiefe",
"tiefe ca 30µm"); without unit the value is taken as µm.

Stage: compute stage. Reads all .rec files under SPT_ROOT on every run and
unconditionally overwrites
  cache/rec_metadata_inherited.pkl       DataFrame indexed by the resolved .rec path; read via
                                         core.io.load_inherited_rec_metadata()
  RESULT_DIR/rec_metadata_inheritance.xlsx  traceability: for every file the stated or inherited
                                         value, the direct predecessor it inherited from, the
                                         recording that originally stated it and the time gap
Consumers: Laser_Intensity/spt_irradiance_overview.py.
"""
import contextlib
import io
import re
from pathlib import Path

import pandas as pd

from hydro_analysis.core.io import DATETIME_RE, POWER_RE, find_rec_tif_files, scan_xml_folder

# ── Configuration ──────────────────────────────────────────────────────────────
SPT_ROOT = Path(r"E:\PhD Data Analysis\SPT 2025 II")
CACHE_PKL = Path(__file__).resolve().parent / "cache" / "rec_metadata_inherited.pkl"
RESULT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder"
                  r"\Experiments and Results - Data\_scratch\REC_Metadata_Inheritance")
LOG_XLSX = RESULT_DIR / "rec_metadata_inheritance.xlsx"

DEPTH_RE = re.compile(r"tiefe\b[^\d\r\n,|]{0,25}(\d+(?:[.,]\d+)?)\s*(mm|[µμu]m)?"
                      r"|(\d+(?:[.,]\d+)?)\s*(mm|[µμu]m)?\s*tiefe\b", re.IGNORECASE)
DEPTH_MENTION_RE = re.compile(r"\btiefe\b|\bdepth\b", re.IGNORECASE)
RELATIVE_RE = {
    "power": re.compile(r"laser\w*\s+(?:erhöht|reduziert|verringert|increased|decreased|lowered|raised)"
                        r"|(?:erhöht|reduziert|increased|decreased)\w*\s+laser", re.IGNORECASE),
    "depth": re.compile(r"\btiefer\b|\bflacher\b|\bdeeper\b|\bshallower\b", re.IGNORECASE),
}


def read_rec(path: Path) -> dict:
    raw = path.read_bytes()
    text = raw.decode("utf-16") if raw[:2] == b"\xff\xfe" else raw.decode("latin-1")
    dt = DATETIME_RE.search(text)
    comment = text.partition("Comment:")[2]
    return {"path": path.resolve(), "folder": path.parent, "comment": comment,
            "time": pd.to_datetime(f"{dt.group(1)} {dt.group(2)}", dayfirst=True) if dt else pd.NaT}


def stated_power(comment: str) -> tuple[str, float] | None:
    m = POWER_RE.search(comment)
    if not m:
        return None
    value = float(m.group(1).replace(",", "."))
    return m.group(0).strip(), value * 1e-3 if m.group(2)[0].lower() in ("µ", "μ", "u") else value


def stated_depth(comment: str) -> tuple[str, float] | None:
    m = DEPTH_RE.search(comment)
    if not m:
        return None
    number, unit = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
    value = float(number.replace(",", "."))
    return m.group(0).strip(), value * 1000.0 if unit and unit.lower() == "mm" else value


def map_folder(series: list[dict]) -> list[dict]:
    """Iterative inheritance along the chronologically ordered recordings of one folder."""
    rows, carried = [], {"power": None, "depth": None}
    for i, rec in enumerate(series):
        row = {"rec_path": str(rec["path"]), "folder": str(rec["folder"].relative_to(SPT_ROOT)),
               "file": rec["path"].name, "recorded_at": rec["time"],
               "comment": " | ".join(line.strip() for line in rec["comment"].splitlines() if line.strip())}
        for q, parse in (("power", stated_power), ("depth", stated_depth)):
            found = parse(rec["comment"])
            info = {"text": None, "value": None, "status": None, "inherited_from": None, "origin": None,
                    "origin_time": pd.NaT}
            if found:
                info.update(text=found[0], value=found[1], status="stated", origin=rec["path"].name,
                            origin_time=rec["time"])
                carried[q] = info
            elif RELATIVE_RE[q].search(rec["comment"]):
                info["status"] = "unknown: relative statement without value"
                carried[q] = None
            elif q == "depth" and DEPTH_MENTION_RE.search(rec["comment"]):
                info["status"] = "unknown: mentioned without value"
                carried[q] = None
            elif carried[q] is not None:
                info.update({k: carried[q][k] for k in ("text", "value", "origin", "origin_time")},
                            status="inherited", inherited_from=series[i - 1]["path"].name)
                carried[q] = info
            else:
                info["status"] = "unknown: predecessor has no value" if i else "unknown: first recording in the folder"
            unit = "laser_power_mW" if q == "power" else "depth_um"
            row.update({f"{q}_status": info["status"], f"{q}_text": info["text"], unit: info["value"],
                        f"{q}_inherited_from": info["inherited_from"], f"{q}_origin": info["origin"],
                        f"{q}_origin_time": info["origin_time"],
                        f"{q}_gap_to_origin_min": round((rec["time"] - info["origin_time"]).total_seconds() / 60.0, 1)
                        if info["status"] == "inherited" else None,
                        f"{q}_origin_other_day": bool(info["status"] == "inherited"
                                                      and info["origin_time"].date() != rec["time"].date())})
        rows.append(row)
    return rows


def paired_rec_paths() -> set[str]:
    paths = set()
    for xml_path in scan_xml_folder(SPT_ROOT):
        # find_rec_tif_files prints an error for every XML without .rec.
        with contextlib.redirect_stdout(io.StringIO()):
            rec = find_rec_tif_files(xml_path)["rec_file"]
        if rec is not None:
            paths.add(str(rec.resolve()))
    return paths


def main() -> None:
    recs = [read_rec(p) for p in sorted(SPT_ROOT.rglob("*.rec"))]
    rows = []
    for folder in sorted({r["folder"] for r in recs}):
        series = sorted((r for r in recs if r["folder"] == folder),
                        key=lambda r: (pd.isna(r["time"]), r["time"] if pd.notna(r["time"]) else pd.Timestamp(0)))
        rows += map_folder(series)
    table = pd.DataFrame(rows)
    table.insert(4, "paired_with_xml", table["rec_path"].isin(paired_rec_paths()))
    table = table.set_index("rec_path")

    CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)
    table.to_pickle(CACHE_PKL)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    table.sort_values(["folder", "recorded_at"]).to_excel(LOG_XLSX)

    spt = table[table["paired_with_xml"]]
    print(f"{len(table)} .rec files mapped, {len(spt)} paired with an XML (SPT measurements)")
    for q in ("power", "depth"):
        print(f"\n{q} (SPT measurements):")
        print(spt[f"{q}_status"].value_counts().to_string())
    print(f"\nCache: {CACHE_PKL}\nLog:   {LOG_XLSX}")


if __name__ == "__main__":
    main()
