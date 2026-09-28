"""Replicate zeta-potential reference figure from a Kalliope XLSX export.

Reads the raw XLSX directly and recomputes the reference statistics on every run.
Reads/writes no pickle cache; no other script depends on these outputs.
Writes only a 600 dpi PNG and an Excel results workbook to the dissertation
figure directory configured as OUTPUT_DIR.
Run directly; paths, size selection and figure options are configured below.
One measurement sheet is one replicate, irrespective of its processed-run count.
The exported instrument SD and distribution peak are NOT replicate statistics.

Missing conditions remain blank. Configured conditions fill missing metadata
only and must describe every selected measurement. Different known conditions
are rejected rather than pooled. The instrument solvent model (e.g. Water)
is preserved separately and is never interpreted as the experimental buffer.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd
from matplotlib.ticker import AutoMinorLocator

from hydro_analysis.MSD_Trackmate.Validation_Claude.Correlations import SIZE_COLORS


# ── Configuration ──────────────────────────────────────────────────────────────
XLSX_PATH = Path(r"E:\Daten Promotion Sicherung\Lite Sizer Particle Measurements\Zeta_repitition_allparticles.xlsx")
OUTPUT_DIR = Path(r"E:\PhD Data Analysis\SPT 2025 II\Visualizations\PhD Dis Bilder\Experiments and Results - Data") / "Zeta_potential"
SIZES = None  # All six nominal diameters; set a list to select a subset.
# Confirmed by the experimenter; the instrument's Water model is kept separately.
BUFFER = "TRIS-HCl"
BUFFER_CONCENTRATION_MM = 10.0
PH = None
PARTICLE_CONCENTRATION = None
PARTICLE_DILUTION = None
STYLE_PATH = Path(__file__).resolve().parents[1] / "thesis.mplstyle"
FIG_SIZE = (4.72, 3.32)  # Dissertation guide: narrow, 0.75 of the text width.
SHOW_HEURISTIC_REFERENCE = True
SHOW_FIGURE = False
# Installed spreadsheet-authoring runtime; only needed for Excel export.
SPREADSHEET_RUNTIME = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node"
FIG_STEM = "Zeta_potential_particle_stability"
CONDITION_FIELDS = (
    "buffer", "buffer_concentration_mM", "pH", "temperature_C",
    "particle_concentration", "particle_dilution", "ionic_strength_mM",
)
SUMMARY_COLUMNS = [
    "nominal_size_nm", "zeta_mean_mV", "zeta_sd_mV", "zeta_sem_mV",
    "n_measurements", "buffer", "pH", "temperature_C",
    "buffer_concentration_mM", "particle_concentration", "particle_dilution",
    "ionic_strength_mM", "temperature_source", "solvent_model",
    "conductivity_mean_mS_cm", "conductivity_min_mS_cm",
    "conductivity_max_mS_cm", "n_conductivity_measurements",
    "metadata_source", "measurement_names", "source_sheets", "source_file",
]


def _text(value):
    return "" if value is None else str(value).strip()


def _number(value, label):
    """Parse finite numeric values, accepting a decimal comma, never coercing errors."""
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a number, got {value!r}") from exc
    if not np.isfinite(number):
        raise ValueError(f"{label}: non-finite value {value!r}")
    return number


def parse_measurement(rows, sheet_name, source_file):
    """Read the labelled A:D metadata block, excluding distribution/trace columns."""
    header = {_text(r[0]).lower(): r[1] for r in rows if r[0] is not None}
    if "measurement name" not in header:
        return None  # Analysis overview is intentionally not counted again.
    if _text(header.get("measurement mode")).lower() != "zeta potential":
        raise ValueError(f"{sheet_name}: expected a zeta-potential measurement.")
    fields = {}
    section = ""
    solvent = ""
    for row in rows:
        label = _text(row[1]).lower()
        if label == "solvent" and row[2] is None:
            section = "solvent"
        elif row[0] is not None:
            section = _text(row[0]).lower()
        if section == "solvent" and label == "name":
            solvent = _text(row[2])
        if label and row[2] is not None:
            if label in fields and fields[label] != (row[2], row[3]):
                # Generic Name/Type labels can occur in different sections.
                if label not in {"name", "type", "serial number"}:
                    raise ValueError(f"{sheet_name}: conflicting {label!r} fields.")
            fields[label] = (row[2], row[3])

    def field(*labels):
        for label in labels:
            if label in fields:
                return fields[label]
        return None, None

    def numeric(labels, allowed_units=None, required=False):
        value, unit = field(*labels)
        if value is None:
            if required:
                raise ValueError(f"{sheet_name}: missing {labels[0]}.")
            return None
        if allowed_units is not None and _text(unit) not in allowed_units:
            raise ValueError(f"{sheet_name}: unsupported unit {unit!r} for {labels[0]}.")
        return _number(value, f"{sheet_name}: {labels[0]}")

    name = _text(header["measurement name"])
    match = re.match(r"^\s*(\d+(?:[.,]\d+)?)\s*nm\b", name, re.IGNORECASE)
    if not match:
        raise ValueError(f"{sheet_name}: cannot read nominal diameter from {name!r}.")
    size = _number(match[1], name)
    if size <= 0:
        raise ValueError(f"{sheet_name}: nominal diameter must be positive.")
    temperature_labels = ("temperature", "target temperature")
    temperature = numeric(temperature_labels, {"°C", "�C", "C", "degC"})
    concentration, concentration_unit = field("particle concentration")
    record = {
        "nominal_size_nm": size,
        "measurement_name": name,
        "zeta_mV": numeric(("mean zeta potential",), {"mV"}, required=True),
        "conductivity_mS_cm": numeric(("conductivity",), {"mS/cm"}),
        "buffer": _text(field("buffer", "buffer composition", "buffer name")[0]),
        "buffer_concentration_mM": numeric(("buffer concentration",), {"mM", "mmol/L"}),
        "pH": numeric(("ph",)),
        "temperature_C": temperature,
        "temperature_source": ("temperature" if "temperature" in fields else "target temperature") if temperature is not None else "",
        "particle_concentration": " ".join(filter(None, (_text(concentration), _text(concentration_unit)))),
        "particle_dilution": _text(field("particle dilution", "dilution")[0]),
        "ionic_strength_mM": numeric(("ionic strength",), {"mM", "mmol/L"}),
        "solvent_model": solvent,
        "comment": _text(header.get("comment")),
        "metadata_source": "workbook",
        "source_sheet": sheet_name,
        "source_file": str(source_file),
    }
    return record


def load_measurements(source: Path) -> pd.DataFrame:
    if not source.is_file():
        raise FileNotFoundError(f"Workbook not found: {source}. Update XLSX_PATH in the configuration.")
    workbook = openpyxl.load_workbook(source, read_only=True, data_only=True)
    records = []
    try:
        for sheet in workbook:
            rows = list(sheet.iter_rows(max_col=4, values_only=True))
            record = parse_measurement(rows, sheet.title, source.resolve())
            if record is not None:
                records.append(record)
    finally:
        workbook.close()
    if not records:
        raise ValueError("No Kalliope zeta-potential measurement sheets found.")
    data = pd.DataFrame(records)
    if data.measurement_name.duplicated().any():
        raise ValueError("Duplicate measurement names: check for repeated exports before counting replicates.")
    return data


def select_measurements(data, sizes):
    available = sorted(data.nominal_size_nm.unique())
    sizes = available if sizes is None else sizes
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError(f"Set SIZES to distinct reference sizes. Available (nm): {available}")
    missing = set(sizes) - set(available)
    if missing:
        raise ValueError(f"No measurements for requested sizes: {sorted(missing)}")
    selected = data[data.nominal_size_nm.isin(sizes)].copy()
    omitted = data[~data.nominal_size_nm.isin(sizes)]
    if not omitted.empty:
        print("Not selected for the reference figure:", ", ".join(omitted.measurement_name))
    return selected


def fill_conditions(data, supplied):
    """Fill missing metadata only; never silently overwrite conflicting source data."""
    data = data.copy()
    for key, value in supplied.items():
        if key not in CONDITION_FIELDS or value is None:
            continue
        if isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f"{key}: must be finite.")
        for index in data.index:
            original = data.at[index, key]
            if pd.isna(original) or original == "":
                data.at[index, key] = value
                data.at[index, "metadata_source"] += f"; {key}: user supplied"
                if key == "temperature_C":
                    data.at[index, "temperature_source"] = "user supplied"
            elif str(original).strip().casefold() != str(value).strip().casefold():
                if not (isinstance(value, (int, float)) and float(original) == value):
                    raise ValueError(f"{key}: supplied value {value!r} conflicts with workbook value {original!r}.")
    return data


def validate_conditions(data):
    """Require one known condition set; missing metadata is stated, not inferred."""
    for key in CONDITION_FIELDS:
        values = data[key].map(lambda x: "" if pd.isna(x) else str(x).strip().casefold())
        if values.nunique() > 1:
            raise ValueError(f"Different or partially missing {key} values: {list(values.unique())}. "
                             "Select measurements from a single documented condition set; do not pool them.")
    if data.buffer.eq("").all():
        print("NOTE: buffer composition is undocumented; the solvent model is not the buffer.")


def summarize(data):
    rows = []
    for size, group in data.groupby("nominal_size_nm", sort=True):
        values = group.zeta_mV.to_numpy(dtype=float)
        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        row = {
            "nominal_size_nm": size, "zeta_mean_mV": float(values.mean()),
            "zeta_sd_mV": sd, "zeta_sem_mV": sd / np.sqrt(len(values)),
            "n_measurements": len(values),
            **{key: group.iloc[0][key] for key in CONDITION_FIELDS},
        }
        conductivity = pd.to_numeric(group.conductivity_mS_cm).dropna()
        row.update({
            "conductivity_mean_mS_cm": conductivity.mean(),
            "conductivity_min_mS_cm": conductivity.min(),
            "conductivity_max_mS_cm": conductivity.max(),
            "n_conductivity_measurements": len(conductivity),
        })
        for key in ("temperature_source", "solvent_model", "metadata_source", "source_file"):
            row[key] = "; ".join(dict.fromkeys(group[key]))
        row["measurement_names"] = "; ".join(group.measurement_name)
        row["source_sheets"] = "; ".join(group.source_sheet)
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def print_values(data, summary):
    for row in summary.itertuples():
        print(f"\n{row.nominal_size_nm:g} nm:")
        for measurement in data[data.nominal_size_nm == row.nominal_size_nm].itertuples():
            print(f"  {measurement.measurement_name} [{measurement.source_sheet}]: {measurement.zeta_mV:.12g} mV")
        spread = f"{row.zeta_sd_mV:.6g}" if np.isfinite(row.zeta_sd_mV) else "undefined (n < 2)"
        print(f"  mean +/- SD = {row.zeta_mean_mV:.6g} +/- {spread} mV; n = {row.n_measurements}")


def stability_message(data, summary):
    """Describe the observed sign and magnitude without assigning stability classes."""
    values = data.zeta_mV.to_numpy()
    sign = "negative" if np.all(values < 0) else "positive" if np.all(values > 0) else None
    if sign is None:
        return "Zeta-potential signs vary across measurements"
    # Reproducible sign separated from zero, with mean magnitudes around or beyond
    # the 30 mV heuristic, supports a cautious stabilization interpretation.
    away_from_zero = np.all(np.abs(summary.zeta_mean_mV) > summary.zeta_sd_mV)
    if away_from_zero and np.all(np.abs(summary.zeta_mean_mV) >= 30):
        return f"Consistently {sign} zeta potential\nSupports electrostatic stabilization"
    return f"Consistently {sign} zeta potential\nStability depends on the complete suspension conditions"


def make_figure(data, summary, heuristic_lines=True):
    """Use the dissertation style with the requested nominal-size replicate design.

    Explicit task requirements retain categorical nominal diameters and jitter.
    Signed zeta potentials use a linear y-axis; the log-log diffusion convention
    does not apply. Export remains PNG by the user's explicit format preference.
    """
    with plt.style.context(STYLE_PATH):
        fig, ax = plt.subplots(figsize=FIG_SIZE, layout="constrained")
        ax.axhline(0, color="0.65", linewidth=0.6, zorder=0)
        bounds = [0.0, *data.zeta_mV]
        low, high = min(bounds), max(bounds)
        pad = max(4.0, (high - low) * 0.10)
        ax.set_ylim(low - pad, high + pad)
        if heuristic_lines:
            for reference in (-30, 30):
                if low - pad < reference < high + pad:
                    ax.axhline(reference, color="0.65", linewidth=0.6,
                               linestyle=(0, (4, 3)), zorder=0)
        for x, row in enumerate(summary.itertuples()):
            base, dark = SIZE_COLORS[row.nominal_size_nm]
            values = data.loc[data.nominal_size_nm == row.nominal_size_nm, "zeta_mV"].to_numpy()
            left_count = len(values) // 2
            offsets = np.concatenate((np.linspace(-0.21, -0.12, left_count),
                                      np.linspace(0.12, 0.21, len(values) - left_count)))
            offsets = np.random.default_rng(1729 + x).permutation(offsets)
            ax.scatter(x + offsets, values, s=10, facecolors=base, alpha=0.65,
                       edgecolors=dark, linewidths=0.45, zorder=3)
        ax.set_xticks(np.arange(len(summary)), [f"{x:g}" for x in summary.nominal_size_nm])
        ax.set_xlim(-0.5, len(summary) - 0.5)
        ax.set_xlabel("Nominal particle diameter (nm)")
        ax.set_ylabel(r"Zeta potential, $\zeta$ (mV)")
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    return fig


def caption_text(data, summary, heuristic_lines):
    parts = [
        "Zeta potential of fluorescent polystyrene particles at the indicated nominal diameters. "
        "Points show individual measurement results. Horizontal offsets are for visibility only. "
        "Means, sample SD (ddof = 1) and SEM are provided in the Statistics worksheet. "
        "The instrument's internal runs are not counted as independent replicates.",
        "Measurements per size: " + "; ".join(f"{r.nominal_size_nm:g} nm, n = {r.n_measurements}" for r in summary.itertuples()) + ".",
    ]
    for key in CONDITION_FIELDS:
        value = data.iloc[0][key]
        parts.append(f"{key}: {value if pd.notna(value) and value != '' else 'not recorded'}.")
    buffer = data.iloc[0].buffer
    concentration = data.iloc[0].buffer_concentration_mM
    if buffer and pd.notna(concentration):
        parts.append(f"Measurement buffer: {concentration:g} mM {buffer}.")
    parts.append("Temperature source: " + "; ".join(dict.fromkeys(data.temperature_source)) + ".")
    cond = pd.to_numeric(data.conductivity_mS_cm).dropna()
    if len(cond):
        parts.append(f"Recorded conductivity: {cond.min():.4g}–{cond.max():.4g} mS/cm (n = {len(cond)}).")
    if heuristic_lines:
        parts.append("Dashed lines show the ±30 mV heuristic reference where it falls within the plotted range. These are not absolute stability thresholds.")
    values = data.zeta_mV.to_numpy()
    same_sign = np.all(values < 0) or np.all(values > 0)
    if same_sign:
        parts.append("All selected measurements have a consistent " + ("negative" if values[0] < 0 else "positive") + " sign.")
        if np.all(np.abs(values) > 30):
            parts.append("All individual magnitudes exceed the commonly used 30 mV heuristic, consistent with relatively strong electrostatic stabilization under the measured conditions.")
        else:
            parts.append("Some magnitudes are at or below the 30 mV heuristic. This does not by itself establish instability.")
    else:
        parts.append("The measurements do not establish a consistently signed zeta potential across the selected particles.")
    parts.append(
        "Electrostatic repulsion can reduce the probability of irreversible aggregation following Brownian particle encounters. "
        "Encounters can still occur without aggregation. Zeta potential is not a direct measurement of interparticle forces "
        "and alone cannot establish complete colloidal stability or exclude an aggregation contribution to diffusion. "
        "Interpretation also depends on electrolyte conditions, surface chemistry, steric stabilization and particle concentration."
    )
    if data.buffer.eq("").any():
        parts.append("Buffer composition is undocumented in this export; correspondence to the tracking buffer must be established before transferring this interpretation to the diffusion experiments.")
    if (summary.n_measurements < 2).any():
        parts.append("SD and SEM are undefined for n = 1.")
    parts.append("Source: " + str(data.iloc[0].source_file))
    return "\n\n".join(parts) + "\n"


def export_excel(data, summary, caption, output_path):
    """Write numeric results, individual measurements and caption into one workbook."""
    helper = Path(__file__).with_name("litesizer_zeta_excel.mjs")
    node = SPREADSHEET_RUNTIME / "bin/node.exe"
    if not node.is_file():
        raise FileNotFoundError(f"Excel runtime missing: {node}. Update SPREADSHEET_RUNTIME.")
    payload = {
        "summary": json.loads(summary.to_json(orient="split", index=False, double_precision=15)),
        "individual": json.loads(data.to_json(orient="split", index=False, double_precision=15)),
        "caption": caption,
        "interpretation": stability_message(data, summary).replace("\n", ". "),
        "output": str(output_path),
    }
    subprocess.run([str(node), str(helper), str(SPREADSHEET_RUNTIME)],
                   input=json.dumps(payload), text=True, encoding="utf-8", check=True,
                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))


def main():
    data = select_measurements(load_measurements(XLSX_PATH), SIZES)
    data = fill_conditions(data, {
        "buffer": BUFFER, "buffer_concentration_mM": BUFFER_CONCENTRATION_MM,
        "pH": PH, "particle_concentration": PARTICLE_CONCENTRATION,
        "particle_dilution": PARTICLE_DILUTION,
    })
    validate_conditions(data)
    summary = summarize(data)
    print_values(data, summary)
    caption = caption_text(data, summary, SHOW_HEURISTIC_REFERENCE)
    caption = stability_message(data, summary).replace("\n", ". ") + ".\n\n" + caption
    fig = make_figure(data, summary, SHOW_HEURISTIC_REFERENCE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{FIG_STEM}.png", dpi=600)
    export_excel(data, summary, caption, OUTPUT_DIR / f"{FIG_STEM}_statistics.xlsx")
    print(f"\nSaved PNG and Excel results to {OUTPUT_DIR}")
    if SHOW_FIGURE:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
