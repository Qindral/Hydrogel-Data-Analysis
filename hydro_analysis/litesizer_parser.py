"""LiteSizer XLSX data parser for particle size measurements."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PeakAnalysis:
    """Peak analysis data (volume, intensity, or number weighted)."""

    peak_values: List[float] = field(default_factory=list)
    area_values: List[float] = field(default_factory=list)
    std_values: List[float] = field(default_factory=list)


@dataclass
class DValues:
    """D values (D10, D50, D90) for volume, intensity, and number."""

    d10_volume: Optional[float] = None
    d50_volume: Optional[float] = None
    d90_volume: Optional[float] = None
    d10_intensity: Optional[float] = None
    d50_intensity: Optional[float] = None
    d90_intensity: Optional[float] = None
    d10_number: Optional[float] = None
    d50_number: Optional[float] = None
    d90_number: Optional[float] = None
    undersize_span_volume: Optional[float] = None
    undersize_span_intensity: Optional[float] = None
    undersize_span_number: Optional[float] = None


@dataclass
class CorrelationFunction:
    """Correlation function data."""

    delay_time: np.ndarray = field(default_factory=lambda: np.array([]))
    g2: np.ndarray = field(default_factory=lambda: np.array([]))
    cumulant_fit: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class IntensityTrace:
    """Intensity trace data."""

    time: np.ndarray = field(default_factory=lambda: np.array([]))
    intensity: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MeasurementData:
    """Complete data for a single measurement."""

    name: str = ""
    measurement_mode: str = ""

    # Summary results
    hydrodynamic_diameter_nm: Optional[float] = None
    polydispersity_pct: Optional[float] = None
    intercept_g2: Optional[float] = None
    baseline: Optional[float] = None
    mean_intensity_kcps: Optional[float] = None
    absolute_intensity_kcps: Optional[float] = None
    fit_error: Optional[float] = None
    diffusion_coefficient_um2s: Optional[float] = None

    # Peak analysis
    peak_analysis_volume: PeakAnalysis = field(default_factory=PeakAnalysis)
    peak_analysis_intensity: PeakAnalysis = field(default_factory=PeakAnalysis)
    peak_analysis_number: PeakAnalysis = field(default_factory=PeakAnalysis)

    # D values
    d_values: DValues = field(default_factory=DValues)

    # Automatic values
    attenuator_optical_density: Optional[float] = None
    focus_position_mm: Optional[float] = None
    auto_run_criteria_pct: Optional[float] = None
    transmittance_pct: Optional[float] = None
    angle: Optional[str] = None
    processed_runs: Optional[int] = None

    # Input parameters
    sample_id: Optional[str] = None
    batch_number: Optional[str] = None
    measurement_cell: Optional[str] = None
    measurement_angle: Optional[str] = None
    target_temperature_c: Optional[float] = None
    equilibration_time: Optional[str] = None
    analysis_model: Optional[str] = None
    cumulant_model: Optional[str] = None

    # Quality
    quality_mode: Optional[str] = None
    max_runs: Optional[int] = None
    accepted_runs_pct: Optional[float] = None
    measurement_time: Optional[str] = None

    # Laser
    laser_attenuation_mode: Optional[str] = None
    optical_density: Optional[float] = None

    # Focus
    focus_mode: Optional[str] = None
    focus_position: Optional[float] = None

    # Material
    material_name: Optional[str] = None
    refractive_index: Optional[float] = None
    absorption_index: Optional[float] = None

    # Solvent
    solvent_name: Optional[str] = None
    solvent_refractive_index: Optional[float] = None
    solvent_viscosity_pas: Optional[float] = None
    relative_permittivity: Optional[float] = None

    # Additional information
    user: Optional[str] = None
    start_time: Optional[str] = None
    software_version: Optional[str] = None
    computer_name: Optional[str] = None

    # Instrument
    instrument_type: Optional[str] = None
    instrument_serial: Optional[str] = None
    module_type: Optional[str] = None
    module_serial: Optional[str] = None

    # Data arrays
    correlation_function: CorrelationFunction = field(default_factory=CorrelationFunction)
    intensity_trace: IntensityTrace = field(default_factory=IntensityTrace)

    # Size distributions
    size_distribution_volume: pd.DataFrame = field(default_factory=pd.DataFrame)
    size_distribution_intensity: pd.DataFrame = field(default_factory=pd.DataFrame)
    size_distribution_number: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class LiteSizerData:
    """Container for all LiteSizer data from an XLSX file."""

    workbook_name: str = ""
    analysis_name: str = ""
    created_by: str = ""
    weighting_model: str = ""

    summary_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    measurements: List[MeasurementData] = field(default_factory=list)

    def get_measurement_by_name(self, name: str) -> Optional[MeasurementData]:
        """Get a measurement by its name."""
        for m in self.measurements:
            if m.name == name:
                return m
        return None

    def to_summary_dataframe(self) -> pd.DataFrame:
        """Convert all measurements to a summary DataFrame."""
        records = []
        for m in self.measurements:
            record = {
                "name": m.name,
                "hydrodynamic_diameter_nm": m.hydrodynamic_diameter_nm,
                "polydispersity_pct": m.polydispersity_pct,
                "diffusion_coefficient_um2s": m.diffusion_coefficient_um2s,
                "transmittance_pct": m.transmittance_pct,
                "mean_intensity_kcps": m.mean_intensity_kcps,
                "material": m.material_name,
                "solvent": m.solvent_name,
                "temperature_c": m.target_temperature_c,
                "angle": m.angle,
                "user": m.user,
                "start_time": m.start_time,
            }
            # Add peak values
            for i, peak in enumerate(m.peak_analysis_volume.peak_values, 1):
                record[f"peak_{i}_volume_nm"] = peak
            for i, peak in enumerate(m.peak_analysis_intensity.peak_values, 1):
                record[f"peak_{i}_intensity_nm"] = peak
            for i, peak in enumerate(m.peak_analysis_number.peak_values, 1):
                record[f"peak_{i}_number_nm"] = peak
            # Add D values
            record["d10_volume"] = m.d_values.d10_volume
            record["d50_volume"] = m.d_values.d50_volume
            record["d90_volume"] = m.d_values.d90_volume
            records.append(record)
        return pd.DataFrame(records)


def _safe_float(value) -> Optional[float]:
    """Safely convert a value to float."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        value = value.replace(",", ".").strip()
        if not value or value == "-":
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    """Safely convert a value to int."""
    f = _safe_float(value)
    return int(f) if f is not None else None


def _get_cell(df: pd.DataFrame, row: int, col: int):
    """Safely get a cell value from a DataFrame."""
    if row < len(df) and col < len(df.columns):
        val = df.iloc[row, col]
        if pd.notna(val):
            return val
    return None


def _get_cell_str(df: pd.DataFrame, row: int, col: int) -> str:
    """Get cell value as lowercase string."""
    val = _get_cell(df, row, col)
    return str(val).lower().strip() if val is not None else ""


def _parse_summary_sheet(df: pd.DataFrame) -> Tuple[Dict[str, str], pd.DataFrame]:
    """Parse the summary sheet to extract header info and data table."""
    header_info = {}

    # Find header information (first few rows)
    for idx in range(min(10, len(df))):
        row = df.iloc[idx]
        for col_idx in range(len(row) - 1):
            val = row.iloc[col_idx]
            if isinstance(val, str):
                val_lower = val.lower().strip()
                if "workbook name" in val_lower:
                    header_info["workbook_name"] = str(row.iloc[col_idx + 1]) if pd.notna(row.iloc[col_idx + 1]) else ""
                elif "analysis name" in val_lower:
                    header_info["analysis_name"] = str(row.iloc[col_idx + 1]) if pd.notna(row.iloc[col_idx + 1]) else ""
                elif "created by" in val_lower:
                    header_info["created_by"] = str(row.iloc[col_idx + 1]) if pd.notna(row.iloc[col_idx + 1]) else ""
                elif "weighting model" in val_lower:
                    header_info["weighting_model"] = str(row.iloc[col_idx + 1]) if pd.notna(row.iloc[col_idx + 1]) else ""

    # Find the data table start (look for "Result" or "Name" header)
    data_start = None
    for idx in range(len(df)):
        row = df.iloc[idx]
        for col_idx, val in enumerate(row):
            if isinstance(val, str) and val.strip().lower() in ("result", "name"):
                data_start = idx
                break
        if data_start is not None:
            break

    if data_start is None:
        return header_info, pd.DataFrame()

    # Find header row (contains "Name", "Temperature", etc.)
    header_row = data_start
    for idx in range(data_start, min(data_start + 5, len(df))):
        row = df.iloc[idx]
        name_found = any(isinstance(v, str) and "name" in v.lower() for v in row)
        temp_found = any(isinstance(v, str) and "temperature" in v.lower() for v in row)
        if name_found and temp_found:
            header_row = idx
            break

    # Find data columns
    header = df.iloc[header_row]
    col_mapping = {}
    for col_idx, val in enumerate(header):
        if isinstance(val, str):
            val_lower = val.lower().strip()
            if "name" in val_lower and "workbook" not in val_lower:
                col_mapping["name"] = col_idx
            elif "temperature" in val_lower:
                col_mapping["temperature"] = col_idx
            elif "angle" in val_lower:
                col_mapping["angle"] = col_idx
            elif "hydrodynamic" in val_lower or "diameter" in val_lower:
                col_mapping["diameter"] = col_idx
            elif "polydispersity" in val_lower:
                col_mapping["polydispersity"] = col_idx
            elif "peak 1" in val_lower:
                col_mapping["peak1"] = col_idx
            elif "peak 2" in val_lower:
                col_mapping["peak2"] = col_idx
            elif "peak 3" in val_lower:
                col_mapping["peak3"] = col_idx
            elif "transmittance" in val_lower:
                col_mapping["transmittance"] = col_idx
            elif "diffusion" in val_lower:
                col_mapping["diffusion"] = col_idx

    # Extract data rows (skip header and unit rows, stop at summary stats)
    data_rows = []
    for idx in range(header_row + 2, len(df)):  # +2 to skip header and unit row
        row = df.iloc[idx]
        first_val = row.iloc[col_mapping.get("name", 0)] if col_mapping else row.iloc[0]
        if isinstance(first_val, str):
            first_lower = first_val.lower().strip()
            if first_lower in ("average", "st.dev.", "rsd%", ""):
                break
        if pd.notna(first_val):
            data_rows.append(row)

    # Build clean DataFrame
    if data_rows and col_mapping:
        records = []
        for row in data_rows:
            record = {}
            for key, col_idx in col_mapping.items():
                val = row.iloc[col_idx] if col_idx < len(row) else None
                record[key] = val
            records.append(record)
        data_df = pd.DataFrame(records)
    else:
        data_df = pd.DataFrame()

    return header_info, data_df


def _parse_measurement_sheet(df: pd.DataFrame) -> MeasurementData:
    """Parse a single measurement sheet.

    Sheet structure:
    - Column 0: Section names (e.g., "Results", "Peak analysis volume")
    - Column 1: Label names (e.g., "Hydrodynamic diameter", "Peak volume 1")
    - Column 2: Values
    """
    measurement = MeasurementData()

    # Track current section and subsection
    current_section = ""
    current_subsection = ""

    for idx in range(len(df)):
        col0 = _get_cell_str(df, idx, 0)
        col1 = _get_cell_str(df, idx, 1)
        col2 = _get_cell(df, idx, 2)

        # Update section tracking
        if col0 and col0 != "nan":
            current_section = col0
            current_subsection = ""

        # Check for subsections in column 1
        if col1 in ("quality", "laser attenuation", "focus", "material", "solvent",
                    "instrument", "module", "measurement", "general"):
            current_subsection = col1

        # Header info (first rows)
        if "measurement name" in col0:
            measurement.name = str(col2) if col2 else str(_get_cell(df, idx, 1)) or ""
        elif "measurement mode" in col0:
            measurement.measurement_mode = str(col2) if col2 else str(_get_cell(df, idx, 1)) or ""

        # Results section
        if current_section == "results":
            if "hydrodynamic diameter" in col1:
                measurement.hydrodynamic_diameter_nm = _safe_float(col2)
            elif "polydispersity" in col1:
                measurement.polydispersity_pct = _safe_float(col2)
            elif "intercept" in col1:
                measurement.intercept_g2 = _safe_float(col2)
            elif "baseline" in col1:
                measurement.baseline = _safe_float(col2)
            elif "mean intensity" in col1:
                measurement.mean_intensity_kcps = _safe_float(col2)
            elif "absolute intensity" in col1:
                measurement.absolute_intensity_kcps = _safe_float(col2)
            elif "fit error" in col1:
                measurement.fit_error = _safe_float(col2)
            elif "diffusion coefficient" in col1:
                measurement.diffusion_coefficient_um2s = _safe_float(col2)

        # Peak analysis volume
        elif current_section == "peak analysis volume":
            val = _safe_float(col2)
            if val is not None:
                if "peak volume" in col1 and "standard deviation" not in col1:
                    measurement.peak_analysis_volume.peak_values.append(val)
                elif "area volume" in col1:
                    measurement.peak_analysis_volume.area_values.append(val)
                elif "standard deviation volume" in col1:
                    measurement.peak_analysis_volume.std_values.append(val)

        # Peak analysis intensity
        elif current_section == "peak analysis intensity":
            val = _safe_float(col2)
            if val is not None:
                if "peak intensity" in col1 and "standard deviation" not in col1:
                    measurement.peak_analysis_intensity.peak_values.append(val)
                elif "area intensity" in col1:
                    measurement.peak_analysis_intensity.area_values.append(val)
                elif "standard deviation intensity" in col1:
                    measurement.peak_analysis_intensity.std_values.append(val)

        # Peak analysis number
        elif current_section == "peak analysis number":
            val = _safe_float(col2)
            if val is not None:
                if "peak number" in col1 and "standard deviation" not in col1:
                    measurement.peak_analysis_number.peak_values.append(val)
                elif "area number" in col1:
                    measurement.peak_analysis_number.area_values.append(val)
                elif "standard deviation number" in col1:
                    measurement.peak_analysis_number.std_values.append(val)

        # D values
        elif current_section == "d values":
            if "d10" in col1:
                if "volume" in col1:
                    measurement.d_values.d10_volume = _safe_float(col2)
                elif "intensity" in col1:
                    measurement.d_values.d10_intensity = _safe_float(col2)
                elif "number" in col1:
                    measurement.d_values.d10_number = _safe_float(col2)
            elif "d50" in col1:
                if "volume" in col1:
                    measurement.d_values.d50_volume = _safe_float(col2)
                elif "intensity" in col1:
                    measurement.d_values.d50_intensity = _safe_float(col2)
                elif "number" in col1:
                    measurement.d_values.d50_number = _safe_float(col2)
            elif "d90" in col1:
                if "volume" in col1:
                    measurement.d_values.d90_volume = _safe_float(col2)
                elif "intensity" in col1:
                    measurement.d_values.d90_intensity = _safe_float(col2)
                elif "number" in col1:
                    measurement.d_values.d90_number = _safe_float(col2)
            elif "undersize span" in col1:
                if "volume" in col1:
                    measurement.d_values.undersize_span_volume = _safe_float(col2)
                elif "intensity" in col1:
                    measurement.d_values.undersize_span_intensity = _safe_float(col2)
                elif "number" in col1:
                    measurement.d_values.undersize_span_number = _safe_float(col2)

        # Automatic values
        elif current_section == "automatic values":
            if "attenuator" in col1:
                measurement.attenuator_optical_density = _safe_float(col2)
            elif "focus position" in col1:
                measurement.focus_position_mm = _safe_float(col2)
            elif "auto run" in col1:
                measurement.auto_run_criteria_pct = _safe_float(col2)
            elif "transmittance" in col1:
                measurement.transmittance_pct = _safe_float(col2)
            elif "angle" in col1:
                measurement.angle = str(col2) if col2 else None
            elif "processed runs" in col1:
                measurement.processed_runs = _safe_int(col2)

        # Input parameters
        elif current_section == "input parameters":
            if current_subsection == "general" or not current_subsection:
                if "sample id" in col1:
                    measurement.sample_id = str(col2) if col2 else None
                elif "batch number" in col1:
                    measurement.batch_number = str(col2) if col2 else None
                elif "measurement cell" in col1:
                    measurement.measurement_cell = str(col2) if col2 else None
                elif "measurement angle" in col1:
                    measurement.measurement_angle = str(col2) if col2 else None
                elif "target temperature" in col1:
                    measurement.target_temperature_c = _safe_float(col2)
                elif "equilibration" in col1:
                    measurement.equilibration_time = str(col2) if col2 else None
                elif "analysis model" in col1:
                    measurement.analysis_model = str(col2) if col2 else None
                elif "cumulant" in col1:
                    measurement.cumulant_model = str(col2) if col2 else None

            # Quality subsection
            if current_subsection == "quality":
                if "mode" in col1:
                    measurement.quality_mode = str(col2) if col2 else None
                elif "max" in col1 and "run" in col1:
                    measurement.max_runs = _safe_int(col2)
                elif "accepted" in col1:
                    measurement.accepted_runs_pct = _safe_float(col2)
                elif "measurement time" in col1:
                    measurement.measurement_time = str(col2) if col2 else None

            # Laser Attenuation subsection
            if current_subsection == "laser attenuation":
                if "mode" in col1:
                    measurement.laser_attenuation_mode = str(col2) if col2 else None
                elif "optical density" in col1:
                    measurement.optical_density = _safe_float(col2)

            # Focus subsection
            if current_subsection == "focus":
                if "mode" in col1:
                    measurement.focus_mode = str(col2) if col2 else None
                elif "position" in col1:
                    measurement.focus_position = _safe_float(col2)

            # Material subsection
            if current_subsection == "material":
                if "name" in col1:
                    measurement.material_name = str(col2) if col2 else None
                elif "refractive" in col1:
                    measurement.refractive_index = _safe_float(col2)
                elif "absorption" in col1:
                    measurement.absorption_index = _safe_float(col2)

            # Solvent subsection
            if current_subsection == "solvent":
                if "name" in col1:
                    measurement.solvent_name = str(col2) if col2 else None
                elif "refractive" in col1:
                    measurement.solvent_refractive_index = _safe_float(col2)
                elif "viscosity" in col1:
                    measurement.solvent_viscosity_pas = _safe_float(col2)
                elif "permittivity" in col1:
                    measurement.relative_permittivity = _safe_float(col2)

        # Additional information
        elif current_section == "additional information":
            if current_subsection == "measurement" or not current_subsection:
                if "user" in col1:
                    measurement.user = str(col2) if col2 else None
                elif "start time" in col1:
                    measurement.start_time = str(col2) if col2 else None
                elif "software" in col1:
                    measurement.software_version = str(col2) if col2 else None
                elif "computer" in col1:
                    measurement.computer_name = str(col2) if col2 else None

            # Instrument subsection
            if current_subsection == "instrument":
                if "type" in col1:
                    measurement.instrument_type = str(col2) if col2 else None
                elif "serial" in col1:
                    measurement.instrument_serial = str(col2) if col2 else None

            # Module subsection
            if current_subsection == "module":
                if "type" in col1:
                    measurement.module_type = str(col2) if col2 else None
                elif "serial" in col1:
                    measurement.module_serial = str(col2) if col2 else None

    # Parse correlation function and intensity trace from right side of sheet
    _parse_correlation_and_trace(df, measurement)

    return measurement


def _parse_correlation_and_trace(df: pd.DataFrame, measurement: MeasurementData) -> None:
    """Parse correlation function and intensity trace data.

    Sheet structure (columns 12+):
    - Column with "Delay time" header -> correlation function delay times
    - Next column (g2) -> correlation function g2 values
    - Column with "Time" header under "Intensity trace" -> time values
    - Next column (Intensity) -> intensity values
    """
    delay_col = None
    g2_col = None
    trace_time_col = None
    trace_intensity_col = None
    header_row = 6  # Default header row

    # Search for column headers in rows 4-8
    for col_idx in range(len(df.columns)):
        for row_idx in range(4, min(10, len(df))):
            val = _get_cell_str(df, row_idx, col_idx)
            if "delay time" in val:
                delay_col = col_idx
                g2_col = col_idx + 1  # g2 is typically next column
                header_row = row_idx
            elif val == "time":
                # Check if this is under "Intensity trace" header
                for check_row in range(max(0, row_idx - 3), row_idx):
                    header_val = _get_cell_str(df, check_row, col_idx)
                    if "intensity trace" in header_val:
                        trace_time_col = col_idx
                        trace_intensity_col = col_idx + 1
                        break

    # Parse correlation function
    if delay_col is not None:
        delay_times = []
        g2_values = []
        cumulant_values = []

        # Find the unit row (contains [s]) and data starts after
        data_start = header_row + 2  # Skip header and unit row
        for row_idx in range(header_row, min(header_row + 5, len(df))):
            val = _get_cell_str(df, row_idx, delay_col)
            if "[s]" in val:
                data_start = row_idx + 1
                break

        # Extract data
        for row_idx in range(data_start, len(df)):
            delay = _safe_float(_get_cell(df, row_idx, delay_col))
            if delay is not None:
                delay_times.append(delay)
                g2_values.append(_safe_float(_get_cell(df, row_idx, g2_col)) or np.nan)
                # Cumulant fit may be in next column
                cumulant_col = g2_col + 1
                cumulant_values.append(_safe_float(_get_cell(df, row_idx, cumulant_col)) or np.nan)

        if delay_times:
            measurement.correlation_function = CorrelationFunction(
                delay_time=np.array(delay_times),
                g2=np.array(g2_values),
                cumulant_fit=np.array(cumulant_values),
            )

    # Parse intensity trace
    if trace_time_col is not None:
        times = []
        intensities = []

        # Find data start (after unit row)
        data_start = header_row + 2
        for row_idx in range(header_row, min(header_row + 5, len(df))):
            val = _get_cell_str(df, row_idx, trace_time_col)
            if "[s]" in val:
                data_start = row_idx + 1
                break

        # Extract data
        for row_idx in range(data_start, len(df)):
            time_val = _safe_float(_get_cell(df, row_idx, trace_time_col))
            if time_val is not None:
                times.append(time_val)
                intensities.append(_safe_float(_get_cell(df, row_idx, trace_intensity_col)) or np.nan)

        if times:
            measurement.intensity_trace = IntensityTrace(
                time=np.array(times),
                intensity=np.array(intensities),
            )


def load_litesizer_xlsx(path: str | Path) -> LiteSizerData:
    """Load a LiteSizer XLSX file and parse all data.

    Parameters
    ----------
    path : str or Path
        Path to the XLSX file.

    Returns
    -------
    LiteSizerData
        Container with summary and all measurements.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    xlsx = pd.ExcelFile(path)
    sheet_names = xlsx.sheet_names

    result = LiteSizerData()

    # Parse first sheet (summary)
    if sheet_names:
        summary_df = pd.read_excel(xlsx, sheet_name=sheet_names[0], header=None)
        header_info, data_df = _parse_summary_sheet(summary_df)
        result.workbook_name = header_info.get("workbook_name", "")
        result.analysis_name = header_info.get("analysis_name", "")
        result.created_by = header_info.get("created_by", "")
        result.weighting_model = header_info.get("weighting_model", "")
        result.summary_df = data_df

    # Parse measurement sheets
    for sheet_name in sheet_names[1:]:
        df = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        measurement = _parse_measurement_sheet(df)
        if not measurement.name:
            measurement.name = sheet_name
        result.measurements.append(measurement)

    return result


def main():
    """Example usage."""
    path = Path(r"D:\Lite Sizer Particle Measurements\Size_repitition_All Sizes.xlsx")

    print(f"Loading: {path}")
    data = load_litesizer_xlsx(path)

    print(f"\nWorkbook: {data.workbook_name}")
    print(f"Analysis: {data.analysis_name}")
    print(f"Created by: {data.created_by}")
    print(f"Weighting model: {data.weighting_model}")

    print(f"\n--- Summary ({len(data.summary_df)} measurements) ---")
    if len(data.summary_df) > 0:
        print(data.summary_df.to_string())

    print(f"\n--- Detailed Measurements ({len(data.measurements)}) ---")
    for m in data.measurements[:3]:  # Show first 3
        print(f"\n  {m.name}:")
        print(f"    Diameter: {m.hydrodynamic_diameter_nm} nm")
        print(f"    PDI: {m.polydispersity_pct}%")
        print(f"    Diffusion: {m.diffusion_coefficient_um2s} µm²/s")
        print(f"    Transmittance: {m.transmittance_pct}%")
        print(f"    Material: {m.material_name}")
        print(f"    Solvent: {m.solvent_name}")
        print(f"    Temperature: {m.target_temperature_c}°C")
        print(f"    User: {m.user}")
        print(f"    Peak Volume: {m.peak_analysis_volume.peak_values}")
        print(f"    D50 Volume: {m.d_values.d50_volume}")
        if len(m.correlation_function.delay_time) > 0:
            print(f"    Correlation function: {len(m.correlation_function.delay_time)} points")
        if len(m.intensity_trace.time) > 0:
            print(f"    Intensity trace: {len(m.intensity_trace.time)} points")

    # Export combined DataFrame
    combined_df = data.to_summary_dataframe()
    print(f"\n--- Combined DataFrame (first 5 rows) ---")
    print(combined_df.head().to_string())

    return data


if __name__ == "__main__":
    main()
