"""
Parser for TA Instruments TRIOS binary run files (.tri, Discovery HR-2) and
for the corresponding TRIOS Excel exports (.xls). Data extraction and parsing
only; here wont be stored any rheological analysis or visualization methods.

File structure (reverse-engineered from TRIOS 5.0 files and verified against
the TRIOS Excel export of the same runs):

- Header: a count followed by (key, value) pairs of .NET length-prefixed
  strings (7-bit encoded length, cp1252/UTF-8 bytes): instrumenttype,
  rundate, ticks (.NET DateTime ticks of the run start), holder (geometry
  name), operator, project, samplename, comments (free text entered in
  TRIOS), proceduresegments ("Frequency sweep" / "Amplitude sweep"),
  instrumentmode, testtype. An embedded PNG thumbnail follows.
- Data channels: records "21 06 <uint32 length> <GUID>" whose content ends
  with "01 00 <uint32 n>" + n little-endian float32 values. The GUID
  identifies the variable, not a column name, so RAW_CHANNELS maps the GUIDs
  found in these files to variable names in SI units. Only raw instrument
  channels are stored; derived variables (G', G'', tan(delta), strain, ...)
  are empty declarations and are recomputed by rheo_analysis.py.
  A channel can occur a second time as a (n+1)-point buffer copy; only the
  first occurrence is used.
- Instrument and geometry rotational inertia are GUID-keyed double records
  (INERTIA_KEYS). Both are required for the TRIOS inertia correction.

Channel identification: angular frequency, step time, temperature, raw
phase, torque and displacement were matched value-by-value to the TRIOS
Excel export; gap was confirmed through the exported strain (strain =
displacement * R / gap). "axial_force_N" is inferred from magnitude and
behaviour (about 0.05 N, about -1 N for a compressed sample) and is not
contained in any export. Two further raw channels are kept under their GUID
prefix because their meaning is not established.
"""
from __future__ import annotations

import re
import struct
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── Channel and property identifiers ─────────────────────────────────────────
RAW_CHANNELS: Dict[str, str] = {
    "fa189102-89f7-4723-860d-82df3fd58ce5": "angular_frequency_rad_s",
    "7b1a8875-39e5-4993-ae1d-67210ee57fb7": "step_time_s",
    "f79c919e-6fa6-4856-92f8-6e5868f792f5": "temperature_C",
    "5f7b2d27-8360-4bc6-be74-eff29aa2fe74": "axial_force_N",
    "94411d75-361a-4cd4-a298-841a22dd9a83": "gap_m",
    "5051699d-7bbe-41e6-b9d9-540c57cc577e": "raw_phase_rad",
    "a393424e-0fb5-4971-93c9-1825d2282c44": "unidentified_a393424e",
    "cf7aa154-70d7-4369-853d-8620b7cdffe8": "unidentified_cf7aa154",
    "360672f5-f937-414b-bd06-5b9d80d6975e": "osc_torque_Nm",
    "bb58e619-9763-45b5-9c1e-5b0a5b90642c": "osc_displacement_rad",
}
REQUIRED_CHANNELS = (
    "angular_frequency_rad_s", "gap_m", "raw_phase_rad", "osc_torque_Nm", "osc_displacement_rad",
)
INERTIA_KEYS: Dict[str, str] = {
    "instrument_inertia_Nms2": "6594d858-7e8d-402c-8554-12377d9f33c0",
    "geometry_inertia_Nms2": "2dad9ebd-bdae-4839-9bd4-44c39e65f684",
}

_BLOCK_RE = re.compile(rb"\x21\x06(.{4})(.{16})", re.S)
_COUNT_RE = re.compile(rb"\xf2\x21\x01\x04\x00{7}\x01\x00(.{4})", re.S)
_DOUBLE_VALUE_RE = re.compile(rb"\x06\x20\x01\x10\x00\x00\x00\x02\x00\x00\x00\x01\x00\x00\x00(.{8})", re.S)
_PLATE_DIAMETER_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*mm\s+parallel\s+plate", re.I)


@dataclass
class TriRun:
    """Content of one .tri file: header strings, raw channels (SI units) and constants."""

    path: Path
    header: Dict[str, str]
    run_datetime: Optional[datetime]
    procedure: str
    geometry_name: str
    plate_diameter_m: float
    instrument_inertia_Nms2: float
    geometry_inertia_Nms2: float
    raw: pd.DataFrame
    parse_notes: List[str] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def comments(self) -> str:
        return self.header.get("comments", "")


# ── Low-level readers ────────────────────────────────────────────────────────

def _read_dotnet_string(buf: bytes, pos: int) -> tuple[str, int]:
    """Read a .NET BinaryWriter string (7-bit encoded byte length) at pos."""
    length, shift = 0, 0
    while True:
        byte = buf[pos]
        pos += 1
        length |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            break
    raw = buf[pos:pos + length]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return text, pos + length


def read_header(buf: bytes) -> Dict[str, str]:
    """Parse the leading (key, value) string table of a .tri file."""
    anchor = buf.find(b"\x0einstrumenttype")
    if anchor < 4:
        raise ValueError("TRIOS header table not found (no 'instrumenttype' key).")
    count = struct.unpack_from("<I", buf, anchor - 4)[0]
    header, pos = {}, anchor
    for _ in range(count):
        key, pos = _read_dotnet_string(buf, pos)
        value, pos = _read_dotnet_string(buf, pos)
        header[key] = value
    return header


def read_channels(buf: bytes) -> Dict[str, np.ndarray]:
    """Return the first complete float32 array of every known raw channel."""
    channels: Dict[str, np.ndarray] = {}
    for match in _BLOCK_RE.finditer(buf):
        guid = str(uuid.UUID(bytes_le=match.group(2)))
        name = RAW_CHANNELS.get(guid)
        if name is None or name in channels:
            continue
        length = struct.unpack("<I", match.group(1))[0]
        start = match.start() + 6
        content = buf[start:start + length]
        count_match = _COUNT_RE.search(content, 16, 64)
        if count_match is None:
            continue
        n = struct.unpack("<I", count_match.group(1))[0]
        for trailing in (3, 0, 1, 2, 4):
            end = len(content) - trailing
            first = end - 4 * n
            if first >= 6 and content[first - 6:first] == b"\x01\x00" + struct.pack("<I", n):
                channels[name] = np.frombuffer(content[first:end], "<f4").astype(float)
                break
    return channels


def read_double_property(buf: bytes, key_guid: str) -> List[float]:
    """Return every double value stored directly after the GUID key record."""
    key = uuid.UUID(key_guid).bytes_le
    values = []
    for match in re.finditer(re.escape(key), buf):
        value = _DOUBLE_VALUE_RE.search(buf, match.end(), match.end() + 120)
        if value is not None:
            values.append(struct.unpack("<d", value.group(1))[0])
    return values


def _ticks_to_datetime(ticks: str) -> Optional[datetime]:
    try:
        return datetime(1, 1, 1) + timedelta(microseconds=int(ticks) // 10)
    except (TypeError, ValueError, OverflowError):
        return None


# ── Public interface ─────────────────────────────────────────────────────────

def read_tri(path: Path | str) -> TriRun:
    """Load one TRIOS .tri file. Raises ValueError if a required constant is missing."""
    path = Path(path)
    buf = path.read_bytes()
    notes: List[str] = []

    header = read_header(buf)
    geometry_name = header.get("holder", "")
    diameter_match = _PLATE_DIAMETER_RE.search(geometry_name)
    if diameter_match is None:
        raise ValueError(f"{path.name}: geometry '{geometry_name}' is not a parallel plate with a parsable diameter.")
    plate_diameter_m = float(diameter_match.group(1).replace(",", ".")) * 1e-3

    inertia = {}
    for name, key in INERTIA_KEYS.items():
        values = read_double_property(buf, key)
        if not values:
            raise ValueError(f"{path.name}: {name} not found; the inertia correction cannot be applied.")
        if not np.allclose(values, values[0], rtol=1e-9):
            notes.append(f"{name}: differing stored values {values}, first one used")
        inertia[name] = values[0]

    channels = read_channels(buf)
    missing = [c for c in REQUIRED_CHANNELS if c not in channels]
    if missing:
        notes.append(f"missing raw channels: {missing}")
    lengths = {len(v) for v in channels.values()}
    if len(lengths) > 1:
        n_min = min(lengths)
        notes.append(f"channels have unequal lengths {sorted(lengths)}; truncated to {n_min}")
        channels = {k: v[:n_min] for k, v in channels.items()}
    n_points = min(lengths) if lengths else 0
    raw = pd.DataFrame({name: channels.get(name, np.full(n_points, np.nan)) for name in RAW_CHANNELS.values()})

    return TriRun(
        path=path,
        header=header,
        run_datetime=_ticks_to_datetime(header.get("ticks", "")),
        procedure=header.get("proceduresegments", "") or header.get("procedurename", ""),
        geometry_name=geometry_name,
        plate_diameter_m=plate_diameter_m,
        instrument_inertia_Nms2=inertia["instrument_inertia_Nms2"],
        geometry_inertia_Nms2=inertia["geometry_inertia_Nms2"],
        raw=raw,
        parse_notes=notes,
    )


def read_trios_xls(path: Path | str) -> tuple[Dict[str, object], pd.DataFrame]:
    """Load a TRIOS Excel export (.xls): Details sheet as dict, first data sheet in SI units.

    Requires xlrd. Moduli are exported in MPa and converted to Pa, torque from
    uN.m to N.m, raw phase from degrees to rad, strain stays in %.
    """
    import xlrd

    book = xlrd.open_workbook(str(path))
    details_sheet = book.sheet_by_index(0)
    details = {
        details_sheet.cell_value(r, 0): details_sheet.cell_value(r, 1)
        for r in range(details_sheet.nrows)
        if details_sheet.ncols > 1
    }
    sheet = book.sheet_by_index(1)
    names = sheet.row_values(1)
    values = np.array([sheet.row_values(r) for r in range(3, sheet.nrows)], dtype=float)
    table = pd.DataFrame(values, columns=names)

    scale = {
        "Storage modulus": ("Gp_Pa", 1e6),
        "Loss modulus": ("Gpp_Pa", 1e6),
        "Tan(delta)": ("tan_delta", 1.0),
        "Angular frequency": ("angular_frequency_rad_s", 1.0),
        "Oscillation torque": ("osc_torque_Nm", 1e-6),
        "Step time": ("step_time_s", 1.0),
        "Temperature": ("temperature_C", 1.0),
        "Raw phase": ("raw_phase_rad", np.pi / 180.0),
        "Oscillation displacement": ("osc_displacement_rad", 1.0),
        "Oscillation strain": ("strain_pct", 1.0),
    }
    converted = pd.DataFrame({new: table[old] * factor for old, (new, factor) in scale.items() if old in table})
    return details, converted
