"""
Scrape TrackMate XML files for analysis parameters:
  - radius (spot detection radius)
  - linking max distance
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import extract_particle_size_from_path

root = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
# root = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
root = Path(r"E:\PhD Data Analysis\SPT 2025 II")
ENCODINGS = ["utf-8-sig", "utf-8", "latin-1", "cp1252"]

RADIUS_RE  = re.compile(r'-\s*radius\s*:\s*([0-9]+\.?[0-9]*)')
LINKING_RE = re.compile(r'-\s*linking max distance\s*:\s*([0-9]+\.?[0-9]*)')


def _read(path: Path) -> str | None:
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None


results = []
n_skipped = 0
xml_files = list(root.rglob("*.xml"))
print(f"Anzahl XML-Dateien: {len(xml_files)}")

for xml_path in xml_files:
    content = _read(xml_path)
    if content is None:
        print(f"  [FEHLER] Konnte nicht lesen: {xml_path.name}")
        n_skipped += 1
        continue

    r_match = RADIUS_RE.search(content)
    l_match = LINKING_RE.search(content)

    if r_match is None and l_match is None:
        print(f"  [SKIP kein Parameter] {xml_path.name}")
        n_skipped += 1
        continue

    results.append({
        "file":             xml_path.name,
        "particle_size_nm": extract_particle_size_from_path(xml_path),
        "radius":           float(r_match.group(1)) if r_match else None,
        "linking_max_dist": float(l_match.group(1)) if l_match else None,
        "path":             str(xml_path),
    })

df = pd.DataFrame(results)
print(f"\nGefunden: {len(df)} Dateien, {n_skipped} übersprungen")
print(df[["file", "particle_size_nm", "radius", "linking_max_dist"]].to_string(index=False))

# ── Plots ─────────────────────────────────────────────────────────────
SIZES = [20, 50, 100, 200, 500, 1000]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

for size in SIZES:
    subset = df[df["particle_size_nm"] == size]

    r_vals = subset["radius"].dropna()
    if not r_vals.empty:
        ax1.scatter([size] * len(r_vals), r_vals, label=f"{size} nm", zorder=3)

    l_vals = subset["linking_max_dist"].dropna()
    if not l_vals.empty:
        ax2.scatter([size] * len(l_vals), l_vals, label=f"{size} nm", zorder=3)

for ax, title, ylabel in [
    (ax1, "Spot detection radius", "Radius (px)"),
    (ax2, "Linking max distance", "Linking max distance (px)"),
]:
    ax.set_xlabel("Particle size (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(SIZES)
    ax.legend(loc="best", fontsize=8)

plt.suptitle(f"TrackMate parameters – {root.name}", fontweight="semibold")
plt.show()
