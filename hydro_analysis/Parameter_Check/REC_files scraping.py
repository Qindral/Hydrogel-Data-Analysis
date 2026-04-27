from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

from hydro_analysis.core.io import extract_particle_size_from_path

root = Path(r"E:\PhD Data Analysis\SPT 2025 II\D_0 Wassermessung")
# root = Path(r"E:\PhD Data Analysis\SPT 2025 II\Hydrogel Messung\20mg C16")
root = Path(r"E:\PhD Data Analysis\SPT 2025 II")
# Matches e.g. "0.4 mW", "50 µW", "23µW", "3.6mW", "3.6MW"
# IGNORECASE covers mw/MW/mW; character class covers µ variants and uW
POWER_RE = re.compile(r'(\d+[.,]\d+|\d+)\s*([µ\xb5\u03bcuU]W|mW)', re.IGNORECASE)

ENCODINGS = ["utf-16", "utf-8-sig", "utf-8", "latin-1", "cp1252"]


def _read_rec(path: Path) -> str | None:
    """Try multiple encodings; return file content or None on failure."""
    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, ValueError):
            continue
    return None


results = []
n_skipped = 0
print("Anzahl der .rec Dateien:", len(list(root.rglob("*.rec"))))

for rec_file in root.rglob("*.rec"):
    content = _read_rec(rec_file)
    if content is None:
        print(f"  [FEHLER] Konnte nicht lesen: {rec_file.name}")
        n_skipped += 1
        continue

    # Only search within the Comment section
    comment_idx = content.find("Comment:")
    if comment_idx == -1:
        print(f"  [SKIP kein Comment] {rec_file.name}")
        n_skipped += 1
        continue

    comment_section = content[comment_idx + len("Comment:"):]

    m = POWER_RE.search(comment_section)
    if m is None:
        print(f"  [SKIP kein mW/µW]  {rec_file.name}  | Comment: {comment_section[:60].strip()!r}")
        n_skipped += 1
        continue

    raw_value  = m.group(1).replace(",", ".")
    power_value = float(raw_value)
    raw_unit    = m.group(2)
    is_micro    = raw_unit[0].lower() in ("µ", "\xb5", "\u03bc", "u")
    power_unit  = "µW" if is_micro else "mW"
    power_mw    = power_value * 1e-3 if is_micro else power_value

    results.append({
        "file":             rec_file.name,
        "particle_size_nm": extract_particle_size_from_path(rec_file),
        "power_value":      power_value,
        "power_unit":       power_unit,
        "power_mW":         power_mw,
        "comment_line":     m.group(0),
        "path":             str(rec_file),
    })

df = pd.DataFrame(results)
print(f"Gefunden: {len(df)} Dateien mit Leistungsangabe, {n_skipped} übersprungen")
print(df[["file", "particle_size_nm", "power_value", "power_unit", "power_mW"]].to_string(index=False))

# ── Plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
for particle_size in [20, 50, 100, 200, 500, 1000]:
    subset = df[df["particle_size_nm"] == particle_size]
    if subset.empty:
        continue
    ax.scatter([particle_size] * len(subset), subset["power_mW"],
               label=f"{particle_size} nm", zorder=3)

ax.set_xlabel("Particle size (nm)")
ax.set_ylabel("Laser power (mW)")
ax.set_title("Laser power per particle size")
ax.legend(loc="best")
plt.tight_layout()
plt.show()
