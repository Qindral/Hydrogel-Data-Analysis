"""
Kleiner geteilter Helfer: fig.savefig() mit Wiederholung bei transienten
OSError (z.B. kurzzeitiger Datei-Lock durch Antivirus/Cloud-Sync auf dem
Auswertungsbilder-Laufwerk direkt nach einem vorherigen Schreibvorgang in
denselben Ordner). Real beobachtet in PSF_Analysis.py und
eMSD_files_histogram.py: mehrere savefig()-Aufrufe kurz hintereinander in
denselben Ordner koennen OSError [Errno 22] werfen, obwohl derselbe Aufruf
isoliert (mit etwas Abstand) sofort erfolgreich ist -- ein neuer Lauf des
gesamten Skripts allein loest das nicht zuverlaessig, wenn genau dieselbe
Abfolge erneut sofort denselben Fehler wirft.

Kein core/-Modul (core/visualization.py baut Abbildungen, dieses Modul
speichert nur eine bereits fertige Figure) -- bewusst hier in
Validation_Claude/, von den Skripten importiert, die mehrere PNGs
hintereinander in denselben Ordner schreiben.
"""
from __future__ import annotations

import time

import matplotlib.pyplot as plt

RETRIES = 3
RETRY_DELAY_S = 1.5


def safe_savefig(fig: plt.Figure, path, retries: int = RETRIES, delay: float = RETRY_DELAY_S, **kwargs) -> None:
    """fig.savefig(path, **kwargs), erneut versucht bei OSError (z.B. Errno 22
    'Invalid argument' durch einen kurzzeitigen Datei-Lock)."""
    for attempt in range(1, retries + 1):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError as exc:
            if attempt == retries:
                raise
            print(f"  [WARN] savefig fehlgeschlagen ({exc}), Versuch {attempt}/{retries} -- "
                  f"erneuter Versuch in {delay:.1f}s...")
            time.sleep(delay)
