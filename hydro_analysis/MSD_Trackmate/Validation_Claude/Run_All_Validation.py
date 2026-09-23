"""
Fuehrt alle aktiven Validation_Claude-Skripte nacheinander aus, in der
Reihenfolge, die ihre gegenseitigen Cache-Abhaengigkeiten verlangt. Kein
Skript zeigt dabei ein Plot-Fenster an (MPLBACKEND=Agg wird jedem Subprozess
erzwungen, macht ein etwaiges plt.show() zum No-Op) -- alle Abbildungen
werden trotzdem unveraendert per savefig() gespeichert, genau wie bei einem
einzelnen manuellen Lauf.

Reihenfolge (SCRIPTS unten) und der Grund dafuer:
  1. TaskB1_TrackLengthBias_Compute    -- schreibt cache/track_length_bias_*.pkl,
                                           gebraucht von iMSD_histograms.py
  2. iMSD_histograms                   -- schreibt cache/imsd_median_stepsize_by_size.pkl,
                                           gebraucht von Loc_Error_Analyse_immob_particle.py
  3. Loc_Error_Analyse_immob_particle  -- schreibt cache/localization_error_immobilized.pkl,
                                           gebraucht von Correlations.py und Validation_Summary.py
  4. PSF_Analysis                      -- unabhaengig (eigener Cache)
  5. eMSD_files_histogram              -- unabhaengig (liest nur msd_*.pkl)
  6. Correlations                      -- braucht Loc_Error_Analyse_immob_particle's Cache
  7. Fittingpoints_robustness          -- unabhaengig (liest nur msd_*.pkl)
  8. SNR_Analysis                      -- unabhaengig, am langsamsten (kein Cache, jedes
                                           Mal echte Neuberechnung ueber viele Frames)
  9. Validation_Summary                -- braucht die CSVs/Caches aller vorigen Skripte,
                                           muss deshalb immer zuletzt laufen

Jedes Skript laeuft als eigener Subprozess (python -m <Modul>), nicht als
Python-Import -- ein Absturz in einem Skript reisst die anderen nicht mit,
und jedes laeuft exakt so wie bei manuellem
`python -m hydro_analysis.MSD_Trackmate.Validation_Claude.<Skript>`.

STOP_ON_FAILURE=True (Standard): bricht beim ersten Fehler ab, da jedes
nachfolgende Skript in der Kette ohnehin an einem fehlenden Cache/CSV
scheitern wuerde (siehe deren eigene _require()-Fehlermeldungen). Auf False
setzen, um trotzdem alle Skripte durchlaufen zu lassen und alle Fehler
gesammelt am Ende zu sehen.

Konsolenausgabe jedes Skripts wird live durchgereicht UND zusaetzlich in
run_all_validation.log (neben diesem Skript) gesammelt.

Nicht Teil dieser Kette (siehe README.md): obsolete/-Skripte,
Compare_1k_vs_1000nm_RawTrajectories.py und TrackMate_Settings_Inventory.py
(Diagnose-/Audit-Werkzeuge, kein Teil des regulaeren Auswertungslaufs),
test_psf_frame_fits.py (Unit-Tests, kein Auswertungsskript).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = "hydro_analysis.MSD_Trackmate.Validation_Claude"

SCRIPTS = [
    "TaskB1_TrackLengthBias_Compute",
    "iMSD_histograms",
    "Loc_Error_Analyse_immob_particle",
    "PSF_Analysis",
    "eMSD_files_histogram",
    "Correlations",
    "Fittingpoints_robustness",
    "SNR_Analysis",
    "Validation_Summary",
]

STOP_ON_FAILURE = True

# Ein fehlgeschlagenes Skript wird bis zu MAX_ATTEMPTS Mal versucht, bevor es
# endgueltig als Fehler zaehlt -- alle Skripte hier sind sicher erneut
# ausfuehrbar (Cache-basiert, kein Seiteneffekt auf Rohdaten), und ein
# vereinzelter OSError beim savefig() (z.B. kurzzeitiger Datei-Lock durch
# Antivirus/Cloud-Sync auf dem Auswertungsbilder-Laufwerk -- real beobachtet
# bei SNR_Analysis.py, beim isolierten Nachstellen sofort wieder erfolgreich)
# ist damit kein Grund, den ganzen Lauf abzubrechen.
MAX_ATTEMPTS = 2
RETRY_DELAY_S = 5.0

LOG_FILE = Path(__file__).parent / "run_all_validation.log"


def run_one(module_name: str, log) -> tuple[bool, float]:
    full_module = f"{PACKAGE}.{module_name}"
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"   # kein Plot-Fenster; savefig() bleibt unveraendert

    total_start = time.monotonic()
    ok = False
    for attempt in range(1, MAX_ATTEMPTS + 1):
        suffix = f" (Versuch {attempt}/{MAX_ATTEMPTS})" if MAX_ATTEMPTS > 1 else ""
        header = f"\n{'=' * 80}\n{module_name}{suffix}\n{'=' * 80}\n"
        print(header, end="")
        log.write(header)
        log.flush()

        start = time.monotonic()
        process = subprocess.Popen(
            [sys.executable, "-m", full_module],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        process.wait()
        attempt_elapsed = time.monotonic() - start
        ok = process.returncode == 0

        status = "OK" if ok else f"FEHLER (exit {process.returncode})"
        footer = f"-- {module_name}: {status} ({attempt_elapsed:.1f}s)\n"
        print(footer, end="")
        log.write(footer)
        log.flush()

        if ok:
            break
        if attempt < MAX_ATTEMPTS:
            note = f"   -> erneuter Versuch in {RETRY_DELAY_S:.0f}s (moeglicherweise transienter Fehler, z.B. Datei-Lock)...\n"
            print(note, end="")
            log.write(note)
            log.flush()
            time.sleep(RETRY_DELAY_S)

    elapsed = time.monotonic() - total_start
    return ok, elapsed


def main() -> None:
    # Windows console defaults to cp1252, which cannot represent every
    # character the child scripts' German console output can produce --
    # without this, print() below crashes with UnicodeEncodeError partway
    # through a run (observed on real data). errors="replace" degrades
    # gracefully (unrepresentable characters become '?') instead of aborting.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    results: list[tuple[str, bool, float]] = []
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        for module_name in SCRIPTS:
            ok, elapsed = run_one(module_name, log)
            results.append((module_name, ok, elapsed))
            if not ok and STOP_ON_FAILURE:
                print(f"\nAbgebrochen nach Fehler in {module_name} (STOP_ON_FAILURE=True).")
                break

    print(f"\n{'=' * 80}\nZusammenfassung\n{'=' * 80}")
    for name, ok, elapsed in results:
        print(f"  {'OK    ' if ok else 'FEHLER'}  {name:<40} {elapsed:6.1f}s")
    total = sum(elapsed for _, _, elapsed in results)
    n_failed = sum(1 for _, ok, _ in results if not ok)
    print(f"\n{len(results)} Skripte gelaufen, {n_failed} fehlgeschlagen, "
          f"Gesamtdauer {total:.1f}s ({total / 60:.1f} min).")
    print(f"Log gespeichert: {LOG_FILE}")

    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
