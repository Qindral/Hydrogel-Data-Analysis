"""Command-line interface for running Differential Dynamic Microscopy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .ddm import run_ddm_analysis


def _prompt_int(prompt: str, default: int, *, min_value: int = 1) -> int:
    while True:
        text = input(f"{prompt} [{default}]: ").strip()
        if not text:
            value = default
        else:
            try:
                value = int(text)
            except ValueError:
                print("Bitte eine ganze Zahl eingeben.")
                continue
        if value < min_value:
            print(f"Der Wert muss ≥ {min_value} sein.")
            continue
        return value


def _prompt_bool(prompt: str, default: bool) -> bool:
    suffix = "Y/n" if default else "y/N"
    text = input(f"{prompt} ({suffix}): ").strip().lower()
    if not text:
        return default
    return text in {"y", "yes", "j", "ja"}


def _prompt_path(prompt: str) -> Path:
    while True:
        text = input(prompt).strip()
        if not text:
            continue
        candidate = Path(text).expanduser().resolve()
        if candidate.exists():
            return candidate
        print(f"❌ Datei nicht gefunden: {candidate}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DDM Analyse für TIFF-Zeitserien")
    parser.add_argument("path", nargs="?", help="Pfad zur TIFF-Datei")
    parser.add_argument("--max-lag", type=int, default=None, help="Maximaler Frame-Abstand Δt")
    parser.add_argument("--q-bins", type=int, default=None, help="Anzahl radialer q-Bins")
    parser.add_argument("--frame-step", type=int, default=None, help="Jedes n-te Bild verwenden")
    parser.add_argument(
        "--no-mean-subtraction",
        action="store_true",
        help="Mittelwert nicht subtrahieren",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Ergebnis als .npz-Datei speichern",
    )
    args = parser.parse_args(argv)

    if args.path:
        path = Path(args.path).expanduser().resolve()
    else:
        path = _prompt_path("Pfad zur TIFF-Datei: ")

    if not path.exists():
        raise FileNotFoundError(path)

    max_lag = args.max_lag if args.max_lag else _prompt_int("Maximaler Lag Δt", 50, min_value=1)
    q_bins = args.q_bins if args.q_bins else _prompt_int("Anzahl q-Bins", 30, min_value=4)
    frame_step = (
        args.frame_step if args.frame_step else _prompt_int("Frame-Schritt", 1, min_value=1)
    )
    subtract_mean = not args.no_mean_subtraction
    if not args.no_mean_subtraction:
        subtract_mean = _prompt_bool("Mittelwert aus dem Stack abziehen?", True)

    print("🔄 Starte DDM-Analyse …")
    result = run_ddm_analysis(
        path,
        max_lag=max_lag,
        q_bins=q_bins,
        frame_step=frame_step,
        subtract_mean=subtract_mean,
    )

    summary: Dict[str, Any] = {
        "lags": result.lags.tolist(),
        "q_values": result.q_values.tolist(),
        "structure_shape": list(result.structure_function.shape),
        "pixel_size_um": result.pixel_size_um,
        "time_step_s": result.time_step_s,
    }
    print("✅ Analyse abgeschlossen. Zusammenfassung:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.save:
        save_path = args.save.expanduser().resolve()
    else:
        if _prompt_bool("Ergebnis als .npz speichern?", False):
            save_input = input("Pfad für Speicherung [ddm_result.npz]: ").strip()
            save_path = (
                Path(save_input).expanduser().resolve() if save_input else path.with_suffix(".ddm.npz")
            )
        else:
            save_path = None

    if save_path:
        result.save(save_path)
        print(f"💾 Ergebnis gespeichert unter {save_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

