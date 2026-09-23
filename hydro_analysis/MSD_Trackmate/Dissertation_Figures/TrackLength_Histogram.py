"""Compatibility entry point; analysis lives in Validation_GPT/TrackLength_Histogram.py."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hydro_analysis.MSD_Trackmate.Validation_GPT.TrackLength_Histogram import main


if __name__ == "__main__":
    main()
