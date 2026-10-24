"""Run a script from this folder with core/ and the script's own folder importable.

Usage (always from extended_studies/):
    python run.py study3_pinn_favoured/score_pinnwin.py [args...]
    python run.py core/test_neural.py -v

Every script reads and writes paths relative to the working directory
(datasets_v2_*, results_*, campaign/...), so run it from this folder.
"""
import runpy
import sys
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit(__doc__)

ROOT = Path(__file__).resolve().parent
script = Path(sys.argv[1]).resolve()
sys.path[:0] = [str(script.parent), str(ROOT / "core"), str(ROOT)]
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
