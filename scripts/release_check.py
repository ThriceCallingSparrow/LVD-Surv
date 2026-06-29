#!/usr/bin/env python
"""Run the supported release-readiness check before a final acceptance run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lvd_surv.cli import main

if __name__ == "__main__":
    sys.argv.insert(1, "release-check")
    main()
