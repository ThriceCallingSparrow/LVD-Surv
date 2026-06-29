#!/usr/bin/env python
"""Single advanced command-line entry for LVD-Surv.

The project root is added to ``sys.path`` so this helper also works before an
editable installation. Installed users should normally call the ``lvd`` entry.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lvd_surv.cli import main

if __name__ == "__main__":
    main()
