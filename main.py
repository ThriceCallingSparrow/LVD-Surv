"""LVD-Surv desktop entry point."""
from __future__ import annotations

import os
from pathlib import Path

# Set before any workflow can import matplotlib.pyplot.  The desktop saves plots
# to files and must not create a second Cocoa/Tk event loop from worker threads.
os.environ.setdefault("MPLBACKEND", "Agg")

from lvd_surv.app.desktop import launch


if __name__ == "__main__":
    default = Path(__file__).resolve().parent / "configs" / "default.yaml"
    launch(str(default) if default.is_file() else None)
