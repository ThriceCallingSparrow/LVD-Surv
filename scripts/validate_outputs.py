#!/usr/bin/env python
from __future__ import annotations

import argparse

import sys
from pathlib import Path
root_path = Path(__file__).parent.parent
sys.path.append(str(root_path))


import pandas as pd
from lvd_surv.modeling.metrics import reliability_validity_report
from lvd_surv.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate reliability outputs")
    parser.add_argument("--prediction-csv", required=True, help="Path to all_reliability_values.csv")
    parser.add_argument("--output-json", default="outputs/metrics/reliability_validity.json")
    args = parser.parse_args()
    pred = pd.read_csv(args.prediction_csv)
    report = reliability_validity_report(pred)
    save_json(report, args.output_json)
    print(report)
    if not report["monotonic_ok"]:
        raise SystemExit("Reliability monotonicity check failed.")


if __name__ == "__main__":
    main()
