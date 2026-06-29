"""metrics 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_validity_report(pred: pd.DataFrame) -> dict:
    """执行 reliability validity report 对应的项目处理逻辑。"""
    report = {}
    report["hazard_min"] = float(pred["hazard"].min())
    report["hazard_max"] = float(pred["hazard"].max())
    report["reliability_min"] = float(pred["reliability"].min())
    report["reliability_max"] = float(pred["reliability"].max())
    violations = 0
    checked = 0
    for _, g in pred.groupby(["unit_id", "current_time"]):
        r = g.sort_values("future_step")["reliability"].values
        violations += int(np.sum(np.diff(r) > 1e-6))
        checked += max(0, len(r) - 1)
    report["monotonic_violations"] = int(violations)
    report["monotonic_pairs_checked"] = int(checked)
    report["monotonic_ok"] = violations == 0
    return report
