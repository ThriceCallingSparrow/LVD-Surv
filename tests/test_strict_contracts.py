"""严格模式的关键回归测试。"""
from pathlib import Path
import pandas as pd
import pytest

from lvd_surv.core.contracts import apply_feature_decision
from lvd_surv.reporting.plots import plot_device_reliability


def test_missing_feature_never_falls_back():
    """缺少契约列时必须失败，不能改用全部传感器。"""
    frame = pd.DataFrame({"unit_id": [1], "cycle": [1], "setting_1": [0], "setting_2": [0], "setting_3": [0]})
    decision = {"schema_version": "1.0", "training_feature_mode": "raw", "selected_features": ["sensor_1"], "raw_selected_features": ["sensor_1"], "residual_selected_features": []}
    with pytest.raises(ValueError, match="missing columns"):
        apply_feature_decision(frame, decision)


def test_current_plot_accepts_single_latest_curve(tmp_path: Path):
    """current 模式应可生成单一当前可靠度图。"""
    rows = []
    for current in (10, 20):
        for step in range(1, 4):
            rows.append({"unit_id": 1, "current_time": current, "absolute_future_time": current + step, "reliability": 1-step/10, "true_failure_time": 30})
    out = tmp_path / "plot.png"
    plot_device_reliability(pd.DataFrame(rows), out, mode="current")
    assert out.is_file()
