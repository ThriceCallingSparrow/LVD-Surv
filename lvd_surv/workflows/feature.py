"""正式特征分析工作流适配器。"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping


def run(cfg: Mapping[str, Any], *, rebuild: bool = False) -> Path:
    """运行现有正式特征算法；依赖在执行时加载，GUI 启动不提前触发重依赖。"""
    from lvd_surv.features.analysis import get_or_build_feature_artifacts
    from lvd_surv.core.artifacts import build_artifact_paths
    from lvd_surv.data.cmapss import CMAPSS_SENSOR_COLS, add_train_rul, read_cmapss_txt
    config = dict(cfg)
    config["features"] = dict(cfg.get("features", {}))
    if rebuild:
        config["features"]["cache_policy"] = "force"
    train_file = config.get("data", {}).get("train_file")
    if not train_file:
        raise ValueError("data.train_file is required")
    frame = add_train_rul(read_cmapss_txt(train_file))
    paths = build_artifact_paths(config)
    sensors = [c for c in CMAPSS_SENSOR_COLS if c in frame.columns]
    get_or_build_feature_artifacts(config, frame, sensors, paths)
    return Path(paths["feature_decision"])
