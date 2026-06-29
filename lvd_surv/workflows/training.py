"""training 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""训练主流程编排"""

from pathlib import Path
from typing import Any, Mapping

from lvd_surv.features.analysis import get_or_build_feature_artifacts
from lvd_surv.lifetime.analysis import get_or_build_lifetime_prior
from lvd_surv.core.artifacts import build_artifact_paths, get_dataset_name, update_run_state
from lvd_surv.core.contracts import apply_feature_decision
from lvd_surv.data.cmapss import CMAPSS_SENSOR_COLS, add_train_rul, normalize_cmapss_schema
from lvd_surv.modeling.trainer import run_training


def _load_training_bundle(cfg: Mapping[str, Any]) -> dict:
    """从配置中明确指定的 C-MAPSS 文本文件加载训练数据。

    该函数不再切换到另一套加载器。缺失任何声明文件都会立即报告路径错误。
    """
    from lvd_surv.data.cmapss import read_cmapss_txt, read_rul_txt
    import pandas as pd
    data_cfg = dict(cfg.get("data", {}))
    if not data_cfg.get("train_file"):
        raise ValueError("data.train_file is required in strict mode.")
    train_df = normalize_cmapss_schema(read_cmapss_txt(data_cfg["train_file"]), add_condition_from_ops=True)
    test_df = normalize_cmapss_schema(read_cmapss_txt(data_cfg["test_file"]), add_condition_from_ops=True) if data_cfg.get("test_file") else None
    rul_df = None
    if data_cfg.get("rul_file"):
        rul_series = read_rul_txt(data_cfg["rul_file"])
        rul_df = pd.DataFrame({"RUL": rul_series.values})
    dataset = data_cfg.get("dataset") or data_cfg.get("fd") or "dataset"
    return {"train_df": train_df, "test_df": test_df, "rul_df": rul_df, "dataset": str(dataset)}


def run_training_pipeline(cfg: dict) -> Path:
    """运行 integrated 训练主流程"""
    paths = build_artifact_paths(cfg)
    dataset = get_dataset_name(cfg)
    update_run_state(paths["run_state"], stage="data_loading", completed={"data_loading": False})
    bundle = _load_training_bundle(cfg)
    train_df = add_train_rul(normalize_cmapss_schema(bundle["train_df"], add_condition_from_ops=True))
    update_run_state(paths["run_state"], stage="data_loading", completed={"data_loading": True}, dataset=dataset)
    sensor_cols = [c for c in CMAPSS_SENSOR_COLS if c in train_df.columns]
    feature_artifacts = get_or_build_feature_artifacts(cfg, train_df, sensor_cols, paths)
    feature_decision = feature_artifacts["feature_decision"]
    transformer = feature_artifacts.get("transformer")
    applied = apply_feature_decision(train_df, feature_decision, transformer=transformer, stage="train")
    prior_artifacts = get_or_build_lifetime_prior(cfg, train_df, paths)
    mixture_prior = prior_artifacts.get("mixture_prior")
    update_run_state(paths["run_state"], stage="training", completed={"training": False})
    ckpt = run_training(
        cfg,
        train_df=applied.dataframe,
        feature_cols=applied.feature_cols,
        feature_decision=feature_decision,
        mixture_prior=mixture_prior,
        feature_transformer_path=str(paths["feature_transformer"]) if transformer is not None else None,
    )
    update_run_state(paths["run_state"], stage="training", completed={"training": True}, checkpoint=str(ckpt))
    return ckpt


def _training_runner(cfg: dict):
    """Testable indirection for the canonical training workflow."""
    return run_training_pipeline(cfg)


def run(cfg: Mapping[str, Any]) -> Path:
    """Desktop/CLI convenience entry that delegates to the canonical workflow."""
    return Path(_training_runner(dict(cfg)))
