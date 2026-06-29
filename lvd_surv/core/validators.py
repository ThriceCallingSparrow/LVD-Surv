"""validators 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""项目产物验证工具。\n\n中文注释：该模块集中验证 checkpoint、feature_decision、mixture_prior 和预测结果，避免验证逻辑散落在脚本中。\n"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def validate_feature_decision_contract(decision: Mapping[str, Any]) -> None:
    """验证 feature_decision 的最小训练/推理字段。"""
    required = {"schema_version", "training_feature_mode", "selected_features", "raw_selected_features", "residual_selected_features"}
    missing = required.difference(decision.keys())
    if missing:
        raise ValueError(f"Feature decision missing keys: {sorted(missing)}")
    if decision.get("training_feature_mode") not in {"raw", "residual", "hybrid"}:
        raise ValueError(f"Unsupported training_feature_mode: {decision.get('training_feature_mode')}")


def validate_mixture_prior_contract(prior: Mapping[str, Any] | None) -> None:
    """验证 mixture_prior 的基础结构。"""
    if prior is None:
        return
    if not isinstance(prior, Mapping):
        raise ValueError("mixture_prior must be a mapping or None")
    if prior.get("enabled", True) and "schema_version" not in prior:
        raise ValueError("Enabled mixture_prior must contain schema_version")


def validate_checkpoint_schema(ckpt: Mapping[str, Any]) -> None:
    """验证模型 checkpoint 是否包含推理所需字段。"""
    required = {"model_state", "model_config", "feature_columns", "scaler_center", "scaler_scale", "cfg"}
    missing = required.difference(ckpt.keys())
    if missing:
        raise ValueError(f"Checkpoint missing keys: {sorted(missing)}")
    features = list(ckpt.get("feature_columns") or [])
    if len(features) != len(ckpt.get("scaler_center") or []) or len(features) != len(ckpt.get("scaler_scale") or []):
        raise ValueError("Checkpoint feature/scaler dimension mismatch")
    if ckpt.get("feature_decision"):
        validate_feature_decision_contract(ckpt["feature_decision"])
    validate_mixture_prior_contract(ckpt.get("mixture_prior"))


def validate_prediction_frame(df: pd.DataFrame) -> None:
    """验证预测结果表的可靠度、风险和 RUL 数值范围。"""
    required = {"unit_id", "current_time", "future_step", "hazard", "reliability", "expected_rul_from_current"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")
    if not df["reliability"].between(0, 1).all():
        raise ValueError("Reliability values must be in [0, 1]")
    if not df["hazard"].between(0, 1).all():
        raise ValueError("Hazard values must be in [0, 1]")
    if (pd.to_numeric(df["expected_rul_from_current"], errors="coerce") < 0).any():
        raise ValueError("Expected RUL must be non-negative")
