"""contracts 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""特征决策契约与寿命先验契约"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lvd_surv.data.cmapss import CMAPSS_SENSOR_COLS, normalize_cmapss_schema
from lvd_surv.features.transformer import build_residual_transformer


TRAINING_REQUIRED_FEATURE_DECISION_KEYS = {
    "schema_version",
    "training_feature_mode",
    "selected_features",
    "raw_selected_features",
    "residual_selected_features",
}


@dataclass
class DecisionApplicationResult:
    """Return object for applying a feature decision."""

    dataframe: pd.DataFrame
    feature_cols: List[str]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _feature_profile_from_result(feature_result: Any) -> Optional[pd.DataFrame]:
    """Extract a feature profile dataframe from known user-script outputs."""
    if isinstance(feature_result, pd.DataFrame):
        return feature_result
    if isinstance(feature_result, Mapping):
        for key in ("feature_profile", "vi_plan", "summary"):
            value = feature_result.get(key)
            if isinstance(value, pd.DataFrame):
                return value
    return None


def _rank_features_from_profile(profile: pd.DataFrame, sensor_cols: Sequence[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Rank/select features from a Chinese or English feature-profile table"""
    if profile is None or profile.empty:
        return list(sensor_cols), {}

    feature_col = "传感器列" if "传感器列" in profile.columns else None
    if feature_col is None:
        for candidate in ("feature", "sensor", "sensor_col", "column"):
            if candidate in profile.columns:
                feature_col = candidate
                break
    if feature_col is None:
        return list(sensor_cols), {}

    rank_words = {
        "核心": 5,
        "强": 5,
        "推荐保留": 5,
        "高": 4,
        "建议保留": 4,
        "中": 3,
        "谨慎保留": 2,
        "弱": 1,
        "剔除": -10,
        "删除": -10,
        "drop": -10,
    }
    selected: List[str] = []
    decisions: Dict[str, Dict[str, Any]] = {}

    for _, row in profile.iterrows():
        feat = str(row.get(feature_col, "")).strip()
        if feat not in sensor_cols:
            continue
        text = " ".join(str(row.get(c, "")) for c in profile.columns)
        score = 0
        for word, weight in rank_words.items():
            if word.lower() in text.lower():
                score += weight
        action = "drop" if score < 0 else "use_raw"
        # First-stage integration is intentionally conservative: complex
        # residual/hybrid actions are preserved for later but not auto-enabled
        # until a reusable feature_transformer is implemented.
        if action != "drop":
            selected.append(feat)
        decisions[feat] = {
            "action": action,
            "score": float(score),
            "confidence": float(max(0.0, min(1.0, 0.5 + score / 10.0))),
            "risk_level": "high" if action == "drop" else "medium" if score < 3 else "low",
            "source_row": {str(k): str(row.get(k, "")) for k in profile.columns},
        }

    if not selected:
        raise ValueError("Feature analysis selected no sensors; adjust analysis thresholds and rerun `lvd feature analyze`.")
    # Preserve original order from sensor_cols for stable training dimensions.
    selected = [c for c in sensor_cols if c in set(selected)]
    return selected, decisions


def build_feature_decision(
    *,
    feature_result: Any,
    train_df: pd.DataFrame,
    sensor_cols: Optional[Sequence[str]] = None,
    cfg: Optional[Mapping[str, Any]] = None,
    dataset: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a stable feature decision JSON contract.

    Stage-1 behavior: use raw features by default.  The decision contract is
    already shaped for future residual/hybrid modes, but this first-stage code
    will not silently request residual columns unless a later feature transformer
    is implemented and explicitly enabled.
    """
    df = normalize_cmapss_schema(train_df, add_condition_from_ops=True)
    sensors = list(sensor_cols or [c for c in CMAPSS_SENSOR_COLS if c in df.columns])
    if not sensors:
        raise ValueError("No sensor columns are available for feature decision building.")
    profile = _feature_profile_from_result(feature_result)
    raw_selected, feature_decisions = _rank_features_from_profile(profile, sensors)

    fs_cfg = dict((cfg or {}).get("features", {}) if isinstance(cfg, Mapping) else {})
    mode = str(fs_cfg.get("default_training_feature_mode", "raw")).lower()
    if mode not in {"raw", "residual", "hybrid"}:
        mode = "raw"
    # Phase 4: residual/hybrid modes are now allowed when requested.  The
    # pipeline will fit/persist a feature_transformer.pkl and inference will
    # require the same transformer before prediction.
    residual_suffix = str(fs_cfg.get("residual_suffix", "_resid"))
    residual_selected = [f"{c}{residual_suffix}" for c in raw_selected]
    if mode == "raw":
        selected = list(raw_selected)
    elif mode == "residual":
        selected = list(residual_selected)
    else:  # hybrid
        selected = list(raw_selected) + list(residual_selected)

    decision = {
        "schema_version": "1.0",
        "dataset": dataset or str((cfg or {}).get("data", {}).get("fd", "unknown") if isinstance(cfg, Mapping) else "unknown"),
        "target_col": "rul",
        "unit_id_col": "unit_id",
        "time_col": "cycle",
        "decision_mode": "phase4_transformer_contract",
        "training_feature_mode": mode,
        "selected_features": selected,
        "raw_selected_features": list(raw_selected),
        "residual_selected_features": list(residual_selected),
        "dropped_features": [c for c in sensors if c not in set(raw_selected)],
        "feature_decisions": feature_decisions,
        "decision_summary": {
            "num_available_sensors": len(sensors),
            "num_selected": len(selected),
            "num_dropped": len([c for c in sensors if c not in set(raw_selected)]),
            "mode": mode,
                "stage": "phase_4_residual_hybrid_transformer_contract",
        },
    }
    validate_feature_decision(decision)
    return decision


def validate_feature_decision(decision: Mapping[str, Any]) -> None:
    """Validate the minimal schema consumed by training and inference."""
    missing = TRAINING_REQUIRED_FEATURE_DECISION_KEYS.difference(decision.keys())
    if missing:
        raise ValueError(f"Feature decision is missing required keys: {sorted(missing)}")
    mode = decision.get("training_feature_mode")
    if mode not in {"raw", "residual", "hybrid"}:
        raise ValueError(f"Unsupported training_feature_mode: {mode}")
    selected = decision.get("selected_features") or []
    if not isinstance(selected, list):
        raise ValueError("feature_decision['selected_features'] must be a list")


def apply_feature_decision(
    df: pd.DataFrame,
    feature_decision: Mapping[str, Any],
    transformer: Optional[Any] = None,
    *,
    stage: str = "train",
) -> DecisionApplicationResult:
    """Apply a feature decision to a dataframe and return selected columns.

    The same function must be used by training and inference so that the model
    sees identical feature semantics in both phases.
    """
    validate_feature_decision(feature_decision)
    out_df = normalize_cmapss_schema(df, add_condition_from_ops=True)
    mode = str(feature_decision.get("training_feature_mode", "raw")).lower()

    if mode == "raw":
        feature_cols = list(feature_decision.get("raw_selected_features") or feature_decision.get("selected_features") or [])
    elif mode == "residual":
        if transformer is None:
            raise ValueError("Residual feature mode requires feature_transformer.pkl, but no transformer was provided.")
        out_df = transformer.transform(out_df)
        feature_cols = list(feature_decision.get("residual_selected_features") or feature_decision.get("selected_features") or [])
    elif mode == "hybrid":
        if transformer is not None:
            out_df = transformer.transform(out_df)
        feature_cols = list(feature_decision.get("selected_features") or [])
    else:  # validate_feature_decision should already catch this.
        raise ValueError(f"Unsupported training_feature_mode: {mode}")

    missing = [c for c in feature_cols if c not in out_df.columns]
    if missing:
        raise ValueError(f"Cannot apply feature decision during {stage}; missing columns: {missing}")
    if not feature_cols:
        raise ValueError("Feature decision selected no columns; rerun strict feature analysis.")
    return DecisionApplicationResult(dataframe=out_df, feature_cols=feature_cols)



def fit_feature_transformer_for_decision(
    df: pd.DataFrame,
    feature_decision: Mapping[str, Any],
    cfg: Optional[Mapping[str, Any]] = None,
) -> Optional[Any]:
    """Fit the transformer required by residual/hybrid feature modes.

    Raw mode returns ``None`` because no extra feature construction is needed.
    Residual and hybrid modes fit a reusable transformer on training data.  The
    caller should persist it as ``feature_transformer.pkl`` and reuse it during
    inference.
    """
    validate_feature_decision(feature_decision)
    mode = str(feature_decision.get("training_feature_mode", "raw")).lower()
    if mode == "raw":
        return None
    raw_features = list(feature_decision.get("raw_selected_features") or [])
    if not raw_features:
        raise ValueError("Residual/hybrid feature mode requires raw_selected_features.")
    transformer = build_residual_transformer(raw_feature_cols=raw_features, cfg=cfg)
    transformer.fit(df)
    return transformer


def mixture_prior_contract_from_user_result(result: Any, *, dataset: str = "unknown") -> Optional[Dict[str, Any]]:
    """Serialize the user's select_best_mixture result into JSON-safe form."""
    if result is None:
        return None
    try:
        best_model, best_bic, best_dist_combo, best_k = result
    except Exception:
        return None

    weights: List[float]
    params_payload: List[Dict[str, Any]] = []
    try:
        pi, params = best_model.get_params()
        weights = [float(x) for x in np.asarray(pi, dtype=float).reshape(-1)]
        for idx, item in enumerate(params):
            # User script returns either list/tuple params or (name, params), depending
            # on version.  Normalize both forms.
            if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                name = item[0]
                values = item[1]
            else:
                name = list(best_dist_combo)[idx] if idx < len(best_dist_combo) else f"component_{idx}"
                values = item
            params_payload.append({"name": str(name), "values": [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]})
    except Exception:
        weights = []
        params_payload = []

    return {
        "schema_version": "1.0",
        "dataset": dataset,
        "best_k": int(best_k),
        "best_bic": float(best_bic),
        "dist_combo": [str(x) for x in list(best_dist_combo)],
        "weights": weights,
        "params": params_payload,
        "enabled": True,
        "source": "lvd_surv.lifetime.analysis.select_best_mixture",
    }
