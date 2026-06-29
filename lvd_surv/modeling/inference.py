"""infer 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""Inference utilities for LVD-Surv.

Phase 3 makes inference consume the same feature contract and mixture-prior
contract saved by integrated training.  This prevents the common deployment bug
where training uses one feature set but prediction silently reconstructs another.
"""

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lvd_surv.core.artifacts import build_artifact_paths, load_pickle_artifact
from lvd_surv.data.cmapss import (
    add_test_rul,
    normalize_cmapss_schema,
    read_cmapss_txt,
    read_rul_txt,
    SlidingWindowSurvivalDataset,
)
from lvd_surv.core.contracts import apply_feature_decision
from lvd_surv.modeling.losses import expected_rul_from_hazard, hazard_to_failure_pmf
from lvd_surv.lifetime.prior import build_prior_from_contract, blend_tail_with_prior
from lvd_surv.modeling.model import LVDSurvModel, ModelConfig
from lvd_surv.utils import ensure_dir
from lvd_surv.reporting.plots import plot_device_reliability, plot_mode_health
from lvd_surv.runtime.cancellation import check_cancelled


def load_model_checkpoint(path: str | Path, device: Optional[torch.device] = None):
    """Load a model checkpoint and return ``(model, checkpoint_dict, device)``."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device)
    mcfg = ModelConfig(**ckpt["model_config"])
    model = LVDSurvModel(mcfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, device


def _minimal_feature_decision_from_checkpoint(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """拒绝缺少正式特征契约的旧 checkpoint。"""
    raise ValueError(
        "Checkpoint has no feature_decision contract. Strict mode does not reconstruct or guess one; retrain the model."
    )


def _load_feature_transformer(cfg: Mapping[str, Any], ckpt: Mapping[str, Any], feature_decision: Mapping[str, Any]) -> Optional[Any]:
    """Load feature_transformer.pkl when residual/hybrid features require it."""
    mode = str(feature_decision.get("training_feature_mode", "raw")).lower()
    if mode == "raw":
        return None
    candidates = []
    if ckpt.get("feature_transformer_path"):
        candidates.append(ckpt["feature_transformer_path"])
    if isinstance(cfg, Mapping):
        candidates.append(build_artifact_paths(cfg)["feature_transformer"])
    for path in candidates:
        p = Path(path)
        if p.exists():
            return load_pickle_artifact(p)
    raise FileNotFoundError(
        "Checkpoint requires residual/hybrid features but feature_transformer.pkl was not found. "
        "Rebuild feature artifacts and retrain; strict mode will not continue with raw features."
    )


def load_inference_data(
    cfg: Mapping[str, Any],
    test_file: Optional[str | Path] = None,
    rul_file: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load test data using the same priority as integrated training.

    Explicit CLI paths take precedence. Otherwise the exact ``data.test_file`` and
    ``data.rul_file`` paths saved in the checkpoint configuration are used.
    """
    data_cfg = dict(cfg.get("data", {})) if isinstance(cfg, Mapping) else {}
    if test_file is not None:
        df = normalize_cmapss_schema(read_cmapss_txt(test_file), add_condition_from_ops=True)
        rul = read_rul_txt(rul_file) if rul_file else None
        return df, rul


    if data_cfg.get("test_file"):
        df = normalize_cmapss_schema(read_cmapss_txt(data_cfg["test_file"]), add_condition_from_ops=True)
        rpath = rul_file or data_cfg.get("rul_file")
        rul = read_rul_txt(rpath) if rpath else None
        return df, rul

    raise ValueError("No test data source was provided. Use --test-file or keep data.test_file in the checkpoint cfg.")


def apply_checkpoint_scaler(df: pd.DataFrame, feature_cols: Sequence[str], ckpt: Mapping[str, Any]) -> pd.DataFrame:
    """Scale inference features with the scaler saved during training."""
    df = df.copy()
    feature_cols = list(feature_cols)
    center = np.asarray(ckpt["scaler_center"], dtype=float)
    scale = np.asarray(ckpt["scaler_scale"], dtype=float)
    if len(center) != len(feature_cols) or len(scale) != len(feature_cols):
        raise ValueError(
            f"Checkpoint scaler dimension mismatch: center={len(center)}, scale={len(scale)}, features={len(feature_cols)}"
        )
    scale[scale == 0] = 1.0
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Inference dataframe is missing checkpoint feature columns: {missing}")
    df[feature_cols] = df.loc[:, feature_cols].astype(float)
    df[feature_cols] = (df.loc[:, feature_cols].values - center) / scale
    return df


def prepare_inference_dataframe(
    cfg: Mapping[str, Any],
    ckpt: Mapping[str, Any],
    test_df: pd.DataFrame,
    rul: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Sequence[str]]:
    """Apply the checkpoint feature contract to raw test data.

    The returned feature columns must exactly match ``checkpoint['feature_columns']``;
    otherwise the trained model input dimension would no longer be trustworthy.
    """
    df_raw = normalize_cmapss_schema(test_df, add_condition_from_ops=True)
    df_with_rul = add_test_rul(df_raw, rul)
    feature_decision = ckpt.get("feature_decision") or _minimal_feature_decision_from_checkpoint(ckpt)
    transformer = _load_feature_transformer(cfg, ckpt, feature_decision)
    applied = apply_feature_decision(df_with_rul, feature_decision, transformer=transformer, stage="inference")
    feature_cols = list(applied.feature_cols)
    ckpt_feature_cols = list(ckpt.get("feature_columns") or [])
    if feature_cols != ckpt_feature_cols:
        raise ValueError(
            "Inference feature contract does not match checkpoint feature_columns. "
            f"contract={feature_cols}, checkpoint={ckpt_feature_cols}"
        )
    scaled = apply_checkpoint_scaler(applied.dataframe, feature_cols, ckpt)
    return scaled, feature_cols


@torch.no_grad()
def predict_dataset(
    checkpoint: str | Path,
    test_file: Optional[str | Path] = None,
    rul_file: Optional[str | Path] = None,
    output_dir: str | Path = "outputs",
    mc_samples: int = 20,
    prior: Optional[object] = None,
    use_checkpoint_prior: bool = True,
    test_df: Optional[pd.DataFrame] = None,
    rul: Optional[pd.Series] = None,
    plot_max_curves_per_device: int = 8,
    plot_mode: str = "current",
) -> pd.DataFrame:
    """Predict device-level reliability curves.

    ``test_df``/``rul`` are primarily for tests or API usage.  CLI users can use
    explicit ``--test-file`` paths or ``--use-checkpoint-config`` to load from the
    checkpoint configuration.
    """
    model, ckpt, device = load_model_checkpoint(checkpoint)
    cfg = ckpt.get("cfg", {})
    if test_df is None:
        test_df, rul = load_inference_data(cfg, test_file=test_file, rul_file=rul_file)
    df, feature_cols = prepare_inference_dataframe(cfg, ckpt, test_df, rul=rul)

    cfg_prior = dict(cfg.get("lifetime_prior", {})) if isinstance(cfg, Mapping) else {}
    if prior is None and use_checkpoint_prior and bool(cfg_prior.get("use_in_inference", True)):
        prior = build_prior_from_contract(ckpt.get("mixture_prior"))
    blend_weight = float(cfg_prior.get("blend_weight", 1.0))

    ds = SlidingWindowSurvivalDataset(
        df,
        feature_cols,
        int(cfg.get("data", {}).get("window_size", 50)),
        int(cfg.get("data", {}).get("max_horizon", 150)),
        stride=1,
    )
    loader = DataLoader(ds, batch_size=256, shuffle=False)
    out_rows = []
    state_rows = []
    for batch in loader:
        check_cancelled("可靠度预测")
        x = batch["x"].to(device)
        cyc = batch["cycle"].to(device)
        hazards = []
        reliabilities = []
        modes = []
        healths = []
        for _ in range(max(1, mc_samples)):
            out = model(x, cyc, sample=mc_samples > 1)
            hazards.append(out["hazard"].cpu())
            reliabilities.append(out["reliability"].cpu())
            modes.append(out["mode_prob"].cpu())
            healths.append(out["health_score"].cpu())
        hazard = torch.stack(hazards).mean(0)
        reliability = torch.stack(reliabilities).mean(0)
        pmf = hazard_to_failure_pmf(hazard).cpu().numpy()
        exp_rul = expected_rul_from_hazard(hazard).cpu().numpy()
        mode_prob = torch.stack(modes).mean(0).numpy()
        health = torch.stack(healths).mean(0).numpy()
        rel_np = reliability.numpy()
        haz_np = hazard.numpy()
        for i in range(x.shape[0]):
            unit = int(batch["unit_id"][i])
            cycle = int(batch["cycle"][i])
            failure_time = float(batch["failure_time"][i])
            rel_i = blend_tail_with_prior(
                rel_np[i],
                cycle,
                prior,
                blend_start=int(0.7 * rel_np.shape[1]),
                blend_weight=blend_weight,
            )
            for step in range(rel_np.shape[1]):
                out_rows.append(
                    {
                        "unit_id": unit,
                        "current_time": cycle,
                        "future_step": step + 1,
                        "absolute_future_time": cycle + step + 1,
                        "hazard": float(haz_np[i, step]),
                        "failure_probability": float(pmf[i, step]),
                        "reliability": float(rel_i[step]),
                        "expected_rul_from_current": float(exp_rul[i]),
                        "true_failure_time": failure_time,
                    }
                )
            state = {"unit_id": unit, "current_time": cycle, "health_score": float(health[i])}
            for k in range(mode_prob.shape[1]):
                state[f"mode_prob_{k}"] = float(mode_prob[i, k])
            state_rows.append(state)
    out_dir = ensure_dir(output_dir)
    rel_dir = ensure_dir(out_dir / "reliability_values")
    plot_dir = ensure_dir(out_dir / "reliability_curves")
    state_dir = ensure_dir(out_dir / "states")
    pred = pd.DataFrame(out_rows)
    states = pd.DataFrame(state_rows)
    pred.to_csv(out_dir / "all_reliability_values.csv", index=False)
    states.to_csv(state_dir / "all_mode_health_states.csv", index=False)
    for unit, g in pred.groupby("unit_id"):
        check_cancelled("预测结果保存与绘图")
        g.to_csv(rel_dir / f"device_{int(unit):03d}_reliability.csv", index=False)
        sg = states[states["unit_id"] == unit]
        sg.to_csv(state_dir / f"device_{int(unit):03d}_mode_health.csv", index=False)
        plot_device_reliability(g, plot_dir / f"device_{int(unit):03d}_reliability.png", max_curves=plot_max_curves_per_device, mode=plot_mode)
        plot_mode_health(sg, state_dir / f"device_{int(unit):03d}_mode_health.png")
    return pred
