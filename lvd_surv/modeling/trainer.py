"""train 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lvd_surv.data.cmapss import (
    CMAPSS_SENSOR_COLS,
    add_train_rul,
    fit_transform_features,
    read_cmapss_txt,
    normalize_cmapss_schema,
    split_by_unit,
    SlidingWindowSurvivalDataset,
)
from lvd_surv.modeling.losses import total_loss
from lvd_surv.modeling.model import LVDSurvModel, ModelConfig
from lvd_surv.utils import count_parameters, ensure_dir, save_json, set_seed
from lvd_surv.runtime.cancellation import check_cancelled


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def train_one_epoch(model, loader, optimizer, device, cfg, epoch: int) -> Dict[str, float]:
    """执行 train one epoch 对应的项目处理逻辑。"""
    model.train()
    beta_final = float(cfg["training"].get("beta_kl_final", 0.01))
    anneal = max(1, int(cfg["training"].get("kl_anneal_epochs", 20)))
    beta_kl = beta_final * min(1.0, epoch / anneal)
    totals: Dict[str, float] = {}
    n = 0
    gui_mode = bool(cfg.get("runtime", {}).get("gui_mode", False))
    for batch in tqdm(loader, desc=f"train epoch {epoch}", leave=False, disable=gui_mode):
        check_cancelled(f"训练 epoch {epoch}")
        batch = _to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["cycle"], sample=True)
        losses = total_loss(
            out,
            batch,
            beta_kl=beta_kl,
            lambda_mono=float(cfg["training"].get("lambda_mono", 0.02)),
            lambda_cond=float(cfg["training"].get("lambda_cond", 0.2)),
            lambda_orth=float(cfg["training"].get("lambda_orth", 0.01)),
            use_reconstruction=bool(cfg["training"].get("use_reconstruction", False)),
            lambda_recon=float(cfg["training"].get("lambda_recon", 0.0)),
        )
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"].get("gradient_clip_norm", 5.0)))
        optimizer.step()
        bs = batch["x"].shape[0]
        n += bs
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bs
    return {k: v / max(1, n) for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> Dict[str, float]:
    """执行 evaluate 对应的项目处理逻辑。"""
    model.eval()
    totals: Dict[str, float] = {}
    n = 0
    for batch in loader:
        check_cancelled("模型验证")
        batch = _to_device(batch, device)
        out = model(batch["x"], batch["cycle"], sample=False)
        losses = total_loss(
            out,
            batch,
            beta_kl=float(cfg["training"].get("beta_kl_final", 0.01)),
            lambda_mono=float(cfg["training"].get("lambda_mono", 0.02)),
            lambda_cond=float(cfg["training"].get("lambda_cond", 0.2)),
            lambda_orth=float(cfg["training"].get("lambda_orth", 0.01)),
        )
        bs = batch["x"].shape[0]
        n += bs
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().cpu()) * bs
    return {"val_" + k: v / max(1, n) for k, v in totals.items()}



def prepare_training_from_dataframe(
    cfg: Dict,
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    val_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, list, object]:
    """Prepare training data from a pipeline-provided dataframe.

    Integrated mode has already made the expensive/complex decisions: which
    loader to use, which features to keep, and which prior artifacts to attach.
    Therefore this function must not reselect features.  It only validates,
    splits, scales, and returns data ready for ``SlidingWindowSurvivalDataset``.
    """
    if train_df is None:
        raise ValueError("train_df is required for integrated training.")
    feature_cols = list(feature_cols or [])
    if not feature_cols:
        raise ValueError("feature_cols must be provided by the integrated pipeline.")

    data_cfg = cfg["data"]
    df = normalize_cmapss_schema(train_df, add_condition_from_ops=True)
    if "failure_time" not in df.columns or "rul" not in df.columns:
        df = add_train_rul(df)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Integrated feature columns are missing from train_df: {missing}")
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if val_df is None:
        train_part, val_part = split_by_unit(df, data_cfg.get("train_split", 0.85), data_cfg.get("random_seed", 42))
    else:
        train_part = df.copy()
        val_part = normalize_cmapss_schema(val_df, add_condition_from_ops=True)
        if "failure_time" not in val_part.columns or "rul" not in val_part.columns:
            val_part = add_train_rul(val_part)
        missing_val = [c for c in feature_cols if c not in val_part.columns]
        if missing_val:
            raise ValueError(f"Integrated feature columns are missing from val_df: {missing_val}")
        for col in feature_cols:
            val_part[col] = pd.to_numeric(val_part[col], errors="raise")

    if val_part.empty:
        # Extremely small synthetic datasets can produce an empty validation split.
        # In that case use a copy of the training split so smoke tests can still
        # exercise the full training path. Real CMAPSS folds should not hit this.
        val_part = train_part.copy()

    train_part, val_part, scaler = fit_transform_features(train_part, val_part, feature_cols)
    return train_part, val_part, list(feature_cols), scaler


def run_training(
    cfg: Dict,
    train_df: Optional[pd.DataFrame] = None,
    val_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[Sequence[str]] = None,
    feature_decision: Optional[Dict[str, Any]] = None,
    mixture_prior: Optional[Dict[str, Any]] = None,
    feature_transformer_path: Optional[str] = None,
) -> Path:
    """执行 run training 对应的项目处理逻辑。"""
    set_seed(int(cfg["data"].get("random_seed", 42)))
    out_dir = ensure_dir(cfg["data"].get("output_dir", "outputs"))
    ckpt_dir = ensure_dir(out_dir / "checkpoints")
    if train_df is None or feature_cols is None:
        raise ValueError("Strict training requires train_df and feature_cols from run_training_pipeline().")
    train_df, val_df, feature_cols, scaler = prepare_training_from_dataframe(
        cfg, train_df=train_df, val_df=val_df, feature_cols=feature_cols
    )
    horizon = int(cfg["data"].get("max_horizon", 150))
    window = int(cfg["data"].get("window_size", 50))
    cond_col = "condition" if bool(cfg["data"].get("known_condition", False)) and "condition" in train_df.columns else None
    train_ds = SlidingWindowSurvivalDataset(train_df, feature_cols, window, horizon, int(cfg["data"].get("stride", 1)), condition_label_column=cond_col)
    val_ds = SlidingWindowSurvivalDataset(val_df, feature_cols, window, horizon, int(cfg["data"].get("stride", 1)), condition_label_column=cond_col)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"].get("batch_size", 128)),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
    )
    val_loader = DataLoader(val_ds, batch_size=int(cfg["training"].get("batch_size", 128)), shuffle=False)
    mcfg = ModelConfig(
        input_dim=len(feature_cols),
        horizon=horizon,
        encoder_type=cfg["model"].get("encoder_type", "tcn"),
        hidden_dim=int(cfg["model"].get("hidden_dim", 96)),
        latent_dim=int(cfg["model"].get("latent_dim", 16)),
        num_modes=int(cfg["model"].get("num_modes", 6)),
        num_tcn_layers=int(cfg["model"].get("num_tcn_layers", 4)),
        tcn_kernel_size=int(cfg["model"].get("tcn_kernel_size", 3)),
        transformer_layers=int(cfg["model"].get("transformer_layers", 2)),
        transformer_heads=int(cfg["model"].get("transformer_heads", 4)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LVDSurvModel(mcfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 1e-3)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-4)),
    )
    best = float("inf")
    patience = int(cfg["training"].get("early_stop_patience", 12))
    bad = 0
    history = []
    for epoch in range(1, int(cfg["training"].get("epochs", 60)) + 1):
        check_cancelled("模型训练")
        tr = train_one_epoch(model, train_loader, optimizer, device, cfg, epoch)
        va = evaluate(model, val_loader, device, cfg)
        row = {"epoch": epoch, **tr, **va}
        history.append(row)
        print(row)
        if va["val_loss"] < best:
            best = va["val_loss"]
            bad = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": mcfg.__dict__,
                    "feature_columns": feature_cols,
                    "feature_decision": feature_decision,
                    "mixture_prior": mixture_prior,
                    "feature_transformer_path": feature_transformer_path,
                    "training_entry": "strict_integrated",
                    "scaler_center": scaler.center_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                    "cfg": cfg,
                },
                ckpt_dir / "best_model.pt",
            )
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    save_json({"parameters": count_parameters(model), "best_val_loss": best}, out_dir / "model_complexity.json")
    return ckpt_dir / "best_model.pt"
