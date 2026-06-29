"""explanation 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""模型解释流程。

解释阶段严格复用 checkpoint 保存的特征契约、残差变换器和 scaler。训练文件可以显式
传入；未传入时读取 checkpoint 配置中的 ``data.train_file``，不会猜测其他文件。
"""

from pathlib import Path
from typing import Optional

from torch.utils.data import DataLoader

from lvd_surv.data.cmapss import SlidingWindowSurvivalDataset, add_train_rul, read_cmapss_txt
from lvd_surv.interpretation.explain import (
    permutation_importance,
    save_feature_importance_plot,
    save_shap_importance_plot,
    shap_gradient_importance,
)
from lvd_surv.core.contracts import apply_feature_decision
from lvd_surv.modeling.inference import apply_checkpoint_scaler, load_model_checkpoint, _load_feature_transformer, _minimal_feature_decision_from_checkpoint
from lvd_surv.runtime.context import RuntimeContext
from lvd_surv.utils import ensure_dir


def run_explanation_pipeline(
    checkpoint: str | Path,
    *,
    train_file: Optional[str | Path] = None,
    output_dir: str | Path = "outputs/explanations",
    repeats: int = 3,
    shap_sample_size: Optional[int] = None,
) -> Path:
    """生成置换重要性和正式 SHAP 解释，并返回输出目录。"""
    model, ckpt, device = load_model_checkpoint(checkpoint)
    shap_sample_size = int(shap_sample_size or ckpt.get("cfg", {}).get("explain", {}).get("shap_sample_size", 128))
    cfg = ckpt.get("cfg", {})
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    project_root = checkpoint_path
    while project_root.parent != project_root and project_root.name != "outputs":
        project_root = project_root.parent
    project_root = project_root.parent if project_root.name == "outputs" else Path.cwd()
    configured_train = cfg.get("data", {}).get("train_file")
    selected_train = Path(train_file).expanduser() if train_file else Path(configured_train or "")
    if not selected_train.is_absolute():
        selected_train = (project_root / selected_train).resolve()
    if not selected_train.is_file():
        raise FileNotFoundError(
            f"Explanation requires the training file, but it was not found: {selected_train}. "
            "Pass --train-file or keep data.train_file in the checkpoint configuration."
        )
    raw = add_train_rul(read_cmapss_txt(selected_train))
    # 解释阶段复用 checkpoint 特征契约，但保留训练数据中的真实 RUL。
    decision = ckpt.get("feature_decision") or _minimal_feature_decision_from_checkpoint(ckpt)
    transformer = _load_feature_transformer(cfg, ckpt, decision)
    applied = apply_feature_decision(raw, decision, transformer=transformer, stage="explanation")
    feature_cols = list(applied.feature_cols)
    checkpoint_features = list(ckpt.get("feature_columns") or [])
    if feature_cols != checkpoint_features:
        raise ValueError(f"Explanation feature contract mismatch: {feature_cols} != {checkpoint_features}")
    prepared = apply_checkpoint_scaler(applied.dataframe, feature_cols, ckpt)
    data_cfg = cfg.get("data", {})
    dataset = SlidingWindowSurvivalDataset(
        prepared,
        feature_cols,
        int(data_cfg.get("window_size", 50)),
        int(data_cfg.get("max_horizon", 150)),
        stride=max(1, int(data_cfg.get("window_size", 50)) // 2),
    )
    if len(dataset) == 0:
        raise ValueError("Training data cannot produce an explanation window; reduce data.window_size.")
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    out = ensure_dir(output_dir)
    importance = permutation_importance(model, loader, device, feature_cols, repeats=repeats)
    importance.to_csv(out / "global_feature_importance.csv", index=False)
    save_feature_importance_plot(importance, out / "global_feature_importance.png")
    shap_importance = shap_gradient_importance(
        model, loader, device, feature_cols, sample_size=shap_sample_size
    )
    shap_importance.to_csv(out / "global_shap_importance.csv", index=False)
    save_shap_importance_plot(shap_importance, out / "global_shap_importance.png")
    return out


def run(cfg, checkpoint: str | Path, output_dir: str | Path) -> Path:
    """Use active explanation settings with the canonical explanation workflow."""
    explain = cfg.get("explain", {})
    return run_explanation_pipeline(
        checkpoint,
        train_file=cfg.get("data", {}).get("train_file"),
        output_dir=output_dir,
        repeats=int(explain.get("permutation_repeats", 3)),
        shap_sample_size=int(explain.get("shap_sample_size", 128)),
    )
