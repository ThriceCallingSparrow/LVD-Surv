"""explain 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch

from lvd_surv.modeling.losses import discrete_survival_nll
from lvd_surv.utils import ensure_dir
from lvd_surv.runtime.cancellation import check_cancelled


@torch.no_grad()
def permutation_importance(model, loader, device, feature_columns: Sequence[str], repeats: int = 3) -> pd.DataFrame:
    """Global feature importance by NLL degradation after feature permutation."""
    model.eval()

    def eval_nll() -> float:
        """执行 eval nll 对应的项目处理逻辑。"""
        vals = []
        for b in loader:
            check_cancelled("置换重要性基线计算")
            x = b["x"].to(device)
            out = model(x, b["cycle"].to(device), sample=False)
            vals.append(float(discrete_survival_nll(out["hazard"], b["event"].to(device), b["mask"].to(device)).cpu()))
        return float(np.mean(vals))

    baseline = eval_nll()
    rows = []
    for j, name in enumerate(feature_columns):
        check_cancelled("置换重要性计算")
        scores = []
        for _ in range(repeats):
            vals = []
            for b in loader:
                check_cancelled("置换重要性计算")
                x = b["x"].clone()
                perm = torch.randperm(x.shape[0])
                x[:, :, j] = x[perm, :, j]
                out = model(x.to(device), b["cycle"].to(device), sample=False)
                vals.append(float(discrete_survival_nll(out["hazard"], b["event"].to(device), b["mask"].to(device)).cpu()))
            scores.append(float(np.mean(vals)))
        rows.append({"feature": name, "baseline_nll": baseline, "permuted_nll": np.mean(scores), "importance": np.mean(scores) - baseline})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)



def shap_gradient_importance(
    model,
    loader,
    device: torch.device,
    feature_columns: Sequence[str],
    sample_size: int = 128,
) -> pd.DataFrame:
    """Compute formal SHAP GradientExplainer attributions for model risk.

    SHAP is a required project dependency. Both sensor windows and observation
    cycle are supplied to the explainer; only sensor-window attributions are
    aggregated into the reported global feature importance.
    """
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is a required dependency. Install the project with `pip install -e .`.") from exc

    class _RiskWrapper(torch.nn.Module):
        """Expose one differentiable risk score per sample to SHAP."""
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, x: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
            out = self.wrapped(x, cycle.view(-1), sample=False)
            return (1.0 - out["reliability"][:, -1]).unsqueeze(-1)

    check_cancelled("SHAP 样本准备")
    xs, cycles = [], []
    for batch in loader:
        check_cancelled("SHAP 样本准备")
        xs.append(batch["x"])
        cycles.append(batch["cycle"].float().unsqueeze(-1))
        if sum(len(x) for x in xs) >= max(2, sample_size):
            break
    if not xs:
        raise ValueError("No explanation samples are available for SHAP.")
    x_all = torch.cat(xs, dim=0)[:sample_size].to(device)
    cycle_all = torch.cat(cycles, dim=0)[:sample_size].to(device)
    background_n = min(max(2, sample_size // 4), len(x_all))
    explain_n = min(sample_size, len(x_all))
    check_cancelled("SHAP 解释器初始化")
    wrapper = _RiskWrapper(model).to(device).eval()
    explainer = shap.GradientExplainer(wrapper, [x_all[:background_n], cycle_all[:background_n]])
    check_cancelled("SHAP 归因计算")
    values = explainer.shap_values([x_all[:explain_n], cycle_all[:explain_n]])
    check_cancelled("SHAP 归因汇总")
    if isinstance(values, list) and len(values) == 2:
        feature_values = values[0]
    elif isinstance(values, list) and len(values) == 1:
        feature_values = values[0][0] if isinstance(values[0], list) else values[0]
    else:
        feature_values = values[0] if isinstance(values, tuple) else values
    arr = np.asarray(feature_values)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 3:
        raise ValueError(f"Unexpected SHAP feature attribution shape: {arr.shape}")
    importance = np.mean(np.abs(arr), axis=(0, 1))
    return pd.DataFrame({"feature": list(feature_columns), "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    )

def save_shap_importance_plot(df: pd.DataFrame, path: str | Path, top_k: int = 20) -> None:
    """Save a global mean-absolute-SHAP feature ranking."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    path = Path(path)
    ensure_dir(path.parent)
    g = df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.3 * len(g))))
    plt.barh(g["feature"], g["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP feature importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

def save_feature_importance_plot(df: pd.DataFrame, path: str | Path, top_k: int = 20) -> None:
    """执行 save feature importance plot 对应的项目处理逻辑。"""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path = Path(path)
    ensure_dir(path.parent)
    g = df.head(top_k).iloc[::-1]
    plt.figure(figsize=(8, max(4, 0.3 * len(g))))
    plt.barh(g["feature"], g["importance"])
    plt.xlabel("NLL increase after permutation")
    plt.title("Global feature importance")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
