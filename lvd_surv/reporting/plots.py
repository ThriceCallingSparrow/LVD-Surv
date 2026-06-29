"""visualize 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from lvd_surv.utils import ensure_dir


def plot_device_reliability(
    df: pd.DataFrame,
    path: str | Path,
    max_curves: int = 8,
    mode: str = "current",
) -> None:
    """绘制设备可靠度。

    ``current`` 只绘制最后观测周期，是默认且最直观的模式；``history`` 采样多个历史周期，
    旧曲线降低透明度并突出最新曲线；``snapshot`` 由上层按周期拆分数据后调用。
    """
    path = Path(path)
    ensure_dir(path.parent)
    unit = int(df["unit_id"].iloc[0])
    cycles = np.array(sorted(df["current_time"].unique()))
    if mode not in {"current", "history", "snapshot"}:
        raise ValueError(f"Unsupported plot mode: {mode}")
    if mode in {"current", "snapshot"}:
        cycles = cycles[-1:]
    elif len(cycles) > max_curves:
        cycles = cycles[np.linspace(0, len(cycles) - 1, max_curves).astype(int)]
    plt.figure(figsize=(10, 6))
    latest = int(cycles[-1])
    for c in cycles:
        g = df[df["current_time"] == c].sort_values("absolute_future_time")
        is_latest = int(c) == latest
        plt.plot(g["absolute_future_time"], g["reliability"], linewidth=2.6 if is_latest else 1.0,
                 alpha=1.0 if is_latest else 0.28, label=f"current t={int(c)}" if is_latest else None)
    failure_vals = df["true_failure_time"].dropna().unique()
    if len(failure_vals) > 0 and np.isfinite(failure_vals[0]):
        plt.axvline(float(failure_vals[0]), linestyle="--", linewidth=2.0, label="Actual failure time")
    plt.xlabel("Absolute cycle")
    plt.ylabel("Reliability")
    plt.title(f"Device {unit:03d} reliability ({mode})")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_mode_health(df: pd.DataFrame, path: str | Path) -> None:
    """执行 plot mode health 对应的项目处理逻辑。"""
    path = Path(path)
    ensure_dir(path.parent)
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.plot(df["current_time"], df["health_score"], label="Health/risk state")
    mode_cols = [c for c in df.columns if c.startswith("mode_prob_")]
    if mode_cols:
        dominant = df[mode_cols].values.argmax(axis=1)
        plt.scatter(df["current_time"], df["health_score"], c=dominant, s=12, label="Dominant mode")
    plt.xlabel("Cycle")
    plt.ylabel("Latent health/risk score")
    plt.title(f"Device {int(df['unit_id'].iloc[0]):03d} latent state")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
