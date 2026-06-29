"""配置、数据、模型与预测结果验证工作流。"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import torch

from lvd_surv.modeling.inference import load_model_checkpoint
from lvd_surv import __release_status__, __version__


def check_dependencies() -> list[str]:
    """返回缺失或导入失败的正式依赖列表。"""
    failures = []
    for name in ("numpy", "pandas", "sklearn", "torch", "scipy", "matplotlib", "yaml", "dcor", "xgboost", "shap"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {exc}")
    return failures


def validate_config(cfg: Mapping[str, Any]) -> None:
    """验证正式数据文件和关键数值配置。"""
    data = cfg.get("data", {})
    for key in ("train_file", "test_file", "rul_file"):
        path = Path(str(data.get(key, "")))
        if not path.is_file():
            raise FileNotFoundError(f"{key} 不存在: {path}")
    window_size = int(data.get("window_size", 0))
    horizon = int(data.get("max_horizon", 0))
    split = float(data.get("train_split", 0))
    if window_size <= 0 or horizon <= 0:
        raise ValueError("window_size 和 max_horizon 必须大于 0")
    if not 0 < split < 1:
        raise ValueError("train_split 必须位于 (0,1)")
    training = cfg.get("training", {})
    if int(training.get("epochs", 0)) <= 0 or int(training.get("batch_size", 0)) <= 0:
        raise ValueError("training.epochs 和 training.batch_size 必须大于 0")
    if int(cfg.get("inference", {}).get("mc_samples", 0)) <= 0:
        raise ValueError("inference.mc_samples 必须大于 0")
    if int(cfg.get("explain", {}).get("shap_sample_size", 0)) <= 0:
        raise ValueError("explain.shap_sample_size 必须大于 0")


def validate_model(checkpoint: str | Path) -> Mapping[str, Any]:
    """验证 checkpoint 可加载且具备严格特征契约，并返回其字典。"""
    _, ckpt, _ = load_model_checkpoint(checkpoint)
    for key in ("model_config", "model_state", "feature_columns", "feature_decision", "scaler_center", "scaler_scale", "cfg"):
        if key not in ckpt:
            raise ValueError(f"checkpoint 缺少字段: {key}")
    if not ckpt.get("feature_columns"):
        raise ValueError("checkpoint.feature_columns 为空")
    if len(ckpt["feature_columns"]) != len(ckpt["scaler_center"]) or len(ckpt["feature_columns"]) != len(ckpt["scaler_scale"]):
        raise ValueError("checkpoint 特征数量与 scaler 维度不一致")
    return ckpt


def validate_checkpoint_compatibility(checkpoint: str | Path, cfg: Optional[Mapping[str, Any]]) -> None:
    """拒绝把明显属于另一数据集或另一特征配置的模型绑定到当前会话。"""
    ckpt = validate_model(checkpoint)
    if not cfg:
        return
    active_data = cfg.get("data", {})
    saved_data = ckpt.get("cfg", {}).get("data", {})
    active_dataset = str(active_data.get("dataset") or active_data.get("fd") or "")
    saved_dataset = str(saved_data.get("dataset") or saved_data.get("fd") or "")
    if active_dataset and saved_dataset and active_dataset != saved_dataset:
        raise ValueError(
            f"checkpoint 数据集不兼容: 当前={active_dataset}, checkpoint={saved_dataset}。"
            "请加载匹配配置或选择正确模型。"
        )
    active_window = int(active_data.get("window_size", 0) or 0)
    saved_window = int(saved_data.get("window_size", 0) or 0)
    active_horizon = int(active_data.get("max_horizon", 0) or 0)
    saved_horizon = int(saved_data.get("max_horizon", 0) or 0)
    if active_window and saved_window and active_window != saved_window:
        raise ValueError(f"checkpoint window_size 不兼容: 当前={active_window}, checkpoint={saved_window}")
    if active_horizon and saved_horizon and active_horizon != saved_horizon:
        raise ValueError(f"checkpoint max_horizon 不兼容: 当前={active_horizon}, checkpoint={saved_horizon}")


def validate_prediction(path: str | Path) -> None:
    """验证预测 CSV 的必要列、值域和基本单调性。"""
    frame = pd.read_csv(path)
    required = {"unit_id", "current_time", "future_step", "hazard", "reliability"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"预测结果缺少列: {sorted(missing)}")
    if frame.empty:
        raise ValueError("预测结果为空")
    if not frame["reliability"].between(0, 1).all():
        raise ValueError("reliability 存在 [0,1] 之外的值")
    if not frame["hazard"].between(0, 1).all():
        raise ValueError("hazard 存在 [0,1] 之外的值")
    ordered = frame.sort_values(["unit_id", "current_time", "future_step"])
    increases = ordered.groupby(["unit_id", "current_time"])["reliability"].diff().fillna(0) > 1e-6
    if bool(increases.any()):
        raise ValueError("预测可靠度存在随未来步长显著上升的曲线")


_REQUIRED_PROJECT_FILES = (
    "main.py",
    "pyproject.toml",
    "README.md",
    "USER_TEST_CHECKLIST.md",
    "configs/default.yaml",
    "lvd_surv/app/desktop.py",
    "lvd_surv/workflows/training.py",
    "lvd_surv/workflows/prediction.py",
    "lvd_surv/workflows/explanation.py",
    "lvd_surv/modeling/model.py",
    "lvd_surv/features/analysis.py",
    "lvd_surv/lifetime/analysis.py",
)


def validate_project_layout(project_root: str | Path) -> None:
    """Validate files required by the supported GUI, CLI, and algorithm workflows.

    This is a release-structure check only. It does not run model training or alter
    the numerical workflow. Missing obsolete compatibility modules are not errors.
    """
    root = Path(project_root).resolve()
    missing = [name for name in _REQUIRED_PROJECT_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"项目发布结构缺少文件: {missing}")
    forbidden = [
        "lvd_surv/model.py", "lvd_surv/infer.py", "lvd_surv/train.py",
        "lvd_surv/cmapss", "lvd_surv/pipelines",
        "scripts/train_lvd_surv.py", "scripts/predict_lvd_surv.py",
        "scripts/explain_lvd_surv.py",
    ]
    remaining = [name for name in forbidden if (root / name).exists()]
    if remaining:
        raise ValueError(f"项目仍包含已淘汰兼容入口: {remaining}")


def release_readiness(
    cfg: Mapping[str, Any],
    project_root: str | Path,
    *,
    require_dependencies: bool = True,
) -> dict[str, Any]:
    """Run deterministic release checks and return a machine-readable summary.

    The check covers package layout, normalized configuration, data files, output
    writability, and optionally all mandatory third-party dependencies. It does not
    claim model-accuracy acceptance; the 60-epoch numerical comparison remains a
    separate user acceptance step.
    """
    root = Path(project_root).resolve()
    validate_project_layout(root)
    validate_config(cfg)
    failures = check_dependencies() if require_dependencies else []
    if failures:
        raise RuntimeError("正式依赖检查失败:\n" + "\n".join(failures))
    output = Path(str(cfg.get("project", {}).get("output_dir") or cfg.get("data", {}).get("output_dir", "")))
    output.mkdir(parents=True, exist_ok=True)
    probe = output / ".lvd_write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
    finally:
        if probe.exists():
            probe.unlink()
    return {
        "project_root": str(root),
        "dataset": str(cfg.get("data", {}).get("dataset", "")),
        "output_dir": str(output.resolve()),
        "dependencies_checked": require_dependencies,
        "package_version": __version__,
        "release_status": __release_status__,
        "gui_entry": "python main.py",
        "cli_entry": "lvd",
        "status": "ready",
    }
