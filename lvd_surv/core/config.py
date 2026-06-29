"""项目配置的唯一加载、规范化与会话覆盖入口。

用户配置采用简化的公开结构；算法内部仍接收稳定的规范化结构。这样可以删除
内部产物路径等容易配置错误的选项，同时不修改训练、特征、寿命先验或推理算法。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping

from lvd_surv.core.errors import LVDWorkflowError
from lvd_surv.utils import load_yaml_like

_INTERNAL_FEATURE_PATH_KEYS = {
    "decision_path",
    "bundle_path",
    "transformer_path",
    "manifest_path",
    "analysis_output_dir",
}
_INTERNAL_LIFETIME_PATH_KEYS = {"prior_json", "prior_pkl", "manifest_path"}


def project_root_from_config(path: str | Path) -> Path:
    """根据配置文件位置推导项目根目录。"""
    resolved = Path(path).expanduser().resolve()
    return resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent


def _require_mapping(cfg: Mapping[str, Any], key: str) -> Dict[str, Any]:
    """读取配置区块并拒绝非映射值。"""
    value = cfg.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise LVDWorkflowError("config", "结构校验", f"{key} 必须是映射", "检查 YAML 缩进")
    return deepcopy(dict(value))


def normalize_config(raw: Mapping[str, Any], project_root: str | Path) -> Dict[str, Any]:
    """把简化公开配置转换为算法使用的规范化配置。

    公开配置使用 ``project/data/features/lifetime/model/training/inference/explanation/runtime``。
    内部产物路径由输出目录自动生成，用户不能再覆盖这些路径。
    """
    root = Path(project_root).resolve()
    cfg = deepcopy(dict(raw))
    project = _require_mapping(cfg, "project")
    data = _require_mapping(cfg, "data")
    features = _require_mapping(cfg, "features")
    lifetime = _require_mapping(cfg, "lifetime")
    model = _require_mapping(cfg, "model")
    training = _require_mapping(cfg, "training")
    inference = _require_mapping(cfg, "inference")
    explanation = _require_mapping(cfg, "explanation")
    runtime = _require_mapping(cfg, "runtime")

    forbidden_features = sorted(_INTERNAL_FEATURE_PATH_KEYS.intersection(features))
    forbidden_lifetime = sorted(_INTERNAL_LIFETIME_PATH_KEYS.intersection(lifetime))
    if forbidden_features or forbidden_lifetime:
        names = forbidden_features + forbidden_lifetime
        raise LVDWorkflowError(
            "config",
            "配置清理",
            f"以下内部产物路径已从公开配置删除: {names}",
            "删除这些字段；系统会根据 project.output_dir 自动生成路径",
        )

    dataset = str(data.get("dataset", "")).strip()
    if not dataset:
        raise LVDWorkflowError("config", "结构校验", "data.dataset 不能为空", "例如设置为 FD004")

    output_value = project.get("output_dir", f"outputs/{dataset}")
    output_path = Path(str(output_value)).expanduser()
    if not output_path.is_absolute():
        output_path = root / output_path

    data_root_value = data.get("root", "datastream/CMAPSSData")
    data_root = Path(str(data_root_value)).expanduser()
    if not data_root.is_absolute():
        data_root = root / data_root

    def resolve_data_file(key: str, default_name: str) -> str:
        value = data.get(key, default_name)
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            # 仅文件名相对 data.root；包含目录的路径相对项目根目录。
            path = data_root / path if path.parent == Path(".") else root / path
        return str(path.resolve())

    normalized_data = deepcopy(data)
    normalized_data.pop("root", None)
    normalized_data["dataset"] = dataset
    normalized_data["fd"] = dataset  # 仅供现有正式加载器内部使用，不再暴露给用户。
    normalized_data["root_path"] = str(data_root.resolve())
    normalized_data["train_file"] = resolve_data_file("train_file", f"train_{dataset}.txt")
    normalized_data["test_file"] = resolve_data_file("test_file", f"test_{dataset}.txt")
    normalized_data["rul_file"] = resolve_data_file("rul_file", f"RUL_{dataset}.txt")
    normalized_data["output_dir"] = str(output_path.resolve())

    normalized_features = deepcopy(features)
    normalized_features.setdefault("enabled", True)
    normalized_features.setdefault("cache_policy", "auto")
    normalized_features.setdefault("default_training_feature_mode", "raw")

    normalized_lifetime = deepcopy(lifetime)
    normalized_lifetime.setdefault("enabled", True)
    normalized_lifetime.setdefault("cache_policy", "auto")
    if "max_components" in normalized_lifetime:
        normalized_lifetime["max_k"] = normalized_lifetime.pop("max_components")
    normalized_lifetime.setdefault("max_k", 4)

    return {
        "project": {**project, "output_dir": str(output_path.resolve())},
        "data": normalized_data,
        "features": normalized_features,
        "lifetime_prior": normalized_lifetime,
        "model": model,
        "training": training,
        "inference": inference,
        "explain": explanation,
        "runtime": runtime,
        "pipeline": {"mode": "integrated"},
    }


def load_config(path: str | Path) -> tuple[Path, Dict[str, Any]]:
    """读取、校验并规范化 YAML/JSON 配置。"""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise LVDWorkflowError("config", "加载", f"配置文件不存在: {resolved}", "使用 load 重新选择配置文件")
    raw = load_yaml_like(resolved)
    if not isinstance(raw, dict):
        raise LVDWorkflowError("config", "结构校验", "配置根节点必须是映射", "检查 YAML 缩进和顶层字段")
    root = project_root_from_config(resolved)
    return resolved, normalize_config(raw, root)


def apply_session_settings(cfg: Mapping[str, Any], settings: Mapping[str, Any]) -> Dict[str, Any]:
    """把会话设置写入配置副本，不修改磁盘 YAML。"""
    merged = deepcopy(dict(cfg))
    runtime = merged.setdefault("runtime", {})
    runtime["verbosity"] = settings.get("log", runtime.get("verbosity", "normal"))
    runtime["save_plots"] = bool(settings.get("plots", runtime.get("save_plots", False)))
    runtime["save_reports"] = bool(settings.get("reports", runtime.get("save_reports", False)))
    runtime["gui_mode"] = bool(settings.get("gui_mode", runtime.get("gui_mode", False)))
    features = merged.setdefault("features", {})
    features["save_plots"] = runtime["save_plots"]
    features["save_tables"] = runtime["save_reports"]
    inference = merged.setdefault("inference", {})
    inference["plot_mode"] = settings.get("plot_mode", inference.get("plot_mode", "current"))
    inference["mc_samples"] = int(settings.get("mc_samples", inference.get("mc_samples", 20)))
    explanation = merged.setdefault("explain", {})
    explanation["shap_sample_size"] = int(settings.get("shap_samples", explanation.get("shap_sample_size", 128)))
    return merged
