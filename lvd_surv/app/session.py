"""桌面终端与 CLI 共用的有状态工作会话。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

from lvd_surv.core.config import apply_session_settings, load_config
from lvd_surv.core.paths import ProjectPaths
from lvd_surv.core.artifacts import build_artifact_paths
from lvd_surv.runtime.context import RuntimeContext


@dataclass
class SessionContext:
    """保存已加载配置、当前模型、会话覆盖设置和最近产物。

    会话设置只覆盖内存中的配置副本，不会修改用户的 YAML。加载新配置会
    清除与旧配置绑定的模型和结果，避免跨数据集误用产物。
    """

    config_path: Optional[Path] = None
    project_root: Optional[Path] = None
    base_config: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[Path] = None
    latest_prediction: Optional[Path] = None
    latest_explanation: Optional[Path] = None
    latest_feature: Optional[Path] = None
    latest_prior: Optional[Path] = None
    latest_resolved_config: Optional[Path] = None
    latest_run_record: Optional[Path] = None
    last_command: str = ""
    last_result: str = ""
    task_status: str = "idle"
    settings: Dict[str, Any] = field(default_factory=lambda: {
        "log": "normal",
        "plots": False,
        "reports": False,
        "plot_mode": "current",
        "mc_samples": 20,
        "shap_samples": 128,
        "gui_mode": False,
    })
    overridden_settings: set[str] = field(default_factory=set)

    def load(self, path: str | Path) -> None:
        """加载配置，并清除依赖旧配置的模型和结果绑定。

        用户通过 ``set`` 明确修改过的会话设置会保留；未覆盖项目从新配置读取。
        这样切换数据集不会丢失用户的界面偏好，也不会把旧模型带入新配置。
        """
        resolved, _ = load_config(path)
        runtime_ctx = RuntimeContext.from_config(resolved)
        cfg = runtime_ctx.config
        self.config_path = resolved
        self.project_root = runtime_ctx.project_root
        self.base_config = cfg
        defaults = self._settings_from_config(cfg)
        for key, value in defaults.items():
            if key not in self.overridden_settings:
                self.settings[key] = value
        self._clear_bound_artifacts()
        self._discover_existing_analysis_artifacts()
        self.last_result = f"已加载配置: {resolved}"

    def reload(self) -> None:
        """从磁盘重新载入当前配置，并保留显式会话覆盖设置。"""
        if self.config_path is None:
            raise ValueError("尚未加载配置。")
        self.load(self.config_path)

    @staticmethod
    def _settings_from_config(cfg: Mapping[str, Any]) -> Dict[str, Any]:
        """提取配置中的会话默认值。"""
        runtime = cfg.get("runtime", {})
        inference = cfg.get("inference", {})
        explain = cfg.get("explain", {})
        return {
            "log": runtime.get("verbosity", "normal"),
            "plots": bool(runtime.get("save_plots", False)),
            "reports": bool(runtime.get("save_reports", False)),
            "plot_mode": inference.get("plot_mode", "current"),
            "mc_samples": int(inference.get("mc_samples", 20)),
            "shap_samples": int(explain.get("shap_sample_size", 128)),
        }

    def _clear_bound_artifacts(self) -> None:
        """清除不能跨配置安全复用的会话绑定。"""
        self.checkpoint = None
        self.latest_prediction = None
        self.latest_explanation = None
        self.latest_feature = None
        self.latest_prior = None
        self.latest_resolved_config = None
        self.latest_run_record = None

    def _discover_existing_analysis_artifacts(self) -> None:
        """发现当前配置目录中的正式特征与寿命先验产物。

        只登记当前配置明确指向且真实存在的分析契约；不会自动绑定 checkpoint，
        因为模型还需要执行严格兼容性检查。
        """
        if not self.base_config:
            return
        paths = build_artifact_paths(self.base_config)
        feature_path = Path(paths["feature_decision"])
        prior_path = Path(paths["mixture_prior"])
        if feature_path.is_file():
            self.latest_feature = feature_path.resolve()
        if prior_path.is_file():
            self.latest_prior = prior_path.resolve()

    @property
    def config(self) -> Dict[str, Any]:
        """返回应用当前会话覆盖后的独立配置副本。"""
        if not self.base_config:
            raise ValueError("尚未加载配置。请先输入 load。")
        return apply_session_settings(self.base_config, self.settings)

    @property
    def paths(self) -> ProjectPaths:
        """返回当前配置的标准输出路径。"""
        if self.project_root is None:
            raise ValueError("尚未加载配置。")
        paths = ProjectPaths.from_config(self.config, self.project_root)
        paths.ensure_output_dirs()
        return paths

    def set_checkpoint(self, path: str | Path) -> None:
        """严格校验并绑定 checkpoint，拒绝跨数据集或不完整模型。"""
        candidate = Path(path).expanduser()
        if not candidate.is_absolute() and self.project_root:
            candidate = self.project_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"checkpoint 不存在: {candidate}")
        from lvd_surv.workflows.validation import validate_checkpoint_compatibility

        validate_checkpoint_compatibility(candidate, self.config if self.base_config else None)
        self.checkpoint = candidate
        self.last_result = f"当前模型: {candidate}"

    def set_value(self, key: str, value: str) -> None:
        """修改当前会话设置，不改写配置文件。"""
        aliases = {"samples": "mc_samples", "shap": "shap_samples", "plot-mode": "plot_mode"}
        key = aliases.get(key, key)
        if key in {"plots", "reports"}:
            if value.lower() not in {"on", "off", "true", "false"}:
                raise ValueError(f"{key} 只接受 on/off")
            self.settings[key] = value.lower() in {"on", "true"}
        elif key in {"mc_samples", "shap_samples"}:
            parsed = int(value)
            if parsed <= 0:
                raise ValueError(f"{key} 必须大于 0")
            self.settings[key] = parsed
        elif key == "log":
            if value not in {"quiet", "normal", "verbose", "debug"}:
                raise ValueError("log 必须是 quiet/normal/verbose/debug")
            self.settings[key] = value
        elif key == "plot_mode":
            if value not in {"current", "history", "snapshot"}:
                raise ValueError("plot-mode 必须是 current/history/snapshot")
            self.settings[key] = value
        else:
            raise ValueError(f"未知设置: {key}")
        self.overridden_settings.add(key)
        self.last_result = f"已设置 {key}={self.settings[key]}"

    def reset_settings(self) -> None:
        """清除显式覆盖并恢复当前配置中的会话默认值。"""
        self.overridden_settings.clear()
        if self.base_config:
            self.settings.update(self._settings_from_config(self.base_config))
        self.last_result = "会话设置已恢复为当前配置默认值。"

    @contextmanager
    def temporary_settings(self, overrides: Mapping[str, Any]) -> Iterator[None]:
        """临时覆盖设置，命令结束后恢复原值。

        ``feature full``、``prior quiet`` 与 ``test history`` 使用该机制，确保
        单次命令不会意外修改后续任务的默认行为。
        """
        original = dict(self.settings)
        try:
            self.settings.update(dict(overrides))
            yield
        finally:
            self.settings.clear()
            self.settings.update(original)

    def register_artifact(self, kind: str, path: str | Path) -> Path:
        """登记工作流产物并返回绝对路径。"""
        resolved = Path(path).expanduser().resolve()
        mapping = {
            "checkpoint": "checkpoint",
            "prediction": "latest_prediction",
            "explanation": "latest_explanation",
            "feature": "latest_feature",
            "prior": "latest_prior",
            "resolved_config": "latest_resolved_config",
            "run_record": "latest_run_record",
        }
        if kind not in mapping:
            raise ValueError(f"未知产物类型: {kind}")
        setattr(self, mapping[kind], resolved)
        return resolved

    def summary(self) -> str:
        """生成状态栏和 ``show`` 命令使用的摘要。"""
        dataset = self.base_config.get("data", {}).get("dataset", "-") if self.base_config else "-"
        return (
            f"配置: {self.config_path or '-'}\n数据集: {dataset}\n"
            f"模型: {self.checkpoint or '-'}\n任务: {self.task_status}\n"
            f"设置: {self.settings}\n最近配置快照: {self.latest_resolved_config or '-'}\n"
            f"最近结果: {self.last_result or '-'}"
        )
