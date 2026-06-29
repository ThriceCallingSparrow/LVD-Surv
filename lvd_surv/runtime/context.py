"""CLI、桌面工作台和脚本共用的运行上下文。"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from lvd_surv.core.config import load_config, project_root_from_config
from lvd_surv.core.errors import LVDWorkflowError


@dataclass
class OutputPolicy:
    """控制日志、分析表格和分析图的生成。"""

    verbosity: str = "normal"
    save_reports: bool = False
    save_plots: bool = False

    def validate(self) -> None:
        """验证输出级别。"""
        allowed = {"quiet", "normal", "verbose", "debug"}
        if self.verbosity not in allowed:
            raise LVDWorkflowError("runtime", "输出策略校验", f"未知 verbosity={self.verbosity}", f"请选择 {sorted(allowed)}")


@dataclass
class RuntimeContext:
    """表示一次 CLI 或桌面会话的稳定运行状态。"""

    project_root: Path
    config_path: Optional[Path] = None
    checkpoint: Optional[Path] = None
    output_policy: OutputPolicy = field(default_factory=OutputPolicy)
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        verbosity: str = "normal",
        save_reports: bool = False,
        save_plots: bool = False,
    ) -> "RuntimeContext":
        """读取简化配置，生成规范化配置和运行上下文。"""
        path, cfg = load_config(config_path)
        policy = OutputPolicy(verbosity=verbosity, save_reports=save_reports, save_plots=save_plots)
        policy.validate()
        ctx = cls(
            project_root=project_root_from_config(path),
            config_path=path,
            output_policy=policy,
            config=cfg,
        )
        ctx.apply_runtime_overrides()
        return ctx

    def resolve(self, value: str | Path | None) -> Optional[Path]:
        """把相对路径稳定解析到项目根目录。"""
        if value is None:
            return None
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def apply_runtime_overrides(self) -> None:
        """仅覆盖输出策略，不改变模型或分析算法参数。"""
        runtime = self.config.setdefault("runtime", {})
        runtime["verbosity"] = self.output_policy.verbosity
        runtime["save_reports"] = self.output_policy.save_reports
        runtime["save_plots"] = self.output_policy.save_plots
        features = self.config.setdefault("features", {})
        features["save_tables"] = self.output_policy.save_reports
        features["save_plots"] = self.output_policy.save_plots

    def require_file(self, value: str | Path | None, *, module: str, label: str) -> Path:
        """解析并验证输入文件。"""
        path = self.resolve(value)
        if path is None or not path.is_file():
            raise LVDWorkflowError(module, "路径校验", f"{label} 不存在: {path}", f"修正配置中的 {label} 后重新运行")
        return path
