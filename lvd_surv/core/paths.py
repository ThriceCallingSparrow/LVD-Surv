"""集中解析项目、数据与输出路径。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from lvd_surv.core.artifacts import build_artifact_paths


@dataclass(frozen=True)
class ProjectPaths:
    """一次运行所需的标准路径集合。"""

    project_root: Path
    output_root: Path
    checkpoints: Path
    predictions: Path
    explanations: Path
    reports: Path
    plots: Path
    logs: Path

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], project_root: str | Path) -> "ProjectPaths":
        """由配置生成标准目录，并保持旧算法产物路径兼容。"""
        root = Path(project_root).resolve()
        artifacts = build_artifact_paths(cfg)
        output = Path(artifacts["output_dir"])
        if not output.is_absolute():
            output = (root / output).resolve()
        return cls(
            project_root=root,
            output_root=output,
            checkpoints=output / "checkpoints",
            predictions=output / "predictions",
            explanations=output / "explanations",
            reports=output / "reports",
            plots=output / "plots",
            logs=output / "logs",
        )

    def ensure_output_dirs(self) -> None:
        """创建标准输出目录。"""
        for path in (self.output_root, self.checkpoints, self.predictions, self.explanations, self.reports, self.plots, self.logs):
            path.mkdir(parents=True, exist_ok=True)
