"""prediction 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""推理主流程编排。"""

from pathlib import Path
from typing import Any, Mapping

from lvd_surv.modeling.inference import predict_dataset


def run_prediction_pipeline(
    checkpoint: str | Path,
    *,
    test_file: str | Path | None = None,
    rul_file: str | Path | None = None,
    output_dir: str | Path = "outputs",
    mc_samples: int = 20,
    use_checkpoint_prior: bool = True,
    plot_max_curves_per_device: int = 8,
    plot_mode: str = "current",
):
    """运行 checkpoint 驱动的推理流程。"""
    return predict_dataset(
        checkpoint=checkpoint,
        test_file=test_file,
        rul_file=rul_file,
        output_dir=output_dir,
        mc_samples=mc_samples,
        use_checkpoint_prior=use_checkpoint_prior,
        plot_max_curves_per_device=plot_max_curves_per_device,
        plot_mode=plot_mode,
    )



def _prediction_runner(checkpoint, **kwargs):
    """Testable indirection for the canonical prediction workflow."""
    return run_prediction_pipeline(checkpoint, **kwargs)

def run(cfg: Mapping[str, Any], checkpoint: str | Path, output_dir: str | Path):
    """Use active inference settings with the canonical prediction workflow."""
    inference = cfg.get("inference", {})
    return _prediction_runner(
        checkpoint,
        test_file=cfg.get("data", {}).get("test_file"),
        rul_file=cfg.get("data", {}).get("rul_file"),
        output_dir=output_dir,
        mc_samples=int(inference.get("mc_samples", 20)),
        use_checkpoint_prior=bool(cfg.get("lifetime_prior", {}).get("use_in_inference", True)),
        plot_max_curves_per_device=int(inference.get("plot_max_curves_per_device", 8)),
        plot_mode=str(inference.get("plot_mode", "current")),
    )
