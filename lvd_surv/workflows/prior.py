"""正式混合寿命分布工作流适配器。"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping


def run(cfg: Mapping[str, Any], *, rebuild: bool = False) -> Path:
    """运行现有正式寿命先验算法，不改变候选分布或拟合逻辑。"""
    from lvd_surv.lifetime.analysis import get_or_build_lifetime_prior
    from lvd_surv.core.artifacts import build_artifact_paths
    from lvd_surv.data.cmapss import add_train_rul, read_cmapss_txt
    config = dict(cfg)
    config["lifetime_prior"] = dict(cfg.get("lifetime_prior", {}))
    if rebuild:
        config["lifetime_prior"]["cache_policy"] = "force"
    train_file = config.get("data", {}).get("train_file")
    if not train_file:
        raise ValueError("data.train_file is required")
    frame = add_train_rul(read_cmapss_txt(train_file))
    paths = build_artifact_paths(config)
    get_or_build_lifetime_prior(config, frame, paths)
    return Path(paths["mixture_prior"])
