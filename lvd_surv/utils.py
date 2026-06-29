"""utils 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """执行 set seed 对应的项目处理逻辑。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    """执行 ensure dir 对应的项目处理逻辑。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    """执行 save json 对应的项目处理逻辑。"""
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_yaml_like(path: str | Path) -> Dict[str, Any]:
    """Load the project YAML or JSON configuration.

    Uses PyYAML if installed. JSON files also work. This keeps the project runnable
    even when PyYAML is unavailable.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception as exc:
        raise RuntimeError(
            "Reading YAML requires PyYAML. Install it with: pip install pyyaml "
            "or provide the configuration as JSON."
        ) from exc


def count_parameters(model: torch.nn.Module) -> int:
    """执行 count parameters 对应的项目处理逻辑。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
