"""artifacts 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""缓存、路径和运行状态管理。

该模块集中管理所有阶段性产物路径、JSON/Pickle 读写、数据指纹、配置指纹、脚本指纹与运行状态。
这样训练流程、推理流程和验证流程不会各自拼接路径，后续维护时只需要检查本文件即可。
"""

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for an artifact and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json_artifact(obj: Mapping[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact using UTF-8 and stable indentation."""
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_json_artifact(path: str | Path) -> Dict[str, Any]:
    """Read a JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle_artifact(obj: Any, path: str | Path) -> Path:
    """Write a pickle artifact.

    Pickle artifacts are for debugging/reproducibility only.  The main pipeline
    consumes JSON contracts whenever possible to avoid version-sensitive object
    loading.
    """
    path = ensure_parent(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle_artifact(path: str | Path) -> Any:
    """Read a pickle artifact."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


def _json_hash(obj: Any) -> str:
    """Return a deterministic SHA256 hash for JSON-serializable content."""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> Optional[str]:
    """Hash a file if it exists; return None for missing files."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_data_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    """Build a lightweight but useful fingerprint for a CMAPSS dataframe.

    The fingerprint avoids hashing every float cell in very large datasets while
    still catching the changes that matter for cache safety: row/column shape,
    unit coverage, cycle totals, and a stable pandas hash sample over all rows.
    """
    cols = list(map(str, df.columns))
    base: Dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(cols)),
        "column_names": cols,
    }
    if "unit_id" in df.columns:
        base["unit_count"] = int(df["unit_id"].nunique())
    if "cycle" in df.columns:
        base["cycle_sum"] = float(pd.to_numeric(df["cycle"], errors="coerce").fillna(0).sum())
        base["cycle_max"] = float(pd.to_numeric(df["cycle"], errors="coerce").max())
    if "rul" in df.columns or "RUL" in df.columns:
        rul_col = "rul" if "rul" in df.columns else "RUL"
        rul = pd.to_numeric(df[rul_col], errors="coerce")
        base["rul_min"] = float(rul.min())
        base["rul_max"] = float(rul.max())
    # pandas hashing is deterministic for the same values and column order.
    hashed = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).values
    base["hash"] = hashlib.sha256(hashed.tobytes()).hexdigest()
    return base


def build_config_fingerprint(cfg: Mapping[str, Any], keys: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Hash the selected configuration sections relevant to cached artifacts."""
    if keys is None:
        view = dict(cfg)
    else:
        view = {k: cfg.get(k) for k in keys}
    return {"hash": _json_hash(view), "keys": list(keys) if keys is not None else "all"}


def build_script_fingerprint(paths: Iterable[str | Path]) -> Dict[str, Any]:
    """Hash the user scripts that affect cached artifacts."""
    out: Dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        out[str(path)] = file_sha256(path)
    out["hash"] = _json_hash(out)
    return out


def now_utc() -> str:
    """Return an ISO timestamp for manifests/run-state files."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_manifest(
    *,
    dataset: str,
    artifact_type: str,
    data_fingerprint: Mapping[str, Any],
    config_fingerprint: Mapping[str, Any],
    script_fingerprint: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> Dict[str, Any]:
    """Create a manifest that decides whether a cache can be reused."""
    return {
        "schema_version": "1.0",
        "artifact_type": artifact_type,
        "dataset": dataset,
        "created_at": now_utc(),
        "data_fingerprint": dict(data_fingerprint),
        "config_fingerprint": dict(config_fingerprint),
        "script_fingerprint": dict(script_fingerprint),
        "artifacts": dict(artifacts),
    }


def is_cache_valid(
    manifest_path: str | Path,
    *,
    expected_dataset: str,
    artifact_paths: Iterable[str | Path],
    data_fingerprint: Mapping[str, Any],
    config_fingerprint: Mapping[str, Any],
    script_fingerprint: Mapping[str, Any],
) -> bool:
    """Return True only when manifest and all artifact paths match expectations."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return False
    if any(not Path(p).exists() for p in artifact_paths):
        return False
    try:
        manifest = load_json_artifact(manifest_path)
    except Exception:
        return False
    if manifest.get("dataset") != expected_dataset:
        return False
    return (
        manifest.get("data_fingerprint") == dict(data_fingerprint)
        and manifest.get("config_fingerprint") == dict(config_fingerprint)
        and manifest.get("script_fingerprint") == dict(script_fingerprint)
    )


def update_run_state(path: str | Path, **updates: Any) -> Path:
    """Merge updates into a simple run-state JSON file."""
    path = ensure_parent(path)
    state: Dict[str, Any] = {}
    if path.exists():
        try:
            state = load_json_artifact(path)
        except Exception:
            state = {}
    state.update(updates)
    state["updated_at"] = now_utc()
    return save_json_artifact(state, path)



def get_dataset_name(cfg: Mapping[str, Any]) -> str:
    """从规范化配置中提取数据集名称。"""
    data_cfg = dict(cfg.get("data", {})) if isinstance(cfg, Mapping) else {}
    return str(data_cfg.get("dataset") or "dataset")


def build_artifact_paths(cfg: Mapping[str, Any]) -> Dict[str, Path]:
    """根据输出根目录自动生成全部内部产物路径。

    用户配置不再暴露 JSON、PKL、manifest 等内部文件路径。所有工作流都使用
    本函数，避免不同模块产生不一致的目录结构。
    """
    data_cfg = dict(cfg.get("data", {})) if isinstance(cfg, Mapping) else {}
    dataset = get_dataset_name(cfg)
    output_dir = Path(data_cfg.get("output_dir") or Path("outputs") / dataset)
    artifacts_dir = output_dir / "artifacts"
    feature_dir = artifacts_dir / "features"
    lifetime_dir = artifacts_dir / "lifetime"
    checkpoint_dir = output_dir / "checkpoints"
    return {
        "output_dir": output_dir,
        "cache_dir": artifacts_dir,
        "feature_dir": feature_dir,
        "lifetime_dir": lifetime_dir,
        "checkpoint_dir": checkpoint_dir,
        "feature_decision": feature_dir / "decision.json",
        "feature_bundle": feature_dir / "analysis.pkl",
        "feature_transformer": feature_dir / "transformer.pkl",
        "feature_manifest": feature_dir / "manifest.json",
        "mixture_prior": lifetime_dir / "prior.json",
        "mixture_prior_pkl": lifetime_dir / "model.pkl",
        "mixture_manifest": lifetime_dir / "manifest.json",
        "run_state": output_dir / "run_state.json",
    }
