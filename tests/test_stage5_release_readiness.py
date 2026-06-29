"""Stage 5 release-readiness and version consistency tests."""
from pathlib import Path

import pytest

import lvd_surv
from lvd_surv.core.config import normalize_config
from lvd_surv.workflows.validation import release_readiness, validate_project_layout


def _raw_config(root: Path):
    return {
        "project": {"output_dir": str(root / "outputs/FD004")},
        "data": {
            "dataset": "FD004",
            "root": str(root / "datastream/CMAPSSData"),
            "train_file": "train_FD004.txt",
            "test_file": "test_FD004.txt",
            "rul_file": "RUL_FD004.txt",
            "window_size": 50,
            "max_horizon": 150,
            "train_split": 0.85,
        },
        "features": {},
        "lifetime": {},
        "model": {},
        "training": {"epochs": 1, "batch_size": 2},
        "inference": {"mc_samples": 1},
        "explanation": {"shap_sample_size": 1},
        "runtime": {},
    }


def test_package_version_matches_stage5():
    assert lvd_surv.__version__ == "1.3.0"


def test_release_layout_accepts_supported_project_tree():
    root = Path(__file__).resolve().parents[1]
    validate_project_layout(root)


def test_release_layout_rejects_obsolete_compatibility_entry(tmp_path):
    root = Path(__file__).resolve().parents[1]
    copy_root = tmp_path / "project"
    import shutil
    shutil.copytree(root, copy_root, ignore=shutil.ignore_patterns("datastream", "outputs", ".git"))
    obsolete = copy_root / "lvd_surv/model.py"
    obsolete.write_text("# obsolete", encoding="utf-8")
    with pytest.raises(ValueError, match="已淘汰兼容入口"):
        validate_project_layout(copy_root)


def test_release_readiness_without_dependency_probe(tmp_path):
    data = tmp_path / "datastream/CMAPSSData"
    data.mkdir(parents=True)
    for name in ("train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"):
        (data / name).write_text("1 1\n", encoding="utf-8")
    cfg = normalize_config(_raw_config(tmp_path), tmp_path)
    root = Path(__file__).resolve().parents[1]
    summary = release_readiness(cfg, root, require_dependencies=False)
    assert summary["status"] == "ready"
    assert summary["dataset"] == "FD004"
