"""Stage 4 public configuration and automatic artifact path tests."""
from pathlib import Path
import pytest

from lvd_surv.core.artifacts import build_artifact_paths
from lvd_surv.core.config import normalize_config
from lvd_surv.core.errors import LVDWorkflowError


def _minimal_config():
    return {
        "project": {"output_dir": "outputs/FD004"},
        "data": {
            "dataset": "FD004",
            "root": "datastream/CMAPSSData",
            "train_file": "train_FD004.txt",
            "test_file": "test_FD004.txt",
            "rul_file": "RUL_FD004.txt",
        },
        "features": {},
        "lifetime": {},
        "model": {},
        "training": {},
        "inference": {},
        "explanation": {},
    }


def test_public_config_is_normalized_without_changing_algorithm_sections(tmp_path):
    cfg = normalize_config(_minimal_config(), tmp_path)
    assert cfg["data"]["dataset"] == "FD004"
    assert cfg["data"]["fd"] == "FD004"
    assert Path(cfg["data"]["train_file"]) == tmp_path / "datastream/CMAPSSData/train_FD004.txt"
    assert "lifetime_prior" in cfg
    assert "explain" in cfg


def test_artifact_paths_are_automatic_and_standardized(tmp_path):
    cfg = normalize_config(_minimal_config(), tmp_path)
    paths = build_artifact_paths(cfg)
    root = tmp_path / "outputs/FD004/artifacts"
    assert paths["feature_decision"] == root / "features/decision.json"
    assert paths["feature_transformer"] == root / "features/transformer.pkl"
    assert paths["mixture_prior"] == root / "lifetime/prior.json"
    assert paths["mixture_prior_pkl"] == root / "lifetime/model.pkl"


def test_internal_artifact_paths_are_rejected(tmp_path):
    raw = _minimal_config()
    raw["features"]["decision_path"] = "custom.json"
    with pytest.raises(LVDWorkflowError):
        normalize_config(raw, tmp_path)
