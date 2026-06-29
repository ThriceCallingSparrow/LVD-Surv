"""Final stable-release markers and release summary tests."""
from pathlib import Path

import lvd_surv
from lvd_surv.core.config import normalize_config
from lvd_surv.workflows.validation import release_readiness


def _config(root: Path):
    data = root / "datastream/CMAPSSData"
    data.mkdir(parents=True, exist_ok=True)
    for name in ("train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt"):
        (data / name).write_text("1 1\n", encoding="utf-8")
    raw = {
        "project": {"output_dir": str(root / "outputs/FD004")},
        "data": {
            "dataset": "FD004",
            "root": str(data),
            "train_file": "train_FD004.txt",
            "test_file": "test_FD004.txt",
            "rul_file": "RUL_FD004.txt",
            "window_size": 50,
            "max_horizon": 150,
            "train_split": 0.85,
        },
        "features": {}, "lifetime": {}, "model": {},
        "training": {"epochs": 1, "batch_size": 2},
        "inference": {"mc_samples": 1},
        "explanation": {"shap_sample_size": 1},
        "runtime": {},
    }
    return normalize_config(raw, root)


def test_package_is_marked_stable():
    assert lvd_surv.__version__ == "1.3.0"
    assert lvd_surv.__release_status__ == "stable"


def test_release_summary_exposes_stable_build(tmp_path):
    summary = release_readiness(
        _config(tmp_path),
        Path(__file__).resolve().parents[1],
        require_dependencies=False,
    )
    assert summary["package_version"] == "1.3.0"
    assert summary["release_status"] == "stable"
    assert summary["gui_entry"] == "python main.py"
    assert summary["cli_entry"] == "lvd"


def test_public_docs_no_longer_mark_release_as_candidate():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    status = (root / "IMPLEMENTATION_STATUS.md").read_text(encoding="utf-8")
    assert "稳定版" in readme
    assert "发布候选版本" not in status
    assert (root / "FINAL_RELEASE_NOTES.md").is_file()
