"""Stage 4 compatibility removal and canonical architecture tests."""
from pathlib import Path
import inspect


def test_legacy_import_adapters_are_removed():
    root = Path(__file__).resolve().parents[1] / "lvd_surv"
    removed = [
        "data.py", "model.py", "infer.py", "train.py", "losses.py", "metrics.py",
        "mixture_prior.py", "visualize.py", "feature_transformer.py", "explain.py",
    ]
    for name in removed:
        assert not (root / name).exists()
    assert not (root / "cmapss").exists()
    assert not (root / "pipelines").exists()


def test_canonical_modules_are_importable():
    from lvd_surv.modeling.model import LVDSurvModel
    from lvd_surv.modeling.inference import predict_dataset
    from lvd_surv.features.analysis import get_or_build_feature_artifacts
    assert LVDSurvModel is not None
    assert predict_dataset is not None
    assert get_or_build_feature_artifacts is not None


def test_workflows_are_canonical():
    from lvd_surv.workflows import training, prediction, explanation
    for module in (training, prediction, explanation):
        source = inspect.getsource(module)
        assert "lvd_surv.pipelines" not in source


def test_obsolete_business_scripts_removed():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "cli.py").is_file()
    assert not (root / "scripts" / "train_lvd_surv.py").exists()
    assert not (root / "scripts" / "predict_lvd_surv.py").exists()
    assert not (root / "scripts" / "explain_lvd_surv.py").exists()
