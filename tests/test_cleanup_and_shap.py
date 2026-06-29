"""Verify removed auxiliary paths and formal SHAP configuration."""
from pathlib import Path


def test_obsolete_auxiliary_feature_module_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "lvd_surv" / "feature_selection.py").exists()


def test_formal_shap_dependency_and_no_legacy_config_aliases():
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    config = (root / "configs" / "default.yaml").read_text(encoding="utf-8")
    assert '"shap>=0.42"' in pyproject
    assert "integrated_gradients_steps" not in config
    assert "fallback_to_all_sensors" not in config
    assert "fallback_to_weibull" not in config


def test_no_legacy_weibull_prior_class():
    from lvd_surv.lifetime import prior
    assert not hasattr(prior, "WeibullMixturePrior")


def test_formal_shap_computes_attributions():
    import torch
    from torch.utils.data import DataLoader
    from lvd_surv.interpretation.explain import shap_gradient_importance

    class TinyModel(torch.nn.Module):
        def forward(self, x, cycle, sample=False):
            risk = torch.sigmoid(x.mean(dim=(1, 2)) + cycle.float() / 100.0)
            reliability = torch.stack([1.0 - 0.2 * risk, 1.0 - 0.4 * risk], dim=1)
            return {"reliability": reliability}

    rows = [{"x": torch.randn(4, 3), "cycle": torch.tensor(i + 1)} for i in range(8)]
    result = shap_gradient_importance(
        TinyModel(), DataLoader(rows, batch_size=4), torch.device("cpu"), ["a", "b", "c"], sample_size=8
    )
    assert list(result.columns) == ["feature", "mean_abs_shap"]
    assert set(result["feature"]) == {"a", "b", "c"}
    assert (result["mean_abs_shap"] >= 0).all()
