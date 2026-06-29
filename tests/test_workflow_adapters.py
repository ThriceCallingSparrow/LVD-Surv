"""确认新工作流只转发到原正式流程，不重写核心算法。"""
from pathlib import Path


def test_training_adapter_delegates(monkeypatch):
    from lvd_surv.workflows import training
    seen = {}
    def fake(cfg):
        seen["cfg"] = cfg
        return Path("best_model.pt")
    monkeypatch.setattr(training, "_training_runner", fake)
    result = training.run({"data": {"dataset": "FD004"}})
    assert result == Path("best_model.pt")
    assert seen["cfg"]["data"]["dataset"] == "FD004"


def test_prediction_adapter_preserves_inference_settings(monkeypatch, tmp_path):
    from lvd_surv.workflows import prediction
    seen = {}
    def fake(checkpoint, **kwargs):
        seen.update(kwargs)
        return "ok"
    monkeypatch.setattr(prediction, "_prediction_runner", fake)
    cfg = {
        "data": {"test_file": "test.txt", "rul_file": "rul.txt"},
        "inference": {"mc_samples": 11, "plot_mode": "history", "plot_max_curves_per_device": 3},
        "lifetime_prior": {"use_in_inference": True},
    }
    assert prediction.run(cfg, "model.pt", tmp_path) == "ok"
    assert seen["mc_samples"] == 11
    assert seen["plot_mode"] == "history"
    assert seen["plot_max_curves_per_device"] == 3
