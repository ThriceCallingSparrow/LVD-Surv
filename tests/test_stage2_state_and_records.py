"""第二阶段：状态继承、临时模式、运行记录和复合命令测试。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lvd_surv.app.commands import CommandEngine
from lvd_surv.app.session import SessionContext
from lvd_surv.core.run_records import append_run_record, save_resolved_config
from lvd_surv.workflows.validation import validate_checkpoint_compatibility


def _session() -> SessionContext:
    root = Path(__file__).resolve().parents[1]
    session = SessionContext()
    session.load(root / "configs" / "default.yaml")
    return session


def test_temporary_modes_do_not_change_persistent_settings(monkeypatch, tmp_path):
    session = _session()
    engine = CommandEngine(session)
    original = dict(session.settings)
    monkeypatch.setattr("lvd_surv.app.commands.feature.run", lambda cfg, rebuild=False: tmp_path / "decision.json")
    engine.execute("feature full")
    assert session.settings == original
    monkeypatch.setattr("lvd_surv.app.commands.prior.run", lambda cfg, rebuild=False: tmp_path / "prior.json")
    engine.execute("prior quiet")
    assert session.settings == original


def test_set_reset_restores_config_defaults():
    session = _session()
    engine = CommandEngine(session)
    engine.execute("set samples 7")
    assert session.settings["mc_samples"] == 7
    engine.execute("set reset")
    assert session.settings["mc_samples"] == int(session.base_config["inference"]["mc_samples"])
    assert not session.overridden_settings


def test_explicit_setting_survives_reload():
    session = _session()
    session.set_value("samples", "9")
    session.reload()
    assert session.settings["mc_samples"] == 9
    assert session.checkpoint is None


def test_run_record_and_resolved_config_are_written(tmp_path):
    cfg = {"data": {"dataset": "FD004"}, "training": {"epochs": 2}}
    snapshot = save_resolved_config(cfg, tmp_path, "train")
    record = append_run_record(
        tmp_path,
        stage="train",
        status="success",
        command="train",
        resolved_config=snapshot,
        primary_artifact="best_model.pt",
    )
    assert snapshot.is_file()
    payload = json.loads(record.read_text(encoding="utf-8").splitlines()[-1])
    assert payload["stage"] == "train"
    assert payload["status"] == "success"
    assert payload["resolved_config"] == str(snapshot)


def test_composite_run_evaluate_uses_current_artifacts(monkeypatch):
    session = _session()
    session.checkpoint = Path("/tmp/model.pt")
    engine = CommandEngine(session)
    seen = []
    monkeypatch.setattr(engine, "_require_checkpoint", lambda: seen.append("require") or session.checkpoint)
    monkeypatch.setattr(engine, "_test", lambda args: seen.append("test") or "pred")
    monkeypatch.setattr(engine, "_validate", lambda target: seen.append(f"validate:{target}") or "valid")
    monkeypatch.setattr(engine, "_explain", lambda: seen.append("explain") or "exp")
    result = engine.execute("run evaluate")
    assert seen == ["require", "test", "validate:prediction", "explain"]
    assert "pred" in result and "exp" in result


def test_checkpoint_dataset_mismatch_is_rejected(monkeypatch):
    fake_ckpt = {"cfg": {"data": {"dataset": "FD001", "window_size": 50, "max_horizon": 150}}}
    monkeypatch.setattr("lvd_surv.workflows.validation.validate_model", lambda path: fake_ckpt)
    with pytest.raises(ValueError, match="数据集不兼容"):
        validate_checkpoint_compatibility(
            "model.pt",
            {"data": {"dataset": "FD004", "window_size": 50, "max_horizon": 150}},
        )
