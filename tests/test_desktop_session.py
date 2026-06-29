"""桌面会话和短命令的无重型算法测试。"""
from pathlib import Path

from lvd_surv.app.commands import CommandEngine
from lvd_surv.app.session import SessionContext


def test_session_load_resolves_paths_and_clears_bound_results():
    root = Path(__file__).resolve().parents[1]
    session = SessionContext()
    session.checkpoint = root / "dummy.pt"
    session.latest_prediction = root / "dummy.csv"
    session.load(root / "configs" / "default.yaml")
    assert session.config_path.is_file()
    assert Path(session.config["data"]["train_file"]).is_absolute()
    assert session.checkpoint is None
    assert session.latest_prediction is None


def test_short_settings_persist_in_session():
    root = Path(__file__).resolve().parents[1]
    session = SessionContext()
    session.load(root / "configs" / "default.yaml")
    engine = CommandEngine(session)
    assert "mc_samples" in engine.execute("set")
    engine.execute("set samples 7")
    engine.execute("set shap 16")
    engine.execute("set plots on")
    engine.execute("set plot-mode history")
    cfg = session.config
    assert cfg["inference"]["mc_samples"] == 7
    assert cfg["explain"]["shap_sample_size"] == 16
    assert cfg["runtime"]["save_plots"] is True
    assert cfg["inference"]["plot_mode"] == "history"


def test_load_once_allows_repeated_short_status_commands():
    root = Path(__file__).resolve().parents[1]
    session = SessionContext()
    engine = CommandEngine(session)
    engine.execute(f'load "{root / "configs" / "default.yaml"}"')
    assert "FD004" in engine.execute("show data")
    assert "配置:" in engine.execute("status")
    assert "load" in engine.execute("help")
