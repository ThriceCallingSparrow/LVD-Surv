"""状态化短命令解析与执行。

命令层只维护会话和调用工作流，不包含特征、分布、模型或解释算法。
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from lvd_surv.app.session import SessionContext
from lvd_surv.core.run_records import append_run_record, save_resolved_config
from lvd_surv.workflows import explanation, feature, prediction, prior, training, validation


HELP_TEXT = """可用命令：
  load [配置文件]       加载配置；GUI 中省略路径会弹出选择窗口
  reload                重新读取当前配置
  show [config|data|model|settings|output|records]
  set <项目> <值>       log/plots/reports/plot-mode/samples/shap
  set reset             恢复当前配置中的会话默认设置
  check [dependencies|config|data|model|all]
  release-check          发布结构、配置、数据、依赖和写权限检查
  feature [full|quiet|rebuild|show]
  prior [full|quiet|rebuild|show]
  train                 使用当前配置训练并绑定最佳模型
  model [checkpoint]    严格校验并绑定模型
  test [current|history|snapshot]
  validate [config|model|prediction|all]
  explain               生成置换重要性和正式 SHAP
  run train             检查后执行训练
  run evaluate          测试、验证并解释当前模型
  run all               特征、先验、训练、测试、验证和解释
  results               查看最近产物
  open [output|plots|reports|model|logs]
  status                查看当前任务和会话
  clear                 清空输出窗口
  history               查看命令历史
  reset                 清空模型与最近结果，保留当前配置
  stop                  请求安全停止当前任务
  exit                  关闭窗口
"""


class CommandEngine:
    """解析简短命令并将重型操作交给统一工作流。"""

    def __init__(
        self,
        session: SessionContext,
        *,
        choose_config: Optional[Callable[[], str]] = None,
        choose_model: Optional[Callable[[], str]] = None,
        clear_output: Optional[Callable[[], None]] = None,
        request_stop: Optional[Callable[[], None]] = None,
        close_app: Optional[Callable[[], None]] = None,
    ) -> None:
        self.session = session
        self.choose_config = choose_config
        self.choose_model = choose_model
        self.clear_output = clear_output
        self.request_stop = request_stop
        self.close_app = close_app
        self.history: list[str] = []

    @staticmethod
    def is_heavy(line: str) -> bool:
        """判断命令是否应在工作线程执行。"""
        tokens = shlex.split(line)
        return bool(tokens and tokens[0].lower() in {"feature", "prior", "train", "test", "predict", "explain", "run"})

    def execute(self, line: str) -> str:
        """执行一条命令并返回适合显示的文本。"""
        line = line.strip()
        if not line:
            return ""
        self.history.append(line)
        self.session.last_command = line
        tokens = shlex.split(line)
        command, args = tokens[0].lower(), tokens[1:]

        if command in {"help", "?"}:
            return HELP_TEXT
        if command == "load":
            selected = args[0] if args else (self.choose_config() if self.choose_config else "")
            if not selected:
                return "未选择配置文件。"
            self.session.load(selected)
            return self.session.last_result
        if command == "reload":
            self.session.reload()
            return self.session.last_result
        if command in {"show", "status"}:
            return self._show(args[0] if args else "all")
        if command == "set":
            if not args:
                return str(self.session.settings)
            if len(args) == 1 and args[0] == "reset":
                self.session.reset_settings()
                return self.session.last_result
            if len(args) != 2:
                raise ValueError("用法: set <项目> <值>，或 set reset")
            self.session.set_value(args[0], args[1])
            return self.session.last_result
        if command == "check":
            return self._check(args[0] if args else "all")
        if command == "release-check":
            summary = validation.release_readiness(self.session.config, self.session.project_root or Path.cwd())
            return "发布检查通过：" + "; ".join(f"{k}={v}" for k, v in summary.items())
        if command == "feature":
            return self._feature(args)
        if command == "prior":
            return self._prior(args)
        if command == "train":
            return self._train()
        if command == "model":
            selected = args[0] if args else (self.choose_model() if self.choose_model else "")
            if not selected:
                return "未选择 checkpoint。"
            self.session.set_checkpoint(selected)
            return self.session.last_result
        if command in {"test", "predict"}:
            return self._test(args)
        if command == "validate":
            return self._validate(args[0] if args else "prediction")
        if command == "explain":
            return self._explain()
        if command == "run":
            return self._run(args[0] if args else "all")
        if command == "results":
            return self._results()
        if command == "open":
            return self._open(args[0] if args else "output")
        if command == "clear":
            if self.clear_output:
                self.clear_output()
            return ""
        if command == "history":
            return "\n".join(f"{i + 1}: {item}" for i, item in enumerate(self.history)) or "暂无历史。"
        if command == "reset":
            self.session._clear_bound_artifacts()
            self.session.last_result = "会话产物绑定已重置。"
            return self.session.last_result
        if command == "stop":
            if self.request_stop:
                self.request_stop()
                return "已提交停止请求。"
            return "当前入口不支持任务停止。"
        if command in {"exit", "quit"}:
            if self.close_app:
                self.close_app()
            return ""
        raise ValueError(f"未知命令: {command}。输入 help 查看命令。")

    def _show(self, target: str) -> str:
        if target in {"all", "status"}:
            return self.session.summary()
        if target == "config":
            return str(self.session.config_path or "尚未加载配置")
        if target == "data":
            return str(self.session.base_config.get("data", {}))
        if target == "model":
            return str(self.session.checkpoint or "尚未绑定模型")
        if target == "settings":
            return str(self.session.settings)
        if target == "output":
            return str(self.session.paths.output_root)
        if target == "records":
            return (
                f"最近配置快照: {self.session.latest_resolved_config or '-'}\n"
                f"运行记录: {self.session.latest_run_record or '-'}"
            )
        raise ValueError(f"未知 show 对象: {target}")

    def _check(self, target: str) -> str:
        messages = []
        if target in {"all", "dependencies"}:
            failures = validation.check_dependencies()
            if failures:
                raise RuntimeError("正式依赖检查失败:\n" + "\n".join(failures))
            messages.append("正式依赖检查通过")
        if target in {"all", "config", "data"}:
            validation.validate_config(self.session.config)
            messages.append("配置和数据检查通过")
        if target in {"all", "model"} and self.session.checkpoint:
            validation.validate_checkpoint_compatibility(self.session.checkpoint, self.session.config)
            messages.append("模型兼容性检查通过")
        elif target == "model":
            raise ValueError("尚未绑定模型")
        return "；".join(messages) or "检查完成"

    def _recorded(self, stage: str, operation: Callable[[], Path | str], *, primary_kind: Optional[str] = None) -> Path | str:
        """保存解析配置和运行记录，并统一登记主要产物。"""
        cfg = self.session.config
        logs = self.session.paths.logs
        snapshot = save_resolved_config(cfg, logs, stage)
        self.session.register_artifact("resolved_config", snapshot)
        try:
            result = operation()
            primary = Path(result).resolve() if isinstance(result, (str, Path)) and str(result) else None
            if primary_kind and primary is not None:
                self.session.register_artifact(primary_kind, primary)
            record = append_run_record(
                logs,
                stage=stage,
                status="success",
                command=self.session.last_command,
                resolved_config=snapshot,
                primary_artifact=primary,
            )
            self.session.register_artifact("run_record", record)
            return result
        except Exception as exc:
            record = append_run_record(
                logs,
                stage=stage,
                status="failed",
                command=self.session.last_command,
                resolved_config=snapshot,
                message=str(exc),
            )
            self.session.register_artifact("run_record", record)
            raise

    def _feature(self, args: list[str]) -> str:
        action = args[0] if args else "normal"
        if action not in {"normal", "full", "quiet", "rebuild", "show"}:
            raise ValueError("feature 只接受 full/quiet/rebuild/show")
        if action == "show":
            return str(self.session.latest_feature or "当前会话尚无特征产物")
        overrides = {}
        if action == "full":
            overrides = {"log": "verbose", "plots": True, "reports": True}
        elif action == "quiet":
            overrides = {"log": "quiet"}
        with self.session.temporary_settings(overrides):
            path = self._recorded("feature", lambda: feature.run(self.session.config, rebuild=action == "rebuild"), primary_kind="feature")
        self.session.last_result = f"特征分析完成: {path}"
        return self.session.last_result

    def _prior(self, args: list[str]) -> str:
        action = args[0] if args else "normal"
        if action not in {"normal", "full", "quiet", "rebuild", "show"}:
            raise ValueError("prior 只接受 full/quiet/rebuild/show")
        if action == "show":
            return str(self.session.latest_prior or "当前会话尚无混合先验产物")
        overrides = {}
        if action == "full":
            overrides = {"log": "verbose", "plots": True, "reports": True}
        elif action == "quiet":
            overrides = {"log": "quiet"}
        with self.session.temporary_settings(overrides):
            path = self._recorded("prior", lambda: prior.run(self.session.config, rebuild=action == "rebuild"), primary_kind="prior")
        self.session.last_result = f"混合先验完成: {path}"
        return self.session.last_result

    def _train(self) -> str:
        checkpoint = self._recorded("train", lambda: training.run(self.session.config), primary_kind="checkpoint")
        validation.validate_checkpoint_compatibility(checkpoint, self.session.config)
        self.session.last_result = f"训练完成，当前模型: {self.session.checkpoint}"
        return self.session.last_result

    def _test(self, args: list[str]) -> str:
        checkpoint = self._require_checkpoint()
        mode = args[0] if args else None
        if mode and mode not in {"current", "history", "snapshot"}:
            raise ValueError("test 只接受 current/history/snapshot")
        overrides = {"plot_mode": mode} if mode else {}
        output = self.session.paths.predictions
        with self.session.temporary_settings(overrides):
            def operation() -> Path:
                prediction.run(self.session.config, checkpoint, output)
                csv_path = output / "all_reliability_values.csv"
                if not csv_path.is_file():
                    raise FileNotFoundError(f"预测流程未生成预期文件: {csv_path}")
                return csv_path
            csv_path = self._recorded("prediction", operation, primary_kind="prediction")
        self.session.last_result = f"预测完成: {csv_path}"
        return self.session.last_result

    def _explain(self) -> str:
        checkpoint = self._require_checkpoint()
        out = self._recorded(
            "explanation",
            lambda: explanation.run(self.session.config, checkpoint, self.session.paths.explanations),
            primary_kind="explanation",
        )
        self.session.last_result = f"解释完成: {out}"
        return self.session.last_result

    def _run(self, target: str) -> str:
        """执行透明的复合工作流；每一步仍调用对应正式工作流。"""
        if target == "train":
            self._check("all")
            return self._train()
        if target == "evaluate":
            self._require_checkpoint()
            outputs = [self._test([]), self._validate("prediction"), self._explain()]
            return "\n".join(outputs)
        if target == "all":
            self._check("all")
            outputs = [
                self._feature([]),
                self._prior([]),
                self._train(),
                self._test([]),
                self._validate("prediction"),
                self._explain(),
            ]
            return "\n".join(outputs)
        raise ValueError("run 只接受 train/evaluate/all")

    def _validate(self, target: str) -> str:
        if target in {"config", "data"}:
            validation.validate_config(self.session.config)
            return "配置和数据验证通过。"
        if target == "model":
            validation.validate_checkpoint_compatibility(self._require_checkpoint(), self.session.config)
            return "模型验证通过。"
        if target == "prediction":
            if not self.session.latest_prediction:
                raise ValueError("当前会话没有预测结果，请先运行 test。")
            validation.validate_prediction(self.session.latest_prediction)
            return "预测结果验证通过。"
        if target == "all":
            self._check("all")
            if self.session.latest_prediction:
                validation.validate_prediction(self.session.latest_prediction)
            return "全部可用对象验证通过。"
        raise ValueError(f"未知验证对象: {target}")

    def _require_checkpoint(self) -> Path:
        if not self.session.checkpoint:
            raise ValueError("尚未绑定模型。请先运行 train，或输入 model 选择 checkpoint。")
        validation.validate_checkpoint_compatibility(self.session.checkpoint, self.session.config)
        return self.session.checkpoint

    def _results(self) -> str:
        return (
            f"模型: {self.session.checkpoint or '-'}\n"
            f"特征: {self.session.latest_feature or '-'}\n"
            f"混合先验: {self.session.latest_prior or '-'}\n"
            f"预测: {self.session.latest_prediction or '-'}\n"
            f"解释: {self.session.latest_explanation or '-'}\n"
            f"解析配置: {self.session.latest_resolved_config or '-'}\n"
            f"运行记录: {self.session.latest_run_record or '-'}"
        )

    def _open(self, target: str) -> str:
        paths = self.session.paths
        mapping = {
            "output": paths.output_root,
            "plots": paths.plots,
            "reports": paths.reports,
            "logs": paths.logs,
            "model": self.session.checkpoint or paths.checkpoints,
        }
        path = Path(mapping.get(target, paths.output_root))
        path = path.parent if path.is_file() else path
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return f"已打开: {path}"
