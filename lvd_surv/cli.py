"""cli 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""统一命令行与交互式终端入口。"""

import argparse
import cmd
import shlex
from pathlib import Path
from typing import Any, Dict, List

from lvd_surv.core.artifacts import build_artifact_paths
from lvd_surv.core.errors import LVDWorkflowError
from lvd_surv.data.cmapss import add_train_rul, read_cmapss_txt
from lvd_surv.workflows.explanation import run_explanation_pipeline
from lvd_surv.workflows.prediction import run_prediction_pipeline
from lvd_surv.workflows.training import run_training_pipeline
from lvd_surv.runtime.context import RuntimeContext


def _common_output_flags(parser: argparse.ArgumentParser) -> None:
    """为分析型命令添加一致的输出控制参数。"""
    parser.add_argument("--verbosity", choices=["quiet", "normal", "verbose", "debug"], default="normal")
    parser.add_argument("--save-reports", action="store_true", help="保存详细表格/报告；不改变算法结果")
    parser.add_argument("--save-plots", action="store_true", help="保存分析图；不改变算法结果")


def build_parser() -> argparse.ArgumentParser:
    """构建顶层命令解析器；每个子命令都带用途与示例。"""
    parser = argparse.ArgumentParser(prog="lvd", description="LVD-Surv 严格可靠性预测控制台")
    sub = parser.add_subparsers(dest="command")

    train = sub.add_parser("train", help="执行严格训练流程")
    train.add_argument("--config", default="configs/default.yaml")
    _common_output_flags(train)

    feature = sub.add_parser("feature", help="单独执行特征分析")
    feature.add_argument("action", choices=["analyze"])
    feature.add_argument("--config", default="configs/default.yaml")
    _common_output_flags(feature)

    prior = sub.add_parser("prior", help="单独拟合混合寿命分布")
    prior.add_argument("action", choices=["analyze"])
    prior.add_argument("--config", default="configs/default.yaml")
    _common_output_flags(prior)

    predict = sub.add_parser("predict", help="使用 checkpoint 预测可靠度")
    predict.add_argument("--checkpoint", required=True)
    predict.add_argument("--test-file")
    predict.add_argument("--rul-file")
    predict.add_argument("--output-dir", default="outputs/predictions")
    predict.add_argument("--mc-samples", type=int, default=20)
    predict.add_argument("--plot-mode", choices=["current", "history", "snapshot"], default="current")
    predict.add_argument("--plot-max-curves", type=int, default=8)
    predict.add_argument("--no-prior", action="store_true")

    explain = sub.add_parser("explain", help="解释模型；默认从 checkpoint 配置读取训练文件")
    explain.add_argument("--checkpoint", required=True)
    explain.add_argument("--train-file")
    explain.add_argument("--output-dir", default="outputs/explanations")
    explain.add_argument("--repeats", type=int, default=3)
    explain.add_argument("--shap-sample-size", type=int, default=None, help="SHAP 正式解释样本上限；默认读取 checkpoint 配置")

    validate = sub.add_parser("config", help="配置操作")
    validate.add_argument("action", choices=["validate", "show"])
    validate.add_argument("--config", default="configs/default.yaml")

    sub.add_parser("doctor", help="检查配置、数据和关键依赖")
    release = sub.add_parser("release-check", help="执行发布结构、配置、数据、依赖和写权限检查")
    release.add_argument("--config", default="configs/default.yaml")
    release.add_argument("--skip-dependencies", action="store_true", help="仅用于离线结构检查；正式发布验收不要使用")
    shell = sub.add_parser("shell", help="进入交互式终端")
    shell.add_argument("--config", default="configs/default.yaml")
    return parser


def _context(args: argparse.Namespace) -> RuntimeContext:
    """由命令参数创建运行上下文。"""
    return RuntimeContext.from_config(
        args.config,
        verbosity=getattr(args, "verbosity", "normal"),
        save_reports=getattr(args, "save_reports", False),
        save_plots=getattr(args, "save_plots", False),
    )


def _load_training_frame(ctx: RuntimeContext):
    """为独立分析命令加载严格指定的训练文件。"""
    path = ctx.require_file(ctx.config.get("data", {}).get("train_file"), module="data", label="data.train_file")
    return add_train_rul(read_cmapss_txt(path))


def execute(args: argparse.Namespace) -> int:
    """执行一个已解析命令并返回进程退出码。"""
    if args.command in {None, "shell"}:
        LVDShell(getattr(args, "config", "configs/default.yaml")).cmdloop()
        return 0
    if args.command == "config":
        ctx = RuntimeContext.from_config(args.config)
        if args.action == "show":
            import pprint
            pprint.pp(ctx.config)
        else:
            data = ctx.config.get("data", {})
            ctx.require_file(data.get("train_file"), module="config", label="data.train_file")
            ctx.require_file(data.get("test_file"), module="config", label="data.test_file")
            ctx.require_file(data.get("rul_file"), module="config", label="data.rul_file")
            print("配置和数据路径校验通过。")
        return 0
    if args.command == "release-check":
        from lvd_surv.core.config import load_config
        from lvd_surv.workflows.validation import release_readiness
        config_path, cfg = load_config(args.config)
        summary = release_readiness(
            cfg,
            config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent,
            require_dependencies=not args.skip_dependencies,
        )
        print("发布检查通过:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        return 0
    if args.command == "doctor":
        import importlib
        required = ["numpy", "pandas", "sklearn", "torch", "scipy", "matplotlib", "yaml", "dcor", "xgboost", "shap"]
        failures = []
        for name in required:
            try:
                importlib.import_module(name)
                print(f"[OK] dependency: {name}")
            except Exception as exc:
                failures.append(f"{name}: {exc}")
                print(f"[FAIL] dependency: {name}: {exc}")
        if failures:
            raise LVDWorkflowError("doctor", "依赖检查", "; ".join(failures), "运行 `pip install -e .` 安装全部正式依赖")
        print("关键依赖检查通过。")
        return 0
    if args.command == "train":
        ctx = _context(args)
        checkpoint = run_training_pipeline(ctx.config)
        print(f"最佳 checkpoint: {checkpoint}")
        return 0
    if args.command == "feature":
        from lvd_surv.features.analysis import get_or_build_feature_artifacts
        from lvd_surv.data.cmapss import CMAPSS_SENSOR_COLS
        ctx = _context(args)
        frame = _load_training_frame(ctx)
        paths = build_artifact_paths(ctx.config)
        result = get_or_build_feature_artifacts(ctx.config, frame, [c for c in CMAPSS_SENSOR_COLS if c in frame.columns], paths)
        print(f"特征契约: {paths['feature_decision']}; cache_hit={result['cache_hit']}")
        return 0
    if args.command == "prior":
        from lvd_surv.lifetime.analysis import get_or_build_lifetime_prior
        ctx = _context(args)
        frame = _load_training_frame(ctx)
        paths = build_artifact_paths(ctx.config)
        result = get_or_build_lifetime_prior(ctx.config, frame, paths)
        print(f"混合寿命先验: {paths['mixture_prior']}; cache_hit={result['cache_hit']}")
        return 0
    if args.command == "predict":
        pred = run_prediction_pipeline(
            args.checkpoint, test_file=args.test_file, rul_file=args.rul_file,
            output_dir=args.output_dir, mc_samples=args.mc_samples,
            use_checkpoint_prior=not args.no_prior,
            plot_max_curves_per_device=args.plot_max_curves, plot_mode=args.plot_mode,
        )
        print(f"已保存 {len(pred)} 行可靠度结果到 {args.output_dir}")
        return 0
    if args.command == "explain":
        out = run_explanation_pipeline(
            args.checkpoint, train_file=args.train_file, output_dir=args.output_dir,
            repeats=args.repeats, shap_sample_size=args.shap_sample_size
        )
        print(f"解释结果已保存到 {out}")
        return 0
    return 2


class LVDShell(cmd.Cmd):
    """保存当前配置和 checkpoint 的交互式终端。"""

    intro = "LVD-Surv Shell。输入 help 查看命令；输入 exit 退出。"
    prompt = "lvd> "

    def __init__(self, config: str = "configs/default.yaml") -> None:
        super().__init__()
        self.config = config
        self.checkpoint = ""

    def default(self, line: str) -> None:
        """把交互输入转发给标准 CLI，确保两种入口行为一致。"""
        try:
            tokens = shlex.split(line)
            if not tokens:
                return
            if tokens[0] in {"train", "feature", "prior"} and "--config" not in tokens:
                tokens += ["--config", self.config]
            if tokens[0] in {"predict", "explain"} and "--checkpoint" not in tokens and self.checkpoint:
                tokens += ["--checkpoint", self.checkpoint]
            execute(build_parser().parse_args(tokens))
        except SystemExit:
            pass
        except Exception as exc:
            print(exc)

    def do_use(self, line: str) -> None:
        """设置会话配置：use configs/default.yaml。"""
        self.config = line.strip() or self.config
        print(f"当前配置: {self.config}")

    def do_checkpoint(self, line: str) -> None:
        """设置会话 checkpoint：checkpoint outputs/.../best_model.pt。"""
        self.checkpoint = line.strip()
        print(f"当前 checkpoint: {self.checkpoint}")

    def do_exit(self, line: str) -> bool:
        """退出终端。"""
        return True

    do_quit = do_exit


def main() -> None:
    """控制台脚本入口。"""
    try:
        raise SystemExit(execute(build_parser().parse_args()))
    except LVDWorkflowError as exc:
        print(exc)
        raise SystemExit(2)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"执行失败: {exc}")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
