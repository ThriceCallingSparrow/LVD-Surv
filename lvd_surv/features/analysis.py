"""features 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""C-MAPSS 时序特征分析正式接口"""

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from lvd_surv.data.cmapss import CMAPSS_SENSOR_COLS, normalize_cmapss_schema
from lvd_surv.core.artifacts import (
    build_config_fingerprint,
    build_data_fingerprint,
    build_manifest,
    build_script_fingerprint,
    get_dataset_name,
    is_cache_valid,
    load_json_artifact,
    load_pickle_artifact,
    save_json_artifact,
    save_pickle_artifact,
    update_run_state,
)
from lvd_surv.core.contracts import build_feature_decision, fit_feature_transformer_for_decision, validate_feature_decision



def run_feature_analysis_pipeline(
    train_df: pd.DataFrame,
    sensor_cols: Sequence[str],
    output_dir: str | Path,
    target_col: str = "rul",
    unit_id_col: str = "unit_id",
    time_col: str = "cycle",
    cfg: Mapping[str, Any] | None = None,
) -> dict:
    """运行正式特征分析；任何分析错误都会终止流程。

    ``save_tables`` 与 ``save_plots`` 只决定产物数量，不改变特征选择算法。函数不会在
    失败时构造相关系数画像或自动选择全部传感器。
    """
    df = normalize_cmapss_schema(train_df, add_condition_from_ops=True)
    output_dir = Path(output_dir)
    feature_cfg = dict((cfg or {}).get("features", {})) if isinstance(cfg, Mapping) else {}
    from lvd_surv.features import time_series as tsa
    target = target_col if target_col in df.columns else ("RUL" if "RUL" in df.columns else "rul")
    gui_mode = bool((cfg or {}).get("runtime", {}).get("gui_mode", False)) if isinstance(cfg, Mapping) else False
    vcfg = tsa.ValidationConfig(
        output_dir=str(output_dir),
        save_plots=bool(feature_cfg.get("save_plots", False)),
        save_tables=bool(feature_cfg.get("save_tables", False)),
        enable_deconfounding=bool(feature_cfg.get("enable_deconfounding", False)),
        show_progress=not gui_mode,
    )
    inspect_cfg = vcfg.to_inspect_config()
    df_all_units, df_summary = tsa.run_sensor_rul_corr_analysis_report(
        df, list(sensor_cols), target_col=target, unit_id_col=unit_id_col, config=inspect_cfg
    )
    if bool(feature_cfg.get("full_validation", True)):
        payload = tsa.automated_correlation_validation(
            df_all_units=df_all_units, df_summary=df_summary, train_df=df,
            sensor_cols=list(sensor_cols), target_col=target,
            unit_id_col=unit_id_col, time_col=time_col, config=vcfg,
        )
    else:
        payload = {"feature_profile": df_summary, "vi_plan": df_summary, "final_conclusion": "full_validation disabled"}
    payload = dict(payload) if isinstance(payload, Mapping) else {"feature_profile": payload}
    payload.setdefault("metadata", {})
    payload["metadata"].update({"source": "lvd_surv.features.time_series", "strict": True})
    return payload


def _feature_mode_requires_transformer(feature_decision: Mapping[str, Any]) -> bool:
    """判断特征契约是否需要可复用的残差变换器。"""
    return str(feature_decision.get("training_feature_mode", "raw")).lower() in {"residual", "hybrid"}


def get_or_build_feature_artifacts(cfg: Mapping[str, Any], train_df: pd.DataFrame, sensor_cols: Sequence[str], paths: Mapping[str, Path]) -> dict:
    """读取或生成特征分析缓存、特征契约和特征变换器"""
    dataset = get_dataset_name(cfg)
    feature_cfg = dict(cfg.get("features", {}))
    policy = str(feature_cfg.get("cache_policy", "auto")).lower()
    data_fp = build_data_fingerprint(train_df)
    cfg_fp = build_config_fingerprint(cfg, keys=["data", "features"])
    script_fp = build_script_fingerprint([__file__])
    cache_hit = policy == "auto" and is_cache_valid(
        paths["feature_manifest"], expected_dataset=dataset, artifact_paths=[paths["feature_decision"]],
        data_fingerprint=data_fp, config_fingerprint=cfg_fp, script_fingerprint=script_fp,
    )
    if cache_hit:
        print(f"[pipeline] Feature cache hit: {paths['feature_decision']}")
        decision = load_json_artifact(paths["feature_decision"])
        validate_feature_decision(decision)
        transformer = None
        if _feature_mode_requires_transformer(decision):
            if not paths["feature_transformer"].exists():
                print("[pipeline] Feature transformer missing; rebuilding feature artifacts.")
            else:
                transformer = load_pickle_artifact(paths["feature_transformer"])
                return {"feature_decision": decision, "analysis_bundle": None, "transformer": transformer, "cache_hit": True}
        else:
            return {"feature_decision": decision, "analysis_bundle": None, "transformer": None, "cache_hit": True}
    if policy == "readonly":
        raise FileNotFoundError("features.cache_policy=readonly but no valid feature cache was found.")
    if policy == "force":
        print("[pipeline] Feature cache force rebuild requested.")
    elif policy == "off":
        print("[pipeline] Feature cache disabled; running feature analysis.")
    else:
        print("[pipeline] No valid feature cache; running feature analysis.")
    update_run_state(paths["run_state"], stage="feature_selection", completed={"feature_selection": False})
    output_dir = paths["output_dir"] / "reports" / "features"
    bundle = run_feature_analysis_pipeline(train_df, sensor_cols, output_dir=output_dir, cfg=cfg)
    decision = build_feature_decision(feature_result=bundle, train_df=train_df, sensor_cols=list(sensor_cols), cfg={**dict(cfg), "features": feature_cfg}, dataset=dataset)
    transformer = fit_feature_transformer_for_decision(train_df, decision, cfg={**dict(cfg), "features": feature_cfg})
    if transformer is not None:
        decision = dict(decision)
        decision["feature_transformer_path"] = str(paths["feature_transformer"])
        decision["feature_transformer"] = transformer.to_contract() if hasattr(transformer, "to_contract") else {"type": type(transformer).__name__}
        save_pickle_artifact(transformer, paths["feature_transformer"])
    save_json_artifact(decision, paths["feature_decision"])
    save_pickle_artifact(bundle, paths["feature_bundle"])
    artifacts = {"feature_decision": str(paths["feature_decision"]), "feature_bundle": str(paths["feature_bundle"])}
    if _feature_mode_requires_transformer(decision):
        artifacts["feature_transformer"] = str(paths["feature_transformer"])
    manifest = build_manifest(dataset=dataset, artifact_type="feature_decision", data_fingerprint=data_fp,
                              config_fingerprint=cfg_fp, script_fingerprint=script_fp, artifacts=artifacts)
    save_json_artifact(manifest, paths["feature_manifest"])
    update_run_state(paths["run_state"], stage="feature_selection", completed={"feature_selection": True})
    return {"feature_decision": decision, "analysis_bundle": bundle, "transformer": transformer, "cache_hit": False}
