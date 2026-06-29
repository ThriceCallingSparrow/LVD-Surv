"""time_series_analysis 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GroupKFold, train_test_split
from sklearn.preprocessing import RobustScaler, PolynomialFeatures

import dcor
from xgboost import XGBRegressor

import shap

from lvd_surv.data.loader import CMAPSSLoader
from lvd_surv.runtime.cancellation import check_cancelled

# 配置区
@dataclass
class InspectConfig:
    """相关性筛选配置"""
    # 统计显著性水平
    alpha: float = 0.05

    # 样本量自适应效应量阈值
    effect_small_n: float = 0.3
    effect_mid_n: float = 0.2
    effect_large_n: float = 0.1

    # 相关强度阈值
    strong_corr_threshold: float = 0.8
    medium_corr_threshold: float = 0.4
    weak_corr_threshold: float = 0.2

    # 异常值处理
    remove_outliers: bool = True
    outlier_method: str = "trend_residual"  # trend_residual / sliding_window / iqr / 3rz
    trend_type: str = "quadratic"           # linear / quadratic
    window_size: int = 20
    window_step: int = 5

    # 距离相关
    enable_distance_corr: bool = True
    max_sample_limit: int = 2000
    small_sample_threshold: int = 500
    distance_corr_resamples: int = 1

    # 数据要求
    min_valid_unit_samples: int = 20
    min_valid_phase_samples: int = 10
    phase_ratios: Tuple[float, float] = (0.6, 0.2)

    # 输出与随机种子
    random_state: int = 42
    show_progress: bool = True
    output_dir: str = "analysis_result"
    save_tables: bool = True
    save_plots: bool = True

    def copy(self, **kwargs: Any) -> "InspectConfig":
        """执行 copy 对应的项目处理逻辑。"""
        data = self.__dict__.copy()
        invalid = set(kwargs) - set(data)
        if invalid:
            raise AttributeError(f"InspectConfig 无属性: {invalid}")
        data.update(kwargs)
        return InspectConfig(**data)

@dataclass
class ValidationConfig:
    """验证、建模与最终结论配置"""
    output_dir: str = "analysis_result"
    save_plots: bool = True
    save_tables: bool = True
    show_progress: bool = True

    check_constant_columns: bool = True
    check_outlier_effect: bool = True

    n_folds: int = 5
    random_state: int = 42

    n_degradation_units: int = 2
    n_corr_examples: int = 3

    min_valid_unit_samples: int = 20
    model_random_state: int = 42
    test_size: float = 0.2

    weak_corr_threshold: float = 0.2
    medium_corr_threshold: float = 0.4
    strong_corr_threshold: float = 0.8
    high_stability_threshold: float = 0.1
    medium_stability_threshold: float = 0.2
    direction_consistency_threshold: float = 0.8

    phase_ratios: Tuple[float, float] = (0.6, 0.2)

    interaction_types: List[str] = field(default_factory=lambda: ["product", "ratio"])
    top_interaction_num: int = 5
    inter_keep_ratio: float = 0.40
    min_keep_inter: int = 5

    # Feature profile阈值
    full_cycle_phase_threshold: float = 0.60
    phase_core_threshold: float = 0.70
    low_direction_threshold: float = 0.60
    shap_high_quantile: float = 0.75
    shap_mid_quantile: float = 0.50

    # 工况去混杂配置: 只使用 op_* 与 condition
    enable_deconfounding: bool = True
    condition_col: str = "condition"
    op_cols: Optional[List[str]] = None
    condition_model_degree: int = 2
    condition_model_alpha: float = 1.0
    deconfounding_n_splits: int = 5
    deconfounding_resid_suffix: str = "_resid"
    condition_reduction_good: float = 0.50
    residual_condition_r2_good: float = 0.10
    residual_condition_r2_warn: float = 0.30
    degradation_retention_min: float = 0.30
    degradation_retention_good: float = 0.50

    def to_inspect_config(self) -> InspectConfig:
        """为验证过程中的重复相关性分析生成统一 InspectConfig。"""
        return InspectConfig(
            output_dir=self.output_dir,
            save_tables=self.save_tables,
            save_plots=self.save_plots,
            min_valid_unit_samples=self.min_valid_unit_samples,
            phase_ratios=self.phase_ratios,
            weak_corr_threshold=self.weak_corr_threshold,
            medium_corr_threshold=self.medium_corr_threshold,
            strong_corr_threshold=self.strong_corr_threshold,
            random_state=self.random_state,
            show_progress=self.show_progress,
        )

METRIC_USAGE: Dict[str, str] = {
    "跨设备显著率": "判断特征是否在多数设备上稳定体现退化相关性。",
    "平均最大相关系数": "判断特征与 RUL 的整体效应强度。",
    "相关系数标准差": "判断跨设备相关性波动，辅助识别不稳定特征。",
    "早期显著率": "判断特征是否在生命周期早期敏感。",
    "中期显著率": "判断特征是否在生命周期中期敏感。",
    "晚期显著率": "判断特征是否在生命周期晚期敏感。",
    "最常见主导类型": "判断特征主要呈线性、单调非线性或复杂非线性关系。",
    "稳定性等级": "判断相关性结论在不同设备折分下是否稳定。",
    "方向一致性": "判断特征与 RUL 的变化方向是否具有可解释性。",
    "SHAP重要性": "判断特征进入模型后的预测贡献。",
    "阶段RMSE": "判断不同生命周期阶段建模是否有效。",
    "原始工况解释R2": "判断传感器读数中有多少可由工况变量解释。",
    "残差工况解释R2": "判断去混杂后的残差中是否仍残留工况信息。",
    "工况影响削弱率": "判断工况影响是否被有效削弱。",
    "退化信号保留率": "判断去工况后是否仍保留与 RUL 相关的退化信息。",
}

# 通用 IO 与辅助函数
def create_output_dir(config: Any) -> None:
    """执行 create output dir 对应的项目处理逻辑。"""
    os.makedirs(config.output_dir, exist_ok=True)
    for subdir in ["plots", "tables", "reports"]:
        os.makedirs(os.path.join(config.output_dir, subdir), exist_ok=True)

def save_plot(fig: plt.Figure, name: str, config: Any) -> None:
    """执行 save plot 对应的项目处理逻辑。"""
    if getattr(config, "save_plots", True):
        create_output_dir(config)
        path = os.path.join(config.output_dir, "plots", f"{name}.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_table(df: Optional[pd.DataFrame], name: str, config: Any) -> None:
    """执行 save table 对应的项目处理逻辑。"""
    if df is not None and getattr(config, "save_tables", True):
        create_output_dir(config)
        path = os.path.join(config.output_dir, "tables", f"{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")

def write_report(content: str, name: str, config: Any) -> None:
    """执行 write report 对应的项目处理逻辑。"""
    create_output_dir(config)
    path = os.path.join(config.output_dir, "reports", f"{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def is_constant_arr(arr: np.ndarray, eps: float = 1e-12) -> bool:
    """执行 is constant arr 对应的项目处理逻辑。"""
    arr = np.asarray(arr).ravel()
    if len(arr) == 0 or np.isnan(arr).all():
        return True
    return np.nanvar(arr) < eps

def safe_mean(series: pd.Series, default: float = 0.0) -> float:
    """执行 safe mean 对应的项目处理逻辑。"""
    if series is None or len(series) == 0:
        return default
    value = series.mean()
    return default if pd.isna(value) else float(value)

def safe_std(series: pd.Series, default: float = 0.0) -> float:
    """执行 safe std 对应的项目处理逻辑。"""
    if series is None or len(series) <= 1:
        return default
    value = series.std()
    return default if pd.isna(value) else float(value)

def get_min_effect_size(n: int, config: InspectConfig) -> float:
    """执行 get min effect size 对应的项目处理逻辑。"""
    if n < 100:
        return config.effect_small_n
    if n < 1000:
        return config.effect_mid_n
    return config.effect_large_n

def is_statistically_significant(corr: float, p_value: float, n: int, config: InspectConfig) -> bool:
    """执行 is statistically significant 对应的项目处理逻辑。"""
    if pd.isna(corr) or pd.isna(p_value):
        return False
    return bool(abs(corr) >= get_min_effect_size(n, config) and p_value < config.alpha)

# 异常值处理与生命周期划分
def remove_outliers_iqr(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """执行 remove outliers iqr 对应的项目处理逻辑。"""
    df = data.copy()
    mask = pd.Series(True, index=df.index)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask &= (df[col] >= lower) & (df[col] <= upper)
    return df[mask]

def remove_outliers_3rz(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """执行 remove outliers 3rz 对应的项目处理逻辑。"""
    df = data.copy()
    scaler = RobustScaler()
    mask = pd.Series(True, index=df.index)
    for col in columns:
        scaled = scaler.fit_transform(df[[col]])
        mask &= np.abs(scaled[:, 0]) < 3
    return df[mask]

def trend_separate_outlier_detect(
    data: pd.DataFrame,
    columns: List[str],
    trend_type: str = "linear",
) -> pd.DataFrame:
    """执行 trend separate outlier detect 对应的项目处理逻辑。"""
    df = data.copy()
    mask = pd.Series(True, index=df.index)

    for col in columns:
        values = df[col].values.astype(float)
        if len(values) < 5 or is_constant_arr(values):
            continue

        x = np.arange(len(df)).reshape(-1, 1)
        if trend_type == "quadratic":
            x = np.hstack([x, x ** 2])

        model = LinearRegression()
        model.fit(x, values)
        residual = values - model.predict(x)
        mad = np.median(np.abs(residual - np.median(residual)))

        if mad <= 1e-12:
            continue
        threshold = 3 * mad
        mask &= np.abs(residual) <= threshold

    return df[mask]

def sliding_window_outlier_detect(
    data: pd.DataFrame,
    columns: List[str],
    window_size: int = 20,
    step: int = 5,
) -> pd.DataFrame:
    """执行 sliding window outlier detect 对应的项目处理逻辑。"""
    df = data.copy()
    mask = pd.Series(True, index=df.index)

    for col in columns:
        values = df[col].values.astype(float)
        outlier_flags = np.zeros(len(values), dtype=bool)
        if len(values) < window_size:
            continue

        for start in range(0, len(values) - window_size + 1, step):
            end = start + window_size
            window_vals = values[start:end]
            q1 = np.percentile(window_vals, 25)
            q3 = np.percentile(window_vals, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_flags[start:end] |= (window_vals < lower) | (window_vals > upper)

        mask &= ~outlier_flags

    return df[mask]

def apply_outlier_filter(data: pd.DataFrame, columns: List[str], config: InspectConfig) -> pd.DataFrame:
    """执行 apply outlier filter 对应的项目处理逻辑。"""
    if not config.remove_outliers:
        return data.copy()
    if config.outlier_method == "trend_residual":
        return trend_separate_outlier_detect(data, columns, config.trend_type)
    if config.outlier_method == "sliding_window":
        return sliding_window_outlier_detect(data, columns, config.window_size, config.window_step)
    if config.outlier_method == "iqr":
        return remove_outliers_iqr(data, columns)
    if config.outlier_method == "3rz":
        return remove_outliers_3rz(data, columns)
    raise ValueError(f"未知异常值处理方法: {config.outlier_method}")

def split_lifecycle_phases(
    data: pd.DataFrame,
    target_col: str,
    phase_ratios: Tuple[float, float] = (0.6, 0.2),
) -> pd.DataFrame:
    """执行 split lifecycle phases 对应的项目处理逻辑。"""
    df = data.copy()
    max_rul = df[target_col].max()
    early_thresh = max_rul * phase_ratios[0]
    mid_thresh = max_rul * phase_ratios[1]

    def get_phase(rul: float) -> str:
        """执行 get phase 对应的项目处理逻辑。"""
        if rul > early_thresh:
            return "early"
        if rul >= mid_thresh:
            return "mid"
        return "late"

    df["lifecycle_phase"] = df[target_col].apply(get_phase)
    return df

# 统一相关性计算
def calculate_distance_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_sample: Optional[int] = None,
    small_sample_threshold: int = 500,
    random_state: int = 42,
    num_resamples: int = 500,
) -> Tuple[float, float]:
    """执行 calculate distance correlation 对应的项目处理逻辑。"""
    rng = np.random.RandomState(random_state)

    x = np.asarray(x)
    y = np.asarray(y)
    x = x.reshape(-1, 1) if x.ndim == 1 else x
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    n_total = len(x)
    if n_total == 0:
        return np.nan, np.nan

    if n_total <= small_sample_threshold:
        res = dcor.independence.distance_covariance_test(
            x, y, num_resamples=num_resamples, random_state=random_state
        )
        return float(dcor.distance_correlation(x, y)), float(res.pvalue)

    if max_sample is None or max_sample > n_total:
        max_sample = n_total

    init_n = max(int(n_total * 0.5), small_sample_threshold)
    step = max(int(init_n * 0.5), 1)

    history: List[Tuple[float, float]] = []
    current_n = min(init_n, max_sample)
    prev_p: Optional[float] = None

    while current_n <= max_sample:
        idx = rng.choice(n_total, size=current_n, replace=False)
        x_sub = x[idx, :]
        y_sub = y[idx, :]
        res = dcor.independence.distance_covariance_test(
            x_sub, y_sub, num_resamples=num_resamples, random_state=random_state
        )
        p_val = float(res.pvalue)
        dcor_val = float(dcor.distance_correlation(x_sub, y_sub))
        history.append((dcor_val, p_val))

        if prev_p is not None and abs(p_val - prev_p) < 1e-3:
            break
        prev_p = p_val
        current_n += step

    return history[-1] if history else (np.nan, np.nan)

def empty_corr_metrics(prefix: str = "") -> Dict[str, Any]:
    """执行 empty corr metrics 对应的项目处理逻辑。"""
    base = {
        "Pearson_r": np.nan,
        "Pearson_p": np.nan,
        "Pearson显著": False,
        "Spearman_r": np.nan,
        "Spearman_p": np.nan,
        "Spearman显著": False,
        "dcorr_r": np.nan,
        "dcorr_p": np.nan,
        "dcorr显著": False,
        "最大相关系数": 0.0,
        "相关强度": "无明显相关",
        "主导相关类型": "无明显相关",
    }
    if prefix:
        return {f"{prefix}_{k}": v for k, v in base.items() if k not in ["相关强度", "主导相关类型"]}
    return base

def compute_corr_metrics(x: np.ndarray, y: np.ndarray, config: InspectConfig) -> Dict[str, Any]:
    """统一计算 Pearson / Spearman / Distance Correlation, 并给出主导关系"""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    x = x[valid_mask]
    y = y[valid_mask]
    n = len(x)

    if n < 2 or is_constant_arr(x) or is_constant_arr(y):
        return empty_corr_metrics()

    pearson_r, pearson_p = pearsonr(x, y)
    spearman_r, spearman_p = spearmanr(x, y)
    pearson_sig = is_statistically_significant(pearson_r, pearson_p, n, config)
    spearman_sig = is_statistically_significant(spearman_r, spearman_p, n, config)

    dcorr_r, dcorr_p, dcorr_sig = np.nan, np.nan, False
    if config.enable_distance_corr:
        dcorr_r, dcorr_p = calculate_distance_correlation(
            x,
            y,
            max_sample=config.max_sample_limit,
            small_sample_threshold=config.small_sample_threshold,
            random_state=config.random_state,
            num_resamples=config.distance_corr_resamples,
        )
        dcorr_sig = is_statistically_significant(dcorr_r, dcorr_p, n, config)

    candidate_scores = {
        "线性相关": abs(pearson_r) if not pd.isna(pearson_r) else 0.0,
        "单调非线性相关": abs(spearman_r) if not pd.isna(spearman_r) else 0.0,
        "复杂非线性相关": dcorr_r if not pd.isna(dcorr_r) else 0.0,
    }
    max_abs_corr = max(candidate_scores.values())
    dominant_type = max(candidate_scores, key=candidate_scores.get)

    if max_abs_corr < config.weak_corr_threshold:
        dominant_type = "无明显相关"
        strength = "无明显相关"
    elif max_abs_corr >= config.strong_corr_threshold:
        strength = "强相关"
    elif max_abs_corr >= config.medium_corr_threshold:
        strength = "中等相关"
    else:
        strength = "弱相关"

    return {
        "Pearson_r": round(float(pearson_r), 4),
        "Pearson_p": round(float(pearson_p), 6),
        "Pearson显著": bool(pearson_sig),
        "Spearman_r": round(float(spearman_r), 4),
        "Spearman_p": round(float(spearman_p), 6),
        "Spearman显著": bool(spearman_sig),
        "dcorr_r": round(float(dcorr_r), 4) if not pd.isna(dcorr_r) else np.nan,
        "dcorr_p": round(float(dcorr_p), 6) if not pd.isna(dcorr_p) else np.nan,
        "dcorr显著": bool(dcorr_sig),
        "最大相关系数": round(float(max_abs_corr), 4),
        "相关强度": strength,
        "主导相关类型": dominant_type,
    }

def prefix_metrics(metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """执行 prefix metrics 对应的项目处理逻辑。"""
    skip = {"相关强度", "主导相关类型"}
    return {f"{prefix}_{k}": v for k, v in metrics.items() if k not in skip}

# 单设备与跨设备相关性分析
def _analyze_single_unit(
    unit_data: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: InspectConfig,
) -> List[Dict[str, Any]]:
    unit_result: List[Dict[str, Any]] = []
    unit_id = unit_data[unit_id_col].iloc[0]
    phases = ["early", "mid", "late"]

    for sen_col in sensor_cols:
        raw_data = unit_data[[unit_id_col, sen_col, target_col]].dropna()
        data = apply_outlier_filter(raw_data, [sen_col], config)
        n = len(data)

        base_row: Dict[str, Any] = {
            "unit_id": unit_id,
            "传感器列": sen_col,
            "有效样本量": n,
        }

        if n < config.min_valid_unit_samples:
            row = {**base_row, **empty_corr_metrics()}
            for phase in phases:
                row.update(empty_corr_metrics(prefix=phase))
            row.update({"筛选建议": "剔除", "备注": "样本量过少"})
            unit_result.append(row)
            continue

        if is_constant_arr(data[sen_col].values):
            row = {**base_row, **empty_corr_metrics()}
            for phase in phases:
                row.update(empty_corr_metrics(prefix=phase))
            row.update({"筛选建议": "剔除", "备注": "常数列"})
            unit_result.append(row)
            continue

        # 全局相关性
        metrics = compute_corr_metrics(data[sen_col].values, data[target_col].values, config)

        # 生命周期阶段相关性: 使用异常值处理后的数据, 保证阶段结论与全局样本一致
        phased_data = split_lifecycle_phases(data, target_col, config.phase_ratios)
        phase_metrics: Dict[str, Any] = {}
        for phase in phases:
            phase_data = phased_data[phased_data["lifecycle_phase"] == phase]
            if len(phase_data) < config.min_valid_phase_samples:
                phase_metrics.update(empty_corr_metrics(prefix=phase))
            else:
                phase_metrics.update(
                    prefix_metrics(
                        compute_corr_metrics(
                            phase_data[sen_col].values,
                            phase_data[target_col].values,
                            config,
                        ),
                        phase,
                    )
                )

        suggestion = "保留" if metrics["最大相关系数"] >= config.weak_corr_threshold else "剔除"
        row = {
            **base_row,
            **phase_metrics,
            **metrics,
            "筛选建议": suggestion,
            "备注": "正常",
        }
        unit_result.append(row)

    return unit_result

def summarize_cross_unit_results(
    df_all_units: pd.DataFrame,
    sensor_cols: List[str],
    config: InspectConfig,
) -> pd.DataFrame:
    """执行 summarize cross unit results 对应的项目处理逻辑。"""
    summary_rows: List[Dict[str, Any]] = []

    for sen_col in sensor_cols:
        sen_data = df_all_units[
            (df_all_units["传感器列"] == sen_col) &
            (~df_all_units["最大相关系数"].isna())
        ].copy()

        if len(sen_data) == 0:
            summary_rows.append({
                "传感器列": sen_col,
                "有效设备数": 0,
                "显著设备数": 0,
                "跨设备显著率": 0.0,
                "早期显著率": 0.0,
                "中期显著率": 0.0,
                "晚期显著率": 0.0,
                "平均最大相关系数": 0.0,
                "相关系数Median": 0.0,
                "相关系数标准差": 0.0,
                "最常见主导类型": "无",
                "最终筛选建议": "剔除",
                "备注": "所有设备均无有效数据",
            })
            continue

        total_units = len(sen_data)
        sig_mask = (
            ((sen_data["Pearson显著"] == True) & (sen_data["Pearson_r"].abs() >= config.weak_corr_threshold)) |
            ((sen_data["Spearman显著"] == True) & (sen_data["Spearman_r"].abs() >= config.weak_corr_threshold)) |
            ((sen_data["dcorr显著"] == True) & (sen_data["dcorr_r"] >= config.weak_corr_threshold))
        )
        sig_units = int(sig_mask.sum())
        sig_rate = sig_units / total_units if total_units else 0.0

        avg_max_corr = safe_mean(sen_data["最大相关系数"])
        std_max_corr = safe_std(sen_data["最大相关系数"])
        median_max_corr = float(sen_data["最大相关系数"].median()) if len(sen_data) else 0.0

        type_counts = sen_data["主导相关类型"].value_counts(normalize=True).to_dict()
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "无"

        early_sig_rate = np.mean([
            safe_mean(sen_data["early_Pearson显著"]),
            safe_mean(sen_data["early_Spearman显著"]),
            safe_mean(sen_data["early_dcorr显著"]),
        ])
        mid_sig_rate = np.mean([
            safe_mean(sen_data["mid_Pearson显著"]),
            safe_mean(sen_data["mid_Spearman显著"]),
            safe_mean(sen_data["mid_dcorr显著"]),
        ])
        late_sig_rate = np.mean([
            safe_mean(sen_data["late_Pearson显著"]),
            safe_mean(sen_data["late_Spearman显著"]),
            safe_mean(sen_data["late_dcorr显著"]),
        ])

        if sig_rate < 0.30:
            final_suggestion = "剔除"
        elif sig_rate >= 0.70 and avg_max_corr >= config.medium_corr_threshold:
            final_suggestion = "推荐保留"
        elif sig_rate >= 0.50:
            final_suggestion = "建议保留"
        else:
            final_suggestion = "谨慎保留 (仅部分设备相关)"

        summary_rows.append({
            "传感器列": sen_col,
            "有效设备数": total_units,
            "显著设备数": sig_units,
            "跨设备显著率": round(sig_rate, 4),
            "早期显著率": round(float(early_sig_rate), 4),
            "中期显著率": round(float(mid_sig_rate), 4),
            "晚期显著率": round(float(late_sig_rate), 4),
            "平均最大相关系数": round(avg_max_corr, 4),
            "相关系数Median": round(median_max_corr, 4),
            "相关系数标准差": round(std_max_corr, 4),
            "最常见主导类型": dominant_type,
            "最终筛选建议": final_suggestion,
            "备注": "正常",
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary.sort_values(
        by=["跨设备显著率", "平均最大相关系数"],
        ascending=False,
    ).reset_index(drop=True)

def analyze_sensor_rul_correlation_by_unit(
    df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: InspectConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行 analyze sensor rul correlation by unit 对应的项目处理逻辑。"""
    all_results: List[Dict[str, Any]] = []
    unit_ids = df[unit_id_col].dropna().unique()

    iterator = tqdm(unit_ids, desc="分析设备相关性") if config.show_progress else unit_ids
    for unit_id in iterator:
        check_cancelled("特征相关性分析")
        unit_data = df[df[unit_id_col] == unit_id]
        all_results.extend(_analyze_single_unit(unit_data, sensor_cols, target_col, unit_id_col, config))

    df_all_units = pd.DataFrame(all_results)
    df_summary = summarize_cross_unit_results(df_all_units, sensor_cols, config)
    return df_all_units, df_summary

def run_sensor_rul_corr_analysis_report(
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: Optional[InspectConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行 run sensor rul corr analysis report 对应的项目处理逻辑。"""
    if config is None:
        config = InspectConfig()
    create_output_dir(config)

    df_all_units, df_summary = analyze_sensor_rul_correlation_by_unit(
        train_df, sensor_cols, target_col, unit_id_col, config
    )

    print("=" * 100)
    print("Sensor cross-unit correlation summary")
    print("=" * 100)
    print(df_summary.to_string(index=False))

    print("\nFinal screening result:")
    print("-" * 50)
    for suggestion in ["推荐保留", "建议保留", "谨慎保留 (仅部分设备相关)", "剔除"]:
        cols = df_summary[df_summary["最终筛选建议"] == suggestion]["传感器列"].tolist()
        print(f"{suggestion} ({len(cols)}): {cols}")

    save_table(df_all_units, "all_units_results", config)
    save_table(df_summary, "all_units_summary", config)
    return df_all_units, df_summary

# 验证模块
def run_quick_self_check(
    df_summary: pd.DataFrame,
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: ValidationConfig,
) -> Tuple[str, Optional[pd.DataFrame]]:
    """执行 run quick self check 对应的项目处理逻辑。"""
    report: List[str] = ["=" * 80, "快速自检", "=" * 80]
    outlier_compare_df: Optional[pd.DataFrame] = None

    if config.check_constant_columns:
        report.append("\n常数列检查")
        unit_constant_counts: Dict[str, int] = {}
        for sen_col in sensor_cols:
            n_constant = train_df.groupby(unit_id_col)[sen_col].apply(lambda x: is_constant_arr(x.values)).sum()
            if n_constant > 0:
                unit_constant_counts[sen_col] = int(n_constant)

        if unit_constant_counts:
            report.append("   设备级常数列统计：")
            for sen, cnt in unit_constant_counts.items():
                report.append(f"   - {sen}: {cnt} 台设备为常数列")
        else:
            report.append("   未发现设备级常数列")

    report.append("\n异常结果检查")
    zero_corr_sensors = df_summary[df_summary["平均最大相关系数"] == 0]["传感器列"].tolist()
    if zero_corr_sensors:
        report.append(f"   发现全零相关传感器: {zero_corr_sensors}")
    else:
        report.append("   未发现全零相关传感器")

    suspicious = df_summary[
        (df_summary["跨设备显著率"] > 0.5) &
        (df_summary["平均最大相关系数"] < config.weak_corr_threshold)
    ]["传感器列"].tolist()
    if suspicious:
        report.append(f"\n   高显著率低相关的可疑传感器: {suspicious}")
        report.append("   可能原因: 大样本 p 值陷阱，建议结合建模验证最终决定")
    else:
        report.append("\n   未发现高显著率低相关的可疑传感器")

    if config.check_outlier_effect:
        report.append("\n异常值处理影响验证")
        no_outlier_cfg = config.to_inspect_config().copy(remove_outliers=False)
        _, df_summary_no_outlier = analyze_sensor_rul_correlation_by_unit(
            train_df, sensor_cols, target_col, unit_id_col, no_outlier_cfg
        )
        outlier_compare_df = pd.merge(
            df_summary[["传感器列", "跨设备显著率"]],
            df_summary_no_outlier[["传感器列", "跨设备显著率"]],
            on="传感器列",
            suffixes=("_with_outlier_filter", "_no_outlier_filter"),
        )
        outlier_compare_df["显著率变化"] = (
            outlier_compare_df["跨设备显著率_with_outlier_filter"] -
            outlier_compare_df["跨设备显著率_no_outlier_filter"]
        ).abs()
        outlier_compare_df = outlier_compare_df.sort_values("显著率变化", ascending=False)
        save_table(outlier_compare_df, "outlier_effect_comparison", config)

        max_change = outlier_compare_df["显著率变化"].max() if len(outlier_compare_df) else 0.0
        if max_change > 0.2:
            report.append(f"   异常值处理对结果影响较大，最大变化: {max_change:.2%}")
            report.append("   建议检查异常值是否是真实退化信号。")
        else:
            report.append(f"   异常值处理对结果影响较小，最大变化: {max_change:.2%}")

    return "\n".join(report), outlier_compare_df

def run_statistical_validation(
    df_all_units: pd.DataFrame,
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: ValidationConfig,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """执行 run statistical validation 对应的项目处理逻辑。"""
    report: List[str] = ["\n" + "=" * 80, "统计稳定性验证", "=" * 80]
    unit_ids = train_df[unit_id_col].dropna().unique()

    n_splits = min(config.n_folds, len(unit_ids))
    if n_splits < 2:
        report.append("设备数量不足，跳过 K 折稳定性验证。")
        empty = pd.DataFrame()
        return "\n".join(report), empty, empty, empty

    report.append(f"\n按设备分组进行 {n_splits} 折交叉验证")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    fold_results: List[pd.DataFrame] = []
    inspect_cfg = config.to_inspect_config()

    fold_iterator = tqdm(kf.split(unit_ids), desc="运行交叉验证", disable=not config.show_progress)
    for fold, (_, test_idx) in enumerate(fold_iterator):
        check_cancelled("特征交叉验证")
        test_units = unit_ids[test_idx]
        fold_df = train_df[train_df[unit_id_col].isin(test_units)]
        _, fold_summary = analyze_sensor_rul_correlation_by_unit(
            fold_df, sensor_cols, target_col, unit_id_col, inspect_cfg
        )
        fold_summary["fold"] = fold
        fold_results.append(fold_summary)

    df_fold = pd.concat(fold_results, ignore_index=True)
    save_table(df_fold, "cross_validation_results", config)

    stability = df_fold.groupby("传感器列")["跨设备显著率"].agg(["mean", "std"]).reset_index()
    stability.columns = ["传感器列", "平均显著率", "显著率标准差"]
    stability["显著率标准差"] = stability["显著率标准差"].fillna(0.0)

    def get_stability_level(std: float) -> str:
        """执行 get stability level 对应的项目处理逻辑。"""
        if std < config.high_stability_threshold:
            return "高稳定"
        if std < config.medium_stability_threshold:
            return "稳定"
        return "不稳定"

    stability["稳定性等级"] = stability["显著率标准差"].apply(get_stability_level)
    stability = stability.sort_values(["稳定性等级", "显著率标准差"])
    save_table(stability, "sensor_stability_scores", config)

    report.append("   稳定性统计:")
    for level in ["高稳定", "稳定", "不稳定"]:
        cnt = int((stability["稳定性等级"] == level).sum())
        report.append(f"   - {level}: {cnt} 个传感器")

    report.append("\n相关性方向一致性检验")
    direction_rows: List[Dict[str, Any]] = []
    for sen_col in sensor_cols:
        sen_data = df_all_units[
            (df_all_units["传感器列"] == sen_col) &
            (~df_all_units["Spearman_r"].isna())
        ]
        if len(sen_data) == 0:
            direction_rows.append({
                "传感器列": sen_col,
                "正相关比例": np.nan,
                "方向一致性": 0.0,
                "主导方向": "无有效方向",
            })
            continue
        pos_ratio = float((sen_data["Spearman_r"] > 0).mean())
        consistency = max(pos_ratio, 1 - pos_ratio)
        dominant_direction = "与RUL正相关" if pos_ratio >= 0.5 else "与RUL负相关"
        direction_rows.append({
            "传感器列": sen_col,
            "正相关比例": round(pos_ratio, 4),
            "方向一致性": round(consistency, 4),
            "主导方向": dominant_direction,
        })

    direction_df = pd.DataFrame(direction_rows).sort_values("方向一致性", ascending=False)
    save_table(direction_df, "direction_consistency", config)

    high_consistency = direction_df[
        direction_df["方向一致性"] >= config.direction_consistency_threshold
    ]["传感器列"].tolist()
    low_consistency = direction_df[
        direction_df["方向一致性"] < config.direction_consistency_threshold
    ]["传感器列"].tolist()
    report.append(f"   高方向一致性(≥{config.direction_consistency_threshold:.0%}): {high_consistency}")
    if low_consistency:
        report.append(f"   低方向一致性: {low_consistency}")

    return "\n".join(report), stability, direction_df, df_fold

def run_visual_validation(
    df_all_units: pd.DataFrame,
    df_summary: pd.DataFrame,
    train_df: pd.DataFrame,
    target_col: str,
    unit_id_col: str,
    time_col: str,
    config: ValidationConfig,
) -> str:
    """执行 run visual validation 对应的项目处理逻辑。"""
    report: List[str] = ["\n" + "=" * 80, "可视化验证", "=" * 80]
    unit_ids = train_df[unit_id_col].dropna().unique()

    report.append("\n生成跨设备相关系数Distribution箱线图")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=df_all_units,
        x="传感器列",
        y="最大相关系数",
        order=df_summary["传感器列"].tolist(),
        ax=ax,
    )
    ax.axhline(y=config.weak_corr_threshold, color="r", linestyle="--", label="Weak threshold")
    ax.set_title("Distribution of Maximum Cross-unit Correlation Coefficient per Sensor")
    ax.set_xlabel("Sensor")
    ax.set_ylabel("Maximum Correlation Coefficient")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_plot(fig, "correlation_distribution_boxplot", config)

    top_sensors = df_summary["传感器列"].head(3).tolist()
    n_sample_units = min(config.n_degradation_units, len(unit_ids))
    rng = np.random.RandomState(config.random_state)

    report.append("\n生成 Top 传感器退化曲线")
    for sensor in top_sensors:
        fig, ax = plt.subplots(figsize=(12, 5))
        sample_units = rng.choice(unit_ids, n_sample_units, replace=False)
        for unit in sample_units:
            unit_data = train_df[train_df[unit_id_col] == unit].sort_values(time_col)
            ax.plot(unit_data[time_col], unit_data[sensor], alpha=0.7, label=f"Unit {unit}")
        ax.set_title(f"{sensor} Degradation Curve")
        ax.set_xlabel("Time Cycle")
        ax.set_ylabel("Sensor Reading")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        save_plot(fig, f"degradation_curve_{sensor}", config)

    report.append("\n生成相关类型验证散点图")
    for sensor in top_sensors:
        sen_data = df_all_units[df_all_units["传感器列"] == sensor].dropna(subset=["最大相关系数"])
        top_units = sen_data.sort_values("最大相关系数", ascending=False).head(config.n_corr_examples)["unit_id"].tolist()
        for unit in top_units:
            unit_data = train_df[train_df[unit_id_col] == unit]
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(x=unit_data[sensor], y=unit_data[target_col], alpha=0.6, ax=ax)
            sns.regplot(x=unit_data[sensor], y=unit_data[target_col], scatter=False, color="r", label="Linear Fit", ax=ax)
            sns.regplot(x=unit_data[sensor], y=unit_data[target_col], scatter=False, color="g", order=2, label="Quadratic Fit", ax=ax)
            corr_info = sen_data[sen_data["unit_id"] == unit].iloc[0]
            ax.set_title(
                f"Unit {unit} - {sensor} vs RUL\n"
                f"Pearson={corr_info['Pearson_r']:.2f}, "
                f"Spearman={corr_info['Spearman_r']:.2f}, "
                f"Dcorr={corr_info['dcorr_r']:.2f}"
            )
            ax.set_xlabel(sensor)
            ax.set_ylabel("RUL")
            ax.legend()
            plt.tight_layout()
            save_plot(fig, f"corr_type_{sensor}_unit_{unit}", config)

    report.append("   所有可视化图表已保存至 plots 目录")
    return "\n".join(report)

# 建模验证与交互特征
def generate_interaction_features(
    data: pd.DataFrame,
    sensor_cols: List[str],
    interaction_types: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """执行 generate interaction features 对应的项目处理逻辑。"""
    if interaction_types is None:
        interaction_types = ["product", "ratio"]

    df = data.copy()
    interaction_dict: Dict[str, pd.Series] = {}
    interaction_cols: List[str] = []

    for i in range(len(sensor_cols)):
        for j in range(i + 1, len(sensor_cols)):
            col1, col2 = sensor_cols[i], sensor_cols[j]
            if "product" in interaction_types:
                new_col = f"{col1}_prod_{col2}"
                interaction_dict[new_col] = df[col1] * df[col2]
                interaction_cols.append(new_col)
            if "ratio" in interaction_types:
                new_col = f"{col1}_ratio_{col2}"
                interaction_dict[new_col] = df[col1] / (df[col2] + 1e-8)
                interaction_cols.append(new_col)

    if interaction_dict:
        df = pd.concat([df, pd.DataFrame(interaction_dict, index=df.index)], axis=1)
    return df, interaction_cols

def run_modeling_validation(
    df_summary: pd.DataFrame,
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str,
    unit_id_col: str,
    config: ValidationConfig,
) -> Tuple[str, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Dict[str, List[str]]]:
    """执行 run modeling validation 对应的项目处理逻辑。"""
    report: List[str] = ["\n" + "=" * 80, "建模有效性验证", "=" * 80]

    unit_sample_count = train_df.groupby(unit_id_col).size()
    valid_units = unit_sample_count[unit_sample_count >= config.min_valid_unit_samples].index.tolist()
    if len(valid_units) < 2:
        report.append("有效设备数量不足，跳过建模验证。")
        return "\n".join(report), pd.DataFrame(), pd.DataFrame(), None, {}

    train_units, test_units = train_test_split(
        valid_units,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_data = train_df[train_df[unit_id_col].isin(train_units)].copy()
    test_data = train_df[train_df[unit_id_col].isin(test_units)].copy()

    base_feature_sets: Dict[str, List[str]] = {
        "全特征集": list(sensor_cols),
        "筛选特征集": df_summary[df_summary["最终筛选建议"].str.contains("保留", na=False)]["传感器列"].tolist(),
        "推荐保留": df_summary[df_summary["最终筛选建议"] == "推荐保留"]["传感器列"].tolist(),
    }
    base_feature_sets = {k: v for k, v in base_feature_sets.items() if len(v) > 0}

    global_model = XGBRegressor(random_state=config.model_random_state)
    global_model.fit(train_data[sensor_cols], train_data[target_col])
    global_pred = global_model.predict(test_data[sensor_cols])
    global_rmse = np.sqrt(mean_squared_error(test_data[target_col], global_pred))
    report.append(f"全局基准模型 RMSE: {global_rmse:.2f}")

    interaction_feature_sets: Dict[str, List[str]] = {}
    group_feat_map: Dict[str, Tuple[List[str], List[str]]] = {}

    for name, features in base_feature_sets.items():
        if len(features) < 2:
            continue

        train_data_inter, all_inter_cols = generate_interaction_features(
            train_data, features, config.interaction_types
        )
        all_candidate_feats = features + all_inter_cols
        temp_model = XGBRegressor(random_state=config.model_random_state)
        temp_model.fit(train_data_inter[all_candidate_feats], train_data_inter[target_col])

        gain_df = pd.DataFrame({
            "特征": all_candidate_feats,
            "Gain重要度": temp_model.feature_importances_,
        })
        inter_gain_df = gain_df[gain_df["特征"].isin(all_inter_cols)].copy()
        keep_num = max(int(len(inter_gain_df) * config.inter_keep_ratio), config.min_keep_inter)
        keep_num = min(keep_num, len(inter_gain_df))
        top_inter_features = inter_gain_df.sort_values("Gain重要度", ascending=False).head(keep_num)["特征"].tolist()

        group_name = f"{name}_筛选后交互"
        interaction_feature_sets[group_name] = features + top_inter_features
        group_feat_map[group_name] = (features, top_inter_features)
        report.append(f"\n{name} 交互特征筛选: 生成 {len(all_inter_cols)} 个，保留 {len(top_inter_features)} 个")

    if group_feat_map:
        add_train = pd.DataFrame(index=train_data.index)
        add_test = pd.DataFrame(index=test_data.index)

        def gen_target_inter(df: pd.DataFrame, raw_feats: List[str], need_inter_list: List[str]) -> pd.DataFrame:
            """执行 gen target inter 对应的项目处理逻辑。"""
            df_new, _ = generate_interaction_features(df.copy(), raw_feats, config.interaction_types)
            exist = [c for c in need_inter_list if c in df_new.columns]
            return df_new[exist]

        for _, (raw_feats, inter_list) in group_feat_map.items():
            add_train = pd.concat([add_train, gen_target_inter(train_data, raw_feats, inter_list)], axis=1)
            add_test = pd.concat([add_test, gen_target_inter(test_data, raw_feats, inter_list)], axis=1)

        add_train = add_train.loc[:, ~add_train.columns.duplicated(keep="first")]
        add_test = add_test.loc[:, ~add_test.columns.duplicated(keep="first")]
        train_data = pd.concat([train_data.reset_index(drop=True), add_train.reset_index(drop=True)], axis=1)
        test_data = pd.concat([test_data.reset_index(drop=True), add_test.reset_index(drop=True)], axis=1)

    all_feature_sets = {**base_feature_sets, **interaction_feature_sets}
    results: Dict[str, Dict[str, Any]] = {}
    phase_rmse_rows: List[Dict[str, Any]] = []

    model_iterator = tqdm(all_feature_sets.items(), desc="训练模型", disable=not config.show_progress)
    for name, features in model_iterator:
        check_cancelled("特征验证模型训练")
        X_train = train_data[features]
        y_train = train_data[target_col]
        X_test = test_data[features]
        y_test = test_data[target_col]

        base_model = XGBRegressor(random_state=config.model_random_state)
        base_model.fit(X_train, y_train)
        y_pred_base = base_model.predict(X_test)

        results[f"{name}_全局模型"] = {
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred_base))), 2),
            "MAE": round(float(mean_absolute_error(y_test, y_pred_base)), 2),
            "R²": round(float(r2_score(y_test, y_pred_base)), 4),
            "特征数": len(features),
            "模型类型": "全局单阶段",
        }

        train_phased = split_lifecycle_phases(train_data, target_col, config.phase_ratios)
        test_phased = split_lifecycle_phases(test_data, target_col, config.phase_ratios)
        phase_models: Dict[str, Optional[XGBRegressor]] = {}
        phase_rmse: Dict[str, float] = {}

        for phase in ["early", "mid", "late"]:
            phase_train = train_phased[train_phased["lifecycle_phase"] == phase]
            if len(phase_train) < 10:
                phase_models[phase] = None
            else:
                model = XGBRegressor(random_state=config.model_random_state)
                model.fit(phase_train[features], phase_train[target_col])
                phase_models[phase] = model

        test_pred_phase = pd.Series(index=test_data.index, dtype=np.float64)
        for phase in ["early", "mid", "late"]:
            phase_test = test_phased[test_phased["lifecycle_phase"] == phase]
            if len(phase_test) == 0:
                phase_rmse[phase] = np.nan
                continue
            model = phase_models[phase]
            if model is not None:
                phase_pred = model.predict(phase_test[features])
            else:
                raise RuntimeError(f"Lifecycle phase {phase} has no fitted model; strict analysis cannot substitute a global model.")
            phase_rmse[phase] = float(np.sqrt(mean_squared_error(phase_test[target_col], phase_pred)))
            test_pred_phase.loc[phase_test.index] = phase_pred

        valid_pred = ~test_pred_phase.isna()
        y_true = y_test.loc[valid_pred]
        y_pred = test_pred_phase.loc[valid_pred]
        phase_total_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        phase_total_mae = float(mean_absolute_error(y_true, y_pred))
        phase_total_r2 = float(r2_score(y_true, y_pred))

        results[f"{name}_分阶段模型"] = {
            "RMSE": round(phase_total_rmse, 2),
            "MAE": round(phase_total_mae, 2),
            "R²": round(phase_total_r2, 4),
            "特征数": len(features),
            "模型类型": "分阶段融合",
        }

        phase_rmse_rows.append({
            "特征集": name,
            "早期RMSE": round(phase_rmse.get("early", np.nan), 2),
            "中期RMSE": round(phase_rmse.get("mid", np.nan), 2),
            "晚期RMSE": round(phase_rmse.get("late", np.nan), 2),
            "整体RMSE": round(phase_total_rmse, 2),
        })

    phase_rmse_df = pd.DataFrame(phase_rmse_rows)
    save_table(phase_rmse_df, "phase_rmse_comparison", config)
    report.append("\n各阶段子模型 RMSE 对比:")
    report.append(phase_rmse_df.to_string(index=False))

    model_df = pd.DataFrame(results).T.sort_values("RMSE") if results else pd.DataFrame()
    save_table(model_df, "model_performance_comparison_phase_inter", config)
    report.append("\n不同特征集 + 模型类型整体性能对比:")
    report.append(model_df.to_string())

    shap_df: Optional[pd.DataFrame] = None
    shap_feature_set_name = "筛选特征集_筛选后交互"
    if shap_feature_set_name in all_feature_sets:
        report.append("\n筛选特征集 SHAP 重要性:")
        features = all_feature_sets[shap_feature_set_name]
        model = XGBRegressor(random_state=config.model_random_state)
        model.fit(train_data[features], train_data[target_col])
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(test_data[features])
        shap_df = pd.DataFrame({
            "特征": features,
            "SHAP重要性": np.abs(shap_values).mean(axis=0),
        }).sort_values("SHAP重要性", ascending=False)
        shap_df["特征类型"] = shap_df["特征"].apply(
            lambda x: "交互特征" if ("_prod_" in x or "_ratio_" in x) else "单特征"
        )
        save_table(shap_df, "shap_interaction_importance", config)
        report.append(shap_df.to_string(index=False))
    else:
        report.append("\n筛选特征集不足以生成交互 SHAP 分析，已跳过。")

    return "\n".join(report), model_df, phase_rmse_df, shap_df, all_feature_sets

# Feature profile与清晰最终结论
def classify_lifecycle_role(row: pd.Series, config: ValidationConfig) -> Tuple[str, str]:
    """执行 classify lifecycle role 对应的项目处理逻辑。"""
    early = float(row.get("早期显著率", 0.0))
    mid = float(row.get("中期显著率", 0.0))
    late = float(row.get("晚期显著率", 0.0))

    if (
        early >= config.full_cycle_phase_threshold and
        mid >= config.full_cycle_phase_threshold and
        late >= config.full_cycle_phase_threshold
    ):
        return "全周期核心特征", "full_cycle"

    stage_values = {"early": early, "mid": mid, "late": late}
    best_stage = max(stage_values, key=stage_values.get)
    best_value = stage_values[best_stage]

    if best_value < config.weak_corr_threshold:
        return "阶段不敏感特征", "none"
    if best_value >= config.phase_core_threshold:
        stage_name = {"early": "早期", "mid": "中期", "late": "晚期"}[best_stage]
        return f"{stage_name}退化敏感特征", best_stage
    return "局部阶段候选特征", best_stage

def classify_shap_level(shap_value: Optional[float], single_shap_values: pd.Series, config: ValidationConfig) -> str:
    """执行 classify shap level 对应的项目处理逻辑。"""
    if shap_value is None or pd.isna(shap_value) or len(single_shap_values.dropna()) == 0:
        return "未验证"
    q_high = single_shap_values.quantile(config.shap_high_quantile)
    q_mid = single_shap_values.quantile(config.shap_mid_quantile)
    if shap_value >= q_high:
        return "高"
    if shap_value >= q_mid:
        return "中"
    return "低"

def build_feature_profile(
    df_summary: pd.DataFrame,
    stability_df: Optional[pd.DataFrame] = None,
    direction_df: Optional[pd.DataFrame] = None,
    shap_df: Optional[pd.DataFrame] = None,
    outlier_compare_df: Optional[pd.DataFrame] = None,
    config: Optional[ValidationConfig] = None,
) -> pd.DataFrame:
    """执行 build feature profile 对应的项目处理逻辑。"""
    if config is None:
        config = ValidationConfig()

    profile = df_summary.copy()

    if stability_df is not None and not stability_df.empty:
        profile = profile.merge(stability_df, on="传感器列", how="left")
    else:
        profile["平均显著率"] = np.nan
        profile["显著率标准差"] = np.nan
        profile["稳定性等级"] = "未验证"

    if direction_df is not None and not direction_df.empty:
        profile = profile.merge(direction_df, on="传感器列", how="left")
    else:
        profile["正相关比例"] = np.nan
        profile["方向一致性"] = np.nan
        profile["主导方向"] = "未验证"

    single_shap_df = None
    if shap_df is not None and not shap_df.empty:
        single_shap_df = shap_df[shap_df["特征类型"] == "单特征"][["特征", "SHAP重要性"]].rename(
            columns={"特征": "传感器列"}
        )
        profile = profile.merge(single_shap_df, on="传感器列", how="left")
    else:
        profile["SHAP重要性"] = np.nan

    if outlier_compare_df is not None and not outlier_compare_df.empty:
        profile = profile.merge(
            outlier_compare_df[["传感器列", "显著率变化"]],
            on="传感器列",
            how="left",
        )
    else:
        profile["显著率变化"] = np.nan

    single_shap_values = profile["SHAP重要性"].dropna()

    roles: List[str] = []
    stages: List[str] = []
    shap_levels: List[str] = []
    final_levels: List[str] = []
    vi_suggestions: List[str] = []
    keep_reasons: List[str] = []
    risk_notes: List[str] = []

    for _, row in profile.iterrows():
        role, stage = classify_lifecycle_role(row, config)
        roles.append(role)
        stages.append(stage)

        shap_level = classify_shap_level(row.get("SHAP重要性", np.nan), single_shap_values, config)
        shap_levels.append(shap_level)

        sig_rate = float(row.get("跨设备显著率", 0.0))
        avg_corr = float(row.get("平均最大相关系数", 0.0))
        stability = row.get("稳定性等级", "未验证")
        direction_cons = row.get("方向一致性", np.nan)
        direction_ok = pd.isna(direction_cons) or direction_cons >= config.direction_consistency_threshold
        suggestion = row.get("最终筛选建议", "剔除")
        dominant_type = row.get("最常见主导类型", "无")

        risks: List[str] = []
        reasons: List[str] = []

        if sig_rate >= 0.70:
            reasons.append("跨设备显著率高")
        elif sig_rate >= 0.50:
            reasons.append("跨设备显著率中等")
        else:
            risks.append("跨设备显著率偏低")

        if avg_corr >= config.medium_corr_threshold:
            reasons.append("相关效应达到中等及以上")
        elif avg_corr >= config.weak_corr_threshold:
            reasons.append("存在弱相关效应")
        else:
            risks.append("相关效应较弱")

        if stability == "高稳定":
            reasons.append("跨折验证稳定")
        elif stability == "不稳定":
            risks.append("跨折验证不稳定")

        if not pd.isna(direction_cons):
            if direction_cons >= config.direction_consistency_threshold:
                reasons.append("方向一致性高")
            elif direction_cons < config.low_direction_threshold:
                risks.append("方向一致性低，物理解释风险较高")
            else:
                risks.append("方向一致性一般")

        if shap_level == "高":
            reasons.append("建模贡献高")
        elif shap_level == "中":
            reasons.append("建模贡献中等")
        elif shap_level == "低":
            risks.append("建模贡献低")

        if "非线性" in str(dominant_type):
            reasons.append("存在非线性退化信息")

        if role in ["全周期核心特征", "早期退化敏感特征", "中期退化敏感特征", "晚期退化敏感特征"]:
            reasons.append(role)

        outlier_change = row.get("显著率变化", np.nan)
        if not pd.isna(outlier_change) and outlier_change > 0.10:
            risks.append("对异常值处理较敏感")

        if suggestion == "剔除" or (sig_rate < 0.30 and avg_corr < config.weak_corr_threshold):
            final_level = "剔除"
            vi = "不输入"
        elif avg_corr >= config.medium_corr_threshold and sig_rate >= 0.70 and direction_ok and stability in ["高稳定", "稳定", "未验证"]:
            final_level = "核心"
            vi = "强先验"
        elif sig_rate >= 0.50 and avg_corr >= config.weak_corr_threshold and direction_ok:
            final_level = "推荐"
            vi = "普通先验"
        else:
            final_level = "候选"
            vi = "稀疏先验"

        # 对高价值非线性特征，即使方向一致性一般，也不直接删除，交给稀疏先验控制
        if final_level == "剔除" and "非线性" in str(dominant_type) and sig_rate >= 0.50:
            final_level = "候选"
            vi = "稀疏先验"
            reasons.append("非线性候选信息，建议由稀疏先验自动选择")

        final_levels.append(final_level)
        vi_suggestions.append(vi)
        keep_reasons.append("; ".join(reasons) if reasons else "未发现足够支持保留的证据")
        risk_notes.append("; ".join(risks) if risks else "低风险")

    profile["特征角色"] = roles
    profile["生命周期敏感阶段"] = stages
    profile["建模贡献等级"] = shap_levels
    profile["最终等级"] = final_levels
    profile["VI建模建议"] = vi_suggestions
    profile["保留原因"] = keep_reasons
    profile["风险提示"] = risk_notes

    preferred_order = [
        "传感器列", "最终等级", "特征角色", "VI建模建议",
        "最常见主导类型", "生命周期敏感阶段",
        "跨设备显著率", "平均最大相关系数", "相关系数标准差",
        "早期显著率", "中期显著率", "晚期显著率",
        "稳定性等级", "显著率标准差", "方向一致性", "主导方向",
        "SHAP重要性", "建模贡献等级", "显著率变化",
        "最终筛选建议", "保留原因", "风险提示",
    ]
    existing_order = [c for c in preferred_order if c in profile.columns]
    remaining = [c for c in profile.columns if c not in existing_order]
    profile = profile[existing_order + remaining]

    rank_map = {"核心": 0, "推荐": 1, "候选": 2, "剔除": 3}
    profile["_rank"] = profile["最终等级"].map(rank_map).fillna(9)
    profile = profile.sort_values(
        by=["_rank", "跨设备显著率", "平均最大相关系数"],
        ascending=[True, False, False],
    ).drop(columns=["_rank"]).reset_index(drop=True)
    return profile

def generate_metric_usage_report() -> str:
    """执行 generate metric usage report 对应的项目处理逻辑。"""
    lines = ["\n" + "=" * 80, "Metric用途说明", "=" * 80]
    for metric, usage in METRIC_USAGE.items():
        lines.append(f"- {metric}: {usage}")
    return "\n".join(lines)

def generate_clear_final_report(
    feature_profile_df: pd.DataFrame,
    model_df: Optional[pd.DataFrame] = None,
    phase_rmse_df: Optional[pd.DataFrame] = None,
    shap_df: Optional[pd.DataFrame] = None,
    config: Optional[ValidationConfig] = None,
) -> str:
    """执行 generate clear final report 对应的项目处理逻辑。"""
    if config is None:
        config = ValidationConfig()

    report: List[str] = ["\n" + "=" * 80, "最终特征筛选结论", "=" * 80]

    for level in ["核心", "推荐", "候选", "剔除"]:
        cols = feature_profile_df[feature_profile_df["最终等级"] == level]["传感器列"].tolist()
        report.append(f"\n{level}特征 ({len(cols)}): {cols}")

    report.append("\n一、核心Feature profile")
    core_cols = ["传感器列", "特征角色", "最常见主导类型", "VI建模建议", "保留原因", "风险提示"]
    core_profile = feature_profile_df[feature_profile_df["最终等级"].isin(["核心", "推荐"])]
    if len(core_profile) > 0:
        report.append(core_profile[core_cols].to_string(index=False))
    else:
        report.append("暂无核心或推荐特征。")

    report.append("\n二、生命周期阶段特征")
    for stage_key, stage_name in [("full_cycle", "全周期"), ("early", "早期"), ("mid", "中期"), ("late", "晚期")]:
        cols = feature_profile_df[
            (feature_profile_df["生命周期敏感阶段"] == stage_key) &
            (feature_profile_df["最终等级"] != "剔除")
        ]["传感器列"].tolist()
        report.append(f"   {stage_name}敏感特征: {cols}")

    report.append("\n三、变分推断输入建议")
    strong_prior = feature_profile_df[feature_profile_df["VI建模建议"] == "强先验"]["传感器列"].tolist()
    normal_prior = feature_profile_df[feature_profile_df["VI建模建议"] == "普通先验"]["传感器列"].tolist()
    sparse_prior = feature_profile_df[feature_profile_df["VI建模建议"] == "稀疏先验"]["传感器列"].tolist()
    excluded = feature_profile_df[feature_profile_df["VI建模建议"] == "不输入"]["传感器列"].tolist()
    report.append(f"   强先验特征: {strong_prior}")
    report.append(f"   普通先验特征: {normal_prior}")
    report.append(f"   稀疏先验候选特征: {sparse_prior}")
    report.append(f"   不建议输入特征: {excluded}")

    if shap_df is not None and not shap_df.empty:
        inter_features = shap_df[shap_df["特征类型"] == "交互特征"].copy()
        if len(inter_features) > 0:
            total_shap = shap_df["SHAP重要性"].sum()
            inter_ratio = inter_features["SHAP重要性"].sum() / total_shap if total_shap > 0 else 0.0
            top_inter = inter_features.head(5)["特征"].tolist()
            report.append("\n四、交互特征建议")
            report.append(f"   交互特征总 SHAP 贡献占比: {inter_ratio:.2%}")
            report.append(f"   Top5 高贡献交互特征: {top_inter}")
            report.append("   建议后续 VI 模型优先加入 Top3-5 个交互特征，避免维度膨胀。")

    if model_df is not None and not model_df.empty:
        report.append("\n五、建模验证摘要")
        best_name = model_df.index[0]
        best_row = model_df.iloc[0]
        report.append(f"   最优模型/特征集: {best_name}")
        report.append(f"   RMSE={best_row['RMSE']}, MAE={best_row['MAE']}, R²={best_row['R²']}, 特征数={best_row['特征数']}")

    if phase_rmse_df is not None and not phase_rmse_df.empty:
        report.append("\n六、阶段 RMSE 摘要")
        report.append(phase_rmse_df.to_string(index=False))

    report.append(generate_metric_usage_report())
    return "\n".join(report)

# 工况去混杂、质量验证与 VI 去混杂特征方案
def get_operating_columns(train_df: pd.DataFrame, config: ValidationConfig) -> Tuple[List[str], Optional[str]]:
    """执行 get operating columns 对应的项目处理逻辑。"""
    op_cols = config.op_cols if config.op_cols is not None else [c for c in train_df.columns if c.startswith("op_")]
    op_cols = [c for c in op_cols if c in train_df.columns]
    condition_col = config.condition_col if config.condition_col in train_df.columns else None
    return op_cols, condition_col

def create_condition_design_matrix(
    df: pd.DataFrame,
    op_cols: List[str],
    condition_col: Optional[str],
    config: ValidationConfig,
    fit_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """构造工况设计矩阵；禁止使用 cycle/unit_id/RUL。"""
    parts: List[pd.DataFrame] = []
    if op_cols:
        op_df = df[op_cols].astype(float).copy()
        parts.append(op_df)
        if config.condition_model_degree >= 2:
            poly = PolynomialFeatures(degree=config.condition_model_degree, include_bias=False)
            arr = poly.fit_transform(op_df.values)
            names = poly.get_feature_names_out(op_cols)
            poly_df = pd.DataFrame(arr, columns=[f"poly_{n}" for n in names], index=df.index)
            poly_df = poly_df[[c for c in poly_df.columns if c.replace("poly_", "") not in op_cols]]
            if not poly_df.empty:
                parts.append(poly_df)
    if condition_col is not None:
        parts.append(pd.get_dummies(df[condition_col].astype("category"), prefix=condition_col, dtype=float))
    if not parts:
        raise ValueError("未找到 op_* 或 condition，无法执行工况去混杂。")
    X = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if fit_columns is not None:
        X = X.reindex(columns=fit_columns, fill_value=0.0)
        return X, fit_columns
    return X, X.columns.tolist()

def safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """执行 safe r2 score 对应的项目处理逻辑。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3 or is_constant_arr(y_true):
        return np.nan
    return float(r2_score(y_true, y_pred))

def make_group_oof_splits(df: pd.DataFrame, unit_id_col: str, config: ValidationConfig) -> List[Tuple[np.ndarray, np.ndarray]]:
    """执行 make group oof splits 对应的项目处理逻辑。"""
    groups = df[unit_id_col].values
    n_units = len(pd.Series(groups).dropna().unique())
    if n_units < 3:
        idx = np.arange(len(df))
        tr_idx, te_idx = train_test_split(idx, test_size=config.test_size, random_state=config.random_state)
        return [(tr_idx, te_idx)]
    n_splits = min(config.deconfounding_n_splits, n_units)
    return list(GroupKFold(n_splits=n_splits).split(df, groups=groups))

def classify_deconfounding_status(
    raw_condition_r2: float,
    residual_condition_r2: float,
    reduction: float,
    raw_rul_corr: float,
    resid_rul_corr: float,
    retention: float,
    config: ValidationConfig,
) -> Tuple[str, str]:
    """执行 classify deconfounding status 对应的项目处理逻辑。"""
    if pd.isna(raw_condition_r2):
        return "无法评估", "原始传感器方差过低或样本不足，无法可靠评估工况解释率"
    resid_r2_high = (not pd.isna(residual_condition_r2)) and residual_condition_r2 > config.residual_condition_r2_warn
    resid_r2_good = pd.isna(residual_condition_r2) or residual_condition_r2 <= config.residual_condition_r2_good
    reduction_good = (not pd.isna(reduction)) and reduction >= config.condition_reduction_good
    retention_low = np.isfinite(retention) and retention < config.degradation_retention_min and raw_rul_corr >= config.weak_corr_threshold
    retention_ok = (not np.isfinite(retention)) or retention >= config.degradation_retention_good
    if resid_r2_high or ((not pd.isna(reduction)) and reduction < 0.20 and raw_condition_r2 > 0.20):
        return "去混杂不足", "残差仍能被工况变量明显解释, 说明工况扰动尚未充分剔除"
    if retention_low and resid_r2_good:
        return "疑似过度去混杂", "工况影响已明显下降, 但 RUL 相关性大幅衰减, 可能把退化信号一起剔除了"
    if reduction_good and retention_ok and resid_rul_corr >= config.weak_corr_threshold:
        return "去混杂有效", "工况影响明显削弱, 且残差仍保留可用的 RUL 退化相关信息"
    if raw_rul_corr < config.weak_corr_threshold and resid_rul_corr >= config.weak_corr_threshold:
        return "揭示被掩盖退化", "原始相关性较弱, 但残差与 RUL 的相关性增强, 说明工况噪声可能掩盖了退化信号"
    if raw_rul_corr >= config.weak_corr_threshold and resid_rul_corr < config.weak_corr_threshold:
        return "工况驱动嫌疑", "原始相关性较强, 但去工况后明显减弱, 原始信号可能主要来自工况差异"
    return "低价值或待观察", "未发现足够证据证明残差特征具有稳定退化价值"

def generate_condition_residual_features(
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    unit_id_col: str,
    target_col: str,
    config: ValidationConfig,
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """按设备分组 OOF 生成 sensor_resid, 并输出去混杂质量Metric"""
    op_cols, condition_col = get_operating_columns(train_df, config)
    resid_df = train_df.copy()
    X_full, _ = create_condition_design_matrix(resid_df, op_cols, condition_col, config)
    splits = make_group_oof_splits(resid_df, unit_id_col, config)
    resid_cols: List[str] = []
    rows: List[Dict[str, Any]] = []
    corr_cfg = InspectConfig(enable_distance_corr=False, remove_outliers=False, random_state=config.random_state, show_progress=False, output_dir=config.output_dir)
    iterator = tqdm(sensor_cols, desc="工况去混杂", disable=not config.show_progress) if len(sensor_cols) > 3 else sensor_cols
    for sensor in iterator:
        check_cancelled("工况去混杂")
        y = resid_df[sensor].astype(float).values
        resid_col = f"{sensor}{config.deconfounding_resid_suffix}"
        resid_cols.append(resid_col)
        if is_constant_arr(y):
            resid_df[resid_col] = 0.0
            rows.append({"传感器列": sensor, "残差特征列": resid_col, "去混杂状态": "常数列", "去混杂说明": "原始传感器为常数或近似常数。"})
            continue
        oof_pred = np.full(len(resid_df), np.nan, dtype=float)
        for tr_idx, te_idx in splits:
            model = Ridge(alpha=config.condition_model_alpha)
            model.fit(X_full.iloc[tr_idx], y[tr_idx])
            oof_pred[te_idx] = model.predict(X_full.iloc[te_idx])
        if np.isnan(oof_pred).any():
            model = Ridge(alpha=config.condition_model_alpha).fit(X_full, y)
            oof_pred[np.isnan(oof_pred)] = model.predict(X_full.iloc[np.isnan(oof_pred)])
        residual = y - oof_pred
        resid_df[resid_col] = residual
        raw_condition_r2 = safe_r2_score(y, oof_pred)
        resid_oof_pred = np.full(len(resid_df), np.nan, dtype=float)
        for tr_idx, te_idx in splits:
            model = Ridge(alpha=config.condition_model_alpha)
            model.fit(X_full.iloc[tr_idx], residual[tr_idx])
            resid_oof_pred[te_idx] = model.predict(X_full.iloc[te_idx])
        if np.isnan(resid_oof_pred).any():
            model = Ridge(alpha=config.condition_model_alpha).fit(X_full, residual)
            resid_oof_pred[np.isnan(resid_oof_pred)] = model.predict(X_full.iloc[np.isnan(resid_oof_pred)])
        residual_condition_r2 = safe_r2_score(residual, resid_oof_pred)
        raw_corr = float(compute_corr_metrics(resid_df[sensor].values, resid_df[target_col].values, corr_cfg).get("最大相关系数", 0.0) or 0.0)
        resid_corr = float(compute_corr_metrics(resid_df[resid_col].values, resid_df[target_col].values, corr_cfg).get("最大相关系数", 0.0) or 0.0)
        reduction = np.nan if pd.isna(raw_condition_r2) or raw_condition_r2 <= 1e-12 else 1 - max(residual_condition_r2 if not pd.isna(residual_condition_r2) else 0.0, 0.0) / max(raw_condition_r2, 1e-12)
        retention = (np.inf if resid_corr > config.weak_corr_threshold else 0.0) if raw_corr <= 1e-12 else resid_corr / raw_corr
        status, note = classify_deconfounding_status(raw_condition_r2, residual_condition_r2, reduction, raw_corr, resid_corr, retention, config)
        rows.append({
            "传感器列": sensor,
            "残差特征列": resid_col,
            "原始工况解释R2": round(raw_condition_r2, 4) if not pd.isna(raw_condition_r2) else np.nan,
            "残差工况解释R2": round(residual_condition_r2, 4) if not pd.isna(residual_condition_r2) else np.nan,
            "工况影响削弱率": round(float(reduction), 4) if not pd.isna(reduction) else np.nan,
            "原始RUL最大相关": round(raw_corr, 4),
            "残差RUL最大相关": round(resid_corr, 4),
            "退化信号保留率": round(float(retention), 4) if np.isfinite(retention) else np.inf,
            "去混杂状态": status,
            "去混杂说明": note,
        })
    return resid_df, resid_cols, pd.DataFrame(rows)

def classify_lifecycle_stage(row: pd.Series, config: ValidationConfig) -> str:
    """为每个设备的全生命周期数据划分 early/mid/late 阶段"""
    early = float(row.get("早期显著率", 0.0))
    mid = float(row.get("中期显著率", 0.0))
    late = float(row.get("晚期显著率", 0.0))

    # 全周期判定: 三阶段均达到全周期阈值
    if (
        early >= config.full_cycle_phase_threshold and
        mid >= config.full_cycle_phase_threshold and
        late >= config.full_cycle_phase_threshold
    ):
        return "full_cycle"

    # 取最大阶段及对应数值
    stage_values = {"early": early, "mid": mid, "late": late}
    best_stage = max(stage_values, key=stage_values.get)
    best_value = stage_values[best_stage]

    # 低于弱相关阈值则判定为无有效阶段
    if best_value < config.weak_corr_threshold:
        return "none"

    # 返回最优单阶段
    return best_stage

def build_residual_feature_profile(resid_summary: pd.DataFrame, quality_df: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    """执行 build residual feature profile 对应的项目处理逻辑。"""
    profile = resid_summary.merge(quality_df, left_on="传感器列", right_on="残差特征列", how="left", suffixes=("", "_原始"))
    levels, roles, priors, reasons, risks = [], [], [], [], []
    for _, row in profile.iterrows():
        status = str(row.get("去混杂状态", "未知"))
        sig = float(row.get("跨设备显著率", 0.0) or 0.0)
        corr = float(row.get("平均最大相关系数", 0.0) or 0.0)
        stage = classify_lifecycle_stage(row, config)
        if status in ["去混杂有效", "揭示被掩盖退化"] and sig >= 0.50 and corr >= config.weak_corr_threshold:
            level = "核心" if sig >= 0.70 and corr >= config.medium_corr_threshold else "推荐"
            prior = "强先验" if level == "核心" else "普通先验"
            role = f"去工况退化特征-{stage}" if stage != "none" else "去工况退化特征"
            reason = f"{status}；残差跨设备显著率 {sig:.2f}，平均相关 {corr:.2f}"
            risk = "低风险" if status == "去混杂有效" else "建议结合下游模型复核"
        elif status in ["去混杂不足", "疑似过度去混杂", "工况驱动嫌疑"]:
            level = "候选" if corr >= config.weak_corr_threshold else "剔除"
            prior = "稀疏先验" if level == "候选" else "不输入"
            role = status
            reason = str(row.get("去混杂说明", "去混杂质量存在风险"))
            risk = status
        else:
            level = "候选" if sig >= 0.50 and corr >= config.weak_corr_threshold else "剔除"
            prior = "稀疏先验" if level == "候选" else "不输入"
            role = "去工况候选特征" if level == "候选" else "低价值残差特征"
            reason = str(row.get("去混杂说明", "缺少足够证据"))
            risk = "需要后续 VI 稀疏先验自动裁剪" if level == "候选" else "退化证据不足"
        levels.append(level); roles.append(role); priors.append(prior); reasons.append(reason); risks.append(risk)
    profile["最终等级"] = levels
    profile["特征角色"] = roles
    profile["VI建模建议"] = priors
    profile["保留原因"] = reasons
    profile["风险提示"] = risks
    order = ["传感器列", "传感器列_原始", "最终等级", "特征角色", "VI建模建议", "去混杂状态", "去混杂说明", "原始工况解释R2", "残差工况解释R2", "工况影响削弱率", "原始RUL最大相关", "残差RUL最大相关", "退化信号保留率", "跨设备显著率", "平均最大相关系数", "早期显著率", "中期显著率", "晚期显著率", "最常见主导类型", "最终筛选建议", "保留原因", "风险提示"]
    cols = [c for c in order if c in profile.columns]
    rank = {"核心":0, "推荐":1, "候选":2, "剔除":3}
    profile = profile[cols + [c for c in profile.columns if c not in cols]].copy()
    profile["_rank"] = profile["最终等级"].map(rank).fillna(9)
    return profile.sort_values(["_rank", "跨设备显著率", "平均最大相关系数"], ascending=[True, False, False]).drop(columns="_rank").reset_index(drop=True)

def compare_raw_and_residual_profiles(raw_profile: pd.DataFrame, residual_profile: pd.DataFrame, quality_df: pd.DataFrame, config: ValidationConfig) -> pd.DataFrame:
    """执行 compare raw and residual profiles 对应的项目处理逻辑。"""
    raw = raw_profile.rename(columns={"传感器列":"原始特征列", "最终等级":"原始最终等级", "特征角色":"原始特征角色", "VI建模建议":"原始VI建议", "跨设备显著率":"原始跨设备显著率", "平均最大相关系数":"原始平均最大相关系数"})
    resid = residual_profile.rename(columns={"最终等级":"残差最终等级", "特征角色":"残差特征角色", "VI建模建议":"残差VI建议", "跨设备显著率":"残差跨设备显著率", "平均最大相关系数":"残差平均最大相关系数"})
    resid["原始特征列"] = resid["残差特征列"].str.replace(config.deconfounding_resid_suffix + "$", "", regex=True)
    q = quality_df.rename(columns={"传感器列":"原始特征列"})
    merged = raw.merge(resid, on="原始特征列", how="outer").merge(q, on="原始特征列", how="left", suffixes=("", "_quality"))
    changes, explanations, sources, features, priors = [], [], [], [], []
    for _, row in merged.iterrows():
        raw_level = str(row.get("原始最终等级", "剔除")); resid_level = str(row.get("残差最终等级", "剔除")); status = str(row.get("去混杂状态", "未知"))
        raw_col = str(row.get("原始特征列", "")); resid_col = str(row.get("残差特征列", f"{raw_col}{config.deconfounding_resid_suffix}"))
        raw_keep = raw_level in ["核心", "推荐"]; resid_keep = resid_level in ["核心", "推荐"]
        if raw_keep and resid_keep and status in ["去混杂有效", "揭示被掩盖退化"]:
            change, exp, src, feat, prior = "稳健退化特征", "原始特征和残差均有价值，优先使用残差通道。", "残差优先", resid_col, row.get("残差VI建议", "普通先验")
        elif raw_keep and not resid_keep:
            change = status if status in ["工况驱动嫌疑", "疑似过度去混杂"] else "原始优于残差"
            exp, src, feat, prior = "原始相关性强但残差证据不足，建议作为稀疏先验候选。", "原始候选", raw_col, "稀疏先验"
        elif (not raw_keep) and resid_keep:
            change, exp, src, feat, prior = "被工况掩盖特征", "原始不突出但去工况后增强，建议进入残差通道。", "残差新增", resid_col, row.get("残差VI建议", "普通先验")
        elif raw_level == "候选" or resid_level == "候选":
            change, exp, src, feat, prior = "候选待裁剪", "仅有弱证据，建议由 VI 稀疏机制裁剪。", "候选", resid_col if resid_level == "候选" else raw_col, "稀疏先验"
        else:
            change, exp, src, feat, prior = "低价值特征", "原始与残差均缺少退化证据，不建议输入。", "剔除", raw_col, "不输入"
        changes.append(change); explanations.append(exp); sources.append(src); features.append(feat); priors.append(prior)
    merged["变化类型"] = changes
    merged["解释"] = explanations
    merged["最终输入来源"] = sources
    merged["最终推荐特征"] = features
    merged["最终VI先验"] = priors
    order = ["原始特征列", "残差特征列", "变化类型", "最终输入来源", "最终推荐特征", "最终VI先验", "解释", "去混杂状态", "原始工况解释R2", "残差工况解释R2", "工况影响削弱率", "退化信号保留率", "原始最终等级", "残差最终等级", "原始跨设备显著率", "残差跨设备显著率", "原始平均最大相关系数", "残差平均最大相关系数", "原始特征角色", "残差特征角色"]
    return merged[[c for c in order if c in merged.columns] + [c for c in merged.columns if c not in order]]

def evaluate_deconfounded_downstream_model(raw_df: pd.DataFrame, resid_df: pd.DataFrame, raw_profile: pd.DataFrame, comparison_df: pd.DataFrame, target_col: str, unit_id_col: str, config: ValidationConfig) -> pd.DataFrame:
    """执行 evaluate deconfounded downstream model 对应的项目处理逻辑。"""
    units = raw_df.groupby(unit_id_col).size()
    valid_units = units[units >= config.min_valid_unit_samples].index.tolist()
    if len(valid_units) < 4:
        return pd.DataFrame()
    tr_units, te_units = train_test_split(valid_units, test_size=config.test_size, random_state=config.random_state)
    tr_mask, te_mask = raw_df[unit_id_col].isin(tr_units), raw_df[unit_id_col].isin(te_units)
    op_cols, condition_col = get_operating_columns(raw_df, config)
    condition_features = op_cols + ([condition_col] if condition_col is not None else [])
    raw_features = raw_profile[(raw_profile["VI建模建议"] != "不输入") & (raw_profile["最终等级"].isin(["核心", "推荐", "候选"]))]["传感器列"].tolist()
    deconf_features = comparison_df[comparison_df["最终VI先验"] != "不输入"]["最终推荐特征"].dropna().astype(str).unique().tolist()
    deconf_features = [f for f in deconf_features if f in resid_df.columns or f in raw_df.columns]
    deconf_features = list(dict.fromkeys(deconf_features + condition_features))
    sets = {}
    if raw_features: sets["原始筛选特征"] = (raw_df, raw_features)
    if deconf_features: sets["去工况残差+工况通道"] = (resid_df, deconf_features)
    rows = []
    for name, (df0, feats) in sets.items():
        feats = [f for f in feats if f in df0.columns]
        if not feats: continue
        Xtr, Xte = df0.loc[tr_mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0), df0.loc[te_mask, feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ytr, yte = df0.loc[tr_mask, target_col], df0.loc[te_mask, target_col]
        model = XGBRegressor(random_state=config.model_random_state)
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        rows.append({"特征方案": name, "特征数": len(feats), "特征列表": ",".join(feats), "RMSE": round(float(np.sqrt(mean_squared_error(yte, pred))), 4), "MAE": round(float(mean_absolute_error(yte, pred)), 4), "R2": round(float(r2_score(yte, pred)), 4)})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True) if rows else pd.DataFrame()

def build_vi_deconfounded_feature_plan(comparison_df: pd.DataFrame, op_cols: List[str], condition_col: Optional[str]) -> pd.DataFrame:
    """执行 build vi deconfounded feature plan 对应的项目处理逻辑。"""
    rows = []
    for _, row in comparison_df.iterrows():
        prior = row.get("最终VI先验", "不输入"); feat = row.get("最终推荐特征")
        if prior == "不输入" or pd.isna(feat): continue
        feat = str(feat)
        channel = "退化残差通道" if feat.endswith("_resid") else "辅助原始通道" if feat.startswith("sensor_") else "其他"
        source = "去工况残差" if channel == "退化残差通道" else "原始传感器" if channel == "辅助原始通道" else "派生特征"
        rows.append({"VI输入通道": channel, "特征名": feat, "来源": source, "建议先验": prior, "原因": row.get("解释", "")})
    for op in op_cols:
        rows.append({"VI输入通道":"工况条件通道", "特征名":op, "来源":"连续操作Params", "建议先验":"条件变量/不做稀疏裁剪", "原因":"用于让 VI 模型显式建模工况扰动。"})
    if condition_col is not None:
        rows.append({"VI输入通道":"工况条件通道", "特征名":condition_col, "来源":"离散工况标签", "建议先验":"Embedding/条件变量", "原因":"建议在深度模型中使用 condition embedding 或 one-hot 条件输入。"})
    return pd.DataFrame(rows)

def generate_deconfounding_report(quality_df: pd.DataFrame, residual_profile: pd.DataFrame, comparison_df: pd.DataFrame, vi_plan: pd.DataFrame, downstream_df: Optional[pd.DataFrame]) -> str:
    """执行 generate deconfounding report 对应的项目处理逻辑。"""
    report = ["\n" + "="*80, "工况去混杂筛选与验证报告", "="*80]
    report.append("\n一、去混杂质量概览")
    if quality_df.empty:
        report.append("未生成去混杂质量结果。")
        return "\n".join(report)
    for status, cnt in quality_df["去混杂状态"].value_counts().to_dict().items():
        report.append(f"   - {status}: {cnt} 个传感器")
    report.append("\n二、原始特征 vs 去工况残差特征变化")
    if not comparison_df.empty:
        for change, cnt in comparison_df["变化类型"].value_counts().to_dict().items():
            report.append(f"   - {change}: {cnt} 个")
        show = ["原始特征列", "残差特征列", "变化类型", "最终输入来源", "最终推荐特征", "最终VI先验", "解释"]
        report.append(comparison_df[[c for c in show if c in comparison_df.columns]].to_string(index=False))
    report.append("\n三、VI 去混杂输入方案")
    report.append(vi_plan.to_string(index=False) if not vi_plan.empty else "暂无建议输入特征。")
    if downstream_df is not None and not downstream_df.empty:
        report.append("\n四、轻量下游建模验证")
        report.append(downstream_df.to_string(index=False))
    report.append("\n五、判定原则")
    report.append("   1. 去混杂不足: 残差仍能被 op/condition 明显解释")
    report.append("   2. 疑似过度去混杂: 工况影响已低, 但 RUL 相关性大幅衰减")
    report.append("   3. 去混杂有效: 工况影响下降, 同时残差保留或增强 RUL 相关性")
    return "\n".join(report)

def run_deconfounded_feature_workflow(train_df: pd.DataFrame, sensor_cols: List[str], raw_df_summary: pd.DataFrame, raw_feature_profile_df: pd.DataFrame, target_col: str, unit_id_col: str, config: ValidationConfig) -> Dict[str, Any]:
    """执行 run deconfounded feature workflow 对应的项目处理逻辑。"""
    create_output_dir(config)
    op_cols, condition_col = get_operating_columns(train_df, config)
    resid_df, resid_sensor_cols, quality_df = generate_condition_residual_features(train_df, sensor_cols, unit_id_col, target_col, config)
    save_table(quality_df, "deconfounding_quality", config)
    resid_inspect = InspectConfig(output_dir=config.output_dir, save_tables=config.save_tables, save_plots=config.save_plots, min_valid_unit_samples=config.min_valid_unit_samples, phase_ratios=config.phase_ratios, weak_corr_threshold=config.weak_corr_threshold, medium_corr_threshold=config.medium_corr_threshold, strong_corr_threshold=config.strong_corr_threshold, random_state=config.random_state, show_progress=False)
    resid_all_units, resid_summary = analyze_sensor_rul_correlation_by_unit(resid_df, resid_sensor_cols, target_col, unit_id_col, resid_inspect)
    save_table(resid_all_units, "residual_all_units_results", config)
    save_table(resid_summary, "residual_all_units_summary", config)
    residual_profile = build_residual_feature_profile(resid_summary, quality_df, config)
    save_table(residual_profile, "residual_feature_profile", config)
    comparison_df = compare_raw_and_residual_profiles(raw_feature_profile_df, residual_profile, quality_df, config)
    save_table(comparison_df, "raw_vs_residual_feature_comparison", config)
    vi_plan = build_vi_deconfounded_feature_plan(comparison_df, op_cols, condition_col)
    save_table(vi_plan, "vi_deconfounded_feature_plan", config)
    downstream_df = evaluate_deconfounded_downstream_model(train_df, resid_df, raw_feature_profile_df, comparison_df, target_col, unit_id_col, config)
    save_table(downstream_df, "deconfounded_model_comparison", config)
    report = generate_deconfounding_report(quality_df, residual_profile, comparison_df, vi_plan, downstream_df)
    write_report(report, "deconfounding_report", config)
    return {"deconfounded_df": resid_df, "residual_sensor_cols": resid_sensor_cols, "deconfounding_quality": quality_df, "residual_all_units": resid_all_units, "residual_summary": resid_summary, "residual_feature_profile": residual_profile, "raw_vs_residual_comparison": comparison_df, "vi_deconfounded_plan": vi_plan, "deconfounded_model_comparison": downstream_df, "deconfounding_report": report}

# 自动化总入口
def automated_correlation_validation(
    df_all_units: pd.DataFrame,
    df_summary: pd.DataFrame,
    train_df: pd.DataFrame,
    sensor_cols: List[str],
    target_col: str = "RUL",
    unit_id_col: str = "unit_id",
    time_col: str = "cycle",
    config: Optional[ValidationConfig] = None,
) -> Dict[str, Any]:
    """执行 automated correlation validation 对应的项目处理逻辑。"""
    if config is None:
        config = ValidationConfig()
    create_output_dir(config)

    print("Running quick self-check...")
    report1, outlier_compare_df = run_quick_self_check(
        df_summary, train_df, sensor_cols, target_col, unit_id_col, config
    )
    write_report(report1, "self_check_report", config)

    print("Running statistical stability validation...")
    report2, stability_df, direction_df, fold_df = run_statistical_validation(
        df_all_units, train_df, sensor_cols, target_col, unit_id_col, config
    )
    write_report(report2, "statistical_report", config)

    print("Running visual validation...")
    report3 = run_visual_validation(
        df_all_units, df_summary, train_df, target_col, unit_id_col, time_col, config
    )
    write_report(report3, "visual_report", config)

    print("Running modeling validation...")
    report4, model_df, phase_rmse_df, shap_df, all_feature_sets = run_modeling_validation(
        df_summary, train_df, sensor_cols, target_col, unit_id_col, config
    )
    write_report(report4, "modeling_report", config)

    print("Building feature profile...")
    feature_profile_df = build_feature_profile(
        df_summary=df_summary,
        stability_df=stability_df,
        direction_df=direction_df,
        shap_df=shap_df,
        outlier_compare_df=outlier_compare_df,
        config=config,
    )
    save_table(feature_profile_df, "feature_profile", config)

    vi_plan_df = feature_profile_df[[
        "传感器列", "最终等级", "特征角色", "VI建模建议", "保留原因", "风险提示"
    ]].copy()
    save_table(vi_plan_df, "vi_feature_recommendation", config)

    print("Generating final conclusion...")
    final_conclusion = generate_clear_final_report(
        feature_profile_df=feature_profile_df,
        model_df=model_df,
        phase_rmse_df=phase_rmse_df,
        shap_df=shap_df,
        config=config,
    )

    deconfounding_results: Optional[Dict[str, Any]] = None
    deconfounding_report = ""
    if config.enable_deconfounding:
        print("Running operating-condition deconfounding validation...")
        deconfounding_results = run_deconfounded_feature_workflow(
            train_df=train_df,
            sensor_cols=sensor_cols,
            raw_df_summary=df_summary,
            raw_feature_profile_df=feature_profile_df,
            target_col=target_col,
            unit_id_col=unit_id_col,
            config=config,
        )
        deconfounding_report = deconfounding_results["deconfounding_report"]

    full_report = "\n".join([report1, report2, report3, report4, final_conclusion, deconfounding_report])
    write_report(final_conclusion, "clear_final_conclusion", config)
    write_report(full_report, "validation_report", config)

    print(f"Validation completed. Outputs saved to: {config.output_dir}")
    print(f"Full report: {os.path.join(config.output_dir, 'reports', 'validation_report.txt')}")
    print(f"Feature profile: {os.path.join(config.output_dir, 'tables', 'feature_profile.csv')}")

    return {
        "final_conclusion": final_conclusion,
        "feature_profile": feature_profile_df,
        "vi_plan": vi_plan_df,
        "stability": stability_df,
        "direction": direction_df,
        "fold_results": fold_df,
        "model_results": model_df,
        "phase_rmse": phase_rmse_df,
        "shap_importance": shap_df,
        "outlier_effect": outlier_compare_df,
        "feature_sets": all_feature_sets,
        "deconfounding": deconfounding_results,
    }

# 主流程
def main() -> Dict[str, Any]:
    """执行 main 对应的项目处理逻辑。"""
    loader = CMAPSSLoader("datastream/CMAPSSData")
    train_df, test_df, rul_df = loader.load_all("FD004")

    unit_id_col = "unit_id"
    sensor_cols = [col for col in train_df.columns if col.startswith("sensor_")]
    target_col = "RUL"
    time_col = "cycle"

    inspect_config = InspectConfig()
    validation_config = ValidationConfig(output_dir=inspect_config.output_dir)

    df_all_units, df_summary = run_sensor_rul_corr_analysis_report(
        train_df=train_df,
        sensor_cols=sensor_cols,
        target_col=target_col,
        unit_id_col=unit_id_col,
        config=inspect_config,
    )

    results = automated_correlation_validation(
        df_all_units=df_all_units,
        df_summary=df_summary,
        train_df=train_df,
        sensor_cols=sensor_cols,
        target_col=target_col,
        unit_id_col=unit_id_col,
        time_col=time_col,
        config=validation_config,
    )

    print("\n" + results["final_conclusion"])
    return results


if __name__ == "__main__":
    main()
