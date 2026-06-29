"""data 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import Dataset

CMAPSS_SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
CMAPSS_SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
CMAPSS_COLUMNS = ["unit_id", "cycle"] + CMAPSS_SETTING_COLS + CMAPSS_SENSOR_COLS


OP_COLS = [f"op_{i}" for i in range(1, 4)]

def normalize_cmapss_schema(df: pd.DataFrame, add_condition_from_ops: bool = False) -> pd.DataFrame:
    """Normalize schemas from either this project or the user's CMAPSSLoader.

    Supported input variants:
    - setting_1/2/3 + sensor_1..sensor_21 + rul
    - op_1/2/3 + sensor_1..sensor_21 + RUL + condition

    Output always contains setting_1/2/3 and lowercase rul when available.
    The original op_* columns are preserved, so the user's analysis scripts can still
    be used on the same dataframe if needed.
    """
    df = df.copy()
    rename = {}
    for i in range(1, 4):
        if f"setting_{i}" not in df.columns and f"op_{i}" in df.columns:
            rename[f"op_{i}"] = f"setting_{i}"
    if rename:
        # Keep op_* aliases as well as creating setting_* names.
        for old, new in rename.items():
            df[new] = df[old]
    if "RUL" in df.columns and "rul" not in df.columns:
        df["rul"] = df["RUL"]
    if "rul" in df.columns and "RUL" not in df.columns:
        df["RUL"] = df["rul"]
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype(int)
    if "unit_id" in df.columns:
        df["unit_id"] = df["unit_id"].astype(int)
    if add_condition_from_ops and "condition" not in df.columns and all(c in df.columns for c in CMAPSS_SETTING_COLS):
        centers = np.array([
            [42.0, 0.84, 100.0],
            [10.0, 0.25, 100.0],
            [25.0, 0.62, 60.0],
            [20.0, 0.70, 100.0],
            [35.0, 0.84, 100.0],
            [0.0, 0.00, 100.0],
        ], dtype=float)
        ops = df[CMAPSS_SETTING_COLS].to_numpy(dtype=float)
        distances = np.sqrt(np.sum((ops[:, None, :] - centers[None, :, :]) ** 2, axis=2))
        df["condition"] = np.argmin(distances, axis=1) + 1
    return df


@dataclass
class DataSpec:
    """封装 DataSpec 相关状态与行为。"""
    feature_columns: List[str]
    condition_columns: List[str]
    scaler_center_: List[float]
    scaler_scale_: List[float]
    max_horizon: int
    window_size: int


def read_cmapss_txt(path: str | Path) -> pd.DataFrame:
    """Read standard NASA C-MAPSS text files.

    The expected layout is unit, cycle, 3 operating settings, 21 sensors.
    Extra blank columns from repeated spaces are ignored.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    if df.shape[1] < len(CMAPSS_COLUMNS):
        raise ValueError(f"{path} has {df.shape[1]} columns; expected at least {len(CMAPSS_COLUMNS)}")
    df = df.iloc[:, : len(CMAPSS_COLUMNS)]
    df.columns = CMAPSS_COLUMNS
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return normalize_cmapss_schema(df, add_condition_from_ops=True)


def read_rul_txt(path: str | Path) -> pd.Series:
    """执行 read rul txt 对应的项目处理逻辑。"""
    vals = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    vals.index = np.arange(1, len(vals) + 1)
    vals.name = "true_rul_after_last_cycle"
    return vals.astype(int)


def add_train_rul(df: pd.DataFrame) -> pd.DataFrame:
    """执行 add train rul 对应的项目处理逻辑。"""
    df = normalize_cmapss_schema(df, add_condition_from_ops=True)
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df["failure_time"] = max_cycle
    df["rul"] = df["failure_time"] - df["cycle"]
    return df


def add_test_rul(df: pd.DataFrame, rul: Optional[pd.Series]) -> pd.DataFrame:
    """执行 add test rul 对应的项目处理逻辑。"""
    df = normalize_cmapss_schema(df, add_condition_from_ops=True)
    last_cycle = df.groupby("unit_id")["cycle"].transform("max")
    if rul is None:
        df["true_rul_after_last_cycle"] = np.nan
        df["failure_time"] = np.nan
        df["rul"] = np.nan
        return df
    rmap = rul.to_dict()
    df["true_rul_after_last_cycle"] = df["unit_id"].map(rmap)
    df["failure_time"] = last_cycle + df["true_rul_after_last_cycle"]
    df["rul"] = df["failure_time"] - df["cycle"]
    return df


def fit_transform_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    feature_columns: Sequence[str],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], RobustScaler]:
    """执行 fit transform features 对应的项目处理逻辑。"""
    scaler = RobustScaler()
    train_df = train_df.copy()
    train_df[list(feature_columns)] = train_df.loc[:, feature_columns].astype(float)
    train_df[list(feature_columns)] = scaler.fit_transform(train_df.loc[:, feature_columns])
    if test_df is not None:
        test_df = test_df.copy()
        test_df[list(feature_columns)] = test_df.loc[:, feature_columns].astype(float)
        test_df[list(feature_columns)] = scaler.transform(test_df.loc[:, feature_columns])
    return train_df, test_df, scaler


def make_survival_target(rul: float, horizon: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return event vector and at-risk mask for discrete hazard likelihood.

    event[k] = 1 when failure occurs at future step k+1.
    mask[k] = 1 for terms included in likelihood.
    For a complete run-to-failure sample with rul=r, include survival terms until r-1
    and event term at r. If r exceeds horizon, treat as right-censored at horizon.
    """
    event = np.zeros(horizon, dtype=np.float32)
    mask = np.zeros(horizon, dtype=np.float32)
    r = int(max(1, np.ceil(rul)))
    if r <= horizon:
        mask[:r] = 1.0
        event[r - 1] = 1.0
        event_index = r - 1
    else:
        mask[:] = 1.0
        event_index = -1
    return event, mask, event_index


class SlidingWindowSurvivalDataset(Dataset):
    """封装 SlidingWindowSurvivalDataset 相关状态与行为。"""
    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: Sequence[str],
        window_size: int,
        horizon: int,
        stride: int = 1,
        condition_label_column: Optional[str] = None,
        min_cycle: Optional[int] = None,
    ) -> None:
        self.df = df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)
        self.feature_columns = list(feature_columns)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.stride = int(stride)
        self.condition_label_column = condition_label_column
        self.samples: List[Tuple[int, int]] = []

        for unit, g in self.df.groupby("unit_id", sort=True):
            idx = g.index.to_numpy()
            cycles = g["cycle"].to_numpy()
            start_pos = self.window_size - 1
            if min_cycle is not None:
                start_pos = max(start_pos, int(np.searchsorted(cycles, min_cycle)))
            for pos in range(start_pos, len(idx), self.stride):
                self.samples.append((int(unit), int(idx[pos])))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        unit, end_idx = self.samples[item]
        row = self.df.loc[end_idx]
        g = self.df[self.df["unit_id"] == unit]
        pos = int(np.where(g.index.to_numpy() == end_idx)[0][0])
        start_pos = max(0, pos - self.window_size + 1)
        w = g.iloc[start_pos : pos + 1]
        x = w[self.feature_columns].to_numpy(dtype=np.float32)
        if len(x) < self.window_size:
            pad = np.repeat(x[:1], self.window_size - len(x), axis=0)
            x = np.concatenate([pad, x], axis=0)
        event, mask, event_index = make_survival_target(row["rul"], self.horizon)
        cond = -1
        if self.condition_label_column and self.condition_label_column in row.index:
            cond = int(row[self.condition_label_column])
            if cond >= 1:
                cond -= 1  # user CMAPSSLoader uses 1..6; torch cross_entropy expects 0..K-1
        return {
            "x": torch.from_numpy(x),
            "event": torch.from_numpy(event),
            "mask": torch.from_numpy(mask),
            "rul": torch.tensor(float(row.get("rul", np.nan)), dtype=torch.float32),
            "unit_id": torch.tensor(unit, dtype=torch.long),
            "cycle": torch.tensor(int(row["cycle"]), dtype=torch.long),
            "failure_time": torch.tensor(float(row.get("failure_time", np.nan)), dtype=torch.float32),
            "condition": torch.tensor(cond, dtype=torch.long),
            "event_index": torch.tensor(event_index, dtype=torch.long),
        }


def split_by_unit(df: pd.DataFrame, train_ratio: float = 0.85, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行 split by unit 对应的项目处理逻辑。"""
    rng = np.random.default_rng(seed)
    units = np.array(sorted(df["unit_id"].unique()))
    rng.shuffle(units)
    n_train = max(1, int(len(units) * train_ratio))
    train_units = set(units[:n_train])
    return df[df["unit_id"].isin(train_units)].copy(), df[~df["unit_id"].isin(train_units)].copy()
