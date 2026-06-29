"""feature_transformer 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""Reusable feature transformers for residual/hybrid feature modes.

Phase 4 introduces the first production-safe transformer used by both training
and inference.  The goal is not to replace the user's rich time-series analysis,
but to persist the exact feature transformation needed to reproduce residual
features after a model has been trained.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from lvd_surv.data.cmapss import CMAPSS_SETTING_COLS, normalize_cmapss_schema


@dataclass
class LinearResidualFeatureTransformer:
    """Create residual sensor features after removing operating-condition effects.

    For each selected raw sensor, this transformer fits a small linear model:

        sensor ~= intercept + setting_1 + setting_2 + setting_3 + condition dummies

    The residual ``sensor - fitted(sensor)`` is then written as
    ``<sensor>_resid``.  The fitted coefficients are stored in the object, so the
    same transformation can be applied to validation/test data and during later
    inference from a checkpoint.

    The implementation intentionally uses NumPy least squares instead of a
    heavyweight external model object.  This keeps the pickle artifact stable and
    easy to inspect across environments.
    """

    raw_feature_cols: Sequence[str]
    condition_columns: Sequence[str] = field(default_factory=lambda: list(CMAPSS_SETTING_COLS))
    include_condition_dummies: bool = True
    residual_suffix: str = "_resid"
    fitted_: bool = False
    design_columns_: List[str] = field(default_factory=list)
    coefficients_: Dict[str, List[float]] = field(default_factory=dict)
    residual_feature_cols_: List[str] = field(default_factory=list)
    metadata_: Dict[str, object] = field(default_factory=dict)

    def _available_condition_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in self.condition_columns if c in df.columns]

    def _build_design(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        """Build a deterministic design matrix for residualization."""
        parts: List[pd.DataFrame] = []
        cond_cols = self._available_condition_columns(df)
        if cond_cols:
            parts.append(df[cond_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float))
        if self.include_condition_dummies and "condition" in df.columns:
            dummies = pd.get_dummies(df["condition"].astype("Int64").astype(str), prefix="condition", dtype=float)
            parts.append(dummies)

        if parts:
            design = pd.concat(parts, axis=1)
        else:
            # Intercept-only residualization; useful for synthetic tests or data
            # where operating settings are intentionally absent.
            design = pd.DataFrame(index=df.index)

        if fit:
            self.design_columns_ = list(design.columns)
        else:
            for c in self.design_columns_:
                if c not in design.columns:
                    design[c] = 0.0
            design = design.loc[:, self.design_columns_]

        x = design.to_numpy(dtype=float) if len(design.columns) else np.empty((len(df), 0), dtype=float)
        intercept = np.ones((len(df), 1), dtype=float)
        return np.concatenate([intercept, x], axis=1)

    def fit(self, df: pd.DataFrame) -> "LinearResidualFeatureTransformer":
        """Fit residualization coefficients on training data."""
        train = normalize_cmapss_schema(df, add_condition_from_ops=True)
        missing = [c for c in self.raw_feature_cols if c not in train.columns]
        if missing:
            raise ValueError(f"Cannot fit residual transformer; missing raw features: {missing}")

        x = self._build_design(train, fit=True)
        self.coefficients_.clear()
        self.residual_feature_cols_ = [f"{c}{self.residual_suffix}" for c in self.raw_feature_cols]
        for col in self.raw_feature_cols:
            y = pd.to_numeric(train[col], errors="coerce").fillna(train[col].median()).to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            self.coefficients_[col] = [float(v) for v in coef]
        self.fitted_ = True
        self.metadata_ = {
            "schema_version": "1.0",
            "transformer_type": self.__class__.__name__,
            "raw_feature_cols": list(self.raw_feature_cols),
            "residual_feature_cols": list(self.residual_feature_cols_),
            "condition_columns": list(self.condition_columns),
            "design_columns": list(self.design_columns_),
            "include_condition_dummies": bool(self.include_condition_dummies),
            "residual_suffix": self.residual_suffix,
        }
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append residual columns to a dataframe using fitted coefficients."""
        if not self.fitted_:
            raise RuntimeError("LinearResidualFeatureTransformer must be fitted before transform().")
        out = normalize_cmapss_schema(df, add_condition_from_ops=True)
        missing = [c for c in self.raw_feature_cols if c not in out.columns]
        if missing:
            raise ValueError(f"Cannot transform residual features; missing raw features: {missing}")
        x = self._build_design(out, fit=False)
        for col in self.raw_feature_cols:
            coef = np.asarray(self.coefficients_[col], dtype=float)
            fitted = x @ coef
            values = pd.to_numeric(out[col], errors="coerce").fillna(out[col].median()).to_numpy(dtype=float)
            out[f"{col}{self.residual_suffix}"] = values - fitted
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on ``df`` and return a transformed copy."""
        return self.fit(df).transform(df)

    @property
    def residual_feature_cols(self) -> List[str]:
        """Residual feature column names created by this transformer."""
        if self.residual_feature_cols_:
            return list(self.residual_feature_cols_)
        return [f"{c}{self.residual_suffix}" for c in self.raw_feature_cols]

    def to_contract(self) -> Dict[str, object]:
        """Return JSON-safe transformer metadata for checkpoint/reporting."""
        return dict(self.metadata_)


def build_residual_transformer(
    *,
    raw_feature_cols: Sequence[str],
    cfg: Optional[Mapping[str, object]] = None,
) -> LinearResidualFeatureTransformer:
    """Factory used by the integrated pipeline.

    Configuration keys are kept under ``features`` so users can change
    the residualization inputs without touching training code.
    """
    fs_cfg = dict((cfg or {}).get("features", {}) if isinstance(cfg, Mapping) else {})
    condition_columns = fs_cfg.get("residual_condition_columns") or fs_cfg.get("condition_columns")
    if condition_columns is None:
        data_cfg = dict((cfg or {}).get("data", {}) if isinstance(cfg, Mapping) else {})
        condition_columns = data_cfg.get("condition_columns") or CMAPSS_SETTING_COLS
    return LinearResidualFeatureTransformer(
        raw_feature_cols=list(raw_feature_cols),
        condition_columns=list(condition_columns),
        include_condition_dummies=bool(fs_cfg.get("include_condition_dummies", True)),
        residual_suffix=str(fs_cfg.get("residual_suffix", "_resid")),
    )
