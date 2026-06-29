"""mixture_prior 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

"""Lifetime-prior utilities used by training artifacts and inference.

Inference consumes only the generic JSON contract produced by the formal
mixed-lifetime analysis. Legacy special-case priors are intentionally rejected.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np



@dataclass
class GenericMixturePrior:
    """Distribution-agnostic lifetime mixture prior.

    Components are dictionaries containing ``name``, ``weight`` and ``values``.
    The values follow scipy's fitted parameter convention for each distribution
    when available, for example Weibull ``[shape, loc, scale]`` and lognormal
    ``[sigma, loc, scale]``.  The class exposes the same ``conditional_tail``
    ``conditional_tail`` interface consumed by inference.
    """

    components: Sequence[Mapping[str, Any]]

    def survival(self, t: np.ndarray) -> np.ndarray:
        """执行 survival 对应的项目处理逻辑。"""
        t = np.asarray(t, dtype=float).clip(min=0.0)
        s = np.zeros_like(t, dtype=float)
        total_weight = 0.0
        for comp in self.components:
            weight = float(comp.get("weight", 0.0))
            if weight <= 0:
                continue
            name = str(comp.get("name", "")).lower()
            values = [float(v) for v in comp.get("values", [])]
            s += weight * _survival_by_distribution(name, t, values)
            total_weight += weight
        if total_weight > 0:
            s = s / total_weight
        return np.clip(s, 0.0, 1.0)

    def conditional_tail(self, current_age: float, future_steps: np.ndarray) -> np.ndarray:
        """执行 conditional tail 对应的项目处理逻辑。"""
        future_abs = float(current_age) + np.asarray(future_steps, dtype=float)
        denom = float(self.survival(np.array([current_age], dtype=float))[0])
        if denom <= 1e-12:
            return np.zeros_like(future_abs, dtype=float)
        return np.clip(self.survival(future_abs) / denom, 0.0, 1.0)


def _survival_by_distribution(name: str, t: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Evaluate a supported scipy-style survival function defensively."""
    lname = name.lower()
    from scipy import stats  # type: ignore

    if lname in {"weibull", "weibull_min"}:
        if len(values) == 2:
            c, scale = values
            loc = 0.0
        elif len(values) >= 3:
            c, loc, scale = values[:3]
        else:
            raise ValueError("Weibull component requires [shape, loc, scale] or [shape, scale].")
        return np.asarray(stats.weibull_min.sf(t, c, loc=loc, scale=max(scale, 1e-12)), dtype=float)
    if lname in {"lognorm", "lognormal"}:
        if len(values) >= 3:
            s, loc, scale = values[:3]
        elif len(values) == 2:
            s, scale = values
            loc = 0.0
        else:
            raise ValueError("Lognormal component requires [sigma, loc, scale] or [sigma, scale].")
        return np.asarray(stats.lognorm.sf(t, s, loc=loc, scale=max(scale, 1e-12)), dtype=float)
    if lname in {"gamma", "gamma_dist"}:
        if len(values) >= 3:
            a, loc, scale = values[:3]
        elif len(values) == 2:
            a, scale = values
            loc = 0.0
        else:
            raise ValueError("Gamma component requires [shape, loc, scale] or [shape, scale].")
        return np.asarray(stats.gamma.sf(t, a, loc=loc, scale=max(scale, 1e-12)), dtype=float)
    if lname in {"expon", "exponential"}:
        if len(values) >= 2:
            loc, scale = values[:2]
        elif len(values) == 1:
            loc, scale = 0.0, values[0]
        else:
            loc, scale = 0.0, 1.0
        return np.asarray(stats.expon.sf(t, loc=loc, scale=max(scale, 1e-12)), dtype=float)
    if lname in {"norm", "normal", "gaussian"}:
        if len(values) >= 2:
            loc, scale = values[:2]
        elif len(values) == 1:
            loc, scale = values[0], 1.0
        else:
            loc, scale = 0.0, 1.0
        return np.asarray(stats.norm.sf(t, loc=loc, scale=max(scale, 1e-12)), dtype=float)
    raise ValueError(f"Unsupported lifetime prior distribution: {name}")


def _normalise_weights(weights: Iterable[float], n: int) -> np.ndarray:
    w = np.asarray(list(weights), dtype=float).reshape(-1)
    if w.size != n:
        w = np.ones(n, dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 0:
        w = np.ones(n, dtype=float)
    return w / float(w.sum())


def build_prior_from_contract(contract: Optional[Mapping[str, Any]]) -> Optional[object]:
    """Build an inference prior object from a JSON-safe prior contract.

    Returns ``None`` for disabled or missing contracts.  The returned object only
    needs to implement ``conditional_tail(current_age, future_steps)``.
    """
    if not contract or not bool(contract.get("enabled", True)):
        return None

    if {"weights", "shapes", "scales"}.issubset(contract.keys()):
        raise ValueError("Legacy Weibull-only prior contracts are unsupported; regenerate the formal mixture prior.")

    params = list(contract.get("params") or [])
    if not params:
        return None
    weights = _normalise_weights(contract.get("weights", []), len(params))
    dist_combo = list(contract.get("dist_combo") or [])
    components = []
    for idx, p in enumerate(params):
        if isinstance(p, Mapping):
            name = str(p.get("name") or (dist_combo[idx] if idx < len(dist_combo) else f"component_{idx}"))
            values = p.get("values") or p.get("params") or []
        else:
            name = str(dist_combo[idx] if idx < len(dist_combo) else f"component_{idx}")
            values = p
        components.append({"name": name, "weight": float(weights[idx]), "values": [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]})
    return GenericMixturePrior(components=components)



def blend_tail_with_prior(
    model_reliability: np.ndarray,
    current_age: float,
    prior: Optional[object],
    blend_start: int,
    blend_weight: float = 1.0,
) -> np.ndarray:
    """Blend model reliability with a lifetime prior tail after ``blend_start``.

    ``prior`` must be a ``GenericMixturePrior``.  The
    blend is intentionally conservative and monotone-clamped after blending.
    """
    if prior is None or blend_start >= len(model_reliability):
        return np.asarray(model_reliability, dtype=float)
    if not hasattr(prior, "conditional_tail"):
        raise TypeError("prior must implement conditional_tail(current_age, future_steps).")
    rel = np.asarray(model_reliability, dtype=float).copy()
    steps = np.arange(1, len(rel) + 1)
    prior_tail = np.asarray(prior.conditional_tail(current_age, steps), dtype=float)
    alpha = np.zeros_like(rel)
    alpha[blend_start:] = np.linspace(0.0, max(0.0, min(1.0, float(blend_weight))), len(rel) - blend_start)
    rel = (1 - alpha) * rel + alpha * prior_tail
    rel = np.minimum.accumulate(rel)
    return np.clip(rel, 0.0, 1.0)
