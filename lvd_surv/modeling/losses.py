"""losses 模块：提供项目内部的明确、可复用实现。"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def discrete_survival_nll(hazard: torch.Tensor, event: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """执行 discrete survival nll 对应的项目处理逻辑。"""
    eps = 1e-7
    log_h = torch.log(hazard.clamp(eps, 1 - eps))
    log_surv = torch.log((1.0 - hazard).clamp(eps, 1 - eps))
    # event=1 at failure bin; mask includes survival bins before failure and event bin.
    nll = -torch.sum(mask * ((1.0 - event) * log_surv + event * log_h), dim=-1)
    denom = torch.sum(mask, dim=-1).clamp_min(1.0)
    return torch.mean(nll / denom)


def monotonic_health_loss(health: torch.Tensor, unit_id: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
    """Softly encourage health/risk score to increase with cycle within each device."""
    if health.numel() < 2:
        return health.new_tensor(0.0)
    loss_terms = []
    for u in torch.unique(unit_id):
        idx = torch.where(unit_id == u)[0]
        if idx.numel() < 2:
            continue
        order = torch.argsort(cycle[idx])
        seq = health[idx][order]
        loss_terms.append(F.relu(seq[:-1] - seq[1:]).mean())
    if not loss_terms:
        return health.new_tensor(0.0)
    return torch.stack(loss_terms).mean()


def mode_entropy_loss(mode_prob: torch.Tensor) -> torch.Tensor:
    # Negative entropy minimization is not wanted. This returns low entropy penalty;
    # use a small negative weight only if forcing hard modes. Default code does not use it.
    """执行 mode entropy loss 对应的项目处理逻辑。"""
    eps = 1e-7
    ent = -torch.sum(mode_prob * torch.log(mode_prob.clamp_min(eps)), dim=-1).mean()
    return ent


def orthogonality_loss(mode_prob: torch.Tensor, health_z: torch.Tensor) -> torch.Tensor:
    # Batch-level correlation penalty between mode state and health latent.
    """执行 orthogonality loss 对应的项目处理逻辑。"""
    m = mode_prob - mode_prob.mean(dim=0, keepdim=True)
    z = health_z - health_z.mean(dim=0, keepdim=True)
    m = F.normalize(m, dim=0)
    z = F.normalize(z, dim=0)
    corr = torch.matmul(m.T, z) / max(1, mode_prob.shape[0] - 1)
    return corr.pow(2).mean()


def total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    beta_kl: float,
    lambda_mono: float,
    lambda_cond: float,
    lambda_orth: float,
    use_reconstruction: bool = False,
    lambda_recon: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """执行 total loss 对应的项目处理逻辑。"""
    loss_surv = discrete_survival_nll(outputs["hazard"], batch["event"], batch["mask"])
    loss_kl = outputs["kl"].mean()
    loss_mono = monotonic_health_loss(outputs["health_score"], batch["unit_id"], batch["cycle"])
    valid_cond = batch["condition"] >= 0
    if valid_cond.any():
        loss_cond = F.cross_entropy(outputs["mode_logits"][valid_cond], batch["condition"][valid_cond])
    else:
        loss_cond = outputs["hazard"].new_tensor(0.0)
    loss_orth = orthogonality_loss(outputs["mode_prob"], outputs["health_z"])
    loss = loss_surv + beta_kl * loss_kl + lambda_mono * loss_mono + lambda_cond * loss_cond + lambda_orth * loss_orth
    loss_recon = outputs["hazard"].new_tensor(0.0)
    if use_reconstruction and lambda_recon > 0:
        target_last = batch["x"][:, -1, :]
        loss_recon = F.mse_loss(outputs["recon_last"], target_last)
        loss = loss + lambda_recon * loss_recon
    return {
        "loss": loss,
        "survival": loss_surv.detach(),
        "kl": loss_kl.detach(),
        "mono": loss_mono.detach(),
        "condition": loss_cond.detach(),
        "orth": loss_orth.detach(),
        "recon": loss_recon.detach(),
    }


def hazard_to_failure_pmf(hazard: torch.Tensor) -> torch.Tensor:
    """执行 hazard to failure pmf 对应的项目处理逻辑。"""
    survival_before = torch.cumprod(torch.cat([torch.ones_like(hazard[:, :1]), 1.0 - hazard[:, :-1]], dim=1), dim=1)
    return hazard * survival_before


def expected_rul_from_hazard(hazard: torch.Tensor) -> torch.Tensor:
    """执行 expected rul from hazard 对应的项目处理逻辑。"""
    pmf = hazard_to_failure_pmf(hazard)
    steps = torch.arange(1, hazard.shape[1] + 1, device=hazard.device, dtype=hazard.dtype).view(1, -1)
    tail_surv = torch.cumprod(1.0 - hazard, dim=-1)[:, -1]
    exp = torch.sum(pmf * steps, dim=-1) + tail_surv * hazard.shape[1]
    return exp
