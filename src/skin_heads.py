# -*- coding: utf-8 -*-
"""
헤드 모듈 + 손실 함수.

헤드:
    - MLPTrunk         : 부위별 공유 trunk
    - CORNHead         : Rank-consistent ordinal regression (Cao et al. 2020)
    - PoissonHead      : 양수 count regression
    - GaussianRegHead  : 평균/분산 동시 추정 (aleatoric uncertainty)

손실:
    - corn_loss        : CORN conditional BCE (벡터화)
    - emd_loss         : Earth Mover's Distance (NIMA-style)  ← auxiliary
    - poisson_nll      : Poisson negative log-likelihood
    - gaussian_nll     : Gaussian NLL with learned variance

다중 손실 결합:
    - UncertaintyWeighting: Kendall & Gal (2018) homoscedastic uncertainty weighting
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Trunk
# ============================================================

class MLPTrunk(nn.Module):
    """입력 feature → 표현 공간 (부위별 공유)."""

    def __init__(self, in_dim: int, hidden: int = 512, dropout: float = 0.1,
                 num_layers: int = 2):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        self.net = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.net(x))


# ============================================================
# Heads
# ============================================================

class CORNHead(nn.Module):
    """
    CORN: Rank-consistent Ordinal Regression (Cao et al. 2020).

    K-class ordinal 라벨을 K-1 binary task로 분해:
        task k 는 'y > k 인가?'
    학습: 각 task는 y >= k 인 샘플에 대해서만 BCE (conditional).
    추론: σ(logit) 의 누적곱이 unconditional P(y > k). > 0.5인 개수가 grade.
          → 단조성 자동 보장.
    """

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        assert num_classes >= 2
        self.num_classes = num_classes
        self.fc = nn.Linear(in_dim, num_classes - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (B, K-1)

    @torch.no_grad()
    def predict_grade(self, x: torch.Tensor):
        """grade: (B,) long in [0, K-1]; cum_probs: (B, K-1)."""
        logits = self.forward(x)
        cond = torch.sigmoid(logits)
        cum = torch.cumprod(cond, dim=1)
        grade = (cum > 0.5).sum(dim=1)
        return grade, cum


class PoissonHead(nn.Module):
    """Poisson regression head. softplus로 λ > 0 보장."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.fc(x).squeeze(-1)) + 1e-6  # (B,)


class GaussianRegHead(nn.Module):
    """
    예측 분포가 Gaussian인 회귀 헤드.
    출력: (μ, logσ²). 손실은 gaussian_nll.
    학습 타깃은 z-score 정규화되어야 하며, 추론 시 역변환은 외부에서.
    """

    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.mu = nn.Linear(in_dim, out_dim)
        self.log_var = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor):
        return self.mu(x), self.log_var(x).clamp(-8.0, 8.0)


# ============================================================
# 손실 함수
# ============================================================

def corn_loss(
    logits: torch.Tensor,           # (B, K-1)
    targets: torch.Tensor,          # (B,) long in [0, K-1]
    valid_mask: torch.Tensor,       # (B,) bool
    class_weights: torch.Tensor = None,  # (K,) float, mean≈1
) -> torch.Tensor:
    """
    벡터화 CORN loss.
        task k loss = BCE( σ(logit_k), 1[y > k] )  단, y >= k 인 샘플만 학습
    class_weights: 샘플의 ground-truth 클래스에 따른 sample weight (불균형 보정).
    """
    if valid_mask.sum() == 0:
        return logits.sum() * 0.0

    logits = logits[valid_mask]
    targets = targets[valid_mask].long()
    B, Km1 = logits.shape

    k_idx = torch.arange(Km1, device=logits.device).unsqueeze(0)   # (1, K-1)
    t = targets.unsqueeze(1)                                       # (B, 1)
    cond_mask = (t >= k_idx).float()                               # (B, K-1)
    bin_target = (t > k_idx).float()                               # (B, K-1)

    bce = F.binary_cross_entropy_with_logits(
        logits, bin_target, reduction="none"
    )
    if class_weights is not None:
        K = class_weights.numel()
        idx = targets.clamp(0, K - 1)
        sw = class_weights.to(logits.device, dtype=logits.dtype)[idx].unsqueeze(1)
        weighted = bce * cond_mask * sw
        return weighted.sum() / ((cond_mask * sw).sum() + 1e-8)
    return (bce * cond_mask).sum() / (cond_mask.sum() + 1e-8)


def emd_loss(
    logits_km1: torch.Tensor,       # (B, K-1)  — CORN logits 재사용
    targets: torch.Tensor,          # (B,) long
    num_classes: int,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary EMD/Wasserstein loss (NIMA-style).
    CORN과 동일한 logits을 사용하기 위해 0번 클래스 logit=0을 prepend하여
    K차원 softmax 분포로 만들고, ground truth one-hot의 CDF와 L2 거리를 계산.

    엄밀하지는 않으므로 auxiliary 용도 (작은 가중치)에 한해 사용 권장.
    """
    if valid_mask.sum() == 0:
        return logits_km1.sum() * 0.0
    logits_km1 = logits_km1[valid_mask]
    targets = targets[valid_mask].long().clamp(0, num_classes - 1)

    z0 = torch.zeros_like(logits_km1[:, :1])
    logits_k = torch.cat([z0, logits_km1], dim=1)                  # (B, K)
    probs = F.softmax(logits_k, dim=1)
    one_hot = F.one_hot(targets, num_classes).float()
    pred_cdf = torch.cumsum(probs, dim=1)
    true_cdf = torch.cumsum(one_hot, dim=1)
    return torch.sqrt(((pred_cdf - true_cdf) ** 2).mean(dim=1) + 1e-8).mean()


def poisson_nll(
    lambda_pred: torch.Tensor,      # (B,) λ>0
    count_gt: torch.Tensor,         # (B,) float
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return lambda_pred.sum() * 0.0
    lp = lambda_pred[valid_mask]
    ct = count_gt[valid_mask].float()
    return (lp - ct * torch.log(lp + 1e-8)).mean()


def gaussian_nll(
    mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if valid_mask.sum() == 0:
        return mu.sum() * 0.0
    mu = mu[valid_mask]
    lv = log_var[valid_mask]
    tg = target[valid_mask]
    return (0.5 * (mu - tg) ** 2 / lv.exp() + 0.5 * lv).mean()


# ============================================================
# 다중 손실 결합
# ============================================================

class UncertaintyWeighting(nn.Module):
    """
    Kendall & Gal (2018), 'Multi-task Learning Using Uncertainty to Weigh Losses'.

    Per-task 학습 파라미터 log σ²_i 를 두고:
        회귀형 : L_i / (2 σ_i²) + 0.5 log σ_i²
        분류형 : L_i /     σ_i² + 0.5 log σ_i²
    초기값 0으로 두면 동등 가중에서 시작해 자동 균형.
    """

    def __init__(self, task_names, task_types: Dict[str, str]):
        super().__init__()
        self.task_types = dict(task_types)
        self.log_vars = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(())) for n in task_names}
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = None
        for name, loss in losses.items():
            lv = self.log_vars[name]
            if self.task_types.get(name) == "regression":
                term = 0.5 * (loss / lv.exp() + lv)
            else:
                term = loss / lv.exp() + 0.5 * lv
            total = term if total is None else total + term
        if total is None:
            # 안전: 빈 dict일 때
            any_lv = next(iter(self.log_vars.values()))
            return any_lv * 0.0
        return total
