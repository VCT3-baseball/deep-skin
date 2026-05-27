# -*- coding: utf-8 -*-
"""
평가 지표.
    - quadratic_weighted_kappa : ordinal classification 표준 지표
    - macro_f1                  : 균형 잡힌 분류 정확도
    - mae                       : count/regression 기본
    - pearson_r                 : 연속값 회귀의 단조 일치도
    - r2_score                  : 분산 설명력
"""

import numpy as np


def quadratic_weighted_kappa(y_true, y_pred, num_classes: int) -> float:
    """Quadratic Weighted Kappa.
    오분류의 거리 제곱을 가중치로 사용해, 인접 클래스 혼동은 가볍게,
    먼 클래스 혼동은 무겁게 페널티 부여."""
    y_true = np.asarray(y_true, dtype=int).clip(0, num_classes - 1)
    y_pred = np.asarray(y_pred, dtype=int).clip(0, num_classes - 1)
    if len(y_true) == 0:
        return 0.0

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1.0

    i = np.arange(num_classes)
    W = ((i[:, None] - i[None, :]) ** 2) / max(((num_classes - 1) ** 2), 1)

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    total = O.sum()
    if total == 0:
        return 0.0
    E = np.outer(act_hist, pred_hist) / total

    num = (W * O).sum()
    den = (W * E).sum()
    if den < 1e-12:
        return 1.0
    return float(1.0 - num / den)


def macro_f1(y_true, y_pred, num_classes: int) -> float:
    """Class-wise F1의 단순 평균 (zero-class 편향에 둔감)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    f1s = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        if prec + rec == 0:
            f1s.append(0.0)
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def mae(pred, target) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    if len(p) == 0:
        return 0.0
    return float(np.mean(np.abs(p - t)))


def pearson_r(pred, target) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    if len(p) < 2 or p.std() < 1e-8 or t.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(p, t)[0, 1])


def r2_score(pred, target) -> float:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(target, dtype=float)
    if len(p) < 2:
        return 0.0
    ss_res = ((t - p) ** 2).sum()
    ss_tot = ((t - t.mean()) ** 2).sum()
    if ss_tot < 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
