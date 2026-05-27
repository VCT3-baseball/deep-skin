# -*- coding: utf-8 -*-
"""
캐시 기반 multi-task 피부 분석 모델 학습 스크립트.

사용 예 (프로젝트 루트 face_cnn/ 에서 실행):
    python src/multivalue/train.py \
        --cache_pattern feature_cache/feat_vith_part{p}_m15_k16.pt \
        --csv face_data/annotations_relative.csv \
        --output_dir runs/vith_v1

라벨은 자동으로 LABEL_REGISTRY ∩ CSV 컬럼만 학습한다.
--labels 로 명시 지정도 가능.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import CachedFacepartDataset, build_split, detect_feat_dim
from .skin_heads import (
    UncertaintyWeighting,
    corn_loss,
    emd_loss,
    gaussian_nll,
    poisson_nll,
)
from .skin_metrics import (
    macro_f1,
    mae,
    pearson_r,
    quadratic_weighted_kappa,
    r2_score,
)
from .model import (
    FACEPART_TO_TRUNK,
    LABEL_REGISTRY,
    MultiTaskSkinModel,
)


# ============================================================
# Collate (facepart 단위 batch)
# ============================================================

def collate_facepart(batch):
    """동일 facepart 샘플들의 배치를 만든다.
    train: features 키만 사용. eval: eval/flip/aug 추가."""
    out = {
        "features": torch.stack([b["features"] for b in batch]),
        "eval_features": torch.stack([b["eval_features"] for b in batch]),
        "flip_features": torch.stack([b["flip_features"] for b in batch]),
        "facepart": int(batch[0]["facepart"]),
    }
    if "aug_features" in batch[0]:
        out["aug_features"] = torch.stack([b["aug_features"] for b in batch])
    label_keys = list(batch[0]["labels"].keys())
    out["labels"] = {
        k: torch.stack([b["labels"][k] for b in batch]) for k in label_keys
    }
    return out


# ============================================================
# 연속 라벨 z-score 통계
# ============================================================

def compute_reg_norm_stats(train_datasets, reg_labels):
    """train split에서만 평균/표준편차 계산."""
    stats = {}
    for fp, ds in train_datasets.items():
        for col in reg_labels:
            if col not in ds.labels:
                continue
            v = ds.labels[col].numpy()
            v = v[~np.isnan(v)]
            if len(v) >= 10:
                stats[col] = (float(v.mean()), float(v.std() + 1e-6))
    return stats


def compute_ord_class_weights(train_datasets, ord_labels, mode: str = "sqrt_inverse"):
    """train split에서 ordinal 라벨별 클래스 가중치 계산.
    mode:
        'none'         → None
        'inverse'      → w[c] = N / (K · count[c])
        'sqrt_inverse' → 위의 sqrt (기본; 덜 공격적)
    NaN/유효 샘플만 카운트. 평균 1로 정규화.
    """
    if mode == "none":
        return {}
    weights = {}
    for fp, ds in train_datasets.items():
        for col in ord_labels:
            if col not in ds.labels:
                continue
            meta = LABEL_REGISTRY[col]
            K = meta["K"]
            v = ds.labels[col].numpy()
            v = v[~np.isnan(v)]
            if len(v) == 0:
                continue
            v = v.astype(int).clip(0, K - 1)
            counts = np.bincount(v, minlength=K).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            w = v.size / (K * counts)
            if mode == "sqrt_inverse":
                w = np.sqrt(w)
            w = w / w.mean()  # mean=1
            weights[col] = torch.tensor(w, dtype=torch.float32)
    return weights


# ============================================================
# 단일 배치 손실
# ============================================================

def compute_losses(model, batch, reg_stats, device, aux_emd_w: float = 0.0,
                   ord_weights: dict = None):
    facepart = batch["facepart"]
    feats = batch["features"].to(device, non_blocking=True)
    labels = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

    repr_ = model.encode(feats, facepart)

    losses = {}
    for name, gt in labels.items():
        if name not in model.label_meta:
            continue
        meta = model.label_meta[name]
        valid = ~torch.isnan(gt)
        if valid.sum() == 0:
            continue
        out = model.head_forward(repr_, name)

        if meta["type"] == "ordinal":
            logits = out  # (B, K-1)
            tgt = gt.clone()
            tgt[~valid] = 0
            cw = (ord_weights or {}).get(name)
            l = corn_loss(logits, tgt, valid, class_weights=cw)
            if aux_emd_w > 0:
                l = l + aux_emd_w * emd_loss(logits, tgt, meta["K"], valid)
            losses[name] = l

        elif meta["type"] == "count":
            lam = out  # (B,)
            losses[name] = poisson_nll(lam, gt, valid)

        elif meta["type"] == "reg":
            mu, log_var = out
            mu = mu.squeeze(-1)
            log_var = log_var.squeeze(-1)
            mean, std = reg_stats.get(name, (0.0, 1.0))
            tgt = (gt - mean) / std
            tgt[~valid] = 0
            losses[name] = gaussian_nll(mu, log_var, tgt, valid)

    return losses


# ============================================================
# 평가 (TTA 포함)
# ============================================================

@torch.no_grad()
def evaluate(model, val_loaders, reg_stats, device, use_tta: bool = True):
    model.eval()
    preds = defaultdict(list)
    gts = defaultdict(list)

    for fp, loader in val_loaders.items():
        for batch in loader:
            facepart = batch["facepart"]
            # ---- 평균 representation (feature-space TTA) ----
            feat_list = [batch["eval_features"].to(device, non_blocking=True)]
            if use_tta:
                feat_list.append(batch["flip_features"].to(device, non_blocking=True))
                if "aug_features" in batch:
                    aug = batch["aug_features"].to(device, non_blocking=True)  # (B, K, D)
                    for k in range(aug.shape[1]):
                        feat_list.append(aug[:, k])
            repr_avg = None
            for f in feat_list:
                r = model.encode(f, facepart)
                repr_avg = r if repr_avg is None else repr_avg + r
            repr_avg = repr_avg / len(feat_list)

            # ---- 각 라벨에 대해 예측 ----
            for name, gt in batch["labels"].items():
                if name not in model.label_meta:
                    continue
                meta = model.label_meta[name]
                valid = ~torch.isnan(gt)
                if valid.sum() == 0:
                    continue

                if meta["type"] == "ordinal":
                    head = model.head(name)
                    grade, _ = head.predict_grade(repr_avg)
                    p = grade.detach().cpu().numpy()
                elif meta["type"] == "count":
                    out = model.head_forward(repr_avg, name)
                    p = out.detach().cpu().numpy()
                elif meta["type"] == "reg":
                    mu, _ = model.head_forward(repr_avg, name)
                    mean, std = reg_stats.get(name, (0.0, 1.0))
                    p = (mu.squeeze(-1).detach().cpu().numpy() * std + mean)
                else:
                    continue

                vmask = valid.cpu().numpy()
                preds[name].append(p[vmask])
                gts[name].append(gt[valid].cpu().numpy())

    metrics = {}
    for name in preds:
        meta = LABEL_REGISTRY[name]
        p = np.concatenate(preds[name])
        g = np.concatenate(gts[name])
        if len(p) == 0:
            continue
        if meta["type"] == "ordinal":
            K = meta["K"]
            metrics[name] = {
                "qwk": quadratic_weighted_kappa(g.astype(int), p.astype(int), K),
                "macro_f1": macro_f1(g.astype(int), p.astype(int), K),
                "acc": float((p.astype(int) == g.astype(int)).mean()),
                "mae": mae(p, g),
                "n": int(len(p)),
            }
        elif meta["type"] == "count":
            metrics[name] = {
                "mae": mae(p, g),
                "r": pearson_r(p, g),
                "n": int(len(p)),
            }
        elif meta["type"] == "reg":
            metrics[name] = {
                "r": pearson_r(p, g),
                "r2": r2_score(p, g),
                "mae": mae(p, g),
                "n": int(len(p)),
            }
    return metrics


NOISY_REG_PREFIXES = ("R1_", "R4_", "R9_")  # 양자화 reg (unique <150) — best 선택에서 제외


def primary_score(metrics, stratified: bool = True):
    """체크포인트 선택용 단일 스칼라.
    stratified=True (기본): ordinal QWK / count r / reg r(노이즈 제외) 그룹별 평균을 다시 평균.
    stratified=False: 전체 라벨 단순 평균 (구버전 호환, score_full로 함께 기록)."""
    ord_s, cnt_s, reg_s = [], [], []
    for name, m in metrics.items():
        meta = LABEL_REGISTRY.get(name, {})
        t = meta.get("type")
        if t == "ordinal":
            ord_s.append(m["qwk"])
        elif t == "count":
            cnt_s.append(m["r"])
        elif t == "reg":
            if stratified and any(name.startswith(p) for p in NOISY_REG_PREFIXES):
                continue
            reg_s.append(m["r"])
    if stratified:
        groups = [s for s in (ord_s, cnt_s, reg_s) if s]
        return float(np.mean([np.mean(g) for g in groups])) if groups else 0.0
    all_s = ord_s + cnt_s + reg_s
    return float(np.mean(all_s)) if all_s else 0.0


# ============================================================
# 학습 루프
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_pattern", type=str, required=True,
                   help="e.g. feature_cache/feat_vith_part{p}_m15_k16.pt")
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--feat_dim", type=int, default=None,
                   help="미지정 시 캐시에서 자동 감지")
    p.add_argument("--trunk_hidden", type=int, default=512)
    p.add_argument("--trunk_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=2)
    p.add_argument("--aux_emd_weight", type=float, default=0.2,
                   help="auxiliary EMD loss 가중치 (0이면 CORN만)")
    p.add_argument("--ord_class_weight", type=str, default="sqrt_inverse",
                   choices=["none", "inverse", "sqrt_inverse"],
                   help="ordinal CORN의 클래스 빈도 가중 방식")
    p.add_argument("--amp", action="store_true", default=True,
                   help="mixed precision (fp16). --no_amp로 끔.")
    p.add_argument("--no_amp", dest="amp", action="store_false")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--labels", type=str, nargs="*", default=None,
                   help="명시 지정 시 이 라벨만 학습. 미지정 시 CSV ∩ LABEL_REGISTRY 자동.")
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.10)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ---- 1. CSV & split ----
    df = pd.read_csv(args.csv)
    train_paths, val_paths, test_paths = build_split(
        df, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed,
    )
    print(f"[split] train={len(train_paths)} val={len(val_paths)} test={len(test_paths)}")

    # ---- 2. feat_dim 자동 감지 ----
    feat_dim = args.feat_dim or detect_feat_dim(args.cache_pattern)
    print(f"[feat_dim] {feat_dim}")

    # ---- 3. 학습할 라벨 결정 ----
    if args.labels:
        active = [l for l in args.labels if l in LABEL_REGISTRY]
        missing = set(args.labels) - set(active)
        if missing:
            print(f"[warn] LABEL_REGISTRY에 없는 라벨 무시: {sorted(missing)}")
    else:
        active = [l for l in LABEL_REGISTRY if l in df.columns]
    if not active:
        print("[fatal] 학습할 라벨이 하나도 없습니다.")
        sys.exit(1)
    print(f"[labels] {active}")

    # ---- 4. facepart별 dataset 빌드 ----
    fp2labels = defaultdict(list)
    for l in active:
        fp2labels[LABEL_REGISTRY[l]["facepart"]].append(l)

    train_ds, val_ds = {}, {}
    for fp, fp_labels in fp2labels.items():
        cp = args.cache_pattern.format(p=fp)
        if not os.path.exists(cp):
            print(f"[warn] cache missing for facepart {fp}: {cp} — skip")
            continue
        train_ds[fp] = CachedFacepartDataset(
            cp, df, fp_labels, keep_image_paths=train_paths,
            mode="train", load_aug=True,
        )
        val_ds[fp] = CachedFacepartDataset(
            cp, df, fp_labels, keep_image_paths=val_paths,
            mode="eval", load_aug=not args.no_tta,
        )
        print(f"  facepart {fp} ({FACEPART_TO_TRUNK[fp]}): "
              f"train={len(train_ds[fp])}, val={len(val_ds[fp])}, "
              f"labels={fp_labels}")

    if not train_ds:
        print("[fatal] 로드된 dataset이 없습니다.")
        sys.exit(1)

    # ---- 5. 연속 라벨 정규화 통계 ----
    reg_labels = [l for l in active if LABEL_REGISTRY[l]["type"] == "reg"]
    reg_stats = compute_reg_norm_stats(train_ds, reg_labels)
    if reg_stats:
        print(f"[reg z-score] {json.dumps(reg_stats, indent=2)}")

    # ordinal class weights (불균형 보정)
    ord_labels = [l for l in active if LABEL_REGISTRY[l]["type"] == "ordinal"]
    ord_weights = compute_ord_class_weights(train_ds, ord_labels, mode=args.ord_class_weight)
    if ord_weights:
        print(f"[ord class weights mode={args.ord_class_weight}]")
        for k, w in ord_weights.items():
            print(f"  {k:32s} {w.tolist()}")

    # ---- 6. 모델 + 가중 모듈 ----
    model = MultiTaskSkinModel(
        feat_dim=feat_dim,
        active_labels=active,
        trunk_hidden=args.trunk_hidden,
        trunk_layers=args.trunk_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model params] {n_params/1e6:.2f}M")

    task_types = {
        l: ("regression" if LABEL_REGISTRY[l]["type"] in ("reg", "count")
            else "classification")
        for l in active
    }
    weighter = UncertaintyWeighting(active, task_types).to(device)

    # ---- 7. 옵티마이저 ----
    params = list(model.parameters()) + list(weighter.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay
    )

    def lr_at(step, total_steps):
        warmup = args.warmup_epochs * steps_per_epoch
        if step < warmup:
            return step / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    # ---- 8. DataLoaders ----
    train_loaders = {
        fp: DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=collate_facepart,
            drop_last=True, pin_memory=True,
        )
        for fp, ds in train_ds.items()
    }
    val_loaders = {
        fp: DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_facepart,
            pin_memory=True,
        )
        for fp, ds in val_ds.items()
    }
    steps_per_epoch = sum(len(l) for l in train_loaders.values())
    total_steps = steps_per_epoch * args.epochs
    print(f"[schedule] steps/epoch={steps_per_epoch}, total={total_steps}")

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"[amp] enabled={use_amp}")

    # ---- 9. 학습 ----
    history = []
    best_score = -float("inf")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighter.train()
        t0 = time.time()

        epoch_losses = defaultdict(list)

        # facepart 순서 무작위로 (한 epoch 안에서 facepart를 round-robin)
        fp_order = list(train_loaders.keys())
        np.random.shuffle(fp_order)

        # 각 facepart의 iterator를 만들고 round-robin
        iters = {fp: iter(train_loaders[fp]) for fp in fp_order}
        remaining = {fp: len(train_loaders[fp]) for fp in fp_order}

        while any(remaining[fp] > 0 for fp in fp_order):
            for fp in fp_order:
                if remaining[fp] <= 0:
                    continue
                try:
                    batch = next(iters[fp])
                except StopIteration:
                    remaining[fp] = 0
                    continue
                remaining[fp] -= 1

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    losses = compute_losses(
                        model, batch, reg_stats, device,
                        aux_emd_w=args.aux_emd_weight,
                        ord_weights=ord_weights,
                    )
                    if not losses:
                        global_step += 1
                        continue
                    total = weighter(losses)

                # warmup + cosine
                lr_scale = lr_at(global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr * lr_scale

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 5.0)
                scaler.step(optimizer)
                scaler.update()

                for k, v in losses.items():
                    epoch_losses[k].append(float(v.detach().item()))
                global_step += 1

        # ---- epoch 로그 ----
        elapsed = time.time() - t0
        avg = {k: float(np.mean(v)) for k, v in epoch_losses.items()}
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch:3d}] {elapsed:.1f}s  lr={cur_lr:.2e}")
        for k in sorted(avg):
            lv = float(weighter.log_vars[k].detach().item())
            print(f"   train/{k:32s} loss={avg[k]:.4f}  log_var={lv:+.3f}")

        # ---- 검증 ----
        metrics = evaluate(model, val_loaders, reg_stats, device,
                           use_tta=not args.no_tta)
        score = primary_score(metrics, stratified=True)
        score_full = primary_score(metrics, stratified=False)
        print(f"   val score stratified={score:.4f}  full={score_full:.4f}")
        for k in sorted(metrics):
            print(f"   val/{k:32s} {metrics[k]}")

        history.append({
            "epoch": epoch, "elapsed": elapsed, "lr": cur_lr,
            "train_loss": avg, "val_metrics": metrics,
            "score": score, "score_full": score_full,
        })
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        # ---- best 저장 ----
        is_best = score > best_score
        if is_best:
            best_score = score
            torch.save({
                "model": model.state_dict(),
                "weighter": weighter.state_dict(),
                "reg_stats": reg_stats,
                "active_labels": active,
                "feat_dim": feat_dim,
                "trunk_hidden": args.trunk_hidden,
                "trunk_layers": args.trunk_layers,
                "dropout": args.dropout,
                "epoch": epoch,
                "val_metrics": metrics,
                "score": score,
                "args": vars(args),
            }, out_dir / "best.pt")
            print(f"   → saved best (score={score:.4f})")

        # 마지막 체크포인트도 저장
        torch.save({
            "model": model.state_dict(),
            "weighter": weighter.state_dict(),
            "reg_stats": reg_stats,
            "active_labels": active,
            "feat_dim": feat_dim,
            "trunk_hidden": args.trunk_hidden,
            "trunk_layers": args.trunk_layers,
            "dropout": args.dropout,
            "epoch": epoch,
        }, out_dir / "last.pt")

    print(f"\n[done] best score = {best_score:.4f}")


if __name__ == "__main__":
    main()
