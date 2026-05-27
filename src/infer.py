# -*- coding: utf-8 -*-
"""
캐시된 DINOv3 특징 → 학습된 MultiTaskSkinModel 추론.

다른 프로젝트로 가져갈 때 필요한 파일:
    skin_model.py, skin_heads.py, skin_data.py (선택), infer.py

체크포인트(best.pt / last.pt)에는 모델 재구성에 필요한 모든 메타가
저장돼 있어서 학습 인자를 외부에서 다시 주지 않아도 됨:
    feat_dim, active_labels, trunk_hidden, trunk_layers, dropout, reg_stats

사용 예 (프로그램):
    inf = SkinInferencer("runs/vits_v2_meas/best.pt", device="cuda")
    # features: (B, D) float tensor, facepart: int (0..8)
    out = inf.predict(features, facepart=5)
    # out = {"l_cheek_pore": np.ndarray(B,), "moisture_l_cheek": ...}

사용 예 (CLI, 캐시 파일에서):
    python infer.py \
        --checkpoint runs/vits_v2_meas/best.pt \
        --cache_pattern feature_cache/feat_vits_part{p}_m15_k16.pt \
        --out predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch

from .model import FACEPART_TO_TRUNK, LABEL_REGISTRY, MultiTaskSkinModel


class SkinInferencer:
    """학습된 체크포인트 한 개를 들고 다니며 추론."""

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.active_labels: List[str] = list(ckpt["active_labels"])
        self.reg_stats: Dict[str, tuple] = dict(ckpt.get("reg_stats", {}))

        self.model = MultiTaskSkinModel(
            feat_dim=int(ckpt["feat_dim"]),
            active_labels=self.active_labels,
            trunk_hidden=int(ckpt["trunk_hidden"]),
            trunk_layers=int(ckpt["trunk_layers"]),
            dropout=float(ckpt["dropout"]),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        # facepart → 그 부위에서 추론 가능한 라벨
        self.fp2labels: Dict[int, List[str]] = {}
        for l in self.active_labels:
            fp = LABEL_REGISTRY[l]["facepart"]
            self.fp2labels.setdefault(fp, []).append(l)

    def labels_for_facepart(self, facepart: int) -> List[str]:
        return list(self.fp2labels.get(facepart, []))

    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        facepart: int,
        flip_features: Optional[torch.Tensor] = None,
        aug_features: Optional[torch.Tensor] = None,
        labels: Optional[Iterable[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        features: (B, D) float
        flip_features: (B, D)   — 있으면 TTA에 합산
        aug_features:  (B, K, D) — 있으면 K개 view도 합산
        labels: 추론할 라벨 부분집합. None이면 해당 facepart의 모든 라벨.
        반환: {label: np.ndarray(B,)}.
              ordinal=정수 grade, count=실수 λ, reg=원 스케일 실수.
        """
        feats = features.to(self.device, non_blocking=True).float()
        feat_list = [feats]
        if flip_features is not None:
            feat_list.append(flip_features.to(self.device).float())
        if aug_features is not None:
            aug = aug_features.to(self.device).float()
            for k in range(aug.shape[1]):
                feat_list.append(aug[:, k])

        repr_avg = None
        for f in feat_list:
            r = self.model.encode(f, facepart)
            repr_avg = r if repr_avg is None else repr_avg + r
        repr_avg = repr_avg / len(feat_list)

        target_labels = list(labels) if labels else self.labels_for_facepart(facepart)
        out: Dict[str, np.ndarray] = {}
        for name in target_labels:
            if name not in self.model.label_meta:
                continue
            meta = self.model.label_meta[name]
            if meta["facepart"] != facepart:
                continue
            if meta["type"] == "ordinal":
                grade, _ = self.model.head(name).predict_grade(repr_avg)
                out[name] = grade.detach().cpu().numpy()
            elif meta["type"] == "count":
                lam = self.model.head_forward(repr_avg, name)
                out[name] = lam.detach().cpu().numpy()
            elif meta["type"] == "reg":
                mu, _ = self.model.head_forward(repr_avg, name)
                mean, std = self.reg_stats.get(name, (0.0, 1.0))
                out[name] = (mu.squeeze(-1).detach().cpu().numpy() * std + mean)
        return out


# ============================================================
# CLI: 캐시 파일에서 배치 추론 → CSV
# ============================================================

def _iter_cache(cache_path: str, batch_size: int):
    """캐시(.pt) 파일을 batch_size씩 (image_paths, eval_feats, flip_feats, aug_feats) 로 yield."""
    c = torch.load(cache_path, map_location="cpu", weights_only=False)
    paths = list(c["image_paths"])
    ev = c["eval_features"]
    fl = c["flip_features"]
    ag = c.get("aug_features", None)
    N = len(paths)
    for s in range(0, N, batch_size):
        e = min(N, s + batch_size)
        yield (
            paths[s:e],
            ev[s:e].float(),
            fl[s:e].float(),
            ag[s:e].float() if ag is not None else None,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--cache_pattern", required=True,
                   help="예: feature_cache/feat_vits_part{p}_m15_k16.pt")
    p.add_argument("--out", required=True, help="결과 CSV 경로")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    inf = SkinInferencer(args.checkpoint, device=args.device)
    print(f"[labels] {inf.active_labels}")

    rows: Dict[str, dict] = {}  # image_path → {label: value}
    for fp in sorted(inf.fp2labels):
        cp = args.cache_pattern.format(p=fp)
        if not Path(cp).exists():
            print(f"[warn] cache missing for facepart {fp}: {cp} — skip")
            continue
        print(f"[infer] facepart={fp} ({FACEPART_TO_TRUNK[fp]}) {cp}")

        for paths, ev, fl, ag in _iter_cache(cp, args.batch_size):
            out = inf.predict(
                ev, fp,
                flip_features=None if args.no_tta else fl,
                aug_features=None if args.no_tta else ag,
            )
            for i, ip in enumerate(paths):
                row = rows.setdefault(ip, {"image_path": ip})
                for name, vec in out.items():
                    row[name] = float(vec[i])

    # CSV 출력 (pandas 없이 stdlib만)
    import csv
    all_cols = ["image_path"] + sorted(
        {k for r in rows.values() for k in r if k != "image_path"}
    )
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_cols)
        w.writeheader()
        for r in rows.values():
            w.writerow(r)
    print(f"[saved] {out_path}  rows={len(rows)}  cols={len(all_cols)}")


if __name__ == "__main__":
    main()
