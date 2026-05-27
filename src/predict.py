# -*- coding: utf-8 -*-
"""
새 얼굴 이미지 한 장 → DINOv3 백본 → MultiTaskSkinModel 추론.

전제: 호출자가 각 facepart의 bbox를 제공한다.
   bboxes = {
       0: (x, y, w, h),   # full   (선택: 전체 얼굴 영역. 없으면 이미지 전체 사용)
       1: (x, y, w, h),   # forehead
       2: (x, y, w, h),   # glabella
       3: (x, y, w, h),   # l_eye
       4: (x, y, w, h),   # r_eye
       5: (x, y, w, h),   # l_cheek
       6: (x, y, w, h),   # r_cheek
       7: (x, y, w, h),   # lips
       8: (x, y, w, h),   # chin
   }
   누락된 facepart는 그 부위 라벨만 건너뛴다.

사용 예:
    from infer_image import EndToEndInferencer
    e2e = EndToEndInferencer(
        head_checkpoint="runs/vits_v2_meas/best.pt",
        dinov3_checkpoint="dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        model_key="vits",
        device="cuda",
    )
    out = e2e.predict("path/to/image.jpg", bboxes={1: (100,80,500,150), 5: (...), ...})
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.amp import autocast
from contextlib import nullcontext

from .infer import SkinInferencer
from .model import FACEPART_TO_TRUNK, LABEL_REGISTRY

# ============================================================
# DINOv3 import (학습 때와 동일한 경로 탐색)
# ============================================================
_DINOV3_REPO = os.path.expanduser(r"~\.cache\torch\hub\facebookresearch_dinov3_main")
if os.path.isdir(_DINOV3_REPO) and _DINOV3_REPO not in sys.path:
    sys.path.insert(0, _DINOV3_REPO)
from dinov3.models.vision_transformer import DinoVisionTransformer  # noqa: E402


# build_feature_cache.py와 동일한 아키텍처 파라미터
MODEL_CONFIGS = {
    "vits": {"embed_dim": 384,  "depth": 12, "num_heads": 6},
    "vitl": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "vith": {"embed_dim": 1280, "depth": 32, "num_heads": 20},
    "vit7b":{"embed_dim": 4096, "depth": 40, "num_heads": 32},
}


def _build_backbone(ckpt_path: str, model_key: str, device: torch.device,
                    img_size: int = 448) -> DinoVisionTransformer:
    params = MODEL_CONFIGS[model_key]
    backbone = DinoVisionTransformer(
        img_size=img_size, patch_size=16, in_chans=3,
        pos_embed_rope_base=100.0, pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2.0, pos_embed_rope_dtype="fp32",
        embed_dim=params["embed_dim"], depth=params["depth"], num_heads=params["num_heads"],
        ffn_ratio=6, qkv_bias=True, drop_path_rate=0.0, layerscale_init=1e-5,
        norm_layer="layernormbf16", ffn_layer="swiglu", ffn_bias=True, proj_bias=True,
        n_storage_tokens=4, mask_k_bias=True,
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"DINOv3 ckpt not found: {ckpt_path}")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(sd, strict=False)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def _expand_bbox(x: int, y: int, w: int, h: int, margin: float,
                 img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    mx, my = int(round(w * margin)), int(round(h * margin))
    return (
        max(0, x - mx),
        max(0, y - my),
        min(img_w, x + w + mx),
        min(img_h, y + h + my),
    )


# ============================================================
# End-to-end inferencer
# ============================================================

class EndToEndInferencer:
    """이미지 한 장 + bbox 딕셔너리 → 라벨 예측 딕셔너리."""

    def __init__(
        self,
        head_checkpoint: str,
        dinov3_checkpoint: str,
        model_key: str = "vits",
        device: str = "cuda",
        img_size: int = 448,
        bbox_margin: float = 0.15,
        use_tta: bool = True,
    ):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        self.img_size = img_size
        self.bbox_margin = bbox_margin
        self.use_tta = use_tta

        # 1) head 모델
        self.skin = SkinInferencer(head_checkpoint, device=str(self.device))

        # 2) DINOv3 백본
        self.backbone = _build_backbone(dinov3_checkpoint, model_key, self.device, img_size)
        backbone_dim = MODEL_CONFIGS[model_key]["embed_dim"]
        if backbone_dim != self.skin.model.feat_dim:
            raise ValueError(
                f"feat_dim mismatch: backbone={backbone_dim} head_ckpt={self.skin.model.feat_dim}. "
                f"model_key='{model_key}' 가 head 학습 때 쓴 백본과 다른 것 같음."
            )

        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        self._eval_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self._flip_t = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        self._amp_ctx = (
            autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" else nullcontext()
        )

    # ---- helpers ----

    def _crop(self, img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
        x, y, w, h = bbox
        W, H = img.size
        return img.crop(_expand_bbox(int(x), int(y), int(w), int(h), self.bbox_margin, W, H))

    @torch.no_grad()
    def _extract(self, pil_crop: Image.Image) -> Dict[str, torch.Tensor]:
        """한 crop → eval/flip feature (1, D)."""
        e = self._eval_t(pil_crop).unsqueeze(0).to(self.device, non_blocking=True)
        feats = {"eval": None, "flip": None}
        with self._amp_ctx:
            feats["eval"] = self.backbone(e).float()
            if self.use_tta:
                f = self._flip_t(pil_crop).unsqueeze(0).to(self.device, non_blocking=True)
                feats["flip"] = self.backbone(f).float()
        return feats

    # ---- public API ----

    @torch.no_grad()
    def predict(
        self,
        image: "str | Path | Image.Image",
        bboxes: Dict[int, Tuple[int, int, int, int]],
    ) -> Dict[str, float]:
        """
        Args:
            image  : 이미지 경로 또는 PIL Image
            bboxes : {facepart_id: (x, y, w, h)}. 누락된 facepart는 건너뜀.
                     facepart 0(full) bbox가 없으면 이미지 전체를 사용.
        Returns:
            {label_name: value} — 각 라벨에 대해 스칼라 (int grade / float λ / float reg).
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB") if image.mode != "RGB" else image

        results: Dict[str, float] = {}
        # 학습된 라벨이 있는 facepart만 순회
        for fp in sorted(self.skin.fp2labels):
            if fp == 0 and fp not in bboxes:
                crop = img  # full face: bbox 없으면 이미지 전체
            elif fp in bboxes:
                crop = self._crop(img, bboxes[fp])
            else:
                continue  # 이 facepart는 추론 불가

            feats = self._extract(crop)
            out = self.skin.predict(
                feats["eval"], fp,
                flip_features=feats["flip"],
                aug_features=None,  # 새 이미지엔 K-view aug 캐시가 없음. flip TTA만.
            )
            for name, vec in out.items():
                results[name] = float(vec[0])
        return results


# ============================================================
# CLI: 이미지 + bbox JSON → 예측 JSON
# ============================================================

def _main():
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--head_checkpoint", required=True)
    p.add_argument("--dinov3_checkpoint", required=True)
    p.add_argument("--model_key", default="vits", choices=list(MODEL_CONFIGS))
    p.add_argument("--image", required=True)
    p.add_argument("--bboxes_json", required=True,
                   help='{"1":[x,y,w,h], "5":[...], ...} 형식의 JSON 파일')
    p.add_argument("--out", default=None, help="결과 JSON 경로 (미지정 시 stdout)")
    p.add_argument("--no_tta", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    with open(args.bboxes_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    bboxes = {int(k): tuple(v) for k, v in raw.items()}

    e2e = EndToEndInferencer(
        head_checkpoint=args.head_checkpoint,
        dinov3_checkpoint=args.dinov3_checkpoint,
        model_key=args.model_key,
        device=args.device,
        use_tta=not args.no_tta,
    )
    results = e2e.predict(args.image, bboxes)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out}  ({len(results)} labels)")
    else:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _main()
