# -*- coding: utf-8 -*-
"""
Multi-task 피부 분석 모델.

설계 원칙:
    - facepart마다 trunk 하나씩 (대칭 부위는 입력에 side bit ±1 concat)
    - 좌/우 대칭 라벨은 같은 head를 공유 (데이터 효율 ↑)
    - 헤드 타입은 라벨 성격에 맞춤:
        * ordinal grade → CORNHead
        * count        → PoissonHead
        * continuous   → GaussianRegHead (z-score 타깃, NLL 학습)
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .skin_heads import MLPTrunk, CORNHead, PoissonHead, GaussianRegHead


# ============================================================
# Label registry — 각 라벨이 어느 facepart에 속하고, 어떤 head 그룹/타입을 쓰는지
# ============================================================
# group: 좌/우 공유 head 식별자
# side : None(비대칭) | -1(left) | +1(right)
# type : 'ordinal' | 'count' | 'reg'
# K    : ordinal일 때 클래스 수
LABEL_REGISTRY: Dict[str, dict] = {
    # ---- Ordinal (전문가 등급) ----
    "forehead_pigmentation":  {"facepart": 1, "group": "forehead_pigm",  "side": None, "type": "ordinal", "K": 4},
    "forehead_wrinkle":       {"facepart": 1, "group": "forehead_wrnk",  "side": None, "type": "ordinal", "K": 5},
    "glabellus_wrinkle":      {"facepart": 2, "group": "glabellus_wrnk", "side": None, "type": "ordinal", "K": 3},
    "l_perocular_wrinkle":    {"facepart": 3, "group": "perocular_wrnk", "side": -1,   "type": "ordinal", "K": 6},
    "r_perocular_wrinkle":    {"facepart": 4, "group": "perocular_wrnk", "side": +1,   "type": "ordinal", "K": 6},
    "l_cheek_pigmentation":   {"facepart": 5, "group": "cheek_pigm",     "side": -1,   "type": "ordinal", "K": 6},
    "r_cheek_pigmentation":   {"facepart": 6, "group": "cheek_pigm",     "side": +1,   "type": "ordinal", "K": 6},
    "l_cheek_pore":           {"facepart": 5, "group": "cheek_pore",     "side": -1,   "type": "ordinal", "K": 6},
    "r_cheek_pore":           {"facepart": 6, "group": "cheek_pore",     "side": +1,   "type": "ordinal", "K": 6},
    "lip_dryness":            {"facepart": 7, "group": "lip_dry",        "side": None, "type": "ordinal", "K": 5},
    "chin_sagging":           {"facepart": 8, "group": "chin_sag",       "side": None, "type": "ordinal", "K": 6},

    # ---- Counts ----
    "acne_count":             {"facepart": 0, "group": "acne_cnt",       "side": None, "type": "count"},
    "pigmentation_count":     {"facepart": 0, "group": "pigm_cnt",       "side": None, "type": "count"},
    "pore_count_l_cheek":     {"facepart": 5, "group": "pore_cnt",       "side": -1,   "type": "count"},
    "pore_count_r_cheek":     {"facepart": 6, "group": "pore_cnt",       "side": +1,   "type": "count"},

    # ---- Continuous regression (z-score 학습) ----
    # 수분 (부위별 측정. CSV에 컬럼이 있는 경우만 활성화됨)
    "moisture_forehead":      {"facepart": 1, "group": "moisture",       "side": None, "type": "reg"},
    "moisture_l_cheek":       {"facepart": 5, "group": "moisture",       "side": -1,   "type": "reg"},
    "moisture_r_cheek":       {"facepart": 6, "group": "moisture",       "side": +1,   "type": "reg"},
    # 탄력 R/Q-파라미터 (볼: 좌우 공유 head)
    **{f"R{i}_l_cheek": {"facepart": 5, "group": f"elasticity_R{i}", "side": -1, "type": "reg"}
       for i in range(10)},
    **{f"R{i}_r_cheek": {"facepart": 6, "group": f"elasticity_R{i}", "side": +1, "type": "reg"}
       for i in range(10)},
    **{f"Q{i}_l_cheek": {"facepart": 5, "group": f"elasticity_Q{i}", "side": -1, "type": "reg"}
       for i in range(4)},
    **{f"Q{i}_r_cheek": {"facepart": 6, "group": f"elasticity_Q{i}", "side": +1, "type": "reg"}
       for i in range(4)},
    # 탄력 (이마)
    **{f"R{i}_forehead": {"facepart": 1, "group": f"elasticity_fh_R{i}", "side": None, "type": "reg"}
       for i in range(10)},
    **{f"Q{i}_forehead": {"facepart": 1, "group": f"elasticity_fh_Q{i}", "side": None, "type": "reg"}
       for i in range(4)},
    # 탄력 (턱)
    **{f"R{i}_chin": {"facepart": 8, "group": f"elasticity_ch_R{i}", "side": None, "type": "reg"}
       for i in range(10)},
    **{f"Q{i}_chin": {"facepart": 8, "group": f"elasticity_ch_Q{i}", "side": None, "type": "reg"}
       for i in range(4)},
    # 거칠기 (눈가: 좌우 공유 head)
    **{f"{m}_l_eye": {"facepart": 3, "group": f"roughness_{m}", "side": -1, "type": "reg"}
       for m in ("Ra", "Rq", "Rmax", "R3z", "Rt", "Rz", "Rp", "Rv")},
    **{f"{m}_r_eye": {"facepart": 4, "group": f"roughness_{m}", "side": +1, "type": "reg"}
       for m in ("Ra", "Rq", "Rmax", "R3z", "Rt", "Rz", "Rp", "Rv")},
}


# facepart → 어느 trunk를 쓰는가
FACEPART_TO_TRUNK = {
    0: "full",
    1: "forehead",
    2: "glabella",
    3: "eye",     # 좌우 공유
    4: "eye",
    5: "cheek",   # 좌우 공유
    6: "cheek",
    7: "lips",
    8: "chin",
}
SYMMETRIC_TRUNKS = {"eye", "cheek"}


# ============================================================
# Model
# ============================================================

class MultiTaskSkinModel(nn.Module):
    """
    Args:
        feat_dim     : 백본 출력 차원
        active_labels: LABEL_REGISTRY 중 실제 사용할 라벨 컬럼명 리스트
        trunk_hidden : trunk 은닉 차원 (head 입력 차원이 됨)
    """

    def __init__(
        self,
        feat_dim: int,
        active_labels: List[str],
        trunk_hidden: int = 512,
        trunk_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.trunk_hidden = trunk_hidden
        self.active_labels = list(active_labels)
        self.label_meta = {l: LABEL_REGISTRY[l] for l in self.active_labels}

        # 필요한 trunk만 빌드
        needed_trunks = set()
        for l in self.active_labels:
            fp = self.label_meta[l]["facepart"]
            needed_trunks.add(FACEPART_TO_TRUNK[fp])
        self.trunks = nn.ModuleDict()
        for tk in needed_trunks:
            in_d = feat_dim + (1 if tk in SYMMETRIC_TRUNKS else 0)
            self.trunks[tk] = MLPTrunk(in_d, trunk_hidden, dropout, trunk_layers)

        # 그룹별 head 하나씩 (좌/우 라벨은 동일 group → 자동 공유)
        self.heads = nn.ModuleDict()
        for l in self.active_labels:
            meta = self.label_meta[l]
            g = meta["group"]
            if g in self.heads:
                continue
            if meta["type"] == "ordinal":
                self.heads[g] = CORNHead(trunk_hidden, meta["K"])
            elif meta["type"] == "count":
                self.heads[g] = PoissonHead(trunk_hidden)
            elif meta["type"] == "reg":
                self.heads[g] = GaussianRegHead(trunk_hidden, 1)
            else:
                raise ValueError(f"Unknown type for label {l}: {meta['type']}")

    # ---- forward helpers ----

    def encode(self, features: torch.Tensor, facepart: int) -> torch.Tensor:
        """facepart에 맞는 trunk를 거쳐 표현 벡터를 만든다."""
        trunk_key = FACEPART_TO_TRUNK[facepart]
        if trunk_key in SYMMETRIC_TRUNKS:
            # left: l_eye=3 / l_cheek=5, right: r_eye=4 / r_cheek=6
            side = -1.0 if facepart in (3, 5) else +1.0
            side_col = torch.full(
                (features.shape[0], 1), side,
                dtype=features.dtype, device=features.device,
            )
            features = torch.cat([features, side_col], dim=1)
        return self.trunks[trunk_key](features)

    def head_forward(self, repr_: torch.Tensor, label: str):
        """label의 head를 거친 raw 출력."""
        meta = self.label_meta[label]
        head = self.heads[meta["group"]]
        return head(repr_)

    def head(self, label: str) -> nn.Module:
        return self.heads[self.label_meta[label]["group"]]
