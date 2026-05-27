# -*- coding: utf-8 -*-
"""
캐시된 DINOv3 특징을 로드하는 Dataset.

각 facepart 캐시 파일(.pt)은 다음 구조를 갖는다 (build_feature_cache.py 출력):
    config         : dict (facepart, feat_dim, k, ...)
    image_paths    : list[str]
    labels         : tensor (primary label - 무시하고 CSV에서 재조회)
    eval_features  : (N, D)        fp16   center crop, no aug
    flip_features  : (N, D)        fp16   horizontal flip
    aug_features   : (N, K, D)     fp16   K개 random aug views

학습 시: aug_features에서 한 view를 랜덤 샘플링 (in-memory augmentation).
평가 시: eval / flip / aug 평균을 TTA로 사용.
"""

import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CachedFacepartDataset(Dataset):
    """
    한 facepart의 캐시 파일 + CSV에서 모든 관련 라벨을 함께 제공한다.

    Args:
        cache_path: facepart 캐시 .pt 파일 경로
        annotation_df: 전체 annotation CSV DataFrame (이미 로드된 것)
        label_cols: 학습/평가에 사용할 라벨 컬럼명 리스트
        keep_image_paths: 사용할 image_path 부분집합 (split 적용). None이면 모두.
        mode: 'train' | 'eval'
        load_aug: 학습용으로 aug_features를 메모리에 로드할지. 평가만 한다면 False로 메모리 절약.
    """

    def __init__(
        self,
        cache_path: str,
        annotation_df: pd.DataFrame,
        label_cols: List[str],
        keep_image_paths: Optional[Iterable[str]] = None,
        mode: str = "train",
        load_aug: bool = True,
    ):
        assert mode in ("train", "eval")
        self.mode = mode
        self.cache_path = cache_path

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        self.config = cache["config"]
        self.facepart = int(self.config["facepart"])
        self.feat_dim = int(self.config["feat_dim"])

        self.image_paths: List[str] = list(cache["image_paths"])
        self.eval_features: torch.Tensor = cache["eval_features"]  # (N, D) fp16
        self.flip_features: torch.Tensor = cache["flip_features"]  # (N, D) fp16
        if load_aug:
            self.aug_features: torch.Tensor = cache["aug_features"]  # (N, K, D) fp16
            self.K = int(self.aug_features.shape[1])
        else:
            self.aug_features = None
            self.K = 0
        del cache

        N = len(self.image_paths)

        # --- CSV에서 라벨 추출 (벡터화: reindex로 cache 순서에 정렬) ---
        sub = annotation_df[annotation_df["facepart"] == self.facepart]
        # 동일 image_path가 두 번 이상 등장하지 않는다고 가정. 안전하게 drop_duplicates.
        sub = sub.drop_duplicates(subset="image_path").set_index("image_path")
        sub_reidx = sub.reindex(self.image_paths)

        self.label_cols = list(label_cols)
        self.labels = {}
        for col in self.label_cols:
            if col in sub_reidx.columns:
                # 문자열/NaN 혼합 안전 변환
                vals = pd.to_numeric(sub_reidx[col], errors="coerce").to_numpy(
                    dtype=np.float32, na_value=np.nan
                )
            else:
                vals = np.full(N, np.nan, dtype=np.float32)
            self.labels[col] = torch.from_numpy(vals)

        # --- split 마스크 ---
        if keep_image_paths is not None:
            keep_set = set(keep_image_paths)
            keep_mask_np = np.array([p in keep_set for p in self.image_paths], dtype=bool)
        else:
            keep_mask_np = np.ones(N, dtype=bool)

        # --- 최소 한 개 이상의 유효 라벨이 있는 샘플만 사용 ---
        any_valid = np.zeros(N, dtype=bool)
        for col, t in self.labels.items():
            any_valid |= ~np.isnan(t.numpy())
        usable_np = keep_mask_np & any_valid
        self.indices = np.where(usable_np)[0].astype(np.int64)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int):
        i = int(self.indices[idx])
        if self.mode == "train" and self.aug_features is not None:
            k = int(torch.randint(0, self.K, (1,)).item())
            feat = self.aug_features[i, k].float()
        else:
            feat = self.eval_features[i].float()

        sample = {
            "features": feat,
            "eval_features": self.eval_features[i].float(),
            "flip_features": self.flip_features[i].float(),
            "facepart": self.facepart,
            "image_path": self.image_paths[i],
            "labels": {col: t[i] for col, t in self.labels.items()},
        }
        # 평가 모드에서만 모든 aug view 반환 (TTA용)
        if self.mode == "eval" and self.aug_features is not None:
            sample["aug_features"] = self.aug_features[i].float()  # (K, D)
        return sample


def build_split(
    annotation_df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
    split_col: str = "split",
    subject_col: str = "subject_id",
):
    """
    Split 전략:
        1) CSV에 'split' 컬럼이 있으면 그대로 사용
        2) 'subject_id' 컬럼이 있으면 subject 단위 split (data leakage 방지)
        3) 그 외엔 image_path 단위 random split

    Returns:
        (train_paths, val_paths, test_paths): 각각 set[str]
    """
    if split_col in annotation_df.columns:
        train = set(annotation_df.loc[annotation_df[split_col] == "train", "image_path"])
        val = set(annotation_df.loc[annotation_df[split_col] == "val", "image_path"])
        test = set(annotation_df.loc[annotation_df[split_col] == "test", "image_path"])
        return train, val, test

    rng = np.random.default_rng(seed)
    if subject_col in annotation_df.columns:
        subjects = annotation_df[subject_col].dropna().unique()
        rng.shuffle(subjects)
        n = len(subjects)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        test_s = set(subjects[:n_test])
        val_s = set(subjects[n_test : n_test + n_val])
        train_s = set(subjects[n_test + n_val :])

        def paths_for(subset):
            return set(
                annotation_df.loc[annotation_df[subject_col].isin(subset), "image_path"]
            )

        return paths_for(train_s), paths_for(val_s), paths_for(test_s)

    paths = np.asarray(annotation_df["image_path"].unique())
    rng.shuffle(paths)
    n = len(paths)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = set(paths[:n_test])
    val = set(paths[n_test : n_test + n_val])
    train = set(paths[n_test + n_val :])
    return train, val, test


def detect_feat_dim(cache_pattern: str, faceparts: Iterable[int] = range(9)) -> int:
    """첫 번째 존재하는 캐시에서 feat_dim 자동 감지."""
    for fp in faceparts:
        p = cache_pattern.format(p=fp)
        if os.path.exists(p):
            c = torch.load(p, map_location="cpu", weights_only=False)
            d = int(c["config"]["feat_dim"])
            del c
            return d
    raise FileNotFoundError(
        f"No cache found matching pattern: {cache_pattern}"
    )
