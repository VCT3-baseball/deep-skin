"""CSV (xyxy) → YOLO 라벨(.txt) + train.txt / val.txt 생성.

사용법:
    python -m src.prepare_dataset
    python -m src.prepare_dataset --sanity 10   # 변환 검증 시각화 N장

산출물:
    {OUTPUT_DIR}/yolo_dataset/labels/<image_stem>.txt
    {OUTPUT_DIR}/yolo_dataset/train.txt
    {OUTPUT_DIR}/yolo_dataset/val.txt
    {OUTPUT_DIR}/yolo_dataset/dataset.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
import yaml

from .utils.config_bbox import (
    CLASS_NAMES, CSV_PATH, DATA_DIR, DATASET_YAML, LABELS_DIR,
    NUM_CLASSES, TRAIN_LIST, VAL_LIST, YOLO_DATASET_DIR,
)
from .utils.bbox_utils import clip01, is_valid_box, xyxy_to_yolo


def _normalize_path(p: str) -> Path:
    """CSV 의 image_path 는 Windows 스타일(\\) 이므로 OS 에 맞게 정규화."""
    return Path(p.replace("\\", "/"))


def _split_of(rel_path: Path) -> str | None:
    parts = rel_path.parts
    if not parts:
        return None
    if parts[0].lower().startswith("train"):
        return "train"
    if parts[0].lower().startswith("val"):
        return "val"
    return None


def build_labels(df: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    """이미지별 라벨 파일을 만들고 (train_imgs, val_imgs) 절대경로 리스트 반환."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    train_imgs: list[Path] = []
    val_imgs: list[Path] = []
    skipped_no_full = 0
    skipped_bad_box = 0

    for rel_path, group in df.groupby("image_path"):
        rel = _normalize_path(rel_path)
        split = _split_of(rel)
        if split is None:
            continue

        # facepart=0 행에서 이미지 크기 획득
        full_rows = group[group["facepart"] == 0]
        if full_rows.empty:
            skipped_no_full += 1
            continue
        full = full_rows.iloc[0]
        # CSV 는 xyxy 포맷이므로 full row 의 bbox_w/bbox_h 가 실제 이미지 (W, H)
        img_w = float(full["bbox_w"])
        img_h = float(full["bbox_h"])
        if img_w <= 0 or img_h <= 0:
            skipped_no_full += 1
            continue

        lines = []
        for _, row in group.iterrows():
            fp = int(row["facepart"])
            if fp == 0:
                continue
            x1, y1 = float(row["bbox_x"]), float(row["bbox_y"])
            x2, y2 = float(row["bbox_w"]), float(row["bbox_h"])
            if not is_valid_box(x1, y1, x2, y2):
                skipped_bad_box += 1
                continue
            cx, cy, w, h = xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h)
            cx, cy, w, h = clip01(cx), clip01(cy), clip01(w), clip01(h)
            if w <= 0 or h <= 0:
                skipped_bad_box += 1
                continue
            cls_id = fp - 1  # 1~8 → 0~7
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not lines:
            continue

        abs_img = (DATA_DIR / rel).resolve()
        label_path = LABELS_DIR / split / f"{rel.stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

        (train_imgs if split == "train" else val_imgs).append(abs_img)

    print(f"[prepare] train images: {len(train_imgs)}, val images: {len(val_imgs)}")
    print(f"[prepare] skipped (no full row): {skipped_no_full}, skipped (bad box): {skipped_bad_box}")
    return train_imgs, val_imgs


def write_lists(train_imgs: list[Path], val_imgs: list[Path]) -> None:
    YOLO_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LIST.write_text("\n".join(str(p) for p in train_imgs), encoding="utf-8")
    VAL_LIST.write_text("\n".join(str(p) for p in val_imgs), encoding="utf-8")
    print(f"[prepare] wrote {TRAIN_LIST} ({len(train_imgs)})")
    print(f"[prepare] wrote {VAL_LIST} ({len(val_imgs)})")


def write_dataset_yaml() -> None:
    # Ultralytics 는 이미지 경로의 '/images/' 를 '/labels/' 로 치환해 라벨을 찾는다.
    # write_lists_with_label_mirror() 로 이미지 미러를 YOLO_DATASET_DIR/images/<split>/
    # 에 두고, 라벨은 YOLO_DATASET_DIR/labels/<split>/ 에 두어 이 규칙을 만족시킨다.
    yaml_dict = {
        "path": str(YOLO_DATASET_DIR),
        "train": str(TRAIN_LIST),
        "val": str(VAL_LIST),
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    DATASET_YAML.write_text(yaml.safe_dump(yaml_dict, sort_keys=False), encoding="utf-8")
    print(f"[prepare] wrote {DATASET_YAML}")


def write_lists_with_label_mirror(train_imgs: list[Path], val_imgs: list[Path]) -> None:
    """이미지를 YOLO_DATASET_DIR/images/<split>/<stem>.jpg 로 hardlink 한다.

    Ultralytics 가 이미지 경로의 '/images/' → '/labels/' 치환으로 라벨을 찾도록 하기
    위함. 라벨은 이미 LABELS_DIR/<split>/<stem>.txt 에 작성되어 있다.
    Hardlink 실패 시 (서로 다른 볼륨 등) copy 로 폴백.
    """
    import os, shutil
    images_dir = YOLO_DATASET_DIR / "images"
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)

    def _link_to(target: Path, link: Path) -> None:
        if link.exists() or link.is_symlink():
            return
        # 1) symlink: Linux (Kaggle) 에서 cross-mount 도 동작
        try:
            os.symlink(target, link)
            return
        except (OSError, NotImplementedError):
            pass
        # 2) hardlink: 같은 파일시스템(Windows NTFS 로컬)에서 동작
        try:
            os.link(target, link)
            return
        except OSError:
            pass
        # 3) 최후: 복사 (디스크 비용 발생)
        shutil.copy2(target, link)

    new_train, new_val = [], []
    for src in train_imgs:
        dst = images_dir / "train" / src.name
        _link_to(src, dst)
        new_train.append(dst)
    for src in val_imgs:
        dst = images_dir / "val" / src.name
        _link_to(src, dst)
        new_val.append(dst)

    TRAIN_LIST.write_text("\n".join(str(p) for p in new_train), encoding="utf-8")
    VAL_LIST.write_text("\n".join(str(p) for p in new_val), encoding="utf-8")
    print(f"[prepare] mirrored {len(new_train)} train / {len(new_val)} val images into {images_dir}")


def sanity_check(n: int, val_imgs: list[Path]) -> None:
    """무작위 N장에 대해 라벨을 다시 박스로 그려 PNG 저장."""
    from PIL import Image
    from .utils.viz import draw_boxes, read_yolo_label

    out_dir = YOLO_DATASET_DIR / "_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = random.sample(val_imgs, min(n, len(val_imgs)))
    for img_path in samples:
        label_path = LABELS_DIR / "val" / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            W, H = im.size
        boxes = read_yolo_label(label_path, W, H)
        draw_boxes(
            img_path, boxes, CLASS_NAMES,
            save_path=out_dir / f"{img_path.stem}.png",
            title=img_path.name,
        )
    print(f"[prepare] sanity images written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sanity", type=int, default=0, help="검증 시각화 장수")
    parser.add_argument("--no-mirror", action="store_true",
                        help="이미지 미러링(hardlink) 생략 — 이미 한 번 만들었을 때")
    args = parser.parse_args()

    print(f"[prepare] reading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"[prepare] {len(df):,} rows")

    train_imgs, val_imgs = build_labels(df)
    if args.no_mirror:
        TRAIN_LIST.write_text("\n".join(str(p) for p in train_imgs), encoding="utf-8")
        VAL_LIST.write_text("\n".join(str(p) for p in val_imgs), encoding="utf-8")
    else:
        write_lists_with_label_mirror(train_imgs, val_imgs)
    write_dataset_yaml()

    if args.sanity:
        sanity_check(args.sanity, val_imgs)


if __name__ == "__main__":
    main()
