"""YOLO 학습 진입점.

사용법:
    python -m src.train                       # 기본 (yolov8s, imgsz=1280, 50ep)
    python -m src.train --model yolo11s.pt --epochs 30
    python -m src.train --sanity              # 빠른 sanity (imgsz=640, 3ep)
"""
from __future__ import annotations

import argparse

from ultralytics import YOLO

from .utils.config_bbox import DATASET_YAML, OUTPUT_DIR


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8s.pt", help="사전학습 가중치 (예: yolov8s.pt, yolo11s.pt)")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0", help="'0', '0,1', 또는 'cpu'")
    p.add_argument("--name", default="face_parts_v1")
    p.add_argument("--sanity", action="store_true",
                   help="imgsz=640, epochs=3 로 빠르게 sanity-check")
    args = p.parse_args()

    if args.sanity:
        args.imgsz = 640
        args.epochs = 3
        args.name = "sanity"

    model = YOLO(args.model)
    model.train(
        data=str(DATASET_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR / "runs"),
        name=args.name,
        patience=10,
        cos_lr=True,
        # 좌·우 부위 클래스가 구분되므로 좌우 반전 금지
        fliplr=0.0,
        flipud=0.0,
        # 얼굴 박스 특성상 mosaic 은 약하게
        mosaic=0.5,
        degrees=10,
        translate=0.05,
        scale=0.2,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        plots=True,
    )


if __name__ == "__main__":
    main()
