"""학습된 YOLO 모델로 이미지에서 8개 얼굴 부위 bbox 추출.

사용법:
    python -m src.predict --weights output/runs/face_parts_v1/weights/best.pt \
                          --source path/to/image_or_dir --save-json --save-viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from .utils.config_bbox import CLASS_NAMES, OUTPUT_DIR
from .utils.viz import draw_boxes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--source", required=True, help="이미지 파일 또는 디렉토리")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--out", default=str(OUTPUT_DIR / "predictions"))
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-viz", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    for r in results:
        img_path = Path(r.path)
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4)
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)  # (N,)
        confs = r.boxes.conf.cpu().numpy()  # (N,)

        detections = []
        for (x1, y1, x2, y2), cid, cf in zip(boxes_xyxy, cls_ids, confs):
            detections.append({
                "class_id": int(cid),
                "class_name": CLASS_NAMES[int(cid)],
                "confidence": float(cf),
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            })

        if args.save_json:
            (out_dir / f"{img_path.stem}.json").write_text(
                json.dumps({"image": str(img_path), "detections": detections},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        if args.save_viz:
            viz_boxes = [(d["class_id"], *d["bbox_xyxy"]) for d in detections]
            draw_boxes(
                img_path, viz_boxes, CLASS_NAMES,
                save_path=out_dir / f"{img_path.stem}_pred.png",
                title=f"{img_path.name}  (n={len(detections)})",
            )

    print(f"[predict] wrote results to {out_dir}")


if __name__ == "__main__":
    main()
