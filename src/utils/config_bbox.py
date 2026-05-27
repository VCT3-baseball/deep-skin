"""환경 감지 및 공용 상수.

로컬(Windows)과 Kaggle 양쪽에서 동일하게 import 되도록 작성한다.
"""
from __future__ import annotations

import os
from pathlib import Path

IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    # Kaggle Dataset slug 는 업로드 시 확정한다. 필요시 환경변수로 오버라이드.
    DATA_DIR = Path(os.environ.get("FACE_BBOX_DATA_DIR", "/kaggle/input/face-bbox"))
    OUTPUT_DIR = Path("/kaggle/working")
else:
    # Local development paths relative to project root
    ROOT = Path(__file__).parents[2]
    DATA_DIR = ROOT / "data"
    OUTPUT_DIR = ROOT / "results" / "bbox"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# YOLO 라벨/리스트 파일은 OUTPUT_DIR/yolo_dataset/ 아래에 둔다.
YOLO_DATASET_DIR = OUTPUT_DIR / "yolo_dataset"
LABELS_DIR = YOLO_DATASET_DIR / "labels"
TRAIN_LIST = YOLO_DATASET_DIR / "train.txt"
VAL_LIST = YOLO_DATASET_DIR / "val.txt"
DATASET_YAML = YOLO_DATASET_DIR / "dataset.yaml"

CSV_PATH = DATA_DIR / "annotations_bbox_only.csv"

# CSV 의 facepart 코드 1~8 → YOLO class id 0~7
# (facepart=0 'full' 은 학습 대상에서 제외)
CLASS_NAMES = [
    "forehead",   # 1
    "glabella",   # 2
    "l_eye",      # 3
    "r_eye",      # 4
    "l_cheek",    # 5
    "r_cheek",    # 6
    "lips",       # 7
    "chin",       # 8
]
NUM_CLASSES = len(CLASS_NAMES)
