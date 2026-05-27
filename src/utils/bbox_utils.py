"""BBox 좌표 변환 유틸.

CSV 의 bbox 컬럼은 이름이 `bbox_x, bbox_y, bbox_w, bbox_h` 지만
실제 값은 **xyxy(x1, y1, x2, y2)** 포맷이다 (full=0,0,2136,3216 이 이미지 크기와 일치).
이 모듈의 모든 함수는 xyxy 를 입력으로 받는다.
"""
from __future__ import annotations

from typing import Tuple


def xyxy_to_yolo(
    x1: float, y1: float, x2: float, y2: float,
    img_w: float, img_h: float,
) -> Tuple[float, float, float, float]:
    """xyxy 절대 좌표 → YOLO (cx, cy, w, h) 정규화 좌표."""
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def is_valid_box(x1: float, y1: float, x2: float, y2: float) -> bool:
    return x2 > x1 and y2 > y1


def yolo_to_xyxy(
    cx: float, cy: float, w: float, h: float,
    img_w: float, img_h: float,
) -> Tuple[float, float, float, float]:
    """YOLO 정규화 좌표 → 원본 픽셀 xyxy."""
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2
