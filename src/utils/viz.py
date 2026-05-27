"""BBox 시각화 유틸 (matplotlib 사용)."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

# 8 부위 고정 색상 (forehead, glabella, l_eye, r_eye, l_cheek, r_cheek, lips, chin)
CLASS_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9A6324",
]


def draw_boxes(
    image_path: str | Path,
    boxes: Iterable[Tuple[int, float, float, float, float]],
    class_names: list[str],
    save_path: str | Path | None = None,
    title: str | None = None,
) -> None:
    """이미지에 박스 그리기.

    boxes: (class_id, x1, y1, x2, y2) 픽셀 좌표 튜플 iterable.
    save_path 가 None 이면 plt.show().
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(img)

    for class_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS[class_id % len(CLASS_COLORS)]
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, y1 - 8, class_names[class_id],
            color="white", fontsize=10,
            bbox=dict(facecolor=color, edgecolor="none", pad=2),
        )

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def read_yolo_label(label_path: str | Path, img_w: int, img_h: int):
    """YOLO 라벨 .txt 를 픽셀 xyxy 튜플 리스트로 변환."""
    from .bbox_utils import yolo_to_xyxy

    out = []
    p = Path(label_path)
    if not p.exists():
        return out
    for line in p.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, w, h, img_w, img_h)
        out.append((cls, x1, y1, x2, y2))
    return out
