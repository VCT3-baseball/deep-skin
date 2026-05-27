# -*- coding: utf-8 -*-
"""measurement_data.csv를 annotations_relative.csv에 병합.

LABEL_REGISTRY의 reg 라벨 (수분/탄력/거칠기)을 subject 단위로 붙여서
multivalue/train.py가 그대로 학습할 수 있는 CSV를 만든다.
"""
import argparse
from pathlib import Path
import pandas as pd

# 한글 컬럼 → LABEL_REGISTRY 이름 매핑
RENAME = {
    # 수분
    "수분_이마": "moisture_forehead",
    "수분_왼쪽볼": "moisture_l_cheek",
    "수분_오른쪽볼": "moisture_r_cheek",
    # 모공 개수
    "모공개수_왼쪽볼": "pore_count_l_cheek",
    "모공개수_오른쪽볼": "pore_count_r_cheek",
}
# 탄력 R0..R9, Q0..Q3 × 4 부위
for i in range(10):
    RENAME[f"탄력_왼쪽볼_R{i}"]   = f"R{i}_l_cheek"
    RENAME[f"탄력_오른쪽볼_R{i}"] = f"R{i}_r_cheek"
    RENAME[f"탄력_이마_R{i}"]    = f"R{i}_forehead"
    RENAME[f"탄력_턱_R{i}"]      = f"R{i}_chin"
for i in range(4):
    RENAME[f"탄력_왼쪽볼_Q{i}"]   = f"Q{i}_l_cheek"
    RENAME[f"탄력_오른쪽볼_Q{i}"] = f"Q{i}_r_cheek"
    RENAME[f"탄력_이마_Q{i}"]    = f"Q{i}_forehead"
    RENAME[f"탄력_턱_Q{i}"]      = f"Q{i}_chin"
# 거칠기 (눈가) — 컬럼명의 Rz=Rtm은 Rz로 단순화
_ROUGHNESS = [
    ("Ra", "Ra"), ("Rq", "Rq"), ("Rmax", "Rmax"), ("R3z", "R3z"),
    ("Rt", "Rt"), ("Rz=Rtm", "Rz"), ("Rp", "Rp"), ("Rv", "Rv"),
]
for src, dst in _ROUGHNESS:
    RENAME[f"주름_왼쪽눈가_{src}"]   = f"{dst}_l_eye"
    RENAME[f"주름_오른쪽눈가_{src}"] = f"{dst}_r_eye"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", default="face_data/annotations_relative.csv")
    p.add_argument("--measurements", default="face_data/measurement_data.csv")
    p.add_argument("--out", default="face_data/annotations_with_measurements.csv")
    args = p.parse_args()

    ann = pd.read_csv(args.annotations)
    meas = pd.read_csv(args.measurements)

    keep_cols = ["subject_no"] + [c for c in RENAME if c in meas.columns]
    missing = set(RENAME) - set(meas.columns)
    if missing:
        print(f"[warn] measurement에 없는 컬럼: {sorted(missing)}")
    meas = meas[keep_cols].rename(columns=RENAME)
    meas = meas.rename(columns={"subject_no": "subject_id"})
    meas = meas.drop_duplicates(subset=["subject_id"])

    merged = ann.merge(meas, on="subject_id", how="left")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    added = [c for c in meas.columns if c != "subject_id"]
    print(f"[done] rows={len(merged)}  added cols={added}")
    print(f"[coverage]")
    for c in added:
        nn = merged[c].notna().sum()
        print(f"  {c:24s} non-null = {nn}/{len(merged)} ({100*nn/len(merged):.1f}%)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
