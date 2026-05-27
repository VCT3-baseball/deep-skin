# Cached-feature Multi-task Skin Analysis Trainer

`build_feature_cache.py`가 만들어 둔 DINOv3 캐시(`feature_cache/feat_*_part*_m*_k*.pt`)를 사용해 multi-task 모델을 학습한다. 백본은 재추론하지 않는다.

## 파일 구성

원래 `src/` 바로 아래에 두려 했으나 파일 수가 많아져 `src/multivalue/` 하위 폴더로 분리했다. 실행은 프로젝트 루트에서 한다.

```
src/multivalue/
├── skin_data.py     CachedFacepartDataset, split helpers
├── skin_heads.py    CORNHead, PoissonHead, GaussianRegHead + losses + UncertaintyWeighting
├── skin_metrics.py  QWK, macro_F1, MAE, Pearson r, R²
├── skin_model.py    MultiTaskSkinModel, LABEL_REGISTRY
└── train.py         학습/검증 메인
```

## 빠른 사용법

프로젝트 루트(`face_cnn/`)에서 실행한다. 그래야 `feature_cache/`, `face_data/` 상대 경로가 그대로 맞는다.

```bash
python src/multivalue/train.py \
  --cache_pattern feature_cache/feat_vith_part{p}_m15_k16.pt \
  --csv face_data/annotations_relative.csv \
  --output_dir runs/vith_v1 \
  --epochs 50 --batch_size 128 --lr 1e-3
```

`{p}` 자리에 facepart id (0~8)가 들어간다. `--labels`를 명시하지 않으면 LABEL_REGISTRY에 있는 컬럼 중 CSV에 실제로 존재하는 것을 모두 학습한다.

특정 라벨만 학습하려면:
```bash
python src/multivalue/train.py ... --labels forehead_wrinkle l_cheek_pore r_cheek_pore acne_count
```

## 설계 결정 (캐시 제약 반영)

원본 설계안 대비 캐시 구조 때문에 바뀐 점:

- **Region-aware extractor는 cross-attention 대신 facepart crop으로 대체**됨. 캐시는 CLS feature만 저장하므로 patch token이 없다. 대신 build_feature_cache.py가 이미 facepart마다 bbox crop을 따로 추론한 결과를 갖고 있어서, region-localized feature가 사실상 확보된 상태.
- **Acne density map은 사용 불가** (patch token 없음). Poisson regression head만 사용.
- 그 외 — CORN ordinal head, Gaussian NLL, 좌/우 공유 trunk + side embedding, uncertainty weighting, feature-space TTA(eval+flip+aug 평균) — 모두 그대로 적용.

## 라벨 추가하기

CSV에 새 컬럼이 있다면 `skin_model.py`의 `LABEL_REGISTRY`에 항목 한 줄 추가:

```python
"my_new_label": {"facepart": 5, "group": "my_group", "side": None, "type": "ordinal", "K": 4},
```

`group`이 같은 라벨끼리는 자동으로 head를 공유한다 (좌/우 대칭에 활용).

## 학습 흐름 한눈에

1. CSV에서 train/val/test split (subject_id 컬럼 있으면 subject 단위, 아니면 image 단위)
2. facepart마다 `CachedFacepartDataset` 빌드 — 캐시의 image_path 순서대로 CSV 라벨 reindex
3. round-robin으로 facepart batch를 하나씩 forward — facepart마다 trunk가 달라서 같은 step에 섞을 수 없음
4. 손실은 `UncertaintyWeighting`이 자동 균형
5. 매 epoch 끝에 TTA validation, primary score(QWK/r 평균)로 best 저장

## 자주 마주칠 함정

- `cache_pattern`의 `{p}`를 따옴표로 감싸지 않으면 셸이 brace를 해석한다. 큰따옴표 권장.
- ViT-7B 같이 큰 모델 캐시는 RAM을 많이 먹는다. 메모리 부족하면 `CachedFacepartDataset(load_aug=False)`로 빌드하거나 `--no_tta`를 쓴다 (이때 aug feature를 로드하지 않음).
- 좌/우 라벨을 둘 다 LABEL_REGISTRY에 등록해야 weight sharing이 발동한다. 한쪽만 학습할 거면 side embedding이 작동하긴 하지만 공유 효과는 없다.
- CSV에 같은 (image_path, facepart) 조합이 중복 행으로 있으면 첫 번째 행만 사용. `drop_duplicates`로 안전 처리.
