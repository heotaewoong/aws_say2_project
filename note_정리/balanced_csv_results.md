# Balanced CSV 학습 결과 (5차)

## 실험 환경

- **모델**: DenseNet-121 (ImageNet pretrained)
- **이미지 크기**: 448×448
- **전처리**: CLAHE
- **인스턴스**: ml.g4dn.16xlarge (Spot)
- **Loss**: BCE + pos_weight (cap=5) + 마스킹(-1 제외)
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Epochs**: 15 / Early stopping patience=5
- **Batch size**: 16
- **Job 이름**: chexnet-mimic448-v1775538521

## 데이터

| 구분 | 파일 |
|------|------|
| Train | `s3://say2-2team-bucket/csv/mimic_cxr_balance1_train.csv` |
| Valid | `s3://say2-2team-bucket/csv/mimic_cxr_balance1_valid.csv` |

- 기존 방식(내부 8:2 분리, No Finding 5000개 캡)과 달리 **이미 균형잡힌 CSV를 그대로 사용**
- WeightedRandomSampler 미적용 (balanced CSV이므로)

---

## Epoch별 결과

| Epoch | mAUROC | F1 | Precision | Recall | val_loss |
|-------|--------|----|-----------|--------|----------|
| 1 | 0.7409 | 0.4557 | 0.3572 | 0.6887 | 0.9291 |
| 2 | 0.7528 | 0.4649 | 0.3763 | 0.6649 | 0.9149 |
| 3 | 0.7595 | 0.4727 | 0.3865 | 0.6687 | 0.9121 |
| **4** | **0.7635** | **0.4778** | **0.4088** | **0.6056** | **0.9016** |
| 5 | 0.7567 | 0.4658 | 0.3863 | 0.6279 | 0.9334 |
| 6 | 0.7564 | 0.4721 | 0.3864 | 0.6454 | 0.9753 |
| 7 | 0.7518 | 0.4679 | 0.3789 | 0.6618 | 0.9868 |
| 8 | 0.7567 | 0.4775 | 0.3939 | 0.6570 | 1.0204 |
| 9 | 0.7524 | 0.4717 | 0.3841 | 0.6420 | 1.0737 |

> Early stopping: epoch 9 (patience=5 소진), Best: epoch 4

---

## 4차 vs 5차(balanced) 비교

| 지표 | 4차 (220K, 내부분리) | **5차 (balanced CSV)** | 변화 |
|------|---------------------|----------------------|------|
| **mAUROC** | 0.7699 | 0.7635 | -0.006 |
| **F1** | 0.3983 | **0.4778** | **+0.080** |
| **Precision** | 0.3318 | **0.4088** | **+0.077** |
| **Recall** | 0.5248 | **0.6056** | **+0.081** |
| Best Epoch | 2/15 | 4/15 | - |

## 분석

### 긍정적 변화
- F1, Precision, Recall 모두 **+0.08 수준 대폭 향상**
- epoch 1부터 mAUROC 0.74로 안정적 시작
- Best epoch가 2→4로 늦어짐 (과적합 완화)

### 아쉬운 점
- mAUROC 0.7699 → 0.7635로 소폭 하락
- epoch 4 이후 val_loss 지속 상승 (여전히 과적합 경향)

---

## 모델 저장 위치

```
s3://say2-2team-bucket/models/mimic-only/chexnet-mimic448-v1775538521/output/model.tar.gz
```

- `chexnet_best.pth` — Best 모델 (epoch 4)
- `best_thresholds.json` — per-class threshold
- `training_history.json` — epoch별 전체 기록
