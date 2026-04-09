# 전처리 기법 및 실험 기록 (통합)

> `training_results.md`, `balanced_csv_results.md`, `binary_2class_results.md`, `3class_results.md`에 없는 내용 보완

---

## 실험 인프라

| 항목 | 값 |
|------|-----|
| **인스턴스** | **ml.g4dn.16xlarge** |
| GPU | NVIDIA T4 × 4 (64GB) |
| 과금 방식 | Spot Instance |
| 스크립트 | run_sagemaker.py / run_sagemaker-2class.py / run_sagemaker-3class.py 모두 동일 |

---

## 전처리 기법 목록

### 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)

```python
def apply_clahe(pil_img):
    img_np = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_np)
    return Image.fromarray(enhanced).convert('RGB')
```

- **목적**: 폐 구조 대비 향상 → 병변 특징 선명화
- **적용**: 1차 시도부터 전 실험에 공통 적용
- **효과**: 육안으로 폐 경계 및 음영 구분 향상

---

### 2. pos_weight (희귀 질환 가중치)

```python
label_vals = full_ds.df[LABELS].values
pos_counts = (label_vals == 1).sum(axis=0)
neg_counts = (label_vals == 0).sum(axis=0)
pw = np.clip(neg_counts / (pos_counts + 1e-6), 1, 10)  # 1차: cap=10, 이후: cap=5
pos_weight = torch.FloatTensor(pw).to(device)
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
```

| 차수 | pos_weight cap | 효과 |
|------|---------------|------|
| 1차 | 10배 | Recall 0.28 → 0.71 (과도하게 Recall 편향) |
| 2차~ | 5배 | Precision/Recall 균형 개선 |

---

### 3. -1 라벨 마스킹 처리

```python
# 변경 전 (잘못된 방식)
df[LABELS] = df[LABELS].fillna(0).replace(-1, 1)  # 불확실 → 양성으로 처리

# 변경 후 (올바른 방식)
df[LABELS] = df[LABELS].fillna(0)   # NaN → 0, -1은 유지
# 마스크: -1인 경우 loss 계산 제외

def masked_loss(outputs, labels, masks):
    loss = bce(outputs, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-6)
    return loss
```

- **문제**: -1을 양성(1)으로 학습 → 모델 혼란
- **해결**: -1 위치는 mask=0으로 설정 → 손실 계산에서 완전 제외

---

### 4. threshold 조정 (Global → Per-class)

#### 4-1. Global threshold 조정

```python
# 변경 전
def compute_metrics(preds, gts, threshold=0.5):

# 변경 후
def compute_metrics(preds, gts, threshold=0.3):  # 또는 0.4
```

| threshold | 효과 |
|-----------|------|
| 0.5 | Precision 높음, Recall 낮음 |
| 0.4 | 중간 균형 |
| 0.3 | Recall 높음, Precision 낮음 |

#### 4-2. Per-class Threshold (3차 시도)

각 질환마다 F1을 최대화하는 threshold를 0.20~0.80 범위에서 0.05 간격으로 탐색.
매 epoch마다 갱신, best 모델의 threshold를 `best_thresholds.json`으로 저장.

```
Fracture    → threshold 0.65 (희귀 → 확신할 때만 양성)
Edema       → threshold 0.35 (흔함 → 조금만 의심돼도 양성)
Pneumothorax→ threshold 0.55
```

---

### 5. WeightedRandomSampler (2차 시도)

```python
from torch.utils.data import WeightedRandomSampler

# 희귀 질환 샘플 → 더 자주 샘플링
# 정상(No Finding) → 희귀질환 수 × 3배로 언더샘플링
```

- **장점**: 데이터 버리지 않고 배치 내 희귀 질환 비율 증가
- **단점**: 같은 희귀 샘플이 반복 등장 가능

---

### 6. ASL Loss (Asymmetric Loss) — 실패

**도입 목적**: BCE의 false negative 문제 해결

```python
# gamma_neg=4, gamma_pos=1, clip=0.05
# 음성(정상): 확실한 정상의 loss를 강하게 억제 → Precision↑
# 양성(질환): 적당히 페널티 → Recall 유지
```

**결과**: loss 붕괴 (0.06~0.08 수준으로 급감)
→ 모델이 모든 예측을 극단값(0 또는 1)으로 출력
→ 실험 중단, BCE + pos_weight 방식으로 복귀

---

### 7. Focal Loss — 실패 (참고)

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()
```

**결과**: loss 붕괴 (γ=2 기준)
→ pos_weight와 병행 시 음성 샘플 억제 과도 → 학습 불안정

---

## 1차 시도 상세 기록

### 설정

| 항목 | 값 |
|------|-----|
| 데이터셋 | MIMIC-CXR JPG 448×448 |
| 데이터 양 | 약 10만장 → 2만장 (No Finding 캡 적용) |
| 데이터 밸런싱 | pos_weight cap=10, -1 마스킹 |
| batch_size | 16 |
| Epoch | 10 |
| 인스턴스 | ml.g4dn.xlarge (Spot) |
| Loss | BCE + pos_weight (cap=10) |
| threshold | 0.3 |

### 주요 적용 기법

1. **pos_weight** cap=10 (희귀 질환 최대 10배 가중치)
2. **-1 처리**: 0으로 변경 → 이후 마스킹 방식으로 개선
3. **threshold**: 0.5 → 0.3 (Recall 우선)
4. **CLAHE**: 적용

### 결과 비교

| 지표 | 이전 (10만장, threshold=0.5) | 1차 (2만장, pos_weight, threshold=0.3) |
|------|------------------------------|----------------------------------------|
| mAUROC | 0.8030 | 0.7749 |
| Recall | 0.28 | **0.71** |
| Precision | 0.51 | 0.23 |
| F1 | 0.31 | 0.35 |

### 분석

- **mAUROC 하락** (0.8030 → 0.7749): 데이터 10만 → 2만으로 감소 영향
- **Recall 급등** (0.28 → 0.71): pos_weight cap=10 + threshold=0.3 효과
- **Precision 폭락** (0.51 → 0.23): Recall 편향 과도 → 이후 cap=5로 조정
- **시사점**: 데이터 양 감소의 mAUROC 영향 > 전처리 개선 효과

---

## 실험별 전처리 기법 적용 현황

| 기법 | 1차 | 2차 | 3차 | 4차(ASL→실패) | 4차(220K) | 5차(balanced) |
|------|-----|-----|-----|--------------|-----------|---------------|
| CLAHE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| pos_weight | ✅ cap=10 | ✅ cap=5 | ✅ cap=5 | - | ✅ cap=5 | ✅ cap=5 |
| -1 마스킹 | 부분적 | ✅ | ✅ | - | ✅ | ✅ |
| threshold | 0.3 | 0.4 | per-class | - | per-class | per-class |
| WeightedSampler | ❌ | ✅ | ✅ | - | ✅ | ❌ |
| Loss | BCE | BCE | BCE | **ASL → 실패** | BCE | BCE |
| 데이터 | 2만 | 2만 | 2만 | - | **22만** | balanced CSV |

---

## 전처리 이외 모델 변경 시도

| 시도 | 내용 | 결과 |
|------|------|------|
| DenseNet-121 → ConvNeXt-L | 198M 파라미터, AdamW + CosineAnnealing | 5차(training_results.md) 진행 예정 |
| DenseNet-121 → DenseNet-169 | 파라미터 증가 | 제안 단계, 미실험 |
| CheXpert 데이터 추가 | 멀티 데이터셋 학습 | 제안 단계, 미실험 |
