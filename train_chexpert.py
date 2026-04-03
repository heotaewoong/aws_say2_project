"""
==========================================================
  CheXpert 데이터로 CheXNet 학습하기
  (train_chexpert.py)
==========================================================

Kaggle에서 CheXpert-v1.0-small 다운로드 후 사용:
  https://www.kaggle.com/datasets/ashery/chexpert

사용법:
  1. 데이터를 data/CheXpert-v1.0-small/ 에 넣기
  2. python train_chexpert.py 실행

필요 라이브러리:
  pip install torch torchvision pandas numpy scikit-learn
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# ─────────────────────────────────────────────
# 설정 (여기만 수정하면 됨!)
# ─────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Kaggle에서 받은 CheXpert 데이터 경로
DATA_ROOT = os.path.join(CURRENT_DIR, "data", "CheXpert-v1.0-small")
# 또는 다른 곳에 풀었으면 직접 경로 지정:
# DATA_ROOT = "/Users/skku_mac08/Downloads/CheXpert-v1.0-small"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
VALID_CSV = os.path.join(DATA_ROOT, "valid.csv")

# 14개 라벨 (이 프로젝트와 동일한 CheXpert 라벨)
LABEL_ORDER = [
    "Atelectasis",                  # 0: 무기폐
    "Cardiomegaly",                 # 1: 심장비대
    "Consolidation",                # 2: 폐 경화
    "Edema",                        # 3: 폐부종
    "Enlarged Cardiomediastinum",   # 4: 종격동 확장
    "Fracture",                     # 5: 골절
    "Lung Lesion",                  # 6: 폐 병변
    "Lung Opacity",                 # 7: 폐 혼탁
    "No Finding",                   # 8: 정상
    "Pleural Effusion",             # 9: 흉막삼출
    "Pleural Other",                # 10: 기타 흉막 이상
    "Pneumonia",                    # 11: 폐렴
    "Pneumothorax",                 # 12: 기흉
    "Support Devices"               # 13: 의료기기
]

# 하이퍼파라미터
BATCH_SIZE = 32     # 메모리 부족하면 16으로 줄이기
NUM_EPOCHS = 10     # 에폭 수
LEARNING_RATE = 1e-4
NUM_WORKERS = 4     # 데이터 로딩 스레드 수


# ─────────────────────────────────────────────
# 1. 데이터셋 클래스
# ─────────────────────────────────────────────
class CheXpertDataset(Dataset):
    """CheXpert CSV + 이미지를 로드하는 데이터셋"""

    def __init__(self, csv_path, data_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.transform = transform

        # 라벨 전처리:
        #   NaN → 0 (라벨 없음 = 음성 처리)
        #   -1  → 1 (불확실 = 양성 처리, U-Ones 방식)
        # U-Ones: CheXpert 논문에서 가장 좋은 성능을 보인 방식
        self.df[LABEL_ORDER] = self.df[LABEL_ORDER].fillna(0).replace(-1, 1)

        # 이미지 경로 컬럼 확인 (CheXpert CSV의 첫 번째 컬럼 = Path)
        self.path_col = self.df.columns[0]  # 보통 'Path'

        print(f"  📂 데이터 로드: {len(self.df)}장 ({csv_path})")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 이미지 경로 구성
        # CheXpert CSV의 Path 컬럼: "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
        img_rel_path = row[self.path_col]

        # data_root가 CheXpert-v1.0-small 폴더이므로,
        # Path에서 "CheXpert-v1.0-small/" 부분 제거
        img_rel_path = img_rel_path.replace("CheXpert-v1.0-small/", "")
        img_path = os.path.join(self.data_root, img_rel_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 손상된 이미지 → 다음 이미지로 대체
            return self.__getitem__((idx + 1) % len(self))

        if self.transform:
            image = self.transform(image)

        labels = torch.FloatTensor(row[LABEL_ORDER].values.astype(float))
        return image, labels


# ─────────────────────────────────────────────
# 2. 학습 함수
# ─────────────────────────────────────────────
def train():
    # ── 데이터 존재 확인 ──
    if not os.path.exists(TRAIN_CSV):
        print("❌ 학습 데이터를 찾을 수 없습니다!")
        print(f"   찾은 경로: {TRAIN_CSV}")
        print()
        print("📦 데이터 다운로드 방법:")
        print("   1. https://www.kaggle.com/datasets/ashery/chexpert 접속")
        print("   2. Kaggle 가입 → Download 클릭")
        print(f"   3. 압축 해제 후 '{DATA_ROOT}' 에 넣기")
        print()
        print("   폴더 구조가 이렇게 되어야 합니다:")
        print(f"   {DATA_ROOT}/")
        print("   ├── train.csv")
        print("   ├── valid.csv")
        print("   ├── train/")
        print("   │   └── patient00001/...")
        print("   └── valid/")
        return

    # ── 디바이스 설정 ──
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac Apple Silicon
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")

    print(f"🖥️  디바이스: {device}")

    # ── 데이터 전처리 ──
    # 논문 기반 데이터 증강 (Medical Image Augmentation Review, Springer 2023)
    #   ❌ RandomHorizontalFlip 제거: Dextrocardia(우심증) 생성 위험
    #   ⚠️ Rotation ±10→±5: 임상 관찰 범위 내로 축소
    #   ✅ RandomCrop: 약한 크롭(256→224, 12.5%)은 안전
    #   ✅ Brightness: [-0.1, 0.1] 범위 내 권장
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),               # 약한 위치 변동 (안전)
        transforms.RandomRotation(5),             # ±5도 (임상 범위)
        transforms.ColorJitter(brightness=0.1),   # 밝기만, 제한적
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ── 데이터셋 & 데이터로더 ──
    print("\n📊 데이터 준비 중...")
    train_ds = CheXpertDataset(TRAIN_CSV, DATA_ROOT, train_transform)
    val_ds   = CheXpertDataset(VALID_CSV, DATA_ROOT, val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # ── 모델 초기화 (ImageNet pretrained → 14개 출력으로 교체) ──
    print("\n🏗️  모델 초기화...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    model = models.densenet121(weights='IMAGENET1K_V1')
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    model = model.to(device)
    print(f"  DenseNet-121 → 14개 출력 (CheXpert 라벨)")

    # ── 손실 함수 & 옵티마이저 ──
    # 클래스 불균형 해결: 희귀 질환에 높은 가중치 부여
    pos_counts = (train_ds.df[LABEL_ORDER] == 1).sum()
    neg_counts = len(train_ds.df) - pos_counts
    pw = (neg_counts / (pos_counts + 1e-6)).values
    pos_weight = torch.FloatTensor(pw).to(device)
    print(f"  📊 pos_weight 적용 (희귀 질환 가중치 강화)")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    # ── 학습 루프 ──
    print(f"\n🚀 학습 시작! (에폭: {NUM_EPOCHS}, 배치: {BATCH_SIZE})")
    print("=" * 65)

    best_auroc = 0.0
    save_path = os.path.join(CURRENT_DIR, "models", "chexnet_mimic_best.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 진행 상황 출력 (100배치마다)
            if (i + 1) % 100 == 0:
                print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation (AUROC 계산) ---
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = torch.sigmoid(model(imgs))
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.numpy())

        val_preds  = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)

        # 14개 라벨 각각의 AUROC 계산
        auroc_per_label = {}
        for c in range(14):
            if len(np.unique(val_labels[:, c])) > 1:
                score = roc_auc_score(val_labels[:, c], val_preds[:, c])
                auroc_per_label[LABEL_ORDER[c]] = score

        mean_auroc = np.mean(list(auroc_per_label.values())) if auroc_per_label else 0.0

        # 결과 출력
        print(f"\n  ✅ Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"     Train Loss: {avg_train_loss:.4f} | mAUROC: {mean_auroc:.4f}")

        # 라벨별 AUROC (5에폭마다 상세 출력)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"     {'라벨':<35} {'AUROC':>7}")
            print(f"     {'─' * 45}")
            for label, score in sorted(auroc_per_label.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int((score - 0.5) * 30)
                print(f"     {label:<35} {score:.4f} {bar}")

        scheduler.step(mean_auroc)

        # Best 모델 저장
        if mean_auroc > best_auroc:
            best_auroc = mean_auroc
            torch.save(model.state_dict(), save_path)
            print(f"     💾 Best 모델 저장! mAUROC: {mean_auroc:.4f} → {save_path}")

        print()

    # ── 최종 결과 ──
    print("=" * 65)
    print(f"  🎉 학습 완료!")
    print(f"  📊 최종 Best mAUROC: {best_auroc:.4f}")
    print(f"  💾 저장 위치: {save_path}")
    print(f"\n  다음 단계: python eval_chexnet_14label.py --demo")
    print("=" * 65)


if __name__ == "__main__":
    train()
