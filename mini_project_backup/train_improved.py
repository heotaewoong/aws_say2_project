"""
============================================================
🚀 train_improved.py – V2보다 성능이 좋은 학습 코드
============================================================

🎯 이 파일이 하는 일 (초보자용 설명):
    이 파일은 "AI 의사의 눈"을 훈련시키는 코드입니다.
    
    비유로 설명하면:
    - 의대생에게 수만 장의 흉부 엑스레이 사진을 보여주면서
    - "이 사진에서는 폐렴이 보여", "이건 정상이야" 라고 가르치는 과정입니다.
    
    V2와의 차이점 (왜 더 좋은가):
    1. [데이터 증강] 같은 사진을 뒤집고, 회전해서 다양한 각도로 배움
       → 실제로 사진이 약간 기울어져도 잘 찾아냄
    2. [EfficientNet] 더 똑똑한 AI 모델을 사용
       → DenseNet(2017년)보다 EfficientNet(2019년)이 더 정확
    3. [Weighted Loss] 희귀한 병을 더 열심히 배움
       → V2는 흔한 병과 희귀 병을 같은 비중으로 학습 (문제!)
    4. [Cosine Annealing] 학습 속도를 점점 줄여서 세밀하게 배움
       → V2는 갑자기 학습 속도를 뚝 떨어뜨림 (덜 정밀)
    5. [Early Stopping] 과적합(너무 외워버리는 것) 방지
       → V2에는 이 기능이 없어서 너무 오래 학습하면 오히려 성능 저하

사용법:
    python train_improved.py
    
    ⚠️ 필요한 것:
    - CheXpert 또는 MIMIC-CXR 데이터셋 (data/ 폴더에)
    - GPU 또는 Apple Silicon Mac (MPS)
    - pip install torch torchvision scikit-learn pandas
============================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import ast
import numpy as np
from sklearn.metrics import roc_auc_score

# ============================================================
# 📋 설정값 (여기만 바꾸면 됨!)
# ============================================================
CONFIG = {
    # --- 데이터 경로 ---
    "img_root": "data",                              # 이미지 최상위 폴더
    "train_csv": "data/mimic_cxr_aug_train.csv",     # 학습용 CSV
    "val_csv": "data/mimic_cxr_aug_validate.csv",    # 검증용 CSV
    "chexpert_csv": "data/mimic-cxr-2.0.0-chexpert.csv",  # 레이블 CSV
    
    # --- 학습 설정 ---
    "model_name": "efficientnet_b4",  # 🔥 V2: densenet121 → 개선: efficientnet_b4
    "batch_size": 16,                 # 메모리 부족 시 8로 낮추세요
    "epochs": 20,                     # 🔥 V2: 10 → 개선: 20 (Early Stopping이 있으니 안전)
    "learning_rate": 3e-4,            # 🔥 V2: 1e-4 → 개선: 3e-4 (EfficientNet에 적합)
    "patience": 5,                    # 🔥 [새 기능] 5번 연속 개선 없으면 중단
    
    # --- 저장 ---
    "save_path": "best_model_v3.pth",
}

# 14개 질환 라벨 (CheXpert 기준)
LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]


# ============================================================
# 📁 데이터 준비 (V2와 동일 로직)
# ============================================================
def prepare_mimic_df(aug_csv_path, chexpert_csv_path, img_root):
    """MIMIC-CXR CSV를 읽어서 [이미지 경로, 라벨] DataFrame으로 변환"""
    labels_df = pd.read_csv(chexpert_csv_path)
    labels_df[LABEL_ORDER] = labels_df[LABEL_ORDER].fillna(0).replace(-1, 1)
    
    aug_df = pd.read_csv(aug_csv_path)
    flat_data = []
    missing_count = 0

    print(f"🔍 '{aug_csv_path}' 데이터 파싱 중...")
    for _, row in aug_df.iterrows():
        for view_col in ['AP', 'PA']:
            raw_string = str(row[view_col])
            if raw_string == 'nan':
                continue
            try:
                img_list = ast.literal_eval(raw_string)
                for img_path in img_list:
                    if 'p10' not in img_path:
                        continue
                    img_full_path = os.path.join(img_root, img_path)
                    if not os.path.exists(img_full_path):
                        missing_count += 1
                        continue
                    study_id = int(img_path.split('/')[-2][1:])
                    label_row = labels_df[labels_df['study_id'] == study_id]
                    if not label_row.empty:
                        flat_data.append({
                            'path': img_path,
                            'labels': label_row[LABEL_ORDER].values[0]
                        })
            except:
                continue

    final_df = pd.DataFrame(flat_data)
    print(f"✅ 파싱 완료: {len(final_df)}장 확보 ({missing_count}장 제외)")
    return final_df


# ============================================================
# 🖼️ 데이터셋 클래스
# ============================================================
class MedicalImageDataset(Dataset):
    """
    초보자 설명:
    PyTorch에게 "데이터를 이렇게 읽어라"고 알려주는 클래스입니다.
    학습할 때 이미지를 한 장씩 불러오고, 전처리(크기 조정, 색상 변환 등)를 적용합니다.
    """
    def __init__(self, df, img_root, transform=None):
        self.df = df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row['path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            # 손상된 이미지 → 다음 이미지로 대체
            return self.__getitem__((idx + 1) % len(self))
        
        label = torch.FloatTensor(row['labels'])
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 🔥 [개선 1] 데이터 증강 (V2에는 없었음!)
# ============================================================
"""
초보자 설명:
    "데이터 증강"이란 같은 사진을 여러 방식으로 변형해서
    AI가 더 다양한 상황에서도 잘 작동하도록 만드는 기법입니다.
    
    예시:
    - 원본 사진 1장 → 뒤집기, 회전, 밝기 변화 등으로 5장처럼 활용
    - 마치 의대생에게 "사진이 약간 기울어져도 폐렴은 폐렴이야"라고 가르치는 것
    
    V2는 이걸 안 했습니다 → 같은 사진만 반복 학습 → 새 사진에 약함
    
    ⚠️ 의료 영상 증강 시 주의사항:
    - ❌ 좌우 반전(HorizontalFlip) 금지!
      → 심장은 왼쪽에 있는데, 반전하면 오른쪽으로 이동
      → AI가 "우심증(Dextrocardia)"으로 오인할 수 있음
      → Cardiomegaly 판단에 치명적 오류 발생
    - ❌ 과도한 회전 금지! (±5도 이내로 제한)
      → 흉부 X-ray는 환자가 정면을 보고 촬영
      → 10도 이상 회전하면 갈비뼈/폐 경계가 왜곡됨
      → 실제 임상에서는 ±3~5도 오차만 발생하므로 이 범위가 현실적
    
    ✅ 안전한 증강: 밝기 변화, 약간의 이동, 가우시안 블러, 랜덤 크롭
"""
train_transform = transforms.Compose([
    transforms.Resize(280),                          # 약간 크게 만든 뒤
    transforms.RandomCrop(224),                      # 랜덤 위치에서 224x224 잘라냄
    # ❌ RandomHorizontalFlip 제거 – 흉부 X-ray에서 심장 위치가 바뀌면 안 됨!
    transforms.RandomRotation(5),                    # ±5도만 (의료 영상 안전 범위)
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 밝기/대비 약간 변화
    transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),  # 아주 약간 이동
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # 약간 흐리게 (노이즈 강건성)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.06)),  # 아주 작은 영역만 가림
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============================================================
# 🏗️ [개선 2] 모델 생성 (EfficientNet-B4)
# ============================================================
def create_model(model_name="efficientnet_b4", num_classes=14):
    """
    초보자 설명:
        AI 모델은 "뇌"라고 생각하면 됩니다.
        
        V2는 DenseNet-121 (2017년 모델)을 사용했는데,
        EfficientNet-B4 (2019년 모델)로 바꾸면 같은 학습량으로 더 정확합니다.
        
        비유: V2는 "중학생 뇌"로 배웠고, 우리는 "대학생 뇌"로 배우는 것
    """
    if model_name == "efficientnet_b4":
        model = models.efficientnet_b4(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "densenet121":
        model = models.densenet121(weights='IMAGENET1K_V1')
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights='IMAGENET1K_V1')
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"알 수 없는 모델: {model_name}")
    
    return model


# ============================================================
# ⚖️ [개선 3] Weighted Loss 계산
# ============================================================
def calculate_pos_weight(train_df, device):
    """
    초보자 설명:
        "Weighted Loss"는 희귀한 병을 더 열심히 배우게 만드는 기법입니다.
        
        문제 상황:
        - 폐렴 데이터: 10,000장 (많음)
        - 탈장 데이터: 100장 (적음!)
        - V2는 둘 다 동일 비중으로 학습 → AI가 탈장을 무시함
        
        해결:
        - 탈장에 100배 가중치를 줌 → "탈장 사진 1장 = 폐렴 사진 100장" 효과
        - AI가 희귀 질환도 놓치지 않게 됨
    """
    all_labels = np.vstack(train_df['labels'].values)
    pos_counts = all_labels.sum(axis=0)           # 각 질환의 양성 개수
    neg_counts = len(train_df) - pos_counts       # 각 질환의 음성 개수
    
    # 0으로 나누기 방지
    pos_counts = np.maximum(pos_counts, 1)
    
    pos_weight = torch.FloatTensor(neg_counts / pos_counts).to(device)
    
    print("⚖️ 질환별 가중치:")
    for i, label in enumerate(LABEL_ORDER):
        print(f"   {label:24s}: {pos_weight[i]:.1f}x")
    
    return pos_weight


# ============================================================
# 📈 AUROC 계산 (모델 성능을 측정하는 지표)
# ============================================================
def calculate_auroc(all_preds, all_labels):
    """
    초보자 설명:
        AUROC는 "AI가 병을 얼마나 잘 구분하는지" 나타내는 점수입니다.
        - 0.5 = 동전 던지기 수준 (최악)
        - 0.8 = 괜찮은 수준
        - 0.9 = 매우 잘함
        - 1.0 = 완벽 (현실에서는 불가능)
    """
    auroc_list = []
    for c in range(14):
        if len(np.unique(all_labels[:, c])) > 1:
            score = roc_auc_score(all_labels[:, c], all_preds[:, c])
            auroc_list.append(score)
    return np.mean(auroc_list) if auroc_list else 0.0


# ============================================================
# 🏋️ 메인 학습 함수
# ============================================================
def train():
    # 디바이스 설정 (GPU > Apple Silicon > CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"🖥️ 사용 디바이스: {device}")
    print(f"📐 사용 모델: {CONFIG['model_name']}")
    
    # 1. 데이터 준비
    train_df = prepare_mimic_df(CONFIG["train_csv"], CONFIG["chexpert_csv"], CONFIG["img_root"])
    val_df = prepare_mimic_df(CONFIG["val_csv"], CONFIG["chexpert_csv"], CONFIG["img_root"])
    
    train_ds = MedicalImageDataset(train_df, CONFIG["img_root"], train_transform)
    val_ds = MedicalImageDataset(val_df, CONFIG["img_root"], val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    # 2. 모델 생성 [개선 2]
    model = create_model(CONFIG["model_name"]).to(device)
    
    # 3. Weighted Loss [개선 3]
    pos_weight = calculate_pos_weight(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # 4. 옵티마이저 + 스케줄러 [개선 4]
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
    
    # 5. Early Stopping [개선 5]
    best_auroc = 0.0
    patience_counter = 0

    print(f"\n🚀 학습 시작! (최대 {CONFIG['epochs']} 에폭, patience={CONFIG['patience']})")
    print("=" * 60)

    for epoch in range(CONFIG["epochs"]):
        # --- 학습 ---
        model.train()
        epoch_loss = 0
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"  Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step()  # [개선 4] 학습률 점점 줄이기

        # --- 검증 ---
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                outputs = torch.sigmoid(model(imgs))
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        auroc = calculate_auroc(val_preds, val_labels)
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 Epoch [{epoch+1}/{CONFIG['epochs']}] Loss: {avg_loss:.4f} | AUROC: {auroc:.4f} | LR: {current_lr:.6f}")

        # --- Best model 저장 & Early Stopping [개선 5] ---
        if auroc > best_auroc:
            best_auroc = auroc
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG["save_path"])
            print(f"💾 ✨ Best Model Saved! AUROC: {auroc:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ 개선 없음 ({patience_counter}/{CONFIG['patience']})")
            
            if patience_counter >= CONFIG["patience"]:
                print(f"\n🛑 Early Stopping! {CONFIG['patience']}번 연속 개선 없어 학습 종료")
                break

    print(f"\n🏆 최종 Best AUROC: {best_auroc:.4f}")
    print(f"💾 모델 저장 위치: {CONFIG['save_path']}")


if __name__ == "__main__":
    train()
