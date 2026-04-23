import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# 🚀 파트너님이 작성하신 U-Net 클래스를 불러옵니다.
# (파일 이름이 unet_model.py 라면 아래와 같이 임포트하세요)
from unet_lung_model import UNet 


def rle_decode(mask_rle, shape):
    '''
    mask_rle: RLE string (example: '1 3 10 5')
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    if pd.isna(mask_rle) or mask_rle == '':
        return np.zeros(shape, dtype=np.uint8)
        
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

# =====================================================================
# 2. CheXmask + 로컬 메타데이터 통합 Dataset
# =====================================================================
class CheXmaskDataset(Dataset):
    def __init__(self, chexmask_csv, local_metadata_csv, image_root, target_size=(512, 512), is_train=True):
        self.image_root = image_root
        self.target_size = target_size
        self.is_train = is_train

        # 🚀 [수정] 멀티라인 필드(Landmarks)를 제대로 인식하도록 설정
        print("📖 CheXmask CSV 로드 중... (멀티라인 파싱 적용)")
        df_mask = pd.read_csv(
            chexmask_csv, 
            engine='python',    # 멀티라인 처리에 더 유연한 파이썬 엔진 사용
            quotechar='"',      # 따옴표로 감싸진 필드 안의 줄바꿈 허용
            on_bad_lines='skip' # 만약 정말 데이터가 깨진 줄이 있다면 건너뜀
        )
        
        # 💡 메모리 절약을 위해 필요한 컬럼만 남깁니다. (Landmarks는 학습에 불필요)
        df_mask = df_mask[['dicom_id', 'Left Lung', 'Right Lung', 'Heart', 'Height', 'Width']]

        df_meta = pd.read_csv(local_metadata_csv)
        self.df = pd.merge(df_mask, df_meta, on='dicom_id', how='inner')
        
        print(f"✅ 데이터 준비 완료: 총 {len(self.df)}장의 이미지를 사용합니다.")

        # 🚀 강력한 증강 파이프라인
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.CLAHE(clip_limit=2.0, p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ], p=1.0) if is_train else A.NoOp(),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. 이미지 로드 (relative_path 활용)
        img_path = os.path.join(self.image_root, row['relative_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지 크기
        h, w = int(row['Height']), int(row['Width'])

        # 2. RLE 마스크 생성 및 병합 (1: Lung, 2: Heart)
        mask_combined = np.zeros((h, w), dtype=np.uint8)
        
        # 폐 영역 (좌/우 합산)
        l_lung = rle_decode(row['Left Lung'], (h, w))
        r_lung = rle_decode(row['Right Lung'], (h, w))
        mask_combined[(l_lung > 0) | (r_lung > 0)] = 1
        
        # 심장 영역
        heart = rle_decode(row['Heart'], (h, w))
        mask_combined[heart > 0] = 2

        # 3. 전처리 적용
        transformed = self.transform(image=image, mask=mask_combined)
        
        return transformed['image'], transformed['mask'].long()
    
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: (B, C, H, W), targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
        
        # 배경을 제외한 클래스(폐, 심장)에 대해서만 Dice 계산
        dims = (0, 2, 3)
        intersection = torch.sum(probs * targets_one_hot, dims)
        cardinality = torch.sum(probs + targets_one_hot, dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice_score[1:].mean() # 배경 제외 평균 Dice Loss

class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        # 🚀 심장(2번 클래스)에 가중치를 3배 더 줍니다.
        weights = torch.tensor([1.0, 1.0, 5.0]).to(device) 
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = SoftDiceLoss()

    def forward(self, inputs, targets):
        # CE는 픽셀 정확도를, Dice는 전체적인 형태를 잡아줍니다.
        return self.ce(inputs, targets) + self.dice(inputs, targets)

# =====================================================================
# 1. 세그멘테이션 평가지표 함수 (Dice & IoU)
# =====================================================================
def calculate_metrics(preds, targets, n_classes=3):
    preds = torch.argmax(preds, dim=1)
    
    # 클래스별 결과를 담을 리스트
    dices = []
    ious = []
    
    # 1: Lung, 2: Heart (배경 0 제외)
    for cls in range(1, n_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        iou = (intersection + 1e-8) / (union - intersection + 1e-8)
        
        dices.append(dice.item())
        ious.append(iou.item())
        
    # [Lung_Dice, Heart_Dice], [Lung_IoU, Heart_IoU] 반환
    return dices, ious

# =====================================================================
# 2. Main Training Loop (Best Saving 로직 포함)
# =====================================================================
def train_unet():
    # 💡 하이퍼파라미터
    batch_size = 8  # 512 해상도이므로 VRAM 고려하여 조정
    lr = 1e-4
    epochs = 20
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    
    # 🎯 경로 설정 (Gitae님의 환경에 맞춤)
    CHEXMASK_CSV = "/Users/skku_aws2_15/med/data/MIMIC-CXR-JPG.csv" # CheXmask 공식 CSV
    MY_METADATA_CSV = "/Users/skku_aws2_15/med/data/my_mimic_metadata.csv"           # 방금 생성하신 파일
    IMAGE_ROOT = "/Users/skku_aws2_15/med/data/official_data_iccv_final/files"
    
    # 데이터셋 로드
    full_dataset = CheXmaskDataset(CHEXMASK_CSV, MY_METADATA_CSV, IMAGE_ROOT)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size])
    
    # RLE 연산 속도를 위해 num_workers 설정을 권장합니다.
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = CombinedLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_dice = 0.0 # 🚀 최고 Dice Score 추적용
    
    print(f"🚀 학습 시작! (Train: {train_size}장, Val: {val_size}장)")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        # --- Validation Phase (지표 계산) ---
        model.eval()
        val_loss = 0.0
        # 🚀 각 클래스별 합계 변수 초기화
        t_l_dice, t_h_dice = 0.0, 0.0
        t_l_iou, t_h_iou = 0.0, 0.0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                
                # 지표 계산 (calculate_metrics가 리스트 두 개를 반 hand한다고 가정)
                (l_dice, h_dice), (l_iou, h_iou) = calculate_metrics(outputs, masks)
                t_l_dice += l_dice
                t_h_dice += h_dice
                t_l_iou += l_iou
                t_h_iou += h_iou

        # 평균 계산
        num_batches = len(val_loader)
        avg_l_dice, avg_h_dice = t_l_dice / num_batches, t_h_dice / num_batches
        avg_l_iou, avg_h_iou = t_l_iou / num_batches, t_h_iou / num_batches
        
        mean_dice = (avg_l_dice + avg_h_dice) / 2
        mean_iou = (avg_l_iou + avg_h_iou) / 2

        scheduler.step(mean_dice)

        print(f"🔥 Epoch [{epoch+1}/{epochs}] | Val Loss: {val_loss/len(val_sub):.4f}")
        print(f"   🫁 Lung  | Dice: {avg_l_dice:.4f} | IoU: {avg_l_iou:.4f}")
        print(f"   ❤️ Heart | Dice: {avg_h_dice:.4f} | IoU: {avg_h_iou:.4f}")
        print(f"   📊 Mean  | Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f}")

        # 🏆 [Best Model 저장] Dice Score가 갱신될 때마다 저장
        if mean_dice > best_dice:
            best_dice = mean_dice
            save_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'dice_score': best_dice,
                'iou_score': mean_iou
            }
            torch.save(save_data, "unet_lung_heart_best.pth")
            print(f"✅ Best Model Saved! (Dice: {best_dice:.4f})")

        # 정기 스냅샷 (5 에폭마다)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"unet_lung_heart_ep{epoch+1}.pth")

if __name__ == "__main__":
    train_unet()