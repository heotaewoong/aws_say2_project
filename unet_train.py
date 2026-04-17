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

# =====================================================================
# 1. 커스텀 Dataset (X-ray와 Mask를 1:1 쌍으로 묶어주는 역할)
# =====================================================================
class LungSegmentationDataset(Dataset):
    def __init__(self, csv_path, image_root, mask_root, target_size=(512, 512), is_train=True):
        self.image_root = image_root
        self.mask_root = mask_root
        self.target_size = target_size
        self.is_train = is_train
        
        # 🚀 강력한 증강 파이프라인 구성
        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1]),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            # 훈련 데이터일 때만 적용하는 증강
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ], p=1.0) if is_train else A.NoOp(),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # 표준화
            ToTensorV2()
        ])
        # 1. CSV 로드
        df = pd.read_csv(csv_path)
        
        valid_rows = []
        print("🔍 파일 존재 여부 확인 중... (Mask Extension: .png 대응)")
        for _, row in df.iterrows():
            # 💡 1. 엑스레이 이미지 경로 처리 (.dcm -> .jpg)
            img_rel_path = row['MIMIC-CXR_path'].replace('.dcm', '.jpg')
            img_path = os.path.join(self.image_root, img_rel_path)
            
            # 💡 2. 마스크 경로 처리 (.jpg -> .png로 강제 변경)
            # CSV에 'heart/101.jpg'라고 적혀 있어도 'heart/101.png'를 찾도록 합니다.
            lung_rel_path = row['lungs_mask_path'].replace('.jpg', '.png')
            heart_rel_path = row['heart_mask_path'].replace('.jpg', '.png')
            
            lung_path = os.path.join(self.mask_root, lung_rel_path)
            heart_path = os.path.join(self.mask_root, heart_rel_path)
            
            # 🚀 [디버깅] 첫 번째 데이터만 경로가 맞는지 출력해봅니다.
            if len(valid_rows) == 0:
                print(f"--- 첫 번째 데이터 경로 테스트 ---")
                print(f"Image: {img_path} ({os.path.exists(img_path)})")
                print(f"Lung : {lung_path} ({os.path.exists(lung_path)})")
                print(f"Heart: {heart_path} ({os.path.exists(heart_path)})")
            
            if os.path.exists(img_path) and os.path.exists(lung_path) and os.path.exists(heart_path):
                valid_rows.append({
                    'image': img_path,
                    'lung': lung_path,
                    'heart': heart_path
                })
        
        self.data = valid_rows
        print(f"✅ 최종 학습 가능 데이터: {len(self.data)}장 (확장자 매칭 성공!)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 1. 이미지 및 마스크 로드 (PIL)
        image_pil = Image.open(item['image']).convert("RGB")
        lung_pil = Image.open(item['lung']).convert("L")
        heart_pil = Image.open(item['heart']).convert("L")

        # 🚀 [핵심 수정] Albumentations에 넣기 전, 동일한 크기로 1차 리사이즈
        # 이렇게 해야 'Height and Width should be equal' 에러가 사라집니다.
        image_pil = image_pil.resize(self.target_size, Image.BILINEAR)
        lung_pil = lung_pil.resize(self.target_size, Image.NEAREST)
        heart_pil = heart_pil.resize(self.target_size, Image.NEAREST)

        # 2. Numpy 배열로 변환
        image_np = np.array(image_pil)
        lung_np = np.array(lung_pil)
        heart_np = np.array(heart_pil)

        # 3. 마스크 병합 (0, 1, 2)
        mask_combined = np.zeros(self.target_size[::-1], dtype=np.uint8) # (H, W)
        mask_combined[lung_np > 127] = 1
        mask_combined[heart_np > 127] = 2

        # 4. 🚀 이미지와 마스크를 동시에 변형!
        # 이제 두 입력의 크기가 target_size로 동일하므로 에러가 나지 않습니다.
        transformed = self.transform(image=image_np, mask=mask_combined)
        
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
    
    # 🎯 경로 설정
    CSV_PATH = "data/mimic_masks/MIMIC_links.csv"
    IMAGE_ROOT = "data/official_data_iccv_final"
    MASK_ROOT = "data/mimic_masks"
    
    # 전체 데이터셋 로드 후 8:2 비율로 분할 (검증 성능 확인을 위해 필수)
    full_dataset = LungSegmentationDataset(CSV_PATH, IMAGE_ROOT, MASK_ROOT, target_size=(512, 512))
   
    # 🚀 디버깅용 로그 추가: 실제로 몇 장이 로드되었는지 꼭 확인하세요.
    if len(full_dataset) == 0:
        print("❌ 에러: 데이터를 찾지 못했습니다. IMAGE_ROOT와 MASK_ROOT 경로를 다시 확인하세요!")
        print(f"현재 설정된 IMAGE_ROOT: {os.path.abspath(IMAGE_ROOT)}")
        print(f"현재 설정된 MASK_ROOT: {os.path.abspath(MASK_ROOT)}")
        return # 학습 중단
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_sub, val_sub = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False)
    
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