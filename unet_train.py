import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 🚀 파트너님이 작성하신 U-Net 클래스를 불러옵니다.
# (파일 이름이 unet_model.py 라면 아래와 같이 임포트하세요)
from unet_lung_model import UNet 

# =====================================================================
# 1. 커스텀 Dataset (X-ray와 Mask를 1:1 쌍으로 묶어주는 역할)
# =====================================================================
class LungSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        
        # 1. 각각의 폴더에서 파일 목록을 Set(집합) 형태로 가져옵니다.
        image_files = set(f for f in os.listdir(image_dir) if not f.startswith('.'))
        mask_files = set(f for f in os.listdir(mask_dir) if not f.startswith('.'))
        
        # 🚀 2. [시니어의 방어 로직] 양쪽에 모두 존재하는 파일만 남깁니다 (교집합)
        valid_files = image_files & mask_files
        self.images = sorted(list(valid_files))
        
        # 3. 누락된 데이터가 얼마나 되는지 터미널에 보고해줍니다.
        missing_masks = image_files - mask_files
        if missing_masks:
            print(f"⚠️ 경고: 짝(마스크)이 없어서 훈련에서 제외된 이미지 수: {len(missing_masks)}장")
            print(f"   (예시: {list(missing_masks)[:3]})")
        
        print(f"✅ 최종 학습에 사용될 완벽한 쌍(Pair) 데이터 수: {len(self.images)}장")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # 1. 이미지 로드 (Input: RGB, Mask: Grayscale)
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 2. 동일한 크기로 Resize
        # 원본은 부드럽게(BILINEAR), 마스크는 픽셀값이 깨지지 않게 엣지를 살려서(NEAREST) 줄입니다.
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # 3. Numpy 배열로 변환 및 정규화 (0.0 ~ 1.0)
        img_np = np.array(image) / 255.0
        mask_np = np.array(mask) / 255.0

        # 4. PyTorch Tensor로 변환
        # 이미지는 (H, W, C) -> (C, H, W)
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1)
        # 마스크는 채널 차원을 추가하여 (1, H, W) 형태로 만듭니다.
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0)

        # 안전 장치: 마스크가 완벽한 0과 1의 이진(Binary) 값을 갖도록 강제
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

# =====================================================================
# 2. 의료 분할 전용 최강의 무기: Dice + BCE Loss
# =====================================================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # 파트너님의 U-Net은 마지막에 Sigmoid가 없으므로 여기서 씌워줍니다 (Logit -> 확률값)
        inputs = torch.sigmoid(inputs)       
        
        # 텐서를 1차원으로 쫙 폅니다
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 1. Dice Loss 계산 (두 영역의 교집합을 최대화)
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # 2. BCE Loss 계산 (픽셀 단위의 정확도)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # 최종 Loss는 두 개를 합쳐서 사용합니다!
        Dice_BCE = bce_loss + dice_loss
        return Dice_BCE

# =====================================================================
# 3. Main Training Loop
# =====================================================================
def train_unet():
    # 💡 하이퍼파라미터 설정
    batch_size = 16  # VRAM 한계에 따라 8이나 16으로 조절하세요.
    lr = 1e-4
    epochs = 20
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"🚀 U-Net 학습 준비 완료! (Device: {device})")

    # 💡 데이터셋 경로를 파트너님의 폴더 구조에 맞게 수정해 주세요!
    TRAIN_IMG_DIR = "data/Lung Segmentation Data/Train/Normal/images"
    TRAIN_MASK_DIR = "data/Lung Segmentation Data/Train/Normal/lung masks"
    
    train_dataset = LungSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, target_size=(224, 224))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 🚀 n_classes=1 로 설정하여 배경/폐 이진 분류를 수행합니다.
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            # 순전파
            outputs = model(images)
            
            # Loss 계산 및 역전파
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * images.size(0)
            
        avg_epoch_loss = epoch_loss / len(train_dataset)
        
        print(f"🔥 Epoch [{epoch+1}/{epochs}] | Train Loss(Dice+BCE): {avg_epoch_loss:.4f}")
        
        # 5 에포크마다 가중치 저장
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"unet_lung_mask_ep{epoch+1}.pth")
            print(f"💾 가중치 저장 완료: unet_lung_mask_ep{epoch+1}.pth")

if __name__ == "__main__":
    train_unet()