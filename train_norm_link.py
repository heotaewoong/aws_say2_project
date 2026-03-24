import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import glob
from loss import HybridLoss

# 1. 커스텀 데이터셋 정의
class NormalXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # 정상(Normal) 이미지만 모여있는 폴더 경로
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png")) + \
                           glob.glob(os.path.join(data_dir, "*.jpeg"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 2. 학습 환경 설정
def train():
    # 하이퍼파라미터
    batch_size = 32
    lr = 1e-3
    epochs = 50
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 데이터 전처리 (정규화는 필수!)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # ImageNet 기준 정규화 (전이학습은 아니지만 표준적으로 사용)
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 데이터 로더 (경로를 기태님의 환경에 맞게 수정하세요)
    dataset = NormalXrayDataset(data_dir="./data/pneumonia_data/train/NORMAL", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화
    from normal_link_model import DeepNormalLinkAE
    model = DeepNormalLinkAE().to(device)
    
    # 손실 함수 및 최적화 도구
    criterion = HybridLoss(alpha=0.8) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"🚀 학습 시작! 기기: {device} | 데이터 수: {len(dataset)}")

    # 3. 학습 루프
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images in dataloader:
            images = images.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, images) # 입력과 출력이 같아야 함
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        avg_loss = train_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

        # 10에폭마다 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"normal_link_v2_ep{epoch+1}.pth")

    print("✅ 모든 학습이 완료되었습니다!")

if __name__ == "__main__":
    train()