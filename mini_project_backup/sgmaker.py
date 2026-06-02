import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import argparse # [추가] 인자 전달을 위해 필요
import glob
from loss import HybridLoss

# 1. 커스텀 데이터셋 정의 (SageMaker 환경에 맞춰 수정)
class NormalXrayDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        # SageMaker Manifest 사용 시 파일들이 폴더 구분 없이 들어올 수 있으므로 확장자로 검색
        self.image_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
            # 만약 하위 폴더가 있다면 recursive=True를 사용하세요
            self.image_paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
            
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# 2. 학습 함수
def train(args):
    # 하이퍼파라미터 (args에서 받아옴)
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    
    # [수정] SageMaker GPU(cuda) 지원
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # [수정] SageMaker가 데이터를 넣어주는 경로(args.train) 사용
    dataset = NormalXrayDataset(data_dir=args.train, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 모델 초기화 (SkipConnection 버전 권장)
    from normal_link_model import SkipNormalLinkAE 
    model = SkipNormalLinkAE().to(device)
    
    criterion = HybridLoss(alpha=0.8) 
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    print(f"🚀 학습 시작! 기기: {device} | 데이터 수: {len(dataset)}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images in dataloader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        avg_loss = train_loss / len(dataloader.dataset)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.6f}")

    # [수정] 모델 저장: 반드시 args.model_dir에 저장해야 S3로 자동 업로드됨
    print("💾 모델 저장 중...")
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pth"))
    print("✅ 모든 학습 및 저장 완료!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker가 자동으로 넣어주는 환경 변수들
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    
    # 경로 관련 인자 (SageMaker 전용)
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    train(parser.parse_args())