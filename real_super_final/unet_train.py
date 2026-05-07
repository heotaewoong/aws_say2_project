import os
import cv2
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 모델 파일 임포트
from unet_lung_model import UNet


def rle_decode(mask_rle, shape):
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
# 2. CheXmask + 로컬 메타데이터 통합 Dataset
# =====================================================================
class CheXmaskDataset(Dataset):
    def __init__(self, metadata_csv, image_root, mask_root, target_size=(256, 256), is_train=True, sample_n=None):
        print("📖 메타데이터 로드 및 필터링 중...")
        df = pd.read_csv(metadata_csv)
        
        # 🚀 [수정된 핵심] 실제 마스크에 1(폐)이나 2(심장)가 있는 유효한 파일만 추려냅니다.
        existing_mask_files = set([f.replace('.png', '') for f in os.listdir(mask_root) if f.endswith('.png')])
        temp_df = df[df['dicom_id'].isin(existing_mask_files)].reset_index(drop=True)
        
        print("🔍 유효 마스크(폐/심장 존재) 정밀 검증 중... (시간이 조금 걸릴 수 있습니다)")
        valid_rows = []
        for idx, row in temp_df.iterrows():
            mask_path = os.path.join(mask_root, row['dicom_id'] + ".png")
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None and np.sum(mask) > 0: # 0이 아닌 픽셀이 하나라도 있으면 합격
                valid_rows.append(row)
                
        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"🚨 완전히 비어있는 까만 마스크 {len(temp_df) - len(self.df)}장 제외 완료!")
        
        if sample_n is not None and len(self.df) > sample_n:
            print(f"🎲 유효 데이터를 {sample_n}장으로 샘플링합니다.")
            self.df = self.df.sample(n=sample_n, random_state=42).reset_index(drop=True)

        self.image_root = image_root
        self.mask_root = mask_root 
        self.target_size = target_size
        self.is_train = is_train
        
        gc.collect()
        print(f"✅ 필터링 완료: {len(df):,}장 중 유효 마스크 {len(self.df):,}장 매칭됨")

        self.transform = A.Compose([
            A.Resize(target_size[0], target_size[1], interpolation=cv2.INTER_NEAREST),
            #A.CLAHE(clip_limit=2.0, p=0.5),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            ], p=1.0) if is_train else A.NoOp(),
            #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        row = self.df.iloc[idx]
        
        # 이미지 로드
        img_path = os.path.join(self.image_root, row['relative_path'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크 로드
        mask_path = os.path.join(self.mask_root, row['dicom_id'] + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image.shape[:2] != mask.shape:
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        transformed = self.transform(image=image, mask=mask)
        return transformed['image'], transformed['mask'].long()
    
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        logits = torch.clamp(logits, min=-10, max=10)

        # logits: (B, C, H, W), targets: (B, H, W)
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        targets_long = targets.long()
        targets_one_hot = F.one_hot(targets_long, num_classes).permute(0, 3, 1, 2).float()
        
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
        weights = torch.tensor([1.0, 5.0, 10.0]).to(device) 
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.dice = SoftDiceLoss()

    def forward(self, inputs, targets):
        # CE는 픽셀 정확도를, Dice는 전체적인 형태를 잡아줍니다.
        return self.ce(inputs, targets) + self.dice(inputs, targets)

# =====================================================================
# 1. 세그멘테이션 평가지표 함수 (Dice & IoU)
# =====================================================================
def calculate_metrics(preds, targets, n_classes=3):
    preds = torch.argmax(preds, dim=1).long() # 예측값 (B, H, W)
    targets = targets.long()                  # 실제값 (B, H, W)
    
    dices, ious = [], []
    for cls in range(1, n_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        intersection = (p * t).sum()
        union = p.sum() + t.sum()
        
        # 만약 해당 클래스가 실제 마스크에 아예 없다면 계산에서 제외하거나 1로 처리
        if t.sum() == 0:
            dice = 1.0 if p.sum() == 0 else 0.0
            iou = 1.0 if p.sum() == 0 else 0.0
        else:
            dice = (2. * intersection + 1e-8) / (union + 1e-8)
            iou = (intersection + 1e-8) / (union - intersection + 1e-8)
        
        dices.append(dice.item())
        ious.append(iou.item())
        
    # [Lung_Dice, Heart_Dice], [Lung_IoU, Heart_IoU] 반환
    return dices, ious

# =====================================================================
# 1. 시각적 지표(Bar) 생성 함수 추가
# =====================================================================
def make_bar(score, length=15):
    """0~1 사이의 점수를 텍스트 진행 바(Bar)로 시각화합니다."""
    # 만약 분모가 0이 되어 NaN이 발생할 경우를 대비
    if np.isnan(score): 
        score = 0.0
    score = max(0.0, min(1.0, score)) # 0~1 사이로 클램핑
    filled = int(round(score * length))
    return '█' * filled + '░' * (length - filled)

# =====================================================================
# 2. 세그멘테이션 평가지표 함수 (버그 수정판: 교집합/합집합 반환)
# =====================================================================
def get_intersection_union(preds, targets, n_classes=3):
    preds = torch.argmax(preds, dim=1).long() # 예측값
    targets = targets.long()                  # 실제값
    
    intersections, unions = [], []
    for cls in range(1, n_classes):
        p = (preds == cls).float()
        t = (targets == cls).float()
        
        intersections.append((p * t).sum().item())
        unions.append((p.sum() + t.sum()).item())
        
    # [Lung_Inter, Heart_Inter], [Lung_Union, Heart_Union] 반환
    return intersections, unions

# =====================================================================
# 2. Main Training Loop (Best Saving 로직 포함)
# =====================================================================
def train_unet():
    # 하이퍼파라미터 (메모리 최적화 세팅)
    batch_size = 16        # VRAM 부족 시 유지
    accumulation_steps = 2 # 실제 배치 사이즈 16의 효과
    lr = 1e-4
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. relative_path가 들어있는 '정제된' 메타데이터 파일
    MY_METADATA_CSV = r"C:\code\data\my_mimic_metadata.csv" 
    
    # 2. 실제 엑스레이 이미지들이 있는 폴더
    IMAGE_ROOT = r"C:\code\data\mimic-cxr\files"
    
    # 3. 전처리로 만든 PNG 마스크들이 있는 폴더 (추가 필요!)
    MASK_ROOT = r"C:\code\data\mimic-cxr\preprocessed_masks"
    
    writer = SummaryWriter(r'C:\code\data\runs\unet_optimized')

    # 데이터 로더 (pin_memory로 GPU 전송 속도 향상)
    full_dataset = CheXmaskDataset(MY_METADATA_CSV, IMAGE_ROOT, MASK_ROOT,sample_n=20000)
    train_size = int(0.8 * len(full_dataset))
    val_sub, train_sub = random_split(full_dataset, [len(full_dataset)-train_size, train_size])
    
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    model = UNet(n_channels=3, n_classes=3).to(device)
    criterion = CombinedLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    start_epoch, best_dice = 0, 0.0
    checkpoint_path = r"C:\code\data\checkpoints\unet_latest.pth"

    # 🚀 이어하기 로직
    if os.path.exists(checkpoint_path):
        print(f"🔄 체크포인트 로드: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_dice = ckpt['best_dice']
        print(f"✅ {start_epoch}에폭부터 재개합니다.")

    print(f"🚀 학습 시작 (Device: {device})")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for i, (images, masks) in enumerate(pbar):
            images = images.to(device, non_blocking=True).float()  / 255.0
            masks = masks.to(device, non_blocking=True)            
            outputs = model(images)

            # 만약 outputs에 nan이 하나라도 있으면 즉시 멈추는 방어 코드
            if torch.isnan(outputs).any():
                print("⚠️ 경고: 모델 출력에 NaN 발생! 학습률을 더 낮춰야 합니다.")
                break

            loss = criterion(outputs, masks) / accumulation_steps

            if torch.isnan(loss):
                print(f"❌ Loss가 NaN입니다!")
                optimizer.zero_grad()
                continue

            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                # Gradient Clipping: 가중치 폭주 방어
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # 더 타이트하게 잡음
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps * images.size(0)
            pbar.set_postfix(loss=loss.item() * accumulation_steps)

        writer.add_scalar('Loss/train', train_loss/len(train_sub), epoch)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        t_l_inter, t_h_inter = 0.0, 0.0
        t_l_union, t_h_union = 0.0, 0.0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        
        with torch.no_grad():
            for images, masks in val_pbar:
                images = images.to(device).float() / 255.0
                masks = masks.to(device)

                #with amp.autocast(device_type='cuda'):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
                if i == 0:
                    print(f"\n🔍 [진단] 이미지 최대값: {images.max().item():.4f}")
                    print(f"🔍 [진단] 모델 출력(Logits) 최대/최소: {outputs.max().item():.4f} / {outputs.min().item():.4f}")
                    print(f"🔍 [진단] 모델이 예측한 클래스 종류: {torch.unique(preds).cpu().numpy()}")
                    print(f"🔍 [진단] 정답 마스크 클래스 종류: {torch.unique(masks).cpu().numpy()}")
                val_loss += criterion(outputs, masks).item() * images.size(0)
                
                # 교집합, 합집합 누적 (배치마다 평균 내지 않음!)
                inters, unions = get_intersection_union(outputs, masks)
                t_l_inter += inters[0]; t_h_inter += inters[1]
                t_l_union += unions[0]; t_h_union += unions[1]

        # 🚀 에폭 종료 후 최종 평균 지표 계산
        avg_l_dice = (2. * t_l_inter + 1e-8) / (t_l_union + 1e-8)
        avg_h_dice = (2. * t_h_inter + 1e-8) / (t_h_union + 1e-8)
        
        avg_l_iou = (t_l_inter + 1e-8) / (t_l_union - t_l_inter + 1e-8)
        avg_h_iou = (t_h_inter + 1e-8) / (t_h_union - t_h_inter + 1e-8)
        
        mean_dice = (avg_l_dice + avg_h_dice) / 2

        print(f"\n🔥 Epoch [{epoch+1}] 결과")
        print(f"   🫁 Lung  | Dice: {avg_l_dice:.4f} {make_bar(avg_l_dice)} | IoU: {avg_l_iou:.4f}")
        print(f"   ❤️ Heart | Dice: {avg_h_dice:.4f} {make_bar(avg_h_dice)} | IoU: {avg_h_iou:.4f}")
        print(f"   📊 Mean  | Dice: {mean_dice:.4f} {make_bar(mean_dice)}")

        # TensorBoard 기록 (기존 유지)
        writer.add_scalar('Dice/Lung', avg_l_dice, epoch)
        writer.add_scalar('Dice/Heart', avg_h_dice, epoch)
        writer.add_scalar('Dice/Mean', mean_dice, epoch)

        # 시각화 (이미지 격자 생성)
        img_s, m_s = next(iter(val_loader))
        with torch.no_grad(), amp.autocast(device_type='cuda'):
            pred_s = torch.argmax(model(img_s[:4].to(device).float()), dim=1).unsqueeze(1).float()
            writer.add_image('Images/True', torchvision.utils.make_grid(m_s[:4].unsqueeze(1).float()), epoch)
            writer.add_image('Images/Pred', torchvision.utils.make_grid(pred_s), epoch)

        # 체크포인트 저장
        ckpt_data = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
            'best_dice': max(best_dice, mean_dice)
        }
        torch.save(ckpt_data, checkpoint_path)
        
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(ckpt_data, "unet_chexmask_best.pth")
            print("🏆 최고 성능 갱신! 모델 저장 완료.")

        # 정기 스냅샷 (5 에폭마다)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"unet_lung_heart_ep{epoch+1}.pth")

        scheduler.step(mean_dice)
        torch.cuda.empty_cache() # 메모리 청소

    writer.close()

if __name__ == "__main__":
    train_unet()