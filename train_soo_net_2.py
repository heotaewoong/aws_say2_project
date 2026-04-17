import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps
import pandas as pd
import os
import ast
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

# 🚀 모듈화된 파일들 임포트
from mimic_dataset import prepare_mimic_df, MimicDataset
from soo_net_2 import SooNetEngine
from unet_lung_model import UNet # 파트너님의 U-Net 모델 임포트
from visualizer import MedicalVisualizer

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def prepare_chexpert_df(csv_path, img_root):
    df = pd.read_csv(csv_path)
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0).replace(-1, 0)
    
    if 'Frontal/Lateral' in df.columns:
        df = df[df['Frontal/Lateral'] == 'Frontal']
    
    flat_data = []
    print(f"🔍 CheXpert 데이터 동적 경로 파싱 중... (FastFile 환경에 최적화됨)")
    missing_count = 0
    
    # 💡 파트너님의 실제 S3 배치 폴더 이름들
    candidate_folders = [
        "CheXpert-v1.0 batch 1 (validate & csv)",
        "CheXpert-v1.0 batch 2 (train 1)",
        "CheXpert-v1.0 batch 3 (train 2)",
        "CheXpert-v1.0 batch 4 (train 3)"
    ]

    for _, row in df.iterrows():
        raw_path = str(row['Path']) # 예: 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'
        
        # 1. 'patient'라는 단어가 시작되는 위치를 찾아 핵심 경로만 잘라냅니다.
        path_parts = raw_path.split('/')
        patient_idx = -1
        for i, part in enumerate(path_parts):
            if part.startswith('patient'):
                patient_idx = i
                break
                
        if patient_idx == -1:
            missing_count += 1
            continue
            
        # core_path = 'patient00001/study1/view1_frontal.jpg'
        core_path = "/".join(path_parts[patient_idx:]) 
        
        # 2. 4개의 배치 폴더 중 이 파일이 진짜로 존재하는 곳을 찾습니다.
        found_real_path = None
        for folder in candidate_folders:
            candidate_full_path = os.path.join(img_root, folder, core_path)
            
            # FastFile 모드에서도 os.path.exists가 S3를 찔러서 파일 유무를 확인해줍니다!
            if os.path.exists(candidate_full_path):
                found_real_path = candidate_full_path
                break
                
        if found_real_path:
            flat_data.append({
                'path': found_real_path,
                'labels': row[LABEL_ORDER].values
            })
        else:
            missing_count += 1
            # 너무 많이 출력되면 로그가 지저분해지므로 생략 (디버깅 시 주석 해제)
            # print(f"⚠️ 파일 없음: {core_path}")

    final_df = pd.DataFrame(flat_data)
    print(f"✅ CheXpert 동적 파싱 완료: 총 {len(final_df)}장 준비 완료! (누락/매칭실패 {missing_count}장)")
    return final_df

class ChexpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_full_path = row['path']
        
        try:
            # 1채널 흑백(Grayscale) 모드로 이미지를 엽니다
            image = Image.open(img_full_path).convert('L')
        except Exception:
            return self.__getitem__((idx + 1) % len(self))
            
        label = torch.FloatTensor(row['labels'].astype(float))
        
        if self.transform:
            image = self.transform(image)
        return image, label
# =====================================================================
# 🚀 1. 전처리 클래스 (U-Net을 위해 3채널 RGB 텐서 반환)
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(448, 448), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        img_padded = ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))
        return img_padded

class Base_Transform:
    def __init__(self, target_size=(448, 448), is_train=False):
        self.target_size = target_size
        self.is_train = is_train
        self.clahe_preprocess = ChestXrayPreprocess(target_size=target_size, clip_limit=2.0)

    def __call__(self, img):
        img = self.clahe_preprocess(img) # RGB 이미지 반환
        if self.is_train:
            img = transforms.RandomRotation(5)(img)
        # U-Net 입력용으로 3채널, 0.0~1.0 스케일의 텐서로 변환
        return transforms.ToTensor()(img)

# =====================================================================
# 🚀 이미지 & 마스크 동시 크롭 및 512x512 리사이즈
# =====================================================================
def process_unet_crops_with_masks(images, masks, padding=10, target_size=(512, 512)):
    batch_size = images.size(0)
    processed_imgs = []
    processed_masks = []

    for i in range(batch_size):
        img = images[i]
        mask = masks[i] # (2, H, W) - 폐와 심장 마스크
        
        # 💡 폐와 심장 마스크를 합쳐서(Union) 전체 흉곽의 Bounding Box를 구합니다.
        combined_mask = mask.sum(dim=0) > 0.5
        rows, cols = torch.any(combined_mask, dim=1), torch.any(combined_mask, dim=0)

        # 예외 처리: 마스크를 아예 못 찾은 경우 (안전 장치)
        if not torch.any(rows) or not torch.any(cols):
            gray_img = TF.resize(img.mean(dim=0, keepdim=True), target_size, antialias=True)
            blank_mask = torch.zeros((mask.shape[0], target_size[0], target_size[1]), device=mask.device)
            processed_imgs.append(gray_img)
            processed_masks.append(blank_mask)
            continue

        # 자를 좌표(Bounding Box) 계산
        y_indices, x_indices = torch.where(rows)[0], torch.where(cols)[0]
        y_min, y_max = y_indices[0].item(), y_indices[-1].item()
        x_min, x_max = x_indices[0].item(), x_indices[-1].item()

        H, W = combined_mask.shape
        y_min, y_max = max(0, y_min - padding), min(H, y_max + padding)
        x_min, x_max = max(0, x_min - padding), min(W, x_max + padding)

        # 💡 [핵심] 원본 엑스레이와 2채널 마스크를 "완벽히 동일한 좌표"로 자릅니다!
        cropped_img = img[:, y_min:y_max, x_min:x_max]
        cropped_mask = mask[:, y_min:y_max, x_min:x_max]

        # 비율이 깨지지 않도록 정방형(정사각형) 검은색 패딩 추가
        c, h, w = cropped_img.shape
        diff = abs(h - w)
        if h > w:
            pad_left, pad_right = diff // 2, diff - diff // 2
            cropped_img = TF.pad(cropped_img, (pad_left, pad_right, 0, 0), fill=0)
            cropped_mask = TF.pad(cropped_mask, (pad_left, pad_right, 0, 0), fill=0)
        elif w > h:
            pad_top, pad_bottom = diff // 2, diff - diff // 2
            cropped_img = TF.pad(cropped_img, (0, 0, pad_top, pad_bottom), fill=0)
            cropped_mask = TF.pad(cropped_mask, (0, 0, pad_top, pad_bottom), fill=0)

        # 💡 512x512 로 나란히 리사이즈
        resized_img = TF.resize(cropped_img, target_size, antialias=True)
        resized_mask = TF.resize(cropped_mask, target_size, antialias=True)

        gray_img = resized_img.mean(dim=0, keepdim=True)
        
        processed_imgs.append(gray_img)
        processed_masks.append(resized_mask)

    # 텐서 스택 및 DenseNet을 위한 TXV 스케일링 (-1024 ~ 1024)
    final_imgs = torch.stack(processed_imgs)
    final_masks = torch.stack(processed_masks)
    txv_scaled_imgs = (final_imgs * 2048.0) - 1024.0 
    
    return txv_scaled_imgs, final_masks

# =====================================================================
# 🚀 메인 학습 루프 (Anatomy-XNet + 7대 시각화 지표 통합 완성본)
# =====================================================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"🚀 Anatomy-XNet 스타일 학습 시작! Device: {device}")
    
    # 💡 1. 학습 트렌드를 기록할 History 딕셔너리와 Visualizer 초기화
    history = {'train_loss': [], 'val_loss': [], 'val_auroc': []}
    visualizer = MedicalVisualizer(labels=LABEL_ORDER, output_dir=args.plot_dir)
    
    # 데이터 로더 (기존 코드 그대로 유지 - Cloud 환경 분기 포함)
    if args.use_mimic:
        IMG_ROOT = args.data_dir 
        TRAIN_CSV = os.path.join(args.data_dir, "mimic_cxr_aug_train.csv")
        VAL_CSV = os.path.join(args.data_dir, "mimic_cxr_aug_validate.csv")
        CHEXPERT_CSV = os.path.join(args.data_dir, "mimic-cxr-2.0.0-chexpert.csv")
        SAVE_MODEL_NAME = os.path.join(args.model_dir, "anatomy_soonet_mimic_best.pth")
        
        train_df = prepare_mimic_df(TRAIN_CSV, CHEXPERT_CSV, IMG_ROOT)
        val_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
        TrainDatasetClass = MimicDataset
    else:
        IMG_ROOT = args.data_dir 
        TRAIN_CSV = os.path.join(args.data_dir, "train.csv")
        VAL_CSV = os.path.join(args.data_dir, "valid.csv")
        CHEXPERT_CSV = None  
        SAVE_MODEL_NAME = os.path.join(args.model_dir, "anatomy_soonet_chexpert_best.pth")
        
        train_df = prepare_chexpert_df(TRAIN_CSV, IMG_ROOT)
        val_df = prepare_chexpert_df(VAL_CSV, IMG_ROOT)
        TrainDatasetClass = ChexpertDataset

    # 512x512 해상도로 트랜스폼 설정
    train_transform = Base_Transform(target_size=(512, 512), is_train=True)
    val_transform = Base_Transform(target_size=(512, 512), is_train=False)
    
    train_loader = DataLoader(TrainDatasetClass(train_df, train_transform), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(TrainDatasetClass(val_df, val_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- U-Net 초기화 ---
    print("✂️ U-Net 모델 로드 중...")
    unet = UNet(n_channels=3, n_classes=2).to(device)
    unet.load_state_dict(torch.load(args.unet_weight_path, map_location=device))
    unet.eval() 

    # --- AnatomySooNet 초기화 ---
    print("🧠 AnatomySooNet (A^3 + PWAP) 로드 중...")
    engine = SooNetEngine(model_path=None) 
    model = engine.model 
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=args.lr_patience)

    best_auroc = 0.0
    
    # 💡 2. 중복된 for문을 제거하고 단일 루프로 구성
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        for i, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            with torch.no_grad():
                raw_masks = torch.sigmoid(unet(imgs))
            
            txv_imgs, final_masks = process_unet_crops_with_masks(imgs, raw_masks, target_size=(512, 512))
            
            outputs = model(txv_imgs, final_masks)
            loss = criterion(outputs, lbls)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Batch [{i+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # 💡 3. Train Loss 기록
        train_avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(train_avg_loss)

        # === 검증(Validation) 루프 ===
        model.eval()
        all_preds, all_labels = [], []
        val_epoch_loss = 0
        
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                
                raw_masks = torch.sigmoid(unet(imgs))
                txv_imgs, final_masks = process_unet_crops_with_masks(imgs, raw_masks, target_size=(512, 512))
                
                # 💡 Loss 계산을 위해 Sigmoid 전의 원본(Logits)을 먼저 받습니다.
                outputs = model(txv_imgs, final_masks) 
                v_loss = criterion(outputs, lbls)
                val_epoch_loss += v_loss.item()
                
                # 💡 확률값으로 변환하여 예측 리스트에 추가
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(lbls.numpy())
        
        # 💡 4. Val Loss 기록
        history['val_loss'].append(val_epoch_loss / len(val_loader))
        
        val_preds = np.vstack(all_preds)
        val_labels = np.vstack(all_labels)
        
        print(f"\n📊 --- Epoch {epoch+1} Validation Report ---")
        print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Prec':<6s} | {'Recall':<6s} | {'F1':<6s}")
        print("-" * 75)

        val_auroc_list, val_f1_list, best_thresholds = [], [], []

        for c, class_name in enumerate(LABEL_ORDER): 
            if len(np.unique(val_labels[:, c])) > 1:
                auroc = roc_auc_score(val_labels[:, c], val_preds[:, c])
                val_auroc_list.append(auroc)
                
                precisions, recalls, thresholds = precision_recall_curve(val_labels[:, c], val_preds[:, c])
                precisions, recalls = precisions[:-1], recalls[:-1]
                
                numerator = 2 * recalls * precisions
                denominator = recalls + precisions
                f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
                
                best_idx = np.argmax(f1_scores)
                best_threshold = thresholds[best_idx]
                
                opt_preds = (val_preds[:, c] >= best_threshold).astype(int)
                val_prec = precision_score(val_labels[:, c], opt_preds, zero_division=0)
                val_rec = recall_score(val_labels[:, c], opt_preds, zero_division=0)
                val_f1 = f1_score(val_labels[:, c], opt_preds, zero_division=0)
                
                val_f1_list.append(val_f1)
                best_thresholds.append(best_threshold) # 💡 Threshold 기록
                
                print(f"{class_name:<27s} | {auroc:.4f} | {best_threshold:.4f}  | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
            else:
                best_thresholds.append(0.5) # 안전망 (정답이 없는 질환의 경우)
                print(f"{class_name:<27s} |  N/A   |  N/A     |  N/A   |  N/A   |  N/A")
        
        print("-" * 75)
        
        avg_auroc = np.mean(val_auroc_list) if len(val_auroc_list) > 0 else 0.0
        avg_f1 = np.mean(val_f1_list) if len(val_f1_list) > 0 else 0.0
        
        # 💡 5. Val AUROC 기록
        history['val_auroc'].append(avg_auroc)
            
        print(f"✅ Epoch [{epoch+1}/{args.epochs}] Train Loss: {train_avg_loss:.4f} | Val Loss: {val_epoch_loss/len(val_loader):.4f} | Macro AUROC: {avg_auroc:.4f} | Macro F1: {avg_f1:.4f}\n")
        
        scheduler.step(avg_auroc)

        # 💡 6. 신기록 경신 시 가중치 저장 및 7대 시각화 지표 생성 발사!
        if avg_auroc > best_auroc:
            best_auroc = avg_auroc
            torch.save(model.state_dict(), SAVE_MODEL_NAME)
            print(f"🏆 Best Model Saved: {SAVE_MODEL_NAME} (AUROC: {avg_auroc:.4f})")
            
            # 🔥 Visualizer 실행
            try:
                visualizer.generate_all_reports(val_labels, val_preds, best_thresholds, history)
            except Exception as e:
                print(f"⚠️ 시각화 지표 생성 중 에러 발생 (학습은 계속 진행됩니다): {e}")

# =====================================================================
# 🚀 4. SageMaker 환경 변수
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', './data'))
    
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--lr-patience', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--use-mimic', type=lambda x: (str(x).lower() == 'true'), default=True)
    
    # 💡 [추가] U-Net 가중치 파일 이름 (SageMaker source_dir 폴더 안에 같이 올려주세요!)
    parser.add_argument('--unet-weight-path', type=str, default='unet_lung_mask_ep10.pth')

    args = parser.parse_args()
    train(args)