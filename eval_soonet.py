import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageOps
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score, precision_recall_curve, precision_score, recall_score, f1_score

# 🚀 모듈화된 파일들을 불러옵니다!
from mimic_dataset import prepare_mimic_df, MimicDataset
from chexpert_dataset import prepare_chexpert_df, ChexpertDataset
from soo_net import SooNetEngine

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
    "Lung Opacity", "No Finding", "Pleural Effusion", 
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

# =====================================================================
# ⚙️ [평가 환경 설정]
# =====================================================================
USE_MIMIC = True  # 평가할 데이터셋 선택 (True: MIMIC, False: CheXpert)

TARGET_SIZE = (448, 448)    
BATCH_SIZE = 32             
NUM_WORKERS = 2             

# 💡 평가할 가중치 파일 경로를 지정하세요!
WEIGHT_PATH = "model/chexnet_1ch_448_chexpert_best.pth" 

if USE_MIMIC:
    IMG_ROOT = "data/mimic-iv-cxr/official_data_iccv_final" 
    VAL_CSV = "data/mimic-iv-cxr/mimic_cxr_aug_validate.csv" # 테스트/검증용 CSV
    CHEXPERT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv"
else:
    IMG_ROOT = "data/chexpert" 
    VAL_CSV = "data/chexpert/valid.csv"
    CHEXPERT_CSV = None  

# =====================================================================
# 1. 전처리 클래스 (학습 환경과 100% 동일하게 유지)
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=TARGET_SIZE, clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        img_padded = ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))
        return img_padded

class TXV_Transform:
    def __init__(self, target_size=TARGET_SIZE):
        self.target_size = target_size
        self.clahe_preprocess = ChestXrayPreprocess(target_size=target_size, clip_limit=2.0)

    def __call__(self, img):
        img = self.clahe_preprocess(img)
        img = img.convert('L')
        img_tensor = transforms.ToTensor()(img)
        img_tensor = (img_tensor * 2048.0) - 1024.0 # TXV 공식 스케일링
        return img_tensor

# =====================================================================
# 2. 메인 평가 루프
# =====================================================================
def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 시작! Device: {device}")
    
    # 1. 데이터셋 및 데이터로더 준비
    if USE_MIMIC:
        test_df = prepare_mimic_df(VAL_CSV, CHEXPERT_CSV, IMG_ROOT)
        TestDatasetClass = MimicDataset
    else:
        test_df = prepare_chexpert_df(VAL_CSV, IMG_ROOT)
        TestDatasetClass = ChexpertDataset

    test_transform = TXV_Transform(target_size=TARGET_SIZE)
    test_loader = DataLoader(TestDatasetClass(test_df, test_transform), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if len(test_loader.dataset) == 0:
        print("🚨 에러: 테스트 데이터가 0장입니다. 경로를 확인해 주세요!")
        return

    # 2. 모델 초기화 및 가중치 로드 (soo_net.py 활용)
    print(f"🧠 SooNetEngine을 통해 모델을 로드합니다... ({WEIGHT_PATH})")
    engine = SooNetEngine(model_path=WEIGHT_PATH) 
    model = engine.model
    model.eval() # 추론 모드 확정

    all_preds, all_labels = [], []
    
    print("📊 테스트 데이터 추론 중 (시간이 조금 걸릴 수 있습니다)...")
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            outputs = torch.sigmoid(model(imgs)) 
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(lbls.numpy())
    
    val_preds = np.vstack(all_preds)
    val_labels = np.vstack(all_labels)
    
    # 3. 평가 지표 계산 및 출력
    print("\n===================================================================")
    print("🏆 Optimal Threshold 적용 후 최종 성적표 (AUROC, Precision, Recall, F1)")
    print("===================================================================")
    print(f"{'Disease':<27s} | {'AUROC':<6s} | {'Best_Th':<7s} | {'Prec':<6s} | {'Recall':<6s} | {'F1':<6s}")
    print("-" * 75)

    val_auroc_list, val_f1_list = [], []

    for c, class_name in enumerate(LABEL_ORDER): 
        if len(np.unique(val_labels[:, c])) > 1:
            # AUROC
            auroc = roc_auc_score(val_labels[:, c], val_preds[:, c])
            val_auroc_list.append(auroc)
            
            # 최적 임계값 도출
            precisions, recalls, thresholds = precision_recall_curve(val_labels[:, c], val_preds[:, c])
            precisions = precisions[:-1]
            recalls = recalls[:-1]
            
            numerator = 2 * recalls * precisions
            denominator = recalls + precisions
            f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator!=0))
            
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            
            # 지표 계산
            opt_preds = (val_preds[:, c] >= best_threshold).astype(int)
            val_prec = precision_score(val_labels[:, c], opt_preds, zero_division=0)
            val_rec = recall_score(val_labels[:, c], opt_preds, zero_division=0)
            val_f1 = f1_score(val_labels[:, c], opt_preds, zero_division=0)
            
            val_f1_list.append(val_f1)
            
            print(f"{class_name:<27s} | {auroc:.4f} | {best_threshold:.4f}  | {val_prec:.4f} | {val_rec:.4f} | {val_f1:.4f}")
        else:
            print(f"{class_name:<27s} |  N/A   |  N/A     |  N/A   |  N/A   |  N/A")
    
    print("-" * 75)
    avg_auroc = np.mean(val_auroc_list) if len(val_auroc_list) > 0 else 0.0
    avg_f1 = np.mean(val_f1_list) if len(val_f1_list) > 0 else 0.0
    print(f"🔥 전체 평균 (Macro) ➡️ AUROC: {avg_auroc:.4f} | F1-Score: {avg_f1:.4f}\n")

if __name__ == "__main__":
    evaluate()