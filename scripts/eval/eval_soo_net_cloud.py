import argparse, os, torch
import torch.nn as nn
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, average_precision_score
)
from soo_net import SooNetEngine
from unet_lung_model import UNet

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def prepare_df(csv_channel_path, image_channel_path, mimic_channel_path, csv_name):
    csv_full_path = os.path.join(csv_channel_path, csv_name)
    print(f"📖 CSV 읽는 중: {csv_full_path}")
    df = pd.read_csv(csv_full_path)
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0)
    df[LABEL_ORDER] = df[LABEL_ORDER].replace(-1, 0)

    def fix_path(p):
        if p.startswith('CheXpert-v1.0'):
            return os.path.join(image_channel_path, p.split('/', 1)[1])
        else:
            return os.path.join(mimic_channel_path, p.split('/', 1)[1])

    df['real_path'] = df['image_path'].apply(fix_path)
    print(f"✅ 데이터 준비 완료: {len(df)}건")
    return df

class ChexpertDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row['real_path']).convert('L')
        except:
            return self.__getitem__((idx + 1) % len(self))
        label = torch.FloatTensor(row[LABEL_ORDER].values.astype(float))
        if self.transform:
            image = self.transform(image)
        return image, label

def process_unet_crops(images, masks, padding=10, target_size=(448, 448)):
    batch_size = images.size(0)
    processed_batch = []
    for i in range(batch_size):
        img, mask = images[i], masks[i].squeeze() > 0.5
        rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
        if not torch.any(rows) or not torch.any(cols):
            processed_batch.append(img.mean(dim=0, keepdim=True))
            continue
        y_idx, x_idx = torch.where(rows)[0], torch.where(cols)[0]
        y_min = max(0, y_idx[0].item() - padding)
        y_max = min(mask.shape[0], y_idx[-1].item() + padding)
        x_min = max(0, x_idx[0].item() - padding)
        x_max = min(mask.shape[1], x_idx[-1].item() + padding)
        cropped = img[:, y_min:y_max, x_min:x_max]
        c, h, w = cropped.shape
        diff = abs(h - w)
        if h > w:
            cropped = TF.pad(cropped, (diff//2, diff - diff//2, 0, 0), fill=0)
        elif w > h:
            cropped = TF.pad(cropped, (0, 0, diff//2, diff - diff//2), fill=0)
        resized = TF.resize(cropped, target_size, antialias=True)
        processed_batch.append(resized.mean(dim=0, keepdim=True))
    final = torch.stack(processed_batch)
    return (final * 2048.0) - 1024.0

def evaluate_and_save_metrics(all_labels, all_preds, output_dir):
    results = []
    for i, class_name in enumerate(LABEL_ORDER):
        y_true, y_prob = all_labels[:, i], all_preds[:, i]
        if len(np.unique(y_true)) < 2:
            continue
        auroc = roc_auc_score(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        p, r, t = precision_recall_curve(y_true, y_prob)
        f1_s = np.divide(2*p*r, p+r, out=np.zeros_like(p), where=(p+r != 0))
        best_t = t[np.argmax(f1_s[:-1])]
        y_pred = (y_prob >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        results.append({
            "Disease": class_name, "AUROC": auroc, "AUPRC": ap,
            "Sensitivity": tp/(tp+fn+1e-8), "Specificity": tn/(tn+fp+1e-8),
            "PPV": tp/(tp+fp+1e-8), "NPV": tn/(tn+fn+1e-8),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "TP": tp, "FP": fp, "TN": tn, "FN": fn
        })
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(output_dir, "medical_metrics_report.csv"), index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x="AUROC", y="Disease", data=df_res.sort_values("AUROC", ascending=False))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "auroc_performance.png"))
    plt.close()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        [[df_res['TN'].mean(), df_res['FP'].mean()],
         [df_res['FN'].mean(), df_res['TP'].mean()]],
        annot=True, fmt='.1f', cmap='Blues',
        xticklabels=['Neg', 'Pos'], yticklabels=['Actual Neg', 'Actual Pos']
    )
    plt.title('Average Confusion Matrix')
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ 평가 리포트 저장 완료: {output_dir}")
    print(f"📊 평균 AUROC: {df_res['AUROC'].mean():.4f}")
    return df_res

EXTRACT_DIR = "/tmp/pretrained_model"

def extract_model_if_needed(pretrained_dir):
    """model.tar.gz를 /tmp/에 압축 해제 (input 채널은 read-only이므로)"""
    import tarfile
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    tar_path = os.path.join(pretrained_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        print(f"📦 model.tar.gz 발견 → {EXTRACT_DIR} 에 압축 해제 중...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        print("✅ 압축 해제 완료")
        return EXTRACT_DIR
    else:
        print(f"ℹ️  model.tar.gz 없음, pretrained_dir 직접 사용: {pretrained_dir}")
        return pretrained_dir

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 평가 시작 (Device: {device})")

    # pretrained 채널에서 model.tar.gz 압축 해제 → 쓰기 가능한 /tmp/ 사용
    model_dir = extract_model_if_needed(args.pretrained_dir)

    # 데이터 로드
    eval_df = prepare_df(
        args.train_dir, args.image_dir, args.mimic_dir, args.eval_csv_name
    )
    eval_loader = DataLoader(
        ChexpertDataset(eval_df, transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])),
        batch_size=16, shuffle=False, num_workers=4
    )

    # U-Net 로드 (weights 채널에서 경로 자동 결정)
    unet_path = args.unet_weight_path or os.path.join(args.weights_dir, 'unet_lung_mask_ep10.pth')
    print(f"✂️ U-Net 가중치 로드 중: {unet_path}")
    unet = UNet(n_channels=3, n_classes=1).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()

    # SooNet 로드 (저장된 가중치)
    model_path = os.path.join(model_dir, args.model_filename)
    print(f"🧠 SooNet 가중치 로드 중: {model_path}")
    engine = SooNetEngine(model_path=None)
    model = engine.model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 추론
    all_preds, all_labels = [], []
    print(f"📊 추론 시작 (총 {len(eval_df)}건)...")
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(eval_loader):
            imgs = imgs.to(device)
            masks = unet(imgs)
            txv_imgs = process_unet_crops(imgs, masks)
            preds = torch.sigmoid(model(txv_imgs))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.numpy())
            if (i+1) % 100 == 0:
                print(f"  배치 {i+1}/{len(eval_loader)} 완료")

    evaluate_and_save_metrics(
        np.vstack(all_labels), np.vstack(all_preds), args.model_dir
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',       type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train-dir',       type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--image-dir',       type=str, default=os.environ.get('SM_CHANNEL_IMAGES'))
    parser.add_argument('--mimic-dir',       type=str, default=os.environ.get('SM_CHANNEL_MIMIC'))
    parser.add_argument('--pretrained-dir',  type=str, default=os.environ.get('SM_CHANNEL_PRETRAINED'))
    parser.add_argument('--eval-csv-name',   type=str, default='chexpert_balanced_u_ones.csv')
    parser.add_argument('--model-filename',  type=str, default='soonet_uones.pth')
    parser.add_argument('--weights-dir',    type=str, default=os.environ.get('SM_CHANNEL_WEIGHTS', '.'))
    parser.add_argument('--unet-weight-path',type=str, default=None)
    evaluate(parser.parse_args())
