"""
3개 모델(U-Ones, U-Ignore, Mixed)을 CheXpert validation set으로 비교 평가
- 입력: valid.csv (234개 공식 validation 이미지)
- 출력: AUROC/F1 비교표, ROC 곡선, Confusion Matrix, 바 차트
"""
import argparse, os, tarfile, torch
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, f1_score,
    confusion_matrix, average_precision_score, roc_curve
)
from soo_net import SooNetEngine
from unet_lung_model import UNet

LABEL_ORDER = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

MODELS = {
    "U-Ones":   "soonet_uones.pth",
    "U-Ignore": "soonet_uignore.pth",
    "Mixed":    "soonet_umixed.pth",
}

COLORS = {"U-Ones": "#E74C3C", "U-Ignore": "#2ECC71", "Mixed": "#3498DB"}

# ── 데이터 준비 ──
def prepare_valid_df(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    # valid.csv 컬럼명: 'Path' → 'image_path' 통일
    if 'Path' in df.columns:
        df = df.rename(columns={'Path': 'image_path'})
    df[LABEL_ORDER] = df[LABEL_ORDER].fillna(0).replace(-1, 0)
    # CheXpert-v1.0/valid/patient.../view.jpg → valid/patient.../view.jpg
    # valid.csv 경로: CheXpert-v1.0/valid/patient.../view.jpg
    # valid_only 채널: patient.../view.jpg (valid/ prefix 없음)
    def fix_valid_path(p):
        parts = p.split('/')  # ['CheXpert-v1.0', 'valid', 'patient...', ...]
        # 'valid/' 이후 부분만 사용
        if 'valid' in parts:
            idx = parts.index('valid')
            return os.path.join(image_dir, *parts[idx+1:])
        return os.path.join(image_dir, p.split('/', 1)[1])
    df['real_path'] = df['image_path'].apply(fix_valid_path)
    print(f"✅ Validation 데이터: {len(df)}건")
    return df

class ValidDataset(Dataset):
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

# ── U-Net 크롭 ──
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

# ── 단일 모델 추론 ──
def run_inference(model, unet, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            masks = unet(imgs)
            txv_imgs = process_unet_crops(imgs, masks)
            preds = torch.sigmoid(model(txv_imgs))
            all_preds.append(preds.cpu().numpy())
            all_labels.append(lbls.numpy())
    return np.vstack(all_labels), np.vstack(all_preds)

# ── 지표 계산 ──
def compute_metrics(all_labels, all_preds):
    results = []
    for i, cls in enumerate(LABEL_ORDER):
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
            "Disease": cls, "AUROC": auroc, "AUPRC": ap,
            "Sensitivity": tp/(tp+fn+1e-8),
            "Specificity": tn/(tn+fp+1e-8),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        })
    return pd.DataFrame(results)

# ── 시각화 ──
def plot_auroc_comparison(metrics_dict, output_dir):
    """3개 모델 AUROC 바 차트"""
    fig, ax = plt.subplots(figsize=(12, 8))
    diseases = LABEL_ORDER
    x = np.arange(len(diseases))
    width = 0.25
    for i, (name, df) in enumerate(metrics_dict.items()):
        vals = [df[df['Disease']==d]['AUROC'].values[0] if d in df['Disease'].values else 0 for d in diseases]
        ax.bar(x + i*width, vals, width, label=name, color=COLORS[name], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('AUROC')
    ax.set_title('Validation AUROC: U-Ones vs U-Ignore vs Mixed Policy')
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auroc_comparison_valid.png'), dpi=150)
    plt.close()
    print("✅ auroc_comparison_valid.png 저장")

def plot_f1_comparison(metrics_dict, output_dir):
    """3개 모델 F1 바 차트"""
    fig, ax = plt.subplots(figsize=(12, 8))
    diseases = LABEL_ORDER
    x = np.arange(len(diseases))
    width = 0.25
    for i, (name, df) in enumerate(metrics_dict.items()):
        vals = [df[df['Disease']==d]['F1'].values[0] if d in df['Disease'].values else 0 for d in diseases]
        ax.bar(x + i*width, vals, width, label=name, color=COLORS[name], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(diseases, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('F1 Score')
    ax.set_title('Validation F1 Score: U-Ones vs U-Ignore vs Mixed Policy')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_comparison_valid.png'), dpi=150)
    plt.close()
    print("✅ f1_comparison_valid.png 저장")

def plot_roc_curves(labels_dict, preds_dict, output_dir):
    """질환별 ROC 곡선 (3개 모델 겹쳐서)"""
    n_diseases = len(LABEL_ORDER)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    axes = axes.flatten()
    for i, disease in enumerate(LABEL_ORDER):
        ax = axes[i]
        for name in MODELS.keys():
            y_true = labels_dict[name][:, LABEL_ORDER.index(disease)]
            y_prob = preds_dict[name][:, LABEL_ORDER.index(disease)]
            if len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = roc_auc_score(y_true, y_prob)
            ax.plot(fpr, tpr, color=COLORS[name], label=f'{name} ({auroc:.3f})', linewidth=2)
        ax.plot([0,1],[0,1],'k--', alpha=0.3)
        ax.set_title(disease, fontsize=10)
        ax.set_xlabel('FPR', fontsize=8)
        ax.set_ylabel('TPR', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    # 빈 subplot 숨기기
    for j in range(n_diseases, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('ROC Curves per Disease (Validation Set)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_valid.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print("✅ roc_curves_valid.png 저장")

def plot_confusion_matrices(labels_dict, preds_dict, metrics_dict, output_dir):
    """각 모델별 평균 Confusion Matrix"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (name, df) in zip(axes, metrics_dict.items()):
        cm = [[df['TN'].mean(), df['FP'].mean()],
              [df['FN'].mean(), df['TP'].mean()]]
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                    xticklabels=['Pred Neg', 'Pred Pos'],
                    yticklabels=['Actual Neg', 'Actual Pos'])
        ax.set_title(f'{name}\n평균 AUROC: {df["AUROC"].mean():.4f}')
    plt.suptitle('Average Confusion Matrix per Model (Validation Set)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_valid.png'), dpi=150)
    plt.close()
    print("✅ confusion_matrix_valid.png 저장")

def plot_avg_auroc_summary(metrics_dict, output_dir):
    """평균 AUROC 요약 바 차트"""
    names = list(metrics_dict.keys())
    avgs = [metrics_dict[n]['AUROC'].mean() for n in names]
    colors = [COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(names, avgs, color=colors, alpha=0.85, edgecolor='black')
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel('Mean AUROC')
    ax.set_title('평균 AUROC 비교 (Validation Set)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_auroc_summary_valid.png'), dpi=150)
    plt.close()
    print("✅ avg_auroc_summary_valid.png 저장")

# ── 메인 ──
def extract_tar(pretrained_dir):
    extract_dir = "/tmp/pretrained_models"
    os.makedirs(extract_dir, exist_ok=True)
    tar_path = os.path.join(pretrained_dir, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print(f"✅ model.tar.gz 압축 해제 → {extract_dir}")
    else:
        # 각 모델별 tar.gz 찾기
        for f in os.listdir(pretrained_dir):
            if f.endswith('.tar.gz'):
                with tarfile.open(os.path.join(pretrained_dir, f), "r:gz") as tar:
                    tar.extractall(path=extract_dir)
                print(f"✅ {f} 압축 해제")
    return extract_dir

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Validation 비교 평가 시작 (Device: {device})")

    # 모델 디렉토리 준비 (각 채널에서 tar.gz 해제)
    model_dirs = {}
    for name, filename in MODELS.items():
        channel_key = name.lower().replace('-', '').replace(' ', '')
        channel_env = f"SM_CHANNEL_{channel_key.upper()}"
        channel_dir = os.environ.get(channel_env, args.pretrained_dir)
        extract_dir = f"/tmp/model_{channel_key}"
        os.makedirs(extract_dir, exist_ok=True)
        tar_path = os.path.join(channel_dir, "model.tar.gz")
        if os.path.exists(tar_path):
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)
            print(f"✅ {name} 압축 해제: {extract_dir}")
        else:
            extract_dir = channel_dir
        model_dirs[name] = extract_dir

    # validation 데이터
    valid_csv = os.path.join(args.train_dir, args.valid_csv_name)
    valid_df = prepare_valid_df(valid_csv, args.image_dir)
    loader = DataLoader(
        ValidDataset(valid_df, transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])),
        batch_size=16, shuffle=False, num_workers=4
    )

    # U-Net 로드
    unet_path = os.path.join(args.weights_dir, 'unet_lung_mask_ep10.pth')
    print(f"✂️ U-Net 로드: {unet_path}")
    unet = UNet(n_channels=3, n_classes=1).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device))
    unet.eval()

    # 3개 모델 순차 추론
    metrics_dict = {}
    labels_dict = {}
    preds_dict = {}

    for name, filename in MODELS.items():
        model_path = os.path.join(model_dirs[name], filename)
        print(f"\n🧠 [{name}] 모델 로드: {model_path}")
        engine = SooNetEngine(model_path=None)
        model = engine.model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"📊 [{name}] 추론 중...")
        labels, preds = run_inference(model, unet, loader, device)
        labels_dict[name] = labels
        preds_dict[name] = preds
        metrics_dict[name] = compute_metrics(labels, preds)
        print(f"✅ [{name}] 평균 AUROC: {metrics_dict[name]['AUROC'].mean():.4f}")

        # 개별 CSV 저장
        metrics_dict[name].to_csv(
            os.path.join(args.model_dir, f"valid_metrics_{name.lower().replace('-','').replace(' ','_')}.csv"),
            index=False
        )

    # 그래프 생성
    print("\n📈 그래프 생성 중...")
    plot_avg_auroc_summary(metrics_dict, args.model_dir)
    plot_auroc_comparison(metrics_dict, args.model_dir)
    plot_f1_comparison(metrics_dict, args.model_dir)
    plot_roc_curves(labels_dict, preds_dict, args.model_dir)
    plot_confusion_matrices(labels_dict, preds_dict, metrics_dict, args.model_dir)

    # 통합 비교 CSV
    all_rows = []
    for name, df in metrics_dict.items():
        df = df.copy()
        df['Model'] = name
        all_rows.append(df)
    combined = pd.concat(all_rows)
    combined.to_csv(os.path.join(args.model_dir, 'valid_comparison_all.csv'), index=False)

    # 최종 요약 출력
    print("\n" + "="*60)
    print("📊 최종 Validation 평균 AUROC 비교")
    print("="*60)
    for name, df in metrics_dict.items():
        print(f"  {name:12s}: AUROC {df['AUROC'].mean():.4f}  F1 {df['F1'].mean():.4f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir',      type=str, default=os.environ.get('SM_MODEL_DIR', '/tmp/output'))
    parser.add_argument('--train-dir',      type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--image-dir',      type=str, default=os.environ.get('SM_CHANNEL_IMAGES'))
    parser.add_argument('--weights-dir',    type=str, default=os.environ.get('SM_CHANNEL_WEIGHTS', '.'))
    parser.add_argument('--pretrained-dir', type=str, default=os.environ.get('SM_CHANNEL_PRETRAINED'))
    parser.add_argument('--valid-csv-name', type=str, default='valid.csv')
    evaluate(parser.parse_args())
