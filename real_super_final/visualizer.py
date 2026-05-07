import os
import numpy as np
import matplotlib
matplotlib.use('Agg') # 클라우드/터미널 환경 에러 방지
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve

class MedicalVisualizer:
    def __init__(self, labels, output_dir):
        self.labels = labels
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # 의료 논문 스타일의 폰트 및 스타일 세팅
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

    def generate_all_reports(self, val_labels, val_preds, best_thresholds, history):
        print(f"\n🎨 [Visualizer] 7대 의료 시각화 지표 생성을 시작합니다...")
        self.plot_roc_curve(val_labels, val_preds)
        self.plot_pr_curve(val_labels, val_preds)
        self.plot_f1_vs_threshold(val_labels, val_preds)
        self.plot_multi_confusion_matrix(val_labels, val_preds, best_thresholds)
        self.plot_co_occurrence(val_preds, best_thresholds)
        self.plot_calibration_curve(val_labels, val_preds)
        if history:
            self.plot_learning_curves(history)
        print(f"✅ 모든 시각화 지표가 '{self.output_dir}' 폴더에 저장되었습니다!")

    # 1. ROC Curve
    def plot_roc_curve(self, y_true, y_pred):
        plt.figure(figsize=(12, 10))
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
                plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc(fpr, tpr):.3f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Multi-label)')
        plt.legend(loc="lower right", fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '01_ROC_Curve.png'), dpi=300)
        plt.close()

    # 2. PR Curve
    def plot_pr_curve(self, y_true, y_pred):
        plt.figure(figsize=(12, 10))
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                prec, rec, _ = precision_recall_curve(y_true[:, i], y_pred[:, i])
                plt.plot(rec, prec, lw=2, label=label)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left", fontsize=9, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '02_PR_Curve.png'), dpi=300)
        plt.close()

    # 3. F1 vs Threshold Curve
    def plot_f1_vs_threshold(self, y_true, y_pred):
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                prec, rec, thresholds = precision_recall_curve(y_true[:, i], y_pred[:, i])
                # 길이가 안 맞는 부분 조정
                prec, rec = prec[:-1], rec[:-1]
                f1_scores = np.divide(2 * rec * prec, rec + prec, out=np.zeros_like(rec), where=(rec + prec) != 0)
                
                axes[i].plot(thresholds, f1_scores, lw=2, color='coral')
                best_idx = np.argmax(f1_scores)
                axes[i].axvline(thresholds[best_idx], color='red', linestyle='--', label=f'Best Th: {thresholds[best_idx]:.2f}')
                axes[i].set_title(label)
                axes[i].set_xlabel('Threshold')
                axes[i].set_ylabel('F1 Score')
                axes[i].legend()
            else:
                axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '03_F1_vs_Threshold.png'), dpi=300)
        plt.close()

    # 4. Multi-label Confusion Matrix (14 Grid)
    def plot_multi_confusion_matrix(self, y_true, y_pred, best_thresholds):
        fig, axes = plt.subplots(4, 4, figsize=(18, 18))
        axes = axes.flatten()
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                # 최적의 임계값으로 이진화(Binarize)
                pred_bin = (y_pred[:, i] >= best_thresholds[i]).astype(int)
                cm = confusion_matrix(y_true[:, i], pred_bin)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
                axes[i].set_title(label)
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
            else:
                axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '04_Multi_Confusion_Matrix.png'), dpi=300)
        plt.close()

    # 5. Co-occurrence Heatmap (동반 질환 상관관계)
    def plot_co_occurrence(self, y_pred, best_thresholds):
        # 14개 질환 예측을 이진화
        bin_preds = np.array([(y_pred[:, i] >= best_thresholds[i]).astype(int) for i in range(len(self.labels))]).T
        # 동시 발생 행렬 계산 (Transpose 행렬곱)
        co_occ = np.dot(bin_preds.T, bin_preds)
        # 대각 성분(자기 자신)은 0으로 만들어 다른 질환과의 관계를 돋보이게 함
        np.fill_diagonal(co_occ, 0)
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(co_occ, xticklabels=self.labels, yticklabels=self.labels, cmap='Reds', annot=True, fmt='d')
        plt.title('Disease Co-occurrence Patterns (Predicted)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '05_Disease_Co_occurrence.png'), dpi=300)
        plt.close()

    # 6. Calibration Curve (신뢰도 다이어그램)
    def plot_calibration_curve(self, y_true, y_pred):
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        for i, label in enumerate(self.labels):
            if len(np.unique(y_true[:, i])) > 1:
                prob_true, prob_pred = calibration_curve(y_true[:, i], y_pred[:, i], n_bins=10)
                axes[i].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
                axes[i].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
                axes[i].set_title(label)
                axes[i].set_xlabel('Mean Predicted Probability')
                axes[i].set_ylabel('Fraction of Positives')
                axes[i].legend()
            else:
                axes[i].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '06_Calibration_Curve.png'), dpi=300)
        plt.close()

    # 7. Learning Curves (학습 트렌드)
    def plot_learning_curves(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss Curve
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # AUROC Curve
        if 'val_auroc' in history and len(history['val_auroc']) > 0:
            ax2.plot(epochs, history['val_auroc'], 'g-', label='Val Macro AUROC')
            ax2.set_title('Validation AUROC Trend')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('AUROC')
            ax2.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '07_Learning_Curves.png'), dpi=300)
        plt.close()