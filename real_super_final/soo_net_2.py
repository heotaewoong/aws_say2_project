import os
import re
import ast
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torchvision.ops as ops
import torchvision.transforms.functional as TF

from unet_lung_model import UNet

# =====================================================================
# 1. Preprocessing Modules
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(512, 512), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        return ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))

# =====================================================================
# 2. Architecture Components
# =====================================================================    
class AnatomicalSoftAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_channel_attention = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, masks):
        attention_map = self.spatial_channel_attention(masks)
        return x * (1 + attention_map)

class AttentionAggregation(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=4):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        return x[:, 0, :]

class AnatomySooNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        densenet = models.densenet121(weights=None)
        densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.features = densenet.features
        
        self.soft_attention = AnatomicalSoftAttention(1024)
        self.a3_module = AttentionAggregation(1024)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, img, masks):
        # 1. Feature Extraction (B, 1024, 16, 16)
        feat_map = F.relu(self.features(img), inplace=True)
        
        # 2. Masks Resize (B, 2, 16, 16)
        masks_resized = F.interpolate(masks, size=feat_map.shape[2:], mode='bilinear')
        
        # 3. Apply Anatomical Soft Attention 🌟 (Anatomy-XNet 핵심)
        attended_feat = self.soft_attention(feat_map, masks_resized)
        
        # 4. PWAP (Part-Wise Average Pooling)
        tokens = []
        # Global
        tokens.append(attended_feat.mean(dim=(2, 3)))
        # Local (Lung, Heart)
        for k in range(2):
            m = masks_resized[:, k:k+1, :, :]
            pooled = (attended_feat * m).sum(dim=(2, 3)) / (m.sum(dim=(2, 3)) + 1e-6)
            tokens.append(pooled)
            
        # 5. Transformer Fusion (B, 3, 1024)
        x = torch.stack(tokens, dim=1)
        aggregated = self.a3_module(x)
        return self.classifier(aggregated)

# =====================================================================
# 3. Inference Engine
# =====================================================================
class SooNetEngine:
    def __init__(self, model_path=None, unet_path=None, num_classes=14):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize and load Anatomy-SooNet
        self.model = AnatomySooNet(num_classes).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"AnatomySooNet loaded from {model_path}")
        self.model.eval()

        # Initialize and load UNet
        self.unet = UNet(n_channels=3, n_classes=3).to(self.device)
        if unet_path and os.path.exists(unet_path):
            checkpoint = torch.load(unet_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.unet.load_state_dict(state_dict)
            print(f"UNet loaded from {unet_path}")
        self.unet.eval()

        self.clahe_preprocess = ChestXrayPreprocess(target_size=(512, 512), clip_limit=2.0)

        # Grad-CAM components        
        self.gradients = None
        self.activations = None
        self.target_layer = self.model.features.norm5
       
        # Labels and HPO Mappings
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
            "Lung Opacity", "No Finding", "Pleural Effusion", 
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        self.hpo_map = {
            "Atelectasis": "HP:0002095",
            "Cardiomegaly": "HP:0001640",
            "Consolidation": "HP:0002113",               # Pulmonary infiltrate/consolidation
            "Edema": "HP:0002111",                       # Pulmonary edema
            "Enlarged Cardiomediastinum": "HP:0034251",  # Widened mediastinum
            "Fracture": "HP:0002757",                    # Bone fracture
            "Lung Lesion": "HP:0025000",                 # Pulmonary nodule/lesion
            "Lung Opacity": "HP:0002088",                # Abnormality of lung morphology
            "No Finding": "Normal (N/A)",                # 정상 소견은 HPO 코드가 없음
            "Pleural Effusion": "HP:0002202",
            "Pleural Other": "HP:0002102",               # Abnormality of the pleura
            "Pneumonia": "HP:0002090",
            "Pneumothorax": "HP:0002107",
            "Support Devices": "Device (N/A)"            # 의료기기는 신체적 질병(Phenotype)이 아님
        }

    def _crop_and_resize(self, img_tensor, mask_tensor, target_size=(512, 512), padding=10):
        B, C_img, H, W = img_tensor.shape
        device = img_tensor.device

        # 1. Bounding Box 계산
        combined_mask = mask_tensor.sum(dim=1) > 0.5
        is_empty = ~torch.any(combined_mask.view(B, -1), dim=1)
        
        rows = torch.any(combined_mask, dim=2)
        cols = torch.any(combined_mask, dim=1)

        y_min_indices = torch.argmax(rows.int(), dim=1)
        y_max_indices = H - 1 - torch.argmax(torch.flip(rows, dims=[1]).int(), dim=1)
        x_min_indices = torch.argmax(cols.int(), dim=1)
        x_max_indices = W - 1 - torch.argmax(torch.flip(cols, dims=[1]).int(), dim=1)

        # 2. 정사각형 BBox로 변환
        y_mins = torch.clamp(y_min_indices - padding, min=0).float()
        y_maxs = torch.clamp(y_max_indices + padding, max=H).float()
        x_mins = torch.clamp(x_min_indices - padding, min=0).float()
        x_maxs = torch.clamp(x_max_indices + padding, max=W).float()

        crop_heights = y_maxs - y_mins
        crop_widths = x_maxs - x_mins
        center_y = (y_mins + y_maxs) / 2
        center_x = (x_mins + x_maxs) / 2
        max_dims = torch.max(crop_heights, crop_widths)
        
        roi_x1 = center_x - max_dims / 2
        roi_y1 = center_y - max_dims / 2
        roi_x2 = center_x + max_dims / 2
        roi_y2 = center_y + max_dims / 2

        # 3. roi_align을 위한 Box 리스트 생성
        batch_indices = torch.arange(B, device=device).view(-1, 1).float()
        boxes_for_roi = torch.stack([roi_x1, roi_y1, roi_x2, roi_y2], dim=1)
        boxes_for_roi[is_empty] = 0.0
        boxes_for_roi = torch.cat([batch_indices, boxes_for_roi], dim=1)

        # 4. roi_align으로 크롭과 리사이즈 수행
        resized_imgs_rgb = ops.roi_align(img_tensor, boxes_for_roi, output_size=target_size, aligned=True)
        final_masks = ops.roi_align(mask_tensor, boxes_for_roi, output_size=target_size, aligned=True)

        # 5. 후처리
        final_imgs = resized_imgs_rgb.mean(dim=1, keepdim=True)
        final_imgs[is_empty] = 0.0
        final_masks[is_empty] = 0.0
        
        # DenseNet specific normalization
        txv_scaled_imgs = (final_imgs * 2048.0) - 1024.0
        return txv_scaled_imgs, final_masks

    def _preprocess(self, image_path):
        pil_img = Image.open(image_path).convert('RGB')
        
        processed_pil = self.clahe_preprocess(pil_img)
        base_tensor = transforms.ToTensor()(processed_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            raw_masks = torch.sigmoid(self.unet(base_tensor))
            raw_masks = raw_masks[:, 1:, :, :] # (1, 2, 512, 512)

        txv_img, final_masks = self._crop_and_resize(base_tensor, raw_masks)   
        return txv_img.to(self.device), final_masks.to(self.device), pil_img

    def predict(self, image_path):
        txv_img, final_masks, _ = self._preprocess(image_path)
        with torch.no_grad():
            outputs = self.model(txv_img, final_masks)
        probs = torch.sigmoid(outputs[0]).cpu().numpy()
        return {label: (probs[i], self.hpo_map.get(label, "N/A")) for i, label in enumerate(self.labels)}
    
    def _extract_gradients(self, grad): 
        self.gradients = grad

    def _save_activations_and_hook_grad(self, module, input, output):
        self.activations = output
        output.register_hook(self._extract_gradients)

    def get_cam_visualize(self, image_path, target_class_index, output_path):
        txv_img, final_masks, pil_image = self._preprocess(image_path)
        txv_img = txv_img.clone().detach().requires_grad_(True)
        
        original_img = cv2.resize(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), (512, 512))

        handle = self.target_layer.register_forward_hook(self._save_activations_and_hook_grad)
        self.model.zero_grad()
        
        outputs = self.model(txv_img, final_masks)
        score = outputs[0][target_class_index]
        score.backward()
        handle.remove()

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.maximum(np.dot(weights, activations.reshape(activations.shape[0], -1)).reshape(activations.shape[1:]), 0)
        
        cam = cv2.resize(cam, (512, 512))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        cv2.imwrite(output_path, result_img)
        return output_path

# =====================================================================
# 4. Evaluation Engine
# =====================================================================
class SooNetTester:
    def __init__(self, engine, gt_csv_path, val_csv_path, img_root):
        self.engine = engine
        self.img_root = img_root
        self.gt_df = pd.read_csv(gt_csv_path) if os.path.exists(gt_csv_path) else None
        self.val_df = pd.read_csv(val_csv_path) if os.path.exists(val_csv_path) else None
        print("Test Engine and Datasets loaded successfully.")

    def get_random_test_path(self):
        if self.val_df is None:
            print("Validation CSV not loaded.")
            return None

        all_paths = []
        for _, row in self.val_df.iterrows():
            for col in ['AP', 'PA']:
                raw_val = str(row[col])
                if raw_val != 'nan':
                    try:
                        path_list = ast.literal_eval(raw_val)
                        all_paths.extend(path_list)
                    except:
                        continue
        
        if not all_paths:
            print("No valid image paths found.")
            return None

        max_attempts = 100
        for _ in range(max_attempts):
            random_rel_path = random.choice(all_paths)
            full_path = os.path.normpath(os.path.join(self.img_root, random_rel_path))
            
            if os.path.exists(full_path):
                print(f"Random image selected: {random_rel_path}")
                return full_path
        
        print(f"Failed to find a valid image file after {max_attempts} attempts.")
        return None

    def _get_gt(self, image_path):
        if self.gt_df is None: return None
        
        try:
            match = re.search(r's(\d{8})', image_path)
            if not match:
                print(f"경로에서 Study ID를 찾을 수 없음: {image_path}")
                return None
            
            study_id = int(match.group(1))            
            row = self.gt_df[self.gt_df['study_id'].astype(int) == study_id]
            
            if not row.empty:
                res_dict = {}
                for label in self.engine.labels:
                    if label in row.columns:
                        val = row.iloc[0][label]
                        res_dict[label] = 1.0 if val == 1.0 or val == -1.0 else 0.0
                return res_dict
            else:
                print(f"CSV 내에 Study ID {study_id}가 존재하지 않음")
        except Exception as e:
            print(f"GT mapping error: {e}")
        return None

    def run_inference_with_gt(self, image_path, threshold=0.4):
        predictions = self.engine.predict(image_path)
        gt_data = self._get_gt(image_path)
        
        print(f"\n[ Inference Report: {os.path.basename(image_path)} ]")
        print(f"{'Disease Finding':<25} | {'Prob':<8} | {'GT':<8} | {'HPO Code':<12} | {'Status'}")
        print("-" * 80)

        detected_indices = []
        for i, label in enumerate(self.engine.labels):
            prob, hpo = predictions[label]
            gt_val = gt_data.get(label, 0.0) if gt_data else "N/A"
            gt_str = "Positive" if gt_val == 1.0 else "Negative" if gt_val == 0.0 else "N/A"
            
            status = ""
            if prob >= threshold:
                detected_indices.append(i)
                status = "DETECTED" if gt_val == 1.0 else "FP (Check)"
            elif gt_val == 1.0:
                status = "MISS (FN)"

            if prob >= threshold or gt_val == 1.0:
                print(f"{label:<25} | {prob:>7.2%} | {gt_str:<8} | {hpo:<12} | {status}")
        
        return detected_indices

    def generate_report_heatmaps(self, image_path, threshold=0.35, output_dir='reports'):
        indices = self.run_inference_with_gt(image_path, threshold)
        if not indices: return

        if not os.path.exists(output_dir): os.makedirs(output_dir)
        for idx in indices:
            label = self.engine.labels[idx].replace(" ", "_")
            output_path = os.path.join(output_dir, f"{label}_cam.png")
            self.engine.get_cam_visualize(image_path, idx, output_path)
        print(f"Generated {len(indices)} heatmaps in '{output_dir}'.")

# =====================================================================
# 5. Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    IMG_ROOT = "data/mimic-iv-cxr/official_data_iccv_final" 
    VAL_CSV = "data/mimic-iv-cxr/mimic_cxr_aug_validate.csv" 
    GT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv" 
    SOONET_WEIGHT_PATH = "model/anatomy_soonet_unified_best.pth"
    UNET_WEIGHT_PATH = "model/unet_lung_heart_best.pth"
    
    engine = SooNetEngine(model_path=SOONET_WEIGHT_PATH, unet_path=UNET_WEIGHT_PATH)
    tester = SooNetTester(engine, gt_csv_path=GT_CSV, val_csv_path=VAL_CSV, img_root=IMG_ROOT)
    
    random_test_img = tester.get_random_test_path()
    
    if random_test_img:
        tester.generate_report_heatmaps(random_test_img, threshold=0.35, output_dir='random_test_results')