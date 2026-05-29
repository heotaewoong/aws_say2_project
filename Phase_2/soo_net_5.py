import os
import re
import ast
import random
import json
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
import torchxrayvision as xrv

from anatomy_preprocessor import AnatomyPreprocessor

# =====================================================================
# 1. Architecture Components (V5)
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
        x = torch.cat((cls_tokens, x), dim=1) # [B, 4, 1024]

        attn_output, _ = self.attention(x, x, x)
        x = self.norm(x + attn_output)
        # Return Global, Lung, Heart tokens (exclude CLS token)
        return x[:, 1:, :] # [B, 3, 1024]


class AnatomySooNetV5(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super().__init__()

        if pretrained:
            print("Loading xrv pretrained backbone (densenet121-res224-all)...")
            xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
            self.features = xrv_model.features
            print("✅ xrv pretrained backbone loaded")
        else:
            densenet = models.densenet121(weights=None)
            densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.features = densenet.features

        self.soft_attention = AnatomicalSoftAttention(1024)
        self.a3_module = AttentionAggregation(1024)
        
        # v5 Anatomy Knowledge Matrix: [Global, Lung, Heart] weights per label
        # Based on CXR_Findings_Guide.docx
        weights = torch.tensor([
            [0.2, 0.7, 0.1], # 0: Atelectasis (Lung-Heavy)
            [0.1, 0.0, 0.9], # 1: Cardiomegaly (Heart-Only)
            [0.1, 0.9, 0.0], # 2: Consolidation (Lung-Only)
            [0.2, 0.4, 0.4], # 3: Edema (Lung/Heart balanced)
            [0.2, 0.0, 0.8], # 4: Enlarged Cardiomediastinum (Heart-Heavy)
            [0.8, 0.1, 0.1], # 5: Fracture (Global/Bone focus)
            [0.1, 0.9, 0.0], # 6: Lung Lesion (Lung-Only)
            [0.1, 0.9, 0.0], # 7: Lung Opacity (Lung-Only)
            [0.8, 0.1, 0.1], # 8: No Finding (Global)
            [0.2, 0.8, 0.0], # 9: Pleural Effusion (Lung-Base/Edge)
            [0.2, 0.8, 0.0], # 10: Pleural Other (Lung-Edge)
            [0.1, 0.9, 0.0], # 11: Pneumonia (Lung-Only)
            [0.2, 0.8, 0.0], # 12: Pneumothorax (Lung-Periphery)
            [0.7, 0.1, 0.2], # 13: Support Devices (Global/Heart focus)
        ], dtype=torch.float32)
        self.register_buffer('anatomy_weights', weights)

        self.classifier = nn.Linear(1024, num_classes)
        self._attended_feat = None

    def forward(self, img, masks):
        feat_map = F.relu(self.features(img), inplace=True)
        masks_resized = F.interpolate(masks, size=feat_map.shape[2:], mode='bilinear')
        attended_feat = self.soft_attention(feat_map, masks_resized)
        self._attended_feat = attended_feat

        tokens = []
        tokens.append(attended_feat.mean(dim=(2, 3))) # Global
        for k in range(2): # Lung, Heart
            m = masks_resized[:, k:k+1, :, :]
            pooled = (attended_feat * m).sum(dim=(2, 3)) / (m.sum(dim=(2, 3)) + 1e-6)
            tokens.append(pooled)

        x = torch.stack(tokens, dim=1) # [B, 3, 1024]
        refined_tokens = self.a3_module(x) # Interaction
        
        # Apply anatomy weights: [B, 3, 1024] * [14, 3] -> [B, 14, 1024]
        # For each label c, we aggregate the 3 tokens using anatomy_weights[c]
        weighted_x = torch.einsum('btd,ct->bcd', refined_tokens, self.anatomy_weights)
        
        # final dot product with classifier weight per class
        # logits: [B, 14]
        logits = (weighted_x * self.classifier.weight.unsqueeze(0)).sum(dim=-1) + self.classifier.bias
        return logits


# =====================================================================
# 2. Inference Engine (SooNetEngineV5)
# =====================================================================
class SooNetEngineV5:
    def __init__(self, model_path=None, unet_path=None, num_classes=14):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

        use_pretrained = (model_path is None) or (not os.path.exists(model_path))
        self.model = AnatomySooNetV5(num_classes, pretrained=use_pretrained).to(self.device)
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"AnatomySooNetV5 loaded from {model_path}")
        self.model.eval()

        self.preprocessor = AnatomyPreprocessor(unet_path, self.device)

        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]

        # Updated HPO Mapping (Corrected in V5)
        self.hpo_map = {
            "Atelectasis": "HP:0100750", 
            "Cardiomegaly": "HP:0001640", 
            "Consolidation": "HP:0032177", 
            "Edema": "HP:0100598", 
            "Enlarged Cardiomediastinum": "HP:0034501", 
            "Fracture": "HP:0002757", 
            "Lung Lesion": "HP:0032338", 
            "Lung Opacity": "HP:0031457", 
            "No Finding": "Normal (N/A)", 
            "Pleural Effusion": "HP:0002202", 
            "Pleural Other": "HP:0002102", 
            "Pneumonia": "HP:0002090",
            "Pneumothorax": "HP:0002107", 
            "Support Devices": "Device (N/A)"
        }

    def _log_error(self, method_name, image_path, error):
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "image_path": image_path,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"error_{method_name}_{timestamp_str}.json"
        log_path = os.path.join(log_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        print(f"Error logged to {log_path}")

    def predict(self, image_path):
        try:
            txv_img, final_masks, _, _ = self.preprocessor.process_image(image_path)
            with torch.no_grad():
                outputs = self.model(txv_img, final_masks)
            probs = torch.sigmoid(outputs[0]).cpu().numpy()
            return {label: (probs[i], self.hpo_map.get(label, "N/A")) for i, label in enumerate(self.labels)}
        except Exception as e:
            self._log_error("predict", image_path, e)
            return None

    def get_cam_visualize(self, image_path, target_class_index, output_path):
        try:
            txv_img, final_masks, pil_image, crop_coords = self.preprocessor.process_image(image_path)
            txv_img = txv_img.clone().detach().requires_grad_(True)
            original_img = cv2.resize(cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), (512, 512))

            self.model.zero_grad()
            outputs = self.model(txv_img, final_masks)
            
            attended = self.model._attended_feat
            if attended is None:
                cv2.imwrite(output_path, original_img)
                return output_path

            attended.retain_grad()
            score = outputs[0][target_class_index]
            score.backward()

            gradients = attended.grad.detach().cpu().numpy()[0]
            activations = attended.detach().cpu().numpy()[0]
            weights = np.mean(gradients, axis=(1, 2))
            cam = np.maximum(np.dot(weights, activations.reshape(activations.shape[0], -1)).reshape(activations.shape[1:]), 0)

            x1, y1 = max(int(crop_coords['x1']), 0), max(int(crop_coords['y1']), 0)
            x2, y2 = min(int(crop_coords['x2']), 512), min(int(crop_coords['y2']), 512)
            crop_h, crop_w = max(y2 - y1, 1), max(x2 - x1, 1)

            full_cam = np.zeros((512, 512), dtype=np.float32)
            cam_resized = cv2.resize(cam, (crop_w, crop_h))
            full_cam[y1:y1+crop_h, x1:x1+crop_w] = cam_resized

            if full_cam.max() > 0:
                full_cam = (full_cam - full_cam.min()) / (full_cam.max() - full_cam.min() + 1e-8)

            heatmap = cv2.applyColorMap(np.uint8(255 * full_cam), cv2.COLORMAP_JET)
            result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
            cv2.imwrite(output_path, result_img)
            return output_path
        except Exception as e:
            self._log_error("get_cam_visualize", image_path, e)
            return None
