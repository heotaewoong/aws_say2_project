
import os
import cv2
import torch
import json
import traceback
import datetime
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.ops as ops
import torchvision.transforms.functional as TF
from unet_lung_model import UNet

# =====================================================================
# 1. Chest X-ray Preprocessing Module (CLAHE & Pad)
# =====================================================================
class ChestXrayPreprocess:
    def __init__(self, target_size=(512, 512), clip_limit=2.0):
        self.target_size = target_size
        self.clip_limit = clip_limit

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        img_np = np.array(img.convert('L'))
        img_clahe = clahe.apply(img_np)
        img_pil = Image.fromarray(img_clahe).convert('RGB')
        return ImageOps.pad(img_pil, self.target_size, method=Image.BILINEAR, color=(0, 0, 0))

# =====================================================================
# 2. Anatomy Preprocessor (Masking & ROI Cropping)
# =====================================================================
class AnatomyPreprocessor:
    """
    Handles Lung/Heart mask extraction via UNet and ROI-based cropping/resizing.
    Separated from the main model for modular use.
    """
    def __init__(self, unet_path, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize and load UNet
        self.unet = UNet(n_channels=3, n_classes=3).to(self.device)
        if unet_path and os.path.exists(unet_path):
            checkpoint = torch.load(unet_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.unet.load_state_dict(state_dict)
        self.unet.eval()

        self.clahe_processor = ChestXrayPreprocess(target_size=(512, 512), clip_limit=2.0)

    def _log_error(self, method_name, input_id, error):
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "image_path": str(input_id),
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"error_{method_name}_{timestamp_str}.json"
        log_path = os.path.join(log_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        print(f"Error logged to {log_path}")

    def crop_and_resize_roi(self, img_tensor, mask_tensor, target_size=(512, 512), padding=10):
        """
        Original ROI Align logic from SooNetEngine.
        """
        B, C_img, H, W = img_tensor.shape
        device = img_tensor.device

        combined_mask = mask_tensor.sum(dim=1) > 0.5
        is_empty = ~torch.any(combined_mask.view(B, -1), dim=1)

        rows = torch.any(combined_mask, dim=2)
        cols = torch.any(combined_mask, dim=1)

        y_min_indices = torch.argmax(rows.int(), dim=1)
        y_max_indices = H - 1 - torch.argmax(torch.flip(rows, dims=[1]).int(), dim=1)
        x_min_indices = torch.argmax(cols.int(), dim=1)
        x_max_indices = W - 1 - torch.argmax(torch.flip(cols, dims=[1]).int(), dim=1)

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

        batch_indices = torch.arange(B, device=device).view(-1, 1).float()
        boxes_for_roi = torch.stack([roi_x1, roi_y1, roi_x2, roi_y2], dim=1)
        boxes_for_roi[is_empty] = 0.0
        boxes_for_roi = torch.cat([batch_indices, boxes_for_roi], dim=1)

        resized_imgs_rgb = ops.roi_align(img_tensor, boxes_for_roi, output_size=target_size, aligned=True)
        final_masks = ops.roi_align(mask_tensor, boxes_for_roi, output_size=target_size, aligned=True)

        final_imgs = resized_imgs_rgb.mean(dim=1, keepdim=True)
        final_imgs[is_empty] = 0.0
        final_masks[is_empty] = 0.0

        # TXV Scaling (-1024 ~ 1024)
        txv_scaled_imgs = (final_imgs * 2048.0) - 1024.0

        crop_coords = {
            'x1': x_mins[0].item(), 'y1': y_mins[0].item(),
            'x2': x_maxs[0].item(), 'y2': y_maxs[0].item(),
            'orig_h': H, 'orig_w': W
        }
        return txv_scaled_imgs, final_masks, crop_coords

    def extract_masks(self, base_tensor):
        """
        Extract Lung and Heart masks from the base tensor.
        """
        with torch.no_grad():
            unet_output = self.unet(base_tensor)
            probs = torch.softmax(unet_output, dim=1)  # (B, 3, 512, 512)
            lung = probs[:, 1:2, :, :]
            heart = probs[:, 2:3, :, :]
            # Clean lung by removing heart overlap
            lung_clean = lung * (1 - (heart > 0.5).float())
            raw_masks = torch.cat([lung_clean, heart], dim=1)  # (B, 2, 512, 512)
        return raw_masks

    def process_image(self, image_input, target_size=(512, 512)):
        """
        Full pipeline: CLAHE -> UNet -> ROI Align -> Final Tensors
        image_input: path (str) or PIL Image
        """
        try:
            if isinstance(image_input, str):
                pil_img = Image.open(image_input).convert('RGB')
            else:
                pil_img = image_input.convert('RGB')

            processed_pil = self.clahe_processor(pil_img)
            base_tensor = transforms.ToTensor()(processed_pil).unsqueeze(0).to(self.device)

            raw_masks = self.extract_masks(base_tensor)
            txv_img, final_masks, crop_coords = self.crop_and_resize_roi(base_tensor, raw_masks, target_size=target_size)
            
            return txv_img, final_masks, pil_img, crop_coords
        except Exception as e:
            input_id = image_input if isinstance(image_input, str) else "PIL.Image"
            self._log_error("process_image", input_id, e)
            return None, None, None, None

    def export_preprocessed_data(self, image_path, output_img_path, output_mask_path):
        """
        Offline batch processing helper: Saves 512x512 image and mask PNGs.
        Mask PNG: R=Lung, G=Heart
        """
        txv_img, final_masks, _, _ = self.process_image(image_path)
        
        # 텐서 -> numpy (0~1 range for saving)
        # txv_img는 (-1024~1024)이므로 다시 (0~1)로 복구
        img_np = ((txv_img.cpu().squeeze().numpy() + 1024.0) / 2048.0 * 255.0).astype(np.uint8)
        mask_np = (final_masks.cpu().squeeze().numpy() * 255.0).astype(np.uint8)
        
        # 이미지 저장
        cv2.imwrite(output_img_path, img_np)
        
        # 마스크 저장 (B: Lung, G: Heart)
        combined_mask = np.zeros((512, 512, 3), dtype=np.uint8)
        combined_mask[:, :, 0] = mask_np[0] # Lung
        combined_mask[:, :, 1] = mask_np[1] # Heart
        cv2.imwrite(output_mask_path, combined_mask)
        
        return output_img_path, output_mask_path
