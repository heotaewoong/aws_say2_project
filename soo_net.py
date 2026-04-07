import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class SooNetEngine:
    def __init__(self, model_path=None):
        # GPU 가속 세팅
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. 아키텍처 정의 (DenseNet-121)
        if model_path and os.path.exists(model_path):
            # 1. 추론 모드: 어차피 덮어쓸 거니까 무거운 ImageNet을 부르지 않음 (None)
            self.model = models.densenet121(weights=None)
        else:
            # 2. 학습 모드: 맨땅에 헤딩하지 않도록 전이 학습(Transfer Learning) 베이스 로드
            print("💡 전이 학습(Transfer Learning)을 위해 ImageNet 가중치를 베이스로 로드합니다.")
            self.model = models.densenet121(weights='IMAGENET1K_V1')
        
        # 🚀 [수정 1] TXV 방식: 1채널(흑백) 입력을 받도록 첫 번째 Conv 레이어 교체
        # 원래 DenseNet은 3채널(RGB)을 받지만, 1채널 입력에 맞게 입구를 뜯어고칩니다.
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 2. 분류기(Classifier) 수정 (14개 질환)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 14)
        
        # 3. 모델 가중치 로드
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            print(f"✅ 고해상도(448) 1채널 가중치 로드 완료: {model_path}")
        else:
            print("⚠️ 가중치 파일이 없어 초기화된 모델을 사용합니다.")
            
        self.model.to(self.device)
        self.model.eval()

        # --- Grad-CAM을 위한 추가 설정 ---
        self.gradients = None
        self.activations = None
        # 448x448 입력 시 이 위치에서 (1024, 14, 14)의 고해상도 피처맵이 잡힙니다.
        self.target_layer = self.model.features.norm5 
        # -------------------------------

        # 4. 14가지 소견 라벨 및 HPO 매핑 (기존과 동일)
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
            "Lung Opacity", "No Finding", "Pleural Effusion", 
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        # 14개 라벨과 1:1로 완벽하게 매칭된 HPO Dictionary
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

    # --- Grad-CAM Hook 메서드 ---
    def _extract_gradients(self, grad):
        self.gradients = grad

    def _save_activations_and_hook_grad(self, module, input, output):
        self.activations = output
        output.register_hook(self._extract_gradients)

    def _preprocess(self, image_path):
        """🚀 [수정 2] 1채널 변환, 448x448 리사이즈 및 TXV 스케일링 적용"""
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor() # 0.0 ~ 1.0 사이 값으로 변환
        ])
        
        # ImageNet 스타일(RGB)이 아닌, 1채널 흑백(Grayscale) 모드 'L'로 불러옵니다.
        pil_image = Image.open(image_path).convert('L') 
        input_tensor = transform(pil_image) # [1, 448, 448]
        
        # 🚀 TXV 공식 스케일링 적용: 0~1 범위를 -1024 ~ 1024 범위로 확장
        input_tensor = (input_tensor * 2048.0) - 1024.0
        
        # 배치 차원 추가 -> [1, 1, 448, 448]
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        
        # 시각화용 BGR 이미지는 448 해상도에 맞춰 3채널로 복원
        original_img = np.array(pil_image.convert('RGB'))
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.resize(original_img, (448, 448))
        
        return input_tensor, original_img

    def extract_vision_hpos(self, image_path, threshold=0.3):
        """이미지를 분석하여 임계값 이상의 소견과 HPO 코드를 반환"""
        input_tensor, _ = self._preprocess(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probabilities = torch.sigmoid(outputs[0]).tolist()
        results = []
        
        print(f"\n--- [Vision 분석 결과: {os.path.basename(image_path)}] ---")
        for i, prob in enumerate(probabilities):
            label = self.labels[i]
            if prob >= threshold:
                hpo_id = self.hpo_map.get(label, "Unknown")
                results.append({
                    'finding': label, 
                    'hpo_id': hpo_id, 
                    'score': prob,
                    'index': i  
                })
                print(f"✅ 검출: {label:<18} | 확률: {prob:.4f} | HPO: {hpo_id}")
                
        return results

    def get_cam_visualize(self, image_path, target_class_index, output_path='heatmap_result.png'):
        """고해상도 피처맵(14x14)을 활용한 정밀 Grad-CAM"""
        print(f"🔥 '{self.labels[target_class_index]}' 질환에 대한 Grad-CAM 생성 중...")
        
        input_tensor, original_img = self._preprocess(image_path)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        handle = self.target_layer.register_forward_hook(self._save_activations_and_hook_grad)

        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        score = outputs[0][target_class_index]
        score.backward()

        handle.remove()

        if self.gradients is None or self.activations is None:
            print("❌ 그래디언트 캡처 실패.")
            return None

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        # 🚀 원래 224였던 리사이즈를 448로 변경하여 더 정밀한 해상도로 뿌려줍니다.
        cam = cv2.resize(cam, (448, 448))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        label_text = f"{self.labels[target_class_index]} ({score.item():.2%})"
        cv2.putText(result_img, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(output_path, result_img)
        print(f"✅ Grad-CAM 결과 저장 완료: {output_path}")
        
        return output_path