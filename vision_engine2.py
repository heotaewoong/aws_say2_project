import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

class CheXNetEngine:
    def __init__(self, model_path=None):
        # 1. 아키텍처 정의 (DenseNet-121)
        self.model = models.densenet121(weights=None)
        num_ftrs = self.model.classifier.in_features
        
        # CheXNet의 14개 질환 출력을 위해 최종 레이어 수정
        self.model.classifier = nn.Linear(num_ftrs, 14)
        
        # 2. 모델 가중치 로드
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(checkpoint)
            print(f"✅ CheXNet 가중치 로드 완료: {model_path}")
        else:
            print("⚠️ 가중치 파일이 없어 초기화된 모델을 사용합니다. (테스트용)")
            
        self.model.eval()

        # --- Grad-CAM을 위한 추가 설정 ---
        self.gradients = None
        self.activations = None
        # DenseNet-121에서 마지막 컨볼루션 레이어 뒤의 정규화 층을 타겟으로 잡습니다.
        self.target_layer = self.model.features.norm5 
        # -------------------------------

        # 3. 14가지 소견 라벨 및 HPO 매핑
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", 
            "Lung Opacity", "No Finding", "Pleural Effusion", 
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        self.hpo_map = {
            "Atelectasis": "HP:0002095", "Cardiomegaly": "HP:0001640",
            "Effusion": "HP:0002202", "Infiltration": "HP:0002113",
            "Mass": "HP:0030048", "Nodule": "HP:0002092",
            "Pneumonia": "HP:0002090", "Pneumothorax": "HP:0002107",
            "Consolidation": "HP:0002113", "Edema": "HP:0002111",
            "Emphysema": "HP:0002097", "Fibrosis": "HP:0006530",
            "Pleural_Thickening": "HP:0005946", "Hernia": "HP:0000857"
        }

    # --- Grad-CAM Hook 메서드 ---
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def _save_activations(self, module, input, output):
        self.activations = output

    def _preprocess(self, image_path):
        """이미지 전처리 및 시각화용 원본 이미지 반환"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        pil_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # OpenCV용 원본 이미지 변환 (BGR)
        original_img = np.array(pil_image)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.resize(original_img, (224, 224)) # 모델 입력 크기에 맞춤
        
        return input_tensor, original_img

    def extract_vision_hpos(self, image_path, threshold=0.3):
        """이미지를 분석하여 임계값 이상의 소견과 HPO 코드를 반환"""
        input_tensor, _ = self._preprocess(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probabilities = outputs[0].tolist()
        results = []
        
        print(f"\n--- [Vision 분석 결과: {os.path.basename(image_path)}] ---")
        for i, prob in enumerate(probabilities):
            label = self.labels[i]
            if prob >= threshold:
                hpo_id = self.hpo_map.get(label, "Unknown")
                # 🔥 'index' 키를 추가하여 KeyError 해결
                results.append({
                    'finding': label, 
                    'hpo_id': hpo_id, 
                    'score': prob,
                    'index': i  
                })
                print(f"✅ 검출: {label:<18} | 확률: {prob:.4f} | HPO: {hpo_id}")
            else:
                # print(f"   미검출: {label:<18} | 확률: {prob:.4f}")
                pass
                
        return results

    def _extract_gradients(self, grad):
        self.gradients = grad

    def _save_activations_and_hook_grad(self, module, input, output):
        # 1. 활성화 맵 저장
        self.activations = output
        # 2. 출력 텐서(output)에 직접 backward hook을 걸어 gradients 가로채기
        # 이 방식은 'view + inplace' 에러를 발생시키지 않습니다.
        output.register_hook(self._extract_gradients)

    def get_cam_visualize(self, image_path, target_class_index, output_path='heatmap_result.png'):
        """최신 PyTorch 에러를 방지하는 안정적인 Grad-CAM 구현"""
        print(f"🔥 '{self.labels[target_class_index]}' 질환에 대한 Grad-CAM 생성 중...")
        
        input_tensor, original_img = self._preprocess(image_path)
        # 텐서 복제본을 사용하여 원본 데이터 변형 방지 (에러 방지 2단계)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        # 1. Forward Hook 등록 (텐서에 직접 Hook을 걸기 위해 수정)
        handle = self.target_layer.register_forward_hook(self._save_activations_and_hook_grad)

        # 2. 순전파 (Forward Pass)
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        # 3. 역전파 (Backward Pass)
        score = outputs[0][target_class_index]
        score.backward()

        # 4. Hook 해제 (메모리 누수 방지)
        handle.remove()

        # 5. Grad-CAM 계산
        # gradients와 activations가 제대로 캡처되었는지 확인
        if self.gradients is None or self.activations is None:
            print("❌ 그래디언트 캡처 실패. 모델의 레이어 설정을 확인하세요.")
            return None

        # 텐서를 CPU로 옮기고 numpy로 변환
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # 가중치 계산 (GAP)
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # ReLU 및 정규화
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        # 0으로 나누기 방지를 위해 작은 값(1e-8) 추가
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        # 6. 히트맵 합성 및 저장
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # 텍스트 오버레이
        label_text = f"{self.labels[target_class_index]} ({score.item():.2%})"
        cv2.putText(result_img, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(output_path, result_img)
        print(f"✅ Grad-CAM 결과 저장 완료: {output_path}")
        
        return output_path