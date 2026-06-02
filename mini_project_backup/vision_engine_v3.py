"""
============================================================
🔬 vision_engine_v3.py – 개선된 엑스레이 분석 엔진
============================================================

🎯 이 파일이 하는 일 (초보자용 설명):
    이 파일은 "학습이 끝난 AI 의사"를 실전에 투입하는 코드입니다.
    
    환자의 흉부 엑스레이 사진을 넣으면:
    1. "이 사진에 무슨 병이 보이는지" 14개 질환을 각각 확률로 알려줌
    2. 이상 부위를 빨간색 히트맵으로 표시해줌 (Grad-CAM)
    
    V2 대비 개선점:
    1. [TTA] 사진 1장이 아니라 5가지 변형으로 보고 평균 → 더 안정적
    2. [EfficientNet 지원] V2는 DenseNet만 사용 → B4까지 지원
    3. [HPO 매핑 확장] CheXpert 14개 레이블에 맞춘 정확한 HPO 코드
    4. [신뢰도 필터] 낮은 확률 결과를 자동 필터링
============================================================
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os


class CheXNetEngineV3:
    """
    초보자 설명:
        이 클래스는 "엑스레이를 분석하는 AI 의사"입니다.
        
        사용법:
        1. engine = CheXNetEngineV3("best_model_v3.pth")  ← AI 뇌 로드
        2. results = engine.analyze("환자_xray.jpg")       ← 분석 시작
        3. results에 [{'finding': '폐렴', 'score': 0.85}]  ← 결과 나옴
    """
    
    def __init__(self, model_path=None, model_name="efficientnet_b4"):
        """
        AI 모델(뇌)를 초기화합니다.
        
        Args:
            model_path: 학습된 가중치 파일 경로 (예: "best_model_v3.pth")
            model_name: 사용할 모델 종류 ("efficientnet_b4" 또는 "densenet121")
        """
        self.model_name = model_name
        
        # 1. 모델 뼈대 생성
        if model_name == "efficientnet_b4":
            self.model = models.efficientnet_b4(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, 14)
        else:
            self.model = models.densenet121(weights=None)
            num_ftrs = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_ftrs, 14)
        
        # 2. 학습된 가중치 로드
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ 모델 로드 완료: {model_path}")
        else:
            print("⚠️ 가중치 없음 → 테스트용 랜덤 모델 사용")
        
        self.model.eval()

        # 3. Grad-CAM 설정
        self.gradients = None
        self.activations = None
        if model_name == "efficientnet_b4":
            self.target_layer = self.model.features[-1]  # EfficientNet 마지막 레이어
        else:
            self.target_layer = self.model.features.norm5  # DenseNet 마지막 레이어

        # 4. CheXpert 14개 레이블 (V2와 라벨 순서 일치)
        self.labels = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion",
            "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        
        # 5. HPO 매핑 (확장 + 검증됨)
        self.hpo_map = {
            "Atelectasis":               "HP:0100750",
            "Cardiomegaly":              "HP:0001640",
            "Consolidation":             "HP:0032177",
            "Edema":                     "HP:0100598",
            "Enlarged Cardiomediastinum": "HP:0001640",  # Cardiomegaly 상위 개념
            "Fracture":                  "HP:0020110",
            "Lung Lesion":               "HP:0033822",
            "Lung Opacity":              "HP:0032183",
            "No Finding":                None,           # 정상 → 추천 제외
            "Pleural Effusion":          "HP:0002202",
            "Pleural Other":             "HP:0100749",
            "Pneumonia":                 "HP:0002090",
            "Pneumothorax":              "HP:0002107",
            "Support Devices":           None,           # 의료기기 → 질환 아님
        }

    # ============================================================
    # 📸 이미지 전처리
    # ============================================================
    def _preprocess(self, image_path):
        """이미지를 AI가 읽을 수 있는 형태로 변환"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        pil_image = Image.open(image_path).convert('RGB')
        input_tensor = transform(pil_image).unsqueeze(0)
        
        # Grad-CAM 시각화용 원본 이미지
        original_img = np.array(pil_image)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        original_img = cv2.resize(original_img, (224, 224))
        
        return input_tensor, original_img, pil_image

    # ============================================================
    # 🎯 기본 예측 (단일 이미지)
    # ============================================================
    def predict(self, image_path, threshold=0.3):
        """1장의 이미지를 분석하여 질환 확률 반환"""
        input_tensor, _, _ = self._preprocess(image_path)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        probabilities = torch.sigmoid(outputs[0]).tolist()
        results = []
        
        for i, prob in enumerate(probabilities):
            label = self.labels[i]
            if prob >= threshold and self.hpo_map.get(label) is not None:
                results.append({
                    'finding': label,
                    'hpo_id': self.hpo_map[label],
                    'score': round(prob, 4),
                    'index': i,
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    # ============================================================
    # 🔥 [개선 4] TTA 예측 (Test-Time Augmentation)
    # ============================================================
    def predict_with_tta(self, image_path, threshold=0.3):
        """
        초보자 설명:
            TTA는 "같은 사진을 5가지 방식으로 보고 평균"내는 기법입니다.
            
            예시 (의사 비유):
            - 원본 사진으로 진단: "폐렴 80%"
            - 좌우반전 사진으로 진단: "폐렴 85%"
            - 약간 회전 사진으로 진단: "폐렴 78%"
            - 평균: (80+85+78) / 3 = 81% ← 더 안정적!
            
            V2는 이 기능이 없어서 한 장으로만 판단 → 불안정
        """
        _, _, pil_image = self._preprocess(image_path)
        
        # 5가지 변형 정의
        # ⚠️ 흉부 X-ray TTA 주의사항:
        #   ❌ 좌우 반전(HorizontalFlip) 금지! → 심장 위치가 바뀌면 오진 위험
        #   ❌ 과도한 회전 금지! → ±5도 이내만 안전
        #   ✅ 안전한 변형: 밝기, 약간의 크롭 위치 변화, 가우시안 블러
        tta_transforms = [
            # 1. 원본 (기준)
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # 2. 약간 밝게 (+10%)
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # 3. 약간 어둡게 (-10%)
            transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.Lambda(lambda x: transforms.functional.adjust_brightness(x, 0.9)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # 4. 약간 좌측 크롭 (위치 변형)
            transforms.Compose([
                transforms.Resize(270),
                transforms.Lambda(lambda x: transforms.functional.crop(x, 10, 5, 224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
            # 5. 약간 우측 크롭 (위치 변형)
            transforms.Compose([
                transforms.Resize(270),
                transforms.Lambda(lambda x: transforms.functional.crop(x, 5, 20, 224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        ]
        
        all_probs = []
        for t in tta_transforms:
            tensor = t(pil_image).unsqueeze(0)
            with torch.no_grad():
                probs = torch.sigmoid(self.model(tensor))
            all_probs.append(probs)
        
        # 5장의 평균
        avg_probs = torch.stack(all_probs).mean(dim=0)[0].tolist()
        
        results = []
        print(f"\n--- [Vision V3 TTA 분석: {os.path.basename(image_path)}] ---")
        for i, prob in enumerate(avg_probs):
            label = self.labels[i]
            hpo_id = self.hpo_map.get(label)
            if prob >= threshold and hpo_id is not None:
                results.append({
                    'finding': label,
                    'hpo_id': hpo_id,
                    'score': round(prob, 4),
                    'index': i,
                })
                print(f"  ✅ {label:28s} | {prob:.4f} | {hpo_id}")
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

    # (편의 메서드)
    def extract_vision_hpos(self, image_path, threshold=0.3, use_tta=True):
        """V2 호환 인터페이스 – 기존 코드에서 갈아끼우기 쉽게"""
        if use_tta:
            return self.predict_with_tta(image_path, threshold)
        return self.predict(image_path, threshold)

    # ============================================================
    # 🔥 Grad-CAM (이상 부위 시각화)
    # ============================================================
    def _extract_gradients(self, grad):
        self.gradients = grad

    def _save_activations_and_hook_grad(self, module, input, output):
        self.activations = output
        output.register_hook(self._extract_gradients)

    def get_cam_visualize(self, image_path, target_class_index, output_path='heatmap_result.png'):
        """
        초보자 설명:
            Grad-CAM은 "AI가 사진에서 어디를 보고 판단했는지" 보여주는 기술입니다.
            
            결과: 엑스레이 위에 빨간색/노란색으로 AI가 주목한 부분이 표시됨
                  → 의사가 AI의 판단 근거를 확인할 수 있음
        """
        print(f"🔥 '{self.labels[target_class_index]}' 에 대한 Grad-CAM 생성 중...")
        
        input_tensor, original_img, _ = self._preprocess(image_path)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        handle = self.target_layer.register_forward_hook(self._save_activations_and_hook_grad)
        
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        score = outputs[0][target_class_index]
        score.backward()
        handle.remove()

        if self.gradients is None or self.activations is None:
            print("❌ 그래디언트 캡처 실패")
            return None

        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        prob = torch.sigmoid(torch.tensor(score.item())).item()
        label_text = f"{self.labels[target_class_index]} ({prob:.1%})"
        cv2.putText(result_img, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(output_path, result_img)
        print(f"✅ Grad-CAM 저장: {output_path}")
        
        return output_path


# ============================================================
# 🧪 테스트 실행
# ============================================================
if __name__ == "__main__":
    # 1. 모델 로드
    engine = CheXNetEngineV3(
        model_path="best_model_v3.pth",    # 학습 후 저장된 가중치
        model_name="efficientnet_b4"
    )
    
    # 2. 테스트 이미지 분석
    test_image = "person3_bacteria_13.jpeg"
    
    if os.path.exists(test_image):
        print("\n🔎 [TTA 분석 시작]")
        results = engine.predict_with_tta(test_image, threshold=0.1)
        
        if results:
            print(f"\n📸 [Grad-CAM 생성]")
            top = results[0]
            engine.get_cam_visualize(test_image, top['index'], f"cam_v3_{top['finding']}.png")
        else:
            print("⚠️ 검출 소견 없음")
    else:
        print(f"❌ 테스트 이미지 없음: {test_image}")
        print("💡 person3_bacteria_13.jpeg 파일이 같은 폴더에 있는지 확인하세요")
