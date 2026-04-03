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
        
        # 2. 모델 가중치 로드 (다양한 포맷 호환)
        if model_path and os.path.exists(model_path):
            print(f"🔄 가중치 로드 중: {model_path} ...")
            checkpoint = torch.load(model_path, map_location='cpu')

            # .pth.tar 구조 대응 (state_dict 키가 있는 경우)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # 분산 학습 모델의 'module.' 접두사 제거
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ CheXNet 가중치 로드 완료!")
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
        
        # CheXpert 14개 라벨에 맞춘 HPO 매핑 (기존 ChestX-ray14 매핑에서 갱신)
        self.hpo_map = {
            "Atelectasis": "HP:0002095",
            "Cardiomegaly": "HP:0001640",
            "Consolidation": "HP:0002113",
            "Edema": "HP:0002111",
            "Enlarged Cardiomediastinum": "HP:0001640",  # 심장비대 계열
            "Fracture": "HP:0020110",                     # 골절
            "Lung Lesion": "HP:0002088",                  # 폐 병변
            "Lung Opacity": "HP:0002113",                 # 폐 혼탁
            "No Finding": None,                           # 정상 → HPO 없음
            "Pleural Effusion": "HP:0002202",             # 흉막삼출
            "Pleural Other": "HP:0002103",                # 기타 흉막 이상
            "Pneumonia": "HP:0002090",
            "Pneumothorax": "HP:0002107",
            "Support Devices": None                       # 의료기기 → HPO 해당 없음
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
        
        probabilities = torch.sigmoid(outputs[0]).tolist()
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
        
        # 텍스트 오버레이 (sigmoid 적용하여 0~100% 확률로 표시)
        prob = torch.sigmoid(score).item()
        label_text = f"{self.labels[target_class_index]} ({prob:.2%})"
        cv2.putText(result_img, label_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imwrite(output_path, result_img)
        print(f"✅ Grad-CAM 결과 저장 완료: {output_path}")
        
        return output_path

if __name__ == "__main__":
    # 현재 스크립트 위치 기준으로 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. 엔진 초기화 (학습된 가중치 로드)
    weights_path = os.path.join(current_dir, "models", "chexnet_mimic_best.pth")
    engine = CheXNetEngine(model_path=weights_path)

    # 2. 테스트용 이미지 경로 설정
    test_image_path = os.path.join(current_dir, "data", "person3_bacteria_13.jpeg")

    # data 폴더에 없으면 backup 폴더에서 찾기
    if not os.path.exists(test_image_path):
        backup_path = os.path.join(current_dir, "..", "mini_project_v3_backup", "person3_bacteria_13.jpeg")
        if os.path.exists(backup_path):
            test_image_path = backup_path
            print(f"📂 backup 폴더에서 이미지를 찾았습니다: {backup_path}")

    if not os.path.exists(test_image_path):
        print(f"❌ 파일을 찾을 수 없습니다: {test_image_path}")
        print("💡 테스트를 위해 실제 이미지 파일 경로를 넣어주세요.")
    else:
        # 3. 비전 분석 및 HPO 추출 테스트
        print("\n🔎 [STEP 1] HPO 추출 테스트 시작...")
        vision_results = engine.extract_vision_hpos(test_image_path, threshold=0.3)

        if vision_results:
            # 4. Grad-CAM 시각화 테스트
            print("\n📸 [STEP 2] Grad-CAM 시각화 테스트 시작...")
            target = vision_results[0]
            output_filename = f"cam_{target['finding']}.png"

            save_path = engine.get_cam_visualize(
                image_path=test_image_path,
                target_class_index=target['index'],
                output_path=output_filename
            )

            if save_path:
                print(f"✅ 시각화 성공! 결과 파일: {os.path.abspath(save_path)}")
        else:
            print("⚠️ 검출된 소견이 없습니다. threshold를 낮춰보세요 (예: 0.1)")