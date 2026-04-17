import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import numpy as np
import pandas as pd
import cv2
import os
import random
import ast
import re

class SooNetEngine:
    def __init__(self, model_path=None):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 모델 구조 정의
        self.model = models.densenet121(weights=None)

        # 3 channels -> 1 channel / first layer modified
        self.model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        
        # engine returns the value of each 14 labels
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, 14)
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"✅ 엔진 가중치 로드 완료")
            
        self.model.to(self.device)
        self.model.eval()

        # --- Grad-CAM을 위한 추가 설정 ---
        self.gradients = None
        self.activations = None
        # 448x448 입력 시 이 위치에서 (1024, 14, 14)의 고해상도 피처맵이 잡힙니다.
        self.target_layer = self.model.features.norm5 
        # -------------------------------

        # 4. 14가지 소견 라벨 및 HPO 매핑
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

    def _preprocess(self, image_path):
        """1채널 변환, 448x448 리사이즈 및 TXV 스케일링 적용"""
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
        
        return input_tensor, pil_image

    def predict(self, image_path):
        input_tensor, _ = self._preprocess(image_path)
        with torch.no_grad():
            outputs = self.model(input_tensor)
        probs = torch.sigmoid(outputs[0]).cpu().numpy()
        
        # {라벨: (확률, HPO)} 형태의 딕셔너리 반환
        return {label: (probs[i], self.hpo_map[label]) for i, label in enumerate(self.labels)}
    
    # Hook 메서드
    def _extract_gradients(self, grad): 
        self.gradients = grad

    def _save_activations_and_hook_grad(self, module, input, output):
        self.activations = output
        output.register_hook(self._extract_gradients)

    def get_cam_visualize(self, image_path, target_class_index, output_path):
        """🚀 엔진 내부에 구현된 정밀 Grad-CAM 생성 로직"""
        input_tensor, pil_image = self._preprocess(image_path)
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # 원본 이미지 복원 (BGR 형식)
        original_img = cv2.resize(cv2.cvtColor(np.array(pil_image.convert('RGB')), cv2.COLOR_RGB2BGR), (448, 448))

        # Hook 등록 및 순전파
        handle = self.target_layer.register_forward_hook(self._save_activations_and_hook_grad)
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        
        # 역전파
        score = outputs[0][target_class_index]
        score.backward()
        handle.remove()

        # CAM 계산
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.maximum(np.dot(weights, activations.reshape(activations.shape[0], -1)).reshape(activations.shape[1:]), 0)
        
        # 정규화 및 히트맵 생성
        cam = cv2.resize(cam, (448, 448))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        result_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
        
        # 결과 저장
        cv2.imwrite(output_path, result_img)
        return output_path

# Model Test Class
class SooNetTester:
    def __init__(self, engine, gt_csv_path, val_csv_path, img_root):
        self.engine = engine
        self.img_root = img_root
        # 정답(GT) 데이터 로드
        self.gt_df = pd.read_csv(gt_csv_path) if os.path.exists(gt_csv_path) else None
        # 검증셋 경로 데이터 로드
        self.val_df = pd.read_csv(val_csv_path) if os.path.exists(val_csv_path) else None
        print(f"✅ 테스트 엔진 및 데이터셋 로드 완료")

    def get_random_test_path(self):
        """🚀 검증 CSV에서 랜덤하게 유효한 이미지 경로 하나를 반환"""
        if self.val_df is None:
            print("❌ 검증 CSV 로드 실패")
            return None

        # 1. 유효한 경로 데이터 수집 (AP 또는 PA 컬럼 활용)
        all_paths = []
        for _, row in self.val_df.iterrows():
            for col in ['AP', 'PA']:
                raw_val = str(row[col])
                if raw_val != 'nan':
                    try:
                        # 문자열 리스트 ['path1', 'path2']를 실제 리스트로 변환
                        path_list = ast.literal_eval(raw_val)
                        all_paths.extend(path_list)
                    except:
                        continue
        
        if not all_paths:
            print("❌ 선택 가능한 이미지 경로가 없습니다.")
            return None

        # 2. 랜덤 선택 및 경로 정규화
        random_rel_path = random.choice(all_paths)
        full_path = os.path.normpath(os.path.join(self.img_root, random_rel_path))
        
        # 실제 파일이 존재하는지 최종 확인
        if os.path.exists(full_path):
            print(f"🎲 랜덤 이미지 선택 완료: {random_rel_path}")
            return full_path
        else:
            # 파일이 없으면 다시 재귀 호출하여 다른 파일을 찾음
            return self.get_random_test_path()
        
    def _get_gt(self, image_path):
        if self.gt_df is None: return None
        
        try:
            # 🚀 [수정] 정규표현식으로 s로 시작하는 8자리 숫자(Study ID)를 안전하게 추출
            match = re.search(r's(\d{8})', image_path)
            if not match:
                print(f"⚠️ 경로에서 Study ID를 찾을 수 없음: {image_path}")
                return None
            
            study_id = int(match.group(1)) # "50591741" -> 50591741 (int 변환)
            
            # 🚀 [수정] CSV의 study_id 컬럼도 int로 확실히 비교
            row = self.gt_df[self.gt_df['study_id'].astype(int) == study_id]
            
            if not row.empty:
                # 라벨 정합성 확인 (대소문자 무시 등)
                res_dict = {}
                for label in self.engine.labels:
                    if label in row.columns:
                        val = row.iloc[0][label]
                        # 불확실(-1.0)은 양성(1.0)으로 처리
                        res_dict[label] = 1.0 if val == 1.0 or val == -1.0 else 0.0
                return res_dict
            else:
                print(f"⚠️ CSV 내에 Study ID {study_id}가 존재하지 않음")
        except Exception as e:
            print(f"❌ GT 매칭 에러: {e}")
        return None

    def run_inference_with_gt(self, image_path, threshold=0.4):
        predictions = self.engine.predict(image_path)
        gt_data = self._get_gt(image_path)
        
        print(f"\n[ 판독 보고서: {os.path.basename(image_path)} ]")
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
                status = "🎯 DETECTED" if gt_val == 1.0 else "❓ FP (Check)"
            elif gt_val == 1.0:
                status = "⚠️ MISS (FN)"

            # 🚀 [핵심] 정답 출력 시 HPO 코드도 함께 출력
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
        print(f"✅ {len(indices)}개의 히트맵이 '{output_dir}' 폴더에 저장되었습니다.")

if __name__ == "__main__":
    # 1. 경로 설정
    IMG_ROOT = "data/mimic-iv-cxr/official_data_iccv_final" #
    VAL_CSV = "data/mimic-iv-cxr/mimic_cxr_aug_validate.csv" #
    GT_CSV = "data/mimic-cxr-2.0.0-chexpert.csv" #
    WEIGHT_PATH = "model/chexnet_1ch_448_chexpert_best.pth"
    
    # 2. 엔진 및 테스터 초기화
    engine = SooNetEngine(model_path=WEIGHT_PATH)
    tester = SooNetTester(engine, gt_csv_path=GT_CSV, val_csv_path=VAL_CSV, img_root=IMG_ROOT)
    
    # 3. 🎲 랜덤 이미지 한 장 가져오기
    random_test_img = tester.get_random_test_path()
    
    if random_test_img:
        # 보고서 출력 및 0.3 이상의 모든 소견 히트맵 생성
        tester.generate_report_heatmaps(random_test_img, threshold=0.35, output_dir='random_test_results')