import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch.nn.functional as F
import numpy as np
import os

class PubMedCLIPEngine:
    def __init__(self, model_path="./models/pubmed-clip", device=None):
        # 1. 장치 설정 (Mac mps 지원)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # 2. 로컬 경로에서 모델 및 프로세서 로드
        if os.path.exists(model_path):
            print(f"📂 로컬 경로에서 모델 로드 중: {model_path}")
            # 로컬 경로를 직접 인자로 전달합니다.
            self.model = CLIPModel.from_pretrained(model_path).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            print(f"✅ 로컬 엔진 로드 완료! (장치: {self.device})")
        else:
            # 로컬에 없을 경우를 대비한 예외 처리
            print(f"⚠️ {model_path}를 찾을 수 없습니다. 허깅페이스에서 직접 로드 시도 중...")
            model_id = "flaviagiammarino/pubmed-clip-vit-base-patch32"
            self.model = CLIPModel.from_pretrained(model_id).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_id)
            
        self.model.eval()

        # 3. HPO 마스터 라이브러리 (필요에 따라 계속 확장 가능)
        self.hpo_library = {
            "ground-glass opacities": "HP:0031087",
            "bilateral pulmonary infiltrates": "HP:0002113",
            "consolidation": "HP:0002094",
            "pleural effusion": "HP:0002202",
            "atelectasis": "HP:0002095",
            "cardiomegaly": "HP:0001640",
            "pneumothorax": "HP:0002107",
            "normal chest x-ray": "NORMAL"
        }
        self.descriptions = list(self.hpo_library.keys())

    def _preprocess(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    def extract_vision_hpos(self, image_path, threshold=0.01):
        """이미지를 분석하여 HPO 코드와 확률을 반환"""
        image = self._preprocess(image_path)
        
        # 1. 로컬 프로세서로 전처리
        inputs = self.processor(
            text=self.descriptions, 
            images=image, 
            return_tensors="pt", 
            padding=True
        ).to(self.device)

        # 2. 추론 (Inference)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1) 

        probabilities = probs[0].cpu().numpy()
        results = []

        print(f"\n--- [Local PubMed-CLIP 분석 결과: {os.path.basename(image_path)}] ---")
        
        for i, prob in enumerate(probabilities):
            desc = self.descriptions[i]
            if prob >= threshold:
                hpo_id = self.hpo_library[desc]
                results.append({
                    'finding': desc,
                    'hpo_id': hpo_id,
                    'score': float(prob)
                })
                status = "✅ 검출" if hpo_id != "NORMAL" else "⚪ 정상"
                print(f"{status}: {desc:<32} | 확률: {prob:.4f} | HPO: {hpo_id}")
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# --- 실행 확인 ---
if __name__ == "__main__":
    # 로컬 경로 설정
    MY_MODEL_PATH = "./models/pubmed-clip"
    engine = PubMedCLIPEngine(model_path=MY_MODEL_PATH)
    
    # 테스트용 이미지 경로 (기태님의 실제 경로로 수정)
    test_img = "data/sub-S11869_ses-E23135_run-1_bp-chest_vp-ap_cr.png"
    
    if os.path.exists(test_img):
        results = engine.extract_vision_hpos(test_img)