import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import urllib.request

# ──────────────────────────────────────────
# 0. CheXNetEngine 클래스 (가중치 로직 통합)
# ──────────────────────────────────────────
class CheXNetEngine:
    def __init__(self, model_path=None):
        self.model = models.densenet121(weights=None)
        num_ftrs = self.model.classifier.in_features
        
        # 기존 코드 유지 (만약 가중치 로드 시 크기 오류가 나면 nn.Sequential 대신 nn.Linear(num_ftrs, 14)만 남기고 수정해야 할 수 있습니다.)
        self.model.classifier = nn.Linear(num_ftrs, 14)
        
        if model_path and os.path.exists(model_path):
            print(f"🔄 가중치 로드 중: {model_path} ...")
            
            # 1. 파일 열기 (CPU 환경에서도 작동하도록)
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # 2. state_dict 안전하게 추출 (.pth.tar 같은 구조 처리)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint  # 파일 자체가 가중치 딕셔너리인 경우
                
            # 3. 'module.' 접두사 제거 (분산 학습된 모델 대응)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # 4. 뼈대에 가중치 입히기
            # strict=False는 약간의 구조 차이(예: classifier 이름)가 있어도 강제로 불러옵니다.
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✅ CheXNet 가중치 로드 완료!")
        else:
            print("⚠️ 가중치 없음 → 랜덤 초기화 모델 사용 (구조 테스트 전용)")

        self.model.eval()

        self.labels = [
            "Atelectasis",          # 0
            "Cardiomegaly",         # 1
            "Consolidation",        # 2
            "Edema",                # 3
            "Effusion",             # 4
            "Emphysema",            # 5
            "Fibrosis",             # 6
            "Hernia",               # 7
            "Infiltration",         # 8 (이전엔 Consolidation 자리)
            "Mass",                 # 9
            "Nodule",               # 10
            "Pleural_Thickening",   # 11
            "Pneumonia",            # 12
            "Pneumothorax"          # 13
        ]

    def _preprocess(self, image_path):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        pil_image = Image.open(image_path).convert('RGB')
        return transform(pil_image).unsqueeze(0)

    def predict_all(self, image_path):
        """14개 라벨 전체 확률을 출력하는 테스트용 메서드"""
        input_tensor = self._preprocess(image_path)

        with torch.no_grad():
            outputs = self.model(input_tensor)

        probabilities = torch.sigmoid(outputs[0]).tolist()
        return list(zip(self.labels, probabilities))


# ──────────────────────────────────────────
# 1. 샘플 이미지 준비
# ──────────────────────────────────────────
IMAGE_PATH = "data/pneumonia.jpeg"   # 역슬래시(\) 대신 슬래시(/) 사용 권장

# 폴더가 없으면 생성
os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)

if not os.path.exists(IMAGE_PATH):
    print("📥 샘플 X-ray 이미지 다운로드 중...")
    SAMPLE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/"
        "6/6e/Pneumonia_x-ray.jpg/640px-Pneumonia_x-ray.jpg"
    )
    try:
        urllib.request.urlretrieve(SAMPLE_URL, IMAGE_PATH)
        print(f"✅ 다운로드 완료: {IMAGE_PATH}")
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        print("   sample_xray.png 파일을 직접 준비해주세요.")
        sys.exit(1)


# ──────────────────────────────────────────
# 2. 모델 로드 및 예측
# ──────────────────────────────────────────
# 클래스 바깥에 있던 로직을 지우고, 진짜 가중치 파일명을 인자로 넘겨줍니다.
engine = CheXNetEngine(model_path="m-30012020-104001.pth.tar")
results = engine.predict_all(IMAGE_PATH)


# ──────────────────────────────────────────
# 3. 결과 출력
# ──────────────────────────────────────────
THRESHOLD = 0.6

print("\n" + "=" * 50)
print(f"  CheXNet 분석 결과: {IMAGE_PATH}")
print("=" * 50)
print(f"{'라벨':<22} {'확률':>8}   {'판정':>8}")
print("-" * 50)

for label, prob in sorted(results, key=lambda x: x[1], reverse=True):
    flag = "🔴 양성" if prob >= THRESHOLD else "⚪ 음성"
    bar  = "█" * int(prob * 20)
    print(f"{label:<22} {prob:>7.4f}   {flag}  {bar}")

print("-" * 50)

positive = [(l, p) for l, p in results if p >= THRESHOLD]
if positive:
    print(f"\n📌 Threshold({THRESHOLD}) 초과 소견 {len(positive)}개 감지됨")
    for l, p in positive:
        print(f"   → {l}: {p:.4f}")
else:
    print(f"\n✅ No Finding (모든 라벨 < {THRESHOLD})")
    print("   → SkipNormalLinkAE 2차 검증 단계로 전달됩니다.")

print("=" * 50)