"""
SageMaker PyTorch 추론 엔트리포인트 (soo_net_2.py 기준)
model.tar.gz 안에 code/inference.py 로 포함됨.

4개 콜백 (SageMaker PyTorchModel이 자동 호출):
    model_fn   → 모델 로드 (컨테이너 시작 시 1회)
    input_fn   → 요청 body 파싱
    predict_fn → 추론 실행
    output_fn  → 응답 직렬화

⚠️ 팀원 확인 필요:
    SOONET_WEIGHT, UNET_WEIGHT 파일명을 실제 가중치 파일명으로 수정할 것.
    soo_net_2.py의 SooNetEngine 생성자 인자도 확인 필요.
"""
import io
import json
import os

from PIL import Image

# soo_net_2.py, unet_lung_model.py 는 model.tar.gz 루트에 같이 포함
from soo_net_2 import SooNetEngine

# ══════════════════════════════════════════════════════════════
# ⚠️ 팀원 확인 필요 — 가중치 파일명
# 배기태(Milk-Case)에게 확인: 최종 가중치 파일명이 뭔지
# ══════════════════════════════════════════════════════════════
SOONET_WEIGHT = "latest_checkpoint.pth"    # S3 Phase_2/ 최신 SooNet 가중치 (배기태 확인)
UNET_WEIGHT   = "unet_lung_heart_best.pth"  # S3 Phase_2/ UNet 가중치


# ══════════════════════════════════════════════════════════════
# 1. 모델 로드 (cold start 시 1회)
# ══════════════════════════════════════════════════════════════
def model_fn(model_dir: str):
    """
    SageMaker가 컨테이너 시작 시 자동 호출.
    model_dir = /opt/ml/model (model.tar.gz 압축 해제 위치)
    """
    soonet_path = os.path.join(model_dir, SOONET_WEIGHT)
    unet_path   = os.path.join(model_dir, UNET_WEIGHT)

    print(f"[model_fn] SooNet 가중치: {soonet_path}")
    print(f"[model_fn] UNet 가중치:   {unet_path}")

    # latest_checkpoint.pth는 학습 체크포인트 형식일 수 있음
    # {'model_state_dict': ..., 'optimizer_state_dict': ...} 형태 처리
    import torch
    ckpt = torch.load(soonet_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        print("[model_fn] 체크포인트 형식 감지 → model_state_dict 추출")
        # model_state_dict를 별도 파일로 저장해서 SooNetEngine에 전달
        state_dict_path = "/tmp/soonet_state_dict.pth"
        torch.save(ckpt['model_state_dict'], state_dict_path)
        soonet_path = state_dict_path

    return SooNetEngine(
        model_path=soonet_path,
        unet_path=unet_path,
    )


# ══════════════════════════════════════════════════════════════
# 2. 요청 파싱
# ══════════════════════════════════════════════════════════════
def input_fn(request_body, content_type: str):
    """
    두 가지 방식 지원:
    - application/x-image : 바이너리 이미지 직접 전송
    - application/json    : base64 인코딩 이미지
    """
    if content_type == "application/x-image":
        return Image.open(io.BytesIO(request_body)).convert("RGB")

    if content_type == "application/json":
        import base64
        data = json.loads(request_body)
        img_bytes = base64.b64decode(data["image_base64"])
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError(f"Unsupported Content-Type: {content_type}")


# ══════════════════════════════════════════════════════════════
# 3. 추론 실행
# ══════════════════════════════════════════════════════════════
def predict_fn(input_image, engine):
    """
    engine.predict()는 파일 경로를 받으므로 임시 저장 후 호출.
    반환 포맷: { label: (probability, hpo_code) }
    """
    tmp_path = "/tmp/sm_input.jpg"
    input_image.save(tmp_path)

    raw = engine.predict(tmp_path)
    # raw 형식: { label: (probability, hpo_code) }

    result = {
        label: {
            "probability": float(prob),
            "hpo_code":    hpo,
        }
        for label, (prob, hpo) in raw.items()
    }

    os.remove(tmp_path)
    return result


# ══════════════════════════════════════════════════════════════
# 4. 응답 직렬화
# ══════════════════════════════════════════════════════════════
def output_fn(prediction, accept: str):
    body = json.dumps(prediction, ensure_ascii=False)
    return body, "application/json"
