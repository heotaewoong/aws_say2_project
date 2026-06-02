from vision_engine import CheXNetEngine
import os

def test_vision_module():
    # 1. 엔진 초기화 (가중치 파일이 있다면 경로 입력, 없으면 None)
    # 예: model_path = "../models/chexnet_weights.pth"
    engine = CheXNetEngine(model_path=None)

    # 2. 테스트 이미지 경로 설정 (로컬에 있는 실제 이미지 경로로 수정하세요)
    # MIMIC-CXR 샘플이나 인터넷에서 구한 흉부 엑스레이 이미지 파일
    test_image = "data\person3_bacteria_13.jpeg" 

    if not os.path.exists(test_image):
        print(f"❌ 테스트할 이미지가 {test_image} 경로에 없습니다.")
        return

    # 3. 분석 실행
    vision_hpos = engine.extract_vision_hpos(test_image, threshold=0.5)

    # 4. 결과 확인
    print(f"\n총 {len(vision_hpos)}개의 시각적 소견이 추출되었습니다.")
    for res in vision_hpos:
        print(f"-> {res['finding']} ({res['hpo_id']})")

if __name__ == "__main__":
    test_vision_module()