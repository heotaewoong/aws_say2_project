from vision_engine import CheXNetEngine
import os
import pandas as pd

def test_cam_implementation():
    # 1. 엔진 초기화
    # 가중치가 없다면 None, 있다면 경로 지정 예: "../models/chexnet_weights.pth"
    engine = CheXNetEngine(model_path=None) 

    # 2. 테스트 이미지 경로 (로컬에 있는 실제 이미지 경로로 수정하세요)
    test_image = "./data/person3_bacteria_13.jpeg" 

    if not os.path.exists(test_image):
        print(f"❌ 테스트할 이미지가 {test_image} 경로에 없습니다.")
        return

    # 3. 먼저 소견 분석 실행
    vision_results = engine.extract_vision_hpos(test_image, threshold=0.5) # 테스트를 위해 임계값 낮춤

    # 4. 분석 결과 중 가장 확률이 높은 질환에 대해 Grad-CAM 생성
    if vision_results:
        # 확률 기준 내림차순 정렬
        sorted_results = sorted(vision_results, key=lambda x: x['score'], reverse=True)
        top_finding = sorted_results[0]
        
        print(f"\n최상위 소견 '{top_finding['finding']}'에 대한 시각화를 진행합니다.")
        
        # Grad-CAM 생성 및 저장
        output_file = f"cam_{top_finding['finding']}.png"
        engine.get_cam_visualize(test_image, top_finding['index'], output_path=output_file)
        
    else:
        print("\n임계값 이상의 소견이 발견되지 않아 시각화를 진행하지 않습니다.")

if __name__ == "__main__":
    test_cam_implementation()