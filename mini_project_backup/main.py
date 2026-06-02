"""
============================================================
🧪 main.py – V3 통합 테스트 (커맨드라인용)
============================================================

🎯 이 파일이 하는 일:
    웹 화면(Streamlit) 없이, 터미널에서 전체 파이프라인을 테스트합니다.
    
사용법:
    GEMINI_API_KEY=your_key python main.py

    또는 아래 API_KEY 변수에 직접 입력
============================================================
"""

import os
from vision_engine_v3 import CheXNetEngineV3
from disease_matcher_v2 import DiseaseMatcherV2
from reporter import ClinicalReporter


def main():
    # --- 설정 ---
    # ⚠️ API 키는 환경변수에서 읽기 (보안)
    API_KEY = os.environ.get("GEMINI_API_KEY", "")
    if not API_KEY:
        print("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   export GEMINI_API_KEY='your_key' 후 다시 실행하세요.")
        print("   또는 이 파일의 API_KEY 변수에 직접 입력하세요.\n")
        # 빈 키로도 Vision만 테스트 가능
    
    TEST_SUBJECT = 10000032
    TEST_IMAGE = "person3_bacteria_13.jpeg"
    MODEL_PATH = "best_model_v3.pth"  # 학습 후 생성되는 가중치
    
    # --- 1. 엔진 초기화 ---
    print("🔧 엔진 초기화 중...")
    vision_engine = CheXNetEngineV3(model_path=MODEL_PATH, model_name="efficientnet_b4")
    matcher = DiseaseMatcherV2()
    reporter = ClinicalReporter()

    print(f"\n🏥 Rare-Link AI V3 파이프라인 가동 (환자: {TEST_SUBJECT})")
    print("=" * 60)

    # --- 2. Phase 1: NLP 텍스트 분석 ---
    nlp_findings = []
    nlp_hpos = []
    if API_KEY:
        try:
            from extractor import PhenotypeExtractor
            nlp_engine = PhenotypeExtractor(api_key=API_KEY)
        except ImportError:
            print("⚠️ google-generativeai 미설치 → pip install google-generativeai")
            nlp_engine = None
        sample_note = "Patient presents with progressive dyspnea and marked digital clubbing. Bilateral lung opacity on chest X-ray."
        if nlp_engine:
            print("\n📝 Phase 1: 텍스트 분석 중...")
            nlp_findings = nlp_engine.extract_from_text_llm(sample_note)
            nlp_hpos = [{'hpo_id': f['hpo_id'], 'score': 1.0, 'finding': f.get('finding', '')} for f in nlp_findings]
            print(f"   → {len(nlp_hpos)}개 HPO 추출")
        else:
            print("\n📝 Phase 1: NLP 건너뜀")
    else:
        print("\n📝 Phase 1: NLP 건너뜀 (API 키 없음)")

    # --- 3. Phase 2: Vision 이미지 분석 ---
    vision_hpos = []
    if os.path.exists(TEST_IMAGE):
        print(f"\n📸 Phase 2: 엑스레이 분석 중... ({TEST_IMAGE})")
        vision_hpos = vision_engine.extract_vision_hpos(TEST_IMAGE, threshold=0.1, use_tta=True)
        print(f"   → {len(vision_hpos)}개 소견 검출")
        
        # Grad-CAM 시각화
        if vision_hpos:
            top = vision_hpos[0]
            cam_path = vision_engine.get_cam_visualize(TEST_IMAGE, top['index'], "cam_output_v3.png")
            print(f"   → Grad-CAM 저장: {cam_path}")
    else:
        print(f"\n📸 Phase 2: 이미지 없음 ({TEST_IMAGE})")

    # --- 4. Phase 3: 질환 매칭 ---
    combined_hpos = nlp_hpos + vision_hpos
    if combined_hpos:
        print(f"\n🧬 Phase 3: 질환 매칭 중... ({len(combined_hpos)}개 HPO)")
        match_results = matcher.match(combined_hpos, top_n=5)
        
        # 요약 출력
        print(match_results['summary'])
    else:
        print("\n🧬 Phase 3: HPO가 없어서 매칭 건너뜀")

    # --- 5. Phase 4: 리포트 생성 ---
    print(f"\n📄 Phase 4: 리포트 생성 중...")
    
    # reporter.py 호환을 위한 변환
    import pandas as pd
    if 'match_results' in locals() and 'layer1' in match_results and not match_results['layer1'].empty:
        rankings_for_report = match_results['layer1'].rename(columns={
            'FinalScore': 'Score', 'Evidence': 'Evidence'
        })
    else:
        rankings_for_report = pd.DataFrame(columns=['DiseaseName', 'Score', 'Evidence'])
    
    final_report = reporter.generate_summary(
        TEST_SUBJECT, nlp_findings, vision_hpos, rankings_for_report
    )
    print("\n" + final_report)
    
    # 파일 저장
    report_path = f"report_{TEST_SUBJECT}_v3.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n✅ 리포트 저장: {report_path}")


if __name__ == "__main__":
    main()