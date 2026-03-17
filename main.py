from extractor import PhenotypeExtractor
from vision_engine import CheXNetEngine
from inference_engine import RareDiseaseInference
from reporter import ClinicalReporter
import os

def main():
    # --- 설정 ---
    API_KEY = "AIzaSyDGiaHsbviOjfYXPkPm3dgj0IcT-YrQ17o"
    TEST_SUBJECT = 10000032
    TEST_IMAGE = "./data/person3_bacteria_13.jpeg"
    
    # --- 1. 엔진 초기화 ---
    nlp_engine = PhenotypeExtractor(api_key=API_KEY)
    vision_engine = CheXNetEngine()
    inference_engine = RareDiseaseInference()
    reporter = ClinicalReporter()

    print(f"Rare-Link AI Pipeline 가동 중... (환자: {TEST_SUBJECT})")

    # --- 2. Phase 2: NLP/Lab 추출 (실제로는 로컬 파일 로드 로직 사용) ---
    # 예시 텍스트 (실제 구현 시 discharge.csv에서 로드)
    sample_note = "Patient presents with progressive dyspnea and marked digital clubbing."
    nlp_findings = nlp_engine.extract_from_text_llm(sample_note)
    nlp_hpos = [f['hpo_id'] for f in nlp_findings]

    # --- 3. Phase 3: Vision 분석 ---
    vision_findings = vision_engine.extract_vision_hpos(TEST_IMAGE, threshold=0.5)
    vision_hpos = [f['hpo_id'] for f in vision_findings]

    # --- 4. Phase 4: 희귀 질환 추론 ---
    combined_hpos = list(set(nlp_hpos + vision_hpos))
    rare_rankings = inference_engine.rank_diseases(combined_hpos)

    # --- 5. Phase 5: 리포트 생성 및 출력 ---
    final_report = reporter.generate_summary(TEST_SUBJECT, nlp_findings, vision_findings, rare_rankings)
    
    print("\n" + final_report)

    # 리포트 파일로 저장
    with open(f"report_{TEST_SUBJECT}.txt", "w", encoding="utf-8") as f:
        f.write(final_report)
    print(f"\n✅ 최종 리포트가 저장되었습니다: report_{TEST_SUBJECT}.txt")

if __name__ == "__main__":
    main()