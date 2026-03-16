from knowledge_base import KnowledgeBaseBuilder
from extractor import PhenotypeExtractor
from vision_engine import VisionPhenotypeExtractor
from inference_engine import InferenceEngine
from reporter import DiagnosticReporter # 추가
import pandas as pd

def rare_link_ai_pipeline(subject_id, text_note, lab_data, image_path):
    # 모듈 초기화
    nlp_engine = PhenotypeExtractor()
    vision_engine = VisionPhenotypeExtractor()
    inference_engine = InferenceEngine()
    reporter = DiagnosticReporter()

    # Step 1 & 2: 텍스트 및 Lab 데이터 변환
    nlp_hpos = nlp_engine.extract_from_text(text_note)
    lab_hpos = [nlp_engine.extract_from_lab(r['label'], r['valuenum']) for _, r in lab_data.iterrows()]
    
    # Step 3: 영상 분석 (일반 폐 소견 탐지 포함)
    # 여기서는 vision_engine이 일반 질환 라벨(common)과 HPO(rare_hpos)를 모두 반환한다고 가정
    vision_hpos = vision_engine.extract_from_image(image_path)
    common_findings = ['Pneumonia'] # 예시로 일반 질환 탐지 결과 추가

    # Step 4: 통합 및 추론
    all_hpos = list(set(nlp_hpos + [h for h in lab_hpos if h] + vision_hpos))
    rare_candidates = inference_engine.rank_diseases(all_hpos)

    # Step 5: 최종 리포트 생성
    final_report = reporter.generate_report(subject_id, common_findings, rare_candidates, all_hpos)
    
    print(final_report)
    return final_report

if __name__ == "__main__":
    # 팀원들에게 보여줄 데모 실행
    sample_note = "호흡 곤란이 심하며, 육안 상 곤봉지가 뚜렷하게 관찰됨."
    sample_labs = pd.DataFrame([{'label': 'Platelet Count', 'valuenum': 120}])
    sample_image = "cxr_scan_001.png"
    
    rare_link_ai_pipeline(10001472, sample_note, sample_labs, sample_image)