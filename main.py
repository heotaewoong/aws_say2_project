from vision_engine import PubMedCLIPEngine
from extractor import TextPhenotypeExtractor
from lab_genomic_agent import LabGenomicAgent
from inference_engine import RareDiseaseInference # 1. 교체된 엔진
from reporter import ClinicalReporter

def standardize_hpo_results(raw_results):
    """모든 에이전트의 출력을 {finding, hpo_id, score} 규격으로 통일"""
    standardized = []
    if not raw_results: return []
    
    # 만약 단일 객체면 리스트로 감쌈
    if isinstance(raw_results, dict):
        raw_results = [raw_results]
        
    for item in raw_results:
        if not isinstance(item, dict): continue
        
        # 1. HPO ID 찾기 (유연하게)
        hpo_id = item.get('hpo_id') or item.get('HPO_ID') or item.get('hpo')
        # 2. Finding 찾기
        finding = item.get('finding') or item.get('label') or "Unknown"
        # 3. Score 찾기 (기본값 1.0)
        score = item.get('score') or item.get('prob') or 1.0
        
        if hpo_id:
            standardized.append({
                'finding': finding,
                'hpo_id': str(hpo_id).strip(), # 공백 제거
                'score': float(score)
            })
    return standardized

def run_diagnostic_system(subject_id, xray_path, note_text, lab_data, variant_data):
    print(f"\n🚀 환자 {subject_id}에 대한 멀티모달 분석을 시작합니다.")
    print("-" * 60)

    # 1. 에이전트 초기화
    agent_a = PubMedCLIPEngine()
    agent_b = TextPhenotypeExtractor()
    agent_c = LabGenomicAgent()
    agent_e = RareDiseaseInference() # 2. 새로운 추론 엔진 초기화
    agent_d = ClinicalReporter()

    # 2. 각 에이전트 분석 및 표준화 수행
    hpo_a = standardize_hpo_results(agent_a.extract_vision_hpos(xray_path))
    hpo_b = standardize_hpo_results(agent_b.extract_from_text(note_text))
    hpo_c = standardize_hpo_results(agent_c.extract_hpos(lab_data, variant_data))

    # 3. 모든 HPO 통합 (standardize_hpo_results 덕분에 이미 리스트임)
    all_hpos = hpo_a + hpo_b + hpo_c

    # 4. 희귀 질환 정밀 매칭 (Agent E - IDF 기반)
    # 새로운 엔진의 rank_diseases는 'score'가 포함된 리스트를 받아 정밀 계산을 수행합니다.
    rare_rankings_df = agent_e.rank_diseases(all_hpos, top_k=5)

    # 5. 리포터(Agent D)에게 전달할 텍스트 포맷팅
    if not rare_rankings_df.empty:
        # 데이터프레임을 리포터가 읽기 좋은 텍스트 형식으로 변환
        ranking_summary = "### Rare Disease Matching Results (by Agent E)\n"
        for _, row in rare_rankings_df.iterrows():
            ranking_summary += f"- 질환명: {row['DiseaseName']} (매칭 점수: {row['Score']})\n"
            ranking_summary += f"  근거: {row['Evidence']}\n"
    else:
        ranking_summary = "매칭되는 희귀 질환 정보가 없습니다."

    # 6. 최종 리포트 생성 (Agent D)
    report = agent_d.generate_summary(
        subject_id, 
        vision_results=hpo_a, 
        nlp_results=hpo_b, 
        lab_genomic_results=hpo_c,
        rare_rankings_text=ranking_summary # 3. 정밀 매칭 결과 전달
    )

    return report

if __name__ == "__main__":
    # 1. 테스트 데이터 설정
    sample_id = "10000032"
    img = "./data/sub-S11869_ses-E23135_run-1_bp-chest_vp-ap_cr.png"
    note = "Patient presents with progressive dyspnea and finger clubbing."
    labs = {'WBC Count': 14.5, 'Oxygen Saturation': 91.0}
    genes = ['SFTPB']

    # 2. 시스템 실행 (리포트 생성)
    final_report = run_diagnostic_system(sample_id, img, note, labs, genes)

    # 3. 콘솔에 출력 (확인용)
    print("\n" + "="*30)
    print("📋 분석이 완료되었습니다. 파일을 생성합니다.")
    print("="*30)

    # 4. .txt 파일로 저장
    # 파일명 예시: report_10000032_20260319.txt
    import datetime
    date_str = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    file_name = f"report_{sample_id}_{date_str}.txt"

    try:
        # 한글 깨짐 방지를 위해 utf-8-sig 인코딩 사용
        with open(file_name, "w", encoding="utf-8-sig") as f:
            f.write(final_report)
        
        print(f"✅ 리포트가 성공적으로 저장되었습니다: {file_name}")
        
        # 저장된 파일 경로 절대 경로로 출력 (찾기 편하게)
        import os
        print(f"📍 파일 위치: {os.path.abspath(file_name)}")

    except Exception as e:
        print(f"❌ 파일 저장 중 오류 발생: {e}")