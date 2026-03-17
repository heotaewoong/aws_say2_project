import pandas as pd
from extractor import PhenotypeExtractor

def run_phase2_prototype(subject_id):
    extractor = PhenotypeExtractor()
    
    # 1. 로컬 파일 로드
    try:
        notes_df = pd.read_csv('./data/discharge.csv')
        labs_df = pd.read_csv('./data/labevents.csv')
    except FileNotFoundError as e:
        print(f"❌ 데이터를 찾을 수 없습니다: {e}")
        return

    # 2. 해당 환자의 텍스트 데이터 추출
    patient_note = notes_df[notes_df['subject_id'] == subject_id]['text'].values[0]
    nlp_results = extractor.extract_from_text_llm(patient_note)
    nlp_hpos = [res['hpo_id'] for res in nlp_results]

    # 3. 해당 환자의 Lab 데이터 추출
    patient_labs = labs_df[labs_df['subject_id'] == subject_id]
    lab_hpos = []
    for _, row in patient_labs.iterrows():
        res = extractor.extract_from_lab_data(row)
        if res:
            lab_hpos.append(res['hpo_id'])

    # 4. 통합 프로필 (Patient HPO Profile)
    combined_profile = list(set(nlp_hpos + lab_hpos))
    
    print(f"\n--- 환자(ID: {subject_id}) Phase 2 결과 ---")
    print(f"📝 NLP 추출 HPO: {nlp_hpos}")
    print(f"🧪 Lab 추출 HPO: {lab_hpos}")
    print(f"🧬 최종 통합 프로필: {combined_profile}")
    
    return combined_profile

# 실행 (예시 subject_id)
run_phase2_prototype(10001472)