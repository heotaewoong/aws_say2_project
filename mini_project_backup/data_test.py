import pandas as pd
from extractor import PhenotypeExtractor

def run_phase2_prototype(subject_id, api_key):
    # Gemini API 키로 초기화
    extractor = PhenotypeExtractor(api_key=api_key)

    # 데이터 로드 및 병합 (이전과 동일)
    notes_df = pd.read_csv('./data/discharge.csv', nrows=100)
    labs_df = pd.read_csv('./data/labevents.csv', nrows=100)
    items_df = pd.read_csv('./data/d_labitems.csv', nrows=100)
    labs_merged = pd.merge(labs_df, items_df[['itemid', 'label']], on='itemid', how='left')

    # NLP 추출 (Gemini 사용)
    patient_notes = notes_df[notes_df['subject_id'] == subject_id]
    if not patient_notes.empty:
        text_content = patient_notes.iloc[0]['text']
        text_content = text_content[:1000]
        nlp_results = extractor.extract_from_text_llm(text_content)
        nlp_hpos = [res['hpo_id'] for res in nlp_results]
    else:
        nlp_hpos = []

    # Lab 추출
    patient_labs = labs_merged[labs_merged['subject_id'] == subject_id]
    lab_hpos = []
    for _, row in patient_labs.iterrows():
        res = extractor.extract_from_lab_data(row)
        if res:
            lab_hpos.append(res['hpo_id'])

    # 결과 통합
    combined_profile = list(set(nlp_hpos + lab_hpos))
    print(f"\n✅ 분석 완료 (Subject ID: {subject_id})")
    print(f"🧬 최종 HPO 프로필: {combined_profile}")
    
    return combined_profile

if __name__ == "__main__":
    GEMINI_API_KEY = "AIzaSyDGiaHsbviOjfYXPkPm3dgj0IcT-YrQ17o" # 실제 키 입력
    run_phase2_prototype(10000032, GEMINI_API_KEY)