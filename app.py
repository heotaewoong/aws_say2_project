import streamlit as st
import pandas as pd
from PIL import Image
import os
from extractor import PhenotypeExtractor
from vision_engine import CheXNetEngine
from inference_engine import RareDiseaseInference
from reporter import ClinicalReporter

# --- 1. 페이지 설정 및 엔진 캐싱 ---
st.set_page_config(page_title="Rare-Link AI Dashboard", layout="wide")

@st.cache_resource
def load_engines(api_key):
    return {
        "nlp": PhenotypeExtractor(api_key=api_key),
        "vision": CheXNetEngine(),
        "inference": RareDiseaseInference(),
        "reporter": ClinicalReporter()
    }

# --- 2. 사이드바 (설정) ---
st.sidebar.title("⚙️ System Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
subject_id = st.sidebar.text_input("Patient Subject ID", value="10000032")

# --- 3. 메인 화면 구성 ---
st.title("🧬 Rare-Link AI: 멀티모달 희귀 폐 질환 진단 시스템")
st.markdown("---")

if not api_key:
    st.warning("사이드바에 Gemini API Key를 입력해주세요.")
else:
    engines = load_engines(api_key)
    
    col1, col2 = st.columns(2)

    # --- 왼쪽 컬럼: 데이터 입력 ---
    with col1:
        st.subheader("📝 1. 환자 데이터 입력 (NLP & Lab)")
        sample_text = "Patient presents with progressive dyspnea and marked digital clubbing. No fever."
        clinical_note = st.text_area("의사 소견서(Clinical Note)", value=sample_text, height=150)
        
        uploaded_img = st.file_uploader("📸 2. 흉부 엑스레이 업로드 (Vision)", type=["jpg", "png", "jpeg"])
        
        if uploaded_img:
            # 임시 파일 저장
            with open("temp_cxr.jpg", "wb") as f:
                f.write(uploaded_img.getbuffer())
            st.image(uploaded_img, caption="업로드된 엑스레이 이미지", use_container_width=True)

    # --- 오른쪽 컬럼: 분석 결과 ---
    with col2:
        st.subheader("🔍 3. AI 분석 결과")
        if st.button("진단 시작 (Run Analysis)"):
            with st.spinner("멀티모달 데이터를 통합 분석 중입니다..."):
                
                # Phase 2: NLP 추출
                nlp_findings = engines["nlp"].extract_from_text_llm(clinical_note)
                nlp_hpos = [f['hpo_id'] for f in nlp_findings]
                
                # Phase 3: Vision 분석 및 Grad-CAM
                vision_hpos = []
                if uploaded_img:
                    vision_results = engines["vision"].extract_vision_hpos("temp_cxr.jpg", threshold=0.1)
                    vision_hpos = [f['hpo_id'] for f in vision_results]
                    
                    # Grad-CAM 생성 (가장 확률 높은 질환)
                    if vision_results:
                        top_finding = sorted(vision_results, key=lambda x: x['score'], reverse=True)[0]
                        cam_path = engines["vision"].get_cam_visualize("temp_cxr.jpg", top_finding['index'], "cam_output.png")
                        st.image(cam_path, caption=f"이상 부위 시각화: {top_finding['finding']}", use_container_width=True)

                # Phase 4: 희귀 질환 추론
                combined_hpos = list(set(nlp_hpos + vision_hpos))
                rankings = engines["inference"].rank_diseases(combined_hpos)

                # 결과 요약 표시
                st.success("분석 완료!")
                st.write(f"🧬 **추출된 HPO 프로필:** {combined_hpos}")
                
                st.dataframe(rankings[['DiseaseName', 'Score', 'Evidence']], use_container_width=True)

    # --- 하단 섹션: 최종 리포트 ---
    st.markdown("---")
    st.subheader("📄 4. 의사용 최종 리포트")
    if 'rankings' in locals():
        final_report = engines["reporter"].generate_summary(subject_id, nlp_findings, vision_results, rankings)
        st.text_area("Clinical Report Output", value=final_report, height=300)
        st.download_button("리포트 다운로드 (.txt)", final_report, file_name=f"report_{subject_id}.txt")