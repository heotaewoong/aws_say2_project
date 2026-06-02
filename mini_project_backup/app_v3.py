"""
============================================================
🏥 app_v3.py – 개선된 통합 대시보드
============================================================

🎯 이 파일이 하는 일 (초보자용 설명):
    이 파일은 "병원 접수처 + 진료 결과 화면"입니다.
    
    웹 브라우저에서:
    1. 환자의 의사 소견서(글)를 입력하고
    2. 환자의 엑스레이 사진을 업로드하면
    3. AI가 분석해서 "어떤 희귀 질환이 의심되는지" 보여줍니다.
    
    V2 대비 개선점:
    1. [V3 Vision] TTA 기반 더 정확한 이미지 분석
    2. [Dual Layer] 비암성 + 암성 질환을 동시에 분석
    3. [UI 개선] 더 깔끔한 결과 표시 + 신뢰도 표시
    
사용법:
    streamlit run app_v3.py
============================================================
"""

import streamlit as st
import pandas as pd
from PIL import Image
import os

from extractor import PhenotypeExtractor       # NLP (텍스트 분석)
from vision_engine_v3 import CheXNetEngineV3    # 🔥 V3 Vision (개선됨)
from disease_matcher_v2 import DiseaseMatcherV2 # 🔥 V2 Matcher (개선됨)
from reporter import ClinicalReporter           # 리포트 생성

# ============================================================
# 📋 페이지 설정
# ============================================================
st.set_page_config(
    page_title="Rare-Link AI V3 Dashboard",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_engines(api_key, model_path):
    """
    초보자 설명:
        AI 엔진들을 한 번만 로드합니다 (캐싱).
        첫 실행 시에만 느리고, 이후에는 빠르게 작동합니다.
    """
    return {
        "nlp": PhenotypeExtractor(api_key=api_key),
        "vision": CheXNetEngineV3(model_path=model_path, model_name="efficientnet_b4"),
        "matcher": DiseaseMatcherV2(),
        "reporter": ClinicalReporter(),
    }

# ============================================================
# 🎨 사이드바 (설정)
# ============================================================
st.sidebar.title("⚙️ System Settings")
api_key = st.sidebar.text_input("Gemini API Key", type="password")
model_path = st.sidebar.text_input("모델 가중치 경로", value="best_model_v3.pth")
subject_id = st.sidebar.text_input("Patient Subject ID", value="10000032")
use_tta = st.sidebar.checkbox("TTA 사용 (더 정확, 더 느림)", value=True)
threshold = st.sidebar.slider("검출 임계값", 0.05, 0.5, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📌 V3 개선사항
- 🔥 EfficientNet-B4
- 🎯 TTA (5-variant 예측)
- ⚖️ Weighted Loss로 학습
- 🧬 COSMIC 유전자 추천
""")

# ============================================================
# 🏥 메인 화면
# ============================================================
st.title("🧬 Rare-Link AI V3: 멀티모달 희귀 폐 질환 진단")
st.caption("AI가 텍스트 + 이미지 + 혈액검사를 통합 분석하여 희귀 질환을 진단합니다")
st.markdown("---")

if not api_key:
    st.warning("👈 사이드바에 Gemini API Key를 입력해주세요.")
    st.info("""
    **처음 사용하시나요?** 이 프로그램은 세 가지 입력을 합쳐서 희귀 질환을 찾습니다:
    1. **의사 소견서** (텍스트) → AI가 증상 키워드를 추출
    2. **흉부 엑스레이** (이미지) → AI가 14가지 소견을 분석
    3. **합산 결과** → 의학 DB에서 가장 유사한 질환 순위 도출
    """)
else:
    engines = load_engines(api_key, model_path)
    
    col1, col2 = st.columns(2)

    # ─── 왼쪽: 데이터 입력 ───
    with col1:
        st.subheader("📝 1단계: 환자 데이터 입력")
        
        sample_text = "Patient presents with progressive dyspnea and marked digital clubbing. Bilateral lung opacity on chest X-ray."
        clinical_note = st.text_area(
            "의사 소견서 (Clinical Note)",
            value=sample_text,
            height=150,
            help="영어로 작성된 임상 소견서를 붙여넣으세요"
        )
        
        uploaded_img = st.file_uploader(
            "📸 2단계: 흉부 엑스레이 업로드",
            type=["jpg", "png", "jpeg"],
            help="흉부 정면(PA) 엑스레이 이미지를 올려주세요"
        )
        
        if uploaded_img:
            with open("temp_cxr.jpg", "wb") as f:
                f.write(uploaded_img.getbuffer())
            st.image(uploaded_img, caption="업로드된 엑스레이", use_container_width=True)

    # ─── 오른쪽: AI 분석 결과 ───
    with col2:
        st.subheader("🔍 3단계: AI 분석 결과")
        
        if st.button("🚀 진단 시작 (Run Analysis)", type="primary", use_container_width=True):
            with st.spinner("🧠 멀티모달 AI가 분석 중입니다..."):
                
                # Phase 1: NLP 텍스트 분석
                st.text("📝 Phase 1: 텍스트 분석 중...")
                nlp_findings = engines["nlp"].extract_from_text_llm(clinical_note)
                nlp_hpos = [{'hpo_id': f['hpo_id'], 'score': 1.0, 'finding': f.get('term', '')} for f in nlp_findings]
                
                # Phase 2: Vision 이미지 분석
                vision_hpos = []
                if uploaded_img:
                    st.text("📸 Phase 2: 엑스레이 분석 중...")
                    vision_hpos = engines["vision"].extract_vision_hpos(
                        "temp_cxr.jpg",
                        threshold=threshold,
                        use_tta=use_tta,
                    )
                    
                    # Grad-CAM 시각화
                    if vision_hpos:
                        top = sorted(vision_hpos, key=lambda x: x['score'], reverse=True)[0]
                        cam_path = engines["vision"].get_cam_visualize(
                            "temp_cxr.jpg", top['index'], "cam_output_v3.png"
                        )
                        if cam_path:
                            st.image(cam_path, caption=f"🔥 AI 주목 영역: {top['finding']} ({top['score']:.1%})", use_container_width=True)
                
                # Phase 3: 통합 매칭
                st.text("🧬 Phase 3: 질환 매칭 중...")
                combined_hpos = nlp_hpos + vision_hpos
                match_results = engines["matcher"].match(combined_hpos, top_n=5)
                
                # ─── 결과 표시 ───
                st.success("✅ 분석 완료!")
                
                # 검출된 HPO 목록
                st.markdown("#### 🔑 검출된 소견")
                hpo_data = [{'소견': h.get('finding', '?'), 'HPO': h['hpo_id'], '확신도': f"{h.get('score', 0):.1%}"} for h in combined_hpos]
                st.dataframe(pd.DataFrame(hpo_data), use_container_width=True)
                
                # Layer 1 결과
                if 'layer1' in match_results and not match_results['layer1'].empty:
                    st.markdown("#### 🔬 비암성 희귀질환 후보")
                    display_df = match_results['layer1'][['DiseaseName', 'FinalScore', 'MatchCount', 'Evidence']].copy()
                    display_df.columns = ['질환명', '점수', '매칭 수', '근거']
                    st.dataframe(display_df, use_container_width=True)
                
                # Layer 2 결과
                if 'layer2' in match_results and not match_results['layer2'].empty:
                    st.markdown("#### 🧬 암성 관련 유전자 검사 추천")
                    st.dataframe(match_results['layer2'].head(5), use_container_width=True)

    # ─── 하단: 최종 리포트 ───
    st.markdown("---")
    st.subheader("📄 4단계: 최종 리포트")
    if 'match_results' in dir() or 'match_results' in locals():
        # 통합 요약
        st.text_area("AI 분석 요약", value=match_results.get('summary', ''), height=250)
        
        # 다운로드
        st.download_button(
            "📥 리포트 다운로드 (.txt)",
            match_results.get('summary', ''),
            file_name=f"v3_report_{subject_id}.txt",
        )
