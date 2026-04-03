import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
from extractor import TextPhenotypeExtractor
from vision_engine import CheXNetEngine
from inference_engine import RareDiseaseInference
from lab_genomic_agent import LabGenomicAgent
from reporter import ClinicalReporter

# ─────────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_FOR_AGENT_PATH = os.path.join(CURRENT_DIR, "data", "lab_for_agent.json")

# ─────────────────────────────────────────────
# 1. 페이지 설정 및 엔진 캐싱
# ─────────────────────────────────────────────
st.set_page_config(page_title="Rare-Link AI Dashboard", layout="wide")

@st.cache_resource
def load_engines():
    return {
        "nlp":       TextPhenotypeExtractor(),
        "vision":    CheXNetEngine(model_path=os.path.join(CURRENT_DIR, "models", "chexnet_mimic_best.pth")),
        "lab":       LabGenomicAgent(),
        "inference": RareDiseaseInference(),
        "reporter":  ClinicalReporter()
    }

@st.cache_data
def load_lab_data():
    """lab_for_agent.json 로드 (76,610명 혈액검사 데이터)"""
    if os.path.exists(LAB_FOR_AGENT_PATH):
        with open(LAB_FOR_AGENT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}

# ─────────────────────────────────────────────
# 2. 사이드바 (설정)
# ─────────────────────────────────────────────
st.sidebar.title("⚙️ System Settings")
subject_id = st.sidebar.text_input("Patient Subject ID (MIMIC-IV)", value="10000032")

# 해당 환자의 혈액검사 데이터 미리보기
lab_db = load_lab_data()
if subject_id in lab_db:
    st.sidebar.success(f"✅ 환자 {subject_id} 혈액검사 데이터 존재")
    st.sidebar.json(lab_db[subject_id])
elif str(subject_id) in lab_db:
    subject_id = str(subject_id)
    st.sidebar.success(f"✅ 환자 {subject_id} 혈액검사 데이터 존재")
    st.sidebar.json(lab_db[subject_id])
else:
    st.sidebar.warning(f"⚠️ 환자 {subject_id} 혈액검사 데이터 없음")

# ─────────────────────────────────────────────
# 3. 메인 화면
# ─────────────────────────────────────────────
st.title("🧬 Rare-Link AI: 멀티모달 희귀 폐 질환 진단 시스템")
st.markdown("---")

engines  = load_engines()

col1, col2 = st.columns(2)

# ── 왼쪽: 데이터 입력 ──
with col1:
    st.subheader("📝 1. 임상 소견서 (NLP 에이전트)")
    sample_text = "Patient presents with progressive dyspnea and marked digital clubbing. No fever."
    clinical_note = st.text_area("의사 소견서 (Clinical Note)", value=sample_text, height=150)

    st.subheader("📸 2. 흉부 엑스레이 (Vision 에이전트)")
    uploaded_img = st.file_uploader("X-ray 이미지 업로드", type=["jpg", "png", "jpeg"])

    if uploaded_img:
        temp_path = os.path.join(CURRENT_DIR, "temp_cxr.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_img.getbuffer())
        st.image(uploaded_img, caption="업로드된 엑스레이", use_container_width=True)

    st.subheader("🩸 3. 혈액검사 (Lab 에이전트)")
    st.info(f"MIMIC-IV 데이터에서 환자 {subject_id}의 수치를 자동으로 불러옵니다.")

# ── 오른쪽: 분석 결과 ──
with col2:
    st.subheader("🔍 4. AI 통합 분석 결과")

    if st.button("진단 시작 (Run Analysis)", type="primary"):
        with st.spinner("멀티모달 데이터를 통합 분석 중..."):

            # ── Agent B: NLP → HPO ──
            st.write("**[Agent B] 소견서 분석 중...**")
            nlp_findings = engines["nlp"].extract_from_text(clinical_note)
            # NLP는 score 없이 반환 → 확정 소견이므로 1.0 부여
            nlp_hpos = [
                {"hpo_id": f["hpo_id"], "score": 1.0, "finding": f["finding"], "source": "NLP"}
                for f in nlp_findings
                if f.get("hpo_id")
            ]
            st.write(f"  → NLP 추출 HPO: {len(nlp_hpos)}개")

            # ── Agent A: Vision → HPO ──
            vision_hpos = []
            cam_path = None
            if uploaded_img:
                st.write("**[Agent A] 엑스레이 분석 중...**")
                temp_path = os.path.join(CURRENT_DIR, "temp_cxr.jpg")
                vision_results = engines["vision"].extract_vision_hpos(temp_path, threshold=0.3)
                vision_hpos = [
                    {"hpo_id": r["hpo_id"], "score": r["score"], "finding": r["finding"], "source": "Vision"}
                    for r in vision_results
                    if r.get("hpo_id")
                ]
                st.write(f"  → Vision 검출 HPO: {len(vision_hpos)}개")

                # Grad-CAM (가장 높은 확률 소견)
                if vision_results:
                    top = max(vision_results, key=lambda x: x["score"])
                    cam_out = os.path.join(CURRENT_DIR, "cam_output.png")
                    cam_path = engines["vision"].get_cam_visualize(temp_path, top["index"], cam_out)
                    if cam_path:
                        st.image(cam_path, caption=f"Grad-CAM: {top['finding']} ({top['score']:.1%})", use_container_width=True)

            # ── Agent C: Lab → HPO ──
            st.write("**[Agent C] 혈액검사 분석 중...**")
            lab_data = lab_db.get(subject_id, lab_db.get(str(subject_id), {}))
            lab_raw_results = engines["lab"].analyze_labs(lab_data)
            lab_hpos = [
                {"hpo_id": r["hpo_id"], "score": r["score"], "finding": r["finding"], "source": "Lab"}
                for r in lab_raw_results
                if r.get("hpo_id")
            ]
            if lab_data:
                st.write(f"  → 혈액검사 수치: {lab_data}")
                st.write(f"  → Lab 검출 HPO: {len(lab_hpos)}개")
            else:
                st.write(f"  → 환자 {subject_id}의 혈액검사 데이터 없음")

            # ── Agent E: 희귀 질환 추론 ──
            st.write("**[Agent E] 희귀 질환 매칭 중...**")
            combined_hpos = nlp_hpos + vision_hpos + lab_hpos

            # 중복 HPO 제거 (같은 HPO_ID는 최고 score만 유지)
            seen = {}
            for h in combined_hpos:
                hid = h["hpo_id"]
                if hid not in seen or h["score"] > seen[hid]["score"]:
                    seen[hid] = h
            combined_hpos_dedup = list(seen.values())

            rankings = engines["inference"].rank_diseases(combined_hpos_dedup)

            # ── 결과 출력 ──
            st.success("✅ 분석 완료!")

            st.write("**추출된 HPO 프로필:**")
            hpo_summary = [f"{h['finding']} [{h['hpo_id']}] ({h['source']})" for h in combined_hpos_dedup]
            for h in hpo_summary:
                st.write(f"  • {h}")

            if not rankings.empty:
                st.write("**희귀 질환 매칭 결과 (Top 5):**")
                st.dataframe(rankings[["DiseaseName", "Score", "Evidence"]], use_container_width=True)
            else:
                st.warning("매칭된 희귀 질환이 없습니다. HPO 데이터를 확인하세요.")

# ─────────────────────────────────────────────
# 4. 최종 리포트
# ─────────────────────────────────────────────
st.markdown("---")
st.subheader("📄 5. 의사용 최종 리포트 (Agent D)")

if st.button("리포트 생성"):
    if "rankings" not in dir() or "combined_hpos_dedup" not in dir():
        st.warning("먼저 '진단 시작' 버튼을 눌러 분석을 완료하세요.")
    else:
        with st.spinner("수석 전문의 에이전트가 리포트 작성 중..."):
            rankings_text = rankings.to_string(index=False) if not rankings.empty else "매칭 결과 없음"
            final_report = engines["reporter"].generate_summary(
                subject_id=subject_id,
                vision_results=vision_hpos,
                nlp_results=nlp_hpos,
                lab_genomic_results=lab_hpos,
                rare_rankings_text=rankings_text
            )
            st.text_area("Clinical Report", value=final_report, height=400)
            st.download_button(
                "리포트 다운로드 (.txt)",
                final_report,
                file_name=f"report_{subject_id}.txt"
            )
