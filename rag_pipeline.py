"""
Rare-Link AI — RAG 파이프라인 오케스트레이터 (5단계)

확정 기준 문서:
  data/최종프롬프트_API시스템_확정문서_v1.docx (2026-04-29)
  data/RAG_프롬프트_API_정리서.md             (명세)
  data/RAG_구현_보고서.md                      (구현 가이드)

확정 5단계:
  ① Phase 1~3       — 증상/X-ray/Lab+Vital+Micro → HPO 변환 + LR 스코어링
  ② 스코어링 분기   — 일반/기타 DB Top10 + 희귀 DB 리스팅
  ③ Phase 4         — LLM이 Phase 3 결과 정리 → Top 3 통합
  ④ RAG 트리거      — Top 3 기준 5개 API 병렬 호출 + 내부 DB 교차검증
  ⑤ LLM 소견서      — Bedrock Claude Sonnet 3.5 → JSON only

5개 API (확정):
  PubCaseFinder · Orphanet · Monarch · PubMed · ClinicalTrials.gov

출력 형식 (확정):
  recommendation { immediate_workup, specialist_referral, treatment_guideline,
                   genetic_test, additional_lab }
  clinical_notes { summary, top1_reasoning, differential_note, rag_evidence,
                   case_comparison, epidemiology_note, disclaimer }
  ※ Markdown 절대 출력 금지

사용법:
  python rag_pipeline.py
  또는: from rag_pipeline import RareLinkPipeline
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# .env 파일 자동 로드 (python-dotenv 없어도 동작하는 간단 구현)
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# ── rag/ 패키지 컴포넌트 ──────────────────────────────────────────
from soo_net import SooNetEngine
from rag.bedrock_extractor import BedrockHPOExtractor
from rag.lab_rules import lab_to_hpo
from rag.lirical_scorer import build_disease_database, rank_diseases
from rag.general_disease_scorer import rank_general_diseases
from rag.pubcasefinder import get_ranked_diseases
from rag.pubmed_fetcher import PubMedFetcher
from rag.clinicaltrials_fetcher import get_clinical_trials
from rag.monarch_fetcher import (
    get_causal_genes,
    format_monarch_for_prompt,
    format_hpo_for_prompt,
)
from rag.orphanet_fetcher import (
    get_orphanet_data,
    format_orphanet_for_prompt,
    cross_validate_genes,
)

# ══════════════════════════════════════════════════════════════════
# 상수 (확정 문서 §1.2 §5)
# ══════════════════════════════════════════════════════════════════
AWS_REGION         = "ap-northeast-2"   # CLAUDE.md 규칙: 절대 변경 금지
XRAY_THRESHOLD     = 0.3                # X-ray HPO 필터 임계
SCORE_RATIO_THRESH = 3.0                # RAG 트리거 (1위/2위 LR 비율)
RARE_LR_THRESHOLD  = 1.0                # 희귀 리스팅 LR 최소값
TOP_K_PER_API      = 3                  # 확정 §5-② Top 3 × 3건
RAG_PARALLEL_TIMEOUT = 60               # 확정 §4.3 병렬 60초

REPORT_MODEL       = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"  # §1.2 ⑤ 확정
PHASE4_MODEL       = "apac.anthropic.claude-3-haiku-20240307-v1:0"     # §1.2 ③ (ap-northeast-2 제공 Haiku)
# 확정 문서 §2.2 — temperature=0.0
REPORT_MAX_TOKENS  = 4096
REPORT_TEMPERATURE = 0.0

# Phase 4 Haiku 호출 파라미터
PHASE4_MAX_TOKENS  = 1024
PHASE4_TEMPERATURE = 0.0

# Phase 4 시스템 프롬프트 (Haiku — 각 트랙 Top 3 선정 전용)
PHASE4_SYSTEM_PROMPT = """You are a clinical prioritization assistant.
Given a disease ranking list, select the Top 3 most clinically relevant diseases.
Output ONLY a JSON array. No explanation, no markdown.

Output format:
[
  {"rank": 1, "disease_name": "...", "orpha_code": "ORPHA:NNN or null", "score": 0.0, "source": "rare or general", "is_rare": true or false},
  {"rank": 2, ...},
  {"rank": 3, ...}
]"""

# ══════════════════════════════════════════════════════════════════
# 시스템 프롬프트 — 확정 문서 §2 (글자 그대로)
# ══════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an elite AI diagnostician specializing in pulmonary and rare diseases.
You synthesize multimodal patient data and RAG-retrieved evidence to generate
a professional diagnostic support report.
Your role is to support physician decision-making, not to make final diagnoses.
Write your final report clearly, logically, and professionally.
Ensure your final output is written in Korean as requested by the clinical team.

[strict rules]

1. Do not make assumptions or definitive conclusions without evidence.
   (근거 없는 추측이나 단정을 포함하지 않습니다.)

2. All claims must be grounded in the provided RAG data or established
   medical guidelines.
   (모든 주장은 제공된 RAG 데이터 또는 공인된 의학 가이드라인에 근거해야 합니다.)

3. If a rare disease (OrphaCode) appears in rare_top3, MDT referral is mandatory.
   (희귀질환이 rare_top3에 있으면 MDT 협진 권고는 필수입니다.)

4. Prioritize diseases cross-validated by both Local DB and Global API.
   (로컬 DB·글로벌 API 양쪽 교차검증 질환을 우선순위로 삼습니다.)

5. Differential Diagnosis: Use the patient's negative findings to
   logically explain why certain candidate diseases should be ruled out.
   (감별 진단: Negative HPO를 활용하여 배제 근거를 논리적으로 설명합니다.)

6. Case Comparison: Compare and contrast the patient's current state
   with the provided PubMed case reports.
   (사례 비교: PubMed 케이스리포트와 현재 환자 상태를 비교·대조합니다.)

7. Actionable Alternatives: Synthesize the Clinical Trials data to
   recommend practical clinical trial opportunities for the patient.
   (실행 가능한 대안: ClinicalTrials 데이터를 종합하여 임상시험 참여 기회를 권고합니다.)

[Output Format Rules]
Output must strictly follow the JSON structure below.
Do not include any text outside this JSON.

{
  "general_diagnosis": [
    {
      "rank": 1,
      "disease_name": "일반 폐질환명",
      "score": 0.0,
      "icd10": ["J코드"],
      "reasoning": "진단 근거 (Positive/Negative HPO + Lab + X-ray 소견)",
      "treatment_guideline": "[질환명] 치료 가이드라인 (우선순위 순)",
      "recent_trend": "PubMed 최신 케이스리포트 기반 최신 치료 동향"
    }
  ],
  "rare_diagnosis": [
    {
      "rank": 1,
      "disease_name": "희귀 폐질환명",
      "orpha_code": "ORPHA:NNN",
      "lr_score": 0.0,
      "evidence": "근거 기반 질환 제시 (케이스리포트 + Orphadata 인용)",
      "genetic_test": ["유전자 검사 권고 항목"],
      "treatment_guideline": "[질환명] 치료 가이드라인",
      "recent_trend": "PubMed 최신 케이스리포트 기반 최신 치료 동향",
      "epidemiology": "Orphanet 유병률/발병연령/유전양식"
    }
  ],
  "recommendation": {
    "immediate_workup": ["즉시 시행할 검사"],
    "specialist_referral": ["협진 권고 (희귀질환 시 MDT 필수)"],
    "additional_lab": ["추가 혈액검사 권고"]
  },
  "clinical_notes": {
    "summary": "환자 종합 요약 (나이/성별/주소/이상 Lab 포함, MRN 제외)",
    "differential_note": "일반 vs 희귀 감별진단 논리 (Negative HPO 활용)",
    "rag_evidence": "RAG 수집 근거 요약 (DB·API 교차검증 결과 포함)",
    "case_comparison": "PubMed 케이스리포트와 현재 환자 비교",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  }
}

[Writing Principles]
1. general_diagnosis: 일반 폐질환 Top 3 각각에 대해 reasoning, treatment_guideline, recent_trend 작성.
2. rare_diagnosis: 희귀 폐질환 Top 3 각각에 대해 evidence(케이스리포트+Orphadata), genetic_test, treatment_guideline, recent_trend, epidemiology 작성.
   - rare_top3가 비어있으면 빈 배열 [] 출력.
3. genetic_test: association_type에 "Disease-causing"이 포함된 유전자만 포함.
4. treatment_guideline: 공인 가이드라인 기반, [질환명] prefix 필수.
5. recent_trend: PubMed 케이스리포트 인용 시 PMID 명시.
6. disclaimer 고정 문구는 반드시 포함. 수정 금지."""


# ══════════════════════════════════════════════════════════════════
# Orphanet CSV 경로 후보
# ══════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(__file__)
_ORPHANET_CSV_CANDIDATES = [
    os.path.join(_HERE, "data", "orphadata_weighted.csv"),
    os.path.join(_HERE, "..", "mini_project", "data", "orphadata_weighted.csv"),
]


# ══════════════════════════════════════════════════════════════════
# 메인 파이프라인 클래스
# ══════════════════════════════════════════════════════════════════
class RareLinkPipeline:
    """
    확정 문서 (2026-04-29) 기반 5단계 RAG 파이프라인

    입력
    ----
    patient_info : dict   환자 기본정보 (name, age, sex, visit_date, visit_type,
                          chief_complaint, allergy)
    xray_path    : str    흉부 X-ray 이미지 경로
    symptom_text : str    증상 원문 (Positive Findings)
    negative_text: str    음성 소견 원문 (Negative Findings)
    lab_results  : dict   Lab + Vital + Micro 수치

    출력
    ----
    dict  확정 JSON 구조 (recommendation + clinical_notes)
    """

    def __init__(
        self,
        vision_model_path: str = "model/chexnet_unet_crop_best.pth",
        orphanet_csv_path: Optional[str] = None,
        aws_region: str = AWS_REGION,
    ):
        print("=" * 60)
        print("🏥  Rare-Link AI 5단계 파이프라인 초기화 (확정 v1.0)")
        print("=" * 60)

        print("\n[1/5] X-ray 분류 모델(SooNet) 로드 중...")
        self.vision = SooNetEngine(model_path=vision_model_path)

        print("[2/5] Bedrock HPO 추출기 초기화 (Phase 1)...")
        self.hpo_extractor = BedrockHPOExtractor(region=aws_region)

        print("[3/5] Orphanet 희귀질환 DB 로드 중 (4335개)...")
        self.disease_db = self._load_disease_db(orphanet_csv_path)

        print("[4/5] Bedrock LLM 클라이언트 초기화 (Phase 4 + 소견서)...")
        self.bedrock = boto3.client("bedrock-runtime", region_name=aws_region)

        print("[5/5] PubMed 검색 엔진 연결 중...")
        self.pubmed = PubMedFetcher()

        print("\n✅ 파이프라인 초기화 완료\n")

    # ──────────────────────────────────────────────────────────────
    def _load_disease_db(self, csv_path: Optional[str] = None) -> dict:
        if csv_path is None:
            csv_path = next(
                (p for p in _ORPHANET_CSV_CANDIDATES if os.path.exists(p)), None
            )
        if csv_path and os.path.exists(csv_path):
            kb_df = pd.read_csv(csv_path)
            db = build_disease_database(kb_df)
            print(f"   ✅ {len(db)}개 질환 로드: {csv_path}")
            return db
        print("   ⚠️  orphadata_weighted.csv 없음 — 희귀 스코어링 비활성")
        return {}

    # ════════════════════════════════════════════════════════════════
    # ① Phase 1~3 — 멀티모달 → HPO 프로파일
    # ════════════════════════════════════════════════════════════════
    def step1_phase123_get_hpo(
        self,
        xray_path: str,
        symptom_text: str,
        negative_text: str,
        lab_results: dict,
    ) -> dict:
        """확정 §1.2 ① Phase 1~3 — 증상/X-ray/Lab+Vital+Micro → HPO 프로파일"""
        print("─" * 60)
        print("① Phase 1~3 — 멀티모달 HPO 변환")
        print("─" * 60)

        # Phase 1: 증상 → HPO (Bedrock Haiku)
        print("\n[Phase 1] 증상 → HPO (Bedrock Claude Haiku)...")
        # 확정 §3.2 — Positive Findings + Negative Findings 별도 처리
        nlp_pos = self.hpo_extractor.extract_hpo(symptom_text)
        pos_nlp = nlp_pos.get("positive_hpo", [])
        # symptom_text에서 자동 추출된 negative + 별도 negative_text 추출 결과 통합
        neg_from_pos = nlp_pos.get("negative_hpo", [])
        if negative_text and negative_text.strip():
            nlp_neg = self.hpo_extractor.extract_hpo(negative_text)
            neg_explicit = nlp_neg.get("positive_hpo", [])  # negative 텍스트의 양성 = 음성 소견
        else:
            neg_explicit = []
        neg_nlp = list(set(neg_from_pos + neg_explicit))
        print(f"  Positive (symptom): {pos_nlp}")
        print(f"  Negative (symptom only): {neg_nlp}")

        # Phase 2: X-ray → HPO (SooNet)
        print("\n[Phase 2] X-ray → HPO% (SooNet/SageMaker)...")
        xray_preds = self.vision.predict(xray_path)
        xray_hpos = [
            hpo
            for label, (prob, hpo) in xray_preds.items()
            if prob >= XRAY_THRESHOLD and "N/A" not in hpo
        ]
        print(f"  X-ray HPO (threshold≥{XRAY_THRESHOLD}): {xray_hpos}")

        # Phase 3: Lab + Vital + Micro → HPO
        print("\n[Phase 3] Lab + Vital + Micro → HPO (Rule-based)...")
        lab_hpos = lab_to_hpo(lab_results, verbose=True)

        # 통합 (Negative 우선 — 같은 코드가 양쪽에 있으면 Negative 우선)
        neg_set = set(neg_nlp)
        all_positive = list(set(xray_hpos + pos_nlp + lab_hpos) - neg_set)
        neg_clean = list(neg_set)

        # source 구분 (확정 §3.2 §3번 섹션 요구: source: symptom/xray)
        # Negative HPO는 "symptom only"로 명시
        pos_with_source = []
        for h in pos_nlp:
            if h in all_positive:
                pos_with_source.append({"hpo": h, "source": "symptom"})
        for h in xray_hpos:
            if h in all_positive and h not in pos_nlp:
                pos_with_source.append({"hpo": h, "source": "xray"})
        for h in lab_hpos:
            if h in all_positive and h not in pos_nlp and h not in xray_hpos:
                pos_with_source.append({"hpo": h, "source": "lab"})

        print(f"\n  통합 Positive HPO ({len(all_positive)}개): {all_positive}")
        print(f"  Negative HPO ({len(neg_clean)}개): {neg_clean}")

        return {
            "positive_hpo":      all_positive,
            "negative_hpo":      neg_clean,
            "positive_with_source": pos_with_source,
            "xray_detail":       xray_preds,
            "symptom_text":      symptom_text,
            "negative_text":     negative_text,
            "lab_results":       lab_results,
        }

    # ════════════════════════════════════════════════════════════════
    # ② 스코어링 분기 — 일반 Top 10 + 희귀 리스팅
    # ════════════════════════════════════════════════════════════════
    def step2_dual_scoring(self, hpo_data: dict) -> tuple:
        """
        확정 §1.2 ② 스코어링 분기 — 일반/기타 DB Top10 + 희귀 DB 리스팅

        Returns
        -------
        (general_top10: list, rare_listing: list)
        """
        print("\n" + "─" * 60)
        print("② 스코어링 분기 — 일반 Top10 + 희귀 리스팅 (병렬 2 트랙)")
        print("─" * 60)

        # 트랙 A: 일반/기타 폐질환 (Rule-based YAML 기반)
        print("\n[트랙 A] 일반/기타 폐질환 Rule-based Top 10...")
        general_ranking = rank_general_diseases(
            positive_hpos=hpo_data["positive_hpo"],
            xray_preds=hpo_data.get("xray_detail", {}),
            lab_results=hpo_data.get("lab_results", {}),
            top_k=10,
        )
        for i, d in enumerate(general_ranking[:5], 1):
            print(f"  {i:2d}. {d['disease_name']:<35} score={d['score']:.3f}")

        # 트랙 B: 희귀질환 (LIRICAL LR Listing)
        print("\n[트랙 B] 희귀질환 LIRICAL LR Listing...")
        if not self.disease_db:
            print("  ⚠️  희귀 DB 없음 — 빈 리스팅 반환")
            rare_listing = []
        else:
            rare_full = rank_diseases(
                positive_hpos=hpo_data["positive_hpo"],
                negative_hpos=hpo_data["negative_hpo"],
                disease_database=self.disease_db,
                top_k=10,
            )
            # LR 임계치 통과한 희귀질환만 Listing
            rare_listing = [
                d for d in rare_full
                if d.get("is_rare", False) and d.get("score", 0) >= RARE_LR_THRESHOLD
            ]
            print(f"  희귀 리스팅 ({len(rare_listing)}개, LR≥{RARE_LR_THRESHOLD}):")
            for i, d in enumerate(rare_listing[:5], 1):
                print(f"  {i:2d}. {d['disease_name']:<40} LR={d['score']:.4f}")

        return general_ranking, rare_listing

    # ════════════════════════════════════════════════════════════════
    # ③ Phase 4 — 일반 Top 3 + 희귀 Top 3 각각 선정
    # ════════════════════════════════════════════════════════════════
    def step3_phase4_organize(
        self,
        general_ranking: list,
        rare_listing: list,
        hpo_data: dict,
    ) -> dict:
        """
        회의록 §1.4 — 일반 질환 Top 3 + 희귀 질환 Top 3 각각 독립 반환

        Returns
        -------
        dict  {
            "general_top3": list[dict],   일반 폐질환 Top 3
            "rare_top3":    list[dict],   희귀 폐질환 Top 3 (없으면 빈 리스트)
        }
        """
        print("\n" + "─" * 60)
        print("③ Phase 4 — 일반 Top 3 + 희귀 Top 3 각각 선정")
        print("─" * 60)

        # ── 일반 질환 Top 3 ──────────────────────────────────────
        general_top3 = self._select_top3(
            candidates=general_ranking[:10],
            track="general",
            label="일반 폐질환",
        )

        # ── 희귀 질환 Top 3 ──────────────────────────────────────
        rare_top3 = self._select_top3(
            candidates=rare_listing[:10],
            track="rare",
            label="희귀 폐질환",
        ) if rare_listing else []

        print(f"\n  ✅ 일반 Top 3:")
        for d in general_top3:
            print(f"     {d['rank']}. {d['disease_name']} (score={d['score']:.3f})")

        if rare_top3:
            print(f"\n  ✅ 희귀 Top 3:")
            for d in rare_top3:
                orpha = f" {d['orpha_code']}" if d.get("orpha_code") else ""
                print(f"     {d['rank']}. {d['disease_name']}{orpha} (LR={d['score']:.3f})")
        else:
            print(f"\n  ℹ️  희귀 질환 후보 없음 (LR≥{RARE_LR_THRESHOLD} 통과 없음)")

        return {"general_top3": general_top3, "rare_top3": rare_top3}

    def _select_top3(self, candidates: list, track: str, label: str) -> list:
        """Haiku로 Top 3 선정. 실패 시 score 순 상위 3개 폴백."""
        if not candidates:
            return []

        # orpha_code 는 이미 "ORPHA:NNN" 형식 → 프롬프트에 그대로 넣어야 이중 prefix 방지
        lines = "\n".join(
            f"  {i:2d}. {d['disease_name']} "
            f"(score={d['score']:.4f}"
            + (f", orpha_code={d.get('orpha_code','null')}" if track == "rare" else "")
            + ")"
            for i, d in enumerate(candidates[:10], 1)
        )
        user_prompt = (
            f"{label} ranking:\n{lines}\n\n"
            "Select the Top 3 most clinically relevant diseases and output JSON only."
        )

        # candidates를 disease_name → 원본 dict 로 인덱싱 (Haiku 환각 방지용)
        candidate_map = {d["disease_name"]: d for d in candidates[:10]}

        try:
            resp = self.bedrock.invoke_model(
                modelId=PHASE4_MODEL,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": PHASE4_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": PHASE4_MAX_TOKENS,
                    "temperature": PHASE4_TEMPERATURE,
                }),
            )
            raw_text = json.loads(resp["body"].read())["content"][0]["text"].strip()
            import re as _re
            m = _re.search(r"\[.*\]", raw_text, _re.DOTALL)
            if m:
                haiku_result = json.loads(m.group())
                result = []
                for i, d in enumerate(haiku_result[:3], 1):
                    name = d.get("disease_name", "")
                    # Haiku 가 반환한 orpha_code 대신 candidates 원본값 우선 사용 (환각 방지)
                    original = candidate_map.get(name, {})
                    d["rank"]        = i
                    d["orpha_code"]  = original.get("orpha_code", d.get("orpha_code"))
                    d["score"]       = original.get("score", d.get("score", 0.0))
                    d["source"]      = track
                    d["is_rare"]     = track == "rare"
                    d.setdefault("genes", original.get("genes", []))
                    result.append(d)
                return result
        except Exception as e:
            print(f"  ⚠️  Haiku {label} 선정 실패: {e} → 폴백")

        # 폴백: score 순 상위 3개
        result = []
        for i, d in enumerate(candidates[:3], 1):
            result.append({
                "rank":         i,
                "disease_name": d["disease_name"],
                "orpha_code":   d.get("orpha_code"),
                "score":        d["score"],
                "source":       track,
                "is_rare":      track == "rare",
                "genes":        d.get("genes", []),
                "icd10":        d.get("icd10", []),
                "prevalence":   d.get("prevalence", "N/A"),
            })
        return result

    # ════════════════════════════════════════════════════════════════
    # ④ RAG 트리거 — 이미지 확정 흐름 (2026-04-29 회의)
    #
    # 흐름:
    #   HPO list
    #     → [1단계] PubCaseFinder (1번 관문) — HPO → 후보 질환 ID/이름/점수
    #     → [2단계] 병렬 메타데이터 보강
    #         ├─ Monarch API  — 후보 질환 ID → 인과 유전자
    #         └─ Orphanet API — 후보 질환 ID → 역학/발현형 메타데이터
    #     → [3단계] 병렬 근거 수집 (Top 3 질환명 기준)
    #         ├─ PubMed        — 질환명 → 케이스리포트 초록
    #         └─ ClinicalTrials — 질환명 → 모집 중 임상시험
    #     → local DB 결과 vs API 결과 비교·검증
    #     → 프롬프트 템플릿 삽입 → Bedrock 전송
    # ════════════════════════════════════════════════════════════════
    def step4_rag_collect(
        self,
        top3_result: dict,
        rare_listing: list,
        hpo_data: dict,
    ) -> dict:
        """
        회의록 §1.4 — 일반 Top 3 + 희귀 Top 3 각각 RAG 수집

        일반 Top 3: PubMed + ClinicalTrials (치료 근거)
        희귀 Top 3: PubCaseFinder + Orphanet + Monarch + PubMed + ClinicalTrials (전부)

        Returns
        -------
        dict  {
            "general": { "top1": {...}, "top2": {...}, "top3": {...} },
            "rare":    { "top1": {...}, "top2": {...}, "top3": {...} },
            "cross_validation": {...},
            "pubcasefinder": [...],
            "rare_listing_text": str,
        }
        """
        print("\n" + "─" * 60)
        print("④ RAG 트리거 — 일반 Top 3 + 희귀 Top 3 각각 수집")
        print("─" * 60)

        general_top3 = top3_result.get("general_top3", [])
        rare_top3    = top3_result.get("rare_top3", [])
        has_rare     = len(rare_top3) > 0

        rare_listing_text = self._format_rare_listing(rare_listing) if rare_listing else "해당 없음"

        api_general = {"top1": {}, "top2": {}, "top3": {}}
        api_rare    = {"top1": {}, "top2": {}, "top3": {}}

        # ── [1단계] PubCaseFinder — 희귀 케이스에만 ──────────────
        print("\n  [1단계] PubCaseFinder (희귀 케이스 전용)")
        pcf_results = None
        if has_rare:
            pcf_results = self._safe_call(
                "pubcasefinder", get_ranked_diseases,
                hpo_data["positive_hpo"],
                "omim",          # target (기본값 명시)
                TOP_K_PER_API,   # top_k
            )
            if pcf_results:
                from rag.pubcasefinder import enrich_pcf_results
                pcf_results = enrich_pcf_results(pcf_results, fetch_pmids=False)
                print(f"  ✅ PubCaseFinder → {len(pcf_results)}개 후보")
            else:
                print("  ⚠️  PubCaseFinder 실패 → 로컬 폴백")
        else:
            print("  ℹ️  희귀 후보 없음 → 스킵")

        # ── [2단계] 희귀 Top 3: Monarch + Orphanet 병렬 보강 ─────
        if has_rare:
            print("\n  [2단계] 희귀 Top 3 — Monarch + Orphanet 병렬 보강")
            with ThreadPoolExecutor(max_workers=6) as executor:
                meta_futures = {}
                for i, disease in enumerate(rare_top3[:3], 1):
                    key = f"top{i}"
                    orpha_code = disease.get("orpha_code")
                    meta_futures[executor.submit(
                        self._safe_call, "monarch_genes", get_causal_genes,
                        orpha_code or disease["disease_name"],
                    )] = (key, "monarch_genes")
                    if orpha_code:
                        meta_futures[executor.submit(
                            self._safe_call, "orphanet", get_orphanet_data,
                            orpha_code,
                        )] = (key, "orphanet")

                for future in as_completed(meta_futures, timeout=RAG_PARALLEL_TIMEOUT):
                    key, api_name = meta_futures[future]
                    try:
                        api_rare[key][api_name] = future.result(timeout=15)
                        print(f"  ✅ 희귀 {key} {api_name} 완료")
                    except Exception as e:
                        print(f"  ⚠️  희귀 {key} {api_name} 실패: {e}")
                        api_rare[key][api_name] = None

        # ── [3단계] 일반 + 희귀 각각 PubMed + ClinicalTrials ─────
        print("\n  [3단계] 일반 + 희귀 각각 PubMed + ClinicalTrials 병렬 수집")
        with ThreadPoolExecutor(max_workers=12) as executor:
            ev_futures = {}
            # 일반 Top 3
            for i, disease in enumerate(general_top3[:3], 1):
                key = f"top{i}"
                name = disease["disease_name"]
                ev_futures[executor.submit(
                    self._safe_call, "pubmed", self.pubmed.get_top_papers, name, TOP_K_PER_API,
                )] = ("general", key, "pubmed")
                ev_futures[executor.submit(
                    self._safe_call, "clinical_trials", get_clinical_trials, name, TOP_K_PER_API,
                )] = ("general", key, "clinical_trials")
            # 희귀 Top 3
            for i, disease in enumerate(rare_top3[:3], 1):
                key = f"top{i}"
                name = disease["disease_name"]
                ev_futures[executor.submit(
                    self._safe_call, "pubmed", self.pubmed.get_top_papers, name, TOP_K_PER_API,
                )] = ("rare", key, "pubmed")
                ev_futures[executor.submit(
                    self._safe_call, "clinical_trials", get_clinical_trials, name, TOP_K_PER_API,
                )] = ("rare", key, "clinical_trials")

            for future in as_completed(ev_futures, timeout=RAG_PARALLEL_TIMEOUT):
                track, key, api_name = ev_futures[future]
                target = api_general if track == "general" else api_rare
                try:
                    target[key][api_name] = future.result(timeout=15)
                    print(f"  ✅ {track} {key} {api_name} 완료")
                except Exception as e:
                    print(f"  ⚠️  {track} {key} {api_name} 실패: {e}")
                    target[key][api_name] = None

        # ── 희귀 교차검증 ─────────────────────────────────────────
        cross_validation = {}
        for i, disease in enumerate(rare_top3[:3], 1):
            key = f"top{i}"
            orpha_data    = api_rare.get(key, {}).get("orphanet")
            monarch_genes = api_rare.get(key, {}).get("monarch_genes") or []
            orpha_genes   = orpha_data.get("genes_from_orphadata", []) if orpha_data else []
            cross_validation[key] = cross_validate_genes(orpha_genes, monarch_genes)

        return {
            "general":          api_general,
            "rare":             api_rare,
            "cross_validation": cross_validation,
            "pubcasefinder":    pcf_results,
            "rare_listing_text": rare_listing_text,
            "general_top3":     general_top3,
            "rare_top3":        rare_top3,
        }

    @staticmethod
    def _safe_call(api_name: str, func, *args, **kwargs):
        """API 호출 wrapper — 예외 발생 시 None 반환"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"     ⚠️  {api_name} 호출 실패: {e}")
            return None

    @staticmethod
    def _format_rare_listing(rare_listing: list) -> str:
        if not rare_listing:
            return "해당 없음"
        lines = []
        for i, d in enumerate(rare_listing[:10], 1):
            orpha_str = d.get("orpha_code", "N/A")
            lines.append(
                f"{i:2d}. {d['disease_name']} "
                f"({orpha_str}, LR={d['score']:.4f})"
            )
        return "\n".join(lines)

    # ════════════════════════════════════════════════════════════════
    # ⑤ LLM 소견서 — 8개 섹션 유저 프롬프트 → JSON
    # ════════════════════════════════════════════════════════════════
    def step5_generate_report(
        self,
        patient_info: dict,
        hpo_data: dict,
        general_ranking: list,
        rare_listing: list,
        rag_context: dict,
    ) -> dict:
        """
        확정 §1.2 ⑤ LLM 소견서 생성 — Bedrock Sonnet 3.5 → JSON only

        확정 §3.2 8개 섹션 유저 프롬프트 구성:
          1. 환자 기본정보
          2. 증상 원문
          3. HPO 프로파일
          4. Lab 수치
          5. 일반/기타 폐질환 랭킹 Top 10
          6. 희귀폐질환 리스팅
          7. 내부 DB 정보 (Top 3 교차검증)
          8. RAG 검색 결과 (외부 API)

        Returns
        -------
        dict  확정 출력 JSON (recommendation + clinical_notes)
        """
        print("\n" + "─" * 60)
        print("⑤ LLM 소견서 생성 — Bedrock Claude Sonnet 3.5 (JSON only)")
        print("─" * 60)

        user_prompt = self._build_user_prompt(
            patient_info, hpo_data,
            general_ranking, rare_listing, rag_context,
        )

        print(f"\n  [Bedrock] {REPORT_MODEL} 호출 중...")
        try:
            resp = self.bedrock.invoke_model(
                modelId=REPORT_MODEL,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": user_prompt}],
                    "max_tokens": REPORT_MAX_TOKENS,
                    "temperature": REPORT_TEMPERATURE,
                }),
            )
            raw_text = json.loads(resp["body"].read())["content"][0]["text"]

            # JSON 파싱 (LLM이 코드블록 감쌀 수 있음)
            structured = self._parse_json_response(raw_text)

            if structured:
                print(f"  ✅ JSON 파싱 성공")
                self._validate_schema(structured)
                return structured
            else:
                print(f"  ⚠️  JSON 파싱 실패 — raw 반환")
                return {"raw_output": raw_text, "error": "json_parse_failed"}

        except ClientError as e:
            err = e.response["Error"]
            print(f"❌ Bedrock 오류: {err['Code']} — {err['Message']}")
            return {"error": err["Message"]}

    # ──────────────────────────────────────────────────────────────
    # 유저 프롬프트 빌더 (확정 §3.2 8개 섹션)
    # ──────────────────────────────────────────────────────────────
    def _build_user_prompt(
        self,
        patient_info: dict,
        hpo_data: dict,
        general_ranking: list,
        rare_listing: list,
        rag_context: dict,
    ) -> str:
        # === 1. 환자 기본정보 (MRN 제외) ===
        safe_info = {
            "name":            patient_info.get("name", ""),
            "age":             patient_info.get("age", ""),
            "sex":             patient_info.get("sex", ""),
            "visit_date":      patient_info.get("visit_date", ""),
            "visit_type":      patient_info.get("visit_type", ""),
            "chief_complaint": patient_info.get("chief_complaint", ""),
            "allergy":         patient_info.get("allergy", ""),
        }

        # === 3. HPO 프로파일 ===
        pos_lines = [
            f"  - {item['hpo']} ({item['source']})"
            for item in hpo_data.get("positive_with_source", [])
        ]
        pos_text = "\n".join(pos_lines) or "  - 없음"
        neg_text = "\n".join(f"  - {h}" for h in hpo_data.get("negative_hpo", [])) or "  - 없음"

        # === X-ray 예측 ===
        xray_detail = hpo_data.get("xray_detail", {})
        if xray_detail:
            xray_sorted = sorted(xray_detail.items(), key=lambda x: x[1][0], reverse=True)
            xray_text = "\n".join(
                f"  - {label}: {prob:.3f} → HPO: {hpo}"
                for label, (prob, hpo) in xray_sorted[:10]
            )
        else:
            xray_text = "  데이터 없음"

        # === 4. Lab 수치 ===
        lab_text = "\n".join(
            f"  - {k}: {v}" for k, v in hpo_data.get("lab_results", {}).items()
        ) or "  데이터 없음"

        # === 5. 일반 폐질환 Top 10 ===
        general_text = "\n".join(
            f"{i:2d}. {d['disease_name']} (score={d['score']:.3f})"
            for i, d in enumerate(general_ranking[:10], 1)
        ) or "데이터 없음"

        # === 6. 희귀 폐질환 리스팅 ===
        rare_listing_text = rag_context.get("rare_listing_text", "해당 없음")

        # === 7. 교차검증 ===
        cv = rag_context.get("cross_validation", {})
        cv_lines = [
            f"희귀 Top {i} 교차검증: {cv.get(f'top{i}', {}).get('summary', '데이터 없음')}"
            for i in range(1, 4)
            if cv.get(f"top{i}")
        ]
        cv_text = "\n".join(cv_lines) or "교차검증 데이터 없음"

        # === 8. RAG 검색 결과 (일반 + 희귀 각각) ===
        section8_text = self._build_section8(rag_context)

        return f"""아래 다중 모달리티 임상 데이터와 RAG 수집 결과를 분석하여,
시스템 프롬프트에 명시된 JSON 형식으로 진단 보조 소견서를 작성하십시오.
일반 폐질환 Top 3와 희귀 폐질환 Top 3를 각각 독립적으로 작성하십시오.

=========================================
=== 1. 환자 기본정보 ===
{json.dumps(safe_info, ensure_ascii=False, indent=2)}

=== 2. 증상 원문 ===
- Positive Findings: {hpo_data.get('symptom_text', '')}
- Negative Findings: {hpo_data.get('negative_text', '')}

=== 2-1. X-ray 예측 결과 (SooNet, 확률값 Top 10) ===
{xray_text}

=== 3. HPO 프로파일 ===
Positive HPO (source: symptom/xray/lab):
{pos_text}
Negative HPO:
{neg_text}

=== 4. Lab 수치 ===
{lab_text}

=== 5. 일반 폐질환 랭킹 Top 10 (로컬 DB 기반) ===
{general_text}

=== 6. 희귀 폐질환 리스팅 (LIRICAL LR 기반) ===
{rare_listing_text}

=== 7. 희귀 질환 교차검증 (로컬 DB vs 외부 API) ===
{cv_text}

=== 8. RAG 검색 결과 ===
{section8_text}
=========================================

위 데이터를 종합하여 규정된 JSON 형식으로 출력하십시오.
general_diagnosis에는 일반 폐질환 Top 3를,
rare_diagnosis에는 희귀 폐질환 Top 3를 각각 작성하십시오.
희귀 후보가 없으면 rare_diagnosis는 빈 배열 []로 출력하십시오."""

    def _build_section8(self, rag_context: dict) -> str:
        """일반 Top 3 + 희귀 Top 3 각각 RAG 블록 구성"""
        general_top3 = rag_context.get("general_top3", [])
        rare_top3    = rag_context.get("rare_top3", [])
        api_general  = rag_context.get("general", {})
        api_rare     = rag_context.get("rare", {})
        pcf          = rag_context.get("pubcasefinder") or []

        sections = []

        # ── 일반 폐질환 RAG ──────────────────────────────────────
        sections.append("=== [일반 폐질환 RAG 결과] ===")
        for i, disease in enumerate(general_top3[:3], 1):
            key = f"top{i}"
            d_name = disease["disease_name"]
            block = [f"--- 일반 Top {i}: {d_name} ---"]

            papers = (api_general.get(key) or {}).get("pubmed") or []
            if papers:
                block.append("[PubMed 케이스리포트]")
                for p in papers:
                    block.append(
                        f"  - PMID:{p.get('pmid')} | {p.get('title', '')} ({p.get('pubdate', '')})\n"
                        f"    {p.get('abstract', '')[:300]}..."
                    )
            else:
                block.append("[PubMed 케이스리포트]\n  데이터 없음")

            trials = (api_general.get(key) or {}).get("clinical_trials") or []
            if trials:
                block.append("[ClinicalTrials (RECRUITING)]")
                for t in trials:
                    block.append(
                        f"  - NCT:{t.get('nct_id')} | {t.get('title', '')[:80]}\n"
                        f"    Phase:{t.get('phase', 'N/A')} | Status:{t.get('status', 'N/A')}"
                    )
            else:
                block.append("[ClinicalTrials]\n  데이터 없음")

            sections.append("\n".join(block))

        # ── 희귀 폐질환 RAG ──────────────────────────────────────
        if rare_top3:
            sections.append("\n=== [희귀 폐질환 RAG 결과] ===")

            # PubCaseFinder (1번 관문)
            if pcf:
                pcf_lines = ["[PubCaseFinder — HPO 기반 희귀질환 후보]"]
                for r in pcf[:5]:
                    pcf_lines.append(
                        f"  - {r.get('disease_name', '')} "
                        f"(score={r.get('score', 0):.3f}, genes={', '.join(r.get('genes', []))})"
                    )
                sections.append("\n".join(pcf_lines))

            for i, disease in enumerate(rare_top3[:3], 1):
                key = f"top{i}"
                d_name = disease["disease_name"]
                orpha  = disease.get("orpha_code") or "N/A"
                block  = [f"--- 희귀 Top {i}: {d_name} (ORPHA:{orpha}) ---"]

                # Orphanet
                orpha_data = (api_rare.get(key) or {}).get("orphanet")
                if orpha_data:
                    block.append(format_orphanet_for_prompt(orpha_data))
                else:
                    block.append("[Orphanet] 데이터 없음")

                # Monarch
                monarch_genes = (api_rare.get(key) or {}).get("monarch_genes") or []
                orpha_genes   = (orpha_data or {}).get("genes_from_orphadata", [])
                block.append(format_monarch_for_prompt(orpha if orpha != "N/A" else d_name, orpha_genes))

                # PubMed
                papers = (api_rare.get(key) or {}).get("pubmed") or []
                if papers:
                    block.append("[PubMed 케이스리포트]")
                    for p in papers:
                        block.append(
                            f"  - PMID:{p.get('pmid')} | {p.get('title', '')} ({p.get('pubdate', '')})\n"
                            f"    {p.get('abstract', '')[:300]}..."
                        )
                else:
                    block.append("[PubMed 케이스리포트]\n  데이터 없음")

                # ClinicalTrials
                trials = (api_rare.get(key) or {}).get("clinical_trials") or []
                if trials:
                    block.append("[ClinicalTrials (RECRUITING)]")
                    for t in trials:
                        block.append(
                            f"  - NCT:{t.get('nct_id')} | {t.get('title', '')[:80]}\n"
                            f"    Phase:{t.get('phase', 'N/A')} | Status:{t.get('status', 'N/A')}"
                        )
                else:
                    block.append("[ClinicalTrials]\n  데이터 없음")

                sections.append("\n".join(block))
        else:
            sections.append("\n=== [희귀 폐질환 RAG 결과] ===\n희귀 질환 후보 없음 — 스킵")

        return "\n\n".join(sections)

    # ──────────────────────────────────────────────────────────────
    # JSON 파싱 + 스키마 검증
    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_json_response(raw: str) -> Optional[dict]:
        """LLM 응답에서 JSON 추출 (코드블록 제거)"""
        import re
        # ```json ... ``` 제거
        m = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
        # raw가 그대로 JSON일 수도
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _validate_schema(data: dict) -> None:
        """출력 스키마 필수 필드 검증 (일반/희귀 분리 구조)"""
        required_top = ["general_diagnosis", "rare_diagnosis", "recommendation", "clinical_notes"]
        required_rec = ["immediate_workup", "specialist_referral", "additional_lab"]
        required_notes = ["summary", "differential_note", "rag_evidence", "case_comparison", "disclaimer"]

        missing = []
        for k in required_top:
            if k not in data:
                missing.append(k)
        if "recommendation" in data:
            for k in required_rec:
                if k not in data["recommendation"]:
                    missing.append(f"recommendation.{k}")
        if "clinical_notes" in data:
            for k in required_notes:
                if k not in data["clinical_notes"]:
                    missing.append(f"clinical_notes.{k}")
        if missing:
            print(f"  ⚠️  스키마 누락 필드: {missing}")
        else:
            print(f"  ✅ 스키마 검증 통과")

    # ════════════════════════════════════════════════════════════════
    # 전체 5단계 실행
    # ════════════════════════════════════════════════════════════════
    def run(
        self,
        patient_info: dict,
        xray_path: str,
        symptom_text: str,
        negative_text: str,
        lab_results: dict,
    ) -> dict:
        """
        확정 5단계 파이프라인 전체 실행

        Parameters
        ----------
        patient_info  : dict  name, age, sex, visit_date, visit_type,
                              chief_complaint, allergy
        xray_path     : str   X-ray 이미지 경로
        symptom_text  : str   양성 증상 원문
        negative_text : str   음성 소견 원문 (없으면 빈 문자열)
        lab_results   : dict  Lab + Vital + Micro 수치

        Returns
        -------
        dict  확정 JSON (recommendation + clinical_notes)
        """
        print("\n" + "=" * 60)
        print("🏥  Rare-Link AI 진단 보조 파이프라인 (5단계 v1.0)")
        print("=" * 60)

        # ① Phase 1~3
        hpo_data = self.step1_phase123_get_hpo(
            xray_path, symptom_text, negative_text, lab_results,
        )

        # ② 스코어링 분기
        general_ranking, rare_listing = self.step2_dual_scoring(hpo_data)

        # ③ Phase 4 — 일반 Top 3 + 희귀 Top 3 각각 선정
        top3_result = self.step3_phase4_organize(general_ranking, rare_listing, hpo_data)

        # ④ RAG 트리거 — 일반/희귀 각각 API 병렬
        rag_context = self.step4_rag_collect(top3_result, rare_listing, hpo_data)

        # ⑤ LLM 소견서
        report_json = self.step5_generate_report(
            patient_info, hpo_data, general_ranking, rare_listing, rag_context,
        )

        print("\n" + "=" * 60)
        print("✅ 파이프라인 완료 (5단계)")
        print("=" * 60)
        return report_json


# ══════════════════════════════════════════════════════════════════
# 직접 실행 — 내장 샘플
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    pipeline = RareLinkPipeline(
        vision_model_path="model/chexnet_unet_crop_best.pth",
    )

    sample_patient = {
        "name":            "익명",
        "age":             40,
        "sex":             "F",
        "visit_date":      "2026-05-04",
        "visit_type":      "외래",
        "chief_complaint": "3주째 지속되는 호흡곤란과 우측 흉통",
        "allergy":         "없음",
    }

    sample_lab = {
        "WBC":   12.5,
        "HGB":    9.8,
        "LDH":   310,
        "CRP":    7.2,
        "SpO2":  92.0,
        "FEV1":  68.0,
    }

    report = pipeline.run(
        patient_info  = sample_patient,
        xray_path     = "test_xray.jpg",
        symptom_text  = (
            "40세 여성. 3주째 지속되는 호흡곤란과 우측 흉통을 호소합니다. "
            "최근 체중 감소가 있었습니다."
        ),
        negative_text = "기침은 없으며 발열도 없습니다.",
        lab_results   = sample_lab,
    )

    print("\n" + "=" * 60)
    print("📋 최종 진단 보조 리포트 (JSON)")
    print("=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2))
