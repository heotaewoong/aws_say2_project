# rag_pipeline.py
# Rare-Link AI — 전체 파이프라인 오케스트레이터
#
# ①INPUT(X-ray + NLP + Lab) → ②SCORE(LIRICAL) → ③TRIG(RAG?) → ④GEN(Claude) → ⑤OUT
#
# 사용법:
#   python rag_pipeline.py
#   또는: from rag_pipeline import RareLinkPipeline

import json
import os

import boto3
import pandas as pd
from botocore.exceptions import ClientError

# ── rag/ 패키지 컴포넌트 ───────────────────────────────────────────
from soo_net import SooNetEngine
from rag.bedrock_extractor import BedrockHPOExtractor
from rag.lab_rules import lab_to_hpo
from rag.lirical_scorer import build_disease_database, rank_diseases
from rag.pubcasefinder import get_ranked_diseases, format_pcf_for_llm

# ── PubMed 논문 검색 ──────────────────────────────────────────────
from rag.pubmed_fetcher import PubMedFetcher
from rag.clinicaltrials_fetcher import get_clinical_trials, format_trials_for_llm
_PUBMED_AVAILABLE = True

# ── 상수 ──────────────────────────────────────────────────────────
XRAY_THRESHOLD      = 0.3    # X-ray HPO 필터링 임계 확률 (0.4→0.3: 테스트 결과 0.4로는 항상 0개)
SCORE_RATIO_THRESH  = 3.0    # 1등/2등 LR 비율 < 이 값이면 불확실 → RAG 실행
                              # ※ 회의록: 미확정. 팀 협의 후 조정 필요
PUBCASE_MAX         = 5      # PubCaseFinder 검색 케이스 최대 수
REPORT_MODEL        = "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
REPORT_MAX_TOKENS   = 2000
AWS_REGION          = "ap-northeast-2"

# Orphanet CSV 기본 탐색 경로 목록
_HERE = os.path.dirname(__file__)
_ORPHANET_CSV_CANDIDATES = [
    os.path.join(_HERE, "data", "orphadata_weighted.csv"),
    os.path.join(_HERE, "..", "mini_project", "data", "orphadata_weighted.csv"),
]


class RareLinkPipeline:
    """
    Rare-Link AI 전체 진단 보조 파이프라인

    입력
    ----
    xray_path    : 흉부 X-ray 이미지 경로 (JPEG / PNG)
    symptom_text : 임상 소견 자유 텍스트 (한국어/영어)
    lab_results  : 혈액·폐기능 검사 수치 딕셔너리
                   예) {"WBC": 15.2, "HGB": 9.1, "SpO2": 91.0, "FEV1": 65.0}

    출력
    ----
    str  마크다운 형식의 진단 보조 리포트

    파이프라인 5단계
    ----------------
    ① INPUT  — X-ray(SooNet) + NLP(Bedrock) + Lab(Rule) → HPO 통합
    ② SCORE  — LIRICAL LR 스코어링 → Top 10 질환 랭킹
    ③ TRIG   — 희귀질환 여부 / 불확실성 → RAG 실행 여부 결정
    ④ GEN    — RAG 검색 + Claude Bedrock → 소견서 생성
    ⑤ OUT    — 최종 리포트 반환
    """

    def __init__(
        self,
        vision_model_path: str = "model/chexnet_unet_crop_best.pth",
        orphanet_csv_path: str = None,
        aws_region: str = AWS_REGION,
    ):
        print("=" * 60)
        print("🏥  Rare-Link AI 파이프라인 초기화")
        print("=" * 60)

        print("\n[1/4] X-ray 분류 모델(SooNet) 로드 중...")
        self.vision = SooNetEngine(model_path=vision_model_path)

        print("[2/4] Bedrock HPO 추출기 초기화...")
        self.hpo_extractor = BedrockHPOExtractor(region=aws_region)

        print("[3/4] Orphanet 질환 DB 로드 중...")
        self.disease_db = self._load_disease_db(orphanet_csv_path)

        print("[4/4] Bedrock LLM 클라이언트 초기화...")
        self.bedrock = boto3.client("bedrock-runtime", region_name=aws_region)

        if _PUBMED_AVAILABLE:
            print("   ✅ PubMed 검색 엔진 연결됨 (rag/pubmed_fetcher.py)")
            self.pubmed = PubMedFetcher()
        else:
            print("   ⚠️  PubMed 검색 비활성 — pip install chromadb sentence-transformers")
            self.pubmed = None

        print("\n✅ 파이프라인 초기화 완료!\n")

    # ────────────────────────────────────────────────────────────────
    def _load_disease_db(self, csv_path: str = None) -> dict:
        if csv_path is None:
            csv_path = next(
                (p for p in _ORPHANET_CSV_CANDIDATES if os.path.exists(p)), None
            )

        if csv_path and os.path.exists(csv_path):
            kb_df = pd.read_csv(csv_path)
            db = build_disease_database(kb_df)
            print(f"   ✅ {len(db)}개 질환 로드 완료: {csv_path}")
            return db

        print("   ⚠️  orphadata_weighted.csv 없음 → 스코어링 비활성화")
        print("       mini_project/knowledge_base.py 로 먼저 데이터를 생성하세요.")
        return {}

    # ════════════════════════════════════════════════════════════════
    # ① INPUT
    # ════════════════════════════════════════════════════════════════
    def step1_get_hpo(
        self,
        xray_path: str,
        symptom_text: str,
        lab_results: dict,
    ) -> dict:
        """3개 입력 소스 → HPO 집합 통합"""
        print("─" * 50)
        print("① INPUT — 다중 모달리티 HPO 변환")
        print("─" * 50)

        # 1-A. X-ray → HPO (SooNetEngine, DenseNet-121 + U-Net)
        print("\n[X-ray] SooNet 분석 중...")
        xray_preds = self.vision.predict(xray_path)
        xray_hpos = [
            hpo
            for label, (prob, hpo) in xray_preds.items()
            if prob >= XRAY_THRESHOLD and "N/A" not in hpo
        ]
        print(f"  HPO (threshold≥{XRAY_THRESHOLD}): {xray_hpos}")

        # 1-B. 임상 소견 텍스트 → HPO (Claude Bedrock Haiku)
        print("\n[NLP] 임상 소견 HPO 추출 중...")
        nlp_result = self.hpo_extractor.extract_hpo(symptom_text)
        pos_nlp = nlp_result.get("positive_hpo", [])
        neg_nlp = nlp_result.get("negative_hpo", [])
        print(f"  Positive: {pos_nlp}")
        print(f"  Negative: {neg_nlp}")

        # 1-C. 혈액·폐기능 검사 → HPO (Rule-based)
        print("\n[Lab] 검사 수치 HPO 변환 중...")
        lab_hpos = lab_to_hpo(lab_results, verbose=True)

        # 1-D. 통합 (중복 제거 + Positive/Negative 충돌 해결)
        # Negative에 있는 HPO는 Positive에서 제거 (같은 코드가 양쪽에 있으면 Negative 우선)
        neg_set = set(neg_nlp)
        all_positive = list(set(xray_hpos + pos_nlp + lab_hpos) - neg_set)
        neg_clean    = list(neg_set)
        print(f"\n  통합 Positive HPO ({len(all_positive)}개): {all_positive}")
        print(f"  Negative HPO ({len(neg_clean)}개): {neg_clean}")

        return {
            "positive_hpo": all_positive,
            "negative_hpo": neg_clean,
            "xray_detail":  xray_preds,
            "symptom_text": symptom_text,
            "lab_results":  lab_results,   # Lab raw 수치 (LLM 인풋용)
        }

    # ════════════════════════════════════════════════════════════════
    # ② SCORE
    # ════════════════════════════════════════════════════════════════
    def step2_score(self, hpo_data: dict) -> list:
        """LIRICAL LR 스코어링 → Top 10 질환 랭킹"""
        print("\n" + "─" * 50)
        print("② SCORE — LIRICAL LR 스코어링")
        print("─" * 50)

        if not self.disease_db:
            print("  ⚠️  질환 DB 없음 — 스코어링 건너뜀")
            return [{"orpha_code": "N/A", "disease_name": "DB 없음 (knowledge_base.py 실행 필요)",
                     "score": 0.0, "is_rare": False, "prevalence": "N/A", "genes": []}]

        ranking = rank_diseases(
            positive_hpos=hpo_data["positive_hpo"],
            negative_hpos=hpo_data["negative_hpo"],
            disease_database=self.disease_db,
            top_k=10,
        )

        print("  Top 10 질환 랭킹:")
        for i, d in enumerate(ranking, 1):
            tag = "[희귀]" if d["is_rare"] else "[일반]"
            print(f"  {i:2d}. {tag} {d['disease_name']:<40} LR={d['score']:.4f}")

        return ranking

    # ════════════════════════════════════════════════════════════════
    # ③ RAG 트리거 판단
    # ════════════════════════════════════════════════════════════════
    def step3_rag_trigger(self, ranking: list) -> bool:
        """희귀질환이거나 1·2등 점수 차이가 불확실하면 RAG 실행"""
        print("\n" + "─" * 50)
        print("③ TRIG — RAG 트리거 판단")
        print("─" * 50)

        if len(ranking) < 2:
            print("  ⚠️  랭킹 항목 부족 → RAG 실행")
            return True

        top1 = ranking[0]
        is_rare      = top1.get("is_rare", False)
        score_ratio  = top1["score"] / max(ranking[1]["score"], 1e-9)
        is_uncertain = score_ratio < SCORE_RATIO_THRESH

        trigger = is_rare or is_uncertain
        print(f"  1등: {top1['disease_name']}")
        print(f"  희귀질환: {is_rare} | 점수비율(1/2위): {score_ratio:.2f} | 불확실: {is_uncertain}")
        print(f"  → {'✅ RAG 실행 (심화 분석)' if trigger else '❌ 일반 리포트 출력'}")
        return trigger

    # ════════════════════════════════════════════════════════════════
    # ④ GEN — RAG + LLM 소견서 생성
    # ════════════════════════════════════════════════════════════════
    def step4_rag_generate(self, hpo_data: dict, ranking: list) -> str:
        """
        Evidence-bound RAG 리포트 생성
        - JSON 구조화 출력 + Markdown 사람용 출력
        - Context 외 정보 사용 금지 (hallucination 방지)
        - Clinical Safety Layer (금기/불확실성 명시)
        """
        print("\n" + "─" * 50)
        print("④ GEN — Evidence-bound RAG 리포트 생성")
        print("─" * 50)

        top_disease  = ranking[0]["disease_name"]
        top_genes    = ranking[0].get("genes", [])
        top_orpha    = ranking[0].get("orpha_code", "N/A")
        positive_hpo = hpo_data["positive_hpo"]
        negative_hpo = hpo_data["negative_hpo"]
        symptom_text = hpo_data["symptom_text"]
        lab_results  = hpo_data.get("lab_results", {})
        xray_detail  = hpo_data.get("xray_detail", {})

        # ── Soft data ①: PubCaseFinder (타임아웃 시 재시도 1회) ──
        print(f"  [PubCaseFinder] {len(positive_hpo)}개 HPO로 질환 매칭 중...")
        pcf_results = get_ranked_diseases(positive_hpo, top_k=PUBCASE_MAX)
        if not pcf_results:
            print("  ↩️ PubCaseFinder 재시도 중 (Top 3 HPO만)...")
            pcf_results = get_ranked_diseases(positive_hpo[:3], top_k=PUBCASE_MAX)
        case_context = format_pcf_for_llm(pcf_results, symptom_text=symptom_text)

        pcf_genes = []
        for r in pcf_results[:3]:
            pcf_genes.extend(r.get("genes", []))
        if pcf_genes and not top_genes:
            top_genes = list(dict.fromkeys(pcf_genes))

        # ── Soft data ②: PubMed ───────────────────────────────────
        pubmed_context = ""
        if self.pubmed is not None:
            print(f"  [PubMed] '{top_disease}' 최신 논문 검색 중...")
            papers = self.pubmed.get_top_papers(top_disease, top_k=3)
            if papers:
                lines = [f"【PubMed 최신 논문 ({len(papers)}편)】"]
                for p in papers:
                    lines.append(
                        f"- PMID:{p['pmid']} | {p['title']} ({p['pubdate']})\n"
                        f"  URL: {p['url']}\n"
                        f"  Abstract: {p.get('abstract','')[:300]}..."
                    )
                pubmed_context = "\n".join(lines)

        rag_context = "\n\n".join(filter(None, [case_context, pubmed_context]))

        # ── Soft data ③: ClinicalTrials.gov ──────────────────────
        trials_context = ""
        print(f"  [ClinicalTrials.gov] '{top_disease}' 모집 중 임상시험 검색 중...")
        trials = get_clinical_trials(top_disease, top_k=3)
        if trials:
            trials_context = format_trials_for_llm(trials, top_disease)

        rag_context = "\n\n".join(filter(None, [case_context, pubmed_context, trials_context]))

        ranking_text = "\n".join(
            f"  {i+1}. {d['disease_name']} "
            f"(LR={d['score']:.4f} | {d.get('orpha_code','N/A')} | 유병률={d['prevalence']})"
            for i, d in enumerate(ranking[:5])
        )
        gene_text = ", ".join(top_genes) if top_genes else "추가 유전자 패널 검토 필요"

        # Top 3 질환 상세 정보 (프롬프트에 명시적 주입)
        top3_detail = ""
        for i, d in enumerate(ranking[:3], 1):
            top3_detail += (
                f"\n[{i}위] {d['disease_name']}\n"
                f"  ORPHA 코드: {d.get('orpha_code', 'N/A')}\n"
                f"  LR 점수: {d['score']:.4f}\n"
                f"  유전자: {', '.join(d.get('genes', [])) or '정보 없음'}\n"
            )

        # ── X-ray Top10 스코어 텍스트 변환 ──────────────────────────
        xray_score_text = "\n".join(
            f"  {label}: {prob:.3f} → HPO={hpo}"
            for label, (prob, hpo) in sorted(
                xray_detail.items(), key=lambda x: x[1][0], reverse=True
            )
        ) if xray_detail else "X-ray 데이터 없음"

        # ── Lab raw 수치 텍스트 변환 ──────────────────────────────
        lab_raw_text = "\n".join(
            f"  {k}: {v}"
            for k, v in lab_results.items()
        ) if lab_results else "Lab 데이터 없음"

        # ── 개선된 프롬프트 (Evidence-bound + JSON + Safety) ──────
        prompt = f"""당신은 희귀 폐질환 진단 보조 AI 시스템입니다. 이 시스템은 반드시 근거 기반(Evidence-based)으로만 판단해야 합니다.

⚠️ 절대 규칙:
- 제공된 Context 밖의 정보는 절대 사용 금지
- PMID / ORPHA 코드 없는 정보는 "불확실"로 표시
- 추측 금지 / 환각 금지
- 반드시 의사 보조 시스템임을 명시

────────────────────────────
[INPUT DATA]

[임상 소견 (Symptom Raw)]
{symptom_text}

[환자 HPO]
Positive: {', '.join(positive_hpo) or '없음'}
Negative: {', '.join(negative_hpo) or '없음'}

[X-ray 분석 결과 (SooNet Top10 확률)]
{xray_score_text}

[혈액·폐기능 검사 수치 (Lab Raw)]
{lab_raw_text}

[질환 랭킹 — Orphanet LR 스코어링]
{ranking_text}

[Top 3 질환 상세]
{top3_detail}

[Context — PubCaseFinder + PubMed]
{rag_context}

────────────────────────────
[작업 목표]
1. Top 3 질환에 대해 근거 기반 분석
2. 유전자 검사 제안
3. 치료 가이드라인 요약
4. 최신 연구 기반 인사이트 제공
5. 다음 임상 단계 제안

────────────────────────────
[출력 형식 — JSON 먼저]

반드시 아래 JSON 구조로 먼저 출력:

```json
{{
  "diagnosis": [
    {{
      "disease": "",
      "orpha_code": "",
      "likelihood_reason": "",
      "positive_hpo_used": [],
      "negative_hpo_used": [],
      "evidence": [
        {{
          "type": "PMID | CASE | ORPHA",
          "source": "",
          "summary": ""
        }}
      ],
      "confidence": "HIGH | MEDIUM | LOW"
    }}
  ],
  "genetic_test": {{
    "recommended_genes": {json.dumps(top_genes)},
    "method": "",
    "reason": ""
  }},
  "treatment": {{
    "guideline": "",
    "contraindications": [],
    "evidence_level": ""
  }},
  "insight": {{
    "recent_findings": [],
    "clinical_notes": []
  }},
  "next_steps": [""],
  "uncertainty": [
    "데이터 부족",
    "근거 부족",
    "추가 검사 필요"
  ]
}}
```

────────────────────────────
[그 다음 Markdown 출력]

## ⚠️ AI 진단 보조 리포트 — 최종 진단은 반드시 담당 의사가 내려야 합니다

### 1. 질환 평가 (Top 3)
각 질환별: 질환명 + ORPHA 코드 + LR 점수 해석 + 근거(PMID 인용, 없으면 "근거 없음" 명시)

### 2. 유전자 검사 권고
관련 유전자: {gene_text}
검사 방법 및 권고 이유 (ORPHA {top_orpha} 기반)

### 3. 치료 가이드라인
1순위 질환({top_disease}) 치료 원칙 + 금기사항

### 4. 최신 동향
Context에 있는 PMID 인용 논문만 사용. 없으면 "최신 논문 없음" 명시.

### 5. 다음 단계
즉시 시행 권고 검사 + 의뢰 과 + 재평가 시점

────────────────────────────
[중요 제약 조건]
- Evidence에 없는 내용 절대 생성 금지
- PMID 없는 연구 언급 금지
- 모르면 반드시 "정보 없음" 표시
- Negative HPO 반드시 감별진단에 반영
- 과도한 확신 표현 금지 ("가능성 있음" 사용)

[Self-Check]
출력 생성 후 아래 확인:
- 모든 근거가 context에 있는가?
- PMID가 실제 context에 존재하는가?
- 추측이 포함되어 있는가?
문제 있으면 수정 후 출력."""

        # ── Bedrock API 호출 (temperature=0.2, top_p=0.9) ─────────
        print(f"  [Claude] {REPORT_MODEL} 소견서 생성 중...")
        try:
            resp = self.bedrock.invoke_model(
                modelId=REPORT_MODEL,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": REPORT_MAX_TOKENS,
                    "temperature": 0.2,
                    "top_p": 0.9,
                }),
            )
            raw_text = json.loads(resp["body"].read())["content"][0]["text"]

            # JSON 파싱 시도 (구조화 데이터 추출)
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if json_match:
                try:
                    structured = json.loads(json_match.group(1))
                    print(f"  ✅ JSON 파싱 성공: diagnosis={len(structured.get('diagnosis',[]))}개, "
                          f"uncertainty={len(structured.get('uncertainty',[]))}개")
                except json.JSONDecodeError:
                    print("  ⚠️ JSON 파싱 실패 — 원문 반환")

            return raw_text

        except ClientError as e:
            err = e.response["Error"]
            print(f"❌ Bedrock 오류: {err['Code']} — {err['Message']}")
            return f"[소견서 생성 실패] {err['Message']}"

    # ════════════════════════════════════════════════════════════════
    # ⑤ OUT — 메인 실행
    # ════════════════════════════════════════════════════════════════
    def run(
        self,
        xray_path: str,
        symptom_text: str,
        lab_results: dict,
    ) -> str:
        """
        전체 파이프라인 실행

        Parameters
        ----------
        xray_path    : str   흉부 X-ray 이미지 경로
        symptom_text : str   임상 소견 자유 텍스트
        lab_results  : dict  혈액·폐기능 검사 수치

        Returns
        -------
        str  마크다운 진단 보조 리포트
        """
        print("\n" + "=" * 60)
        print("🏥  Rare-Link AI 진단 보조 파이프라인 시작")
        print("=" * 60)

        hpo_data = self.step1_get_hpo(xray_path, symptom_text, lab_results)
        ranking  = self.step2_score(hpo_data)
        use_rag  = self.step3_rag_trigger(ranking)

        if use_rag:
            report = self.step4_rag_generate(hpo_data, ranking)
        else:
            top    = ranking[0]
            report = (
                f"## 진단 보조 결과 (일반)\n\n"
                f"가장 유력한 질환: **{top['disease_name']}** (LR={top['score']:.4f})\n\n"
                f"> 희귀질환 임계값 미달 — 추가 RAG 분석 생략\n"
            )

        print("\n" + "=" * 60)
        print("✅ 파이프라인 완료")
        print("=" * 60)
        return report


# ── 실행 테스트 ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = RareLinkPipeline(
        vision_model_path="model/chexnet_unet_crop_best.pth",
    )

    report = pipeline.run(
        xray_path="test_xray.jpg",
        symptom_text=(
            "40세 여성. 3주째 지속되는 호흡곤란과 우측 흉통을 호소합니다. "
            "기침은 없으며 발열도 없습니다. 최근 체중 감소가 있었습니다."
        ),
        lab_results={
            "WBC":   12.5,
            "HGB":    9.8,
            "LDH":   310,
            "CRP":    7.2,
            "SpO2":  92.0,
            "FEV1":  68.0,
        },
    )

    print("\n" + "=" * 60)
    print("📋 최종 진단 보조 리포트")
    print("=" * 60)
    print(report)
