# ragas_eval.py
# RAG 시스템 검증 — 2차 RAG 회의록 섹션 3 기반
#
# 회의에서 정의한 3단계 검증:
#   1단계: 정량 자동 평가 (RAGAS — Faithfulness, Answer Relevancy 등)
#   2단계: 의료 도메인 특화 수동 평가 (PMID 환각 체크)
#   3단계: 프롬프트 A/B 테스트 (허태웅·배기태·권미라 비교)
#
# 설치 필요:
#   pip install ragas datasets requests
#   (ragas가 없어도 PMID 체크와 A/B 테스트는 동작)

import json
import re

import requests

# ── RAGAS 임포트 (없어도 나머지 기능 동작) ────────────────────────────
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False

PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


# ════════════════════════════════════════════════════════════════════
# 1단계: RAGAS 자동 평가
# ════════════════════════════════════════════════════════════════════
def evaluate_with_ragas(
    questions: list,
    answers: list,
    contexts: list,
    ground_truths: list = None,
) -> dict:
    """
    RAGAS로 RAG 출력 품질 자동 평가

    RAGAS란?
      RAG 시스템 전용 평가 라이브러리.
      "모델이 지어낸 말(환각)이 있는가", "답변이 질문과 관련 있는가" 등을
      자동으로 수치화함.

    측정 지표:
      - faithfulness     : 답변이 컨텍스트에 근거하는가 (환각 없음 = 1.0)
      - answer_relevancy : 답변이 질문과 관련 있는가 (높을수록 좋음)
      - context_precision: 검색된 컨텍스트가 정답과 관련 있는가 (ground_truth 필요)
      - context_recall   : 필요한 정보가 컨텍스트에 포함됐는가  (ground_truth 필요)

    Parameters
    ----------
    questions     : list[str]
        입력 질문 목록
        예) ["40세 여성 흉통+호흡곤란, LAM 가능성은?"]

    answers       : list[str]
        모델이 생성한 소견서 목록 (rag_pipeline.run() 결과)

    contexts      : list[list[str]]
        각 질문에 대해 RAG가 검색한 컨텍스트 목록의 목록
        예) [["케이스1 내용", "케이스2 내용"], [...]]

    ground_truths : list[str] or None
        정답 텍스트 목록 (없으면 precision/recall 측정 생략)

    Returns
    -------
    dict
        {"faithfulness": 0.85, "answer_relevancy": 0.91, ...}
        점수 범위: 0.0 ~ 1.0 (높을수록 좋음)

    Example
    -------
    >>> results = evaluate_with_ragas(
    ...     questions=["LAM 진단 근거?"],
    ...     answers=[report_text],
    ...     contexts=[["케이스리포트1", "케이스리포트2"]],
    ... )
    >>> print(results)
    {'faithfulness': 0.87, 'answer_relevancy': 0.93}
    """
    if not _RAGAS_AVAILABLE:
        print("❌ ragas 미설치.")
        print("   설치 방법: pip install ragas datasets")
        return {}

    data = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    }
    if ground_truths:
        data["ground_truth"] = ground_truths

    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy]
    if ground_truths:
        metrics += [context_precision, context_recall]

    print(f"[RAGAS] {len(questions)}개 샘플 평가 중...")
    result = evaluate(dataset, metrics=metrics)
    scores = dict(result)

    print("\n=== RAGAS 평가 결과 ===")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {metric:<20}: {score:.3f}  |{bar:<20}|")

    return scores


# ════════════════════════════════════════════════════════════════════
# 2단계: PMID 환각 체크 — 가장 중요한 의료 AI 검증
# ════════════════════════════════════════════════════════════════════
def verify_pmids(text: str, verbose: bool = True) -> dict:
    """
    소견서 텍스트에서 PMID를 추출하고 PubMed에서 실제 존재 여부 확인

    왜 중요한가?
      LLM은 "PMID: 12345678" 같은 실제로 없는 논문 ID를 지어낼 수 있음.
      의료 AI에서 없는 논문을 인용하면 신뢰성이 무너지므로 반드시 체크.

    Parameters
    ----------
    text    : str   — 모델이 생성한 소견서 텍스트
    verbose : bool  — True면 결과 상세 출력

    Returns
    -------
    dict
        {
            "total":      5,                    # 발견된 PMID 수
            "valid":      4,                    # PubMed에 실제 존재하는 수
            "invalid":    ["99999999"],          # 가짜 PMID 목록
            "valid_list": ["32386464", ...],    # 유효 PMID 목록
            "rate":       0.8,                  # 유효율 (valid/total)
        }

    Example
    -------
    >>> report = "...PMID: 32386464 에 따르면... PMID: 99999999..."
    >>> result = verify_pmids(report)
    >>> print(result["invalid"])  # ['99999999']
    """
    # "PMID: 12345678" 또는 "PMID 12345678" 패턴 추출 (7~9자리)
    pmids = list(set(re.findall(r"PMID[:\s]*(\d{7,9})", text)))

    if not pmids:
        if verbose:
            print("  ℹ️  소견서에 PMID 없음 (인용 없음)")
        return {"total": 0, "valid": 0, "invalid": [], "valid_list": [], "rate": None}

    if verbose:
        print(f"  [PMID 체크] 발견된 PMID {len(pmids)}개: {pmids}")

    valid, invalid = [], []
    for pmid in pmids:
        try:
            resp = requests.get(
                PUBMED_ESUMMARY,
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
                timeout=5,
            )
            data = resp.json()
            # PubMed API: 실제 존재하면 result[pmid]["uid"] 에 pmid가 들어옴
            entry = data.get("result", {}).get(pmid, {})
            uid = entry.get("uid", "")
            if uid == pmid and "error" not in entry:
                valid.append(pmid)
                if verbose:
                    title = data["result"][pmid].get("title", "")[:60]
                    print(f"    ✅ {pmid} — {title}...")
            else:
                invalid.append(pmid)
                if verbose:
                    print(f"    ❌ {pmid} — PubMed에 없음 (환각)")
        except Exception as e:
            invalid.append(pmid)
            if verbose:
                print(f"    ⚠️ {pmid} — 확인 실패: {e}")

    rate = len(valid) / len(pmids) if pmids else None

    if verbose:
        print(f"\n  결과: 유효 {len(valid)}/{len(pmids)} (유효율 {rate:.0%})")
        if invalid:
            print(f"  ❌ 가짜 PMID: {invalid}")

    return {
        "total":      len(pmids),
        "valid":      len(valid),
        "invalid":    invalid,
        "valid_list": valid,
        "rate":       rate,
    }


# ════════════════════════════════════════════════════════════════════
# 3단계: 프롬프트 A/B 테스트
# ════════════════════════════════════════════════════════════════════
def run_ab_comparison(test_cases: list, pipeline_a, pipeline_b) -> list:
    """
    동일 입력 케이스에 두 파이프라인을 각각 실행하여 비교

    회의 확정:
      허태웅·배기태·권미라 3인이 각자 구현한 프롬프트를
      동일 입력 케이스 5~10개에 적용하여 비교

    비교 기준:
      - PMID 유효율 (환각 없는 인용)
      - 자동 RAGAS 점수 (faithfulness)
      - 수동 검토 (팀 내 임상 정확성 평가)

    Parameters
    ----------
    test_cases : list[dict]
        [{"xray_path": "...", "symptom_text": "...", "lab_results": {...}}]
        최소 5개 권장 (일반/기타/희귀 질환 유형별)

    pipeline_a : RareLinkPipeline  — 첫 번째 프롬프트 버전
    pipeline_b : RareLinkPipeline  — 두 번째 프롬프트 버전

    Returns
    -------
    list[dict]
        [{"case_idx": 1, "output_a": "...", "output_b": "...",
          "pmid_a": {...}, "pmid_b": {...}}, ...]
    """
    results = []
    print(f"\n=== A/B 테스트 시작 ({len(test_cases)}개 케이스) ===\n")

    for i, case in enumerate(test_cases, 1):
        print(f"--- 케이스 {i}/{len(test_cases)} ---")

        print("[Pipeline A] 실행 중...")
        out_a = pipeline_a.run(**case)

        print("[Pipeline B] 실행 중...")
        out_b = pipeline_b.run(**case)

        print("\n[PMID 환각 체크]")
        print("  Pipeline A:")
        pmid_a = verify_pmids(out_a, verbose=True)
        print("  Pipeline B:")
        pmid_b = verify_pmids(out_b, verbose=True)

        results.append({
            "case_idx": i,
            "output_a": out_a,
            "output_b": out_b,
            "pmid_a":   pmid_a,
            "pmid_b":   pmid_b,
        })

    # ── 요약 출력 ─────────────────────────────────────────────────
    print("\n=== A/B 테스트 요약 ===")
    print(f"{'케이스':<6} {'A PMID 유효율':<16} {'B PMID 유효율'}")
    print("-" * 40)
    for r in results:
        a_rate = f"{r['pmid_a']['rate']:.0%}" if r["pmid_a"]["rate"] is not None else "N/A"
        b_rate = f"{r['pmid_b']['rate']:.0%}" if r["pmid_b"]["rate"] is not None else "N/A"
        print(f"  {r['case_idx']:<4} {a_rate:<16} {b_rate}")

    # 결과를 JSON 파일로 저장
    out_path = "ab_test_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {out_path}")

    return results


# ════════════════════════════════════════════════════════════════════
# 1-B단계: Bedrock 직접 평가 (ragas 라이브러리 대체)
# ragas 0.4.x는 InstructorLLM만 지원, 0.2.x는 Python 3.14 비호환
# → Claude Haiku로 동일 지표를 직접 구현
# ════════════════════════════════════════════════════════════════════
def evaluate_with_bedrock(
    question: str,
    answer: str,
    contexts: list,
    aws_region: str = "ap-northeast-2",
) -> dict:
    """
    Bedrock Claude Haiku로 faithfulness·answer_relevancy 직접 평가

    ragas 라이브러리 없이 동일 지표 측정.
    Python 버전·의존성 제약 없이 동작.

    Parameters
    ----------
    question : str        임상 질문
    answer   : str        모델 생성 소견서
    contexts : list[str]  RAG 검색 컨텍스트 목록

    Returns
    -------
    dict
        {
            "faithfulness":      0.0~1.0,  # 컨텍스트 근거 충실도
            "answer_relevancy":  0.0~1.0,  # 질문 관련성
            "faithfulness_reason":    str,
            "answer_relevancy_reason": str,
        }
    """
    import boto3, json as _json
    client = boto3.client("bedrock-runtime", region_name=aws_region)
    MODEL  = "anthropic.claude-3-haiku-20240307-v1:0"

    ctx_block = "\n---\n".join(contexts)

    # ── Faithfulness 평가 ──────────────────────────────────────────
    faith_prompt = f"""당신은 의료 AI 출력물의 품질을 평가하는 전문가입니다.

[컨텍스트]
{ctx_block}

[AI 소견서]
{answer}

위 AI 소견서의 모든 주요 주장이 [컨텍스트]에 근거하는지 평가하세요.
컨텍스트에 없는 정보를 AI가 지어냈으면 환각(hallucination)입니다.

다음 JSON 형식으로만 답하세요:
{{"score": 0.0~1.0, "reason": "한 줄 이유"}}
score 기준: 1.0=완전 근거 있음, 0.5=일부 환각, 0.0=대부분 환각"""

    # ── Answer Relevancy 평가 ─────────────────────────────────────
    relevancy_prompt = f"""당신은 의료 AI 출력물의 품질을 평가하는 전문가입니다.

[질문]
{question}

[AI 소견서]
{answer}

위 AI 소견서가 [질문]에 얼마나 관련 있고 직접적으로 답변하는지 평가하세요.

다음 JSON 형식으로만 답하세요:
{{"score": 0.0~1.0, "reason": "한 줄 이유"}}
score 기준: 1.0=완전히 관련·직접 답변, 0.5=부분 관련, 0.0=무관"""

    scores = {}
    for metric, prompt in [("faithfulness", faith_prompt), ("answer_relevancy", relevancy_prompt)]:
        try:
            resp = client.invoke_model(
                modelId=MODEL,
                body=_json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                }),
            )
            raw = _json.loads(resp["body"].read())["content"][0]["text"].strip()
            parsed = _json.loads(raw)
            scores[metric]             = round(float(parsed["score"]), 3)
            scores[f"{metric}_reason"] = parsed.get("reason", "")
        except Exception as e:
            print(f"  ⚠️ {metric} 평가 실패: {e}")
            scores[metric]             = -1.0
            scores[f"{metric}_reason"] = str(e)

    print("\n=== Bedrock 품질 평가 결과 ===")
    for metric in ("faithfulness", "answer_relevancy"):
        score = scores[metric]
        bar   = "█" * int(score * 20) if score >= 0 else "ERROR"
        print(f"  {metric:<22}: {score:.3f}  |{bar:<20}|")
        print(f"    이유: {scores.get(metric+'_reason','')}")

    return scores


# ════════════════════════════════════════════════════════════════════
# 편의 함수: 단일 소견서 빠른 검증
# ════════════════════════════════════════════════════════════════════
def quick_check(report_text: str) -> None:
    """
    단일 소견서에 대해 PMID 환각 체크만 빠르게 실행

    파이프라인 완성 전 개발 중 빠른 테스트용.

    Example
    -------
    >>> from ragas_eval import quick_check
    >>> quick_check(my_report)
    """
    print("\n" + "=" * 45)
    print("빠른 환각 체크 (PMID 검증)")
    print("=" * 45)
    verify_pmids(report_text, verbose=True)


# ════════════════════════════════════════════════════════════════════
# 단독 실행 예시
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=== ragas_eval.py 단위 테스트 ===\n")

    # PMID 환각 체크 테스트
    sample_report = """
    ## AI 진단 보조 리포트

    1위 질환: Lymphangioleiomyomatosis (LAM) [ORPHA:723]
    PMID: 32386464 에 따르면 TSC2 유전자 변이가 LAM의 주요 원인입니다.
    PMID: 99999999 (이것은 테스트용 가짜 PMID입니다)

    유전자 검사 권고: TSC1, TSC2 시퀀싱
    """

    print("--- PMID 환각 체크 ---")
    result = verify_pmids(sample_report)
    print(f"\n최종: 유효 {result['valid']}/{result['total']}, 가짜: {result['invalid']}")

    print("\n--- RAGAS 설치 확인 ---")
    if _RAGAS_AVAILABLE:
        print("✅ ragas 설치됨. evaluate_with_ragas() 사용 가능")
    else:
        print("⚠️ ragas 미설치. pip install ragas datasets 로 설치하세요.")
