"""
RAG 출력 구조 검증 — 2차 회의록(2026-04-21) §1.4 기준
=====================================================
회의록에서 확정된 출력 4가지가 파이프라인 JSON에 모두 존재하는지 측정

회의록 §1.4 확정 사항:
  주요 질환 → 랭킹 형태 출력 (Top 10)
  희귀질환 → 유전자 리스트 + 아래 4가지:
    ① 근거 기반 질환 제시 (케이스리포트 + Orphadata)
    ② 유전자 검사 권고
    ③ 치료 가이드라인
    ④ 최신 치료 동향

사용:
    python rag/valid/eval_output_structure.py
"""
import os
import sys
import json
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def check_structure(report: dict) -> dict:
    """회의록 §1.4 요구사항별 출력 존재 여부 체크"""
    rec = report.get('recommendation', {}) or {}
    cn  = report.get('clinical_notes', {}) or {}

    # ① 근거 기반 질환 제시 — rag_evidence/case_comparison/epidemiology_note
    rag_evidence = (cn.get('rag_evidence') or '').strip()
    case_comp    = (cn.get('case_comparison') or '').strip()
    epi_note     = (cn.get('epidemiology_note') or '').strip()

    has_casereport = bool(
        re.search(r'PMID\s*[:：]?\s*\d+', rag_evidence + case_comp, re.IGNORECASE)
    )
    has_orphadata = bool(
        re.search(r'ORPHA[:：]?\s*\d+|orphanet|유병률|prevalence|발병연령',
                  rag_evidence + epi_note, re.IGNORECASE)
    )

    # ② 유전자 검사 권고
    genetic_test = rec.get('genetic_test', [])
    has_genetic = bool(genetic_test and len(genetic_test) > 0)

    # ③ 치료 가이드라인 — [질환명] prefix가 있는 항목 있어야
    treatments = rec.get('treatment_guideline', [])
    disease_tagged = [t for t in treatments if re.match(r'\s*\[', str(t))]
    has_treatment = bool(disease_tagged)

    # ④ 최신 치료 동향 — PubMed case_comparison 에 PMID 있으면 "최신 동향" 반영
    has_recent_trend = bool(
        re.search(r'PMID[:：]?\s*\d+|202[0-9]|최신|recent',
                  case_comp + rag_evidence, re.IGNORECASE)
    )

    # 부가: Top-1 reasoning (주요 질환 랭킹 설명)
    has_ranking_reason = bool((cn.get('top1_reasoning') or '').strip())
    has_differential  = bool((cn.get('differential_note') or '').strip())

    return {
        "① 근거 기반 질환 제시": {
            "case_report_cited": has_casereport,
            "orphadata_cited":   has_orphadata,
            "passed":            has_casereport or has_orphadata,
        },
        "② 유전자 검사 권고": {
            "genes_count": len(genetic_test),
            "sample":      genetic_test[:3],
            "passed":      has_genetic,
        },
        "③ 치료 가이드라인": {
            "total_items":         len(treatments),
            "disease_tagged_items": len(disease_tagged),
            "sample":              disease_tagged[:3],
            "passed":              has_treatment,
        },
        "④ 최신 치료 동향": {
            "pubmed_or_recent": has_recent_trend,
            "passed":           has_recent_trend,
        },
        "주요 질환 랭킹 설명": {
            "top1_reasoning_present":   has_ranking_reason,
            "differential_note_present": has_differential,
            "passed":                   has_ranking_reason and has_differential,
        },
    }


def main():
    # MIMIC 검증 결과 로드
    results_path = "rag/valid/rag_pipeline_mimic_results.json"
    if not os.path.exists(results_path):
        print(f"❌ {results_path} 없음 — 먼저 eval_rag_pipeline_mimic.py 실행 필요")
        return

    with open(results_path) as f:
        data = json.load(f)

    print("=" * 70)
    print("RAG 출력 구조 검증 — 2차 회의록(2026-04-21) §1.4 기준")
    print(f"대상: MIMIC-IV 폐질환 환자 {data['n_patients']}명")
    print("=" * 70)

    # 각 환자별 체크
    per_patient = []
    for r in data['results']:
        rep = r.get('full_report', {})
        if not isinstance(rep, dict) or 'recommendation' not in rep:
            continue
        check = check_structure(rep)
        per_patient.append({
            "subject_id": r['subject_id'],
            "checks":     check,
        })

    # 요구사항별 통과율 집계
    print(f"\n[환자 {len(per_patient)}명 전원 기준 — 요구사항별 통과율]\n")

    requirements = [
        "① 근거 기반 질환 제시",
        "② 유전자 검사 권고",
        "③ 치료 가이드라인",
        "④ 최신 치료 동향",
        "주요 질환 랭킹 설명",
    ]

    summary = {}
    for req in requirements:
        passed = sum(1 for p in per_patient if p['checks'][req]['passed'])
        total = len(per_patient)
        rate = passed / total if total else 0
        mark = "✅" if rate >= 0.9 else ("⚠️ " if rate >= 0.5 else "❌")
        print(f"  {mark} {req:<30}  {passed}/{total}  ({rate*100:.0f}%)")
        summary[req] = {"passed": passed, "total": total, "rate": round(rate, 4)}

    # 개별 환자 상세
    print("\n" + "=" * 70)
    print("개별 환자 상세")
    print("=" * 70)
    for p in per_patient[:3]:  # 처음 3명만 상세 출력
        sid = p['subject_id']
        print(f"\n환자 {sid}")
        for req in requirements:
            c = p['checks'][req]
            mark = "✅" if c['passed'] else "❌"
            print(f"  {mark} {req}")
            if req == "② 유전자 검사 권고" and c.get('sample'):
                for s in c['sample']:
                    print(f"       - {s}")
            elif req == "③ 치료 가이드라인" and c.get('sample'):
                for s in c['sample']:
                    print(f"       - {s}")

    # 결과 저장
    out_json = "rag/valid/output_structure_check.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_patients": len(per_patient),
            "summary": summary,
            "per_patient": per_patient,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n📄 결과 저장: {out_json}")


if __name__ == "__main__":
    main()
