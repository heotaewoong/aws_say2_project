"""
환자별 파이프라인 출력 리포트 생성기
===================================
rag_pipeline_mimic_results.json 에 저장된 10명 환자의
실제 파이프라인 출력을 사람이 보기 좋은 MD 파일로 변환.

회의록 §1.4 구조에 맞춰 출력:
  - 주요 질환 랭킹 설명
  - ① 근거 기반 질환 제시
  - ② 유전자 검사 권고
  - ③ 치료 가이드라인
  - ④ 최신 치료 동향
  - 진단 정확도 (실제 vs AI Top 3)

실행:
    python rag/valid/generate_patient_reports.py

결과:
    rag/valid/patient_reports/patient_<sid>.md  (10명 개별)
    rag/valid/patient_reports/INDEX.md          (종합 인덱스)
"""
import os
import sys
import json
import re
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

OUT_DIR = "rag/valid/patient_reports"
os.makedirs(OUT_DIR, exist_ok=True)


def format_patient_report(r: dict, idx: int, total: int) -> str:
    """단일 환자의 실측 출력을 MD로 포맷 (신 포맷: general_diagnosis / rare_diagnosis)"""
    sid = r['subject_id']
    actual = r['actual_diagnosis']
    ai_top3 = r.get('ai_top3_names', [])
    hit1 = r.get('hit@1', False)
    hit3 = r.get('hit@3', False)
    matched_rank = r.get('matched_rank')

    rep  = r.get('full_report', {})
    rec  = rep.get('recommendation', {}) or {}
    cn   = rep.get('clinical_notes', {}) or {}
    gen  = rep.get('general_diagnosis', []) or []
    rare = rep.get('rare_diagnosis', []) or []

    summary      = cn.get('summary', '')
    diff_note    = cn.get('differential_note', '')
    rag_evidence = cn.get('rag_evidence', '')
    case_comp    = cn.get('case_comparison', '')
    disclaimer   = cn.get('disclaimer', '')

    immediate  = rec.get('immediate_workup', [])
    referral   = rec.get('specialist_referral', [])
    additional = rec.get('additional_lab', [])

    hit_mark = "✅ Hit@1" if hit1 else ("⚠️ Hit@3" if hit3 else "❌ Miss")
    rank_str = f"Top {matched_rank}" if matched_rank else "매칭 없음"

    lines = []
    lines.append(f"# 환자 {sid} — 파이프라인 실제 출력 ({idx}/{total})")
    lines.append("")
    lines.append(f"> **MIMIC-IV 실환자 검증 결과** | {datetime.now().strftime('%Y-%m-%d')} 실측")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 0. 진단 정확도 요약")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---|")
    lines.append(f"| **MIMIC 실제 퇴원 진단** | {actual} |")
    lines.append(f"| **AI Top 1** | {ai_top3[0] if len(ai_top3) > 0 else '—'} |")
    lines.append(f"| **AI Top 2** | {ai_top3[1] if len(ai_top3) > 1 else '—'} |")
    lines.append(f"| **AI Top 3** | {ai_top3[2] if len(ai_top3) > 2 else '—'} |")
    lines.append(f"| **매칭 결과** | {hit_mark} ({rank_str}) |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 1. 환자 종합 요약 ──────────────────────────────────────────
    lines.append("## 1. 환자 종합 요약")
    lines.append("")
    lines.append("```")
    lines.append(summary or "(없음)")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── 2. 일반 폐질환 Top 3 ──────────────────────────────────────
    lines.append("## 2. 일반 폐질환 Top 3")
    lines.append("")
    if gen:
        for d in gen[:3]:
            rank = d.get('rank', '—')
            dname = d.get('disease_name', '—')
            score = d.get('score', 0)
            icd = ', '.join(d.get('icd10', []))
            reasoning = d.get('reasoning', '')
            tg = d.get('treatment_guideline', '')
            trend = d.get('recent_trend', '')
            lines.append(f"### {rank}. {dname}  `score={score:.3f}`  `ICD-10: {icd}`")
            lines.append("")
            lines.append(f"**진단 근거**: {reasoning}")
            lines.append("")
            lines.append(f"**치료 가이드라인**: {tg}")
            lines.append("")
            lines.append(f"**최신 동향 (PubMed)**: {trend or '(없음)'}")
            lines.append("")
    else:
        lines.append("(데이터 없음)")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 3. 희귀 폐질환 Top 3 ──────────────────────────────────────
    lines.append("## 3. 희귀 폐질환 Top 3")
    lines.append("")
    if rare:
        for d in rare[:3]:
            rank = d.get('rank', '—')
            dname = d.get('disease_name', '—')
            orpha = d.get('orpha_code', '—')
            lr = d.get('lr_score', 0)
            evidence = d.get('evidence', '')
            genetic = d.get('genetic_test', [])
            tg = d.get('treatment_guideline', '')
            trend = d.get('recent_trend', '')
            epi = d.get('epidemiology', '')
            lines.append(f"### {rank}. {dname}  `{orpha}`  `LR={lr:.1f}`")
            lines.append("")
            lines.append(f"**근거**: {evidence}")
            lines.append("")
            lines.append(f"**역학 (Orphanet)**: {epi or '(없음)'}")
            lines.append("")
            if genetic:
                lines.append(f"**유전자 검사 권고**: {', '.join(genetic)}")
                lines.append("")
            lines.append(f"**치료 가이드라인**: {tg}")
            lines.append("")
            lines.append(f"**최신 동향 (PubMed)**: {trend or '(없음)'}")
            lines.append("")
    else:
        lines.append("(희귀질환 후보 없음)")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 4. 감별진단 & RAG 근거 ────────────────────────────────────
    lines.append("## 4. 감별진단 & RAG 근거")
    lines.append("")
    lines.append("### 감별진단 (differential_note)")
    lines.append("```")
    lines.append(diff_note or "(없음)")
    lines.append("```")
    lines.append("")
    lines.append("### RAG 수집 근거 (rag_evidence)")
    lines.append("```")
    lines.append(rag_evidence or "(없음)")
    lines.append("```")
    lines.append("")
    lines.append("### PubMed 사례 비교 (case_comparison)")
    lines.append("```")
    lines.append(case_comp or "(없음)")
    lines.append("```")
    lines.append("")
    pmids = re.findall(r'PMID\s*[:：]?\s*(\d+)', (case_comp or '') + (rag_evidence or ''))
    if pmids:
        lines.append("**인용된 PMID**:")
        for p in sorted(set(pmids)):
            lines.append(f"- [PMID:{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)")
        lines.append("")
    lines.append("---")
    lines.append("")

    # ── 5. 즉시 권고사항 ──────────────────────────────────────────
    lines.append("## 5. 즉시 권고사항")
    lines.append("")
    lines.append("### 즉시 시행할 검사 (immediate_workup)")
    for w in immediate or ["(없음)"]:
        lines.append(f"- {w}")
    lines.append("")
    lines.append("### 협진 권고 (specialist_referral)")
    for rv in referral or ["(없음)"]:
        lines.append(f"- {rv}")
    lines.append("")
    lines.append("### 추가 검사 (additional_lab)")
    for a in additional or ["(없음)"]:
        lines.append(f"- {a}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Disclaimer")
    lines.append("")
    lines.append(f"> {disclaimer or 'AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다.'}")
    lines.append("")
    return "\n".join(lines)


def format_index(data: dict) -> str:
    """모든 환자 요약 인덱스"""
    lines = []
    lines.append("# 환자별 파이프라인 출력 — 종합 인덱스")
    lines.append("")
    lines.append(f"> 생성일: {datetime.now().strftime('%Y-%m-%d')} | 환자 수: {data['n_patients']}명")
    lines.append(f"> Hit@1: {data['hit_at_1']}/{data['n_patients']} | Hit@3: {data['hit_at_3']}/{data['n_patients']}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 환자별 요약 테이블")
    lines.append("")
    lines.append("| # | subject_id | 실제 진단 | AI Top 1 | 결과 | 리포트 |")
    lines.append("|---|-----------|-----------|----------|:----:|--------|")

    for i, r in enumerate(data['results'], 1):
        sid = r['subject_id']
        actual = r['actual_diagnosis'].replace('|', '\\|')[:40]
        top1 = (r.get('ai_top3_names', ['—'])[0] if r.get('ai_top3_names') else '—').replace('|', '\\|')
        hit1 = r.get('hit@1', False)
        hit3 = r.get('hit@3', False)
        mark = "✅ Hit@1" if hit1 else ("⚠️ Hit@3" if hit3 else "❌")
        link = f"[patient_{sid}.md](patient_{sid}.md)"
        lines.append(f"| {i} | {sid} | {actual} | {top1} | {mark} | {link} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 주요 사례 하이라이트")
    lines.append("")

    # Hit 성공 사례
    lines.append("### ✅ 성공 사례 (Hit@1 또는 Hit@3)")
    for r in data['results']:
        if r.get('hit@3') or r.get('hit@1'):
            sid = r['subject_id']
            actual = r['actual_diagnosis'][:60]
            top3 = r.get('ai_top3_names', [])
            hit1 = r.get('hit@1', False)
            mark = "✅ Top 1" if hit1 else "⚠️ Top 3"
            lines.append(f"- **환자 {sid}** {mark} — 실제: *{actual}* → AI: {top3}")
    lines.append("")

    # 실패 사례
    lines.append("### ❌ 미매칭 사례 (원인 분석)")
    for r in data['results']:
        if not (r.get('hit@3') or r.get('hit@1')):
            sid = r['subject_id']
            actual = r['actual_diagnosis'][:60]
            top3 = r.get('ai_top3_names', [])
            lines.append(f"- **환자 {sid}** — 실제: *{actual}* → AI: {top3}")
    lines.append("")
    lines.append("**공통 원인**: 희귀질환 LR이 매우 높아 일반 폐질환(COPD, ILD, PE)을 Top에서 밀어냄 + Lab 가상값")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 파일 구조")
    lines.append("")
    lines.append("```")
    lines.append("patient_reports/")
    lines.append("├── INDEX.md             ← 이 파일")
    for r in data['results']:
        lines.append(f"├── patient_{r['subject_id']}.md")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main():
    results_path = "rag/valid/rag_pipeline_mimic_results.json"
    if not os.path.exists(results_path):
        print(f"❌ {results_path} 없음")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    total = len(data['results'])
    print(f"📝 환자별 리포트 생성 중 ({total}명)...")

    for i, r in enumerate(data['results'], 1):
        sid = r['subject_id']
        content = format_patient_report(r, i, total)
        out_path = os.path.join(OUT_DIR, f"patient_{sid}.md")
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ [{i}/{total}] {out_path}")

    # 인덱스 생성
    index_content = format_index(data)
    index_path = os.path.join(OUT_DIR, "INDEX.md")
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    print(f"\n📄 인덱스 저장: {index_path}")


if __name__ == "__main__":
    main()
