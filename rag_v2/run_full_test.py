"""
전체 5단계 파이프라인 실제 실행 테스트
X-ray 이미지 + 증상 + Lab → 최종 소견서 (일반 Top 3 + 희귀 Top 3)
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(__file__))

# .env 로드
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    with open(_env) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from rag_pipeline import RareLinkPipeline

print("=" * 60)
print("🏥 Rare-Link AI — 전체 파이프라인 실행 테스트")
print("=" * 60)

pipeline = RareLinkPipeline(
    vision_model_path="model/chexnet_unet_crop_best.pth",
)

sample_patient = {
    "name":            "익명",
    "age":             40,
    "sex":             "F",
    "visit_date":      "2026-05-06",
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
    xray_path     = "cam_results/흉막삼출_Pleural_Effusion.png",
    symptom_text  = "40세 여성. 3주째 지속되는 호흡곤란과 우측 흉통을 호소합니다. 최근 체중 감소가 있었습니다.",
    negative_text = "기침은 없으며 발열도 없습니다.",
    lab_results   = sample_lab,
)

print()
print("=" * 60)
print("최종 출력 구조")
print("=" * 60)
print("Top-level keys:", list(report.keys()))
print()

if "general_diagnosis" in report:
    print("[일반 폐질환 Top 3]")
    for d in report["general_diagnosis"]:
        print(f"  {d['rank']}. {d['disease_name']}")
        treat = str(d.get("treatment_guideline", ""))[:80]
        print(f"     치료: {treat}")
    print()

if "rare_diagnosis" in report:
    print("[희귀 폐질환 Top 3]")
    for d in report["rare_diagnosis"]:
        orpha = d.get("orpha_code", "N/A")
        print(f"  {d['rank']}. {d['disease_name']} (ORPHA:{orpha})")
        print(f"     유전자: {d.get('genetic_test', [])}")
        treat = str(d.get("treatment_guideline", ""))[:80]
        print(f"     치료: {treat}")
        trend = str(d.get("recent_trend", ""))[:80]
        print(f"     최신동향: {trend}")
    print()

if "recommendation" in report:
    rec = report["recommendation"]
    print("[권고사항]")
    print("  즉시검사:", rec.get("immediate_workup", [])[:3])
    print("  협진:", rec.get("specialist_referral", [])[:2])

if "clinical_notes" in report:
    cn = report["clinical_notes"]
    print()
    print("[임상 노트]")
    print("  요약:", cn.get("summary", "")[:120])
    print("  감별진단:", cn.get("differential_note", "")[:120])
    print("  disclaimer:", cn.get("disclaimer", ""))

# 결과 저장
out_path = "run_full_test_result.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print()
print("=" * 60)
print(f"결과 저장: {out_path}")
print("전체 5단계 오류 없이 완주")
print("=" * 60)
