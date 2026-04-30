"""
일반 폐질환 스코어링 — lung_disease_profiles_v2.yaml 기반
HPO 코드 + X-ray 소견 + Lab 수치를 가중치 합산으로 Top10 일반 질환 랭킹 생성

회의 확정 (2026-04-29):
  - 일반 질환 Ranking을 LLM 인풋으로 넣기로 결정
  - aws_say2_project_vision 단독으로 구현 (외부 연동 없음)
  - YAML 기반 Rule-based 스코어링 (LIRICAL과 별도 파이프라인)
"""
import os
import yaml

_HERE = os.path.dirname(__file__)
_YAML_CANDIDATES = [
    os.path.join(_HERE, "..", "..", "aws_say2_project", "data", "lung_disease_profiles_v2.yaml"),
    os.path.join(_HERE, "..", "data", "lung_disease_profiles_v2.yaml"),
]

# X-ray 소견 → 질환 연관 키워드 매핑
_XRAY_LABEL_MAP = {
    "Consolidation":  ["consolidation", "infiltrate", "airspace disease"],
    "Lung Opacity":   ["opacity", "infiltrate", "consolidation"],
    "Pleural Effusion": ["pleural effusion", "effusion"],
    "Pneumothorax":   ["pneumothorax"],
    "Cardiomegaly":   ["cardiomegaly", "cardiac enlargement"],
    "Atelectasis":    ["atelectasis", "collapse"],
    "Edema":          ["edema", "pulmonary edema"],
    "Pneumonia":      ["consolidation", "infiltrate", "opacity"],
}


def _load_profiles() -> dict:
    for path in _YAML_CANDIDATES:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    return {}


def rank_general_diseases(
    positive_hpos: list,
    xray_preds: dict,
    lab_results: dict,
    top_k: int = 10,
) -> list:
    """
    일반 폐질환 Top-K 랭킹 생성

    Parameters
    ----------
    positive_hpos : list[str]   환자 Positive HPO 코드 목록
    xray_preds    : dict        SooNet 예측 결과 {label: (prob, hpo)}
    lab_results   : dict        혈액검사 수치 {항목명: 값}
    top_k         : int         반환할 상위 질환 수

    Returns
    -------
    list[dict]
        [
            {
                "disease_key":  "community_acquired_pneumonia",
                "disease_name": "지역사회획득 폐렴 (CAP)",
                "icd10":        ["J13", "J14", ...],
                "score":        0.72,
                "score_detail": {"hpo": 0.3, "xray": 0.25, "lab": 0.17},
            },
            ...
        ]
    """
    profiles = _load_profiles()
    if not profiles:
        print("  ⚠️ lung_disease_profiles_v2.yaml 없음 — 일반 질환 스코어링 건너뜀")
        return []

    hpo_set = set(positive_hpos)
    results = []

    for disease_key, profile in profiles.items():
        if not isinstance(profile, dict):
            continue

        weights = profile.get("weights", {"symptoms": 0.25, "lab": 0.25, "radiology": 0.5})
        w_sym = weights.get("symptoms", 0.25)
        w_lab = weights.get("lab", 0.25)
        w_rad = weights.get("radiology", 0.5)

        # ── HPO 매칭 점수 ──────────────────────────────────────────
        hpo_map = profile.get("hpo_symptom_map", {})
        disease_hpos = set(hpo_map.values())
        if disease_hpos:
            hpo_score = len(hpo_set & disease_hpos) / len(disease_hpos)
        else:
            hpo_score = 0.0

        # ── X-ray 소견 매칭 점수 ───────────────────────────────────
        radiology_findings = [f.lower() for f in profile.get("radiology_findings", [])]
        xray_score = 0.0
        xray_match_count = 0
        for label, (prob, _) in xray_preds.items():
            if prob < 0.3:
                continue
            keywords = _XRAY_LABEL_MAP.get(label, [label.lower()])
            for kw in keywords:
                if any(kw in finding for finding in radiology_findings):
                    xray_score += prob
                    xray_match_count += 1
                    break
        if xray_match_count > 0:
            xray_score = min(xray_score / max(len(radiology_findings), 1), 1.0)

        # ── Lab 패턴 매칭 점수 ─────────────────────────────────────
        lab_patterns = [p.lower() for p in profile.get("lab_patterns", [])]
        lab_score = 0.0
        if lab_patterns and lab_results:
            matched = _match_lab_patterns(lab_results, lab_patterns)
            lab_score = matched / len(lab_patterns)

        # ── 가중치 합산 ────────────────────────────────────────────
        total_score = (
            w_sym * hpo_score +
            w_rad * xray_score +
            w_lab * lab_score
        )

        disease_name = profile.get("disease_kr", disease_key)
        icd10 = profile.get("icd10", [])

        results.append({
            "disease_key":  disease_key,
            "disease_name": disease_name,
            "icd10":        icd10[:3] if icd10 else [],
            "score":        round(total_score, 4),
            "score_detail": {
                "hpo":  round(hpo_score, 3),
                "xray": round(xray_score, 3),
                "lab":  round(lab_score, 3),
            },
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_k]

    print(f"  [일반 질환 스코어링] Top {len(top)}개 질환 랭킹:")
    for i, d in enumerate(top, 1):
        print(f"  {i:2d}. {d['disease_name']:<40} score={d['score']:.4f}")

    return top


def _match_lab_patterns(lab_results: dict, patterns: list) -> int:
    """Lab 수치와 패턴 키워드 매칭 (단순 키워드 기반)"""
    matched = 0
    lab_keys_lower = {k.lower(): v for k, v in lab_results.items()}

    pattern_rules = {
        "leukocytosis":    lambda: lab_keys_lower.get("wbc", 0) > 11.0,
        "leukopenia":      lambda: 0 < lab_keys_lower.get("wbc", 99) < 4.0,
        "neutrophilia":    lambda: lab_keys_lower.get("wbc", 0) > 11.0,
        "elevated crp":    lambda: lab_keys_lower.get("crp", 0) > 5.0,
        "elevated ldh":    lambda: lab_keys_lower.get("ldh", 0) > 250,
        "hypoxemia":       lambda: 0 < lab_keys_lower.get("spo2", 100) < 95,
        "anemia":          lambda: 0 < lab_keys_lower.get("hgb", 99) < 12.0,
        "thrombocytopenia": lambda: 0 < lab_keys_lower.get("plt", 999) < 150,
        "elevated fev1":   lambda: lab_keys_lower.get("fev1", 100) < 70,
        "decreased dlco":  lambda: lab_keys_lower.get("dlco", 100) < 70,
    }

    for pattern in patterns:
        p_lower = pattern.lower()
        for key, rule in pattern_rules.items():
            if key in p_lower:
                try:
                    if rule():
                        matched += 1
                except Exception:
                    pass
                break

    return matched


def format_general_ranking_for_llm(ranking: list) -> str:
    """
    일반 질환 랭킹 → LLM 프롬프트용 텍스트 변환

    Parameters
    ----------
    ranking : list[dict]   rank_general_diseases() 반환값

    Returns
    -------
    str   프롬프트 삽입용 텍스트
    """
    if not ranking:
        return "일반 질환 랭킹: 데이터 없음"

    lines = [f"【일반 폐질환 랭킹 Top {len(ranking)} (HPO + X-ray + Lab 가중치 합산)】"]
    for i, d in enumerate(ranking, 1):
        icd_str = ", ".join(d["icd10"]) if d["icd10"] else "N/A"
        detail = d["score_detail"]
        lines.append(
            f"{i:2d}. {d['disease_name']}\n"
            f"    ICD-10: {icd_str} | 종합점수: {d['score']:.4f}\n"
            f"    (HPO={detail['hpo']:.2f}, X-ray={detail['xray']:.2f}, Lab={detail['lab']:.2f})"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== 일반 질환 스코어링 테스트 ===\n")

    test_hpos = ["HP:0002094", "HP:0001945", "HP:0012735"]
    test_xray = {
        "Consolidation": (0.82, "HP:0002113"),
        "Lung Opacity":  (0.65, "HP:0002113"),
        "Pleural Effusion": (0.31, "HP:0002202"),
    }
    test_lab = {"WBC": 14.5, "CRP": 45.0, "SpO2": 92.0}

    ranking = rank_general_diseases(test_hpos, test_xray, test_lab, top_k=5)
    print("\n--- LLM 주입용 텍스트 ---")
    print(format_general_ranking_for_llm(ranking))
