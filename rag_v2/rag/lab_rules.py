# lab_rules.py
# 혈액·폐기능 검사 수치 → HPO 코드 Rule-based 변환
# 폐질환 특화 (lung disease context)

LAB_HPO_RULES = {
    # ── 혈구 계산 (CBC) ─────────────────────────────────────────
    "WBC": {
        "unit": "×10³/μL",
        "high": {"threshold": 11.0, "hpo": "HP:0011897", "name": "Leukocytosis(백혈구 증가)"},
        "low":  {"threshold": 4.0,  "hpo": "HP:0001882", "name": "Leukopenia(백혈구 감소)"},
    },
    "HGB": {
        "unit": "g/dL",
        "high": {"threshold": 17.5, "hpo": "HP:0001902", "name": "Polycythemia(적혈구 증가)"},
        "low":  {"threshold": 12.0, "hpo": "HP:0001903", "name": "Anemia(빈혈)"},
    },
    "PLT": {
        "unit": "×10³/μL",
        "high": {"threshold": 400,  "hpo": "HP:0001894", "name": "Thrombocytosis(혈소판 증가)"},
        "low":  {"threshold": 150,  "hpo": "HP:0001873", "name": "Thrombocytopenia(혈소판 감소)"},
    },
    "NEUTROPHIL": {
        "unit": "%",
        "high": {"threshold": 75.0, "hpo": "HP:0001875", "name": "Neutrophilia(호중구 증가)"},
    },
    "LYMPHOCYTE": {
        "unit": "%",
        "low":  {"threshold": 20.0, "hpo": "HP:0001888", "name": "Lymphopenia(림프구 감소)"},
    },
    "EOSINOPHIL": {
        "unit": "%",
        "high": {"threshold": 5.0,  "hpo": "HP:0001880", "name": "Eosinophilia(호산구 증가)"},
    },
    # ── 폐 기능 / 조직 손상 마커 ────────────────────────────────
    "LDH": {
        "unit": "U/L",
        "high": {"threshold": 250,  "hpo": "HP:0003155", "name": "Elevated LDH(조직 손상)"},
    },
    "CRP": {
        "unit": "mg/L",
        "high": {"threshold": 5.0,  "hpo": "HP:0012116", "name": "Elevated CRP(염증 반응)"},
    },
    "D_DIMER": {
        "unit": "mg/L FEU",
        "high": {"threshold": 0.5,  "hpo": "HP:0003560", "name": "Elevated D-dimer(혈전 위험)"},
    },
    "ALT": {
        "unit": "U/L",
        "high": {"threshold": 40.0, "hpo": "HP:0002910", "name": "Elevated liver enzymes"},
    },
    "AST": {
        "unit": "U/L",
        "high": {"threshold": 40.0, "hpo": "HP:0002910", "name": "Elevated liver enzymes"},
    },
    # ── 폐기능 검사 (PFT) ───────────────────────────────────────
    "FEV1": {
        "unit": "% predicted",
        "low": {"threshold": 80.0, "hpo": "HP:0002093", "name": "Reduced FEV1(기도 폐쇄)"},
    },
    "FVC": {
        "unit": "% predicted",
        "low": {"threshold": 80.0, "hpo": "HP:0002091", "name": "Reduced FVC(제한성 환기 장애)"},
    },
    "DLCO": {
        "unit": "% predicted",
        "low": {"threshold": 70.0, "hpo": "HP:0002878", "name": "Reduced DLCO(확산능 저하)"},
    },
    # ── 산소화 ──────────────────────────────────────────────────
    "SpO2": {
        "unit": "%",
        "low": {"threshold": 95.0, "hpo": "HP:0012418", "name": "Hypoxemia(저산소증)"},
    },
    "PaO2": {
        "unit": "mmHg",
        "low": {"threshold": 80.0, "hpo": "HP:0012418", "name": "Hypoxemia(저산소증)"},
    },
}

# 입력 키 정규화 (다양한 표기 → 표준 키)
_ALIAS_MAP = {
    "d-dimer":              "D_DIMER",
    "d_dimer":              "D_DIMER",
    "spo2":                 "SpO2",
    "oxygen saturation":    "SpO2",
    "o2 sat":               "SpO2",
    "wbc count":            "WBC",
    "platelet count":       "PLT",
    "hemoglobin":           "HGB",
    "hgb":                  "HGB",
    "neutrophil (%)":       "NEUTROPHIL",
    "lymphocyte (%)":       "LYMPHOCYTE",
    "eosinophil (%)":       "EOSINOPHIL",
    "alanine aminotransferase": "ALT",
    "aspartate aminotransferase": "AST",
}


def _normalize_key(key: str) -> str:
    return _ALIAS_MAP.get(key.lower().strip(), key)


def lab_to_hpo(lab_results: dict, verbose: bool = True) -> list:
    """
    혈액/폐기능 검사 수치 딕셔너리 → 이상 HPO 코드 목록

    Parameters
    ----------
    lab_results : dict
        {"WBC": 15.2, "HGB": 9.1, "SpO2": 91.0, "FEV1": 65.0}
    verbose : bool
        True 이면 판정 결과를 출력

    Returns
    -------
    list[str]
        이상 소견 HPO 코드 목록 (정상이면 빈 리스트)

    Example
    -------
    >>> lab_to_hpo({"WBC": 15.2, "HGB": 9.1, "LDH": 320})
    ['HP:0011897', 'HP:0001903', 'HP:0003155']
    """
    hpo_codes = []

    for raw_key, value in lab_results.items():
        key = _normalize_key(raw_key)

        if key not in LAB_HPO_RULES:
            if verbose:
                print(f"  ⚠️  {raw_key}: 매핑 규칙 없음 (LAB_HPO_RULES에 추가 필요)")
            continue

        rules = LAB_HPO_RULES[key]

        if "high" in rules and value > rules["high"]["threshold"]:
            hpo = rules["high"]["hpo"]
            hpo_codes.append(hpo)
            if verbose:
                print(
                    f"  🔴 {key}={value} {rules['unit']}"
                    f" > {rules['high']['threshold']}"
                    f" → {rules['high']['name']} ({hpo})"
                )
        elif "low" in rules and value < rules["low"]["threshold"]:
            hpo = rules["low"]["hpo"]
            hpo_codes.append(hpo)
            if verbose:
                print(
                    f"  🔵 {key}={value} {rules['unit']}"
                    f" < {rules['low']['threshold']}"
                    f" → {rules['low']['name']} ({hpo})"
                )
        else:
            if verbose:
                print(f"  ✅ {key}={value} {rules['unit']} → 정상 범위")

    return hpo_codes


if __name__ == "__main__":
    print("=== 혈액·폐기능 검사 HPO 변환 테스트 ===\n")
    sample = {
        "WBC":   15.2,   # 백혈구 증가
        "HGB":    9.1,   # 빈혈
        "PLT":   350,    # 정상
        "LDH":   320,    # LDH 상승
        "CRP":    8.5,   # CRP 상승
        "SpO2":  91.0,   # 저산소증
        "FEV1":  65.0,   # 폐기능 저하
    }
    result = lab_to_hpo(sample)
    print(f"\n감지된 이상 HPO 코드 ({len(result)}개): {result}")
