# lirical_scorer.py
# LIRICAL (Likelihood Ratio Interpretation of Clinical AbnormaLities) 방식 질환 스코어링
# 참고 논문: Jacobsen et al. 2020, AJHG — https://doi.org/10.1016/j.ajhg.2020.06.004
#
# 핵심 개념:
#   LR(질환) = ∏ P(증상|질환) / P(증상|배경)  ← 있는 증상
#            × ∏ P(~증상|질환) / P(~증상|배경) ← 없는 증상
#
# Orphanet 빈도 가중치 매핑 (knowledge_base.py와 동일):
#   Always(100%) → 1.0 | Very frequent(99-80%) → 0.9
#   Frequent(79-30%) → 0.5 | Occasional(29-5%) → 0.1 | Unknown → 0.3

import math


DEFAULT_BG_FREQ    = 0.05   # 일반 인구에서 해당 증상 빈도 기본값 (5%)
DEFAULT_SENSITIVITY = 0.5   # 질환 프로파일에 HPO 없을 때 기본 민감도


def compute_lr_score(
    positive_hpos: list,
    negative_hpos: list,
    disease_hpo_profile: dict,
    bg_freq_default: float = DEFAULT_BG_FREQ,
    sensitivity_default: float = DEFAULT_SENSITIVITY,
) -> float:
    """
    단일 질환에 대한 LIRICAL LR 점수 계산

    Parameters
    ----------
    positive_hpos : list[str]
        환자에게 존재하는 증상의 HPO 코드 목록
    negative_hpos : list[str]
        환자에게 명시적으로 없는 증상의 HPO 코드 목록
    disease_hpo_profile : dict
        {"HP:0002202": {"sensitivity": 0.90, "bg_freq": 0.05}, ...}
        (sensitivity = Orphanet 빈도 가중치, bg_freq = 배경 빈도)

    Returns
    -------
    float
        LR 점수. 1.0 기준, 높을수록 해당 질환일 가능성 높음.
    """
    lr = 1.0

    # 있는 증상: sens/bg — 이 증상이 이 질환에 특이적일수록 LR 상승
    for hpo in positive_hpos:
        if hpo in disease_hpo_profile:
            sens = disease_hpo_profile[hpo]["sensitivity"]
            bg   = disease_hpo_profile[hpo]["bg_freq"]
        else:
            # 프로파일에 없는 증상 → 질환과 무관한 증상으로 취급
            sens = bg_freq_default
            bg   = bg_freq_default

        bg   = max(bg, 1e-6)
        lr  *= sens / bg

    # 없는 증상: (1-sens)/(1-bg) — 필수 증상이 없으면 LR 감소
    for hpo in negative_hpos:
        if hpo in disease_hpo_profile:
            sens = disease_hpo_profile[hpo]["sensitivity"]
            bg   = disease_hpo_profile[hpo]["bg_freq"]
        else:
            sens = sensitivity_default
            bg   = bg_freq_default

        bg   = min(max(bg,   1e-6), 1.0 - 1e-6)
        sens = min(max(sens, 1e-6), 1.0 - 1e-6)
        lr  *= (1 - sens) / (1 - bg)

    return lr


def build_hpo_profile_from_orphanet(disease_rows) -> dict:
    """
    knowledge_base.py 로 생성된 orphadata_weighted.csv의
    단일 질환 행 DataFrame → HPO 프로파일 딕셔너리 변환

    Parameters
    ----------
    disease_rows : pd.DataFrame
        OrphaCode 가 동일한 행들 (컬럼: HPO_ID, Weight)

    Returns
    -------
    dict
        {"HP:0002202": {"sensitivity": 0.90, "bg_freq": 0.05}, ...}
    """
    profile = {}
    for _, row in disease_rows.iterrows():
        hpo_id = str(row.get("HPO_ID", "")).strip()
        if not hpo_id:
            continue
        weight = float(row.get("Weight", 0.3))
        profile[hpo_id] = {
            "sensitivity": weight,
            "bg_freq":     DEFAULT_BG_FREQ,
        }
    return profile


def build_disease_database(kb_df) -> dict:
    """
    전체 orphadata_weighted.csv DataFrame → LIRICAL용 질환 DB 딕셔너리

    Parameters
    ----------
    kb_df : pd.DataFrame
        knowledge_base.py 로 생성된 전체 DataFrame
        (컬럼: OrphaCode, DiseaseName, HPO_ID, HPO_Term, Weight)

    Returns
    -------
    dict
        {
            "ORPHA:723": {
                "name": "Lymphangioleiomyomatosis",
                "hpo_profile": {"HP:0002202": {...}, ...},
                "is_rare": True,
                "prevalence": "Unknown",
                "genes": [],
            }, ...
        }
    """
    db = {}
    for orpha_code, group in kb_df.groupby("OrphaCode"):
        disease_name = group["DiseaseName"].iloc[0]
        hpo_profile  = build_hpo_profile_from_orphanet(group)

        db[f"ORPHA:{orpha_code}"] = {
            "name":        disease_name,
            "hpo_profile": hpo_profile,
            "is_rare":     True,       # Orphanet 수록 질환 = 희귀질환
            "prevalence":  "Unknown",  # en_product9.xml 에서 별도 파싱 가능
            "genes":       [],         # en_product6.xml 에서 별도 파싱 가능
        }

    return db


def rank_diseases(
    positive_hpos: list,
    negative_hpos: list,
    disease_database: dict,
    top_k: int = 10,
    log_scale: bool = False,
) -> list:
    """
    전체 질환 DB에 대해 LR 점수를 계산하고 점수 내림차순 상위 K개 반환

    Parameters
    ----------
    positive_hpos : list[str]
    negative_hpos : list[str]
    disease_database : dict    build_disease_database() 반환값
    top_k : int
    log_scale : bool
        True 이면 log(LR)로 정규화 (수치 매우 클 때 안정성 향상)

    Returns
    -------
    list[dict]
        점수 내림차순 Top K 질환 목록
    """
    if not positive_hpos and not negative_hpos:
        return []

    # Positive HPO 없으면 스코어링 의미 없음
    if not positive_hpos:
        return []

    scores = []
    for orpha_code, info in disease_database.items():
        score = compute_lr_score(
            positive_hpos,
            negative_hpos,
            info.get("hpo_profile", {}),
        )
        if log_scale:
            score = math.log(max(score, 1e-9))

        scores.append({
            "orpha_code":   orpha_code,
            "disease_name": info["name"],
            "score":        score,
            "is_rare":      info.get("is_rare",     True),
            "prevalence":   info.get("prevalence",  "Unknown"),
            "genes":        info.get("genes",        []),
        })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    print("=== LIRICAL Scorer 단위 테스트 ===\n")

    mock_db = {
        "ORPHA:723": {
            "name": "Lymphangioleiomyomatosis (LAM)",
            "hpo_profile": {
                "HP:0002202": {"sensitivity": 0.90, "bg_freq": 0.05},  # Pleural effusion
                "HP:0002094": {"sensitivity": 0.80, "bg_freq": 0.08},  # Dyspnea
                "HP:0012418": {"sensitivity": 0.70, "bg_freq": 0.04},  # Hypoxemia
            },
            "is_rare": True, "prevalence": "1-9/100000", "genes": ["TSC1", "TSC2"],
        },
        "ORPHA:2302": {
            "name": "Idiopathic Pulmonary Fibrosis (IPF)",
            "hpo_profile": {
                "HP:0002094": {"sensitivity": 0.95, "bg_freq": 0.08},  # Dyspnea
                "HP:0002206": {"sensitivity": 0.60, "bg_freq": 0.03},  # Pulmonary fibrosis
                "HP:0002202": {"sensitivity": 0.20, "bg_freq": 0.05},  # Pleural effusion (드묾)
            },
            "is_rare": True, "prevalence": "1-9/100000", "genes": ["TERT", "TERC"],
        },
    }

    pos = ["HP:0002202", "HP:0002094", "HP:0012418"]  # 흉막삼출 + 호흡곤란 + 저산소증
    neg = ["HP:0002206"]                               # 폐섬유화 없음

    ranking = rank_diseases(pos, neg, mock_db)
    print("랭킹 결과:")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['disease_name']}")
        print(f"     LR 점수: {r['score']:.4f} | 유전자: {r['genes']}")
