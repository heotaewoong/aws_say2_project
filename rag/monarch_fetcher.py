"""
Monarch Initiative API — HPO 코드 → 한국어/영어 증상명 변환
https://api.monarchinitiative.org/v3/api/

역할:
  - HPO 코드(HP:0002202)를 사람이 읽을 수 있는 이름으로 변환
  - 예: HP:0002202 → "Pleural effusion (흉막삼출)"
  - 리포트 가독성 향상 + 의사가 HPO 코드 몰라도 이해 가능

무료, 등록 불필요
"""
import time
import requests

MONARCH_API  = "https://api.monarchinitiative.org/v3/api/entity"
TIMEOUT      = 8   # 빠른 실패 (HPO 이름 조회는 보조 기능)
DELAY        = 0.2 # rate limit 준수

# 로컬 캐시 (같은 HPO 코드 반복 조회 방지)
_HPO_CACHE: dict = {}

# 자주 쓰는 HPO 코드 사전 정의 (API 실패 시 폴백)
_HPO_FALLBACK = {
    "HP:0002094": "호흡곤란 (Dyspnea)",
    "HP:0002202": "흉막삼출 (Pleural effusion)",
    "HP:0002107": "기흉 (Pneumothorax)",
    "HP:0002206": "폐섬유화 (Pulmonary fibrosis)",
    "HP:0012418": "저산소증 (Hypoxemia)",
    "HP:0001903": "빈혈 (Anemia)",
    "HP:0011897": "백혈구 증가 (Leukocytosis)",
    "HP:0001873": "혈소판 감소 (Thrombocytopenia)",
    "HP:0001882": "백혈구 감소 (Leukopenia)",
    "HP:0002113": "폐침윤 (Pulmonary infiltrates)",
    "HP:0006530": "폐간질 이상 (Interstitial lung abnormality)",
    "HP:0025179": "간유리음영 (Ground-glass opacity)",
    "HP:0100750": "무기폐 (Atelectasis)",
    "HP:0002092": "폐동맥 고혈압 (Pulmonary arterial hypertension)",
    "HP:0002088": "폐 이상 형태 (Abnormal lung morphology)",
    "HP:0012735": "기침 (Cough)",
    "HP:0100749": "흉통 (Chest pain)",
    "HP:0002091": "제한성 환기 장애 (Restrictive ventilatory defect)",
    "HP:0045051": "DLCO 감소 (Decreased DLCO)",
    "HP:0100759": "곤봉 손가락 (Clubbing)",
    "HP:0032177": "폐경화 (Pulmonary consolidation)",
    "HP:0003565": "ESR 상승 (Elevated ESR)",
    "HP:0010741": "발 부종 (Pedal edema)",
    "HP:0100763": "림프관계 이상 (Abnormality of the lymphatic system)",
    "HP:0002095": "호흡부전 (Respiratory failure)",
    "HP:0030830": "수포음 (Crackles)",
}


def get_hpo_name(hpo_id: str) -> str:
    """
    HPO 코드 → 사람이 읽을 수 있는 이름 반환

    Parameters
    ----------
    hpo_id : str   예) "HP:0002202"

    Returns
    -------
    str   예) "Pleural effusion (흉막삼출)"
          실패 시 hpo_id 그대로 반환
    """
    if hpo_id in _HPO_CACHE:
        return _HPO_CACHE[hpo_id]

    # 로컬 폴백 먼저 확인 (API 호출 절약)
    if hpo_id in _HPO_FALLBACK:
        _HPO_CACHE[hpo_id] = _HPO_FALLBACK[hpo_id]
        return _HPO_FALLBACK[hpo_id]

    # Monarch API 호출
    url = f"{MONARCH_API}/{hpo_id}"
    try:
        resp = requests.get(url, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        name = data.get("name", hpo_id)
        result = f"{name} ({hpo_id})"
        _HPO_CACHE[hpo_id] = result
        time.sleep(DELAY)
        return result
    except Exception:
        # 실패해도 파이프라인 중단 없이 코드 그대로 반환
        _HPO_CACHE[hpo_id] = hpo_id
        return hpo_id


def enrich_hpo_list(hpo_ids: list) -> list:
    """
    HPO 코드 목록 → 이름 포함 목록으로 변환

    Parameters
    ----------
    hpo_ids : list[str]   예) ["HP:0002202", "HP:0002094"]

    Returns
    -------
    list[str]   예) ["Pleural effusion (흉막삼출)", "호흡곤란 (Dyspnea)"]
    """
    return [get_hpo_name(hpo_id) for hpo_id in hpo_ids]


def format_hpo_for_prompt(positive_hpos: list, negative_hpos: list) -> str:
    """
    HPO 코드 목록을 LLM 프롬프트용 텍스트로 변환
    코드 + 이름을 함께 표시해서 LLM이 증상을 더 잘 이해하도록 함

    Parameters
    ----------
    positive_hpos : list[str]   있는 증상 HPO 코드 목록
    negative_hpos : list[str]   없는 증상 HPO 코드 목록

    Returns
    -------
    str   프롬프트 삽입용 텍스트
    """
    lines = []

    if positive_hpos:
        lines.append("Positive HPO (환자에게 있는 증상):")
        for hpo in positive_hpos:
            name = get_hpo_name(hpo)
            lines.append(f"  + {hpo} — {name}")
    else:
        lines.append("Positive HPO: 없음")

    lines.append("")

    if negative_hpos:
        lines.append("Negative HPO (환자에게 없는 증상 — 감별진단 핵심):")
        for hpo in negative_hpos:
            name = get_hpo_name(hpo)
            lines.append(f"  - {hpo} — {name}")
    else:
        lines.append("Negative HPO: 없음")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=== Monarch Initiative API 테스트 ===\n")

    test_hpos = ["HP:0002202", "HP:0002094", "HP:0012418", "HP:0002107"]

    print("개별 조회:")
    for hpo in test_hpos:
        name = get_hpo_name(hpo)
        print(f"  {hpo} → {name}")

    print("\n프롬프트용 포맷:")
    positive = ["HP:0002202", "HP:0002094", "HP:0012418"]
    negative = ["HP:0002107", "HP:0002206"]
    print(format_hpo_for_prompt(positive, negative))
