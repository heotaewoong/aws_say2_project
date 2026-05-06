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


# ══════════════════════════════════════════════════════════════
# 확정 문서 (2026-04-29) §4 — 인과 유전자 조회 (Monarch causal_genes)
# ══════════════════════════════════════════════════════════════
# 확정 문서 §3.2 §8번 섹션 — Top N 질환별 [Monarch] 블록:
#   - 인과 유전자: {monarch_genes_N}
#   - Orphanet 교차검증: {cross_validation_N}
#
# Monarch Initiative는 OrphaCode를 ORDO IRI로 변환해서 association을 조회.
# Endpoint: /api/entity/{ORPHA:NNN}/biolink:Disease+to+biolink:Gene+associations

_GENE_CACHE: dict = {}
_DISEASE_INFO_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════
# 확정 문서 §2.2 — Monarch 반환 데이터 (name/label, description, 인과 유전자)
# disease_id (URL path parameter) → 질환명·설명·유전자 보강
# ══════════════════════════════════════════════════════════════
def get_disease_info(disease_id: str) -> dict:
    """
    확정 문서 §2.2 표 — Monarch Initiative (v3) 반환 데이터:
      - name / label (질환명)
      - description (질환 설명)
      - 인과 유전자

    disease_id 는 URL path 파라미터로 사용 (예: "Orphanet:58", "OMIM:606693", "MONDO:0007139")

    Returns
    -------
    dict {
        "id":          str,
        "name":        str,   # name / label
        "description": str,   # 질환 설명
        "causal_genes": list[str],
    }
    """
    cache_key = f"info:{disease_id}"
    if cache_key in _DISEASE_INFO_CACHE:
        return _DISEASE_INFO_CACHE[cache_key]

    info = {"id": disease_id, "name": "", "description": "", "causal_genes": []}

    # 1차 — direct entity 조회
    try:
        url = f"{MONARCH_API}/{disease_id}"
        resp = requests.get(url, timeout=TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            info["id"]          = data.get("id", disease_id)
            info["name"]        = data.get("name") or data.get("label", "")
            info["description"] = data.get("description", "") or ""
        time.sleep(DELAY)
    except Exception:
        pass

    # 2차 — direct 실패 시 search 엔드포인트 (OMIM/MESH 등 cross-ref 매핑)
    if not info["name"]:
        try:
            search_url = "https://api.monarchinitiative.org/v3/api/search"
            resp = requests.get(
                search_url,
                params={"q": disease_id, "limit": 1, "category": "biolink:Disease"},
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    item = items[0]
                    info["id"]          = item.get("id", disease_id)
                    info["name"]        = item.get("name") or item.get("label", "")
                    info["description"] = item.get("description", "") or ""
            time.sleep(DELAY)
        except Exception:
            pass

    # OrphaCode면 causal_genes도 같이 채움
    if "Orphanet" in disease_id or "ORPHA" in disease_id:
        code_only = disease_id.replace("Orphanet:", "").replace("ORPHA:", "").strip()
        info["causal_genes"] = get_causal_genes(code_only)

    _DISEASE_INFO_CACHE[cache_key] = info
    return info


def get_causal_genes(orpha_code: str, top_k: int = 10) -> list:
    """
    OrphaCode → Monarch 인과 유전자 목록

    확정 문서 §2.4-3: causal_genes (Monarch)
      "Orphanet 교차검증: {cross_validation}"
      "Monarch·Orphanet 양 소스에서 TSC1/TSC2 확인"

    Parameters
    ----------
    orpha_code : str   예) "58" 또는 "ORPHA:58"
    top_k      : int   반환할 유전자 최대 개수

    Returns
    -------
    list[str]  유전자 심볼 목록 (예: ["TSC1", "TSC2"])
               실패 시 빈 리스트
    """
    code_str = str(orpha_code).replace("ORPHA:", "").strip()
    cache_key = f"genes:{code_str}"
    if cache_key in _GENE_CACHE:
        return _GENE_CACHE[cache_key]

    # Monarch는 Orphanet 코드 그대로 받지 않고 ORDO 형태 IRI 또는 MONDO ID 사용
    # 우선 시도: /api/entity/Orphanet:{code}/biolink:DiseaseToGeneAssociation
    # 폴백: /api/search?q={code}&category=biolink:Disease
    candidates = [
        f"{MONARCH_API}/Orphanet:{code_str}",
        f"{MONARCH_API}/ORPHA:{code_str}",
    ]

    for url in candidates:
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            if resp.status_code != 200:
                continue
            data = resp.json()
            assoc = data.get("association_counts", {}) or {}
            # Monarch v3 응답 구조: associations endpoint 별도 호출 필요
            entity_id = data.get("id") or f"Orphanet:{code_str}"

            assoc_url = (
                f"https://api.monarchinitiative.org/v3/api/entity/"
                f"{entity_id}/biolink:DiseaseToGeneAssociation"
            )
            assoc_resp = requests.get(assoc_url, timeout=TIMEOUT,
                                     params={"limit": top_k, "offset": 0})
            if assoc_resp.status_code == 200:
                items = assoc_resp.json().get("items", [])
                genes = []
                for item in items:
                    obj = item.get("object_label") or item.get("object", "")
                    if obj and obj not in genes:
                        genes.append(obj)
                _GENE_CACHE[cache_key] = genes
                time.sleep(DELAY)
                return genes
        except Exception:
            continue

    # 모든 시도 실패
    _GENE_CACHE[cache_key] = []
    return []


def format_monarch_for_prompt(orpha_code: str, orphanet_genes: list) -> str:
    """
    확정 문서 §3.2 §8번 [Monarch] 블록 형식

    Parameters
    ----------
    orpha_code      : str  Top N 질환의 OrphaCode
    orphanet_genes  : list 교차검증할 Orphanet 유전자 (genes_from_orphadata)

    Returns
    -------
    str  프롬프트 §8번에 삽입될 [Monarch] 블록
    """
    monarch_genes = get_causal_genes(orpha_code)

    # 교차검증 (orphanet_fetcher.cross_validate_genes 와 동일 로직)
    orpha_set   = set()
    for g in orphanet_genes:
        if isinstance(g, dict):
            orpha_set.add(g.get("gene", "").upper())
        else:
            orpha_set.add(str(g).upper())
    monarch_set = {str(g).upper() for g in monarch_genes}

    matched    = sorted(orpha_set & monarch_set)
    mismatched = sorted((orpha_set | monarch_set) - (orpha_set & monarch_set))

    if matched and not mismatched:
        cross_text = "DB·API 교차검증 일치"
    elif matched and mismatched:
        cross_text = f"부분 일치 — 일치: {', '.join(matched)} / 불일치: {', '.join(mismatched)}"
    elif mismatched:
        cross_text = "DB·API 불일치 — 추가 확인 필요"
    else:
        cross_text = "교차검증 데이터 없음"

    gene_text = ", ".join(monarch_genes) if monarch_genes else "정보 없음"

    return (
        f"[Monarch]\n"
        f"- 인과 유전자: {gene_text}\n"
        f"- Orphanet 교차검증: {cross_text}"
    )


if __name__ == "__main__":
    print("=== Monarch Initiative API 테스트 ===\n")

    test_hpos = ["HP:0002202", "HP:0002094", "HP:0012418", "HP:0002107"]

    print("개별 HPO 조회:")
    for hpo in test_hpos:
        name = get_hpo_name(hpo)
        print(f"  {hpo} → {name}")

    print("\nHPO 프롬프트 포맷:")
    positive = ["HP:0002202", "HP:0002094", "HP:0012418"]
    negative = ["HP:0002107", "HP:0002206"]
    print(format_hpo_for_prompt(positive, negative))

    print("\n=== 인과 유전자 조회 (확정 문서 §4) ===")
    # 예: ORPHA:79318 (PMM2-CDG)
    genes = get_causal_genes("79318")
    print(f"ORPHA:79318 인과 유전자: {genes}")
    print("\n[Monarch] 프롬프트 블록:")
    print(format_monarch_for_prompt("79318", orphanet_genes=["PMM2"]))
