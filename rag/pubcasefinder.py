# pubcasefinder.py
# PubCaseFinder API 연동 — HPO 코드 → 희귀질환 랭킹 + 유전자 정보
#
# 회의 확정 (2차 RAG 회의록):
#   - PubCaseFinder API 확정 (HPO 코드로 케이스 검색 트리거)
#   - Soft data: 실시간 질환 매칭 + 유전자 정보 수집
#
# PubCaseFinder란?
#   DBCLS(일본 생명과학통합데이터베이스) 운영.
#   HPO 코드 → 유사 증상의 희귀질환 랭킹 + 유전자 목록 반환.
#   LIRICAL(Orphanet 빈도 기반)과 다른 알고리즘으로 교차검증 가능.
#   API: https://pubcasefinder.dbcls.jp/api/get_ranked_list
#
# 설치 필요:
#   pip install requests

import json
import os
import hashlib
import requests

PCF_API        = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"
REQUEST_TIMEOUT = 15  # 30 → 15초로 단축 (빠른 실패)

# 캐시 디렉토리 (같은 HPO 조합은 재요청 없이 로컬에서 반환)
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pcf_cache")

# Orphanet CSV 경로 (로컬 폴백용)
_HERE = os.path.dirname(__file__)
_ORPHANET_CSV = os.path.join(_HERE, "..", "data", "orphadata_weighted.csv")


def _cache_key(hpo_ids: list, target: str) -> str:
    key = f"{target}::{','.join(sorted(hpo_ids))}"
    return hashlib.md5(key.encode()).hexdigest()


def _load_cache(cache_key: str) -> list | None:
    path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_cache(cache_key: str, data: list) -> None:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = os.path.join(_CACHE_DIR, f"{cache_key}.json")
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _local_fallback(hpo_ids: list, top_k: int) -> list:
    """
    PubCaseFinder 불가 시 로컬 Orphanet CSV로 HPO 매칭 (폴백)
    orphadata_weighted.csv에서 입력 HPO와 겹치는 질환을 점수화해서 반환
    """
    if not os.path.exists(_ORPHANET_CSV):
        return []

    try:
        import pandas as pd
        df = pd.read_csv(_ORPHANET_CSV)

        # 입력 HPO와 매칭되는 행만 필터
        matched = df[df["HPO_ID"].isin(hpo_ids)]
        if matched.empty:
            return []

        # 질환별 매칭 HPO 수 + 가중치 합산
        scores = (
            matched.groupby(["OrphaCode", "DiseaseName"])
            .agg(matched_count=("HPO_ID", "count"), weight_sum=("Weight", "sum"))
            .reset_index()
        )
        scores["score"] = scores["weight_sum"] / (scores["matched_count"] + 1)
        scores = scores.sort_values("score", ascending=False).head(top_k)

        results = []
        for _, row in scores.iterrows():
            results.append({
                "orpha_id":    f"ORPHA:{row['OrphaCode']}",
                "name":        row["DiseaseName"],
                "score":       round(float(row["score"]), 3),
                "genes":       [],
                "description": f"로컬 Orphanet 매칭 (HPO {int(row['matched_count'])}개 일치)",
                "matched_hpo": "",
                "orpha_url":   f"https://www.orpha.net/en/disease/detail/{row['OrphaCode']}",
            })

        print(f"  [로컬 폴백] Orphanet CSV에서 {len(results)}개 질환 매칭")
        return results

    except Exception as e:
        print(f"  ⚠️ 로컬 폴백 실패: {e}")
        return []


def get_ranked_diseases(
    hpo_ids: list,
    target: str = "orphanet",
    top_k: int = 5,
) -> list:
    """
    HPO 코드 목록으로 PubCaseFinder에서 희귀질환 랭킹 검색

    LIRICAL과의 차이:
      - LIRICAL: Orphanet 빈도(sensitivity) 기반 LR 곱 계산 (로컬)
      - PubCaseFinder: 케이스리포트 유사도 + Orphanet 기반 자체 알고리즘 (API)
      → 두 랭킹을 비교하면 더 신뢰할 수 있는 결과 선별 가능

    Parameters
    ----------
    hpo_ids : list[str]
        양성 HPO 코드 목록 (예: ["HP:0002202", "HP:0002094"])
    target : str
        검색 대상 DB — "orphanet" (기본) 또는 "omim"
    top_k : int
        반환할 상위 질환 수 (기본 5)

    Returns
    -------
    list[dict]
        [
            {
                "orpha_id":    "ORPHA:723",
                "name":        "Lymphangioleiomyomatosis",
                "score":       0.95,              # PCF 유사도 점수 (0~1)
                "genes":       ["TSC1", "TSC2"],  # 관련 유전자
                "description": "...",             # 질환 설명
                "orpha_url":   "https://..."
            },
            ...
        ]

    Example
    -------
    >>> results = get_ranked_diseases(["HP:0002202", "HP:0002094"])
    >>> for r in results:
    ...     print(r["name"], r["score"], r["genes"])
    """
    if not hpo_ids:
        return []

    ck = _cache_key(hpo_ids, target)
    cached = _load_cache(ck)
    if cached is not None:
        print(f"  [PubCaseFinder] 캐시 히트 → {len(cached)}개 질환 반환")
        return cached[:top_k]

    params = {
        "format": "json",
        "target": target,
        "hpo_id": ",".join(hpo_ids),
    }

    try:
        resp = requests.get(PCF_API, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()

    except requests.exceptions.Timeout:
        print(f"⚠️ PubCaseFinder 타임아웃 ({REQUEST_TIMEOUT}초) → 로컬 폴백 사용")
        return _local_fallback(hpo_ids, top_k)
    except requests.exceptions.ConnectionError:
        print("⚠️ PubCaseFinder 연결 실패 → 로컬 폴백 사용")
        return _local_fallback(hpo_ids, top_k)
    except requests.exceptions.HTTPError as e:
        print(f"⚠️ PubCaseFinder HTTP 오류: {e} → 로컬 폴백 사용")
        return _local_fallback(hpo_ids, top_k)
    except ValueError:
        print("⚠️ PubCaseFinder 응답 파싱 실패 → 로컬 폴백 사용")
        return _local_fallback(hpo_ids, top_k)

    raw = data if isinstance(data, list) else data.get("result", [])

    results = []
    for item in raw[:top_k]:
        genes = item.get("hgnc_gene_symbol", [])
        if isinstance(genes, str):
            genes = [genes]

        results.append({
            "orpha_id":    item.get("id", ""),
            "name":        item.get("orpha_disease_name_en", "Unknown"),
            "score":       float(item.get("score", 0)),
            "genes":       genes,
            "description": item.get("description", "")[:300],
            "matched_hpo": item.get("matched_hpo_id", ""),
            "orpha_url":   item.get("orpha_url", ""),
        })

    print(f"  [PubCaseFinder] HPO {len(hpo_ids)}개 → 상위 {len(results)}개 질환 매칭")
    if results:
        _save_cache(ck, results)
    return results


def format_pcf_for_llm(pcf_results: list, symptom_text: str = "") -> str:
    """
    PubCaseFinder 결과 + 자연어 증상 → LLM 주입용 컨텍스트 블록

    회의 확정 방식:
      "자연어 증상 컨텍스트 + HPO를 LLM에 함께 주입"
      HPO만으로는 임상 맥락이 손실되므로 원본 소견도 포함.

    Parameters
    ----------
    pcf_results : list[dict]
        get_ranked_diseases() 반환값
    symptom_text : str
        의사가 입력한 원본 자연어 증상 텍스트

    Returns
    -------
    str
        LLM 프롬프트 삽입용 컨텍스트

    Example
    -------
    >>> ctx = format_pcf_for_llm(results, "40세 여성 호흡곤란...")
    >>> prompt = f"...{ctx}..."
    """
    lines = []

    if symptom_text:
        lines.append("【환자 임상 맥락 — 원본 소견】")
        lines.append(symptom_text.strip())
        lines.append("")

    if not pcf_results:
        lines.append("【PubCaseFinder 매칭 결과】")
        lines.append("매칭 결과 없음 — Orphanet/LIRICAL 결과만 활용하세요.")
        return "\n".join(lines)

    lines.append(f"【PubCaseFinder 희귀질환 매칭 결과 ({len(pcf_results)}건)】")
    lines.append("아래는 환자 HPO 코드와 가장 유사한 희귀질환 목록입니다.")
    lines.append("LIRICAL 랭킹과 비교하여 교차 검증하세요.")
    lines.append("")

    for i, r in enumerate(pcf_results, 1):
        gene_str = ", ".join(r["genes"]) if r["genes"] else "유전자 정보 없음"
        lines.append(
            f"{i}. {r['name']} ({r['orpha_id']}) — 유사도: {r['score']:.3f}\n"
            f"   관련 유전자: {gene_str}\n"
            f"   {r['description']}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 55)
    print("PubCaseFinder API 테스트")
    print("=" * 55)

    test_hpos = ["HP:0002202", "HP:0002094", "HP:0012418"]
    test_symptom = "40세 여성. 3주째 호흡곤란과 흉통. 기침·발열 없음."

    print(f"\n입력 HPO: {test_hpos}")
    results = get_ranked_diseases(test_hpos, top_k=5)

    if results:
        print(f"\n--- 상위 {len(results)}개 질환 ---")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['name']} ({r['orpha_id']})")
            print(f"  유전자: {r['genes']}")

        print("\n--- LLM 주입용 컨텍스트 ---")
        print(format_pcf_for_llm(results, test_symptom))
    else:
        print("결과 없음")
