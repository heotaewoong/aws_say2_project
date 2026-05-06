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

# 확정 문서 §2.2 — PubCaseFinder 공식 엔드포인트 (검증 완료 2026-05-04)
# 파라미터: target=omim, format=json, phenotype=HPO코드(쉼표구분)
PCF_API        = "https://pubcasefinder.dbcls.jp/api/get_diseases"
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
            disease_id = f"ORPHA:{row['OrphaCode']}"
            results.append({
                # 확정 스펙 §2.2 필드
                "disease_id":   disease_id,
                "disease_name": row["DiseaseName"],
                "score":        round(float(row["score"]), 3),
                "pmid_list":    [],
                "hpo_list":     [],
                # 추가 호환 필드
                "orpha_id":     disease_id,
                "name":         row["DiseaseName"],
                "genes":        [],
                "rank":         None,
                "matched_hpo":  "",
                "description":  f"로컬 Orphanet 매칭 (HPO {int(row['matched_count'])}개 일치)",
                "orpha_url":    f"https://www.orpha.net/en/disease/detail/{row['OrphaCode']}",
            })

        print(f"  [로컬 폴백] Orphanet CSV에서 {len(results)}개 질환 매칭")
        return results

    except Exception as e:
        print(f"  ⚠️ 로컬 폴백 실패: {e}")
        return []


def get_ranked_diseases(
    hpo_ids: list,
    target: str = "omim",   # 확정 문서 §2.2 — 기본 target은 "omim"
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
                "disease_id":   "OMIM:617300",    # 확정 스펙 §2.2 — OMIM ID (target=omim)
                "disease_name": "Lymphangioleiomyomatosis",
                "score":        0.95,             # PCF 유사도 점수 (0~1)
                "pmid_list":    ["12345", ...],   # 관련 논문 PMID (enrich 후 채워짐)
                "hpo_list":     ["HP:0002202", ...],  # 매칭된 HPO 코드 list
                # 추가 호환 필드 (enrich_pcf_results() 후 보강)
                "orpha_id":    "OMIM:617300",     # disease_id 와 동일 (target=omim 기준)
                "name":        "Lymphangioleiomyomatosis",
                "genes":       ["GENEID:2050"],   # 원시 gene_id (심볼 변환은 enrich 단계)
                "description": "",                # raw API에선 빈값, enrich 후 Monarch 보강
                "orpha_url":   "",                # raw API에선 빈값 (target=omim)
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

    # 확정 문서 §2.2 — 파라미터: target, format, phenotype (HPO 코드 쉼표 구분)
    params = {
        "format":    "json",
        "target":    target,
        "phenotype": ",".join(hpo_ids),
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
        # 확정 문서 §2.2 — 실제 API 응답 필드:
        # id, score, matched_hpo_id, gene_id, rank, annotation_hp_num, annotation_hp_sum_ic
        disease_id = item.get("id", "")  # 예: "OMIM:617300"

        # gene_id 는 "GENEID:2050" 형식 → 심볼 변환은 후속 enrich 단계에서
        gene_id_str = item.get("gene_id", "")
        genes = [gene_id_str] if gene_id_str else []

        # 확정 스펙 §2.2 — HPO 코드 list (매칭된 HPO)
        matched_hpo = item.get("matched_hpo_id", "")
        if isinstance(matched_hpo, str) and matched_hpo:
            hpo_list = [h.strip() for h in matched_hpo.split(",") if h.strip()]
        elif isinstance(matched_hpo, list):
            hpo_list = matched_hpo
        else:
            hpo_list = []

        # 확정 스펙 §2.2 — pmid_list (PubCaseFinder 응답엔 직접 없음 → 별도 케이스리포트
        # 엔드포인트 또는 PubMed 보강이 필요. 응답에 있으면 그대로 사용)
        pmid_list = item.get("pmid_list") or item.get("pmids") or []
        if isinstance(pmid_list, str):
            pmid_list = [p.strip() for p in pmid_list.split(",") if p.strip()]

        # disease_name 은 PCF 응답엔 직접 없음 → Monarch get_disease_info 로 후속 보강
        disease_name = item.get("disease_name") or item.get("orpha_disease_name_en") or ""

        results.append({
            # 확정 스펙 필드 (그대로 유지)
            "disease_id":   disease_id,
            "disease_name": disease_name,
            "score":        float(item.get("score", 0)),
            "pmid_list":    pmid_list,
            "hpo_list":     hpo_list,
            # 추가 정보 (호환성 + enrich 용)
            "orpha_id":     disease_id,
            "name":         disease_name,
            "genes":        genes,
            "rank":         item.get("rank"),
            "matched_hpo":  matched_hpo,
            "description":  item.get("description", "")[:300] if item.get("description") else "",
            "orpha_url":    item.get("orpha_url", ""),
        })

    print(f"  [PubCaseFinder] HPO {len(hpo_ids)}개 → 상위 {len(results)}개 질환 매칭")
    if results:
        _save_cache(ck, results)
    return results


def enrich_pcf_results(pcf_results: list, fetch_pmids: bool = True) -> list:
    """
    확정 문서 §2.2 — PubCaseFinder 결과에 disease_name + pmid_list 보강

    PCF API 응답에는 id, score, matched_hpo_id, gene_id 만 있고
    disease_name과 pmid_list는 직접 반환되지 않는다.
    이 함수는:
      - disease_name : Monarch /entity/{disease_id} → name
      - pmid_list    : PubMed eSearch (disease_name + "case reports") → idlist[:3]
    를 호출하여 확정 스펙 §2.2 필드를 모두 채운다.

    Parameters
    ----------
    pcf_results : list[dict]   get_ranked_diseases() 반환값
    fetch_pmids : bool         PubMed eSearch 호출 여부 (False면 disease_name만 채움)
    """
    try:
        from rag.monarch_fetcher import get_disease_info
    except ImportError:
        get_disease_info = None

    enriched = []
    for r in pcf_results:
        new_r = dict(r)
        disease_id = new_r.get("disease_id", "")

        # 1) disease_name 보강 (Monarch)
        if not new_r.get("disease_name") and disease_id and get_disease_info:
            try:
                info = get_disease_info(disease_id)
                if info.get("name"):
                    new_r["disease_name"] = info["name"]
                    new_r["name"] = info["name"]
                if info.get("description") and not new_r.get("description"):
                    new_r["description"] = info["description"][:300]
            except Exception:
                pass

        # 2) pmid_list 보강 (PubMed eSearch — 확정 §2.2 term: 질환명 + case reports)
        if fetch_pmids and not new_r.get("pmid_list") and new_r.get("disease_name"):
            try:
                from rag.pubmed_fetcher import PubMedFetcher
                fetcher = PubMedFetcher()
                pmids = fetcher._search_pmids(new_r["disease_name"], max_results=3)
                new_r["pmid_list"] = pmids
            except Exception:
                pass

        enriched.append(new_r)
    return enriched


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
