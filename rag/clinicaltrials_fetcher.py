"""
ClinicalTrials.gov API — 희귀질환 관련 현재 모집 중인 임상시험 검색
NIH 공식 API v2: https://clinicaltrials.gov/api/v2/studies
무료, 등록 불필요

회의 확정 (2026-04-27):
  - 희귀질환 환자에게 "현재 이 질환으로 모집 중인 임상시험이 있습니다" 정보 제공
  - 리포트 "5. 다음 단계 임상 권고" 섹션 강화
"""
import time
import requests

CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"
TIMEOUT   = 15
DELAY     = 0.3  # rate limit 준수


def get_clinical_trials(disease_name: str, top_k: int = 3) -> list:
    """
    질환명으로 현재 모집 중인 임상시험 검색

    Parameters
    ----------
    disease_name : str   질환명 (예: "Lymphangioleiomyomatosis")
    top_k        : int   반환할 최대 임상시험 수

    Returns
    -------
    list[dict]
        [
            {
                "nct_id":    "NCT05XXXXXX",
                "title":     "임상시험 제목",
                "status":    "RECRUITING",
                "phase":     "Phase 2",
                "condition": "Lymphangioleiomyomatosis",
                "locations": ["서울대병원", "Mayo Clinic"],
                "url":       "https://clinicaltrials.gov/study/NCT05XXXXXX"
            },
            ...
        ]
    """
    params = {
        "query.cond":  disease_name,
        "filter.overallStatus": "RECRUITING",
        "pageSize":    top_k,
        "format":      "json",
        "fields":      "NCTId,BriefTitle,OverallStatus,Phase,Condition,LocationFacility",
    }

    try:
        resp = requests.get(CTGOV_URL, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.Timeout:
        print(f"  ⚠️ ClinicalTrials.gov 타임아웃 ({TIMEOUT}초)")
        return []
    except requests.exceptions.ConnectionError:
        print("  ⚠️ ClinicalTrials.gov 연결 실패")
        return []
    except Exception as e:
        print(f"  ⚠️ ClinicalTrials.gov 오류: {e}")
        return []

    studies = data.get("studies", [])
    results = []

    for s in studies[:top_k]:
        proto = s.get("protocolSection", {})
        id_mod      = proto.get("identificationModule", {})
        status_mod  = proto.get("statusModule", {})
        design_mod  = proto.get("designModule", {})
        cond_mod    = proto.get("conditionsModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})

        nct_id = id_mod.get("nctId", "")
        title  = id_mod.get("briefTitle", "")
        status = status_mod.get("overallStatus", "")
        phases = design_mod.get("phases", [])
        phase  = phases[0] if phases else "N/A"
        conditions = cond_mod.get("conditions", [])
        condition  = conditions[0] if conditions else disease_name

        # 기관 목록 (최대 3개)
        locations = []
        for loc in contacts_mod.get("locations", [])[:3]:
            facility = loc.get("facility", "")
            if facility:
                locations.append(facility)

        results.append({
            "nct_id":    nct_id,
            "title":     title,
            "status":    status,
            "phase":     phase,
            "condition": condition,
            "locations": locations,
            "url":       f"https://clinicaltrials.gov/study/{nct_id}",
        })

    time.sleep(DELAY)
    print(f"  [ClinicalTrials.gov] '{disease_name}' → 모집 중 임상시험 {len(results)}건")
    return results


def format_trials_for_llm(trials: list, disease_name: str = "") -> str:
    """
    임상시험 결과 → LLM 주입용 컨텍스트 블록

    Parameters
    ----------
    trials       : list[dict]   get_clinical_trials() 반환값
    disease_name : str          질환명 (헤더용)

    Returns
    -------
    str   LLM 프롬프트 삽입용 컨텍스트
    """
    if not trials:
        return ""

    lines = [f"【ClinicalTrials.gov — 현재 모집 중인 임상시험 ({len(trials)}건)】"]
    if disease_name:
        lines.append(f"질환: {disease_name}")
    lines.append("")

    for i, t in enumerate(trials, 1):
        loc_str = ", ".join(t["locations"]) if t["locations"] else "기관 정보 없음"
        lines.append(
            f"{i}. {t['title']}\n"
            f"   NCT ID: {t['nct_id']} | 단계: {t['phase']} | 상태: {t['status']}\n"
            f"   기관: {loc_str}\n"
            f"   URL: {t['url']}"
        )

    lines.append("\n→ 담당 의사에게 임상시험 참여 가능 여부 문의 권고")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== ClinicalTrials.gov API 테스트 ===\n")

    test_disease = "Lymphangioleiomyomatosis"
    print(f"질환: {test_disease}")
    trials = get_clinical_trials(test_disease, top_k=3)

    if trials:
        print(f"\n--- 모집 중 임상시험 {len(trials)}건 ---")
        for t in trials:
            print(f"  [{t['phase']}] {t['title']}")
            print(f"  {t['nct_id']} | {t['url']}")
            if t["locations"]:
                print(f"  기관: {', '.join(t['locations'])}")

        print("\n--- LLM 주입용 컨텍스트 ---")
        print(format_trials_for_llm(trials, test_disease))
    else:
        print("결과 없음")
