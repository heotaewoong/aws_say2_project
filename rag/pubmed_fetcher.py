"""
PubMed E-utilities API — chromadb 없이 직접 호출
NIH/NLM 공식 API: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/

질환명 → 최신 논문 Top K 반환 (제목 + abstract + PMID + URL)
"""
import time
import requests

PUBMED_SEARCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
PUBMED_FETCH_URL   = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI 권장 rate limit: API key 없이 초당 3 req
REQUEST_DELAY = 0.4
TIMEOUT       = 15


class PubMedFetcher:
    """
    PubMed E-utilities 직접 호출 (chromadb 불필요)
    질환명 → 최신 관련 논문 Top K 반환
    """

    def _search_pmids(self, disease_name: str, max_results: int = 20) -> list:
        params = {
            "db":      "pubmed",
            "term":    (
                f'"{disease_name}"[Title/Abstract] AND '
                "(treatment[Title/Abstract] OR diagnosis[Title/Abstract] "
                "OR management[Title/Abstract])"
            ),
            "retmax":  max_results,
            "sort":    "date",
            "retmode": "json",
        }
        try:
            resp = requests.get(PUBMED_SEARCH_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"⚠️ PubMed 검색 오류: {e}")
            return []

    def _fetch_summaries(self, pmid_list: list) -> list:
        if not pmid_list:
            return []
        params = {
            "db":      "pubmed",
            "id":      ",".join(pmid_list),
            "retmode": "json",
        }
        try:
            resp = requests.get(PUBMED_SUMMARY_URL, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json().get("result", {})
            papers = []
            for pmid in pmid_list:
                info = data.get(pmid, {})
                if not info or pmid == "uids":
                    continue
                papers.append({
                    "pmid":    pmid,
                    "title":   info.get("title", ""),
                    "pubdate": info.get("pubdate", ""),
                    "source":  info.get("source", ""),
                })
            return papers
        except Exception as e:
            print(f"⚠️ PubMed summary 오류: {e}")
            return []

    def _fetch_abstracts(self, pmid_list: list) -> dict:
        """abstract 텍스트 → {pmid: abstract}"""
        if not pmid_list:
            return {}
        params = {
            "db":      "pubmed",
            "id":      ",".join(pmid_list),
            "rettype": "abstract",
            "retmode": "text",
        }
        try:
            resp = requests.get(PUBMED_FETCH_URL, params=params, timeout=20)
            resp.raise_for_status()
            raw = resp.text
        except Exception as e:
            print(f"⚠️ PubMed fetch 오류: {e}")
            return {}

        # PMID 마커로 분할
        abstract_map = {}
        current_pmid, current_lines = None, []
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("PMID-") or stripped.startswith("PMID:"):
                if current_pmid and current_lines:
                    abstract_map[current_pmid] = "\n".join(current_lines).strip()
                current_pmid = stripped.split()[-1]
                current_lines = []
            else:
                current_lines.append(line)
        if current_pmid and current_lines:
            abstract_map[current_pmid] = "\n".join(current_lines).strip()

        return abstract_map

    def get_top_papers(self, disease_name: str, top_k: int = 5) -> list:
        """
        질환명 → 최신 관련 논문 Top K

        Returns
        -------
        list[dict]:
            [{"pmid", "title", "pubdate", "url", "abstract"}, ...]
        """
        print(f"  🔍 PubMed에서 '{disease_name}' 논문 검색 중...")
        pmid_list = self._search_pmids(disease_name, max_results=top_k * 2)

        if not pmid_list:
            print(f"  ⚠️ '{disease_name}' 관련 논문 없음")
            return []

        time.sleep(REQUEST_DELAY)
        summaries = self._fetch_summaries(pmid_list[:top_k])

        time.sleep(REQUEST_DELAY)
        abstract_map = self._fetch_abstracts(pmid_list[:top_k])

        results = []
        for s in summaries[:top_k]:
            results.append({
                "pmid":     s["pmid"],
                "title":    s["title"],
                "pubdate":  s["pubdate"],
                "url":      f"https://pubmed.ncbi.nlm.nih.gov/{s['pmid']}/",
                "abstract": abstract_map.get(s["pmid"], "")[:600],
            })

        print(f"  ✅ PubMed 논문 {len(results)}편 수집 완료")
        return results


if __name__ == "__main__":
    fetcher = PubMedFetcher()
    papers = fetcher.get_top_papers("Lymphangioleiomyomatosis", top_k=3)
    for p in papers:
        print(f"\n[{p['pubdate']}] {p['title']}")
        print(f"  PMID: {p['pmid']} | {p['url']}")
        print(f"  {p['abstract'][:200]}...")
