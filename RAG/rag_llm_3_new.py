import os
import json
import datetime
import asyncio
import aiohttp
import pandas as pd
import xml.etree.ElementTree as ET
import boto3
import re
import requests
import traceback

# =====================================================================
# 0. PubMed Verifier (이식된 코드)
# =====================================================================
PUBMED_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

def verify_pmids(text: str, verbose: bool = True) -> dict:
    """
    소견서 텍스트에서 PMID를 추출하고 PubMed에서 실제 존재 여부 확인
    """
    # "PMID: 12345678" 또는 "PMID 12345678" 패턴 추출 (7~9자리)
    pmids = list(set(re.findall(r"PMID[:\s]*(\d{7,9})", text)))

    if not pmids:
        if verbose:
            print("  ℹ️  소견서에 PMID 없음 (인용 없음)")
        return {"total": 0, "valid": 0, "invalid": [], "valid_list": [], "rate": None}

    if verbose:
        print(f"  [PMID 체크] 발견된 PMID {len(pmids)}개: {pmids}")

    valid, invalid = [], []
    for pmid in pmids:
        try:
            resp = requests.get(
                PUBMED_ESUMMARY,
                params={"db": "pubmed", "id": pmid, "retmode": "json"},
                timeout=5,
            )
            data = resp.json()
            # PubMed API: 실제 존재하면 result[pmid]["uid"] 에 pmid가 들어옴
            entry = data.get("result", {}).get(pmid, {})
            uid = entry.get("uid", "")
            if uid == pmid and "error" not in entry:
                valid.append(pmid)
                if verbose:
                    title = data["result"][pmid].get("title", "")[:60]
                    print(f"    ✅ {pmid} — {title}...")
            else:
                invalid.append(pmid)
                if verbose:
                    print(f"    ❌ {pmid} — PubMed에 없음 (환각)")
        except Exception as e:
            invalid.append(pmid)
            if verbose:
                print(f"    ⚠️ {pmid} — 확인 실패: {e}")

    rate = len(valid) / len(pmids) if pmids else None

    if verbose:
        print(f"\n  결과: 유효 {len(valid)}/{len(pmids)} (유효율 {rate:.0%})")
        if invalid:
            print(f"  ❌ 가짜 PMID: {invalid}")

    return {
        "total":      len(pmids),
        "valid":      len(valid),
        "invalid":    invalid,
        "valid_list": valid,
        "rate":       rate,
    }

# =====================================================================
# 1. 시스템 프롬프트 정의
# =====================================================================
SYSTEM_PROMPT = """You are an elite AI diagnostician specializing in pulmonary and rare diseases.
You synthesize multimodal patient data and RAG-retrieved evidence to generate a professional diagnostic support report.
Your role is to support physician decision-making, not to make final diagnoses.
Write your final report clearly, logically, and professionally.
Ensure your final output is written in Korean as requested by the clinical team.

[strict rules]
1. Evidence-Based Reasoning: Do not make assumptions or definitive conclusions without objective evidence. All claims must be grounded in the provided RAG data. If data from OMIM and Orphanet differ, you must note this discrepancy.
2. Final Ranking Strategy: You will receive up to 6 candidates (Local Top 3 + Global API Top 3). You must independently evaluate all 6 based on HPO-matching scores from PubCaseFinder (Orphanet/OMIM/Gene targets) and internal DB, Lab findings, and PubMed evidence to select the final Top 1~3 for the report.
3. Hybrid Discovery Logic:
   - Priority 1: Candidates cross-validated by both Local DB and Global API.
   - Priority 2: Candidates found ONLY in Global API but showing high HPO similarity and specific clinical fit.
   - Priority 3: Candidates found ONLY in Local DB with strong internal evidence.
4. MDT Mandatory: If any rare disease (OrphaCode) is included in the final Top 3, an MDT (Multidisciplinary Team) referral is mandatory regardless of its ranking.
5. Cross-Validation Marking: In the 'rag_evidence' field, you MUST explicitly state whether each diagnosis was found in both sources ("DB·API 교차검증 일치") or discovered newly via Global API ("글로벌 API 신규 발굴").
6. Differential Diagnosis: Use the patient's negative findings and mismatched Lab values to logically explain why certain candidate diseases (especially those in Top 2~3) should be ruled out or considered less likely than Top 1.
7. Case Comparison: Conduct a deep-dive comparison between the patient's current presentation and the retrieved PubMed case reports. Note specific similarities in genomic markers or atypical symptoms.
8. Actionable Alternatives: Synthesize the Clinical Trials data to recommend specific, recruiting clinical trial opportunities that match the patient's condition and location (if applicable).

[Output Format Rules]
Output must strictly follow the JSON structure below. Do not include any text outside this JSON.
{
  "recommendation": {
    "immediate_workup": ["Examination / procedure items"],
    "specialist_referral": ["Referral recommendation (with source and rationale)"],
    "treatment_guideline": [
      "[Disease 1] Treatment guideline + PMID",
      "[Disease 2] Treatment guideline + PMID",
      "[Disease 3] Treatment guideline + PMID"
    ],
    "clinical_trial_info": ["Specific recruiting trials (NCT ID + Title) matching patient's condition"],
    "genetic_test": ["Target genes + (복수 소스 확인) if applicable"],
    "additional_lab": ["Additional lab recommendations"]
  },
  "clinical_notes": {
    "summary": "Comprehensive summary of chief complaint and AI analysis (include age, sex, chief complaint; exclude MRN)",
    "top1_reasoning": "Clinical rationale for Top 1 disease (use Positive HPO + Negative HPO + Lab findings)",
    "differential_note": "Rule out Top 2~3 by citing the absence of symptoms (Negative HPO) that would otherwise be expected. Rare disease flags must be included regardless of probability",
    "rag_evidence": "Key clinical evidence from RAG results (cite sources). Include internal DB vs external API cross-validation results",
    "case_comparison": "Comparison of current patient with PubMed case reports (similarities, differences, implications)",
    "epidemiology_note": "Orphanet prevalence, age of onset, inheritance pattern (note DB vs API agreement). Empty string for common diseases",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단과 추가 검사 결과를 종합하여 확정합니다."
  },
  "confidence_metrics": {
    "overall_confidence_score": 0.0,
    "rationale": "Score reasoning (e.g., 'High confidence due to consistent PMID evidence and DB match')",
    "data_sufficiency": {
      "genomic_evidence": "High/Medium/Low",
      "clinical_case_match": "High/Medium/Low",
      "trial_availability": "High/Medium/Low"
    }
  }
}

[작성 원칙]
1. summary: Include age, sex, chief complaint, and key abnormal Lab values. Never include MRN number.
2. top1_reasoning: Explicitly mention all Positive HPO, Negative HPO, and abnormal Lab values with specific formatting.
3. rag_evidence must cite the following data if present in the RAG JSON:
   - genes_from_orphadata: State gene name and association_type explicitly.
   - phenotypes_from_orphadata: Cite at least 2 HPO terms with frequency "Very frequent" or "Frequent".
   - epidemiology.prevalence: Include prevalence value or range.
   - epidemiology.age_of_onset: Compare onset age with patient's age.
   - causal_genes (Monarch): Note agreement or disagreement with Orphadata genes.
   - Matched items (internal DB + external API): Mark as "DB·API 교차검증 일치".
   - Mismatched items: Mark as "DB·API 불일치 — 추가 확인 필요".
4. treatment_guideline: Use the provided [PubMed 가이드라인/리뷰] data to formulate the treatment plan for each of the Top 3 diseases.
   - Prefix each entry with [Disease Name].
   - Sort by priority.
   - CRITICAL: You MUST append the exact PMID as the source (e.g., ""). If no specific guideline data is available in the prompt, explicitly state "(출처: General medical consensus)".
5. If a rare disease (OrphaCode) is within Top 3, specialist_referral must include MDT referral.
6. differential_note: Describe Top 2~3 differential considerations. Always include rare disease flagged items even if probability is low.
7. genetic_test: Must include genes where association_type contains "Disease-causing". If Monarch and Orphadata agree, append "(복수 소스 확인)".
8. case_comparison: Describe similarities, differences, and genetic testing implications between the provided [PubMed 케이스리포트] and this patient.
9. epidemiology_note: Empty string for common diseases. For rare diseases, describe prevalence, age of onset, and inheritance pattern based on Orphanet data. Interpret various prevalence formats (e.g., "1-9 / 100,000", "0.01%", "Rare") into a standardized clinical context in Korean.
10. clinical_trial_info: Must include specific NCT IDs and study titles retrieved from the RAG data. If no relevant trials are found, explicitly state "현재 모집 중인 적합한 임상시험이 없음" (No suitable clinical trials currently recruiting).
11. confidence_metrics: Generate a score between 0.0 and 1.0.
    - 0.9~1.0: Internal DB and external API (Monarch, PubMed) are perfectly consistent and match the patient's HPO.
    - 0.7~0.8: Most sources agree, but some metadata (prevalence or onset) is missing.
    - 0.5~0.6: Recommended treatment/diagnosis relies heavily on General medical consensus due to lack of specific RAG data.
    - Below 0.5: Significant mismatch between DB and API, or retrieved PubMed articles are not directly relevant.
    - rationale: Briefly explain why this score was assigned in Korean.
12. disclaimer fixed phrase must always be included. Do not modify.
"""

# =====================================================================
# 2. 하이브리드 듀얼 RAG 클래스 정의
# =====================================================================
class RareLinkHybridDualRAG:
    def __init__(self, orphadata_csv_path, hpo_dict_path="hpo_official.json", api_key=None):
        self.pubcasefinder_url = "https://pubcasefinder.dbcls.jp/api/get_ranked_list"
        self.pubcasefinder_record_url = "https://pubcasefinder.dbcls.jp/api/get_data_record"
        self.api_key = api_key
        self.pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.pubmed_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.monarch_url = "https://api-v3.monarchinitiative.org/v3/api/entity/"
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"
        
        # 🛡️ 수정 1: PubMed 속도 제한을 더 엄격히 관리 (세마포어 강화)
        self.pubmed_semaphore = asyncio.Semaphore(1) # 동시 요청 1개로 제한하여 서버 부하 방지
        self.local_db = None
        if os.path.exists(orphadata_csv_path):
            self.local_db = pd.read_csv(orphadata_csv_path)
            
        self.hpo_dict = {}
        if os.path.exists(hpo_dict_path):
            try:
                with open(hpo_dict_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # OBO Graph 구조(hpo_official.json)인지 확인
                if isinstance(data, dict) and 'graphs' in data:
                    print(f"⚙️ OBO Graph 구조에서 HPO 데이터 추출 중: {hpo_dict_path}")
                    extracted_dict = {}
                    for node in data['graphs'][0].get('nodes', []):
                        node_id = node.get('id', '')
                        if node_id.startswith('http://purl.obolibrary.org/obo/HP_'):
                            hpo_code = node_id.split('/')[-1].replace('_', ':')
                            term_name = node.get('lbl', 'Unknown')
                            extracted_dict[hpo_code] = term_name
                    self.hpo_dict = extracted_dict
                else:
                    # 기존 flat map 방식 (hpo_whitelist.json)
                    self.hpo_dict = data
                
                print(f"✅ HPO 사전 로드 완료 (총 {len(self.hpo_dict)}개 용어)")
            except Exception as e:
                print(f"❌ HPO 사전 로딩 중 오류 발생: {e}")
        else:
            # 기본 파일이 없을 경우 hpo_whitelist.json 시도
            fallback_path = "hpo_whitelist.json"
            if hpo_dict_path == "hpo_official.json" and os.path.exists(fallback_path):
                print(f"ℹ️ {hpo_dict_path}가 없어 {fallback_path}로 대체 로드합니다.")
                with open(fallback_path, "r", encoding="utf-8") as f:
                    self.hpo_dict = json.load(f)
                print(f"✅ HPO 사전 로드 완료 (총 {len(self.hpo_dict)}개 용어)")
            else:
                print(f"⚠️ HPO 사전 파일({hpo_dict_path})이 없습니다. 코드만 출력됩니다.")

    def _log_error(self, method_name, input_id, error):
        import traceback
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "image_path": input_id,
            "error_message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)) if isinstance(error, Exception) else traceback.format_exc()
        }
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"error_{method_name}_{timestamp_str}.json"
        log_path = os.path.join(log_dir, log_filename)
        
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        print(f"Error logged to {log_path}")

    def format_hpo_list(self, hpo_list):
        formatted_list = []
        for hpo_code in hpo_list:
            hpo_name = self.hpo_dict.get(hpo_code, "Unknown")
            formatted_list.append(f"{hpo_code} ({hpo_name})")
        return formatted_list

    async def fetch_global_top_n(self, session, hpo_list, top_k=3):
        hpo_query = ",".join(hpo_list)
        params = {
            "target": "orphanet",
            "format": "json",
            "phenotype": hpo_query
        } 
        try:
            async with session.get(self.pubcasefinder_url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    global_candidates = []
                    for item in data[:top_k]:
                        # PubCaseFinder API key changed: name -> orpha_disease_name_en
                        pcf_name = item.get("orpha_disease_name_en") or item.get("omim_disease_name_en") or item.get("name")
                        global_candidates.append({
                            "id": item.get("id"),
                            "name": pcf_name,
                            "source": "Global API (PubCaseFinder)"
                        })
                    return global_candidates
        except Exception as e:
            self._log_error("fetch_global_top_n", hpo_query, e)
        return []

    async def fetch_monarch(self, session, disease_id, original_name):
        try:
            async def search_monarch_entity(query_term):
                if not query_term: return None
                search_url = "https://api-v3.monarchinitiative.org/v3/api/search"
                params = {"q": query_term, "category": "biolink:Disease", "limit": 1}
                async with session.get(search_url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("items", [])
                        return items[0].get("id") if items else None
                return None

            search_query = disease_id.replace("ORPHA:", "Orphanet:")
            m_id = await search_monarch_entity(search_query)
            
            if not m_id and original_name:
                m_id = await search_monarch_entity(original_name)

            if not m_id:
                return {"id": disease_id, "name": original_name, "description": "N/A", "synonyms": "N/A", "monarch_genes": "N/A"}

            entity_url = f"https://api-v3.monarchinitiative.org/v3/api/entity/{m_id}"
            genes = []
            description = "No definition available."
            synonyms_str = "None"

            async with session.get(entity_url, timeout=10) as ent_res:
                if ent_res.status == 200:
                    ent_data = await ent_res.json()
                    description = ent_data.get("description") or description
                    synonyms_list = ent_data.get("synonym", [])
                    synonyms_str = ", ".join(synonyms_list[:3]) if synonyms_list else "None"
                    causal_list = ent_data.get("causal_gene", [])
                    if causal_list:
                        for gene_node in causal_list:
                            gene_name = gene_node.get("symbol") or gene_node.get("name")
                            if gene_name and gene_name not in genes:
                                genes.append(gene_name)

            if not genes:
                import re
                potential = re.findall(r'\b[A-Z][A-Z0-9]{2,7}\b', description)
                exclude = ["TAAD", "FTAAD", "PMID", "HPO", "DNA", "RNA", "AMI", "MDT", "JSON", "RAG"]
                for g in potential:
                    if g not in exclude and g not in genes:
                        genes.append(g)

            return {
                "id": disease_id,
                "name": ent_data.get("label") if 'ent_data' in locals() else original_name,
                "description": description,
                "synonyms": synonyms_str,
                "monarch_genes": ", ".join(genes[:5]) if genes else "No linked genes found"
            }
        except Exception as e:
            self._log_error("fetch_monarch", f"{disease_id} | {original_name}", e)
            return {"id": disease_id, "name": original_name, "description": "N/A", "monarch_genes": "N/A"}


    async def fetch_pubmed_cases(self, session, disease_name, top_k=2):
        if not disease_name or "Unknown" in disease_name: return "No relevant case reports found."
        filter_query = "(case reports[Filter])"
        return await self._execute_pubmed_search(session, disease_name, filter_query, top_k)

    def _normalize_disease_name(self, name):
        if not name: return ""
        parts = [p.strip() for p in name.split(',')]
        if len(parts) > 1:
            name = " ".join(reversed(parts))
        return name.replace(';', '').strip()

    async def fetch_pubmed_guidelines(self, session, disease_name, top_k=2):
        if not disease_name or "Unknown" in disease_name: return "No relevant guidelines found."
        filter_query = '("Practice Guideline"[PT] OR "Review"[PT] OR "Meta-Analysis"[PT] OR "Systematic Review"[PT])'
        return await self._execute_pubmed_search(session, disease_name, filter_query, top_k)

    async def _execute_pubmed_search(self, session, disease_name, filter_query, top_k):
        short_name = disease_name.split(';')[0].split(',')[0].strip()
        core_phrase = short_name.split(' and ')[0].split(' with ')[0].strip()
        
        search_variants = [
            f'"{disease_name}"',
            disease_name.replace('"', ''),
            self._normalize_disease_name(disease_name),
            short_name
        ]
        
        if len(short_name.split()) > 3:
            search_variants.append(core_phrase)

        for variant in search_variants:            
            query = f"({variant}[Title/Abstract]) AND {filter_query}"
            params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": top_k}
            if self.api_key: params["api_key"] = self.api_key
            
            async with self.pubmed_semaphore:
                # 🛡️ 수정 2: 요청 간 고정 지연 추가 (0.5초) - 서버 부하 방지 및 Rate Limit 준수
                await asyncio.sleep(0.5)
                
                for attempt in range(3): # 시도 횟수 3회로 증가
                    try:
                        async with session.get(self.pubmed_url, params=params, timeout=10) as response:
                            if response.status == 200:
                                data = await response.json()
                                pmids = data.get("esearchresult", {}).get("idlist", [])
                                if pmids:
                                    return await self._fetch_abstracts_by_pmids(session, pmids)
                                break
                            elif response.status == 429:
                                await asyncio.sleep(1.5 * (attempt + 1))
                                continue
                            else:
                                break
                    except Exception as e:
                        print(f"⚠️ PubMed 시도 중 에러 ({variant}, 시도 {attempt+1}): {e}")
                        await asyncio.sleep(1)
                        continue
        return "No relevant articles found."
    
    async def _fetch_abstracts_by_pmids(self, session, pmids):
        if not pmids: return "No PMIDs provided."
        
        # 🛡️ 수정 3: efetch 시에도 지연 추가
        await asyncio.sleep(0.3)
        
        fetch_params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}
        if self.api_key: fetch_params["api_key"] = self.api_key
        
        async with session.get(self.pubmed_fetch_url, params=fetch_params, timeout=12) as fetch_res:
            content = await fetch_res.read()
            if not content or b"<?xml" not in content or b"ERROR" in content:
                return "No valid XML content found in PubMed (Server busy or empty)."
            try:
                root = ET.fromstring(content)
                abstracts = []
                for article in root.findall('.//PubmedArticle'):
                    pmid_elem = article.find('.//PMID')
                    title_elem = article.find('.//ArticleTitle')
                    if pmid_elem is None or title_elem is None: continue
                    pmid = pmid_elem.text
                    title = ''.join(title_elem.itertext())
                    abs_elem = article.find('.//AbstractText')
                    abs_txt = ''.join(abs_elem.itertext())[:800] + "..." if abs_elem is not None else "No abstract"
                    abstracts.append(f"[PMID: {pmid}] {title}\nAbstract: {abs_txt}")
                return "\n\n".join(abstracts) if abstracts else "No abstracts found in XML."
            except Exception as e:
                return f"Failed to parse PubMed XML response: {e}"
    
    async def fetch_pcf_disease_data(self, session, disease_id):
        if "ORPHA" in disease_id.upper():
            target = "orphanet"
            clean_id = disease_id.split(":")[-1]
        elif "OMIM" in disease_id.upper():
            target = "omim"
            clean_id = disease_id.split(":")[-1]
        else:
            return []

        url = self.pubcasefinder_record_url
        params = {"target": target, "id": clean_id}
        try:
            async with session.get(url, params=params, timeout=5) as res:
                if res.status == 200:
                    data = await res.json()
                    target_key = "ORPHA" if target == "orphanet" else target.upper()
                    key = f"{target_key}:{clean_id}"
                    content = data.get(key, {})
                    return content.get("hgnc_gene_symbol", [])
        except Exception as e:
            self._log_error("fetch_pcf_disease_data", disease_id, e)
        return []

    async def fetch_clinicaltrials(self, session, disease_name, top_k=3):
        if not disease_name or "Name Fetch Failed" in disease_name:
            return "No currently recruiting clinical trials found."
        params = {"query.term": disease_name, "filter.overallStatus": "RECRUITING", "pageSize": top_k}
        try:
            async with session.get(self.clinical_trials_url, params=params, timeout=5) as response:
                if response.status == 200:
                    data = await response.json()
                    studies = data.get("studies", [])
                    if not studies: return "No currently recruiting clinical trials found."
                    trials_info = []
                    for study in studies:
                        ident = study.get("protocolSection", {}).get("identificationModule", {})
                        trials_info.append(f"- [{ident.get('nctId', 'NCT_UNKNOWN')}] {ident.get('briefTitle', 'No Title')}")
                    return "\n".join(trials_info)
        except Exception as e:
            self._log_error("fetch_clinicaltrials", disease_name, e)
            return "Failed to load clinical trial information."

    async def fetch_pubcasefinder(self, session, hpo_list, top_k=2):
        hpo_query = ",".join(hpo_list)
        targets = ["orphanet", "omim", "gene"]
        
        async def fetch_target(target):
            params = {"target": target, "format": "json", "phenotype": hpo_query}
            # 타겟별로 타임아웃 차별화 (gene은 더 길게)
            timeout_sec = 30 if target == "gene" else 20
            
            for attempt in range(2): # 최대 2회 시도
                try:
                    async with session.get(self.pubcasefinder_url, params=params, timeout=timeout_sec) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 429: # Rate Limit
                            await asyncio.sleep(2 * (attempt + 1))
                            continue
                        return None
                except Exception as e:
                    if attempt == 0: # 첫 번째 실패 시 잠시 대기 후 재시도
                        await asyncio.sleep(1)
                        continue
                    return e

        tasks = [fetch_target(t) for t in targets]
        try:
            responses = await asyncio.gather(*tasks)
            combined_results = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    self._log_error(f"fetch_pubcasefinder_{targets[i]}", hpo_query, response)
                    continue
                if not response: continue
                
                target_name = targets[i].upper()
                for item in response[:top_k]:
                    d_id = item.get("id", "UNKNOWN")
                    score = item.get("score", "0.0")
                    combined_results.append(f"[{target_name}] ID: {d_id} (Score: {score})")
            return "\n".join(combined_results) if combined_results else "No relevant data found."
        except Exception as e:
            self._log_error("fetch_pubcasefinder_gather", hpo_query, e)
            return f"Failed to load PubCaseFinder data: {e}"

    async def gather_rag_data(self, pos_hpos, local_top_3):
        connector = aiohttp.TCPConnector(ssl=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            global_top_3 = await self.fetch_global_top_n(session, pos_hpos, top_k=3)

            # 🛡️ 수정 5: 로컬 데이터와 글로벌 데이터를 병합할 때 데이터 손실 방지
            combined_dict = {d['id']: d for d in local_top_3}
            for g_dis in global_top_3:
                g_id = g_dis['id']
                if g_id in combined_dict:
                    # 이름이 없는 경우 로컬 이름 유지
                    if not g_dis.get('name') and combined_dict[g_id].get('name'):
                        g_dis['name'] = combined_dict[g_id]['name']
                    combined_dict[g_id].update(g_dis)
                else:
                    combined_dict[g_id] = g_dis

            combined_candidates = list(combined_dict.values())
            num_candidates = len(combined_candidates)

            # 🛡️ 수정 6: PubCaseFinder 결과 캐싱 (중복 호출 방지)
            pcf_cache = None
            if any(c.get('id', '').startswith("ORPHA") for c in combined_candidates):
                pcf_cache = await self.fetch_pubcasefinder(session, pos_hpos)

            rag_results = []
            # 🛡️ 수정 4: 질환별 RAG 수집을 순차적으로 수행 (병렬성 제어하여 PubMed 부하 감소)
            for disease in combined_candidates:
                d_id = disease.get("id", "")
                is_rare = d_id.startswith("ORPHA")
                original_name = disease.get("name", d_id)

                # PubMed 요청 수 조절
                pubmed_k = 1 if num_candidates > 3 else 2

                # 비-PubMed 태스크 (병렬 가능)
                non_pubmed_tasks = {
                    "monarch": self.fetch_monarch(session, d_id, original_name),
                    "clinical_trials": self.fetch_clinicaltrials(session, original_name),
                    "pcf_genes": self.fetch_pcf_disease_data(session, d_id)
                }

                non_pubmed_results = await asyncio.gather(*non_pubmed_tasks.values(), return_exceptions=True)
                non_pubmed_dict = dict(zip(non_pubmed_tasks.keys(), non_pubmed_results))

                # 캐시된 PubCaseFinder 결과 사용
                if is_rare:
                    non_pubmed_dict["pubcasefinder"] = pcf_cache

                # PubMed 태스크 (순차적 수행 유도)

                pubmed_cases = await self.fetch_pubmed_cases(session, original_name, top_k=pubmed_k)
                pubmed_guidelines = await self.fetch_pubmed_guidelines(session, original_name, top_k=pubmed_k)
                
                disease_data = {
                    "id": d_id, "name": original_name, "is_rare": is_rare,
                    "source": disease.get("source", "Internal DB"),
                    "orphanet_genes": disease.get("orphanet_genes", "N/A"),
                    "orphanet_hpo": disease.get("orphanet_hpo", "N/A"),
                    "orphanet_prev": disease.get("orphanet_prev", "N/A"),
                    "orphanet_age": disease.get("orphanet_age", "N/A"),
                    "pubmed_cases": pubmed_cases,
                    "pubmed_guidelines": pubmed_guidelines
                }
                
                for key, result in non_pubmed_dict.items():
                    disease_data[key] = result if not isinstance(result, Exception) else "Error fetching data"
                
                pcf_genes_list = non_pubmed_dict.get("pcf_genes", [])
                disease_data["pcf_genes"] = ", ".join(pcf_genes_list) if isinstance(pcf_genes_list, list) else "N/A"
                
                def normalize_genes(gene_str):
                    if not gene_str or gene_str == "N/A" or "found" in str(gene_str).lower(): return set()
                    import re
                    if isinstance(gene_str, list): gene_str = " ".join(map(str, gene_str))
                    return {g.strip().upper() for g in re.split(r'[,\s;]+', str(gene_str)) if g.strip()}

                internal_gene_set = normalize_genes(disease.get("internal_genes", "N/A"))
                monarch_gene_set = normalize_genes(disease_data.get("monarch", {}).get("monarch_genes", "N/A"))
                pcf_gene_set = normalize_genes(pcf_genes_list)

                if internal_gene_set & (monarch_gene_set | pcf_gene_set):
                    disease_data["cross_validation"] = "DB·API 교차검증 일치"
                else:
                    disease_data["cross_validation"] = "DB·API 불일치 — 추가 확인 필요"
                
                if disease_data["source"] != "Internal DB" and disease_data["is_rare"]:
                    monarch_info = disease_data.get("monarch", {})
                    disease_data["orphanet_genes"] = f"{monarch_info.get('monarch_genes', 'N/A')} (글로벌 API 신규 발굴)"
                rag_results.append(disease_data)
        return rag_results

    def build_prompt(self, patient_input, rag_results):
        mapped_pos_hpos = "\n".join(self.format_hpo_list(patient_input.get('pos_hpos', [])))
        mapped_neg_hpos = "\n".join(self.format_hpo_list(patient_input.get('neg_hpos', [])))
        prompt = f"""You are analyzing a patient case using multimodal clinical data and
RAG-retrieved evidence from both internal DB and external APIs.
Based on the structured data below, generate a comprehensive diagnostic
support report following the output format specified in the system prompt.

=========================================
=== 1. 환자 기본정보 ===
{{
  "name": "{patient_input.get('name', 'N/A')}",
  "age": {patient_input.get('age', 0)},
  "sex": "{patient_input.get('sex', 'N/A')}",
  "visit_date": "{patient_input.get('visit_date', 'N/A')}",
  "visit_type": "{patient_input.get('visit_type', 'N/A')}",
  "chief_complaint": "{patient_input.get('chief_complaint', 'N/A')}",
  "allergy": "{patient_input.get('allergy', 'N/A')}"
}}
=== 2. 증상 원문 ===
- Positive Findings (양성 증상): {patient_input.get('symptoms_raw', 'N/A')}
- Negative Findings (음성 소견): {patient_input.get('negative_raw', 'N/A')}

=== 3. HPO 프로파일 ===
[Positive HPO]
{mapped_pos_hpos}

[Negative HPO (증상에서만)]
{mapped_neg_hpos}

=== 4. Lab 수치 ===
{patient_input.get('lab_data', 'N/A')}

=== 5. 일반/기타 폐질환 랭킹 Top 10 (로컬 DB 기반) ===
{patient_input.get('ranking_general', 'N/A')}

=== 6. 희귀폐질환 리스팅 (로컬 DB 기반) ===
{patient_input.get('ranking_rare', '해당 없음')}

=== 7. 내부 DB 정보 — Top 3 교차검증용 ===
{patient_input.get('internal_db_context', 'N/A')}
※ 외부 API 결과와 대조하여 일치/불일치 여부를 rag_evidence에 반드시 명시할 것.

=== 8. RAG 검색 결과 (외부 API) ===
"""
        for idx, res in enumerate(rag_results):
            prompt += f"\n--- Top {idx+1}: {res.get('name')} ({res.get('id')}) ---\n"
            if res.get('is_rare'):
                prompt += f"[Orphanet]\n- 유전자: {res.get('orphanet_genes')}\n- Very frequent / Frequent HPO: {res.get('orphanet_hpo')}\n- 유병률: {res.get('orphanet_prev')}\n- 발병연령: {res.get('orphanet_age')}\n"
            prompt += f"[Monarch Metadata]\n- 공식 정의: {res.get('monarch', {}).get('description', 'N/A')}\n- 동의어: {res.get('monarch', {}).get('synonyms', 'N/A')}\n- 인과 유전자 (Monarch): {res.get('monarch', {}).get('monarch_genes', 'N/A')}\n- 인과 유전자 (PubCaseFinder): {res.get('pcf_genes', 'N/A')} 💡 추가됨\n- 교차검증 결과: {res.get('cross_validation', 'N/A')}\n\n[PubMed 가이드라인/리뷰 (치료법 참고용)]\n{res.get('pubmed_guidelines', 'N/A')}\n\n[PubMed 케이스리포트 (환자 비교용)]\n{res.get('pubmed_cases', 'N/A')}\n"
            if res.get('is_rare'):
                prompt += f"\n[PubCaseFinder]\n{res.get('pubcasefinder', 'N/A')}\n"
            prompt += f"\n[ClinicalTrials (RECRUITING + COMPLETED)]\n{res.get('clinical_trials', 'N/A')}\n"
        prompt += "\n=========================================\n위 데이터를 종합하여 규정된 JSON 형식으로 출력하십시오."
        return prompt


    def run_pipeline(self, patient_input):
        try:
            top_3 = patient_input.get("top_3", [])
            rag_results = asyncio.run(self.gather_rag_data(patient_input["pos_hpos"], top_3))
            return self.build_prompt(patient_input, rag_results)
        except Exception as e:
            self._log_error("run_pipeline", patient_input.get("name", "Unknown Patient"), e)
            return None

# =====================================================================
# 테스트 실행부 (AWS Bedrock 연동)
# =====================================================================
if __name__ == "__main__":
    patient_orpha_91387 = {
        "name": "이환자", "age": 42, "sex": "M",
        "visit_date": "2026-05-06", "visit_type": "응급실",
        "chief_complaint": "갑작스럽고 찢어지는 듯한 가슴 통증과 호흡곤란 (Chest pain & Dyspnea)", "allergy": "없음",
        "symptoms_raw": "환자는 심한 흉통과 호흡곤란으로 응급실 내원함. 흉부 X-ray 및 영상 검 상 기흉(Pneumothorax)과 심비대(Cardiomegaly) 소견이 뚜렷하게 관찰됨. 문진 상 부친이 흉부 대동맥류로 수술받은 가족력(Familial occurrence)이 있음.",
        "negative_raw": "상지 부종(Upper limb edema)이나 연하곤란(Difficulty swallowing)은 관찰되지 않음. 빈혈 소견은 명확히 배제됨.",
        "pos_hpos": ["HP:0002107", "HP:0001640"],
        "neg_hpos": ["HP:0001903", "HP:0002264"],
        "lab_data": "D-dimer: 2500 ng/mL (상승 - 대동맥 박리 의심 소견), Troponin I: 정상 범위",
        "ranking_general": "1. 원발성 자발성 기흉(Primary spontaneous pneumothorax)\n2. 급성 심근경색(AMI)",
        "ranking_rare": "1. ORPHA:91387 (Familial thoracic aortic aneurysm and aortic dissection)",
        "internal_db_context": "ORPHA:91387의 원인 유전자는 ACTA2, MYH11, TGFBR1, TGFBR2, FBN1 등이 빈번하게 보고됨.",
        "top_3": [
            {
                "id": "ORPHA:91387", "name": "Familial thoracic aortic aneurysm and aortic dissection", 
                "internal_genes": "ACTA2, ELN, TGFB3, TGFBR2, SMAD3, MYH11, FBN1",
                "orphanet_genes": "ACTA2 (Disease-causing), TGFBR2 (Disease-causing)",
                "orphanet_hpo": "Pneumothorax (Frequent), Cardiomegaly (Frequent)",
                "orphanet_prev": "1-9 / 100 000", "orphanet_age": "Adult"
            },
            {"id": "OMIM:173600", "name": "PNEUMOTHORAX, PRIMARY SPONTANEOUS", "internal_genes": "N/A"},
            {"id": "OMIM:192600", "name": "CARDIOMYOPATHY, FAMILIAL HYPERTROPHIC, 1", "internal_genes": "N/A"}
        ]
    }
    
    rag_system = RareLinkHybridDualRAG("dummy.csv") 
    generated_prompt = rag_system.run_pipeline(patient_orpha_91387)
    
    print("\n" + "="*50)
    print("✅ [RAG 시스템이 조립한 최종 프롬프트]")
    print("="*50)
    print(generated_prompt)

    # AWS Bedrock 호출
    REGION_NAME = "ap-northeast-2"
    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0" 
    
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": generated_prompt}
            ],
            "temperature": 0.0
        })
        
        print(f"\n🧠 Bedrock ({MODEL_ID}) 진단 추론 시작...")
        
        response = bedrock_client.invoke_model(
            body=body,
            modelId=MODEL_ID,
            accept="application/json",
            contentType="application/json"
        )
        
        response_body = json.loads(response.get("body").read())
        final_report = response_body["content"][0]["text"]
        
        # --- PMID 검증 단계 추가 ---
        print("\n🔍 생성된 리포트의 PMID 검증 중...")
        pmid_results = verify_pmids(final_report)
        
        if pmid_results['total'] > 0:
            print(f"✅ PMID 검증 완료: 유효 {pmid_results['valid']}/{pmid_results['total']} (유효율: {pmid_results['rate']:.1%})")
            if pmid_results['invalid']:
                print(f"❌ 가짜(환각) PMID 발견: {pmid_results['invalid']}")
        else:
            print("ℹ️ 리포트 내에 인용된 PMID가 없습니다.")
        # --------------------------
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnosis_report_orpha91387_{timestamp}.json"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(final_report)
            
        print(f"\n✅ 진단 레포트(JSON)가 성공적으로 저장되었습니다: {filename}")
        
    except Exception as e:
        print(f"❌ Bedrock API 호출 에러: {e}")
