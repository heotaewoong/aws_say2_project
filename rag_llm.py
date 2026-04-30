import os
import requests
import pandas as pd
import xml.etree.ElementTree as ET
import datetime
import json
import boto3 # 💡 Bedrock 사용을 위한 Boto3 임포트

class HybridDualRAG:
    def __init__(self, orphadata_csv_path):
        # 1. API Endpoints
        self.pubcasefinder_url = "https://pubcasefinder.dbcls.jp/pcf_get_ranking_by_hpo_id"
        self.pubmed_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.monarch_url = "https://api.monarchinitiative.org/v3/api/entity/"
        self.clinical_trials_url = "https://clinicaltrials.gov/api/v2/studies"
        
        # 2. 로컬 Orphadata 로드
        self.local_db = None
        if os.path.exists(orphadata_csv_path):
            self.local_db = pd.read_csv(orphadata_csv_path)
            print(f"✅ 로컬 Orphadata 로드 완료 (총 {len(self.local_db)}개 질환)")
        else:
            print("⚠️ 로컬 Orphadata 파일이 없습니다. API 검색만 수행합니다.")
    
    # =========================================================
    # 🛡️ [Track 1] 로컬 Orphadata 탐색
    # =========================================================
    def search_local_orphadata(self, positive_hpos, negative_hpos, top_k=3):
        if self.local_db is None: return []
        print("🔍 [Track 1] 로컬 Orphadata DB 탐색 중...")
        pos_set = set(positive_hpos)
        neg_set = set(negative_hpos)
        candidates = []
        
        for _, row in self.local_db.iterrows():
            disease_id = row['disease_id']
            disease_name = row['disease_name']
            disease_hpos = set(str(row['hpo_codes']).split(';')) 
            
            if len(disease_hpos.intersection(neg_set)) > 0: continue
                
            intersection = len(disease_hpos.intersection(pos_set))
            union = len(disease_hpos.union(pos_set))
            similarity = intersection / union if union > 0 else 0.0
            
            if similarity > 0:
                candidates.append({
                    "id": disease_id, "name": disease_name, 
                    "score": round(similarity, 4), "source": "LOCAL_ORPHANET"
                })
                
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[:top_k]

    # =========================================================
    # 🌐 [Track 2] PubCaseFinder API
    # =========================================================
    def get_api_diseases_and_pmids(self, hpo_list, top_k=5):
        print("🌐 [Track 2] PubCaseFinder API 확장 검색 중...")
        hpo_query = ",".join(hpo_list)
        targets = ["omim"] 
        all_candidates = []
        pmids = set()
        
        for target_db in targets:
            params = {"target": target_db, "phenotype": hpo_query}
            headers = {"Accept": "application/json"}
            
            try:
                response = requests.get(self.pubcasefinder_url, params=params, headers=headers)
                if response.status_code == 200:
                    results = response.json()
                    for item in results[:top_k]:
                        disease_id = item.get("disease_id") or item.get("id")
                        if not disease_id: continue
                        
                        disease_name = item.get("disease_name") or item.get("disease_name_en")
                        
                        all_candidates.append({
                            "id": disease_id, 
                            "name": disease_name,
                            "score": item.get("score", 0),
                            "source": f"API_{target_db.upper()}"
                        })
                        
                        if "pmid_list" in item:
                            for pmid in item["pmid_list"][:3]: pmids.add(str(pmid))
                else:
                    print(f"⚠️ API 요청 실패 ({target_db}): HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ API 통신 에러 ({target_db}): {e}")
                
        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        return all_candidates[:top_k], list(pmids)

    # =========================================================
    # 🧬 [API 2] 질병 메타데이터 (상용화 레벨: Local DB Fallback)
    # =========================================================
    def get_disease_metadata(self, disease_id):
        # 1. 플랜 A: Monarch API (가장 풍부한 최신 설명과 이름을 가져옴)
        try:
            res = requests.get(f"{self.monarch_url}{disease_id}", timeout=3).json()
            name = res.get("name") or res.get("label") or None
            desc = res.get("description") or "No detailed description available."
            if name: return name, desc
        except Exception:
            print(f"⚠️ Monarch API 지연. 로컬 DB(Plan B)로 병명 복구 시도 중... ({disease_id})")
            
        # 2. 플랜 B: 프로덕션용 로컬 DB (orphadata.csv)에서 병명 검색
        # 하드코딩 사전을 버리고, 메모리에 올라가 있는 수만 개의 판다스(Pandas) DB에서 이름을 찾아냅니다!
        if self.local_db is not None:
            # Pandas Dataframe에서 disease_id가 일치하는 행(Row)을 검색
            matched_row = self.local_db[self.local_db['disease_id'] == disease_id]
            if not matched_row.empty:
                # 일치하는 병이 있다면 이름을 가져옵니다.
                local_disease_name = matched_row.iloc[0]['disease_name']
                return local_disease_name, "Local DB fallback (Description not available)."

        # 3. 최후의 보루: API도 죽고 로컬 DB에도 없는 초희귀 케이스
        return f"{disease_id} (Name Fetch Failed)", "No detailed description available."

    # =========================================================
    # 📚 [API 3] PubMed (증례 보고서)
    # =========================================================
    def search_pubmed_pmids_directly(self, disease_name, disease_id, top_k=3):
        print(f"🔎 PubMed 직접 검색 중... (타겟: {disease_name})")
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        
        if "(Name Fetch Failed)" in disease_name or "Unknown" in disease_name:
            clean_id = disease_id.replace(":", " ")
            query = f'"{clean_id}" AND "case reports"[Publication Type]'
        else:
            query = f'"{disease_name}"[Title/Abstract] AND "case reports"[Publication Type]'
            
        params = {"db": "pubmed", "term": query, "retmode": "json", "retmax": top_k}
        
        try:
            response = requests.get(search_url, params=params, timeout=5)
            if response.status_code == 200:
                return response.json().get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            print(f"⚠️ PubMed 검색 실패: {e}")
        return []

    def get_pubmed_abstracts(self, pmid_list):
        if not pmid_list: return "No relevant case reports found."
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pubmed", "id": ",".join(pmid_list), "retmode": "xml"}
        try:
            root = ET.fromstring(requests.get(fetch_url, params=params, timeout=10).content)
            abstracts = []
            for article in root.findall('.//PubmedArticle'):
                pmid = article.find('.//PMID').text
                title = article.find('.//ArticleTitle').text
                abs_txt = article.find('.//AbstractText')
                if abs_txt is not None:
                    abstracts.append(f"[PMID: {pmid}] {title}\nAbstract: {''.join(abs_txt.itertext())[:400]}...")
            return "\n\n".join(abstracts)
        except Exception:
            return "Failed to fetch articles."

    # =========================================================
    # 💊 [API 4] ClinicalTrials.gov
    # =========================================================
    def get_clinical_trials(self, disease_name, top_k=3):
        if not disease_name or "(Name Fetch Failed)" in disease_name:
            return "Cannot search clinical trials due to uncertain diagnosis name."
            
        print(f"💊 ClinicalTrials API 검색 중... (모집 중인 타겟: {disease_name})")
        params = {
            "query.cond": disease_name,
            "filter.overallStatus": "RECRUITING", 
            "pageSize": top_k
        }
        try:
            res = requests.get(self.clinical_trials_url, params=params, timeout=5)
            if res.status_code == 200:
                studies = res.json().get("studies", [])
                if not studies:
                    return "No currently recruiting clinical trials found."
                
                trials_info = []
                for study in studies:
                    ident = study.get("protocolSection", {}).get("identificationModule", {})
                    nct_id = ident.get("nctId", "NCT_UNKNOWN")
                    title = ident.get("briefTitle", "No Title")
                    trials_info.append(f"- [{nct_id}] {title}")
                return "\n".join(trials_info)
        except Exception as e:
            print(f"⚠️ ClinicalTrials API 에러: {e}")
        return "Failed to load clinical trial information."

    # =========================================================
    # 🚀 최종 파이프라인 조립 (English Prompt)
    # =========================================================
    def run_pipeline(self, patient_input):
        pos_hpos = patient_input["positive_hpos"]
        neg_hpos = patient_input["negative_hpos"]
        
        local_candidates = self.search_local_orphadata(pos_hpos, neg_hpos, top_k=3)
        api_candidates, _ = self.get_api_diseases_and_pmids(pos_hpos, top_k=3)
        
        merged_candidates = {}
        for cand in local_candidates + api_candidates:
            d_id = cand["id"]
            if d_id not in merged_candidates:
                merged_candidates[d_id] = cand
            else:
                merged_candidates[d_id]["source"] += f" & {cand['source']}"
        
        final_candidates = list(merged_candidates.values())
        
        disease_info = ""
        top_disease_name = ""
        top_disease_id = ""
        
        for i, cand in enumerate(final_candidates):
            monarch_name, desc = self.get_disease_metadata(cand["id"])
            final_name = cand["name"] if cand["name"] else (monarch_name if monarch_name else "Unknown Disease")
            disease_info += f"- [{cand['source']}] {final_name} ({cand['id']}): {desc}\n"
            
            if i == 0:
                top_disease_name = final_name
                top_disease_id = cand["id"]
                
        direct_pmids = []
        if top_disease_id:
            direct_pmids = self.search_pubmed_pmids_directly(top_disease_name, top_disease_id, top_k=3)
            
        case_reports = self.get_pubmed_abstracts(direct_pmids)
        clinical_trials = self.get_clinical_trials(top_disease_name)
        
        # 💡 영어 프롬프트로 전면 교체
        prompt = f"""
You are a world-class AI specialized in rare disease diagnosis and clinical informatics.
Based on the provided raw patient data and the literature retrieved by our dual RAG system, formulate a comprehensive and highly professional diagnostic report.

=========================================
[1. Patient Clinical Data]
- Positive Findings: {patient_input["symptoms_raw"]}
- Negative Findings (Ruled out): {patient_input["negative_raw"]}

[2. HPO Mapping Information]
- Positive HPOs: {pos_hpos}
- Negative HPOs: {neg_hpos}

[3. High-Probability Candidate Diseases (Dual RAG)]
{disease_info}

[4. Real Medical Case Reports (PubMed)]
{case_reports}

[5. Alternative Clinical Trials (Currently Recruiting)]
{clinical_trials}
=========================================

[Instructions]
1. Primary Diagnosis: Prioritize diseases cross-validated by both the Local DB and the Global API. Provide a clinical rationale.
2. Differential Diagnosis: Use the patient's negative findings to logically explain why certain candidate diseases should be ruled out.
3. Case Comparison: Compare and contrast the patient's current state with the provided PubMed case reports.
4. Actionable Alternatives: Synthesize the [5. Alternative Clinical Trials] data to recommend practical clinical trial opportunities for the patient.
"""
        return prompt

# =====================================================================
# 테스트 실행부 (AWS Bedrock 연동)
# =====================================================================
if __name__ == "__main__":
    # 💡 파트너님이 제공하신 데이터를 완벽하게 이식한 테스트 케이스
    patient_omim_235200 = {
        "symptoms_raw": "가족력 조사 상 상염색체 열성 유전(Autosomal recessive inheritance) 패턴이 의심됨.",
        "negative_raw": "빈혈 관련 어지러움이나 창백함 등의 소견은 명확히 배제됨.",
        "positive_hpos": [
            "HP:0002202", # Pleural effusion (흉수)
            "HP:0001640", # Cardiomegaly (심비대)
            "HP:0000819",  # Autosomal recessive inheritance (상염색체 열성 유전) - 추가 단서!
            "HP:0003281",
            "HP:0002202"
        ],
        "negative_hpos": [
            "HP:0001903"  # Anemia (빈혈 배제)
        ]
    }
    
    rag_system = HybridDualRAG("dummy.csv") 
    generated_prompt = rag_system.run_pipeline(patient_omim_235200)
    
    print("\n" + "="*50)
    print("✅ [RAG 시스템이 조립한 최종 프롬프트]")
    print("="*50)
    print(generated_prompt)
    
    # 💡 AWS Bedrock 설정
    REGION_NAME = "ap-northeast-2"
    MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0" # 최상급 추론 모델
    
    try:
        bedrock_client = boto3.client(service_name='bedrock-runtime', region_name=REGION_NAME)
        
        system_prompt = "You are an elite rare disease diagnostician. Write your final report clearly, logically, and professionally. Ensure your final output is written in Korean as requested by the clinical team."
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": generated_prompt}
            ],
            "temperature": 0.0 # 진단의 일관성과 사실 기반 추론을 위해 0으로 고정
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
        
        # txt 파일 저장 로직
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnosis_report_{timestamp}.txt"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write("==================================================\n")
            f.write(f"🩺 Rare Disease Diagnosis Report (Generated: {timestamp})\n")
            f.write("==================================================\n\n")
            f.write("--- [1. GENERATED PROMPT] ---\n")
            f.write(generated_prompt)
            f.write("\n\n--- [2. LLM DIAGNOSIS RESULT] ---\n")
            f.write(final_report)
            
        print(f"\n✅ 진단 리포트가 성공적으로 저장되었습니다: {filename}")
        print("\n" + "🩺 "*20)
        print(final_report)
        
    except Exception as e:
        print(f"❌ Bedrock API 호출 에러: {e}")
        print("💡 팁: AWS CLI 자격 증명(aws configure)이 올바르게 설정되어 있는지 확인하세요.")