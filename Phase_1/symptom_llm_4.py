import boto3
import json
import os
import difflib
import datetime
import re
import traceback
from typing import Optional, Dict, List, Tuple, Any

class BedrockHPOExtractor:
    """
    Extracts HPO terms from clinical notes using AWS Bedrock (Claude 3.5 Sonnet)
    with direct reference to the official HPO OBO Graph JSON.
    """
    
    def __init__(self, region_name: str = "ap-northeast-2", hpo_json_path: str = "hpo_official.json"):
        """
        Initializes the AWS Bedrock client and parses the official HPO JSON.
        """
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )

        self.model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        # HPO Database storage
        self.hpo_id_to_term: Dict[str, str] = {}
        self.term_to_hpo_id: Dict[str, str] = {}
        
        # Load and parse official HPO JSON
        if os.path.exists(hpo_json_path):
            print(f"📥 Loading official HPO data from {hpo_json_path}...")
            try:
                with open(hpo_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract nodes (terms)
                for node in data.get('graphs', [{}])[0].get('nodes', []):
                    node_id_raw = node.get('id', '')
                    if node_id_raw.startswith('http://purl.obolibrary.org/obo/HP_'):
                        hpo_id = node_id_raw.split('/')[-1].replace('_', ':')
                        primary_label = node.get('lbl', '')
                        
                        if primary_label:
                            self.hpo_id_to_term[hpo_id] = primary_label
                            self.term_to_hpo_id[primary_label.lower().strip()] = hpo_id
                            
                            # Extract synonyms
                            meta = node.get('meta', {})
                            synonyms = meta.get('synonyms', [])
                            for syn in synonyms:
                                syn_val = syn.get('val')
                                if syn_val:
                                    self.term_to_hpo_id[syn_val.lower().strip()] = hpo_id
                
                print(f"✅ HPO Database initialized: {len(self.hpo_id_to_term)} primary terms and {len(self.term_to_hpo_id)} total labels/synonyms.")
            except Exception as e:
                print(f"❌ Error parsing HPO JSON: {e}")
        else:
            print(f"⚠️ Warning: {hpo_json_path} not found.")

    def _log_error(self, method_name, input_id, error):
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method": method_name,
            "image_path": input_id,
            "error_message": str(error),
            "traceback": traceback.format_exc()
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

    def _get_reference_candidates(self, keywords: List[str]) -> str:
        """
        Searches the HPO database for terms matching the identified medical keywords.
        Uses both substring matching and fuzzy matching for better robustness.
        """
        candidates = set()
        all_hpo_terms = list(self.term_to_hpo_id.keys())

        for kw in keywords:
            kw_clean = kw.lower().strip()
            if len(kw_clean) < 3: continue

            # 1. Substring matching
            match_count = 0
            for term in all_hpo_terms:
                if kw_clean in term:
                    hpo_id = self.term_to_hpo_id[term]
                    official_label = self.hpo_id_to_term.get(hpo_id, term)
                    candidates.add(f"{official_label} ({hpo_id})")
                    match_count += 1
                if match_count > 15: break # Limit per keyword

            # 2. Fuzzy matching
            fuzzy_matches = difflib.get_close_matches(kw_clean, all_hpo_terms, n=5, cutoff=0.7)
            for match in fuzzy_matches:
                hpo_id = self.term_to_hpo_id[match]
                official_label = self.hpo_id_to_term.get(hpo_id, match)
                candidates.add(f"{official_label} ({hpo_id})")

            if len(candidates) > 150: break # Overall limit

        return "\n".join(list(candidates))


    def _map_term_to_hpo(self, english_term: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Maps the LLM's extracted term back to the official HPO ID and Label.
        """
        term_lower = english_term.lower().strip()
        
        # 1. Exact Match (including synonyms)
        if term_lower in self.term_to_hpo_id:
            hpo_id = self.term_to_hpo_id[term_lower]
            return hpo_id, self.hpo_id_to_term.get(hpo_id, english_term)
        
        # 2. Substring Match
        for term_in_db, hpo_id in self.term_to_hpo_id.items():
            if term_lower in term_in_db or term_in_db in term_lower:
                return hpo_id, self.hpo_id_to_term.get(hpo_id, term_in_db)
            
        # 3. Fuzzy Match
        matches = difflib.get_close_matches(term_lower, self.term_to_hpo_id.keys(), n=1, cutoff=0.8)
        if matches:
            best_match = matches[0]
            hpo_id = self.term_to_hpo_id[best_match]
            return hpo_id, self.hpo_id_to_term.get(hpo_id, best_match)
            
        return None, None

    def extract_hpo_from_clinical_note(self, input_source: str) -> Dict[str, Any]:
        """
        Extracts HPO terms using a 2-step Discovery -> Reference -> Extraction process.
        """
        try:
            clinical_note = input_source
            try:
                data = json.loads(input_source)
                if isinstance(data, dict):
                    clinical_note = data.get("clinical_note", input_source)
            except: pass

            # --- STEP 1: Discovery (Identify medical keywords) ---
            print("\n[Step 1] Discovering clinical keywords from note...")
            discovery_prompt = f"Identify 5-10 core medical/phenotypic keywords in English for this clinical note: \"{clinical_note}\". Return ONLY a comma-separated list."
            
            try:
                discovery_body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": discovery_prompt}],
                    "temperature": 0.0
                })
                
                disc_response = self.bedrock_client.invoke_model(body=discovery_body, modelId=self.model_id)
                keywords_raw = json.loads(disc_response.get("body").read().decode())["content"][0]["text"]
                keywords = [k.strip() for k in keywords_raw.split(",")]
                print(f"🔍 Keywords identified: {keywords}")
                
                # --- STEP 2: Search HPO Reference ---
                hpo_reference = self._get_reference_candidates(keywords)
                
            except Exception as e:
                print(f"⚠️ Discovery step failed: {e}. Falling back to general knowledge.")
                hpo_reference = "Search official HPO terminology."

            # --- STEP 3: Final Extraction with Reference ---
            system_prompt = f"""
            You are an expert clinical informatician.
            Your task is to analyze clinical notes written in Korean and extract clinical findings.
            
            ### REFERENCE HPO TERMS (Use these EXACT labels if they match the context):
            {{hpo_reference}}
            
            ### Instructions:
            1. MANDATORY ALIGNMENT: You MUST prioritize using the EXACT labels provided in the REFERENCE HPO TERMS list above.
            2. ATOMIC EXTRACTION: Extract EXACTLY ONE symptom per JSON object.
               - Example: If the note says "Nausea exists but no vomiting", you MUST create TWO entries:
                 a) Positive: "Nausea"
                 b) Negative: "Vomiting"
               - DO NOT use combined terms like "Nausea and vomiting" unless BOTH are present or absent together.
            3. NO NEGATION: The `english_term` must be the core symptom without negation words. The negation is indicated by the "negative_findings" category.
            4. ACCURACY: Ensure every clinical finding in the note is captured individually.
            5. FORMAT: Return ONLY raw JSON.
            
            JSON Output Schema:
            {{
                "positive_findings": [{{ "exact_quote_from_text": "...", "english_term": "..." }}],
                "negative_findings": [{{ "exact_quote_from_text": "...", "english_term": "..." }}]
            }}
            """

            user_prompt = f"Extract symptoms in HPO format from this note: \"{{clinical_note}}\""

            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.0
            })

            print("[Step 2] Invoking AWS Bedrock for final HPO-aligned extraction...")
            response = self.bedrock_client.invoke_model(
                body=body, modelId=self.model_id,
                accept="application/json", contentType="application/json"
            )
            
            raw_response_body = response.get("body").read().decode("utf-8")
            response_json = json.loads(raw_response_body)
            extracted_text = response_json["content"][0]["text"].strip()
            
            start_idx = extracted_text.find("{")
            end_idx = extracted_text.rfind("}")
            
            if start_idx != -1 and end_idx != -1:
                json_str = extracted_text[start_idx : end_idx + 1]
                raw_findings = json.loads(json_str)
            else:
                raise ValueError("No JSON object found.")
            
            final_hpo_result = {
                "positive_hpos": [],
                "negative_hpos": [],
                "unmapped_findings": []
            }
            
            print("[Step 3] Mapping findings to IDs...")
            
            for finding_type, target_key in [("positive_findings", "positive_hpos"), ("negative_findings", "negative_hpos")]:
                for item in raw_findings.get(finding_type, []):
                    english_term = item.get("english_term", "")
                    hpo_id, official_term = self._map_term_to_hpo(english_term)
                    
                    if hpo_id:
                        final_hpo_result[target_key].append({
                            "exact_quote_from_text": item["exact_quote_from_text"],
                            "hpo_id": hpo_id,
                            "official_term": official_term,
                            "llm_extracted_term": english_term
                        })
                    else:
                        final_hpo_result["unmapped_findings"].append(item)

            return final_hpo_result
        except Exception as e:
            self._log_error("extract_hpo_from_clinical_note", input_source[:100], e)
            return {"positive_hpos": [], "negative_hpos": [], "unmapped_findings": []}

if __name__ == "__main__":
    extractor = BedrockHPOExtractor(hpo_json_path="hpo_official.json")
    
    json_input = json.dumps({
        "clinical_note": """
        45세 남성 환자, 최근 2주간 지속되는 만성 피로와 양손 검지/중지 관절통(Arthralgia)으로 내원함. 
        최근 햇빛 노출이 없었음에도 피부가 청동색으로 어두워지는 양상 보임. 
        가슴 답답함이나 창백함은 호소하지 않으며, 어지러움(빈혈 소견)은 없음. 
        과거력 상 외계인 납치 증후군(Alien Abduction Syndrome) 의심됨.
        """
    })
    
    print("🚀 Starting HPO extraction process...")
    result = extractor.extract_hpo_from_clinical_note(json_input)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"hpo_extraction_{timestamp}.json"
    
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ Extraction Complete. Results saved to: {output_filename}")
    print("Validated HPO Data Structure:")
    print(json.dumps(result, indent=4, ensure_ascii=False))
