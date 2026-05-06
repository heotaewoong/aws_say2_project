import xml.etree.ElementTree as ET
import pandas as pd
import os

class KnowledgeBaseBuilder:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        # 빈도를 수치 가중치로 변환 (Orphadata 표준 기준)
        self.freq_weight_map = {
            'Always (100%)': 1.0,
            'Very frequent (99-80%)': 0.9,
            'Frequent (79-30%)': 0.5,
            'Occasional (29-5%)': 0.1,
            'Unknown': 0.3
        }

    def build_csv(self, output_path):
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"❌ 원본 XML 파일을 찾을 수 없습니다: {self.xml_path}")

        print(f"📂 로컬 XML 분석 시작: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        records = []
        # 실제 en_product4.xml의 Disorder 태그 순회
        for disorder in root.findall(".//Disorder"):
            orpha_code = disorder.findtext("OrphaCode")
            name_node = disorder.find("Name")
            disease_name = name_node.text if name_node is not None else "Unknown"
            
            # HPO 협회 정보 추출
            for assoc in disorder.findall(".//HPODisorderAssociation"):
                hpo_node = assoc.find("HPO")
                if hpo_node is not None:
                    hpo_id = hpo_node.findtext("HPOId")
                    hpo_term = hpo_node.findtext("HPOTerm")
                
                    freq_node = assoc.find(".//HPOFrequency/Name")
                    freq_text = freq_node.text if freq_node is not None else "Unknown"
                    weight = self.freq_weight_map.get(freq_text, 0.3)
                    
                    records.append({
                        "OrphaCode": orpha_code,
                        "DiseaseName": disease_name,
                        "HPO_ID": hpo_id,
                        "HPO_Term": hpo_term,
                        "Weight": weight
                    })
        
        df = pd.DataFrame(records)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ 지식 베이스 저장 완료: {output_path} ({len(df)} 행)")
        return df

if __name__ == "__main__":
    # 로컬 경로 설정
    XML_FILE = "./data/en_product4.xml"
    OUTPUT_CSV = "./data/orphadata_weighted.csv"
    
    builder = KnowledgeBaseBuilder(XML_FILE)
    builder.build_csv(OUTPUT_CSV)