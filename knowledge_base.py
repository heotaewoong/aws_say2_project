import xml.etree.ElementTree as ET
import pandas as pd
import os

class KnowledgeBaseBuilder:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.freq_weight_map = {
            'Always (100%)': 1.0,
            'Very frequent (99-80%)': 0.9,
            'Frequent (79-30%)': 0.5,
            'Occasional (29-5%)': 0.1,
            'Unknown': 0.3
        }

    def build_csv(self, output_path='orphadata_weighted.csv'):
        print(f"📂 XML 파싱 시작: {self.xml_path}")
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        
        records = []
        for disorder in root.findall(".//Disorder"):
            orpha_code = disorder.findtext("OrphaCode")
            disease_name = disorder.find(".//Name").text if disorder.find(".//Name") is not None else "Unknown"
            
            for assoc in disorder.findall(".//HPODisorderAssociation"):
                hpo_node = assoc.find("HPO")
                hpo_id = hpo_node.findtext("HPOId") if hpo_node is not None else "N/A"
                hpo_term = hpo_node.findtext("HPOTerm") if hpo_node is not None else "N/A"
                
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
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 생성 완료: {output_path} ({len(df)} 행)")
        return df