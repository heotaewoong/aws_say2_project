import requests
import json
import os

def build_hpo_official_data():
    """
    Downloads the official HPO OBO Graph JSON and saves it as hpo_official.json.
    This file is used by symtom_llm_4.py for high-precision extraction.
    """
    url = "http://purl.obolibrary.org/obo/hp.json"
    output_filename = "hpo_official.json"
    
    print(f"📥 공식 HPO 데이터베이스(hp.json) 다운로드 중... (약 50MB+)")
    print("이 파일은 모든 동의어와 계층 구조를 포함하며, symtom_llm_4.py의 핵심 데이터 소스가 됩니다.")
    
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"✅ 다운로드 완료: '{output_filename}'")
        
        # 파일 검증
        with open(output_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            nodes = data.get('graphs', [{}])[0].get('nodes', [])
            print(f"📊 검증 성공: 총 {len(nodes)}개의 노드(용어 및 관계)가 로드되었습니다.")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    build_hpo_official_data()
