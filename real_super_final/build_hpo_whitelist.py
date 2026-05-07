import requests
import json

def build_hpo_whitelist():
    url = "http://purl.obolibrary.org/obo/hp.json"
    
    print("📥 공식 HPO 데이터베이스 다운로드 중... (약 50MB, 잠시만 기다려주세요)")
    response = requests.get(url)
    
    if response.status_code != 200:
        print("❌ 다운로드 실패!")
        return
        
    data = response.json()
    whitelist = {}
    
    print("⚙️ OBO Graph 구조에서 HPO 코드와 영문명 추출 중...")
    # OBO Graph JSON 구조 파싱
    for node in data['graphs'][0]['nodes']:
        node_id = node.get('id', '')
        
        # 'http://purl.obolibrary.org/obo/HP_0000001' 형태의 ID만 필터링
        if node_id.startswith('http://purl.obolibrary.org/obo/HP_'):
            # URL을 우리가 쓰는 'HP:0000001' 형태로 변환
            hpo_code = node_id.split('/')[-1].replace('_', ':')
            # 해당 HPO의 공식 영문 명칭
            term_name = node.get('lbl', 'Unknown')
            
            whitelist[hpo_code] = term_name

    # 최종 결과물을 json 파일로 저장
    output_filename = "hpo_whitelist.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(whitelist, f, indent=4)
        
    print(f"✅ 추출 완료! 총 {len(whitelist)}개의 HPO 코드가 '{output_filename}'에 저장되었습니다.")

if __name__ == "__main__":
    build_hpo_whitelist()