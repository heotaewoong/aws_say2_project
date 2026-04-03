import pandas as pd
import numpy as np

class LabGenomicAgent:
    def __init__(self):
        # 1. Lab 판정 규칙 (나중에 데이터 준비 단계에서 확장 가능)
        self.lab_rules = {
            'WBC Count': {'high': 11.0, 'low': 4.0, 'hpo': 'HP:0001890'},  # Leukocytosis
            'Platelet Count': {'low': 150, 'hpo': 'HP:0001873'},         # Thrombocytopenia
            'CRP': {'high': 10.0, 'hpo': 'HP:0001945'},                  # Inflammation
            'Oxygen Saturation': {'low': 95, 'hpo': 'HP:0012418'}         # Hypoxemia
        }

        # 2. 유전체-HPO 매핑 (타겟 유전자 리스트)
        self.genomic_kb = {
            'SFTPB': 'HP:0006527',  # Pulmonary surfactant metabolism dysfunction
            'ABCA3': 'HP:0002094',  # Pulmonary alveolar proteinosis
            'CFTR': 'HP:0002100',   # Cystic fibrosis
            'NKX2-1': 'HP:0001631'  # Atrial septal defect / Lung involvement
        }

    def _calculate_lab_confidence(self, value, threshold, direction='high'):
        """수치가 정상 범위를 많이 벗어날수록 높은 점수를 부여하는 로직"""
        # 간단한 거리 기반 점수 (예: 1.0 + (차이/임계값))
        diff = abs(value - threshold)
        score = min(1.0, 0.7 + (diff / threshold)) # 기본 0.7에서 최대 1.0까지
        return round(score, 4)

    def analyze_labs(self, lab_data_dict):
        """
        lab_data_dict: {'WBC Count': 15.2, 'CRP': 50.0} 형태
        """
        results = []
        for label, value in lab_data_dict.items():
            if label in self.lab_rules:
                rule = self.lab_rules[label]
                
                # 고수치 판정
                if 'high' in rule and value > rule['high']:
                    score = self._calculate_lab_confidence(value, rule['high'], 'high')
                    results.append({
                        'finding': f'High {label}',
                        'hpo_id': rule['hpo'],
                        'score': score,
                        'source': 'Lab_Analysis'
                    })
                
                # 저수치 판정
                elif 'low' in rule and value < rule['low']:
                    score = self._calculate_lab_confidence(value, rule['low'], 'low')
                    results.append({
                        'finding': f'Low {label}',
                        'hpo_id': rule['hpo'],
                        'score': score,
                        'source': 'Lab_Analysis'
                    })
        return results

    def analyze_genomics(self, variant_list):
        """
        variant_list: ['SFTPB', 'CFTR'] 등 검출된 유전자 리스트
        """
        results = []
        for gene in variant_list:
            if gene in self.genomic_kb:
                results.append({
                    'finding': f'Pathogenic variant in {gene}',
                    'hpo_id': self.genomic_kb[gene],
                    'score': 1.0,  # 유전 변이는 확정적 정보이므로 1.0 부여
                    'source': 'Genomic_Analysis'
                })
        return results

    def extract_hpos(self, lab_data, variant_data):
        """에이전트 C의 통합 실행 메서드"""
        print("🧬 에이전트 C가 Lab 및 유전체 데이터를 분석 중입니다...")
        
        lab_results = self.analyze_labs(lab_data)
        genomic_results = self.analyze_genomics(variant_data)
        
        combined_results = lab_results + genomic_results
        
        # 결과 요약 출력
        for res in combined_results:
            print(f"✅ 검출: [{res['source']}] {res['finding']} -> {res['hpo_id']} (Score: {res['score']})")
            
        return combined_results

# --- 테스트 실행 ---
if __name__ == "__main__":
    agent_c = LabGenomicAgent()
    
    # 가상의 입력 데이터
    sample_labs = {'WBC Count': 18.5, 'CRP': 120.0, 'Oxygen Saturation': 92.0}
    sample_genomics = ['SFTPB']
    
    hpo_list = agent_c.extract_hpos(sample_labs, sample_genomics)