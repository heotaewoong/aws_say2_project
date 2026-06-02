import pandas as pd
import numpy as np
import os

class RareDiseaseInference:
    def __init__(self, kb_path=None):
        # 1. 파일의 현재 위치를 기준으로 절대 경로 계산
        if kb_path is None:
            # 현재 파일(inference_engine.py)의 디렉토리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # 프로젝트 루트로 나간 뒤 data 폴더로 접근
            kb_path = os.path.join(current_dir, "..", "data", "orphadata_weighted.csv")

        # 2. 경로 정규화 (역슬래시/슬래시 문제 방지)
        kb_path = os.path.normpath(kb_path)

        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"❌ 지식 베이스가 없습니다. Phase 1을 먼저 실행하세요: {kb_path}")
        
        self.kb_df = pd.read_csv(kb_path)
        self.hpo_specificity = self._calculate_hpo_idf()
        print(f"🧬 추론 엔진 로드 완료 (지식 베이스 크기: {len(self.kb_df)} 행)")

    def _calculate_hpo_idf(self):
        """증상별 희귀도(IDF) 계산: 특정 증상이 나타나는 질환이 적을수록 점수가 높음"""
        total_diseases = self.kb_df['OrphaCode'].nunique()
        # 각 HPO ID가 몇 개의 질환에 등장하는지 카운트
        hpo_counts = self.kb_df.groupby('HPO_ID')['OrphaCode'].nunique()
        # IDF = log(전체 질환 수 / 해당 증상을 가진 질환 수)
        return np.log(total_diseases / hpo_counts)

    def rank_diseases(self, patient_hpo_list, top_k=5):
        """환자의 HPO 리스트를 입력받아 가능성 높은 질환 순위 반환"""
        # 1. 지식 베이스에 있는 증상만 필터링
        valid_hpos = [h for h in patient_hpo_list if h in self.hpo_specificity.index]
        
        if not valid_hpos:
            return pd.DataFrame(columns=['OrphaCode', 'DiseaseName', 'Score', 'Evidence'])

        results = []
        # 2. 각 질환별로 점수 합산
        for (orpha_code, disease_name), group in self.kb_df.groupby(['OrphaCode', 'DiseaseName']):
            # 환자 증상 중 이 질환과 일치하는 것 찾기
            matches = group[group['HPO_ID'].isin(valid_hpos)]
            
            if not matches.empty:
                current_score = 0
                evidence_list = []
                
                for _, row in matches.iterrows():
                    hpo_id = row['HPO_ID']
                    hpo_term = row['HPO_Term']
                    weight = row['Weight']
                    idf = self.hpo_specificity[hpo_id]
                    
                    # 최종 기여도 계산
                    contribution = weight * idf
                    current_score += contribution
                    evidence_list.append(f"{hpo_term}(+{contribution:.2f})")
                
                results.append({
                    'OrphaCode': orpha_code,
                    'DiseaseName': disease_name,
                    'Score': round(current_score, 2),
                    'Evidence': ", ".join(evidence_list)
                })
        
        # 3. 점수 내림차순 정렬 및 Top-K 추출
        ranking_df = pd.DataFrame(results).sort_values(by='Score', ascending=False).head(top_k)
        return ranking_df