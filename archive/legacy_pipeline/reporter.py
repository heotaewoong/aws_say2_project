import datetime
import ollama
import json

class ClinicalReporter:
    def __init__(self, model_name="llama3.1"):
        self.model_name = model_name
        # 시스템 프롬프트: 에이전트 D의 페르소나 설정
        self.system_instruction = (
            "You are a Senior Pulmonologist and Medical Geneticist (Agent D). "
            "Your role is to synthesize multi-modal findings (Vision, NLP, Lab, Genomics) "
            "to provide a comprehensive diagnostic summary. "
            "Use professional medical terminology. Be concise but thorough. "
            "Structure the report clearly in Korean."
        )

    def _format_findings(self, source_name, findings):
        """각 에이전트의 결과를 텍스트로 정리"""
        if not findings:
            return f"- {source_name}: 특이 소견 없음\n"
        
        formatted = f"- {source_name}:\n"
        for f in findings:
            score_str = f" (Score: {f['score']:.2f})" if 'score' in f else ""
            formatted += f"   * {f['finding']} [{f['hpo_id']}]{score_str}\n"
        return formatted

    def generate_summary(self, subject_id, vision_results, nlp_results, lab_genomic_results, rare_rankings_text=""):
        """
        Agent D를 사용하여 최종 리포트 생성
        """
        # 1. 모든 소견 통합 (Context Building)
        clinical_context = "### Multi-modal Clinical Evidence\n"
        clinical_context += self._format_findings("영상 분석 (Agent A)", vision_results)
        clinical_context += self._format_findings("임상 메모 분석 (Agent B)", nlp_results)
        clinical_context += self._format_findings("검사 및 유전체 분석 (Agent C)", lab_genomic_results)

        # 2. 에이전트 D에게 보낼 프롬프트 구성
        prompt = f"""
        당신은 수석 전문의 에이전트 D입니다. 
        아래 환자 ID {subject_id}의 멀티모달 분석 데이터와 희귀 질환 매칭 결과를 바탕으로 최종 진단 리포트를 작성하세요.
        
        [분석 데이터]
        {clinical_context}
        
        [희귀 질환 매칭 순위 (Agent E 결과)]
        {rare_rankings_text}
        
        리포트에는 다음 항목이 반드시 포함되어야 합니다:
        1. 주요 발견 소견 (Key Findings)
        2. 의심 질환 (Differential Diagnosis) - 희귀 질환 가능성과 근거(Evidence) 포함
        3. 추가 권고 검사 및 관리 계획 (Recommendations)
        
        작성 언어: 한국어
        """

        try:
            print("👨‍⚕️ 수석 전문의(Agent D)가 리포트를 작성 중입니다...")
            response = ollama.generate(
                model=self.model_name,
                system=self.system_instruction,
                prompt=prompt,
                options={"temperature": 0.2} # 약간의 창의성 허용
            )
            
            final_content = response['response']
            
            # 3. 리포트 포맷팅 (Header/Footer 추가)
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            report = [
                "STRICTLY CONFIDENTIAL - Rare-Link AI Diagnostic Systems",
                f"발행 일시: {now} | 환자 ID: {subject_id}",
                "="*60,
                final_content,
                "="*60,
                "⚠️ 본 리포트는 AI 에이전트 연합에 의해 생성되었습니다. 최종 진단은 반드시 담당 전문의의 확인이 필요합니다."
            ]
            
            return "\n".join(report)

        except Exception as e:
            return f"❌ 리포트 생성 중 오류 발생: {e}"

# --- 실행 예시 ---
if __name__ == "__main__":
    reporter = ClinicalReporter()
    
    # 에이전트들로부터 받은 가상의 데이터
    v_res = [{'finding': 'Bilateral infiltrates', 'hpo_id': 'HP:0002113', 'score': 0.76}]
    n_res = [{'finding': 'Finger clubbing', 'hpo_id': 'HP:0001217'}]
    l_res = [{'finding': 'Pathogenic variant in SFTPB', 'hpo_id': 'HP:0006527', 'score': 1.0, 'source': 'Genomic'}]
    
    final_report = reporter.generate_summary("10000032", v_res, n_res, l_res)
    print(final_report)