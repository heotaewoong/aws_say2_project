# 요구사항 문서

## 소개

PaperHub는 다중 학술 소스에서 논문을 수집·요약·추천하는 서버리스 기반 개인 논문 서비스이다. 본 문서는 PaperHub에 9가지 신규 기능을 추가하기 위한 요구사항을 정의한다. 신규 기능은 논문 본문 기반 Q&A(RAG), 오디오 요약(Podcast Mode), 다국어 번역, 관련 유명 논문 Top 3 추천, 자동 문헌 고찰, Paper-to-Code/Data 매칭, 피어 리뷰 AI 어시스턴트, 다중 논문 소스 수집, 요약 가독성 향상 UI/UX이다.

## 용어 정의

- **PaperHub_System**: PaperHub 서비스 전체를 구성하는 AWS 서버리스 아키텍처(CloudFront, API Gateway, Lambda, DynamoDB, S3, EventBridge, Step Functions)
- **RAG_Engine**: PDF 텍스트를 추출하고 OpenSearch Serverless 벡터 스토어에 임베딩을 저장하여 Bedrock 기반 질의응답을 수행하는 컴포넌트
- **Text_Extractor**: PDF 파일에서 텍스트를 추출하는 컴포넌트
- **Vector_Store**: OpenSearch Serverless 기반 벡터 인덱스로, 논문 텍스트 청크의 임베딩을 저장하고 유사도 검색을 수행하는 저장소
- **Audio_Generator**: Amazon Polly를 사용하여 텍스트를 음성으로 변환하는 컴포넌트
- **Translation_Engine**: Amazon Translate 또는 Bedrock 번역 API를 사용하여 텍스트를 다국어로 번역하는 컴포넌트
- **Related_Paper_Recommender**: 현재 논문의 요약을 기반으로 관련 유명 논문 Top 3를 추천하는 컴포넌트
- **Literature_Review_Generator**: 사용자가 입력한 연구 주제에 대해 10~20편의 핵심 논문을 수집·분석하여 문헌 고찰 초안을 자동 생성하는 컴포넌트
- **Code_Data_Matcher**: 논문과 관련된 GitHub 리포지토리 및 Hugging Face/Kaggle 데이터셋을 자동으로 검색하고 재현성 점수를 산출하는 컴포넌트
- **Peer_Review_AI**: 사용자가 업로드한 논문 초고에 대해 논리적 결함, 누락된 인용, 통계적 타당성 등을 검토하여 비평적 피드백을 제공하는 컴포넌트
- **Audio_Cache**: S3에 저장되는 오디오 요약 파일의 캐시 저장소
- **Reproducibility_Score**: 논문의 코드·데이터 공개 여부, GitHub 스타 수, 라이선스 등을 종합하여 0~100 사이로 산출하는 재현성 점수
- **Frontend**: S3/CloudFront에 호스팅되는 PaperHub 단일 페이지 HTML 애플리케이션
- **Multi_Source_Ingestor**: PubMed, arXiv, Semantic Scholar, CrossRef, IEEE Xplore, DBLP, bioRxiv/medRxiv, Google Scholar 등 다중 학술 소스에서 논문 메타데이터를 수집하는 컴포넌트
- **Source_Adapter**: 각 학술 소스의 API 규격에 맞춰 논문 메타데이터를 조회하고 Unified_Paper_Schema로 변환하는 개별 커넥터
- **Unified_Paper_Schema**: 다양한 소스에서 수집된 논문 메타데이터를 단일 형식으로 통합하는 표준 스키마 (paperId, source, title, abstract, authors, doi, publishedDate, category, citationCount, sourceUrl 필드 포함)
- **Summary_Display**: 논문 AI 요약을 가독성 높은 구조화된 UI로 렌더링하는 Frontend 컴포넌트
- **Summary_Section_Card**: 요약의 각 섹션(배경/목적, 방법론, 결과, 결론)을 개별 스타일 카드로 표시하는 UI 요소

---

## 요구사항

### 요구사항 1: 논문 PDF 텍스트 추출 및 벡터 인덱싱

**사용자 스토리:** 연구자로서, 논문 PDF의 본문 텍스트가 벡터 스토어에 인덱싱되기를 원한다. 이를 통해 논문 내용에 대한 질의응답이 가능해진다.

#### 인수 조건

1. WHEN 사용자가 특정 논문의 Q&A 기능을 최초 요청하면, THE Text_Extractor SHALL 해당 논문의 PDF에서 전체 텍스트를 추출한다
2. WHEN 텍스트 추출이 완료되면, THE RAG_Engine SHALL 추출된 텍스트를 500~1000 토큰 단위의 청크로 분할한다
3. WHEN 청크 분할이 완료되면, THE RAG_Engine SHALL 각 청크를 Bedrock 임베딩 모델을 사용하여 벡터로 변환하고 Vector_Store에 저장한다
4. WHEN 벡터 인덱싱이 완료되면, THE RAG_Engine SHALL 해당 논문의 인덱싱 상태를 DynamoDB papers 테이블에 'indexed'로 업데이트한다
5. IF PDF 텍스트 추출에 실패하면, THEN THE Text_Extractor SHALL 오류 메시지와 함께 실패 상태를 반환한다
6. WHILE 논문이 이미 인덱싱된 상태이면, THE RAG_Engine SHALL 중복 인덱싱을 수행하지 않고 기존 인덱스를 재사용한다

---

### 요구사항 2: 논문 본문 기반 Q&A (Chat with Paper)

**사용자 스토리:** 연구자로서, 특정 논문에 대해 자연어로 질문하고 논문 본문에 기반한 답변을 받고 싶다.

#### 인수 조건

1. WHEN 사용자가 특정 논문에 대해 질문을 입력하면, THE RAG_Engine SHALL Vector_Store에서 질문과 가장 유사한 상위 5개 청크를 검색한다
2. WHEN 관련 청크가 검색되면, THE RAG_Engine SHALL 검색된 청크를 컨텍스트로 포함하여 Bedrock Nova Pro 모델에 질문을 전달하고 답변을 생성한다
3. THE RAG_Engine SHALL 생성된 답변과 함께 참조한 청크의 출처 정보(페이지 번호 또는 섹션)를 반환한다
4. WHEN 사용자가 동일 논문에 대해 후속 질문을 입력하면, THE RAG_Engine SHALL 이전 대화 컨텍스트를 유지하여 연속적인 대화를 지원한다
5. IF Vector_Store에 해당 논문의 인덱스가 존재하지 않으면, THEN THE RAG_Engine SHALL 자동으로 인덱싱 프로세스를 시작한 후 질문에 답변한다
6. IF 질문이 논문 내용과 관련이 없으면, THEN THE RAG_Engine SHALL 논문 범위 밖의 질문임을 사용자에게 안내한다

---

### 요구사항 3: 오디오 요약본 생성 (Podcast Mode)

**사용자 스토리:** 연구자로서, 논문의 AI 요약을 오디오로 들으며 이동 중에도 논문 내용을 파악하고 싶다.

#### 인수 조건

1. WHEN 사용자가 특정 논문의 오디오 요약을 요청하면, THE Audio_Generator SHALL 해당 논문의 AI 페이지 요약 텍스트를 Amazon Polly를 사용하여 음성으로 변환한다
2. THE Audio_Generator SHALL 생성된 오디오 파일의 길이를 1분 이내로 제한한다
3. WHEN 오디오 변환이 완료되면, THE Audio_Generator SHALL 생성된 오디오 파일을 Audio_Cache에 저장한다
4. WHEN 사용자가 오디오 요약을 요청하면, THE PaperHub_System SHALL Audio_Cache에 캐시된 오디오 파일이 존재하는 경우 캐시된 파일의 presigned URL을 반환한다
5. IF 해당 논문의 AI 요약이 아직 생성되지 않았으면, THEN THE Audio_Generator SHALL 먼저 AI 요약을 생성한 후 오디오 변환을 수행한다
6. THE Audio_Generator SHALL 한국어 음성(Seoyeon 보이스)을 기본 음성으로 사용한다
7. WHEN Frontend에서 오디오 요약을 재생할 때, THE Frontend SHALL 인라인 오디오 플레이어를 표시한다

---

### 요구사항 4: 다국어 번역 지원

**사용자 스토리:** 연구자로서, 영어 논문의 특정 단락을 한국어 또는 다른 언어로 번역하여 이해도를 높이고 싶다.

#### 인수 조건

1. WHEN 사용자가 논문의 특정 텍스트와 대상 언어를 지정하여 번역을 요청하면, THE Translation_Engine SHALL 해당 텍스트를 지정된 언어로 번역한다
2. THE Translation_Engine SHALL 한국어, 영어, 일본어, 중국어(간체) 번역을 지원한다
3. THE Translation_Engine SHALL 번역 시 학술 용어의 정확성을 유지하기 위해 원문 용어를 괄호 안에 병기한다
4. WHEN 번역 요청의 텍스트 길이가 5000자를 초과하면, THE Translation_Engine SHALL 텍스트를 5000자 이하의 단위로 분할하여 순차적으로 번역한다
5. IF 지원하지 않는 언어가 요청되면, THEN THE Translation_Engine SHALL 지원 가능한 언어 목록과 함께 오류 메시지를 반환한다

---

### 요구사항 5: 관련 유명 논문 Top 3 추천

**사용자 스토리:** 연구자로서, 현재 읽고 있는 논문과 관련된 유명 논문 3편을 추천받아 연구 맥락을 넓히고 싶다.

#### 인수 조건

1. WHEN 사용자가 특정 논문의 요약 페이지를 조회하면, THE Related_Paper_Recommender SHALL 해당 논문의 제목과 초록을 기반으로 관련 유명 논문 3편을 추천한다
2. THE Related_Paper_Recommender SHALL 각 추천 논문에 대해 제목, 저자, 발행 연도, 인용 수(가용한 경우), 추천 이유를 포함한다
3. THE Related_Paper_Recommender SHALL Bedrock Nova Pro 모델을 사용하여 논문 간 관련성을 분석하고 추천 이유를 한국어로 생성한다
4. WHEN 추천 결과가 생성되면, THE PaperHub_System SHALL 추천 결과를 DynamoDB에 캐싱하여 동일 논문에 대한 반복 요청 시 재사용한다
5. THE Frontend SHALL 논문 상세보기 모달에서 요약 아래에 관련 논문 Top 3 섹션을 표시한다

---

### 요구사항 6: 자동 문헌 고찰 생성

**사용자 스토리:** 연구자로서, 특정 연구 주제를 입력하면 관련 핵심 논문을 자동으로 수집·분석하여 문헌 고찰 초안을 받고 싶다.

#### 인수 조건

1. WHEN 사용자가 연구 주제를 입력하면, THE Literature_Review_Generator SHALL PubMed API를 통해 해당 주제와 관련된 10~20편의 핵심 논문을 수집한다
2. WHEN 논문 수집이 완료되면, THE Literature_Review_Generator SHALL 수집된 각 논문의 초록을 분석하여 주요 연구 동향, 방법론, 결과를 정리한다
3. WHEN 분석이 완료되면, THE Literature_Review_Generator SHALL 다음 구조로 문헌 고찰 초안을 생성한다: 서론, 연구 동향 분석, 방법론 비교, 주요 발견 요약, 연구 공백 및 향후 방향, 참고문헌 목록
4. THE Literature_Review_Generator SHALL 생성된 문헌 고찰의 길이를 3000~5000 단어 범위로 유지한다
5. THE Literature_Review_Generator SHALL 문헌 고찰 내 모든 인용에 대해 해당 논문의 제목, 저자, 발행 연도를 명시한다
6. WHEN 문헌 고찰 생성이 완료되면, THE Frontend SHALL 생성된 문헌 고찰을 마크다운 형식으로 렌더링하고 다운로드 옵션을 제공한다
7. IF 입력된 주제에 대해 관련 논문이 5편 미만으로 검색되면, THEN THE Literature_Review_Generator SHALL 사용자에게 주제를 구체화하거나 키워드를 변경할 것을 안내한다
8. THE Literature_Review_Generator SHALL Step Functions 워크플로우를 사용하여 논문 수집, 분석, 생성 단계를 순차적으로 실행한다

---

### 요구사항 7: Paper-to-Code 및 Paper-to-Data 매칭

**사용자 스토리:** 연구자로서, 논문과 관련된 GitHub 코드 리포지토리와 Hugging Face/Kaggle 데이터셋을 자동으로 찾아 연구 재현성을 확인하고 싶다.

#### 인수 조건

1. WHEN 사용자가 특정 논문의 코드/데이터 매칭을 요청하면, THE Code_Data_Matcher SHALL 해당 논문의 제목과 DOI를 기반으로 GitHub API에서 관련 리포지토리를 검색한다
2. WHEN GitHub 검색이 완료되면, THE Code_Data_Matcher SHALL Hugging Face API와 Kaggle API를 사용하여 관련 데이터셋을 검색한다
3. THE Code_Data_Matcher SHALL 각 검색 결과에 대해 리포지토리/데이터셋 이름, URL, 설명, 스타 수(GitHub의 경우), 다운로드 수(데이터셋의 경우)를 포함한다
4. THE Code_Data_Matcher SHALL 검색된 코드와 데이터의 가용성을 기반으로 Reproducibility_Score를 0~100 범위로 산출한다
5. THE Code_Data_Matcher SHALL Reproducibility_Score 산출 시 다음 기준을 적용한다: 공식 코드 존재 여부(40점), 데이터셋 공개 여부(30점), 라이선스 명시 여부(15점), 문서화 수준(15점)
6. WHEN 매칭 결과가 생성되면, THE PaperHub_System SHALL 매칭 결과와 Reproducibility_Score를 DynamoDB에 저장한다
7. THE Frontend SHALL 논문 상세보기에서 코드/데이터 매칭 결과와 Reproducibility_Score를 시각적 게이지로 표시한다
8. IF 관련 코드나 데이터셋이 검색되지 않으면, THEN THE Code_Data_Matcher SHALL Reproducibility_Score를 0으로 설정하고 해당 논문에 대한 공개 코드/데이터가 없음을 안내한다

---

### 요구사항 8: 피어 리뷰 AI 어시스턴트

**사용자 스토리:** 연구자로서, 작성 중인 논문 초고를 업로드하여 AI로부터 비평적 피드백을 받아 논문 품질을 향상시키고 싶다.

#### 인수 조건

1. WHEN 사용자가 논문 초고 PDF를 업로드하면, THE Peer_Review_AI SHALL PDF에서 텍스트를 추출하고 논문 구조(서론, 방법론, 결과, 결론)를 식별한다
2. WHEN 텍스트 추출이 완료되면, THE Peer_Review_AI SHALL 다음 항목에 대해 검토를 수행한다: 논리적 일관성, 주장과 근거의 연결성, 방법론의 적절성, 통계적 타당성, 결론의 타당성
3. THE Peer_Review_AI SHALL 각 검토 항목에 대해 구체적인 피드백과 개선 제안을 한국어로 제공한다
4. THE Peer_Review_AI SHALL 논문에서 인용이 필요하지만 누락된 부분을 식별하고 관련 논문을 제안한다
5. WHEN 검토가 완료되면, THE Peer_Review_AI SHALL 전체 검토 결과를 구조화된 리포트 형식으로 반환한다: 총평, 강점, 약점, 세부 피드백(섹션별), 누락 인용 제안
6. THE Peer_Review_AI SHALL 업로드된 논문의 최대 파일 크기를 20MB로 제한한다
7. IF 업로드된 파일이 PDF 형식이 아니면, THEN THE Peer_Review_AI SHALL 지원 형식 안내와 함께 오류 메시지를 반환한다
8. IF 업로드된 PDF에서 텍스트를 추출할 수 없으면, THEN THE Peer_Review_AI SHALL 스캔된 이미지 PDF는 지원하지 않음을 안내한다

---

### 요구사항 9: Q&A 대화 이력 파싱 및 표시

**사용자 스토리:** 연구자로서, 논문에 대한 Q&A 대화 이력을 저장하고 다시 확인하고 싶다.

#### 인수 조건

1. WHEN 사용자가 논문 Q&A에서 질문과 답변을 주고받으면, THE PaperHub_System SHALL 대화 이력을 DynamoDB에 저장한다
2. WHEN 사용자가 이전에 Q&A를 수행한 논문의 Q&A 화면을 다시 열면, THE Frontend SHALL 저장된 대화 이력을 시간순으로 표시한다
3. THE PaperHub_System SHALL 대화 이력을 JSON 형식으로 직렬화하여 저장하고, 조회 시 동일한 형식으로 역직렬화하여 반환한다
4. FOR ALL 유효한 대화 이력 객체에 대해, 직렬화 후 역직렬화하면 원본과 동일한 객체가 생성된다 (라운드트립 속성)
5. IF 대화 이력 조회에 실패하면, THEN THE PaperHub_System SHALL 빈 대화 이력과 함께 새 대화를 시작할 수 있도록 안내한다


---

### 요구사항 10: 다중 논문 소스 수집 (Multi-Source Paper Ingestion)

**사용자 스토리:** 연구자로서, PubMed뿐만 아니라 arXiv, Semantic Scholar, CrossRef, IEEE Xplore, DBLP, bioRxiv/medRxiv, Google Scholar 등 전 세계 주요 학술 소스에서 논문을 수집하여 더 넓은 범위의 연구 자료에 접근하고 싶다.

#### 인수 조건

1. THE Multi_Source_Ingestor SHALL 다음 8개 학술 소스에서 논문 메타데이터를 수집한다: PubMed, arXiv, Semantic Scholar, CrossRef, IEEE Xplore, DBLP, bioRxiv/medRxiv, Google Scholar
2. THE Multi_Source_Ingestor SHALL 각 학술 소스에 대해 독립적인 Source_Adapter를 사용하여 해당 소스의 API를 호출한다
3. WHEN Source_Adapter가 논문 메타데이터를 수집하면, THE Source_Adapter SHALL 수집된 데이터를 Unified_Paper_Schema로 변환한다
4. THE Unified_Paper_Schema SHALL 다음 필드를 포함한다: paperId, source, title, abstract, authors, doi, publishedDate, category, citationCount, sourceUrl
5. WHEN 동일 논문이 여러 소스에서 수집되면, THE Multi_Source_Ingestor SHALL DOI 또는 제목 기반으로 중복을 감지하고 가장 풍부한 메타데이터를 가진 레코드를 우선 저장한다
6. WHEN arXiv Source_Adapter가 논문을 수집하면, THE arXiv Source_Adapter SHALL arXiv API(export.arxiv.org)를 사용하여 컴퓨터 과학, 물리학, 수학 분야의 논문 메타데이터를 조회한다
7. WHEN Semantic Scholar Source_Adapter가 논문을 수집하면, THE Semantic Scholar Source_Adapter SHALL Semantic Scholar Academic Graph API를 사용하여 논문 메타데이터와 인용 수를 조회한다
8. WHEN CrossRef Source_Adapter가 논문을 수집하면, THE CrossRef Source_Adapter SHALL CrossRef REST API를 사용하여 DOI 기반 저널 논문 메타데이터를 조회한다
9. WHEN IEEE Xplore Source_Adapter가 논문을 수집하면, THE IEEE Xplore Source_Adapter SHALL IEEE Xplore API를 사용하여 공학 및 컴퓨터 과학 분야의 논문 메타데이터를 조회한다
10. WHEN DBLP Source_Adapter가 논문을 수집하면, THE DBLP Source_Adapter SHALL DBLP API를 사용하여 컴퓨터 과학 서지 정보를 조회한다
11. WHEN bioRxiv/medRxiv Source_Adapter가 논문을 수집하면, THE bioRxiv/medRxiv Source_Adapter SHALL bioRxiv/medRxiv API를 사용하여 생물학 및 의학 프리프린트 메타데이터를 조회한다
12. WHEN Google Scholar Source_Adapter가 논문을 수집하면, THE Google Scholar Source_Adapter SHALL 보조 소스로서 다른 소스에서 검색되지 않은 논문을 보완 수집한다
13. IF 특정 Source_Adapter의 API 호출이 실패하면, THEN THE Multi_Source_Ingestor SHALL 해당 소스의 오류를 로깅하고 나머지 소스의 수집을 계속 진행한다
14. IF 특정 Source_Adapter의 API 호출이 연속 3회 실패하면, THEN THE Multi_Source_Ingestor SHALL 해당 소스를 일시적으로 비활성화하고 다음 수집 주기에서 재시도한다
15. THE Multi_Source_Ingestor SHALL 각 소스별 수집 결과(성공 건수, 실패 건수, 중복 건수)를 로그로 기록한다
16. WHEN 수집이 완료되면, THE PaperHub_System SHALL 수집된 논문을 DynamoDB papers 테이블에 Unified_Paper_Schema 형식으로 저장한다
17. THE Frontend SHALL 논문 목록에서 각 논문의 출처 소스를 시각적 배지로 표시한다
18. FOR ALL 유효한 Unified_Paper_Schema 객체에 대해, 직렬화 후 역직렬화하면 원본과 동일한 객체가 생성된다 (라운드트립 속성)

---

### 요구사항 11: 요약 가독성 향상 UI/UX (Summary Readability Design)

**사용자 스토리:** 연구자로서, AI 논문 요약을 읽을 때 명확한 시각적 구조와 가독성 높은 디자인으로 핵심 내용을 빠르게 파악하고 싶다.

#### 인수 조건

1. THE Summary_Display SHALL 요약 텍스트를 배경/목적, 방법론, 결과, 결론의 4개 섹션으로 구분하여 표시한다
2. THE Summary_Display SHALL 각 섹션을 독립적인 Summary_Section_Card로 렌더링하고, 각 카드에 섹션 제목과 구분선을 포함한다
3. THE Summary_Display SHALL 각 Summary_Section_Card에 섹션별 고유 색상 코드를 적용한다: 배경/목적(파란색 계열), 방법론(보라색 계열), 결과(초록색 계열), 결론(주황색 계열)
4. THE Summary_Display SHALL 명확한 제목 계층 구조를 적용한다: 논문 제목(24px 이상), 섹션 제목(18px 이상), 본문 텍스트(14px 이상)
5. THE Summary_Display SHALL 본문 텍스트의 줄 간격을 1.6 이상으로 설정하고, 섹션 간 여백을 16px 이상으로 유지한다
6. WHEN 요약에서 핵심 발견(Key Findings)이 식별되면, THE Summary_Display SHALL 해당 텍스트를 강조 색상 배경으로 하이라이트 처리한다
7. THE Summary_Display SHALL 다크 모드 테마에서 WCAG AA 기준(4.5:1 이상)의 텍스트 대비율을 유지한다
8. THE Summary_Display SHALL 화면 너비 768px 이하에서 Summary_Section_Card를 단일 컬럼 레이아웃으로 전환하고, 터치 영역을 44px 이상으로 유지한다
9. THE Summary_Display SHALL 각 Summary_Section_Card에 접기/펼치기 토글 기능을 제공한다
10. WHEN 사용자가 Summary_Section_Card의 접기/펼치기 토글을 클릭하면, THE Summary_Display SHALL 해당 섹션의 본문을 애니메이션과 함께 접거나 펼친다
11. THE Summary_Display SHALL 요약 페이지 상단에 시각적 읽기 진행률 표시기를 제공한다
12. WHEN 사용자가 요약 페이지를 스크롤하면, THE Summary_Display SHALL 읽기 진행률 표시기를 현재 스크롤 위치에 비례하여 업데이트한다
13. THE Summary_Display SHALL 각 Summary_Section_Card에 해당 섹션 텍스트를 클립보드에 복사하는 버튼을 제공한다
14. THE Summary_Display SHALL 요약 전체를 공유할 수 있는 공유 버튼을 제공한다
15. WHEN 사용자가 복사 버튼을 클릭하면, THE Summary_Display SHALL 해당 섹션 텍스트를 클립보드에 복사하고 복사 완료 피드백을 0.5초 이상 표시한다
