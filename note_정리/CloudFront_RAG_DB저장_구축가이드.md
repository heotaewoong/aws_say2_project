# CloudFront 생성 + RAG 최종 아웃풋 + DB 저장 구축 가이드

> 담당: 허태웅 | 작성: 2026-05-13 | 완전 초보자용

---

## 목차

1. [전체 그림 먼저 이해하기](#1-전체-그림)
2. [CloudFront 생성 (권한 받은 후)](#2-cloudfront-생성)
3. [RAG 최종 아웃풋이 뭔지 이해하기](#3-rag-최종-아웃풋)
4. [DynamoDB 테이블 구조](#4-dynamodb-테이블-구조)
5. [rag_llm_3.py에 DB 저장 코드 삽입](#5-rag_llm_3py에-db-저장-코드-삽입)
6. [전체 수정 코드 (복붙용)](#6-전체-수정-코드)
7. [순서 요약 (오늘 할 일)](#7-순서-요약)

---

## 1. 전체 그림

### 지금 만들려는 것

```
[의사 브라우저]
      │ HTTPS
      ▼
[Route53] ← DNS (도메인 없으면 스킵)
      │
      ▼
[WAF] ← 해커 차단 (us-east-1에 생성)
      │
      ▼
[CloudFront] ← HTTPS + 전세계 캐시 (S3 React 앱 서빙)
      │
      ▼
[S3: say2-2team-bucket/frontend/] ← React 앱 파일들 ← 이미 업로드 완료 ✅
      
      별도로:
[RAG 파이프라인 실행 (rag_llm_3.py)]
      │
      ├→ S3에 JSON + PDF 저장
      └→ DynamoDB에 진단 결과 기록  ← 오늘 추가할 부분
```

### 왜 DynamoDB가 필요하냐

```
지금:  결과가 .json 파일로만 저장됨 → 나중에 찾을 방법 없음

추가 후:
  파일 저장    → S3 (JSON + PDF)
  검색 가능    → DynamoDB (환자 MRN, 날짜로 조회 가능)
  프론트 연동  → 프론트에서 DynamoDB로 이전 진단 기록 불러오기 가능
```

---

## 2. CloudFront 생성

> ⚠️ 먼저 강사에게 권한 요청 필요 (`cloudfront:*`, `wafv2:*`)

### 강사에게 보낼 메시지

```
안녕하세요. say2-2team 계정 프론트엔드 배포 중 아래 권한이 필요합니다.

필요 권한: cloudfront:*, wafv2:* (us-east-1)
S3에 프론트엔드 파일은 이미 업로드 완료했습니다.
(say2-2team-bucket/frontend/)
권한만 있으면 콘솔에서 5분 만에 완료됩니다. 감사합니다.
```

### 권한 받은 후 콘솔 순서

#### STEP A: CloudFront 배포 생성 (ap-northeast-2 리전 무관 — CloudFront는 글로벌)

1. AWS 콘솔 → **CloudFront** → `Create distribution`
2. **Origin 설정**
   - Origin domain: `say2-2team-bucket.s3.ap-northeast-2.amazonaws.com`
   - Origin path: `/frontend`  ← 중요! 슬래시 포함
   - Origin access: `Origin access control settings (recommended)` 선택
     - `Create new OAC` 클릭
     - Name: `say2-2team-oac`
     - Signing behavior: `Sign requests (recommended)`
     - Create
3. **뷰어 설정**
   - Viewer protocol policy: `Redirect HTTP to HTTPS`
4. **기본 루트 객체**
   - Default root object: `index.html`
5. **에러 페이지 설정** (React SPA에 필수!)
   - `Add custom error response` 클릭
   - HTTP error code: `403` → Response page path: `/index.html` → HTTP response code: `200`
   - 같은 방법으로 `404`도 추가
6. **가격 등급**
   - Price class: `Use only North America, Europe, Asia, Middle East, and Africa`
7. **태그**
   - Key: `project` / Value: `pre-cloudfront-2-2-team`
8. `Create distribution` 클릭 → **15~20분 대기** (Deploying → Enabled)

#### 생성 후 S3 버킷 정책 업데이트 (필수!)

CloudFront 배포 완료 후 노란 배너가 뜸 → `Copy policy` 클릭 → S3 콘솔로 이동

```
S3 → say2-2team-bucket → Permissions → Bucket policy → Edit
→ 복사한 내용 붙여넣기 → Save changes
```

이 정책이 없으면 CloudFront가 S3에서 파일을 못 가져옴.

#### 완료 후 기록할 것 (infra/resource_ids.md에 기입)

```
Distribution ID: E_____________
CloudFront URL: https://_____________.cloudfront.net
```

---

#### STEP B: WAF 생성 (반드시 us-east-1 리전!)

> WAF가 us-east-1인 이유: CloudFront는 글로벌 서비스 → WAF도 글로벌(us-east-1) 필수

1. **AWS 콘솔 우측 상단 리전을 `us-east-1 (미국 동부 버지니아 북부)`로 변경**
2. **WAF & Shield** → `Create web ACL`
3. 설정
   - Resource type: `Amazon CloudFront distributions`
   - Name: `say2-2team-waf`
   - Region: `Global (CloudFront)` (자동)
4. **규칙 추가** → `Add managed rule groups`
   - `AWS-AWSManagedRulesCommonRuleSet` ✅ 추가 (기본 SQL인젝션, XSS 차단)
   - `AWS-AWSManagedRulesKnownBadInputsRuleSet` ✅ 추가 (악성 입력 차단)
5. Default action: `Allow`
6. **Associated AWS resources**
   - `Add AWS resources` → CloudFront distribution 선택 (STEP A에서 만든 것)
7. **태그**: Key `project` / Value `pre-waf-2-2-team`
8. `Create web ACL`

---

## 3. RAG 최종 아웃풋

### 현재 rag_llm_3.py가 하는 일

```
환자 데이터 입력
    ↓
RAG 수집 (PubMed, Orphanet, ClinicalTrials, Monarch)
    ↓
LLM (Bedrock Claude) 호출
    ↓
LLM이 이런 JSON 텍스트 반환:
{
  "recommendation": {
    "immediate_workup": ["검사1", "검사2"],
    "specialist_referral": ["호흡기내과 MDT 의뢰"],
    "treatment_guideline": ["[ORPHA:91387] PMID: 12345678"],
    "clinical_trial_info": ["NCT12345678 — Recruiting"],
    "genetic_test": ["ACTA2, FBN1 (복수 소스 확인)"],
    "additional_lab": ["D-dimer 재측정"]
  },
  "clinical_notes": {
    "summary": "42세 남성, 흉통 및 호흡곤란으로 응급실 내원...",
    "top1_reasoning": "Positive HPO HP:0002107 기흉이 Orphanet Frequent와 일치...",
    "differential_note": "Top2 AMI는 Troponin 정상으로 배제...",
    "rag_evidence": "ACTA2 DB·API 교차검증 일치...",
    "case_comparison": "PubMed PMID 37654321 케이스와 유사...",
    "epidemiology_note": "유병률 1-9/100,000, 성인 발병",
    "disclaimer": "AI 결과는 진단 보조이며 최종 진단은 주치의의 임상 판단..."
  },
  "confidence_metrics": {
    "overall_confidence_score": 0.87,
    "rationale": "DB·API 일치율 높고 PubMed 근거 충분",
    "data_sufficiency": {
      "genomic_evidence": "High",
      "clinical_case_match": "Medium",
      "trial_availability": "Low"
    }
  }
}
    ↓
현재: .json 파일로만 저장 (로컬)
추가: S3 업로드 + PDF 생성 + DynamoDB 저장
```

---

## 4. DynamoDB 테이블 구조

### 먼저 DynamoDB 테이블 2개를 생성해야 함 (Step 18)

#### 테이블 1: 진단 이력 (say2-2team-diagnosis-history)

```
목적: 환자별 진단 기록 저장 + 나중에 "이 환자 이전 결과 보여줘" 가능

구조:
  patient_mrn (PK, String)       ← 환자 고유번호 (예: "20-145982")
  diagnosis_timestamp (SK, String) ← 진단 시간 (예: "2026-05-13T14:30:22")
  
  추가 필드들:
  diagnosis_id    ← "orpha91387_20260513_143022" 같은 고유 ID
  patient_name    ← "이환자"
  chief_complaint ← "흉통 및 호흡곤란"
  top1_disease    ← "Familial thoracic aortic aneurysm..."
  confidence      ← 0.87
  s3_json_key     ← "RAG/reports/diagnosis_report_xxx.json"
  s3_pdf_key      ← "RAG/reports/diagnosis_report_xxx.pdf"
  report_summary  ← LLM이 만든 clinical_notes.summary 텍스트
  ttl             ← (선택) 자동 삭제 기한
```

#### 테이블 2: 희귀 케이스 수집 (say2-2team-rare-case-collection)

```
목적: 희귀질환 케이스 축적 → 100개 모이면 MLOps 재학습 자동 트리거

구조:
  disease_orpha_id (PK, String) ← "ORPHA:91387"
  case_id (SK, String)          ← "20260513_143022_이환자"
  
  추가 필드들:
  pos_hpos        ← ["HP:0002107", "HP:0001640"]
  neg_hpos        ← ["HP:0001903"]
  lab_data        ← "D-dimer: 2500 ng/mL..."
  confidence      ← 0.87
  diagnosis_id    ← diagnosis-history 테이블과 연결용
```

### DynamoDB 테이블 생성하는 법 (AWS 콘솔)

```
1. DynamoDB 콘솔 → Create table

테이블 1:
  Table name: say2-2team-diagnosis-history
  Partition key: patient_mrn (String)
  Sort key: diagnosis_timestamp (String)
  Settings: Customize → On-demand → Create

테이블 2:
  Table name: say2-2team-rare-case-collection
  Partition key: disease_orpha_id (String)
  Sort key: case_id (String)
  Settings: Customize → On-demand → Create

둘 다 Tags: project = pre-dynamodb-2-2-team
```

---

## 5. rag_llm_3.py에 DB 저장 코드 삽입

### 어디를 바꾸냐

현재 파일의 맨 아래 `if __name__ == "__main__":` 블록에서
이 부분을 찾아서 교체합니다 (637~646줄):

```python
# 현재 코드 (이 부분을 교체)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"diagnosis_report_orpha91387_{timestamp}.json"

with open(filename, "w", encoding="utf-8") as f:
    f.write(final_report)
    
print(f"\n✅ 진단 레포트(JSON)가 성공적으로 저장되었습니다: {filename}")
```

### 교체할 내용 (아래 6번 섹션에 전체 코드 있음)

```
저장하는 함수 2개 추가:
  1. save_to_s3_and_dynamodb()  ← JSON + PDF S3 업로드 + DynamoDB 저장
  2. generate_pdf()             ← JSON → PDF 변환

기존 저장 코드 → 새 함수 호출로 교체
```

---

## 6. 전체 수정 코드

### 추가할 라이브러리 (requirements.txt에도 추가)

```python
pip install fpdf2
```

### rag_llm_3.py 맨 위 import에 추가

```python
# 기존 import들 아래에 추가
from fpdf import FPDF
import uuid
```

### 추가할 함수들 (if __name__ 블록 바로 위에 삽입)

```python
# =====================================================================
# PDF 생성 함수
# =====================================================================
def generate_pdf(report_json: dict, patient_input: dict, output_path: str):
    """LLM JSON 결과 → PDF 보고서 생성"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # 한글 폰트 (fpdf2 내장 폰트는 한글 미지원 → 영문으로 핵심 정보만)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Rare-Link AI Diagnostic Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)

    # 환자 기본정보
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Patient Information", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  Name: {patient_input.get('name','N/A')}  |  Age: {patient_input.get('age','N/A')}  |  Sex: {patient_input.get('sex','N/A')}", ln=True)
    pdf.cell(0, 6, f"  Visit: {patient_input.get('visit_date','N/A')} ({patient_input.get('visit_type','N/A')})", ln=True)
    pdf.cell(0, 6, f"  Chief Complaint: {patient_input.get('chief_complaint','N/A')}", ln=True)
    pdf.ln(3)

    notes = report_json.get("clinical_notes", {})
    rec   = report_json.get("recommendation", {})
    conf  = report_json.get("confidence_metrics", {})

    # 임상 요약
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. Clinical Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    summary = notes.get("summary", "N/A")
    pdf.multi_cell(0, 6, f"  {summary[:500]}")  # 500자 제한
    pdf.ln(3)

    # Top 1 진단 근거
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Top 1 Diagnosis Reasoning", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, f"  {notes.get('top1_reasoning','N/A')[:500]}")
    pdf.ln(3)

    # 권고사항
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "4. Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for item in rec.get("immediate_workup", [])[:3]:
        pdf.cell(0, 6, f"  [Workup] {item[:100]}", ln=True)
    for item in rec.get("specialist_referral", [])[:2]:
        pdf.cell(0, 6, f"  [Referral] {item[:100]}", ln=True)
    for item in rec.get("genetic_test", [])[:2]:
        pdf.cell(0, 6, f"  [Genetic] {item[:100]}", ln=True)
    pdf.ln(3)

    # RAG 근거
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "5. RAG Evidence", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, f"  {notes.get('rag_evidence','N/A')[:400]}")
    pdf.ln(3)

    # 신뢰도
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "6. Confidence Metrics", ln=True)
    pdf.set_font("Helvetica", "", 10)
    score = conf.get("overall_confidence_score", 0)
    pdf.cell(0, 6, f"  Overall Score: {score:.0%}  |  {conf.get('rationale','')[:100]}", ln=True)
    data_s = conf.get("data_sufficiency", {})
    pdf.cell(0, 6, f"  Genomic: {data_s.get('genomic_evidence','N/A')}  |  Case Match: {data_s.get('clinical_case_match','N/A')}  |  Trial: {data_s.get('trial_availability','N/A')}", ln=True)
    pdf.ln(3)

    # 면책조항
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, f"  DISCLAIMER: {notes.get('disclaimer','AI results are for diagnostic support only.')}")

    pdf.output(output_path)
    print(f"  ✅ PDF 생성: {output_path}")


# =====================================================================
# S3 + DynamoDB 저장 함수
# =====================================================================
def save_results(
    final_report: str,
    patient_input: dict,
    region: str = "ap-northeast-2",
    s3_bucket: str = "say2-2team-bucket",
    dynamo_history_table: str = "say2-2team-diagnosis-history",
    dynamo_rare_table: str = "say2-2team-rare-case-collection",
):
    """
    진단 결과를 JSON + PDF로 S3에 저장하고 DynamoDB에 기록한다.
    """
    timestamp     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    iso_timestamp = datetime.datetime.now().isoformat()
    patient_name  = patient_input.get("name", "unknown").replace(" ", "_")
    diagnosis_id  = f"{patient_name}_{timestamp}"

    # ── JSON 파싱 ────────────────────────────────────────────────
    try:
        report_json = json.loads(final_report)
    except json.JSONDecodeError:
        print("  ⚠️ LLM 응답이 유효한 JSON이 아님 — 텍스트로 저장")
        report_json = {"raw": final_report}

    # ── 로컬 파일 저장 ───────────────────────────────────────────
    local_json = f"/tmp/diagnosis_{diagnosis_id}.json"
    local_pdf  = f"/tmp/diagnosis_{diagnosis_id}.pdf"

    with open(local_json, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2)

    generate_pdf(report_json, patient_input, local_pdf)

    # ── S3 업로드 ────────────────────────────────────────────────
    s3 = boto3.client("s3", region_name=region)
    s3_json_key = f"RAG/reports/{diagnosis_id}.json"
    s3_pdf_key  = f"RAG/reports/{diagnosis_id}.pdf"

    s3.upload_file(local_json, s3_bucket, s3_json_key,
                   ExtraArgs={"ContentType": "application/json"})
    print(f"  ✅ S3 JSON: s3://{s3_bucket}/{s3_json_key}")

    s3.upload_file(local_pdf, s3_bucket, s3_pdf_key,
                   ExtraArgs={"ContentType": "application/pdf"})
    print(f"  ✅ S3 PDF:  s3://{s3_bucket}/{s3_pdf_key}")

    # ── DynamoDB 저장 ────────────────────────────────────────────
    dynamo = boto3.resource("dynamodb", region_name=region)

    # 테이블 1: 진단 이력
    notes  = report_json.get("clinical_notes", {})
    conf   = report_json.get("confidence_metrics", {})
    top_1  = patient_input.get("top_3", [{}])[0].get("name", "Unknown") if patient_input.get("top_3") else "Unknown"

    history_table = dynamo.Table(dynamo_history_table)
    history_table.put_item(Item={
        "patient_mrn":          patient_input.get("name", "N/A"),   # 실제 운영 시 MRN으로 교체
        "diagnosis_timestamp":  iso_timestamp,
        "diagnosis_id":         diagnosis_id,
        "patient_name":         patient_input.get("name", "N/A"),
        "age":                  patient_input.get("age", 0),
        "sex":                  patient_input.get("sex", "N/A"),
        "chief_complaint":      patient_input.get("chief_complaint", "N/A"),
        "top1_disease":         top_1,
        "confidence":           str(conf.get("overall_confidence_score", 0)),
        "s3_json_key":          s3_json_key,
        "s3_pdf_key":           s3_pdf_key,
        "report_summary":       notes.get("summary", "")[:500],
        "rag_evidence":         notes.get("rag_evidence", "")[:500],
    })
    print(f"  ✅ DynamoDB [{dynamo_history_table}] 저장 완료")

    # 테이블 2: 희귀 케이스 (희귀질환만)
    for candidate in patient_input.get("top_3", []):
        if "ORPHA" in candidate.get("id", ""):
            rare_table = dynamo.Table(dynamo_rare_table)
            rare_table.put_item(Item={
                "disease_orpha_id": candidate["id"],
                "case_id":          f"{iso_timestamp}_{patient_input.get('name','unknown')}",
                "disease_name":     candidate.get("name", "N/A"),
                "pos_hpos":         patient_input.get("pos_hpos", []),
                "neg_hpos":         patient_input.get("neg_hpos", []),
                "lab_data":         patient_input.get("lab_data", "N/A")[:300],
                "confidence":       str(conf.get("overall_confidence_score", 0)),
                "diagnosis_id":     diagnosis_id,
            })
            print(f"  ✅ DynamoDB [{dynamo_rare_table}] 희귀케이스 저장: {candidate['id']}")

    return {
        "diagnosis_id": diagnosis_id,
        "s3_json":      f"s3://{s3_bucket}/{s3_json_key}",
        "s3_pdf":       f"s3://{s3_bucket}/{s3_pdf_key}",
    }
```

### if __name__ == "__main__": 블록에서 교체

```python
# ── 기존 저장 코드 (삭제) ─────────────────────
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# filename = f"diagnosis_report_orpha91387_{timestamp}.json"
# with open(filename, "w", encoding="utf-8") as f:
#     f.write(final_report)
# print(f"\n✅ 진단 레포트(JSON)가 성공적으로 저장되었습니다: {filename}")

# ── 새 저장 코드 (추가) ───────────────────────
print("\n💾 결과 저장 중 (S3 + DynamoDB)...")
saved = save_results(
    final_report   = final_report,
    patient_input  = patient_orpha_91387,
    region         = REGION_NAME,
    s3_bucket      = "say2-2team-bucket",
    dynamo_history_table = "say2-2team-diagnosis-history",
    dynamo_rare_table    = "say2-2team-rare-case-collection",
)
print(f"\n✅ 저장 완료:")
print(f"   JSON: {saved['s3_json']}")
print(f"   PDF:  {saved['s3_pdf']}")
print(f"   ID:   {saved['diagnosis_id']}")
```

---

## 7. 순서 요약 (오늘 할 일)

### A. DynamoDB 테이블 먼저 만들기 (콘솔, 5분)

```
DynamoDB 콘솔 → Create table × 2

테이블 1: say2-2team-diagnosis-history
  PK: patient_mrn (String), SK: diagnosis_timestamp (String)
  태그: project = pre-dynamodb-2-2-team

테이블 2: say2-2team-rare-case-collection
  PK: disease_orpha_id (String), SK: case_id (String)
  태그: project = pre-dynamodb-2-2-team
```

### B. fpdf2 설치 (터미널, 1분)

```bash
pip install fpdf2
```

### C. rag_llm_3.py 수정 (위 코드 삽입, 10분)

```
1. import 맨 위에:  from fpdf import FPDF, uuid 추가
2. if __name__ 블록 위에:  generate_pdf() + save_results() 함수 추가
3. if __name__ 블록 안:  기존 저장 코드 → save_results() 호출로 교체
```

### D. Lambda Role에 DynamoDB 권한 추가 (IAM 콘솔, 3분)

```
IAM → say2-2team-lambda-role → Add permissions → Create inline policy

{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "dynamodb:PutItem",
      "dynamodb:GetItem",
      "dynamodb:Query"
    ],
    "Resource": [
      "arn:aws:dynamodb:ap-northeast-2:666803869796:table/say2-2team-diagnosis-history",
      "arn:aws:dynamodb:ap-northeast-2:666803869796:table/say2-2team-rare-case-collection"
    ]
  }]
}

Policy name: DynamoDBAccess
```

### E. CloudFront + WAF (강사 권한 승인 후)

```
위 2번 섹션 참고
```

### F. 테스트

```bash
python rag_llm_3.py
# 성공 시:
# ✅ S3 JSON: s3://say2-2team-bucket/RAG/reports/이환자_20260513_143022.json
# ✅ S3 PDF:  s3://say2-2team-bucket/RAG/reports/이환자_20260513_143022.pdf
# ✅ DynamoDB [say2-2team-diagnosis-history] 저장 완료
# ✅ DynamoDB [say2-2team-rare-case-collection] 희귀케이스 저장: ORPHA:91387
```

---

## 비용 참고

| 리소스 | 비용 |
|--------|------|
| DynamoDB (온디맨드) | ~$0.00/월 (소량) |
| S3 PDF/JSON 저장 | $0.023/GB/월 |
| CloudFront | $0.0085/10,000건 |
| WAF | $5/WebACL/월 + $1/백만 요청 |
| **합계** | **~$6/월 (데모 수준)** |

---

> 문의: 허태웅 | 최종 업데이트: 2026-05-13
