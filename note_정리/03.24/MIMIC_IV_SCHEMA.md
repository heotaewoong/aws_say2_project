# MIMIC-IV 데이터 스키마 정리 (실제 데이터 기반)

> 경로: `data/mimic-iv/`, `data/mimic-iv-note/`
> 총 11개 테이블 (hosp 6 + icu 3 + note 2)
> 실제 파일을 읽어서 컬럼, 샘플, 통계를 정리함

---

## 📁 hosp/ — 병원 입원 데이터

### 1. patients.csv.gz (364,627명)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `subject_id` | int | 10000032 | 환자 고유 ID (PK) |
| `gender` | str | F | 성별 (M/F) |
| `anchor_age` | int | 52 | anchor_year 기준 나이 |
| `anchor_year` | int | 2180 | 기준 연도 (프라이버시 보호용 시프트) |
| `anchor_year_group` | str | 2014-2016 | 실제 연도 그룹 |
| `dod` | datetime | (빈값) | 사망일 (없으면 빈값) |

---

### 2. admissions.csv.gz (546,028건)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `subject_id` | int | 10000032 | 환자 ID (FK) |
| `hadm_id` | int | 22595853 | 입원 ID (PK) |
| `admittime` | datetime | 2180-05-06 22:23 | 입원 시각 |
| `dischtime` | datetime | 2180-05-07 17:15 | 퇴원 시각 |
| `deathtime` | datetime | (빈값) | 원내 사망 시각 |
| `admission_type` | str | EMERGENCY | 입원 유형 |
| `admit_provider_id` | str | - | 입원 담당 의사 ID |
| `admission_location` | str | EMERGENCY ROOM | 입원 경로 |
| `discharge_location` | str | HOME | 퇴원 후 행선지 |
| `insurance` | str | Medicare | 보험 종류 |
| `language` | str | ENGLISH | 환자 언어 |
| `marital_status` | str | MARRIED | 결혼 상태 |
| `race` | str | WHITE | 인종 |
| `edregtime` | datetime | - | 응급실 등록 시각 |
| `edouttime` | datetime | - | 응급실 퇴실 시각 |
| `hospital_expire_flag` | int | 0 | 원내 사망 여부 (0/1) |

---

### 3. diagnoses_icd.csv.gz (6,364,520건)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `subject_id` | int | 10000032 | 환자 ID |
| `hadm_id` | int | 22595853 | 입원 ID |
| `seq_num` | int | 1 | 진단 순서 (1=주진단) |
| `icd_code` | str | 5723 | ICD 코드 |
| `icd_version` | int | 9 | ICD 버전 (9 또는 10) |

---

### 4. d_icd_diagnoses.csv.gz (112,107개)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `icd_code` | str | 0010 | ICD 코드 (PK) |
| `icd_version` | int | 9 | ICD 버전 |
| `long_title` | str | Cholera due to vibrio cholerae | 진단명 |

---

### 5. labevents.csv.gz (158,478,383건) ⚠️ 매우 큰 파일

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `labevent_id` | int | 1 | 검사 이벤트 ID (PK) |
| `subject_id` | int | 10000032 | 환자 ID |
| `hadm_id` | int | - | 입원 ID (외래는 빈값) |
| `specimen_id` | int | 2704548 | 검체 ID |
| `itemid` | int | 50931 | 검사 항목 ID (FK → d_labitems) |
| `order_provider_id` | str | - | 처방 의사 ID |
| `charttime` | datetime | - | 검사 시각 |
| `storetime` | datetime | - | 결과 저장 시각 |
| `value` | str | 7.4 | 결과값 (문자열) |
| `valuenum` | float | 7.4 | 결과값 (숫자) |
| `valueuom` | str | mEq/L | 단위 |
| `ref_range_lower` | float | - | 정상 범위 하한 |
| `ref_range_upper` | float | - | 정상 범위 상한 |
| `flag` | str | abnormal | 이상 여부 |
| `priority` | str | ROUTINE | 우선순위 |
| `comments` | str | - | 비고 |

---

### 6. d_labitems.csv.gz (1,650개)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `itemid` | int | 50801 | 검사 항목 ID (PK) |
| `label` | str | Alveolar-arterial Gradient | 검사명 |
| `fluid` | str | Blood | 검체 종류 |
| `category` | str | Blood Gas | 카테고리 |

---

## 📁 icu/ — 중환자실 데이터

### 7. icustays.csv.gz (94,458건)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `subject_id` | int | 10000032 | 환자 ID |
| `hadm_id` | int | 29079034 | 입원 ID |
| `stay_id` | int | 39553978 | ICU 입실 ID (PK) |
| `first_careunit` | str | MICU | 최초 ICU 유형 |
| `last_careunit` | str | MICU | 마지막 ICU 유형 |
| `intime` | datetime | 2180-07-23 | ICU 입실 시각 |
| `outtime` | datetime | 2180-07-25 | ICU 퇴실 시각 |
| `los` | float | 2.3 | ICU 재원 기간 (일) |

**ICU 유형**: MICU(내과), SICU(외과), CVICU(심혈관), NICU(신생아), TSICU(외상)

---

### 8. chartevents.csv.gz ⚠️ 매우 큰 파일 (수억 건)

실제 샘플 데이터:

| subject_id | stay_id | charttime | itemid | value | valueuom |
|-----------|---------|-----------|--------|-------|----------|
| 10000032 | 39553978 | 2180-07-23 12:36 | 226512 | 39.4 | kg (체중) |
| 10000032 | 39553978 | 2180-07-23 12:36 | 226707 | 60 | Inch (키) |
| 10000032 | 39553978 | 2180-07-23 14:00 | 220048 | SR (Sinus Rhythm) | (심장리듬) |
| 10000032 | 39553978 | 2180-07-23 14:00 | 223761 | 98.7 | °F (체온) |
| 10000032 | 39553978 | 2180-07-23 14:11 | 220179 | 84 | mmHg (수축기혈압) |
| 10000032 | 39553978 | 2180-07-23 14:11 | 220180 | 48 | mmHg (이완기혈압) |
| 10000032 | 39553978 | 2180-07-23 14:11 | 220181 | 56 | mmHg (평균혈압) |

**컬럼 설명**:

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `subject_id` | int | 환자 ID |
| `hadm_id` | int | 입원 ID |
| `stay_id` | int | ICU 입실 ID |
| `caregiver_id` | int | 담당 간호사/의사 ID |
| `charttime` | datetime | 기록 시각 |
| `storetime` | datetime | 저장 시각 |
| `itemid` | int | 측정 항목 ID (FK → d_items) |
| `value` | str | 측정값 (문자열) |
| `valuenum` | float | 측정값 (숫자, 없으면 빈값) |
| `valueuom` | str | 단위 |
| `warning` | int | 경고 여부 (0/1) |

---

### 9. d_items.csv.gz (4,095개 항목)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `itemid` | int | 항목 ID (PK) |
| `label` | str | 항목명 |
| `abbreviation` | str | 약어 |
| `linksto` | str | 연결 테이블 |
| `category` | str | 카테고리 |
| `unitname` | str | 단위 |
| `param_type` | str | 파라미터 타입 |
| `lownormalvalue` | float | 정상 하한 |
| `highnormalvalue` | float | 정상 상한 |

**카테고리별 항목 수 (주요)**:

| 카테고리 | 항목 수 |
|---------|--------|
| Skin - Impairment | 412 |
| Access Lines - Invasive | 372 |
| Respiratory | 170 |
| Labs | 161 |
| Medications | 139 |
| Routine Vital Signs | 50 |
| Neurological | 88 |
| Pain/Sedation | 97 |

**주요 활력징후 itemid (자주 사용)**:

| itemid | label | 단위 | 설명 |
|--------|-------|------|------|
| 220045 | Heart Rate | bpm | 심박수 |
| 220050 | Arterial Blood Pressure systolic | mmHg | 수축기 혈압 |
| 220051 | Arterial Blood Pressure diastolic | mmHg | 이완기 혈압 |
| 220052 | Arterial Blood Pressure mean | mmHg | 평균 혈압 |
| 220179 | Non Invasive Blood Pressure systolic | mmHg | 비침습 수축기 혈압 |
| 220180 | Non Invasive Blood Pressure diastolic | mmHg | 비침습 이완기 혈압 |
| 220181 | Non Invasive Blood Pressure mean | mmHg | 비침습 평균 혈압 |
| 220210 | Respiratory Rate | insp/min | 호흡수 |
| 220277 | O2 saturation pulseoxymetry | % | SpO2 (산소포화도) |
| 223761 | Temperature Fahrenheit | °F | 체온 (화씨) |
| 223762 | Temperature Celsius | °C | 체온 (섭씨) |
| 220739 | GCS - Eye Opening | - | GCS 눈 반응 |
| 223900 | GCS - Verbal Response | - | GCS 언어 반응 |
| 223901 | GCS - Motor Response | - | GCS 운동 반응 |
| 226512 | Admission Weight (Kg) | kg | 입원 시 체중 |
| 226730 | Height (cm) | cm | 키 |
| 220048 | Heart Rhythm | - | 심장 리듬 |

---

## � mimic-iv-note/ — 임상 노트 (비정형 텍스트)

### 10. discharge.csv (퇴원 요약)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `note_id` | str | 10000032-DS-21 | 노트 ID (PK) |
| `subject_id` | int | 10000032 | 환자 ID |
| `hadm_id` | int | 22595853 | 입원 ID |
| `note_type` | str | DS | 노트 유형 (DS=Discharge Summary) |
| `note_seq` | int | 21 | 노트 순서 |
| `charttime` | datetime | 2180-05-07 | 작성 시각 |
| `storetime` | datetime | 2180-05-09 | 저장 시각 |
| `text` | str | Name: ___ Unit No: ___ ... | 퇴원 요약 전문 (자유 텍스트) |

**활용**: NLP, 진단명 추출, 임상 결과 예측

---

### 11. radiology.csv (영상 판독 보고서)

| 컬럼 | 타입 | 샘플 | 설명 |
|------|------|------|------|
| `note_id` | str | 10000032-RR-14 | 노트 ID (PK) |
| `subject_id` | int | 10000032 | 환자 ID |
| `hadm_id` | int | 22595853 | 입원 ID |
| `note_type` | str | RR | 노트 유형 (RR=Radiology Report) |
| `note_seq` | int | 14 | 노트 순서 |
| `charttime` | datetime | 2180-05-06 21:19 | 촬영 시각 |
| `storetime` | datetime | 2180-05-06 23:32 | 저장 시각 |
| `text` | str | EXAMINATION: CHEST (PA AND LAT)... | 영상 판독 전문 |

**활용**: 흉부 X-ray 판독 NLP, CheXpert 레이블과 연계

---

## 🔗 테이블 관계도

```
patients (subject_id)
    │
    ├── admissions (subject_id → hadm_id)
    │       ├── diagnoses_icd (hadm_id) → d_icd_diagnoses (icd_code)
    │       ├── labevents (hadm_id) → d_labitems (itemid)
    │       ├── discharge.csv (hadm_id)  ← 퇴원 요약 텍스트
    │       ├── radiology.csv (hadm_id)  ← 영상 판독 텍스트
    │       └── icustays (hadm_id → stay_id)
    │               └── chartevents (stay_id) → d_items (itemid)
```

---

## 💡 주요 분석 시나리오

### 1. 패혈증 코호트 추출
```python
import pandas as pd, gzip

diag = pd.read_csv('data/mimic-iv/hosp/diagnoses_icd.csv.gz')
d_icd = pd.read_csv('data/mimic-iv/hosp/d_icd_diagnoses.csv.gz')

# ICD-10 A41.x (패혈증)
sepsis = diag[diag['icd_code'].str.startswith('A41') & (diag['icd_version']==10)]
print(f"패혈증 입원: {sepsis['hadm_id'].nunique():,}건")
```

### 2. ICU 사망률 분석
```python
icu = pd.read_csv('data/mimic-iv/icu/icustays.csv.gz')
adm = pd.read_csv('data/mimic-iv/hosp/admissions.csv.gz')

icu_mort = icu.merge(adm[['hadm_id','hospital_expire_flag']], on='hadm_id')
print(f"ICU 사망률: {icu_mort['hospital_expire_flag'].mean():.1%}")
```

### 3. 활력징후 시계열 추출 (청크 처리)
```python
vital_items = [220045, 220050, 220051, 220210, 220277, 223761]  # HR, BP, RR, SpO2, Temp

chunks = pd.read_csv('data/mimic-iv/icu/chartevents.csv.gz', chunksize=500000)
vitals = []
for chunk in chunks:
    filtered = chunk[chunk['itemid'].isin(vital_items)]
    vitals.append(filtered)
vitals_df = pd.concat(vitals)
```

### 4. 특정 혈액검사 추출
```python
labitems = pd.read_csv('data/mimic-iv/hosp/d_labitems.csv.gz')

# Creatinine (신장 기능)
creat_id = labitems[labitems['label']=='Creatinine']['itemid'].values[0]

chunks = pd.read_csv('data/mimic-iv/hosp/labevents.csv.gz', chunksize=500000)
creat_data = []
for chunk in chunks:
    creat_data.append(chunk[chunk['itemid']==creat_id])
creat_df = pd.concat(creat_data)
```

### 5. 퇴원 요약 텍스트 + 진단 연계
```python
notes = pd.read_csv('data/mimic-iv-note/discharge.csv')
diag = pd.read_csv('data/mimic-iv/hosp/diagnoses_icd.csv.gz')

# 주진단(seq_num=1)과 퇴원 요약 조인
main_diag = diag[diag['seq_num']==1]
notes_with_diag = notes.merge(main_diag, on='hadm_id')
```

---

## ⚠️ 주의사항

1. **날짜 시프트**: 실제 날짜가 아닌 프라이버시 보호용 시프트 날짜. 상대적 시간 차이는 유효
2. **ICD 버전 혼재**: `icd_version` 컬럼으로 9/10 구분 필수
3. **대용량 파일**: `labevents`(1.5억), `chartevents`(수억) → 청크 처리 필수
4. **결측값**: `hadm_id`가 빈 경우 있음 (외래 검사 등)
5. **텍스트 익명화**: 노트에서 환자명, 날짜 등은 `___`로 마스킹됨
6. **mimic-cxr 연계**: `data/mimic-cxr-2.0.0-chexpert.csv`로 흉부 X-ray 레이블 연계 가능
