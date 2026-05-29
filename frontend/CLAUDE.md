# Rare-Link AI · Project Context

이 파일은 Claude Code가 세션 시작 시 자동으로 읽는 프로젝트 헌법입니다.
향후 프롬프트와 충돌 시 이 파일의 규칙이 우선합니다.

## 1. Project Overview

- **이름**: Rare-Link AI · Clinical Decision Support for Rare Pulmonary Disease
- **소속**: SKKU AWS SAY 2기 · 2팀
- **성격**: 파이널 프로젝트 프론트엔드 프로토타입 (연구·교육용, SaMD 허가 전)
- **최종 발표**: 2026-05-28
- **현재 단계**: Final Phase · Week 1 (2026-04-20 ~ 04-26)

시스템 개요: CXR + 증상(HPO) + 검사결과 3축 입력 → 528개 폐질환 랭킹 출력.
DenseNet-121 (내부명 SooNet) + HPO 기반 Likelihood Ratio 엔진.

## 2. Team & Roles

- **박성수** (user · Frontend Lead) — 이 CLAUDE.md를 읽는 Claude Code의 주 대화 상대
- **배기태, 허태웅** — Model training, AWS infra
- **권미라, 양희인** — MIMIC-IV data extraction, KB management
- **이희찬** — AWS mentor

## 3. Tech Stack (IMMUTABLE)

결정 확정됨. 변경 시 사용자에게 재확인 필수.

- **Frontend**: React 18 + Vite + Tailwind CSS
- **UI**: shadcn/ui 스타일 커스텀 컴포넌트 (lucide-react 아이콘)
- **Typography**: IBM Plex Sans KR / Serif / Mono (Google Fonts)
- **State**: useState local · 아직 전역 상태관리 도입 안 함
- **Backend 연동 (W3+)**: SMART on FHIR v2.2 + SMART Health IT Sandbox + Synthea 합성 환자
- **추론**: AWS SageMaker Endpoint (DenseNet-121) + Lambda (HPO-LR 엔진)
- **배포**: 미정 (AWS Amplify 또는 S3+CloudFront 후보)

**금지 사항**:
- ❌ Streamlit MVP (PDF 초안의 §8.2 한계 때문에 React로 이미 이관 결정)
- ❌ Inter·Roboto·Arial 등 generic font
- ❌ 보라색 그라데이션 (AI slop aesthetic)
- ❌ localStorage/sessionStorage에 **환자 정보** 저장 (개인정보보호법·HIPAA)
  - 예외: 로그인 세션 토큰만 sessionStorage 허용 · **TTL 1시간** · `rl-session` 키 · `doctor` 프로파일 + `issuedAt` + `expiresAt`만 저장 · 환자 MRN/영상/lab/진단 절대 금지 (2026-04-23 결정)

## 4. File Structure

```
rare-link-ai-frontend/
├── CLAUDE.md                  # 이 파일
├── README.md                  # 한글 실행 가이드
├── package.json
├── vite.config.js
├── tailwind.config.js         # rl.* 컬러 토큰 정의됨
├── postcss.config.js
├── index.html                 # IBM Plex 폰트 preconnect
└── src/
    ├── main.jsx
    ├── App.jsx                # 우하단 APP↔DESIGN SYSTEM 토글 스위처
    ├── LoginWorklist.jsx      # 로그인 + 환자 목록 + 프리뷰 드로어
    ├── DesignSystem.jsx       # v0.1 디자인 시스템 쇼케이스
    └── index.css              # Tailwind directives + body font
```

## 5. Design Tokens (CSS Variables)

모든 컬러는 `:root` CSS 변수로 관리. Tailwind 클래스는 `rl-*` prefix.

| Token | Hex | 의미 |
|---|---|---|
| `--rl-primary` | `#0C447C` | 브랜드 · 1차 액션 · 환자 배너 |
| `--rl-teal` | `#0E8574` | LR 지지 · 정상 수치 · 확정 |
| `--rl-critical` | `#A32D2D` | LR 반박 · 심각 이상치 · 기각 |
| `--rl-amber` | `#B45309` | Don't miss 플래그 · 주의 |
| `--rl-rare` | `#6B21A8` | 희귀질환 배지 (Orphadata 연결) |
| `--rl-ink` | `#0A1628` | 본문 텍스트 |
| `--rl-bg-2` | `#F8FAFC` | 페이지 배경 |

## 6. Design Principles (with citations)

반드시 준수해야 하는 원칙들. 각 원칙은 논문 또는 시장 레퍼런스 근거 있음.

1. **3-Panel Layout** (좌 입력 / 중 CXR+모델 / 우 감별진단)
   - 근거: Epic Hyperdrive · BESTCare 2.0 공통 EMR 패턴. 의사 근육 기억.

2. **LR 막대 시각화** (지지 녹색 우측 / 반박 빨강 좌측)
   - 근거: Robinson et al. *Am J Hum Genet* 2020;107:403-417, Fig. 2 직접 차용.

3. **"Don't miss" 플래그**
   - 근거: Ramnarayan et al. *BMC Med Inform Decis Mak* 2003 (Isabel). 하위 랭킹이어도 위중한 희귀질환은 빨간 테두리로 구분.

4. **Human-in-the-Loop (HITL) 배너 상시 노출**
   - 근거: EU AI Act 2024/1680/EU Article 22. "본 결과는 진단 보조용이며 의사의 최종 판단이 필요합니다" 배너를 모든 AI 출력 화면 하단에 고정.

5. **Explainability 기본값 ON**
   - 근거: Neri et al. *Radiol Med* 2023;128:755-764. Heatmap overlay, LR 막대, 지지/반박 분리를 토글이 아닌 기본 표시.

6. **Sticky 환자 배너**
   - 근거: EMR 표준 · 환자 안전 직결. Streamlit에서 불가능했던 점이 React 이전의 핵심 사유.

7. **EMR 정보 밀도**
   - 근거: Epic, BESTCare 공통. 여백은 인색, hairline(1px) 구분선이 위계 담당. 시작업 SaaS의 넓은 여백 지양.

## 7. Screen Map (9 screens)

| # | 화면 | 파일 | 상태 |
|---|---|---|---|
| 01 | 로그인 (SMART SSO sim) | `src/LoginWorklist.jsx` | ✓ W1 완료 |
| 02 | 당일 외래 환자 목록 | `src/LoginWorklist.jsx` | ✓ W1 완료 |
| 03 | **진단 워크스페이스 (3-Panel)** | TBD | W2~W3 |
| 04 | **질환 상세 · LR 막대** | TBD | W3 |
| 05 | 이미지 확대 뷰어 (PACS) | TBD | W3 |
| 06 | 리포트 뷰어 | TBD | W4 |
| 07 | 유사 케이스 비교 (RAG) | TBD | W4 |
| 08 | 진단 히스토리 | TBD | optional |
| 09 | 설정 | TBD | optional |

**#03, #04 는 발표 메인 화면**. Robinson 2020 Fig. 2 수준의 LR 막대 구현이 핵심 차별점.

## 8. Mock Data Conventions

- **환자명**: `김○○` 스타일 마스킹 (의료 데모 관행 + 개인정보보호법 대응)
- **MRN 형식**: `YY-NNNNNN` (예: `20-145982`)
- **HPO ID**: `HP:0002094` 형식 (monospace 폰트)
- **Orpha ID**: `ORPHA:2032` 형식
- **날짜**: 한국 시간 KST, `08:32` 24h format
- **의사 기본값**: 정민수 과장 · 성균관대학교병원 · 호흡기내과

## 9. Roadmap (5 weeks)

| Week | 기간 | 산출물 |
|---|---|---|
| W1 (현재) | 04/20 ~ 04/26 | 디자인 시스템 · 로그인 · Worklist ✓ |
| W2 | 04/27 ~ 05/03 | 진단 워크스페이스 · SMART 샌드박스 세팅 |
| W3 | 05/04 ~ 05/10 | SageMaker 연결 · LR 상세 · 이미지 뷰어 |
| W4 | 05/11 ~ 05/17 | Heatmap · RAG · 리포트 · 폴리싱 |
| W5 | 05/18 ~ 05/24 | 시나리오 · 녹화 · 발표 연동 |
| 발표 | 05/25 ~ 05/28 | 리허설 + 최종 발표 |

## 10. Working Style (user preferences)

- **언어**: 한국어 대화. 기술 용어는 영어 원어 유지.
- **톤**: 간결하되 근거는 생략하지 말 것. 중요 결정은 논문·시장 레퍼런스 인용.
- **반복**: 수소님은 "디테일 많은 출력을 다듬는 것"을 "빈약한 출력을 확장하는 것"보다 선호.
- **제안**: 사용자가 놓친 gap을 능동적으로 flag. 예: "미라님과 상의 필요", "이 변경은 PDF §X 와 상충".
- **버전 관리**: 큰 변경 뒤엔 명시적 changelog 제공.
- **단위**: 코드 변경 시 변경 전/후 diff 또는 요약 동반.

## 11. Anti-patterns (what NOT to do)

- ❌ 변경 없이 "잘 됐어요"만 말하기 → 항상 무엇이 바뀌었는지 구체 명시
- ❌ 불필요한 emoji 남발 (사용자가 쓰거나 이전 메시지에 있을 때만)
- ❌ Streamlit·Next.js로 이관 제안 (React+Vite 확정됨)
- ❌ PDF §4.3 원칙 5가지 훼손 (HITL 배너 삭제 금지, 환자 배너 스티키 해제 금지 등)
- ❌ 희귀질환 배지와 Don't miss 플래그를 하나의 색으로 병합 (두 개념은 분리 유지)

## 12. Key References

- Robinson PN et al. *Am J Hum Genet* 2020;107:403-417 — LIRICAL LR paradigm
- Raghu G et al. *Am J Respir Crit Care Med* 2022;205:e18-e47 — IPF guideline
- Ramnarayan P et al. *BMC Med Inform Decis Mak* 2003;3:8 — Isabel DDx
- Neri E et al. *Radiol Med* 2023;128:755-764 — Explainable AI in radiology
- Baltruschat IM et al. *Eur Radiol* 2021;31:3837-3845 — AI worklist triage
- Mandel JC et al. *JAMIA* 2016;23:899-908 — SMART on FHIR
- KLAS Research 2025 — US Acute Care EHR Market Share
- EU AI Act 2024/1680/EU — Art. 22 human-in-the-loop

## 13. Project Artifacts

- **PDF 설계 문서**: `Rare-Link-AI_UIUX_설계제안서.pdf` (박성수, 2026-04-21) — 프로젝트 설계 근거 원본
- **Slack 채널**: `#skku-2기-2팀` (C0AKTN43XT4)
- **주요 캔버스**: F0AU8REJ5FB (Final 업무보고서)
- **AWS**: us-east-1, 버킷 `say2-2team-bucket`, 태그 `pre-2-2team`

---

**이 문서는 프로젝트 "헌법"입니다.** 프롬프트와 충돌 시 이 파일이 우선합니다.
변경이 필요하면 박성수와 먼저 상의 후 이 파일을 업데이트하세요.
