"""진단 세션 라우터.

문서: §5.6, §6.3 — Phase 1~5 비동기 파이프라인 트리거 + 폴링 조회.

엔드포인트:
  POST   /api/v1/sessions             · 새 세션 생성 (Step Functions 시작 X)
  POST   /api/v1/sessions/{id}/run    · 파이프라인 실행 시작 (202 Accepted)
  GET    /api/v1/sessions/{id}        · 세션 상태 + Phase 결과 (frontend 가 2초 간격 polling)
  GET    /api/v1/sessions/{id}/result · 최종 통합 리포트
  POST   /api/v1/sessions/{id}/rerun  · 재진단 (새 세션 생성)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...shared.db_models import (
    DiagnosisSession, FinalRagReport,
    Phase1Result, Phase2Result, Phase3Result, Phase4Result, Phase5Result,
)
from ...shared.schemas import (
    SessionCreateRequest, SessionCreateResponse,
    SessionDetailResponse, SessionFinalReport, SessionStatus,
)
from ..config import Settings
from ..deps import Clinician, get_current_clinician, get_db
from ..services.audit_log import log_session_access
from ..services.stepfunctions import start_diagnosis_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


def _settings() -> Settings:
    return Settings.from_env()


def _progress_from_phases(*phases) -> float:
    done = sum(1 for p in phases if p is not None)
    return round(done / max(1, len(phases)), 2)


@router.post("", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    payload: SessionCreateRequest,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> SessionCreateResponse:
    sess = DiagnosisSession(
        patient_fhir_id=payload.patient_fhir_id,
        clinician_id=clinician.id,
        symptom_text=payload.symptom_text,
        cxr_s3_key=payload.cxr_s3_key,
        status=SessionStatus.created.value,
    )
    db.add(sess)
    await db.commit()
    await db.refresh(sess)

    await log_session_access(db,
                             clinician_id=clinician.id,
                             session_id=sess.id,
                             patient_fhir_id=payload.patient_fhir_id,
                             action="session.create")

    return SessionCreateResponse(session_id=str(sess.id), status=SessionStatus.created)


@router.post("/{session_id}/run", status_code=status.HTTP_202_ACCEPTED)
async def run_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> dict[str, Any]:
    sess: DiagnosisSession | None = await db.get(DiagnosisSession, session_id)
    if sess is None or sess.clinician_id != clinician.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    s = _settings()
    arn = await start_diagnosis_pipeline(
        session_id=str(sess.id),
        patient_fhir_id=sess.patient_fhir_id,
        symptom_text=sess.symptom_text or "",
        cxr_s3_key=sess.cxr_s3_key,
        state_machine_arn=s.stepfn_state_machine_arn,
        region=s.aws_region,
    )
    sess.status = SessionStatus.running.value
    sess.execution_arn = arn
    sess.updated_at = datetime.utcnow()
    await db.commit()

    await log_session_access(db,
                             clinician_id=clinician.id,
                             session_id=sess.id,
                             action="session.run",
                             payload={"execution_arn": arn})

    return {"status": "started", "execution_arn": arn}


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> SessionDetailResponse:
    """문서: Frontend 가 2초 간격으로 폴링."""
    sess: DiagnosisSession | None = await db.get(DiagnosisSession, session_id)
    if sess is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if sess.clinician_id != clinician.id:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Forbidden")

    p1 = await db.scalar(select(Phase1Result).where(Phase1Result.session_id == session_id))
    p2 = await db.scalar(select(Phase2Result).where(Phase2Result.session_id == session_id))
    p3 = await db.scalar(select(Phase3Result).where(Phase3Result.session_id == session_id))
    p4 = await db.scalar(select(Phase4Result).where(Phase4Result.session_id == session_id))
    p5 = await db.scalar(select(Phase5Result).where(Phase5Result.session_id == session_id))

    await log_session_access(db,
                             clinician_id=clinician.id,
                             session_id=sess.id,
                             action="session.read_status")

    return SessionDetailResponse(
        session_id=str(sess.id),
        patient_fhir_id=sess.patient_fhir_id,
        status=SessionStatus(sess.status),
        progress=_progress_from_phases(p1, p2, p3, p4, p5),
        phase1=_to_phase1(p1) if p1 else None,
        phase2=_to_phase2(p2) if p2 else None,
        phase3=_to_phase3(p3) if p3 else None,
        phase4=_to_phase4(p4) if p4 else None,
        phase5=_to_phase5(p5) if p5 else None,
        created_at=sess.created_at,
        updated_at=sess.updated_at or sess.created_at,
    )


@router.get("/{session_id}/result", response_model=SessionFinalReport)
async def get_session_result(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> SessionFinalReport:
    sess: DiagnosisSession | None = await db.get(DiagnosisSession, session_id)
    if sess is None or sess.clinician_id != clinician.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    final: FinalRagReport | None = await db.scalar(
        select(FinalRagReport).where(FinalRagReport.session_id == session_id)
    )
    if final is None:
        raise HTTPException(status.HTTP_409_CONFLICT,
                            "Final report not yet ready. Poll /sessions/{id} until status=completed.")

    await log_session_access(db,
                             clinician_id=clinician.id,
                             session_id=sess.id,
                             action="session.read_result")

    return SessionFinalReport(
        session_id=str(sess.id),
        patient_fhir_id=sess.patient_fhir_id,
        final_dx=final.final_dx,
        confidence=final.confidence,
        full_report_md=final.full_report_md or "",
        generated_at=final.generated_at,
    )


@router.post("/{session_id}/rerun", response_model=SessionCreateResponse, status_code=status.HTTP_201_CREATED)
async def rerun_session(
    session_id: UUID,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> SessionCreateResponse:
    """재진단 = 같은 환자/증상으로 새 세션 생성. 원본 세션은 history 보존."""
    orig: DiagnosisSession | None = await db.get(DiagnosisSession, session_id)
    if orig is None or orig.clinician_id != clinician.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")

    new = DiagnosisSession(
        patient_fhir_id=orig.patient_fhir_id,
        clinician_id=clinician.id,
        symptom_text=orig.symptom_text,
        cxr_s3_key=orig.cxr_s3_key,
        status=SessionStatus.created.value,
    )
    db.add(new)
    await db.commit()
    await db.refresh(new)

    await log_session_access(db,
                             clinician_id=clinician.id,
                             session_id=new.id,
                             action="session.rerun",
                             payload={"original_session_id": str(session_id)})

    return SessionCreateResponse(session_id=str(new.id), status=SessionStatus.created)


# ---- ORM → Pydantic 변환 ----
def _to_phase1(p: Phase1Result):
    from ...shared.schemas import Phase1Result as S
    return S(positive_hpo=p.positive_hpo or [], negative_hpo=p.negative_hpo or [],
             extracted_at=p.created_at)


def _to_phase2(p: Phase2Result):
    from ...shared.schemas import Phase2Finding, Phase2Result as S
    findings = [Phase2Finding(**f) for f in (p.findings or [])]
    return S(findings=findings, cxr_s3_key=p.cxr_s3_key, inferred_at=p.created_at)


def _to_phase3(p: Phase3Result):
    from ...shared.schemas import Phase3LRScore, Phase3Result as S
    cands = [Phase3LRScore(**c) for c in (p.lr_scores or [])]
    return S(top_candidates=cands, computed_at=p.created_at)


def _to_phase4(p: Phase4Result):
    from ...shared.schemas import Phase4Result as S, Phase4Verification
    vs = [Phase4Verification(**v) for v in (p.verifications or [])]
    return S(verifications=vs, verified_at=p.created_at)


def _to_phase5(p: Phase5Result):
    from ...shared.schemas import Phase5Citation, Phase5Result as S
    cs = [Phase5Citation(**c) for c in (p.citations or [])]
    return S(citations=cs, summary_text=p.summary_text, completed_at=p.created_at)
