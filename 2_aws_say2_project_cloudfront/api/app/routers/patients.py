"""환자 정보 라우터 — HAPI FHIR proxy + cache.

문서: §6.2 — Frontend 는 HAPI 직접 호출 안 함. 모든 환자 데이터 접근은 FastAPI 경유.
이유: ① JWT 검증, ② 감사 로그, ③ 캐싱, ④ 데이터 정규화

엔드포인트:
  GET  /api/v1/patients/{fhir_id}   · 환자 detail (cache hit 우선, miss 시 HAPI fetch)
  POST /api/v1/patients/import      · HAPI 에서 강제 재 fetch + cache 갱신
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from ...shared.db_models import FhirBundleArchive, PatientCache
from ...shared.schemas import PatientDetail
from ..config import Settings
from ..deps import Clinician, get_current_clinician, get_db
from ..services import s3_emr
from ..services.audit_log import log_session_access
from ..services.hapi_client import HapiClient

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/patients", tags=["patients"])


class PatientImportRequest(BaseModel):
    patient_fhir_id: str


def _settings() -> Settings:
    return Settings.from_env()


@router.get("/{fhir_id}", response_model=PatientDetail)
async def get_patient(
    fhir_id: str,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> PatientDetail:
    s = _settings()
    # ── s3-mock: 정적 JSON 직접 fetch (데모) ─────────────────────
    if s.emr_data_source == "s3-mock":
        payload = s3_emr.get_patient(fhir_id)
        if payload is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND,
                                f"Patient {fhir_id} not found in mock EMR")
        return PatientDetail(**payload)

    # ── hapi (기본): DB cache hit ────────────────────────────────
    cache: PatientCache | None = await db.get(PatientCache, fhir_id)
    if cache is None or cache.cached_payload is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Patient not in cache. POST /patients/import first.",
        )
    await log_session_access(db,
                             clinician_id=clinician.id,
                             patient_fhir_id=fhir_id,
                             action="patient.read")
    return PatientDetail(**cache.cached_payload)


@router.post("/import", response_model=PatientDetail, status_code=status.HTTP_201_CREATED)
async def import_patient(
    body: PatientImportRequest,
    db: AsyncSession = Depends(get_db),
    clinician: Clinician = Depends(get_current_clinician),
) -> PatientDetail:
    """HAPI 에서 환자 정보 fetch → 정규화 → cache 저장."""
    s = _settings()
    if not s.fhir_base_url:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "FHIR_BASE_URL not configured")

    hapi = HapiClient(s.fhir_base_url, token=s.fhir_auth_token)
    try:
        patient_resource = await hapi.get_patient(body.patient_fhir_id)
        observations = await hapi.get_patient_observations(body.patient_fhir_id)
        imaging = await hapi.get_patient_imaging(body.patient_fhir_id)
    finally:
        await hapi.aclose()

    # 정규화 (FHIR → PatientDetail) — 실 구현은 lung_dx/api/fhirAdapter 와 같은 변환 로직
    normalized = _normalize_patient(patient_resource, observations, imaging)

    # cache 갱신
    cache: PatientCache | None = await db.get(PatientCache, body.patient_fhir_id)
    if cache is None:
        cache = PatientCache(fhir_id=body.patient_fhir_id, cached_payload=normalized.model_dump(mode="json"),
                             last_synced=datetime.utcnow())
        db.add(cache)
    else:
        cache.cached_payload = normalized.model_dump(mode="json")
        cache.last_synced = datetime.utcnow()

    # 원본 Bundle archive
    db.add(FhirBundleArchive(
        fhir_id=body.patient_fhir_id,
        bundle={"patient": patient_resource, "observations": observations, "imaging": imaging},
    ))

    await db.commit()
    await log_session_access(db,
                             clinician_id=clinician.id,
                             patient_fhir_id=body.patient_fhir_id,
                             action="patient.import")
    return normalized


def _normalize_patient(patient: dict[str, Any],
                       observations: list[dict[str, Any]],
                       imaging: list[dict[str, Any]]) -> PatientDetail:
    """FHIR resource → PatientDetail.

    TODO (백엔드 팀): 프론트의 fhirAdapter.toUIShape() 와 동일한 매핑.
    이 stub 는 최소 필드만 채워서 schema validation 통과.
    """
    from ...shared.schemas import (
        CxrStudy, LabPanelsByCategory, VitalsEntry,
    )

    name = (patient.get("name") or [{}])[0]
    given = " ".join(name.get("given") or [])
    family = name.get("family") or ""
    masked = f"{family[:1]}○○" if family else (given or "환자")

    sex = {"male": "M", "female": "F"}.get(patient.get("gender", ""), "?")
    birth = patient.get("birthDate")
    age = 0
    if birth:
        try:
            age = (datetime.utcnow() - datetime.fromisoformat(birth)).days // 365
        except Exception:
            pass

    cxr_studies = [
        CxrStudy(
            studyId=s.get("id", ""),
            capturedAt=s.get("started"),
            view=(s.get("series") or [{}])[0].get("bodySite", {}).get("display", "PA · Frontal"),
            modality=(s.get("modality") or [{}])[0].get("code", "CR"),
        )
        for s in imaging
    ]

    # 실제 vitals/labs 매핑은 TODO. 빈 배열로 schema validate.
    return PatientDetail(
        mrn=patient.get("id", ""),
        name=masked,
        sex=sex,
        age=age,
        time=datetime.now().strftime("%H:%M"),
        visit="재진",
        complaint="",
        allergy=None,
        cxr="arrived" if imaging else "pending",
        status="ready",
        rare=False,
        dontMiss=False,
        acknowledged=None,
        pendingEmrUpdates=0,
        topDx=None,
        preview=None,
        vitals=None,
        vitalsHistory=[],
        labs=LabPanelsByCategory(),
        cxrStudies=cxr_studies,
    )
