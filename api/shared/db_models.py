"""SQLAlchemy ORM 모델 — Aurora PostgreSQL `rarelinkai` 스키마.

문서: §5.4, §5.6
스키마는 production schema 일치. Phase Lambda 와 main backend 가 같은 모델 import.
실제 DB 연결은 db_session.py 의 engine 사용. 마이그레이션은 alembic 권장.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()
SCHEMA = "rarelinkai"


# ----------------------------------------------------------------
# Diagnosis Session · 한 번의 진단 요청 = 한 행
# ----------------------------------------------------------------
class DiagnosisSession(Base):
    __tablename__ = "diagnosis_session"
    __table_args__ = {"schema": SCHEMA}

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_fhir_id = Column(String(64), nullable=False, index=True)
    clinician_id    = Column(String(64), nullable=False, index=True)
    symptom_text    = Column(Text)
    cxr_s3_key      = Column(String(512))
    status          = Column(String(32), default="created", nullable=False, index=True)
    execution_arn   = Column(String(512))  # Step Functions ARN
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ----------------------------------------------------------------
# Phase 1~5 Result · session_id FK 로 연결
# ----------------------------------------------------------------
class Phase1Result(Base):
    """증상 → HPO 추출."""
    __tablename__ = "phase1_result"
    __table_args__ = {"schema": SCHEMA}

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id   = Column(UUID(as_uuid=True),
                          ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    positive_hpo = Column(JSON)  # ["HP:0012735", ...]
    negative_hpo = Column(JSON)
    raw_response = Column(JSON)  # Bedrock raw response (디버그용)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class Phase2Result(Base):
    """CXR 14-label 추론."""
    __tablename__ = "phase2_result"
    __table_args__ = {"schema": SCHEMA}

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id   = Column(UUID(as_uuid=True),
                          ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    findings     = Column(JSON)  # [{label, score}, ...]
    cxr_s3_key   = Column(String(512))
    model_version = Column(String(64))
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class Phase3Result(Base):
    """LIRICAL LR 점수."""
    __tablename__ = "phase3_result"
    __table_args__ = {"schema": SCHEMA}

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id    = Column(UUID(as_uuid=True),
                           ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    lr_scores     = Column(JSON)  # [{orpha_code, name, lr_score}, ...]
    candidates_total = Column(Integer)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)


class Phase4Result(Base):
    """Bedrock 검증."""
    __tablename__ = "phase4_result"
    __table_args__ = {"schema": SCHEMA}

    id            = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id    = Column(UUID(as_uuid=True),
                           ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                           nullable=False, index=True)
    verifications = Column(JSON)  # [{candidate, confidence, evidence}, ...]
    raw_response  = Column(JSON)
    created_at    = Column(DateTime, default=datetime.utcnow, nullable=False)


class Phase5Result(Base):
    """RAG 케이스 검색."""
    __tablename__ = "phase5_result"
    __table_args__ = {"schema": SCHEMA}

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id   = Column(UUID(as_uuid=True),
                          ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                          nullable=False, index=True)
    citations    = Column(JSON)  # [{title, source, url, relevance}, ...]
    summary_text = Column(Text)
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


class FinalRagReport(Base):
    """최종 통합 리포트 — Phase 5 가 작성, Frontend `/result` 가 조회."""
    __tablename__ = "final_rag_report"
    __table_args__ = {"schema": SCHEMA}

    id             = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id     = Column(UUID(as_uuid=True),
                            ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                            nullable=False, unique=True, index=True)
    final_dx       = Column(String(256))
    confidence     = Column(String(16))
    full_report_md = Column(Text)
    generated_at   = Column(DateTime, default=datetime.utcnow, nullable=False)


# ----------------------------------------------------------------
# Patient cache · HAPI 에서 import 한 환자 정보 보관
# ----------------------------------------------------------------
class PatientCache(Base):
    __tablename__ = "patient_cache"
    __table_args__ = {"schema": SCHEMA}

    fhir_id       = Column(String(64), primary_key=True)
    name_masked   = Column(String(64))
    sex           = Column(String(8))
    birth_date    = Column(String(16))
    cached_payload = Column(JSON)  # 정규화된 환자 detail (PatientDetail 모양)
    last_synced   = Column(DateTime, default=datetime.utcnow, nullable=False)


class FhirBundleArchive(Base):
    """원본 FHIR Bundle 보존 (감사용)."""
    __tablename__ = "fhir_bundle_archive"
    __table_args__ = {"schema": SCHEMA}

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    fhir_id     = Column(String(64), nullable=False, index=True)
    bundle      = Column(JSON, nullable=False)
    archived_at = Column(DateTime, default=datetime.utcnow, nullable=False)


# ----------------------------------------------------------------
# Feedback · Audit
# ----------------------------------------------------------------
class Feedback(Base):
    __tablename__ = "feedback"
    __table_args__ = {"schema": SCHEMA}

    id                = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id        = Column(UUID(as_uuid=True),
                               ForeignKey(f"{SCHEMA}.diagnosis_session.id", ondelete="CASCADE"),
                               nullable=False, index=True)
    clinician_id      = Column(String(64), nullable=False)
    final_dx_correct  = Column(Boolean, nullable=False)
    correction        = Column(String(256))
    note              = Column(Text)
    created_at        = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    """누가 언제 무엇을 봤는지 — HIPAA · 의료 데이터 접근 기록."""
    __tablename__ = "audit_log"
    __table_args__ = {"schema": SCHEMA}

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    clinician_id = Column(String(64), nullable=False, index=True)
    session_id   = Column(UUID(as_uuid=True), nullable=True, index=True)
    patient_fhir_id = Column(String(64), nullable=True, index=True)
    action       = Column(String(64), nullable=False)
    payload      = Column(JSON)  # extra metadata
    ip_addr      = Column(String(64))
    user_agent   = Column(String(256))
    created_at   = Column(DateTime, default=datetime.utcnow, nullable=False)
