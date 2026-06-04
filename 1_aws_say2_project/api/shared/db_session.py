"""Async SQLAlchemy 엔진 + 세션 — Aurora PostgreSQL.

문서: §5.4
환경변수: DATABASE_URL (예: postgresql+asyncpg://user:pwd@host:5432/rarelinkai)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def _database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        # 개발 fallback — sqlite (인-메모리 안 되는 모델은 구조만 검증)
        url = "sqlite+aiosqlite:///./rarelinkai_dev.db"
    return url


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_async_engine(_database_url(), echo=False, future=True, pool_pre_ping=True)
    return _engine


def get_sessionmaker():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(get_engine(), expire_on_commit=False, class_=AsyncSession)
    return _SessionLocal


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """프레임워크 외부 코드 (Lambda 등) 가 세션을 쓸 때."""
    SessionLocal = get_sessionmaker()
    async with SessionLocal() as s:
        try:
            yield s
            await s.commit()
        except Exception:
            await s.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI Depends 용 — app/deps.py 에서 re-export."""
    SessionLocal = get_sessionmaker()
    async with SessionLocal() as s:
        try:
            yield s
        finally:
            await s.close()
