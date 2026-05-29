"""공통 의존성 — DI · 인증.

문서: §5.5 (JWT) + §5.6
production 에선 SMART on FHIR 가 발급한 RS256 JWT 를 검증.
dev mode 에서는 환경변수 DEV_BYPASS_AUTH=1 이면 dummy clinician 반환.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import AsyncIterator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ..shared.db_session import get_db as _get_db_session

logger = logging.getLogger(__name__)


@dataclass
class Clinician:
    id: str
    scope: str = "patient/*.read"
    audience: str = ""


# ---- DB ----
async def get_db() -> AsyncIterator[AsyncSession]:
    async for s in _get_db_session():
        yield s


# ---- Auth ----
def _bypass() -> bool:
    return os.getenv("DEV_BYPASS_AUTH", "0") in ("1", "true", "yes")


async def get_current_clinician(
    authorization: str | None = Header(default=None),
) -> Clinician:
    """SMART on FHIR JWT 검증.

    JWT 검증 라이브러리는 의도적으로 lazy import — 백엔드 팀이 jwt 라이브러리
    선택 (pyjwt vs python-jose) 을 마무리하지 않아도 dev mode 부팅 가능.
    """
    if _bypass():
        return Clinician(id="dev-bypass", scope="patient/*.read",
                         audience=os.getenv("JWT_AUDIENCE", "rare-link-ai"))

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing Bearer token")

    token = authorization[len("Bearer "):]

    try:
        import jwt  # type: ignore[import-not-found]
    except ImportError as e:
        # 백엔드 팀에 알림 메시지로
        logger.warning("pyjwt 미설치 — JWT 검증 우회 불가")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "JWT validation unavailable (pyjwt missing)") from e

    pub_key_path = os.getenv("JWT_PUBLIC_KEY_PATH")
    if not pub_key_path:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "JWT_PUBLIC_KEY_PATH not configured")

    try:
        with open(pub_key_path, "r", encoding="utf-8") as f:
            public_key = f.read()
        payload = jwt.decode(
            token, public_key,
            algorithms=[os.getenv("JWT_ALGORITHM", "RS256")],
            audience=os.getenv("JWT_AUDIENCE", "rare-link-ai"),
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, f"Invalid token: {e}")

    return Clinician(
        id=payload["sub"],
        scope=payload.get("scope", ""),
        audience=payload.get("aud", ""),
    )
