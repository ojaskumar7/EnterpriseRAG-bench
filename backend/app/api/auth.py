"""JWT authentication and request scope validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated
from uuid import UUID

import jwt
from fastapi import Header, HTTPException, Request, status

from app.settings import Settings


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: UUID
    knowledge_base_ids: frozenset[UUID]


async def get_current_principal(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Principal:
    settings: Settings = request.app.state.settings
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    token = authorization[7:].strip()
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing bearer token")
    if not settings.jwt_secret:
        raise HTTPException(status_code=500, detail="JWT_SECRET is not configured")

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            options={"require": ["sub", "tenant_id", "knowledge_base_ids", "exp"]},
        )
        subject = str(payload["sub"])
        tenant_id = UUID(str(payload["tenant_id"]))
        knowledge_base_ids = frozenset(UUID(str(value)) for value in payload["knowledge_base_ids"])
    except (jwt.InvalidTokenError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid bearer token",
        ) from exc

    if not subject or not knowledge_base_ids:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="token scope is incomplete",
        )
    return Principal(subject, tenant_id, knowledge_base_ids)


def require_knowledge_base(principal: Principal, knowledge_base_id: UUID) -> None:
    if knowledge_base_id not in principal.knowledge_base_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="knowledge base is not authorized for this token",
        )
