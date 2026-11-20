"""JWT issuing (dev only) and verification.

company_id is read exclusively from the verified token. No request field can override it.
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings
from app.schemas import Principal

_bearer = HTTPBearer(auto_error=False)


def issue_token(
    sub: str, company_id: str, role: str = "employee", settings: Settings | None = None
) -> str:
    s = settings or get_settings()
    now = datetime.now(UTC)
    payload = {
        "sub": sub,
        "company_id": company_id,
        "role": role,
        "iss": s.jwt_issuer,
        "iat": now,
        "exp": now + timedelta(minutes=s.jwt_ttl_minutes),
    }
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_token(token: str, settings: Settings | None = None) -> Principal:
    s = settings or get_settings()
    claims = jwt.decode(
        token,
        s.jwt_secret,
        algorithms=[s.jwt_algorithm],  # pinned: rejects alg=none / alg confusion
        issuer=s.jwt_issuer,
        options={"require": ["sub", "company_id", "exp", "iss"]},
    )
    return Principal(
        sub=claims["sub"],
        company_id=claims["company_id"],
        role=claims.get("role", "employee"),
    )


def current_principal(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        return decode_token(creds.credentials)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc


def require_admin(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return principal
