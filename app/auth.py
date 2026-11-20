"""JWT issuing (dev only) and verification.

company_id is read exclusively from the verified token. No request field can override it.

Status codes:
- 401: no token, bad signature, wrong alg/issuer, expired, missing claims (not authenticated)
- 403: valid token, but unknown company or role, or role not allowed here (not authorised)
"""

from datetime import UTC, datetime, timedelta
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, Request, status
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
        options={"require": ["sub", "company_id", "role", "exp", "iss"]},
    )
    return Principal(sub=claims["sub"], company_id=claims["company_id"], role=claims["role"])


def current_principal(
    request: Request,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> Principal:
    # The app's own settings (set at startup), so tests and server never disagree on secrets.
    s: Settings = getattr(request.app.state, "settings", None) or get_settings()
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    try:
        principal = decode_token(creds.credentials, s)
    except jwt.PyJWTError as exc:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token") from exc
    if principal.company_id not in s.companies:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Unknown company")
    if principal.role not in s.roles:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Unknown role")
    return principal


def require_admin(principal: Annotated[Principal, Depends(current_principal)]) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return principal
