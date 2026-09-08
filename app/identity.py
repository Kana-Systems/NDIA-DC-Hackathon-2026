"""Demo and OIDC identity adapters for principal-aware retrieval."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from fastapi import HTTPException, status
from jwt import PyJWKClient, PyJWTError

from app.config import Settings
from app.models import PrincipalContext


def issue_demo_token(
    username: str,
    password: str,
    settings: Settings,
) -> str:
    """Issue a short-lived local token; this path is explicitly non-production."""

    if not settings.demo_identity_enabled or settings.identity_mode != "demo":
        raise HTTPException(status_code=404, detail="Demo identity is disabled")
    valid_username = secrets.compare_digest(
        username.encode(),
        settings.gradio_username.encode(),
    )
    valid_password = secrets.compare_digest(
        password.encode(),
        settings.gradio_password.get_secret_value().encode(),
    )
    if not (valid_username and valid_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid demo credentials",
        )
    now = datetime.now(UTC)
    payload = {
        "sub": username,
        "iss": "contract-review-demo",
        "aud": "j2-intelligence-demo",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=30)).timestamp()),
        "groups": ["contract-reviewers", "mission-analysts"],
        "security_domain": settings.security_domain,
        "scope": "rag:query entities:read objects:draft objects:review",
    }
    return jwt.encode(
        payload,
        settings.demo_jwt_secret.get_secret_value(),
        algorithm="HS256",
    )


def decode_principal(token: str, settings: Settings) -> PrincipalContext:
    try:
        if settings.identity_mode == "demo":
            if not settings.demo_identity_enabled:
                raise _unauthorized()
            payload = jwt.decode(
                token,
                settings.demo_jwt_secret.get_secret_value(),
                algorithms=["HS256"],
                audience="j2-intelligence-demo",
                issuer="contract-review-demo",
            )
        else:
            if not all((settings.oidc_issuer, settings.oidc_audience, settings.oidc_jwks_url)):
                raise RuntimeError("OIDC identity settings are incomplete")
            signing_key = PyJWKClient(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=list(settings.oidc_algorithms),
                audience=settings.oidc_audience,
                issuer=settings.oidc_issuer,
            )
    except PyJWTError as exc:
        raise _unauthorized() from exc
    return _principal_from_claims(payload, settings)


def _principal_from_claims(
    payload: dict[str, Any],
    settings: Settings,
) -> PrincipalContext:
    subject = str(payload.get("sub") or "").strip()
    if not subject:
        raise _unauthorized()
    raw_groups = payload.get("groups", [])
    groups = (
        [str(group) for group in raw_groups if str(group).strip()]
        if isinstance(raw_groups, list)
        else []
    )
    raw_scope = payload.get("scope", "")
    scopes = str(raw_scope).split() if raw_scope else []
    domain = str(payload.get("security_domain") or settings.security_domain)
    if domain != settings.security_domain:
        raise _unauthorized()
    return PrincipalContext(
        subject=subject,
        groups=groups,
        security_domain=domain,
        scopes=scopes,
    )


def _unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Valid bearer authentication is required",
        headers={"WWW-Authenticate": "Bearer"},
    )
