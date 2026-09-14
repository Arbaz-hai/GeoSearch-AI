from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_SECRET = os.environ.get("GEOSEARCH_TOKEN_SECRET", "geosearch-ai-local-secret-change-me")
_BEARER = HTTPBearer(auto_error=False)
_TOKEN_TTL = 86400


def create_token(user_id: str, username: str) -> str:
    """Create a signed local auth token (HMAC-SHA256). No external JWT library needed."""
    expiry = int(time.time()) + _TOKEN_TTL
    payload = f"{user_id}:{username}:{expiry}"
    sig = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{sig}"


def verify_token(credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER)) -> dict[str, Any]:
    """FastAPI dependency — validates the bearer token. Raises 401 on failure."""
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    token = credentials.credentials
    try:
        parts = token.rsplit(":", 1)
        if len(parts) != 2:
            raise ValueError("malformed")
        payload, sig = parts
        expected = hmac.new(_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            raise ValueError("invalid signature")
        user_id, username, expiry_str = payload.split(":", 2)
        if int(time.time()) > int(expiry_str):
            raise ValueError("expired")
        return {"user_id": user_id, "username": username}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
        )


def optional_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_BEARER),
) -> dict[str, Any] | None:
    """FastAPI dependency — returns user info or None (for optional auth)."""
    if credentials is None:
        return None
    try:
        return verify_token(credentials)
    except HTTPException:
        return None
