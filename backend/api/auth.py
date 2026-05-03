"""
api/auth.py — JWT-based Authentication

Provides:
  - create_access_token()  : Generate signed JWT
  - get_current_user()     : FastAPI dependency to protect routes
  - optional_user()        : Same but doesn't raise if no token (for anonymous use)
"""

from __future__ import annotations
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import get_settings
from database.db import get_db, User
from database.crud import get_user_by_id

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


def create_access_token(user_id: int, username: str) -> str:
    """Create a signed JWT access token."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload = {
        "sub": str(user_id),
        "username": username,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    FastAPI dependency — extracts and validates JWT, returns User.
    Raises 401 if token is missing or invalid.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        raise credentials_exception

    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload.get("sub", 0))
    except (JWTError, ValueError):
        raise credentials_exception

    user = await get_user_by_id(db, user_id)
    if user is None or not user.is_active:
        raise credentials_exception

    return user


async def optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Like get_current_user but returns None instead of raising.
    Use for routes that support both anonymous and authenticated access.
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
        user_id = int(payload.get("sub", 0))
        return await get_user_by_id(db, user_id)
    except Exception:
        return None
