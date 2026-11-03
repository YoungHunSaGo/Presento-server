"""JWT 발급·검증과 비밀번호 해시."""

import base64
import hashlib
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import bcrypt
import jwt

from app.core.config import Settings

JWT_ALGORITHM = "HS256"


def _prehash(password: str) -> bytes:
    # bcrypt는 72바이트까지만 받으므로(5.x는 초과 시 에러) SHA-256으로 먼저 고정 길이로 만든다.
    return base64.b64encode(hashlib.sha256(password.encode("utf-8")).digest())


def hash_password(password: str, rounds: int) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(rounds)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("ascii"))
    except ValueError:
        return False


@lru_cache
def _dummy_hash(rounds: int) -> str:
    return hash_password("presento-dummy-password", rounds)


def warm_dummy_hash(rounds: int) -> None:
    _dummy_hash(rounds)


def burn_password_check(password: str, rounds: int) -> None:
    """없는 사용자로 로그인할 때도 같은 시간을 써서 가입 여부가 응답 시간으로 드러나지 않게 한다."""
    verify_password(password, _dummy_hash(rounds))


def create_access_token(user_id: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(hours=settings.jwt_expire_hours),
    }
    return jwt.encode(payload, settings.jwt_secret.get_secret_value(), algorithm=JWT_ALGORITHM)


def decode_access_token(token: str, settings: Settings) -> str | None:
    """유효한 토큰이면 사용자 ID, 아니면 None."""
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret.get_secret_value(),
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "iat", "sub"]},
        )
    except jwt.PyJWTError:
        return None
    sub = payload.get("sub")
    return sub if isinstance(sub, str) and sub else None
