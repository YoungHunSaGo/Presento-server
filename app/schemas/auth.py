from typing import Annotated

from email_validator import EmailNotValidError, validate_email
from pydantic import ConfigDict, EmailStr, Field, StringConstraints

from app.schemas.common import CamelModel, KstDatetime

PASSWORD_MAX_LENGTH = 128


def normalize_email(raw: str) -> str:
    """가입·로그인이 같은 규칙으로 이메일을 정규화한다(퓨니코드 도메인 → 유니코드, 소문자)."""
    try:
        return validate_email(raw.strip(), check_deliverability=False).normalized.lower()
    except EmailNotValidError:
        return raw.strip().lower()


class SignupRequest(CamelModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=PASSWORD_MAX_LENGTH)
    nickname: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=50)]


class LoginRequest(CamelModel):
    email: str = Field(min_length=1)
    password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)


class UserResponse(CamelModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    nickname: str
    created_at: KstDatetime


class AuthResponse(CamelModel):
    access_token: str
    user: UserResponse
