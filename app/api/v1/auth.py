"""A1 회원가입, A2 로그인, A3 내 정보."""

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import (
    burn_password_check,
    create_access_token,
    hash_password,
    verify_password,
)
from app.models import User
from app.schemas.auth import (
    AuthResponse,
    LoginRequest,
    SignupRequest,
    UserResponse,
    normalize_email,
)
from app.schemas.common import ErrorResponse

router = APIRouter(tags=["auth"])

_UNAUTHORIZED_DOC = {401: {"model": ErrorResponse, "description": "UNAUTHORIZED"}}


def _email_exists() -> AppError:
    return AppError("EMAIL_ALREADY_EXISTS", "이미 가입된 이메일이에요.", 409)


def _invalid_credentials() -> AppError:
    return AppError("INVALID_CREDENTIALS", "이메일 또는 비밀번호가 맞지 않아요.", 401)


def _auth_response(user: User, settings: Settings) -> AuthResponse:
    return AuthResponse(
        access_token=create_access_token(user.id, settings),
        user=UserResponse.model_validate(user),
    )


# bcrypt가 CPU를 쓰므로 async가 아닌 def로 두어 스레드풀에서 돌게 한다.
@router.post(
    "/auth/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"model": ErrorResponse, "description": "EMAIL_ALREADY_EXISTS"}},
)
def signup(
    body: SignupRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    email = normalize_email(body.email)
    # bcrypt를 DB 트랜잭션 밖에서 먼저 돌려 커넥션을 오래 잡지 않는다.
    password_hash = hash_password(body.password, settings.bcrypt_rounds)
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise _email_exists()

    user = User(email=email, password_hash=password_hash, nickname=body.nickname)
    db.add(user)
    try:
        db.commit()
    except IntegrityError as e:
        db.rollback()
        # 동시에 같은 이메일로 가입한 경우만 409로 바꾸고 나머지 제약 위반은 그대로 올린다.
        # PostgreSQL은 제약 이름, SQLite는 "users.email"을 메시지에 담는다.
        msg = str(e.orig)
        if "uq_users_email" in msg or "users.email" in msg:
            raise _email_exists() from None
        raise
    db.refresh(user)
    return _auth_response(user, settings)


@router.post(
    "/auth/login",
    response_model=AuthResponse,
    responses={401: {"model": ErrorResponse, "description": "INVALID_CREDENTIALS"}},
)
def login(
    body: LoginRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == normalize_email(body.email)))
    if user is not None:
        db.expunge(user)
    # bcrypt 비교 전에 트랜잭션을 끝내 커넥션을 풀에 돌려준다.
    db.rollback()

    if user is None:
        burn_password_check(body.password, settings.bcrypt_rounds)
        raise _invalid_credentials()
    if not verify_password(body.password, user.password_hash):
        raise _invalid_credentials()
    return _auth_response(user, settings)


@router.get("/users/me", response_model=UserResponse, responses=_UNAUTHORIZED_DOC)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user)
