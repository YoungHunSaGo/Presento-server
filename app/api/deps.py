"""엔드포인트 공통 의존성: DB 세션, 현재 사용자."""

from collections.abc import Iterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.security import decode_access_token
from app.models import Analysis, Presentation, User
from app.services.media import ProbeDuration
from app.services.queue import TaskQueue
from app.services.storage import Storage

_bearer = HTTPBearer(auto_error=False)


def unauthorized() -> AppError:
    return AppError("UNAUTHORIZED", "로그인이 필요해요.", 401)


def get_db(request: Request) -> Iterator[Session]:
    session: Session = request.app.state.session_factory()
    try:
        yield session
    finally:
        session.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> User:
    if credentials is None:
        raise unauthorized()
    user_id = decode_access_token(credentials.credentials, settings)
    if user_id is None:
        raise unauthorized()
    user = db.get(User, user_id)
    if user is None:
        raise unauthorized()
    return user


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_owned_presentation(
    presentation_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Presentation:
    """없으면 404 NOT_FOUND, 다른 사용자의 발표면 403 FORBIDDEN."""
    presentation = db.get(Presentation, presentation_id)
    if presentation is None:
        raise AppError("NOT_FOUND", "발표를 찾을 수 없어요.", 404)
    if presentation.user_id != user.id:
        raise AppError("FORBIDDEN", "이 발표에 접근할 권한이 없어요.", 403)
    return presentation


def get_owned_analysis(
    analysis_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> Analysis:
    """없으면 404 NOT_FOUND, 다른 사용자의 회차면 403 FORBIDDEN."""
    analysis = db.get(Analysis, analysis_id)
    if analysis is None:
        raise AppError("NOT_FOUND", "분석 회차를 찾을 수 없어요.", 404)
    if analysis.presentation.user_id != user.id:
        raise AppError("FORBIDDEN", "이 분석 회차에 접근할 권한이 없어요.", 403)
    return analysis


def get_queue(request: Request) -> TaskQueue:
    return request.app.state.queue


def get_probe(request: Request) -> ProbeDuration:
    return request.app.state.probe_duration
