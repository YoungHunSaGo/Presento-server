"""P1~P7 발표 API."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy import and_, func, select, update
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_db,
    get_owned_presentation,
    get_queue,
    get_storage,
)
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.models import Analysis, AnalysisStatus, Presentation, User
from app.schemas.common import ErrorResponse
from app.schemas.presentation import (
    AttemptSummary,
    HistoryItem,
    HistoryResponse,
    PresentationCreate,
    PresentationDetailResponse,
    PresentationListResponse,
    PresentationResponse,
    PresentationUpdate,
    SlidesUploadRequest,
    UploadUrlResponse,
)
from app.services.cleanup import delete_objects_later
from app.services.queue import TaskQueue
from app.services.slides import new_slides_key, refresh_pending_slides
from app.services.storage import Storage

router = APIRouter(prefix="/presentations", tags=["presentations"])

PDF_CONTENT_TYPE = "application/pdf"
MAX_PAGE = 10_000  # 큰 OFFSET이 DB 정수 범위를 넘지 않게

_AUTH_DOCS = {401: {"model": ErrorResponse, "description": "UNAUTHORIZED"}}
_OWNED_DOCS = {
    **_AUTH_DOCS,
    403: {"model": ErrorResponse, "description": "FORBIDDEN"},
    404: {"model": ErrorResponse, "description": "NOT_FOUND"},
}


# ---------- 공통 ----------


def _stats(db: Session, ids: list[str]) -> dict[str, tuple[int, int | None]]:
    """발표별 (회차 수, 최근 완료 회차 점수). 쿼리 2번으로 N+1을 피한다."""
    if not ids:
        return {}
    counts: dict[str, int] = dict(
        db.execute(
            select(Analysis.presentation_id, func.count(Analysis.id))
            .where(Analysis.presentation_id.in_(ids))
            .group_by(Analysis.presentation_id)
        ).all()
    )
    latest = (
        select(Analysis.presentation_id, func.max(Analysis.attempt_no).label("attempt_no"))
        .where(
            Analysis.presentation_id.in_(ids),
            Analysis.status == AnalysisStatus.COMPLETED,
        )
        .group_by(Analysis.presentation_id)
        .subquery()
    )
    scores: dict[str, int | None] = dict(
        db.execute(
            select(Analysis.presentation_id, Analysis.overall_score).join(
                latest,
                and_(
                    Analysis.presentation_id == latest.c.presentation_id,
                    Analysis.attempt_no == latest.c.attempt_no,
                ),
            )
        ).all()
    )
    return {pid: (counts.get(pid, 0), scores.get(pid)) for pid in ids}


def _to_response(p: Presentation, attempt_count: int, latest_score: int | None) -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "purpose": p.purpose,
        "time_limit_sec": p.time_limit_sec,
        "has_slides": p.slides_key is not None,
        "attempt_count": attempt_count,
        "latest_score": latest_score,
        "created_at": p.created_at,
    }


def _one_response(db: Session, p: Presentation) -> PresentationResponse:
    count, score = _stats(db, [p.id])[p.id]
    return PresentationResponse(**_to_response(p, count, score))


def _refresh_slides(
    db: Session, storage: Storage, settings: Settings, presentations: list[Presentation]
) -> None:
    refresh_pending_slides(
        db,
        storage,
        presentations,
        max_bytes=settings.max_slides_bytes,
        url_expires_sec=settings.slides_upload_url_expires_sec,
    )


# ---------- P1 ----------


@router.post(
    "",
    response_model=PresentationResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_AUTH_DOCS,
)
def create_presentation(
    body: PresentationCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> PresentationResponse:
    p = Presentation(
        user_id=user.id,
        title=body.title,
        purpose=body.purpose,
        time_limit_sec=body.time_limit_sec,
    )
    db.add(p)
    db.commit()
    return PresentationResponse(**_to_response(p, 0, None))


# ---------- P2 ----------


@router.get("", response_model=PresentationListResponse, responses=_AUTH_DOCS)
def list_presentations(
    page: int = Query(1, ge=1, le=MAX_PAGE),
    size: int = Query(20, ge=1, le=100),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> PresentationListResponse:
    total = db.scalar(select(func.count(Presentation.id)).where(Presentation.user_id == user.id))
    presentations = list(
        db.scalars(
            select(Presentation)
            .where(Presentation.user_id == user.id)
            .order_by(Presentation.updated_at.desc(), Presentation.id.desc())
            .offset((page - 1) * size)
            .limit(size)
        )
    )
    _refresh_slides(db, storage, settings, presentations)
    stats = _stats(db, [p.id for p in presentations])
    items = [PresentationResponse(**_to_response(p, *stats[p.id])) for p in presentations]
    return PresentationListResponse(items=items, page=page, size=size, total=total or 0)


# ---------- P3 ----------


@router.get("/{presentation_id}", response_model=PresentationDetailResponse, responses=_OWNED_DOCS)
def get_presentation(
    p: Presentation = Depends(get_owned_presentation),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> PresentationDetailResponse:
    _refresh_slides(db, storage, settings, [p])
    analyses = list(
        db.scalars(
            select(Analysis)
            .where(Analysis.presentation_id == p.id)
            .order_by(Analysis.attempt_no.desc())
        )
    )
    completed = [a for a in analyses if a.status == AnalysisStatus.COMPLETED]
    latest_score = completed[0].overall_score if completed else None
    attempts = [
        AttemptSummary(
            analysis_id=a.id,
            attempt_no=a.attempt_no,
            status=a.status,
            overall_score=a.overall_score,
            duration_sec=a.duration_sec,
            created_at=a.created_at,
        )
        for a in analyses
    ]
    return PresentationDetailResponse(
        **_to_response(p, len(analyses), latest_score), attempts=attempts
    )


# ---------- P4 ----------


@router.patch("/{presentation_id}", response_model=PresentationResponse, responses=_OWNED_DOCS)
def update_presentation(
    body: PresentationUpdate,
    p: Presentation = Depends(get_owned_presentation),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> PresentationResponse:
    changed = False
    for field in body.model_fields_set:
        value = getattr(body, field)
        if getattr(p, field) != value:
            setattr(p, field, value)
            changed = True
    if changed:
        # "최근 수정순"은 사용자가 실제로 바꾼 경우에만 갱신한다.
        p.updated_at = datetime.now(UTC)
        db.commit()
    _refresh_slides(db, storage, settings, [p])
    return _one_response(db, p)


# ---------- P5 ----------


@router.delete(
    "/{presentation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    responses=_OWNED_DOCS,
)
def delete_presentation(
    p: Presentation = Depends(get_owned_presentation),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    queue: TaskQueue = Depends(get_queue),
) -> Response:
    keys = [p.slides_key, p.slides_pending_key] + [a.video_key for a in p.analyses]
    db.delete(p)  # 회차·리포트는 cascade로 함께 삭제
    db.commit()
    delete_objects_later(queue, storage, keys)  # S3 정리는 워커가 한다
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ---------- P6 ----------


@router.post(
    "/{presentation_id}/slides/upload-url",
    response_model=UploadUrlResponse,
    responses={
        **_OWNED_DOCS,
        413: {"model": ErrorResponse, "description": "FILE_TOO_LARGE"},
        415: {"model": ErrorResponse, "description": "UNSUPPORTED_FILE_TYPE"},
    },
)
def create_slides_upload_url(
    body: SlidesUploadRequest,
    p: Presentation = Depends(get_owned_presentation),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> UploadUrlResponse:
    if body.content_type.lower() != PDF_CONTENT_TYPE:
        raise AppError("UNSUPPORTED_FILE_TYPE", "발표 자료는 PDF 파일만 올릴 수 있어요.", 415)
    if body.file_size > settings.max_slides_bytes:
        raise AppError(
            "FILE_TOO_LARGE",
            f"발표 자료는 {settings.max_slides_bytes // (1024 * 1024)}MB 이하만 올릴 수 있어요.",
            413,
        )

    # 이전 URL로 이미 올린 파일이 있으면 대기 키를 덮어쓰기 전에 먼저 반영한다.
    _refresh_slides(db, storage, settings, [p])
    expires = settings.slides_upload_url_expires_sec

    if p.slides_key is not None:
        # 이미 확인된 슬라이드는 같은 키로 덮어쓴다 → 다시 확인(HEAD)할 필요가 없다.
        key = p.slides_key
    else:
        key = new_slides_key(p.id)
        # 아직 대기 중인 이전 키는 대개 업로드되지 않은 것이라 지울 파일이 없다.
        # URL 발급은 사용자가 보는 수정이 아니므로 updated_at을 바꾸지 않는다.
        result = db.execute(
            update(Presentation)
            .where(Presentation.id == p.id, Presentation.slides_key.is_(None))
            .values(slides_pending_key=key, slides_pending_at=datetime.now(UTC))
        )
        db.commit()
        if result.rowcount == 0:
            # 그 사이 다른 요청이 업로드를 확인했다 → 확인된 키로 덮어쓴다.
            db.refresh(p)
            key = p.slides_key or key

    url = storage.presign_put(key, PDF_CONTENT_TYPE, expires)
    return UploadUrlResponse(
        upload_url=url,
        headers={"Content-Type": PDF_CONTENT_TYPE},
        expires_in_sec=expires,
    )


# ---------- P7 ----------


@router.get("/{presentation_id}/history", response_model=HistoryResponse, responses=_OWNED_DOCS)
def get_history(
    p: Presentation = Depends(get_owned_presentation),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    analyses = db.scalars(
        select(Analysis)
        .where(
            Analysis.presentation_id == p.id,
            Analysis.status == AnalysisStatus.COMPLETED,
            # 수치를 지어내지 않도록 요약값이 모두 계산된 회차만 보여준다.
            Analysis.overall_score.is_not(None),
            Analysis.scores.is_not(None),
            Analysis.filler_count.is_not(None),
            Analysis.syllables_per_min.is_not(None),
        )
        .order_by(Analysis.attempt_no.asc())
    )
    items = [
        HistoryItem(
            attempt_no=a.attempt_no,
            analysis_id=a.id,
            created_at=a.created_at,
            overall_score=a.overall_score,
            scores=a.scores,
            filler_count=a.filler_count,
            syllables_per_min=a.syllables_per_min,
        )
        for a in analyses
    ]
    return HistoryResponse(items=items)
