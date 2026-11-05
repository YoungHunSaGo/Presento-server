"""N1~N6 분석 API."""

import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Response, status
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.api.deps import (
    get_current_user,
    get_db,
    get_owned_analysis,
    get_owned_presentation,
    get_probe,
    get_queue,
    get_storage,
)
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.models import Analysis, AnalysisStatus, AnalysisUsage, Presentation, Report, User
from app.schemas.analysis import (
    AnalysisCreateRequest,
    AnalysisCreateResponse,
    AnalysisError,
    AnalysisStartResponse,
    AnalysisStatusResponse,
    VideoUrlResponse,
)
from app.schemas.common import KST, ErrorResponse
from app.schemas.presentation import UploadUrlResponse
from app.schemas.report import ReportResponse
from app.services.analysis_state import fail_if_stale
from app.services.cleanup import delete_objects_later
from app.services.media import DEMUXERS, MediaError, ProbeDuration, ProbeUnavailable
from app.services.queue import TaskQueue
from app.services.storage import Storage, StorageError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analyses"])

ALLOWED_VIDEO_TYPES = frozenset(DEMUXERS)  # video/mp4, video/quicktime, video/webm


def _docs(*extra: tuple[int, str]) -> dict:
    base = {
        401: {"model": ErrorResponse, "description": "UNAUTHORIZED"},
        403: {"model": ErrorResponse, "description": "FORBIDDEN"},
        404: {"model": ErrorResponse, "description": "NOT_FOUND"},
    }
    return base | {code: {"model": ErrorResponse, "description": desc} for code, desc in extra}


def _start_of_today_kst(now: datetime) -> datetime:
    # UTC로 바꿔 돌려준다. SQLite는 시간대를 버리고 문자열로 비교해서(저장값은 UTC),
    # KST 그대로 넘기면 KST 0~9시에 오늘 사용량이 0으로 잡힌다.
    start = now.astimezone(KST).replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(UTC)


def _used_today(db: Session, user_id: str, now: datetime) -> int:
    """오늘(KST) 시작한 분석 수. 기록은 회차를 지워도 남는다."""
    return (
        db.scalar(
            select(func.count(AnalysisUsage.id)).where(
                AnalysisUsage.user_id == user_id,
                AnalysisUsage.created_at >= _start_of_today_kst(now),
            )
        )
        or 0
    )


def _rate_limited(settings: Settings) -> AppError:
    return AppError(
        "RATE_LIMITED",
        f"오늘 분석 횟수({settings.daily_analysis_limit}회)를 모두 썼어요. "
        "내일 다시 시도해 주세요.",
        429,
    )


VIDEO_TOO_LONG_MESSAGE = "영상은 10분 이하만 분석할 수 있어요."


def _video_too_long() -> AppError:
    return AppError("VIDEO_TOO_LONG", VIDEO_TOO_LONG_MESSAGE, 422)


def _size_label(n: int) -> str:
    """바이트 수를 안내 문구용으로 쓴다. GB로 나누어떨어지면 GB, 아니면 MB."""
    gb, mb = 1024**3, 1024**2
    return f"{n // gb}GB" if n % gb == 0 else f"{n // mb}MB"


def _storage_error() -> AppError:
    return AppError(
        "INTERNAL_ERROR", "파일 저장소에 연결하지 못했어요. 잠시 후 다시 시도해 주세요.", 500
    )


# ---------- N1 ----------


@router.post(
    "/presentations/{presentation_id}/analyses",
    response_model=AnalysisCreateResponse,
    status_code=status.HTTP_201_CREATED,
    responses=_docs((413, "FILE_TOO_LARGE"), (415, "UNSUPPORTED_FILE_TYPE"), (429, "RATE_LIMITED")),
)
def create_analysis(
    body: AnalysisCreateRequest,
    p: Presentation = Depends(get_owned_presentation),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> AnalysisCreateResponse:
    # "video/webm;codecs=vp9,opus"처럼 오는 경우 기본 형식만 본다.
    content_type = body.content_type.split(";", 1)[0].strip().lower()
    if content_type not in ALLOWED_VIDEO_TYPES:
        raise AppError("UNSUPPORTED_FILE_TYPE", "mp4, mov, webm 영상만 올릴 수 있어요.", 415)
    if body.file_size > settings.max_video_bytes:
        raise AppError(
            "FILE_TOO_LARGE",
            f"영상은 {_size_label(settings.max_video_bytes)} 이하만 업로드할 수 있어요.",
            413,
        )

    now = datetime.now(UTC)
    # 차감은 N2에서 한다. 여기서는 이미 한도에 찼으면 업로드 전에 미리 알려준다.
    if _used_today(db, user.id, now) >= settings.daily_analysis_limit:
        raise _rate_limited(settings)

    # 회차 번호는 원자적으로 하나 올린다(동시 요청에도 겹치지 않고, 지운 번호를 재사용하지 않음).
    attempt_no = db.execute(
        update(Presentation)
        .where(Presentation.id == p.id)
        .values(attempt_seq=Presentation.attempt_seq + 1, updated_at=now)
        .returning(Presentation.attempt_seq)
    ).scalar_one_or_none()
    if attempt_no is None:  # 그 사이 발표가 삭제됨
        db.rollback()
        raise AppError("NOT_FOUND", "발표를 찾을 수 없어요.", 404)

    analysis = Analysis(
        presentation_id=p.id,
        attempt_no=attempt_no,
        status=AnalysisStatus.WAITING_UPLOAD,
        source=body.source,
        content_type=content_type,
        file_size=body.file_size,
        created_at=now,
    )
    db.add(analysis)
    db.flush()  # id 확정
    analysis.video_key = f"videos/{analysis.id}"

    expires = settings.video_upload_url_expires_sec
    url = storage.presign_put(analysis.video_key, content_type, expires)
    db.commit()

    return AnalysisCreateResponse(
        analysis_id=analysis.id,
        attempt_no=attempt_no,
        status=AnalysisStatus.WAITING_UPLOAD,
        upload=UploadUrlResponse(
            upload_url=url, headers={"Content-Type": content_type}, expires_in_sec=expires
        ),
    )


# ---------- N2 ----------


@router.post(
    "/analyses/{analysis_id}/start",
    response_model=AnalysisStartResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses=_docs(
        (409, "UPLOAD_NOT_FOUND"),
        (413, "FILE_TOO_LARGE"),
        (422, "VIDEO_TOO_LONG"),
        (429, "RATE_LIMITED"),
    ),
)
def start_analysis(
    a: Analysis = Depends(get_owned_analysis),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    queue: TaskQueue = Depends(get_queue),
    probe: ProbeDuration = Depends(get_probe),
    settings: Settings = Depends(get_settings),
) -> AnalysisStartResponse:
    if a.status == AnalysisStatus.FAILED and a.error_code == "VIDEO_TOO_LONG":
        raise _video_too_long()  # 다시 불러도 같은 답
    if a.status != AnalysisStatus.WAITING_UPLOAD:
        # 이미 시작한 회차: 작업을 다시 넣지 않고 현재 상태만 알려준다.
        return AnalysisStartResponse(analysis_id=a.id, status=a.status)

    key = a.video_key or ""
    try:
        size = storage.head(key)
    except StorageError:
        logger.warning("Video HEAD failed for %s", a.id, exc_info=True)
        raise _storage_error() from None
    if size is None:
        raise AppError(
            "UPLOAD_NOT_FOUND",
            "영상 업로드가 아직 끝나지 않았어요. 업로드를 마친 뒤 다시 시도해 주세요.",
            409,
        )
    if size > settings.max_video_bytes:
        # presigned PUT은 크기를 강제할 수 없어 여기서 거르고, 다시 올릴 수 있게 지운다.
        delete_objects_later(queue, storage, [key])
        raise AppError(
            "FILE_TOO_LARGE",
            f"영상은 {_size_label(settings.max_video_bytes)} 이하만 업로드할 수 있어요.",
            413,
        )

    # 길이를 헤더에서 바로 알 수 있으면 여기서 거른다. 모르면 워커가 다시 잰다.
    try:
        duration = probe(
            storage.presign_get_internal(key, 300),
            a.content_type or "",
            settings.start_probe_timeout_sec,
            False,
        )
    except (MediaError, ProbeUnavailable):
        duration = None  # 손상 여부·서버 문제는 워커가 다시 확인해 판단한다
    if duration is not None and duration > settings.max_video_seconds:
        result = db.execute(
            update(Analysis)
            .where(Analysis.id == a.id, Analysis.status == AnalysisStatus.WAITING_UPLOAD)
            .values(
                status=AnalysisStatus.FAILED,
                duration_sec=round(duration, 2),
                error_code="VIDEO_TOO_LONG",
                error_message=VIDEO_TOO_LONG_MESSAGE,
            )
        )
        db.commit()
        if result.rowcount == 1:
            # 분석하지 않을 영상이라 저장소에서 지운다.
            delete_objects_later(queue, storage, [key])
        raise _video_too_long()

    now = datetime.now(UTC)
    user_id = a.presentation.user_id
    # 하루 횟수 차감은 여기서 한다(실제 비용이 드는 시점).
    # 같은 사용자의 동시 요청이 한도를 함께 넘지 않게 사용자 행을 잠근다(PostgreSQL).
    db.execute(select(User.id).where(User.id == user_id).with_for_update())
    # 잠금을 기다리는 사이 같은 회차의 다른 N2가 먼저 시작했으면 429가 아니라 현재 상태를 준다.
    db.refresh(a)
    if a.status != AnalysisStatus.WAITING_UPLOAD:
        db.rollback()
        return AnalysisStartResponse(analysis_id=a.id, status=a.status)
    if _used_today(db, user_id, now) >= settings.daily_analysis_limit:
        db.rollback()
        raise _rate_limited(settings)

    # WAITING_UPLOAD → QUEUED 전환에 성공한 요청만 차감하고 작업을 넣는다(중복 호출 방지).
    claimed = db.execute(
        update(Analysis)
        .where(Analysis.id == a.id, Analysis.status == AnalysisStatus.WAITING_UPLOAD)
        .values(status=AnalysisStatus.QUEUED, queued_at=now)
    ).rowcount
    if claimed != 1:
        db.rollback()
        db.refresh(a)
        return AnalysisStartResponse(analysis_id=a.id, status=a.status)
    usage = AnalysisUsage(user_id=user_id, analysis_id=a.id, created_at=now)
    db.add(usage)
    db.commit()

    try:
        queue.enqueue_analysis(a.id)
    except Exception:
        logger.exception("Failed to enqueue analysis %s", a.id)
        # 다시 시도할 수 있게 상태를 되돌리고 차감도 돌려준다.
        reverted = db.execute(
            update(Analysis)
            .where(Analysis.id == a.id, Analysis.status == AnalysisStatus.QUEUED)
            .values(status=AnalysisStatus.WAITING_UPLOAD, queued_at=None)
        ).rowcount
        if reverted == 1:
            db.execute(delete(AnalysisUsage).where(AnalysisUsage.id == usage.id))
        db.commit()
        raise AppError(
            "INTERNAL_ERROR", "분석을 시작하지 못했어요. 잠시 후 다시 시도해 주세요.", 500
        ) from None

    return AnalysisStartResponse(analysis_id=a.id, status=AnalysisStatus.QUEUED)


# ---------- N3 ----------


@router.get("/analyses/{analysis_id}", response_model=AnalysisStatusResponse, responses=_docs())
def get_analysis(
    a: Analysis = Depends(get_owned_analysis), db: Session = Depends(get_db)
) -> AnalysisStatusResponse:
    # 워커가 죽어 멈춘 회차는 여기서 FAILED로 끝내 프론트 폴링이 끝나게 한다.
    fail_if_stale(db, a)
    error = None
    if a.status == AnalysisStatus.FAILED and a.error_code:
        error = AnalysisError(code=a.error_code, message=a.error_message or "")
    return AnalysisStatusResponse(
        analysis_id=a.id,
        presentation_id=a.presentation_id,
        attempt_no=a.attempt_no,
        status=a.status,
        stage=a.stage if a.status == AnalysisStatus.PROCESSING else None,
        progress=a.progress,
        error=error,
        created_at=a.created_at,
        completed_at=a.completed_at,
    )


# ---------- N4 ----------


@router.get(
    "/analyses/{analysis_id}/report",
    response_model=ReportResponse,
    responses=_docs((409, "ANALYSIS_NOT_READY")),
)
def get_report(
    a: Analysis = Depends(get_owned_analysis), db: Session = Depends(get_db)
) -> ReportResponse:
    report = db.get(Report, a.id) if a.status == AnalysisStatus.COMPLETED else None
    if report is None:
        raise AppError("ANALYSIS_NOT_READY", "분석이 아직 끝나지 않았어요.", 409)
    return ReportResponse.model_validate(report.data)


# ---------- N5 ----------


@router.get(
    "/analyses/{analysis_id}/video-url",
    response_model=VideoUrlResponse,
    responses=_docs((409, "UPLOAD_NOT_FOUND")),
)
def get_video_url(
    a: Analysis = Depends(get_owned_analysis),
    storage: Storage = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> VideoUrlResponse:
    if a.status == AnalysisStatus.WAITING_UPLOAD or not a.video_key:
        raise AppError("UPLOAD_NOT_FOUND", "아직 업로드된 영상이 없어요.", 409)
    expires = settings.video_play_url_expires_sec
    return VideoUrlResponse(
        video_url=storage.presign_get(a.video_key, expires), expires_in_sec=expires
    )


# ---------- N6 ----------


@router.delete(
    "/analyses/{analysis_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    response_model=None,
    responses=_docs(),
)
def delete_analysis(
    a: Analysis = Depends(get_owned_analysis),
    db: Session = Depends(get_db),
    storage: Storage = Depends(get_storage),
    queue: TaskQueue = Depends(get_queue),
) -> Response:
    key = a.video_key
    db.delete(a)  # 리포트는 cascade로 함께 삭제. 진행 중인 워커는 다음 갱신에서 멈춘다.
    db.commit()
    delete_objects_later(queue, storage, [key])
    return Response(status_code=status.HTTP_204_NO_CONTENT)
