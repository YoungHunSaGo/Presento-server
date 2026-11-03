"""회차 상태 보정: 워커가 죽거나 큐 메시지가 사라져 멈춘 회차를 FAILED로 끝낸다.

프론트는 N3를 COMPLETED/FAILED가 될 때까지 폴링하므로, 멈춘 회차가 영원히
'분석 중'으로 남지 않게 N3 조회 시점에 확인한다.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from app.models import Analysis, AnalysisStatus

# 작업 강제 종료 시간(16분)보다 길게 잡는다.
STALE_PROCESSING = timedelta(minutes=20)
# 대기열이 밀려도 이 정도면 충분하다고 본다.
STALE_QUEUED = timedelta(minutes=30)

STALE_ERROR_CODE = "INTERNAL_ERROR"
STALE_ERROR_MESSAGE = "분석이 제시간에 끝나지 않았어요. 다시 시도해 주세요."


def _aware(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def is_stale(a: Analysis, now: datetime) -> bool:
    if a.status == AnalysisStatus.PROCESSING:
        started = _aware(a.started_at)
        return started is not None and now - started > STALE_PROCESSING
    if a.status == AnalysisStatus.QUEUED:
        queued = _aware(a.queued_at)
        return queued is not None and now - queued > STALE_QUEUED
    return False


def fail_if_stale(db: Session, a: Analysis, now: datetime | None = None) -> bool:
    """멈춘 회차면 조건부 UPDATE로 FAILED 처리하고 True.

    그 사이 상태가 바뀌었으면 건드리지 않는다.
    """
    now = now or datetime.now(UTC)
    if not is_stale(a, now):
        return False
    result = db.execute(
        update(Analysis)
        .where(
            Analysis.id == a.id,
            or_(
                (Analysis.status == AnalysisStatus.PROCESSING)
                & (Analysis.started_at < now - STALE_PROCESSING),
                (Analysis.status == AnalysisStatus.QUEUED)
                & (Analysis.queued_at < now - STALE_QUEUED),
            ),
        )
        .values(
            status=AnalysisStatus.FAILED,
            stage=None,
            error_code=STALE_ERROR_CODE,
            error_message=STALE_ERROR_MESSAGE,
        )
        # 세션 객체와 맞추려고 WHERE를 파이썬에서 다시 평가하지 않게 한다(아래에서 refresh).
        .execution_options(synchronize_session=False)
    )
    db.commit()
    db.refresh(a)
    return result.rowcount == 1
