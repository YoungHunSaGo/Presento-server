"""M1~M3 마이페이지 API (docs/mypage-spec.md)."""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.models import Analysis, AnalysisStatus, Presentation, User
from app.models.enums import Purpose
from app.schemas.common import ErrorResponse
from app.schemas.mypage import (
    MyInsightsResponse,
    MyStatsResponse,
    Period,
    ScoreTimelineResponse,
)
from app.services.mypage import (
    DEFAULT_TIMELINE_LIMIT,
    MAX_TIMELINE_LIMIT,
    CompletedRecord,
    build_insights,
    build_score_timeline,
    build_stats,
)
from app.services.pipeline.artifacts import backfill_analysis_summary
from app.services.pipeline.scoring import grade_for

router = APIRouter(prefix="/users/me", tags=["mypage"])

_DOCS = {
    400: {"model": ErrorResponse, "description": "VALIDATION_ERROR"},
    401: {"model": ErrorResponse, "description": "UNAUTHORIZED"},
}


def _query_rows(db: Session, user_id: str) -> list:
    stmt = (
        select(
            Analysis.id,
            Analysis.presentation_id,
            Presentation.title,
            Presentation.purpose,
            Analysis.attempt_no,
            Analysis.created_at,
            Analysis.completed_at,
            Analysis.duration_sec,
            Analysis.overall_score,
            Analysis.grade,
            Analysis.scores,
            Analysis.metric_details,
            Analysis.filler_counts,
            Analysis.feedback_counts,
        )
        .join(Presentation, Presentation.id == Analysis.presentation_id)
        .where(
            Presentation.user_id == user_id,
            Analysis.status == AnalysisStatus.COMPLETED,
            Analysis.overall_score.is_not(None),
            Analysis.scores.is_not(None),
        )
        .order_by(Analysis.completed_at.asc(), Analysis.created_at.asc(), Analysis.id.asc())
    )
    return list(db.execute(stmt).all())


def _load_completed_records(db: Session, user_id: str) -> list[CompletedRecord]:
    """사용자의 완료된 회차 요약 컬럼만 단일 조인 쿼리로 읽는다(Report.data는 읽지 않음).

    만약 마이그레이션 0006 직후 아직 백필되지 않은 레코드(metric_details가 None)가 발견되면
    최초 1회 자동 백필하여 지표 누락을 막는다.
    """
    rows = _query_rows(db, user_id)
    if any(r[11] is None for r in rows):
        backfill_analysis_summary(db, dry_run=False, allow_old_version=True)
        rows = _query_rows(db, user_id)
    out: list[CompletedRecord] = []
    for row in rows:
        (
            aid,
            pid,
            title,
            purpose_raw,
            attempt_no,
            created_at,
            completed_at,
            duration_sec,
            overall_score,
            grade,
            scores,
            metric_details,
            filler_counts,
            feedback_counts,
        ) = row
        comp = completed_at or created_at
        score_int = int(overall_score)
        out.append(
            CompletedRecord(
                analysis_id=aid,
                presentation_id=pid,
                presentation_title=title,
                purpose=Purpose(purpose_raw),
                attempt_no=int(attempt_no),
                created_at=created_at,
                completed_at=comp,
                duration_sec=float(duration_sec or 0.0),
                overall_score=score_int,
                grade=grade or grade_for(score_int),
                scores=dict(scores or {}),
                metric_details=dict(metric_details or {}),
                filler_counts=list(filler_counts or []),
                feedback_counts=dict(feedback_counts or {}),
            )
        )
    return out


@router.get("/stats", response_model=MyStatsResponse, responses=_DOCS)
def get_my_stats(
    period: Period = Query(Period.ALL),
    purpose: Purpose | None = Query(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MyStatsResponse:
    records = _load_completed_records(db, user.id)
    return build_stats(records, period=period, purpose=purpose, now=datetime.now(UTC))


@router.get("/insights", response_model=MyInsightsResponse, responses=_DOCS)
def get_my_insights(
    period: Period = Query(Period.ALL),
    purpose: Purpose | None = Query(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> MyInsightsResponse:
    records = _load_completed_records(db, user.id)
    return build_insights(records, period=period, purpose=purpose, now=datetime.now(UTC))


@router.get("/score-timeline", response_model=ScoreTimelineResponse, responses=_DOCS)
def get_my_score_timeline(
    period: Period = Query(Period.ALL),
    purpose: Purpose | None = Query(None),
    limit: int = Query(DEFAULT_TIMELINE_LIMIT, ge=1, le=MAX_TIMELINE_LIMIT),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ScoreTimelineResponse:
    records = _load_completed_records(db, user.id)
    return build_score_timeline(
        records, period=period, purpose=purpose, limit=limit, now=datetime.now(UTC)
    )
