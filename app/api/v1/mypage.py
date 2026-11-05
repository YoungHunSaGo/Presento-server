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
from app.services.pipeline.scoring import grade_for

router = APIRouter(prefix="/users/me", tags=["mypage"])

_DOCS = {
    400: {"model": ErrorResponse, "description": "VALIDATION_ERROR"},
    401: {"model": ErrorResponse, "description": "UNAUTHORIZED"},
}


def _load_completed_records(db: Session, user_id: str) -> list[CompletedRecord]:
    """사용자의 완료된 회차 요약 컬럼만 단일 조인 쿼리로 읽는다(Report.data 조회·DB 쓰기 없음).

    마이그레이션 전 완료 회차라 metric_details·filler_counts·feedback_counts가 None이면
    '세부 항목 측정 없음'으로 두어 M1·M3 전체/카테고리 점수에는 포함하고 M2 세부 집계에서만 뺀다.
    기존 회차 백필은 CLI(`python -m app.tools.recompute --backfill`)로 수행한다.
    """
    stmt = (
        select(
            Analysis.id.label("analysis_id"),
            Analysis.presentation_id.label("presentation_id"),
            Presentation.title.label("presentation_title"),
            Presentation.purpose.label("purpose"),
            Analysis.attempt_no.label("attempt_no"),
            Analysis.created_at.label("created_at"),
            Analysis.completed_at.label("completed_at"),
            Analysis.duration_sec.label("duration_sec"),
            Analysis.overall_score.label("overall_score"),
            Analysis.grade.label("grade"),
            Analysis.scores.label("scores"),
            Analysis.metric_details.label("metric_details"),
            Analysis.filler_counts.label("filler_counts"),
            Analysis.feedback_counts.label("feedback_counts"),
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
    out: list[CompletedRecord] = []
    for row in db.execute(stmt):
        m = row._mapping
        comp = m["completed_at"] or m["created_at"]
        score_int = int(m["overall_score"])
        md = m["metric_details"]
        fc = m["filler_counts"]
        fbc = m["feedback_counts"]
        out.append(
            CompletedRecord(
                analysis_id=m["analysis_id"],
                presentation_id=m["presentation_id"],
                presentation_title=m["presentation_title"],
                purpose=Purpose(m["purpose"]),
                attempt_no=int(m["attempt_no"]),
                created_at=m["created_at"],
                completed_at=comp,
                duration_sec=float(m["duration_sec"] or 0.0),
                overall_score=score_int,
                grade=m["grade"] or grade_for(score_int),
                scores=dict(m["scores"] or {}),
                metric_details=dict(md) if md is not None else None,
                filler_counts=list(fc) if fc is not None else None,
                feedback_counts=dict(fbc) if fbc is not None else None,
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
