"""마이페이지 M1~M3 응답 스키마 (docs/mypage-spec.md 4·5장)."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from app.models.enums import Purpose, ScoreCategory
from app.schemas.common import CamelModel, KstDatetime
from app.schemas.report import FeedbackType, FillerStat


class Period(StrEnum):
    DAYS_30 = "30d"
    DAYS_90 = "90d"
    ALL = "all"


class Trend(StrEnum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class MetricKey(StrEnum):
    PACE = "PACE"
    FILLER = "FILLER"
    SILENCE = "SILENCE"
    TIME = "TIME"
    GAZE = "GAZE"
    POSTURE = "POSTURE"
    GESTURE = "GESTURE"
    STRUCTURE = "STRUCTURE"
    CLARITY = "CLARITY"
    LANGUAGE = "LANGUAGE"


class InsightStatus(StrEnum):
    READY = "READY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


GradeLiteral = Literal["A+", "A", "B+", "B", "C+", "C", "D"]


class PurposeBreakdownItem(CamelModel):
    purpose: Purpose
    analysis_count: int
    average_score: int | None


class WeeklyActivityItem(CamelModel):
    week_start: str
    analysis_count: int
    practice_sec: float
    average_score: int | None


class MyStatsResponse(CamelModel):
    period: Period
    from_: KstDatetime | None = Field(default=None, serialization_alias="from", alias="from")
    to: KstDatetime
    presentation_count: int
    analysis_count: int
    total_practice_sec: float
    average_score: int | None
    best_score: int | None
    best_analysis_id: str | None
    recent_change: int | None
    streak_days: int
    category_averages: dict[ScoreCategory, int | None]
    grade_distribution: dict[str, int]
    purpose_breakdown: list[PurposeBreakdownItem]
    weekly_activity: list[WeeklyActivityItem]


class InsightItem(CamelModel):
    metric: MetricKey
    category: ScoreCategory
    label: str
    score: int
    trend: Trend
    change: int | None
    sample_count: int
    value_text: str
    message: str


class FrequentFeedbackItem(CamelModel):
    type: FeedbackType
    label: str
    count: int


class MyInsightsResponse(CamelModel):
    status: InsightStatus
    analysis_count: int
    required_count: int
    based_on_recent: int
    strengths: list[InsightItem]
    weaknesses: list[InsightItem]
    metrics: list[InsightItem]
    focus_tip: str | None
    top_fillers: list[FillerStat]
    frequent_feedback_types: list[FrequentFeedbackItem]


class ScoreTimelineItem(CamelModel):
    analysis_id: str
    presentation_id: str
    presentation_title: str
    purpose: Purpose
    attempt_no: int
    created_at: KstDatetime
    overall_score: int
    grade: str
    scores: dict[ScoreCategory, int]


class ScoreTimelineResponse(CamelModel):
    items: list[ScoreTimelineItem]
    total: int
