"""N4 리포트 스키마 (docs/api-spec.md 5장)."""

from enum import StrEnum

from app.models.enums import Purpose, ScoreCategory
from app.schemas.common import CamelModel, KstDatetime


class FeedbackType(StrEnum):
    PACE_FAST = "PACE_FAST"
    PACE_SLOW = "PACE_SLOW"
    FILLER = "FILLER"
    LONG_SILENCE = "LONG_SILENCE"
    OVER_TIME = "OVER_TIME"
    UNDER_TIME = "UNDER_TIME"
    GAZE_AWAY = "GAZE_AWAY"
    POSTURE_UNSTABLE = "POSTURE_UNSTABLE"
    GESTURE_NONE = "GESTURE_NONE"
    GESTURE_EXCESS = "GESTURE_EXCESS"
    STRUCTURE = "STRUCTURE"
    CLARITY = "CLARITY"
    SLIDE_MISSED = "SLIDE_MISSED"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class WordTag(StrEnum):
    FILLER = "FILLER"
    REPEAT = "REPEAT"


class Overall(CamelModel):
    score: int
    grade: str
    summary: str
    strengths: list[str]
    improvements: list[str]


class CategoryScore(CamelModel):
    category: ScoreCategory
    label: str
    score: int


class PacePoint(CamelModel):
    t: int
    value: int


class FillerStat(CamelModel):
    word: str
    count: int


class SpeechMetrics(CamelModel):
    syllables_per_min: int
    pace_series: list[PacePoint]
    filler_count: int
    fillers: list[FillerStat]
    silence_count: int
    longest_silence_sec: float
    time_limit_sec: int | None
    over_time_sec: float


class TimeRange(CamelModel):
    start: float
    end: float


class NonverbalMetrics(CamelModel):
    # 측정 불가 시 null(명세 5장)
    gaze_front_ratio: float | None
    posture_stability: float | None
    gesture_per_min: float | None
    low_confidence_ranges: list[TimeRange]


class Metrics(CamelModel):
    speech: SpeechMetrics
    nonverbal: NonverbalMetrics


class Feedback(CamelModel):
    id: str
    category: ScoreCategory
    type: FeedbackType
    severity: Severity
    start: float
    end: float
    title: str
    description: str
    evidence: str | None
    suggestion: str


class Word(CamelModel):
    text: str
    start: float
    end: float
    tag: WordTag | None


class Segment(CamelModel):
    id: str
    start: float
    end: float
    text: str
    words: list[Word]


class Transcript(CamelModel):
    segments: list[Segment]


class ImprovedScript(CamelModel):
    segment_id: str
    start: float
    end: float
    original: str
    improved: str
    reason: str


class ExpectedQuestion(CamelModel):
    question: str
    intent: str
    answer_tip: str


class SlideCoverage(CamelModel):
    page: int
    covered: bool
    note: str | None


class ReportResponse(CamelModel):
    analysis_id: str
    presentation_id: str
    attempt_no: int
    purpose: Purpose
    duration_sec: float
    created_at: KstDatetime
    overall: Overall
    scores: list[CategoryScore]
    metrics: Metrics
    feedbacks: list[Feedback]
    transcript: Transcript
    improved_scripts: list[ImprovedScript]
    expected_questions: list[ExpectedQuestion]
    slide_coverage: list[SlideCoverage] | None
