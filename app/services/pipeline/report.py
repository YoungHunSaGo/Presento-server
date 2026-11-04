"""최종 리포트 조립(명세 5장 N4 응답 그대로).

각 모듈 결과를 명세 5장 JSON으로 모은 뒤, 명세를 옮긴 ReportResponse 모델로 검증한다.
검증은 "모델로 읽고 다시 쓴 JSON이 조립한 JSON과 똑같은지"로 한다. 빠진 필드는 검증 오류,
명세에 없는 필드나 형식이 다른 값은 왕복 결과가 달라져 ReportInvalid가 된다.
P2·P3·P7이 쓰는 요약값(overallScore, scores, fillerCount, syllablesPerMin)은 이 JSON에서 꺼낸다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from app.models.enums import Purpose, ScoreCategory
from app.schemas.report import (
    CategoryScore,
    Metrics,
    NonverbalMetrics,
    Overall,
    ReportResponse,
)
from app.services.pipeline.content import ContentAnalysis
from app.services.pipeline.feedback import CATEGORY_LABEL, FeedbackResult
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.scoring import ScoreResult
from app.services.pipeline.speech_metrics import SpeechAnalysis

CATEGORY_ORDER = (ScoreCategory.DELIVERY, ScoreCategory.NONVERBAL, ScoreCategory.CONTENT)


class ReportInvalid(Exception):
    """조립한 리포트가 명세 형식과 다르다(코드 버그)."""


@dataclass(frozen=True)
class ReportParts:
    analysis_id: str
    presentation_id: str
    attempt_no: int
    purpose: Purpose
    duration_sec: float
    created_at: datetime
    scores: ScoreResult
    speech: SpeechAnalysis
    nonverbal: NonverbalAnalysis | None
    content: ContentAnalysis
    feedback: FeedbackResult


def _nonverbal_metrics(nv: NonverbalAnalysis | None) -> NonverbalMetrics:
    if nv is None:
        return NonverbalMetrics(
            gaze_front_ratio=None, posture_stability=None, gesture_per_min=None,
            low_confidence_ranges=[],
        )  # fmt: skip
    return nv.metrics


def build_report(p: ReportParts) -> dict[str, Any]:
    """명세 5장 리포트 JSON(camelCase). 검증을 통과한 것만 돌려준다."""
    fb = p.feedback
    model = ReportResponse(
        analysis_id=p.analysis_id,
        presentation_id=p.presentation_id,
        attempt_no=p.attempt_no,
        purpose=p.purpose,
        duration_sec=round(p.duration_sec, 2),
        created_at=p.created_at,
        overall=Overall(
            score=p.scores.overall,
            grade=p.scores.grade,
            summary=fb.overall.summary,
            strengths=fb.overall.strengths,
            improvements=fb.overall.improvements,
        ),
        scores=[
            CategoryScore(category=c, label=CATEGORY_LABEL[c], score=p.scores.categories[c])
            for c in CATEGORY_ORDER
        ],
        metrics=Metrics(speech=p.speech.metrics, nonverbal=_nonverbal_metrics(p.nonverbal)),
        feedbacks=fb.feedbacks,
        transcript=p.speech.transcript,
        improved_scripts=fb.improved_scripts,
        expected_questions=p.content.expected_questions,
        slide_coverage=p.content.slide_coverage,
    )
    data = model.model_dump(mode="json", by_alias=True)
    validate_report(data)
    return data


def validate_report(data: dict[str, Any]) -> ReportResponse:
    """명세 모델로 읽고 다시 쓴 결과가 원래 JSON과 같아야 한다."""
    try:
        model = ReportResponse.model_validate(data)
    except ValidationError as e:
        raise ReportInvalid(f"report does not match spec: {e.error_count()} error(s)") from None
    if model.model_dump(mode="json", by_alias=True) != data:
        raise ReportInvalid("report has fields or values outside the spec")
    _check_invariants(model)
    return model


def _check_invariants(r: ReportResponse) -> None:
    """명세 표에 적힌 값 범위."""
    problems: list[str] = []
    if not 0 <= r.overall.score <= 100:
        problems.append("overall.score")
    if [s.category for s in r.scores] != list(CATEGORY_ORDER):
        problems.append("scores order")
    if any(not 0 <= s.score <= 100 for s in r.scores):
        problems.append("scores range")
    for f in r.feedbacks:
        if not (0 <= f.start < f.end <= r.duration_sec + 0.01):
            problems.append(f"feedback {f.id} range")
    nv = r.metrics.nonverbal
    for v in (nv.gaze_front_ratio, nv.posture_stability):
        if v is not None and not 0 <= v <= 1:
            problems.append("nonverbal ratio")
    if problems:
        raise ReportInvalid("report invariants failed: " + ", ".join(problems))


def history_values(data: dict[str, Any]) -> dict[str, Any]:
    """P2·P3·P7이 쓰는 회차 요약값(Analysis 컬럼)."""
    return {
        "overall_score": data["overall"]["score"],
        "scores": {s["category"]: s["score"] for s in data["scores"]},
        "filler_count": data["metrics"]["speech"]["fillerCount"],
        "syllables_per_min": data["metrics"]["speech"]["syllablesPerMin"],
    }
