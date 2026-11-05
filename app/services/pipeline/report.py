"""최종 리포트 조립(명세 5장 N4 응답 그대로).

각 모듈 결과를 명세 5장 JSON으로 모은 뒤, 명세를 옮긴 ReportResponse 모델로 검증한다.
검증은 "모델로 읽고 다시 쓴 JSON이 조립한 JSON과 똑같은지"로 한다. 빠진 필드는 검증 오류,
명세에 없는 필드나 형식이 다른 값은 왕복 결과가 달라져 ReportInvalid가 된다.
P2·P3·P7이 쓰는 요약값(overallScore, scores, fillerCount, syllablesPerMin)은 이 JSON에서 꺼낸다.
"""

import math
from collections import Counter
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
    SpeechMetrics,
    Transcript,
)
from app.services.pipeline.content import ContentAnalysis, ContentIssue, IssueType
from app.services.pipeline.feedback import CATEGORY_LABEL, FeedbackResult
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.scoring import ScoreResult, build_metric_details, grade_for
from app.services.pipeline.speech_metrics import SpeechAnalysis, analyze_speech

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
    problems += [f"{path} not 2 decimals" for path, v in _time_values(r) if not _two_decimals(v)]
    if problems:
        raise ReportInvalid("report invariants failed: " + ", ".join(problems))


def _two_decimals(v: float) -> bool:
    if not math.isfinite(v):
        return False
    return abs(v * 100 - round(v * 100)) < 1e-6


def _time_values(r: ReportResponse) -> list[tuple[str, float]]:
    """리포트의 모든 시간 값(초)과 위치."""
    out: list[tuple[str, float]] = [("durationSec", r.duration_sec)]
    sp = r.metrics.speech
    out += [("longestSilenceSec", sp.longest_silence_sec), ("overTimeSec", sp.over_time_sec)]
    for i, rg in enumerate(r.metrics.nonverbal.low_confidence_ranges):
        out += [
            (f"lowConfidenceRanges[{i}].start", rg.start),
            (f"lowConfidenceRanges[{i}].end", rg.end),
        ]
    for f in r.feedbacks:
        out += [(f"feedbacks.{f.id}.start", f.start), (f"feedbacks.{f.id}.end", f.end)]
    for seg in r.transcript.segments:
        out += [(f"{seg.id}.start", seg.start), (f"{seg.id}.end", seg.end)]
        for j, w in enumerate(seg.words):
            out += [(f"{seg.id}.words[{j}].start", w.start), (f"{seg.id}.words[{j}].end", w.end)]
    for s in r.improved_scripts:
        out += [(f"improvedScripts.{s.segment_id}.start", s.start)]
        out += [(f"improvedScripts.{s.segment_id}.end", s.end)]
    return out


def _metric_details_from_report_json(
    data: dict[str, Any],
) -> dict[str, dict[str, float | None] | None]:
    """아티팩트 없이 리포트 JSON만 있을 때 세부 항목 {score, value}를 계산한다."""
    duration = float(data["durationSec"])
    sp_data = data["metrics"]["speech"]
    nv_data = data["metrics"]["nonverbal"]
    limit = sp_data.get("timeLimitSec")
    over = float(sp_data.get("overTimeSec") or 0.0)
    t = Transcript.model_validate(data["transcript"])
    raw_t = Transcript(
        segments=[
            s.model_copy(update={"words": [w.model_copy(update={"tag": None}) for w in s.words]})
            for s in t.segments
        ]
    )
    speech_from_t = analyze_speech(raw_t, duration, limit)
    sp_model = SpeechMetrics.model_validate(sp_data)
    # 가짜 파이프라인 예시 리포트처럼 대본은 일부만 있고 speech 지표가 따로 채워진 경우를 보정
    speaking_sec = (
        speech_from_t.speaking_sec
        if speech_from_t.speaking_sec >= 10.0 or duration <= 10.0
        else duration
    )
    span_sec = (
        limit + over
        if (limit is not None and over > 0)
        else max(speech_from_t.speech_span_sec, duration)
    )
    ratio = round(span_sec / limit, 3) if limit else None
    speech = SpeechAnalysis(
        metrics=sp_model,
        transcript=t,
        syllable_count=speech_from_t.syllable_count,
        speaking_sec=speaking_sec,
        silences=speech_from_t.silences,
        filler_hits=speech_from_t.filler_hits,
        repeat_hits=speech_from_t.repeat_hits,
        speech_span_sec=span_sec,
        under_time_sec=max(0.0, (limit - span_sec) if limit else 0.0),
        time_used_ratio=ratio,
        is_under_time=bool(limit and span_sec < limit * 0.7),
        profanity=speech_from_t.profanity,
    )
    nv = NonverbalAnalysis(
        metrics=NonverbalMetrics.model_validate(nv_data),
        frame_count=1,
        face_frame_ratio=1.0,
        posture_measured=nv_data.get("postureStability") is not None,
        gesture_measured=nv_data.get("gesturePerMin") is not None,
        gesture_times=[],
        gaze_away_ranges=[],
        posture_unstable_ranges=[],
        gesture_none_ranges=[],
        gesture_excess_ranges=[],
    )
    issues: list[ContentIssue] = []
    for f in data.get("feedbacks", []):
        if f.get("type") in ("STRUCTURE", "CLARITY"):
            issues.append(
                ContentIssue(
                    type=IssueType(f["type"]),
                    segment_ids=[],
                    start=float(f["start"]),
                    end=float(f["end"]),
                    reason=str(f.get("description") or ""),
                )
            )
    content = ContentAnalysis(
        sections=[],
        key_message=None,
        issues=issues,
        slide_coverage=None,
        expected_questions=[],
        usage=None,
    )
    return build_metric_details(speech, nv, content)


def history_values(
    data: dict[str, Any],
    metric_details: dict[str, dict[str, float | None] | None] | None = None,
    *,
    compute_fallback: bool = True,
) -> dict[str, Any]:
    """P2·P3·P7 및 M1~M3 마이페이지가 쓰는 회차 요약값(Analysis 컬럼)."""
    sp = data["metrics"]["speech"]
    fb_counts = Counter(f["type"] for f in data.get("feedbacks", []))
    overall_score = int(data["overall"]["score"])
    grade = data["overall"].get("grade") or grade_for(overall_score)
    resolved_details = (
        metric_details
        if metric_details is not None or not compute_fallback
        else _metric_details_from_report_json(data)
    )
    return {
        "overall_score": overall_score,
        "grade": grade,
        "scores": {s["category"]: s["score"] for s in data["scores"]},
        "metric_details": resolved_details,
        "filler_counts": [
            {"word": str(item["word"]), "count": int(item["count"])}
            for item in sp.get("fillers", [])
            if int(item.get("count", 0)) > 0
        ],
        "feedback_counts": dict(fb_counts),
        "filler_count": sp["fillerCount"],
        "syllables_per_min": sp["syllablesPerMin"],
    }
