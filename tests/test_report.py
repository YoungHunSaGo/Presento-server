"""report: 명세 5장 리포트 JSON 조립·검증."""

import copy
import json
from datetime import UTC, datetime

import pytest

from app.models.enums import Purpose, ScoreCategory
from app.schemas.report import (
    ExpectedQuestion,
    Feedback,
    FeedbackType,
    FillerStat,
    ImprovedScript,
    NonverbalMetrics,
    PacePoint,
    Segment,
    Severity,
    SlideCoverage,
    SpeechMetrics,
    TimeRange,
    Transcript,
    Word,
    WordTag,
)
from app.services.pipeline import report as rp
from app.services.pipeline.content import ContentAnalysis
from app.services.pipeline.fake import SAMPLE_REPORT_PATH
from app.services.pipeline.feedback import FeedbackResult, OverallText
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.scoring import ScoreResult
from app.services.pipeline.speech_metrics import SpeechAnalysis

SPEC = json.loads(SAMPLE_REPORT_PATH.read_text(encoding="utf-8"))  # 명세 5장 예시 그대로


def parts(**over) -> rp.ReportParts:
    transcript = Transcript(
        segments=[
            Segment(
                id="s-001", start=0.0, end=6.4, text="안녕하세요 음 오늘",
                words=[
                    Word(text="안녕하세요", start=0.0, end=0.9, tag=None),
                    Word(text="음", start=1.1, end=1.4, tag=WordTag.FILLER),
                    Word(text="오늘", start=2.0, end=2.5, tag=None),
                ],
            )
        ]
    )  # fmt: skip
    speech = SpeechAnalysis(
        metrics=SpeechMetrics(
            syllables_per_min=341, pace_series=[PacePoint(t=0, value=402)], filler_count=9,
            fillers=[FillerStat(word="음", count=9)], silence_count=3, longest_silence_sec=3.8,
            time_limit_sec=300, over_time_sec=0.0,
        ),
        transcript=transcript, syllable_count=7, speaking_sec=6.0, silences=[], filler_hits=[],
        repeat_hits=[], speech_span_sec=6.4, under_time_sec=0.0, time_used_ratio=0.02,
        is_under_time=True,
    )  # fmt: skip
    nv = NonverbalAnalysis(
        metrics=NonverbalMetrics(
            gaze_front_ratio=0.78, posture_stability=0.86, gesture_per_min=None,
            low_confidence_ranges=[TimeRange(start=120.0, end=128.5)],
        ),
        frame_count=10, face_frame_ratio=1.0, posture_measured=True, gesture_measured=False,
        gesture_times=[], gaze_away_ranges=[], posture_unstable_ranges=[], gesture_none_ranges=[],
        gesture_excess_ranges=[],
    )  # fmt: skip
    ct = ContentAnalysis(
        sections=[], key_message="핵심", issues=[],
        slide_coverage=[SlideCoverage(page=1, covered=True, note=None)],
        expected_questions=[ExpectedQuestion(question="질문?", intent="의도", answer_tip="팁")],
        usage=None,
    )  # fmt: skip
    fb = FeedbackResult(
        feedbacks=[
            Feedback(
                id="f-01", category=ScoreCategory.DELIVERY, type=FeedbackType.PACE_FAST,
                severity=Severity.WARNING, start=0.0, end=31.2, title="말이 빨라요",
                description="빨라요.", evidence="안녕하세요 음 오늘", suggestion="쉬어 보세요.",
            )
        ],
        overall=OverallText(summary="좋아요.", strengths=["시선"], improvements=["속도"]),
        improved_scripts=[
            ImprovedScript(
                segment_id="s-001", start=0.0, end=6.4, original="안녕하세요 음 오늘",
                improved="안녕하세요, 오늘", reason="추임새를 뺐어요.",
            )
        ],
        candidates=[],
    )  # fmt: skip
    scores = ScoreResult(
        overall=82, grade="B+",
        categories={
            ScoreCategory.DELIVERY: 80, ScoreCategory.NONVERBAL: 84, ScoreCategory.CONTENT: 81,
        },
        details={}, measured={},
    )  # fmt: skip
    base = dict(
        analysis_id="a-9d21", presentation_id="p-7f3c", attempt_no=3, purpose=Purpose.CLASS,
        duration_sec=287.4321, created_at=datetime(2026, 11, 5, 11, 0, tzinfo=UTC),
        scores=scores, speech=speech, nonverbal=nv, content=ct, feedback=fb,
    )  # fmt: skip
    base.update(over)
    return rp.ReportParts(**base)


def shape(v):
    """값 대신 구조(키·타입)만."""
    if isinstance(v, dict):
        return {k: shape(x) for k, x in v.items()}
    if isinstance(v, list):
        return [shape(v[0])] if v else []
    return type(v).__name__


def keys_tree(v):
    if isinstance(v, dict):
        return {k: keys_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [keys_tree(v[0])] if v and isinstance(v[0], dict) else []
    return None


def test_report_has_exactly_spec_section_5_keys():
    data = rp.build_report(parts())
    built = keys_tree(data)
    spec = keys_tree(SPEC)
    # 리스트 원소 구조까지 같은 키
    assert built.keys() == spec.keys()
    for k in spec:
        if isinstance(spec[k], dict) or (spec[k] and isinstance(spec[k], list)):
            assert built[k] == spec[k], k


def test_report_values():
    data = rp.build_report(parts())
    assert data["analysisId"] == "a-9d21" and data["attemptNo"] == 3
    assert data["durationSec"] == 287.43
    assert data["createdAt"] == "2026-11-05T20:00:00+09:00"  # KST
    assert data["overall"] == {
        "score": 82, "grade": "B+", "summary": "좋아요.", "strengths": ["시선"],
        "improvements": ["속도"],
    }  # fmt: skip
    assert data["scores"] == SPEC["scores"]  # 순서·라벨까지 명세와 같음
    assert data["metrics"]["nonverbal"]["gesturePerMin"] is None  # 측정 불가 → null
    assert data["transcript"]["segments"][0]["words"][1]["tag"] == "FILLER"
    assert data["improvedScripts"][0]["segmentId"] == "s-001"
    assert data["expectedQuestions"][0] == {
        "question": "질문?",
        "intent": "의도",
        "answerTip": "팁",
    }


def test_no_slides_gives_null():
    ct = parts().content
    no_slides = ContentAnalysis(**{**ct.__dict__, "slide_coverage": None})
    assert rp.build_report(parts(content=no_slides))["slideCoverage"] is None


def test_missing_nonverbal_gives_nulls():
    data = rp.build_report(parts(nonverbal=None))
    assert data["metrics"]["nonverbal"] == {
        "gazeFrontRatio": None, "postureStability": None, "gesturePerMin": None,
        "lowConfidenceRanges": [],
    }  # fmt: skip


def test_spec_sample_passes_validation():
    rp.validate_report(copy.deepcopy(SPEC))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra=1),  # 명세에 없는 필드
        lambda d: d["overall"].update(extra=1),
        lambda d: d["feedbacks"][0].update(extra=1),
        lambda d: d.pop("transcript"),  # 빠진 필드
        lambda d: d["metrics"]["speech"].pop("fillerCount"),
        lambda d: d["overall"].update(score="82"),  # 형식이 다름(문자열)
        lambda d: d["overall"].update(score=101),  # 범위
        lambda d: d["scores"].reverse(),  # 순서
        lambda d: d["feedbacks"][0].update(end=999.0),  # 영상 길이 밖
        lambda d: d["feedbacks"][0].update(type="SOMETHING"),
        lambda d: d["metrics"]["nonverbal"].update(gazeFrontRatio=1.5),
    ],
)
def test_validation_rejects_off_spec(mutate):
    data = copy.deepcopy(SPEC)
    mutate(data)
    with pytest.raises(rp.ReportInvalid):
        rp.validate_report(data)


def test_history_values():
    data = rp.build_report(parts())
    assert rp.history_values(data) == {
        "overall_score": 82,
        "scores": {"DELIVERY": 80, "NONVERBAL": 84, "CONTENT": 81},
        "filler_count": 9,
        "syllables_per_min": 341,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["feedbacks"][0].update(end=44.517656),
        lambda d: d.update(durationSec=287.401),
        lambda d: d["transcript"]["segments"][0]["words"][0].update(start=0.123),
        lambda d: d["improvedScripts"][0].update(end=108.001),
        lambda d: d["metrics"]["nonverbal"]["lowConfidenceRanges"][0].update(end=128.555),
    ],
)
def test_time_values_must_have_two_decimals(mutate):
    data = copy.deepcopy(SPEC)
    mutate(data)
    with pytest.raises(rp.ReportInvalid) as exc_info:
        rp.validate_report(data)
    assert "2 decimals" in str(exc_info.value)


def test_infinite_time_value_is_invalid_not_crash():
    data = copy.deepcopy(SPEC)
    data["feedbacks"][0]["end"] = float("inf")
    with pytest.raises(rp.ReportInvalid):
        rp.validate_report(data)
