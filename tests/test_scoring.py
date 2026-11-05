"""scoring: 코드만으로 0~100점. LLM 없음."""

import pytest

from app.models.enums import Purpose, ScoreCategory
from app.schemas.report import NonverbalMetrics, SpeechMetrics, Transcript
from app.services.pipeline import scoring as sc
from app.services.pipeline.content import ContentAnalysis, ContentIssue, IssueType
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.speech_metrics import SpeechAnalysis

D, N, C = ScoreCategory.DELIVERY, ScoreCategory.NONVERBAL, ScoreCategory.CONTENT


def speech(
    spm=340, fillers=0, silences=0, speaking=120.0, ratio=None, limit=None
) -> SpeechAnalysis:
    return SpeechAnalysis(
        metrics=SpeechMetrics(
            syllables_per_min=spm, pace_series=[], filler_count=fillers, fillers=[],
            silence_count=silences, longest_silence_sec=0.0, time_limit_sec=limit,
            over_time_sec=0.0,
        ),
        transcript=Transcript(segments=[]), syllable_count=0, speaking_sec=speaking,
        silences=[], filler_hits=[], repeat_hits=[], speech_span_sec=speaking,
        under_time_sec=0.0, time_used_ratio=ratio, is_under_time=False,
    )  # fmt: skip


def nonverbal(gaze=0.9, posture=0.9, gpm=6.0) -> NonverbalAnalysis:
    return NonverbalAnalysis(
        metrics=NonverbalMetrics(
            gaze_front_ratio=gaze, posture_stability=posture, gesture_per_min=gpm,
            low_confidence_ranges=[],
        ),
        frame_count=100, face_frame_ratio=1.0, posture_measured=posture is not None,
        gesture_measured=gpm is not None, gesture_times=[], gaze_away_ranges=[],
        posture_unstable_ranges=[], gesture_none_ranges=[], gesture_excess_ranges=[],
    )  # fmt: skip


def content(structure=0, clarity=0) -> ContentAnalysis:
    issues = [ContentIssue(IssueType.STRUCTURE, ["s-001"], 0, 1, "x")] * structure
    issues += [ContentIssue(IssueType.CLARITY, ["s-001"], 0, 1, "x")] * clarity
    return ContentAnalysis(
        sections=[], key_message="핵심", issues=issues, slide_coverage=None,
        expected_questions=[], usage=None,
    )  # fmt: skip


# ---------- 세부 지표 ----------


@pytest.mark.parametrize(
    ("spm", "score"),
    [(340, 100), (300, 100), (380, 100), (240, 50), (180, 0), (100, 0), (440, 50), (500, 0)],
)
def test_pace_band(spm, score):
    assert sc.delivery_details(speech(spm=spm))["pace"] == pytest.approx(score)


@pytest.mark.parametrize(("per_min", "score"), [(0, 100), (1, 100), (5.5, 50), (10, 0), (20, 0)])
def test_filler_score(per_min, score):
    d = sc.delivery_details(speech(fillers=int(per_min * 2), speaking=120.0))
    assert d["filler"] == pytest.approx(score)


@pytest.mark.parametrize(
    ("ratio", "score"), [(0.9, 100), (1.0, 100), (1.15, 50), (1.3, 0), (0.625, 50), (0.3, 0)]
)
def test_time_score(ratio, score):
    assert sc.delivery_details(speech(ratio=ratio, limit=300))["time"] == pytest.approx(score)


def test_no_time_limit_is_unmeasured():
    assert sc.delivery_details(speech())["time"] is None


@pytest.mark.parametrize(
    ("gaze", "score"), [(0.9, 100), (0.85, 100), (0.575, 50), (0.3, 0), (0.1, 0)]
)
def test_gaze_score(gaze, score):
    assert sc.nonverbal_details(nonverbal(gaze=gaze))["gaze"] == pytest.approx(score)


@pytest.mark.parametrize(
    ("gpm", "score"), [(0, 50), (1, 75), (2, 100), (12, 100), (26, 50), (40, 0)]
)
def test_gesture_score(gpm, score):
    assert sc.nonverbal_details(nonverbal(gpm=gpm))["gesture"] == pytest.approx(score)


def test_content_penalties():
    d = sc.content_details(content(structure=2, clarity=3))
    assert d == {"structure": 70.0, "clarity": 70.0, "language": 100.0}
    assert sc.content_details(content(structure=10))["structure"] == 0.0


@pytest.mark.parametrize(
    ("prof_count", "serious", "minor", "expected_penalty", "expected_lang_detail"),
    [
        (0, 0, 0, 0.0, 100.0),
        (1, 0, 0, 40.0, 60.0),
        (3, 0, 0, 70.0, 30.0),  # 첫 1회 -40 + 추가 2회(-15×2) = -70
        (0, 1, 1, 40.0, 60.0),  # SERIOUS(-30) + MINOR(-10) = -40
        (1, 1, 1, 80.0, 20.0),  # 욕설 1회(-40) + SERIOUS(-30) + MINOR(-10) = -80
        (3, 2, 0, 130.0, 0.0),  # 100점 초과 감점도 0점 미만으로 내려가지 않는다
    ],
)
def test_language_penalties_and_clamp(
    prof_count, serious, minor, expected_penalty, expected_lang_detail
):
    from app.services.pipeline.content import LanguageSeverity
    from app.services.pipeline.profanity_ko import ProfanityHit, ProfanityResult

    # structure=85, clarity=100 -> 평균 92.5
    issues = [ContentIssue(IssueType.STRUCTURE, ["s-001"], 0, 1, "x")]
    issues += [
        ContentIssue(IssueType.LANGUAGE, ["s-001"], 0, 1, "x", severity=LanguageSeverity.SERIOUS)
    ] * serious
    issues += [
        ContentIssue(IssueType.LANGUAGE, ["s-001"], 0, 1, "x", severity=LanguageSeverity.MINOR)
    ] * minor
    ct = ContentAnalysis(
        sections=[],
        key_message="핵심",
        issues=issues,
        slide_coverage=None,
        expected_questions=[],
        usage=None,
    )
    hits = tuple(ProfanityHit("x같네요", "x같", 0.0, 1.0, "s-001") for _ in range(prof_count))
    sp = speech()
    sp = SpeechAnalysis(**{**sp.__dict__, "profanity": ProfanityResult(hits=hits)})

    assert sc.language_penalty(ct, prof_count) == pytest.approx(expected_penalty)
    d = sc.content_details(ct, profanity_count=prof_count)
    assert d["language"] == pytest.approx(expected_lang_detail)

    r = sc.score_presentation(Purpose.CLASS, sp, nonverbal(), ct)
    assert 0 <= r.categories[C] <= 100
    expected_c = sc._round_int(max(0.0, 92.5 - expected_penalty))
    assert r.categories[C] == expected_c
    if expected_penalty == 0:
        # 언어 문제가 없으면 기존 점수와 완전히 같다
        baseline = sc.score_presentation(Purpose.CLASS, speech(), nonverbal(), content(structure=1))
        assert r.categories == baseline.categories and r.overall == baseline.overall


# ---------- 측정 불가 → 가중치 재분배 ----------


def test_weighted_redistributes_missing():
    w = {"a": 0.5, "b": 0.25, "c": 0.25}
    assert sc.weighted({"a": 80, "b": None, "c": 40}, w) == pytest.approx(
        (0.5 * 80 + 0.25 * 40) / 0.75
    )
    assert sc.weighted({"a": None, "b": None, "c": None}, w) is None


def test_null_nonverbal_metrics_are_excluded():
    nv = nonverbal(gaze=0.575, posture=None, gpm=None)  # 시선 50점만 남음
    r = sc.score_presentation(Purpose.CLASS, speech(), nv, content())
    assert r.categories[N] == 50
    assert r.details[N] == {"gaze": pytest.approx(50.0), "posture": None, "gesture": None}


def test_delivery_without_time_limit_uses_remaining_weights():
    # 말 속도 100, 추임새 50(분당 5.5), 침묵 100 → (0.35×100 + 0.35×50 + 0.15×100) / 0.85
    r = sc.score_presentation(Purpose.CLASS, speech(fillers=11), nonverbal(), content())
    assert r.categories[D] == round((35 + 17.5 + 15) / 0.85)


def test_category_without_any_metric_is_neutral_and_excluded():
    r = sc.score_presentation(Purpose.CLASS, speech(), None, content(structure=2))
    assert r.categories[N] == sc.NEUTRAL_SCORE and not r.measured[N]
    # 종합 = D, C만 다시 나눈 가중치(0.35, 0.40)
    expected = (0.35 * r.categories[D] + 0.40 * 85) / 0.75
    assert r.overall == round(expected)


# ---------- 목적별 가중치·등급 ----------


def test_purpose_weights_change_overall():
    args = (speech(fillers=20), nonverbal(gaze=0.95), content(structure=1))
    by_purpose = {p: sc.score_presentation(p, *args).overall for p in Purpose}
    assert len(set(by_purpose.values())) > 1
    for p, w in sc.PURPOSE_WEIGHTS.items():
        assert sum(w.values()) == pytest.approx(1.0), p


@pytest.mark.parametrize(
    ("score", "grade"),
    [(100, "A+"), (95, "A+"), (94, "A"), (90, "A"), (89, "B+"), (82, "B+"), (80, "B+"),
     (79, "B"), (70, "B"), (69, "C+"), (60, "C+"), (50, "C"), (49, "D"), (0, "D")],
)  # fmt: skip
def test_grade_mapping(score, grade):
    assert sc.grade_for(score) == grade


def test_scores_are_ints_in_range_and_grade_matches():
    r = sc.score_presentation(
        Purpose.PITCH, speech(spm=470, fillers=40), nonverbal(0.2, 0.1, 50), content(5, 5)
    )
    for v in [r.overall, *r.categories.values()]:
        assert isinstance(v, int) and 0 <= v <= 100
    assert r.grade == sc.grade_for(r.overall)


def test_llm_text_does_not_affect_score():
    a = content(structure=1)
    b = ContentAnalysis(
        sections=[], key_message="완전히 다른 핵심 메시지", issues=a.issues, slide_coverage=None,
        expected_questions=[], usage=None,
    )  # fmt: skip
    ra = sc.score_presentation(Purpose.CLASS, speech(), nonverbal(), a)
    rb = sc.score_presentation(Purpose.CLASS, speech(), nonverbal(), b)
    assert ra == rb


def test_perfect_presentation():
    r = sc.score_presentation(
        Purpose.INTERVIEW, speech(ratio=0.95, limit=300), nonverbal(), content()
    )
    assert r.categories == {D: 100, N: 98, C: 100}  # 자세 0.9 → 90점이 비언어에 반영
    assert r.grade in ("A+", "A")


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_metrics_are_unmeasured(bad):
    nv = nonverbal(gaze=0.9, posture=0.9, gpm=bad)
    r = sc.score_presentation(Purpose.CLASS, speech(), nv, content())
    assert r.details[N]["gesture"] is None
    # 시선 100(0.5)·자세 90(0.25)만 다시 나눔: (50 + 22.5) / 0.75 = 96.7 → 97
    assert r.categories[N] == 97
