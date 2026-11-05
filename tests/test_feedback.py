"""feedback: 후보 생성·정리는 코드, 문장은 LLM(가짜 클라이언트). 실제 호출 없음."""

import json
import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

from app.models.enums import Purpose, ScoreCategory
from app.models.enums import ScoreCategory as Cat
from app.schemas.report import (
    FeedbackType,
    FillerStat,
    NonverbalMetrics,
    PacePoint,
    Segment,
    Severity,
    SpeechMetrics,
    TimeRange,
    Transcript,
    Word,
)
from app.services.pipeline import feedback as fb
from app.services.pipeline.content import ContentAnalysis, ContentIssue, IssueType
from app.services.pipeline.feedback import (
    FeedbackInput,
    FeedbackServiceError,
    build_candidates,
    generate_feedback,
)
from app.services.pipeline.korean_style import formal_sentences
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.speech_metrics import Silence, SpeechAnalysis, WordHit

DURATION = 120.0


def transcript(n: int = 12) -> Transcript:
    """10초마다 한 문장(s-001 0~9.5초 …). 문장마다 단어 두 개."""
    segs = []
    for i in range(n):
        s = i * 10.0
        segs.append(
            Segment(
                id=f"s-{i + 1:03d}",
                start=s,
                end=s + 9.5,
                text=f"문장 {i + 1} 내용이에요",
                words=[
                    Word(text="문장", start=s, end=s + 4.0, tag=None),
                    Word(text="내용이에요", start=s + 4.5, end=s + 9.5, tag=None),
                ],
            )
        )
    return Transcript(segments=segs)


def speech(**over) -> SpeechAnalysis:
    metrics = SpeechMetrics(
        syllables_per_min=over.pop("spm", 320),
        pace_series=over.pop("pace", [PacePoint(t=t, value=320) for t in (0, 30, 60, 90)]),
        filler_count=len(over.get("filler_hits", [])),
        fillers=over.pop("fillers", []),
        silence_count=len(over.get("silences", [])),
        longest_silence_sec=0.0,
        time_limit_sec=over.pop("time_limit", None),
        over_time_sec=over.pop("over", 0.0),
    )
    base = dict(
        metrics=metrics,
        transcript=transcript(),
        syllable_count=600,
        speaking_sec=110.0,
        silences=[],
        filler_hits=[],
        repeat_hits=[],
        speech_span_sec=119.5,
        under_time_sec=0.0,
        time_used_ratio=None,
        is_under_time=False,
    )
    base.update(over)
    return SpeechAnalysis(**base)


def nonverbal(**over) -> NonverbalAnalysis:
    base = dict(
        metrics=NonverbalMetrics(
            gaze_front_ratio=over.pop("gaze", 0.8),
            posture_stability=over.pop("posture", 0.9),
            gesture_per_min=over.pop("gpm", 6.0),
            low_confidence_ranges=[],
        ),
        frame_count=120,
        face_frame_ratio=1.0,
        posture_measured=True,
        gesture_measured=True,
        gesture_times=[],
        gaze_away_ranges=[],
        posture_unstable_ranges=[],
        gesture_none_ranges=[],
        gesture_excess_ranges=[],
    )
    base.update(over)
    return NonverbalAnalysis(**base)


def content_with(*issues: ContentIssue) -> ContentAnalysis:
    return ContentAnalysis(
        sections=[],
        key_message="핵심은 연습이에요.",
        issues=list(issues),
        slide_coverage=None,
        expected_questions=[],
        usage=None,
    )


def issue(kind: IssueType, ids: list[str], reason: str = "모호해요.") -> ContentIssue:
    t = transcript()
    idx = {s.id: s for s in t.segments}
    return ContentIssue(
        type=kind,
        segment_ids=ids,
        start=min(idx[i].start for i in ids),
        end=max(idx[i].end for i in ids),
        reason=reason,
    )


def data(**over) -> FeedbackInput:
    base = dict(
        purpose=Purpose.CLASS,
        duration_sec=DURATION,
        transcript=transcript(),
        speech=speech(),
        nonverbal=None,
        content=None,
    )
    base.update(over)
    return FeedbackInput(**base)


def hit(word: str, t: float, seg: str = "s-001") -> WordHit:
    return WordHit(word=word, start=t, end=t + 0.3, segment_id=seg)


def types(cands) -> list[FeedbackType]:
    return [c.type for c in cands]


# ---------- 말 속도 ----------


@pytest.mark.parametrize(
    ("value", "expected"),
    [(399, None), (400, (FeedbackType.PACE_FAST, Severity.WARNING)),
     (450, (FeedbackType.PACE_FAST, Severity.CRITICAL)),
     (251, None), (250, (FeedbackType.PACE_SLOW, Severity.WARNING)),
     (200, (FeedbackType.PACE_SLOW, Severity.CRITICAL)), (0, None)],
)  # fmt: skip
def test_pace_thresholds(value, expected):
    pace = [PacePoint(t=0, value=320), PacePoint(t=30, value=value), PacePoint(t=60, value=320)]
    cands = build_candidates(data(speech=speech(pace=pace)))
    if expected is None:
        assert cands == []
    else:
        (c,) = cands
        assert (c.type, c.severity) == expected
        assert (c.start, c.end) == (30.0, 60.0)
        assert c.category is ScoreCategory.DELIVERY
        assert c.values == {"pace": value, "avgPace": 320}
        assert c.segment_ids == ("s-004", "s-005", "s-006")


def test_adjacent_fast_windows_merge():
    pace = [PacePoint(t=0, value=410), PacePoint(t=30, value=460), PacePoint(t=60, value=320)]
    (c,) = build_candidates(data(speech=speech(pace=pace)))
    assert (c.start, c.end) == (0.0, 60.0)
    assert c.severity is Severity.CRITICAL
    assert c.values["pace"] == 460


# ---------- 추임새 ----------


def test_filler_cluster():
    hits = [hit("음", 10), hit("어", 15), hit("음", 24.0, "s-003")]
    (c,) = build_candidates(data(speech=speech(filler_hits=hits)))
    assert c.type is FeedbackType.FILLER and c.severity is Severity.WARNING
    assert (c.start, c.end) == (10.0, 24.3)
    assert c.values == {"fillerCount": 3, "topFiller": "음"}
    assert c.segment_ids == ("s-001", "s-003")


def test_filler_cluster_gap_and_minimum():
    # 10초 넘게 떨어지면 다른 무리, 2개짜리 무리는 후보가 아님
    hits = [hit("음", 0), hit("음", 5), hit("어", 20), hit("어", 25)]
    assert build_candidates(data(speech=speech(filler_hits=hits, speaking_sec=600.0))) == []


def test_filler_cluster_critical():
    hits = [hit("음", t) for t in range(0, 12, 2)]
    (c,) = build_candidates(data(speech=speech(filler_hits=hits)))
    assert c.severity is Severity.CRITICAL and c.values["fillerCount"] == 6


def test_spread_fillers_become_one_info_candidate():
    # 무리는 없지만 110초에 8개 → 분당 4.4개
    hits = [hit("이제", t) for t in range(0, 120, 15)]
    stats = [FillerStat(word="이제", count=8)]
    (c,) = build_candidates(data(speech=speech(filler_hits=hits, fillers=stats)))
    assert c.type is FeedbackType.FILLER and c.severity is Severity.INFO
    assert c.values == {"fillerCount": 8, "topFiller": "이제", "fillerPerMin": 4.4}


# ---------- 침묵 ----------


@pytest.mark.parametrize(
    ("dur", "sev"),
    [(2.9, None), (3.0, Severity.INFO), (5.0, Severity.WARNING), (8.0, Severity.CRITICAL)],
)
def test_long_silence_thresholds(dur, sev):
    cands = build_candidates(data(speech=speech(silences=[Silence(40.0, 40.0 + dur)])))
    if sev is None:
        assert cands == []
    else:
        (c,) = cands
        assert c.type is FeedbackType.LONG_SILENCE and c.severity is sev
        assert c.values == {"silenceSec": dur}


# ---------- 시간 ----------


def test_over_time_candidate():
    sp = speech(time_limit=100, over=19.5)
    (c,) = build_candidates(data(speech=sp))
    assert c.type is FeedbackType.OVER_TIME
    assert c.severity is Severity.CRITICAL  # 19.5 > 10
    assert (c.start, c.end) == (100.0, 119.5)  # 첫 단어 + 제한 시간 ~ 마지막 단어
    assert c.values == {"overTimeSec": 19.5, "timeLimitMin": 1.7}


def test_slight_over_time_is_warning():
    (c,) = build_candidates(data(speech=speech(time_limit=115, over=4.5)))
    assert c.severity is Severity.WARNING


@pytest.mark.parametrize(("ratio", "sev"), [(0.4, Severity.WARNING), (0.6, Severity.INFO)])
def test_under_time_candidate(ratio, sev):
    sp = speech(time_limit=300, is_under_time=True, time_used_ratio=ratio, under_time_sec=180.5)
    (c,) = build_candidates(data(speech=sp))
    assert c.type is FeedbackType.UNDER_TIME and c.severity is sev
    assert (c.start, c.end) == (0.0, 119.5)
    assert c.values == {"underTimeSec": 180.5, "timeLimitMin": 5}


def test_no_time_limit_no_time_candidates():
    assert build_candidates(data(speech=speech(is_under_time=True))) == []


# ---------- 비언어 ----------


def test_nonverbal_candidates():
    nv = nonverbal(
        gaze_away_ranges=[TimeRange(start=10, end=14), TimeRange(start=40, end=58)],
        posture_unstable_ranges=[TimeRange(start=60, end=70)],
        gesture_none_ranges=[TimeRange(start=0, end=90)],
        gesture_excess_ranges=[TimeRange(start=90, end=120)],
        gesture_times=[1.0, 92.0, 95.0, 99.0, 104.0, 110.0, 115.0],
        posture=0.4,
    )
    cands = build_candidates(data(nonverbal=nv))
    got = {(c.type, c.start): (c.severity, c.values) for c in cands}
    assert got[(FeedbackType.GAZE_AWAY, 10.0)] == (Severity.INFO, {"awaySec": 4.0})
    assert got[(FeedbackType.GAZE_AWAY, 40.0)] == (Severity.CRITICAL, {"awaySec": 18.0})
    # 구간별 자세 값이 없으므로 전체 값을 쓰지 않고 수치 없이 WARNING
    assert got[(FeedbackType.POSTURE_UNSTABLE, 60.0)] == (Severity.WARNING, {})
    assert got[(FeedbackType.GESTURE_NONE, 0.0)] == (Severity.WARNING, {"noneSec": 90.0})
    # 구간 안 손동작 6번 / 0.5분 = 분당 12(전체 값이 아니라 그 구간 값)
    assert got[(FeedbackType.GESTURE_EXCESS, 90.0)] == (
        Severity.INFO, {"gestureCount": 6, "gesturePerMin": 12.0},
    )  # fmt: skip
    assert all(c.category is ScoreCategory.NONVERBAL for c in cands)


def test_unmeasured_nonverbal_values_are_omitted():
    nv = nonverbal(posture=None, gpm=None, posture_unstable_ranges=[TimeRange(start=0, end=10)])
    (c,) = build_candidates(data(nonverbal=nv))
    assert c.values == {} and c.severity is Severity.WARNING


# ---------- 내용 ----------


def test_content_candidates():
    ct = content_with(
        issue(IssueType.STRUCTURE, ["s-002", "s-003"], "순서가 바뀌었어요."),
        issue(IssueType.CLARITY, ["s-010"]),
    )
    cands = build_candidates(data(content=ct))
    s, c = cands
    assert (s.type, s.severity, s.segment_ids, s.context) == (
        FeedbackType.STRUCTURE, Severity.WARNING, ("s-002", "s-003"), "순서가 바뀌었어요.",
    )  # fmt: skip
    assert (s.start, s.end) == (10.0, 29.5)
    assert (c.type, c.severity) == (FeedbackType.CLARITY, Severity.INFO)
    assert s.category is ScoreCategory.CONTENT


# ---------- 정리: 영상 길이, 병합, 상한 ----------


def test_candidates_outside_video_are_dropped_or_clipped():
    sil = [Silence(118.0, 125.0), Silence(130.0, 140.0), Silence(-2.0, 4.0)]
    cands = build_candidates(data(speech=speech(silences=sil)))
    assert [(c.start, c.end) for c in cands] == [(0.0, 4.0), (118.0, 120.0)]
    assert all(0 <= c.start < c.end <= DURATION for c in cands)


def test_merge_within_gap_only_same_type():
    nv = nonverbal(
        gaze_away_ranges=[
            TimeRange(start=10, end=14),
            TimeRange(start=19, end=23),  # 5초 간격 → 합침
            TimeRange(start=28.1, end=32),  # 5.1초 → 따로
        ],
        posture_unstable_ranges=[TimeRange(start=12, end=20)],  # 다른 type은 안 합침
    )
    cands = build_candidates(data(nonverbal=nv))
    gaze = [c for c in cands if c.type is FeedbackType.GAZE_AWAY]
    assert [(c.start, c.end) for c in gaze] == [(10.0, 23.0), (28.1, 32.0)]
    # 실제로 벗어난 시간의 합(4 + 4), 사이 정상 구간 5초는 빼고 → 8초면 WARNING으로 다시 계산
    assert gaze[0].values == {"awaySec": 8.0}
    assert gaze[0].severity is Severity.WARNING
    assert FeedbackType.POSTURE_UNSTABLE in types(cands)


def test_overlapping_merge_keeps_highest_severity_and_joins_context():
    ct = content_with(
        issue(IssueType.CLARITY, ["s-002"], "주어가 없어요."),
        issue(IssueType.CLARITY, ["s-003"], "근거가 없어요."),
    )
    (c,) = build_candidates(data(content=ct))
    assert c.segment_ids == ("s-002", "s-003")
    assert c.context == "주어가 없어요. / 근거가 없어요."


def test_merge_value_policies():
    a = fb.Candidate(
        FeedbackType.FILLER, Severity.WARNING, 0, 5, {"fillerCount": 3, "topFiller": "음"}
    )
    b = fb.Candidate(
        FeedbackType.FILLER, Severity.INFO, 8, 12, {"fillerCount": 4, "topFiller": "어"}
    )
    (m,) = fb._merge([a, b], {})
    assert m.values == {"fillerCount": 7, "topFiller": "음"}  # 수는 더하고 글자는 앞 것
    slow1 = fb.Candidate(FeedbackType.PACE_SLOW, Severity.WARNING, 0, 30, {"pace": 240})
    slow2 = fb.Candidate(FeedbackType.PACE_SLOW, Severity.WARNING, 30, 60, {"pace": 220})
    assert fb._merge([slow1, slow2], {})[0].values == {"pace": 220}  # 느림은 가장 느린 값
    s1 = fb.Candidate(FeedbackType.LONG_SILENCE, Severity.INFO, 0, 3, {"silenceSec": 3.0})
    s2 = fb.Candidate(FeedbackType.LONG_SILENCE, Severity.INFO, 6, 10, {"silenceSec": 4.0})
    (ms,) = fb._merge([s1, s2], {})
    assert ms.values == {"silenceSec": 4.0}  # 침묵은 합치지 않고 가장 긴 것
    assert ms.severity is Severity.INFO


def test_merged_filler_severity_is_recomputed():
    a = fb.Candidate(FeedbackType.FILLER, Severity.WARNING, 0, 5, {"fillerCount": 3})
    b = fb.Candidate(FeedbackType.FILLER, Severity.WARNING, 8, 12, {"fillerCount": 3})
    assert fb._merge([a, b], {})[0].severity is Severity.CRITICAL  # 3 + 3 = 6


def test_max_feedbacks_keeps_most_severe_then_sorts_by_time():
    # LONG_SILENCE: CRITICAL 4개 + INFO 6개, GAZE_AWAY: WARNING 4개, CLARITY: INFO 3개 → 17개
    sil = [Silence(t, t + 9.0) for t in (60, 75, 90, 105)]  # CRITICAL(간격 6초, 안 합침)
    sil += [Silence(t, t + 3.0) for t in range(0, 60, 10)]  # INFO(간격 7초)
    nv = nonverbal(gaze_away_ranges=[TimeRange(start=t, end=t + 9) for t in (2, 33, 64, 95)])
    ct = content_with(*(issue(IssueType.CLARITY, [f"s-{i:03d}"]) for i in (1, 5, 9)))
    cands = build_candidates(data(speech=speech(silences=sil), nonverbal=nv, content=ct))
    assert len(cands) == 11  # type별 상한 때문에 INFO 침묵 6개는 못 들어간다
    by_type = {t: [c for c in cands if c.type is t] for t in set(types(cands))}
    # type마다 최대 4개: 침묵은 CRITICAL 4개만, INFO 침묵은 빠짐
    assert [c.severity for c in by_type[FeedbackType.LONG_SILENCE]] == [Severity.CRITICAL] * 4
    assert len(by_type[FeedbackType.GAZE_AWAY]) == 4
    assert len(by_type[FeedbackType.CLARITY]) == 3
    assert [c.start for c in cands] == sorted(c.start for c in cands)
    assert [c.id for c in cands] == [f"c-{i:02d}" for i in range(1, 12)]


def test_severity_priority_when_over_limit():
    # 서로 다른 type 5종 × 각 3개 = 15개 → INFO 3개가 빠진다
    sil = [Silence(t, t + 9.0) for t in (0, 20, 40)]  # CRITICAL
    nv = nonverbal(
        gaze_away_ranges=[TimeRange(start=t, end=t + 9) for t in (60, 75, 90)],  # WARNING
        gesture_none_ranges=[TimeRange(start=t, end=t + 4) for t in (2, 22, 42)],  # INFO
        posture_unstable_ranges=[TimeRange(start=t, end=t + 4) for t in (62, 77, 92)],  # WARNING
    )
    ct = content_with(*(issue(IssueType.STRUCTURE, [f"s-{i:03d}"]) for i in (2, 6, 10)))
    cands = build_candidates(data(speech=speech(silences=sil), nonverbal=nv, content=ct))
    assert len(cands) == 12
    assert FeedbackType.GESTURE_NONE not in types(cands)


def test_clipped_range_values_and_severity_follow_clip():
    nv = nonverbal(gaze_away_ranges=[TimeRange(start=110, end=130)])  # 20초 → 잘려 10초
    (c,) = build_candidates(data(nonverbal=nv))
    assert (c.start, c.end) == (110.0, 120.0)
    assert c.values == {"awaySec": 10.0}
    assert c.severity is Severity.WARNING  # 20초면 CRITICAL, 잘린 10초면 WARNING


def test_filler_cluster_span_is_capped():
    # 9초 간격 추임새가 2분 내내 → 한 무리가 되지 않고 60초 단위로 나뉜다
    hits = [hit("음", float(t)) for t in range(0, 120, 9)]
    cands = build_candidates(data(speech=speech(filler_hits=hits)))
    assert all(c.end - c.start <= fb.FILLER_CLUSTER_MAX_SEC for c in cands)
    assert len(cands) >= 2


# ---------- evidence ----------


def test_evidence_rules():
    t = transcript()
    pace = fb.Candidate(
        FeedbackType.PACE_FAST,
        Severity.WARNING,
        25,
        55,
        segment_ids=("s-003", "s-004", "s-005", "s-006"),
    )
    gaze = fb.Candidate(FeedbackType.GAZE_AWAY, Severity.INFO, 0, 5, segment_ids=("s-001",))
    clar = fb.Candidate(FeedbackType.CLARITY, Severity.INFO, 0, 20, segment_ids=("s-001", "s-002"))
    sil = fb.Candidate(FeedbackType.LONG_SILENCE, Severity.INFO, 0, 5)
    under = fb.Candidate(FeedbackType.UNDER_TIME, Severity.INFO, 0, 5, segment_ids=("s-001",))
    # 말 속도: 구간(25~55)과 가장 많이 겹치는 문장(s-004 30~39.5)
    assert fb._evidence(pace, t) == "문장 4 내용이에요"
    assert fb._evidence(gaze, t) is None
    assert fb._evidence(clar, t) == "문장 1 내용이에요 문장 2 내용이에요"
    assert fb._evidence(sil, t) is None
    assert fb._evidence(under, t) is None


def test_content_evidence_is_first_contiguous_run_in_order():
    t = transcript()
    c = fb.Candidate(
        FeedbackType.STRUCTURE, Severity.WARNING, 0, 50, segment_ids=("s-005", "s-002", "s-003")
    )
    ev = fb._evidence(c, t)
    assert ev == "문장 2 내용이에요 문장 3 내용이에요"
    whole = " ".join(s.text for s in t.segments)
    assert ev in whole  # 근거는 대본에 그대로 있는 글


def test_silence_evidence_is_sentence_before():
    sp = speech(silences=[Silence(39.5, 45.0)])
    (c,) = build_candidates(data(speech=sp))
    assert c.segment_ids == ("s-004",)
    assert fb._evidence(c, transcript()) == "문장 4 내용이에요"


# ---------- LLM ----------


def resp(payload, status="completed", tokens=(3000, 1500)):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        status=status,
        output_text=text,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


class FakeLlm:
    def __init__(self, *results):
        self.results = list(results)
        self.calls: list[dict] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, Exception):
            raise r
        return r


@pytest.fixture
def llm_settings(settings):
    settings.openai_llm_model = "gpt-5-mini"
    return settings


def rich_input() -> FeedbackInput:
    """후보 3개: c-01 PACE_FAST(0~30), c-02 GAZE_AWAY(40~50), c-03 CLARITY(s-008)."""
    pace = [PacePoint(t=0, value=420), PacePoint(t=30, value=320), PacePoint(t=60, value=320)]
    return data(
        speech=speech(pace=pace),
        nonverbal=nonverbal(gaze_away_ranges=[TimeRange(start=40, end=50)]),
        content=content_with(issue(IssueType.CLARITY, ["s-008"], "근거가 없어요.")),
    )


def item(cid, title="제목이에요", desc="설명이에요.", sugg="이렇게 해 보세요."):
    return {"candidateId": cid, "title": title, "description": desc, "suggestion": sugg}


def payload(**over):
    p = {
        "feedbacks": [
            item("c-01", "도입부 말이 빨라요", "분당 {pace}음절로 평균({avgPace})보다 빨라요."),
            item("c-02", "시선이 벗어나요", "{awaySec}초 동안 화면을 보지 않았어요."),
            item("c-03", "근거가 부족해요", "주장 뒤에 근거가 없어요."),
        ],
        "overall": {
            "summary": "전체 말 속도는 분당 {syllablesPerMin}음절이에요. 근거를 보완하면 좋아요.",
            "strengths": ["핵심이 분명해요"],
            "improvements": ["도입부 속도", "근거 보완"],
        },
        "improvedScripts": [
            {
                "segmentId": "s-008",
                "improved": "문장 8의 근거는 이 자료예요.",
                "reason": "근거를 붙였어요.",
            }
        ],
    }
    p.update(over)
    return p


def test_zero_candidates_does_not_call_llm(llm_settings):
    client = FakeLlm(resp(payload()))
    r = generate_feedback(data(), client, llm_settings)
    assert client.calls == []
    assert r.feedbacks == [] and r.improved_scripts == []
    # LLM을 안 불러도 overall은 코드 기본 문구로 채운다(절대 null 아님)
    assert r.overall.summary and r.overall.strengths and r.overall.improvements
    assert r.overall_fallback
    assert r.usage is None


def test_happy_path_fills_placeholders_and_code_fields(llm_settings):
    client = FakeLlm(resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1
    f1, f2, f3 = r.feedbacks
    assert [f.id for f in r.feedbacks] == ["f-01", "f-02", "f-03"]
    assert f1.type is FeedbackType.PACE_FAST and f1.category is ScoreCategory.DELIVERY
    assert (f1.start, f1.end, f1.severity) == (0.0, 30.0, Severity.WARNING)
    assert f1.description == "분당 420음절로 평균(320)보다 빨라요."
    assert f1.evidence == "문장 1 내용이에요"
    assert f2.description == "10초 동안 화면을 보지 않았어요." and f2.evidence is None
    assert f3.type is FeedbackType.CLARITY and f3.evidence == "문장 8 내용이에요"
    assert r.overall.summary.startswith("전체 말 속도는 분당 320음절")
    (s,) = r.improved_scripts
    assert (s.segment_id, s.original, s.start, s.end) == ("s-008", "문장 8 내용이에요", 70.0, 79.5)
    assert r.usage.attempts == 1 and r.dropped_items == 0


def test_request_shape(llm_settings):
    client = FakeLlm(resp(payload()))
    generate_feedback(rich_input(), client, llm_settings)
    kw = client.calls[0]
    assert kw["reasoning"] == {"effort": "low"}
    assert kw["max_output_tokens"] == fb.MAX_OUTPUT_TOKENS
    assert kw["text"]["format"]["strict"] is True
    assert kw["text"]["format"]["schema"] == fb.RESPONSE_SCHEMA
    assert kw["instructions"] == fb.load_system_prompt()
    text = kw["input"]
    assert "- c-01 | 종류: PACE_FAST(DELIVERY)" in text and "- c-03" in text
    assert "{pace}=420" in text and "{awaySec}=10" in text
    assert "[대본]\ns-001: 문장 1 내용이에요" in text
    assert "[핵심 메시지] 핵심은 연습이에요." in text


def test_unknown_candidate_id_is_dropped_without_retry(llm_settings):
    p = payload()
    p["feedbacks"].append(item("c-99", "새 후보"))
    client = FakeLlm(resp(p))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1
    assert len(r.feedbacks) == 3 and r.dropped_items == 1


@pytest.mark.parametrize(
    "bad_desc",
    [
        "분당 420음절이에요.",
        "{pace}음절, 3배 빨라요.",
        "１０초 동안",
        "{unknown}이에요.",
        "{awaySec}초",
    ],
)
def test_digits_or_foreign_placeholder_trigger_drop_and_retry(llm_settings, bad_desc):
    bad = payload()
    bad["feedbacks"][0]["description"] = bad_desc  # c-01(PACE)에는 awaySec 없음
    client = FakeLlm(resp(bad), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2
    assert len(r.feedbacks) == 3 and r.usage.attempts == 2


def test_after_retries_best_attempt_used_and_bad_item_dropped(llm_settings):
    bad = payload()
    bad["feedbacks"][0]["title"] = "분당 420음절"
    client = FakeLlm(resp(bad))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1 + fb.MAX_RETRIES
    assert [f.type for f in r.feedbacks] == [FeedbackType.GAZE_AWAY, FeedbackType.CLARITY]
    assert [f.id for f in r.feedbacks] == ["f-01", "f-02"]


def test_missing_candidate_triggers_retry(llm_settings):
    partial = payload()
    partial["feedbacks"] = partial["feedbacks"][:2]
    client = FakeLlm(resp(partial), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2 and len(r.feedbacks) == 3


def test_overall_with_digit_is_violation(llm_settings):
    bad = payload()
    bad["overall"]["strengths"] = ["시선 80% 유지"]
    client = FakeLlm(resp(bad), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2 and r.overall.strengths == ["핵심이 분명해요"]


def test_improved_script_rules(llm_settings):
    p = payload()
    p["improvedScripts"] = [
        {"segmentId": "s-404", "improved": "없는 문장", "reason": "이유예요."},  # 없는 ID: 버림
        {"segmentId": "s-008", "improved": "문장 8 내용이에요", "reason": "같아요."},  # 원문과 같음
        {
            "segmentId": "s-005",
            "improved": "문장 5와 2024년 자료",
            "reason": "다듬었어요.",
        },  # 2024 없음
        {"segmentId": "s-005", "improved": "문장 5를 또렷하게 말해요.", "reason": "다듬었어요."},
    ]
    client = FakeLlm(resp(p), resp(p), resp(p))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert [(s.segment_id, s.improved) for s in r.improved_scripts] == [
        ("s-005", "문장 5를 또렷하게 말해요.")
    ]
    assert r.improved_scripts[0].original == "문장 5 내용이에요"


@pytest.mark.parametrize("bad", ["not json", "", {"feedbacks": []}])
def test_schema_violation_retries(llm_settings, bad):
    client = FakeLlm(resp(bad), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2 and len(r.feedbacks) == 3


def test_all_attempts_invalid_is_error(llm_settings):
    client = FakeLlm(resp("not json"))
    with pytest.raises(FeedbackServiceError):
        generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 3


def _cut(reason="max_output_tokens"):
    r = resp(payload(), status="incomplete")
    r.incomplete_details = SimpleNamespace(reason=reason)
    return r


def test_token_limit_raises_cap_once(llm_settings):
    client = FakeLlm(_cut(), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert [c["max_output_tokens"] for c in client.calls] == [
        fb.MAX_OUTPUT_TOKENS, fb.MAX_OUTPUT_TOKENS_RETRY,
    ]  # fmt: skip
    assert len(r.feedbacks) == 3


def test_token_limit_twice_fails(llm_settings):
    client = FakeLlm(_cut())
    with pytest.raises(FeedbackServiceError):
        generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2


def test_incomplete_or_refusal_fails_without_retry(llm_settings):
    client = FakeLlm(_cut("content_filter"))
    with pytest.raises(FeedbackServiceError):
        generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1

    refusal = resp(payload())
    refusal.output = [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]
    client = FakeLlm(refusal)
    with pytest.raises(FeedbackServiceError):
        generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1


def test_api_error_single_call(llm_settings):
    err = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    client = FakeLlm(err)
    with pytest.raises(FeedbackServiceError) as exc_info:
        generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1
    assert exc_info.value.__context__ is None


def test_missing_model_is_error(settings):
    settings.openai_llm_model = ""
    with pytest.raises(FeedbackServiceError):
        generate_feedback(rich_input(), FakeLlm(resp(payload())), settings)


def test_cost_log_and_haeyo_conversion(llm_settings, caplog):
    p = payload()
    p["feedbacks"][2]["suggestion"] = "근거를 먼저 말하십시오."
    p["feedbacks"][0]["title"] = "도입부 말이 빠릅니다"
    p["overall"]["summary"] = "핵심이 분명합니다. 근거를 보완하면 좋습니다."
    p["improvedScripts"][0]["reason"] = "근거를 붙였습니다."
    with caplog.at_level(logging.INFO):
        r = generate_feedback(rich_input(), FakeLlm(resp(p)), llm_settings)
    assert "feedback llm call: model=gpt-5-mini" in caplog.text
    assert "est_cost_usd=0.00375" in caplog.text  # 3000×0.25/1M + 1500×2.0/1M
    assert r.feedbacks[2].suggestion == "근거를 먼저 말하세요."
    assert r.feedbacks[0].title == "도입부 말이 빨라요"
    assert r.overall.summary == "핵심이 분명해요. 근거를 보완하면 좋아요."
    assert r.improved_scripts[0].reason == "근거를 붙였어요."
    assert "해요체" not in caplog.text
    assert llm_settings.openai_api_key.get_secret_value() not in caplog.text
    assert "문장 8 내용이에요" not in caplog.text


def test_prompt_rules():
    prompt = fb.load_system_prompt()
    assert "해요체" in prompt and "자리표시자" in prompt
    assert "후보를 새로 만들거나 빼지 않는다" in prompt


def test_feedback_validates_against_report_schema(llm_settings):
    r = generate_feedback(rich_input(), FakeLlm(resp(payload())), llm_settings)
    for f in r.feedbacks:
        dumped = f.model_dump(by_alias=True)
        assert set(dumped) == {
            "id", "category", "type", "severity", "start", "end", "title", "description",
            "evidence", "suggestion",
        }  # fmt: skip
        assert 0 <= dumped["start"] < dumped["end"] <= DURATION


# ---------- 리뷰 반영: 재시도 실패, 숫자 규칙, 중괄호, overall ----------


def _with_bad_strength():
    p = payload()
    p["overall"]["strengths"] = ["시선 80% 유지", "핵심이 분명해요"]
    return p


@pytest.mark.parametrize(
    "second",
    [
        openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com")),
        "refusal",
        "cut",
    ],
)
def test_failed_retry_keeps_earlier_usable_result(llm_settings, second):
    if second == "refusal":
        second = resp(payload())
        second.output = [SimpleNamespace(content=[SimpleNamespace(type="refusal")])]
    elif second == "cut":
        second = _cut("content_filter")
    client = FakeLlm(resp(_with_bad_strength()), second)
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2
    assert len(r.feedbacks) == 3
    assert r.overall.strengths == ["핵심이 분명해요"]  # 80% 항목만 빠짐


@pytest.mark.parametrize(
    "bad",
    [
        "추임새를 세 번 썼어요.",
        "분당 사백오십 음절이에요.",
        "열두 초 동안 멈췄어요.",
        "두 번 반복했어요.",
    ],
)
def test_measurements_as_hangul_numerals_are_violations(llm_settings, bad):
    p = payload()
    p["feedbacks"][0]["description"] = bad
    client = FakeLlm(resp(p), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2 and len(r.feedbacks) == 3


@pytest.mark.parametrize(
    "ok", ["한 번 쉬고 말해 보세요.", "한 문장으로 줄여 보세요.", "이번에는 천천히요."]
)
def test_advice_numerals_are_allowed(llm_settings, ok):
    p = payload()
    p["feedbacks"][0]["suggestion"] = ok
    client = FakeLlm(resp(p))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1 and r.feedbacks[0].suggestion == ok


def test_numbers_quoted_from_evidence_are_allowed(llm_settings):
    # CLARITY 근거 문장(s-008)이 "문장 8 내용이에요" → "8"은 인용으로 허용
    p = payload()
    p["feedbacks"][2]["description"] = "'문장 8'이 무엇을 말하는지 모호해요."
    client = FakeLlm(resp(p))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 1
    assert r.feedbacks[2].description == "'문장 8'이 무엇을 말하는지 모호해요."


def test_number_not_in_evidence_is_still_rejected(llm_settings):
    p = payload()
    p["feedbacks"][2]["description"] = "'문장 9'가 모호해요."
    client = FakeLlm(resp(p), resp(payload()))
    generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2


@pytest.mark.parametrize("bad", ["{{pace}} 빨라요.", "{음} 이 많아요.", "닫는 괄호 } 하나"])
def test_leftover_braces_are_violations(llm_settings, bad):
    p = payload()
    p["feedbacks"][0]["description"] = bad
    client = FakeLlm(resp(p), resp(payload()))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 2
    assert all("{" not in f.description and "}" not in f.description for f in r.feedbacks)


def test_retry_input_names_the_broken_rule(llm_settings):
    bad = payload()
    bad["feedbacks"][0]["description"] = "분당 420음절"
    bad["feedbacks"] = bad["feedbacks"][:2]  # c-03 빠짐
    client = FakeLlm(resp(bad), resp(payload()))
    generate_feedback(rich_input(), client, llm_settings)
    first, second = (c["input"] for c in client.calls)
    assert "[다시 쓸 때 지킬 것]" not in first
    assert "[다시 쓸 때 지킬 것]" in second
    assert "빠짐없이" in second and "자리표시자로만" in second
    assert second.startswith(first)  # 원래 입력 + 안내만 붙인다


def test_best_attempt_prefers_valid_overall(llm_settings):
    no_overall = payload()
    no_overall["overall"]["summary"] = "말 속도가 420이에요."  # summary 무효 → overall None
    worse = payload()
    worse["feedbacks"][0]["title"] = "3"  # 피드백 하나 빠짐
    # 1차: 피드백 3개, overall 없음 / 2차: 피드백 3개, overall 있음(한 항목만 위반) / 3차: 2개
    second = _with_bad_strength()
    client = FakeLlm(resp(no_overall), resp(second), resp(worse))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert len(client.calls) == 3
    assert r.overall.summary.startswith("전체 말 속도는") and len(r.feedbacks) == 3
    assert not r.overall_fallback


def test_improved_script_reason_digit_and_braces_rejected(llm_settings):
    p = payload()
    p["improvedScripts"] = [
        {"segmentId": "s-008", "improved": "문장 8을 다듬었어요.", "reason": "3군데 고쳤어요."},
        {"segmentId": "s-009", "improved": "{pace} 문장이에요", "reason": "다듬었어요."},
    ]
    client = FakeLlm(resp(p))
    r = generate_feedback(rich_input(), client, llm_settings)
    assert r.improved_scripts == []


# ---------- overall 기본 문구(절대 null 아님) ----------


def _all_text(o) -> list[str]:
    return [o.summary, *o.strengths, *o.improvements]


def test_fallback_uses_scores_and_top_feedback_titles():
    scores = {Cat.DELIVERY: 72, Cat.NONVERBAL: 85, Cat.CONTENT: 64}
    f_info = SimpleNamespace(title="시선이 벗어나요", severity=Severity.INFO, start=5.0)
    f_crit = SimpleNamespace(title="말이 너무 빨라요", severity=Severity.CRITICAL, start=50.0)
    o = fb.fallback_overall(scores, [f_info, f_crit])
    assert o.summary == "비언어가 가장 좋았어요. 내용 구성을 보완하면 더 좋아질 거예요."
    assert o.strengths == ["비언어가 좋아요"]
    assert o.improvements == ["말이 너무 빨라요", "시선이 벗어나요"]  # 심각한 것부터


def test_fallback_without_feedback_uses_low_categories():
    o = fb.fallback_overall({Cat.DELIVERY: 90, Cat.NONVERBAL: 70, Cat.CONTENT: 60}, [])
    assert o.strengths == ["전달력이 좋아요"]
    assert o.improvements == ["내용 구성을 더 다듬어 보세요", "비언어를 더 다듬어 보세요"]


def test_fallback_without_feedback_uses_candidate_titles():
    c = fb.Candidate(FeedbackType.FILLER, Severity.WARNING, 0, 5)
    o = fb.fallback_overall({Cat.DELIVERY: 70, Cat.NONVERBAL: 70, Cat.CONTENT: 70}, [], [c])
    assert o.summary == "세 영역이 고르게 나왔어요. 아래 피드백을 하나씩 연습해 보세요."
    assert o.improvements == ["추임새가 많아요"]


@pytest.mark.parametrize(
    "scores",
    [None, {}, {Cat.DELIVERY: 95, Cat.NONVERBAL: 95, Cat.CONTENT: 95},
     {Cat.DELIVERY: 10, Cat.NONVERBAL: 20, Cat.CONTENT: 30}],
)  # fmt: skip
def test_fallback_is_never_empty_and_haeyo(scores):
    o = fb.fallback_overall(scores, [])
    assert o.summary and o.strengths and o.improvements
    assert all(formal_sentences(t) == [] for t in _all_text(o))
    assert not any(ch.isdigit() for t in _all_text(o) for ch in t)


def test_default_titles_cover_every_candidate_type():
    for t in fb.CATEGORY_OF:
        if t is not FeedbackType.SLIDE_MISSED:
            assert t in fb.DEFAULT_TITLE


def test_llm_overall_invalid_falls_back(llm_settings):
    bad = payload()
    bad["overall"]["summary"] = "분당 320음절이에요."  # 매번 위반 → overall 없음
    client = FakeLlm(resp(bad))
    d = rich_input()
    d = FeedbackInput(
        **{**d.__dict__, "category_scores": {Cat.DELIVERY: 70, Cat.NONVERBAL: 88, Cat.CONTENT: 75}}
    )
    r = generate_feedback(d, client, llm_settings)
    assert len(client.calls) == 3
    assert r.overall_fallback
    assert r.overall.summary == "비언어가 가장 좋았어요. 전달력을 보완하면 더 좋아질 거예요."
    assert r.overall.improvements[0] == "도입부 말이 빨라요"  # 상위 피드백 제목(LLM이 쓴 것)


def test_llm_overall_empty_lists_are_filled(llm_settings):
    p = payload()
    p["overall"]["strengths"] = []
    r = generate_feedback(rich_input(), FakeLlm(resp(p)), llm_settings)
    assert r.overall.summary.startswith("전체 말 속도는")  # LLM summary는 그대로
    assert r.overall.strengths == ["끝까지 발표를 마쳤어요"]  # 점수 없음 → 기본 문구
    assert r.overall_fallback


def test_prompt_v2_count_wording_rule():
    prompt = fb.load_system_prompt()
    assert fb.PROMPT_VERSION == "v2"
    assert "'여러 번'" in prompt.replace('"', "'") and "'한 번 더'" in prompt.replace('"', "'")


def test_improved_script_keeps_presentation_style(llm_settings, caplog):
    # 발표자가 말할 개선 대본은 합쇼체가 자연스러우므로 바꾸지도 경고하지도 않는다
    p = payload()
    p["improvedScripts"] = [
        {
            "segmentId": "s-008",
            "improved": "들어주셔서 감사합니다.",
            "reason": "마무리를 다듬었어요.",
        }
    ]
    with caplog.at_level(logging.WARNING):
        r = generate_feedback(rich_input(), FakeLlm(resp(p)), llm_settings)
    assert r.improved_scripts[0].improved == "들어주셔서 감사합니다."
    assert "해요체" not in caplog.text


def test_clipped_end_is_rounded_to_two_decimals():
    # 영상 길이 44.517656초를 넘는 말 속도 구간(30~60초) → 끝은 44.52
    pace = [PacePoint(t=0, value=320), PacePoint(t=30, value=240)]
    (c,) = build_candidates(data(speech=speech(pace=pace), duration_sec=44.517656))
    assert (c.start, c.end) == (30.0, 44.52)


def test_language_and_profanity_prioritized_in_clarity_and_profanity_not_repeated(llm_settings):
    from app.services.pipeline.content import LanguageSeverity
    from app.services.pipeline.profanity_ko import ProfanityHit, ProfanityResult

    # 일반 CLARITY 4개(s-001, s-003, s-005, s-007) + LANGUAGE 1개(s-009) + 욕설 1개(s-011)
    ct = content_with(
        issue(IssueType.CLARITY, ["s-001"], "모호해요 1."),
        issue(IssueType.CLARITY, ["s-003"], "모호해요 2."),
        issue(IssueType.CLARITY, ["s-005"], "모호해요 3."),
        issue(IssueType.CLARITY, ["s-007"], "모호해요 4."),
        ContentIssue(
            type=IssueType.LANGUAGE,
            segment_ids=["s-009"],
            start=80.0,
            end=89.5,
            reason="발표를 거부하는 표현이 있어요.",
            severity=LanguageSeverity.SERIOUS,
        ),
    )
    sp = speech(
        profanity=ProfanityResult(
            hits=(ProfanityHit("x같네요", "x같", 100.0, 105.0, "s-011"),)
        )
    )
    inp = data(speech=sp, content=ct)
    cands = build_candidates(inp)
    # CLARITY 상한(MAX_PER_TYPE=4) 안에서도 언어 문제(s-009, s-011)가 최우선으로 살아남는다
    assert len(cands) == fb.MAX_PER_TYPE == 4
    seg_sets = [c.segment_ids for c in cands]
    assert ("s-009",) in seg_sets and ("s-011",) in seg_sets

    # LLM이 실수로 욕설 단어를 설명에 반복하면 거부하고 재시도한다
    bad_p = {
        "feedbacks": [
            item("c-01", "모호해요", "설명이 모호해요."),
            item("c-02", "모호해요", "근거가 부족해요."),
            item(
                "c-03",
                "발표 태도를 다듬어 보세요",
                "발표를 포기하는 표현이 있어요.",
                "발표 자리에서는 이 표현 대신 준비한 핵심만 간단히 말해 보세요.",
            ),
            item(
                "c-04",
                "부적절한 표현이 있어요",
                "X같네요 같은 표현은 부적절해요.",  # 욕설 단어 반복 → 위반!
                "발표 자리에서는 이 표현 대신 정중한 마무리 인사를 써 보세요.",
            ),
        ],
        "overall": {
            "summary": "표현을 다듬으면 좋아요.",
            "strengths": ["목소리"],
            "improvements": ["표현"],
        },
        "improvedScripts": [],
    }
    good_p = json.loads(json.dumps(bad_p))
    good_p["feedbacks"][3]["description"] = "발표 자리에 맞지 않는 거친 표현이 있어요."
    client = FakeLlm(resp(bad_p), resp(good_p))
    r = generate_feedback(inp, client, llm_settings)
    assert len(client.calls) == 2
    assert len(r.feedbacks) == 4
    assert all(f.type is FeedbackType.CLARITY for f in r.feedbacks)
    assert all(fb.verify_evidence(f.evidence, inp.transcript) for f in r.feedbacks)
