"""speech_metrics: 말 속도, 추임새, 반복어, 침묵, 시간. 외부 호출 없음."""

import json
from pathlib import Path

import pytest

from app.schemas.report import Segment, Transcript, Word, WordTag
from app.services.pipeline import fillers_ko
from app.services.pipeline.speech_metrics import (
    MIN_WINDOW_SPEECH_SEC,
    SILENCE_SEC,
    analyze_speech,
    count_syllables,
    word_core,
)
from app.services.pipeline.stt import normalize

FIXTURE = Path(__file__).parent / "fixtures" / "whisper_verbose_ko.json"


def tr(*segments: list[tuple[str, float, float]]) -> Transcript:
    """[(단어, 시작, 끝), ...] 목록들로 transcript를 만든다(목록 하나가 문장 하나)."""
    segs = []
    for n, words in enumerate(segments, start=1):
        ws = [Word(text=t, start=s, end=e, tag=None) for t, s, e in words]
        segs.append(
            Segment(
                id=f"s-{n:03d}",
                start=ws[0].start if ws else 0.0,
                end=ws[-1].end if ws else 0.0,
                text=" ".join(w.text for w in ws),
                words=ws,
            )
        )
    return Transcript(segments=segs)


def tags(result) -> list[tuple[str, WordTag | None]]:
    return [(w.text, w.tag) for s in result.transcript.segments for w in s.words]


def tag_of(result, text: str) -> WordTag | None:
    found = [t for w, t in tags(result) if w == text]
    assert len(found) == 1, f"{text!r} appears {len(found)} times"
    return found[0]


# ---------- 기본 함수 ----------


@pytest.mark.parametrize(
    ("text", "n"),
    [("서비스입니다.", 6), ("안녕하세요,", 5), ("2024년", 1), ("AI는", 1), ("...", 0), ("", 0)],
)
def test_count_syllables_counts_only_hangul(text, n):
    assert count_syllables(text) == n


def test_word_core_strips_punctuation():
    assert word_core("음,") == "음"
    assert word_core("이제...") == "이제"
    assert word_core("“그”") == "그"
    assert word_core("AI!") == "ai"


# ---------- 항상 추임새 ----------


@pytest.mark.parametrize("filler", ["음", "어", "으음", "어어", "음,", "어...", "어어어", "음음음"])
def test_always_filler_is_tagged_without_pause(filler):
    r = analyze_speech(
        tr([("저희가", 0.0, 0.5), (filler, 0.5, 0.7), ("발표", 0.7, 1.0)]), 2.0, None
    )
    assert tag_of(r, filler) is WordTag.FILLER


def test_elongated_filler_counts_under_short_form():
    r = analyze_speech(tr([("어어어", 0.0, 0.5), ("어어", 0.5, 1.0), ("음", 1.0, 1.2)]), 2.0, None)
    assert [(f.word, f.count) for f in r.metrics.fillers] == [("어어", 2), ("음", 1)]


def test_always_filler_with_suffix_is_not_filler():
    # "음식", "어디" 같은 단어는 추임새가 아니다
    r = analyze_speech(tr([("음식", 0.0, 0.5), ("어디", 1.0, 1.5), ("음은", 2.0, 2.3)]), 3.0, None)
    assert all(t is None for _, t in tags(r))


# ---------- 문맥 추임새: 뒤 쉼 기준 ----------


@pytest.mark.parametrize("word", sorted(fillers_ko.CONTEXT_FILLERS))
def test_context_filler_with_pause_after_is_filler(word):
    r = analyze_speech(tr([("저희가", 0.0, 0.5), (word, 0.5, 0.7), ("발표", 1.0, 1.4)]), 2.0, None)
    assert tag_of(r, word) is WordTag.FILLER


@pytest.mark.parametrize("word", sorted(fillers_ko.CONTEXT_FILLERS))
def test_context_filler_with_only_pause_before_is_not_filler(word):
    # 앞 쉼 1초, 뒤 쉼 없음 → 세지 않는다("…했습니다. [쉼] 그 결과는")
    r = analyze_speech(tr([("저희가", 0.0, 0.5), (word, 1.5, 1.7), ("발표", 1.7, 2.0)]), 3.0, None)
    assert tag_of(r, word) is None


@pytest.mark.parametrize("word", sorted(fillers_ko.CONTEXT_FILLERS))
def test_context_filler_without_pause_is_not_filler(word):
    r = analyze_speech(tr([("저희가", 0.0, 0.5), (word, 0.5, 0.7), ("발표", 0.8, 1.2)]), 2.0, None)
    assert tag_of(r, word) is None


def test_pause_after_exactly_threshold_counts():
    assert fillers_ko.CONTEXT_PAUSE_SEC == 0.3
    # 뒤 쉼 0.3초 (0.7 - 0.4는 부동소수점으로 0.2999…)
    r = analyze_speech(
        tr([("저희가", 0.0, 0.2), ("이제", 0.2, 0.4), ("발표", 0.7, 1.0)]), 2.0, None
    )
    assert tag_of(r, "이제") is WordTag.FILLER


def test_pause_after_just_under_threshold_does_not_count():
    # 뒤 쉼 0.29초
    r = analyze_speech(
        tr([("저희가", 0.0, 0.5), ("약간", 0.5, 0.7), ("발표", 0.99, 1.3)]), 2.0, None
    )
    assert tag_of(r, "약간") is None


def test_sentence_start_demonstrative_is_not_filler():
    words = [("했습니다.", 0.0, 0.8), ("그", 1.5, 1.6), ("결과는", 1.6, 2.1)]
    r = analyze_speech(tr(words), 3.0, None)
    assert tag_of(r, "그") is None


def test_context_filler_at_end_is_not_filler():
    # 마지막 단어 뒤는 쉼으로 보지 않는다
    r = analyze_speech(tr([("시작", 1.0, 1.5), ("좀", 1.5, 1.7)]), 5.0, None)
    assert tag_of(r, "좀") is None


@pytest.mark.parametrize(
    "word", ["그래서", "이제는", "저희", "뭐가", "막상", "약간의", "좀더", "그게"]
)
def test_words_with_particles_or_endings_are_never_fillers(word):
    # 앞뒤로 긴 쉼이 있어도 세지 않는다
    r = analyze_speech(tr([("네", 0.0, 0.3), (word, 1.0, 1.4), ("발표", 2.0, 2.4)]), 3.0, None)
    assert tag_of(r, word) is None
    assert r.metrics.filler_count == 0


def test_context_filler_with_punctuation_still_matches():
    r = analyze_speech(tr([("저희가", 0.0, 0.5), ("그,", 0.5, 0.6), ("발표", 1.0, 1.3)]), 2.0, None)
    assert tag_of(r, "그,") is WordTag.FILLER
    assert r.metrics.fillers[0].word == "그"


def test_pause_after_is_measured_across_segments():
    r = analyze_speech(
        tr([("저희가", 0.0, 0.5), ("이제", 0.5, 0.7)], [("발표", 1.2, 1.5)]), 2.0, None
    )
    assert tag_of(r, "이제") is WordTag.FILLER


# ---------- 문맥 추임새: 연속 ----------


@pytest.mark.parametrize("word", sorted(fillers_ko.CONTEXT_FILLERS))
def test_consecutive_same_context_filler_counts_without_pause(word):
    r = analyze_speech(
        tr([("네", 0.0, 0.2), (word, 0.2, 0.3), (word, 0.3, 0.4), ("사람", 0.4, 0.7)]), 1.0, None
    )
    assert [t for _, t in tags(r)] == [None, WordTag.FILLER, WordTag.FILLER, None]
    assert r.repeat_hits == []
    assert [(f.word, f.count) for f in r.metrics.fillers] == [(word, 2)]


def test_consecutive_context_filler_ignores_long_pause_between():
    r = analyze_speech(tr([("그", 0.0, 0.1), ("그", 3.0, 3.1), ("사람", 3.1, 3.5)]), 4.0, None)
    assert [t for _, t in tags(r)] == [WordTag.FILLER, WordTag.FILLER, None]


def test_triple_context_filler_all_count():
    r = analyze_speech(
        tr([("저", 0.0, 0.1), ("저,", 0.1, 0.2), ("저", 0.2, 0.3), ("발표", 0.3, 0.6)]), 1.0, None
    )
    assert r.metrics.filler_count == 3


def test_different_context_fillers_in_a_row_are_not_consecutive():
    # "그 저"는 같은 추임새가 아니다. 뒤 쉼이 없으면 세지 않는다
    r = analyze_speech(tr([("그", 0.0, 0.1), ("저", 0.1, 0.2), ("사람", 0.2, 0.5)]), 1.0, None)
    assert all(t is None for _, t in tags(r))


def test_context_filler_next_to_attached_form_is_not_consecutive():
    # "그 그래서": 조사·어미 붙은 단어와는 연속으로 보지 않는다
    r = analyze_speech(tr([("그", 0.0, 0.1), ("그래서", 0.1, 0.5), ("끝", 0.5, 0.7)]), 1.0, None)
    assert all(t is None for _, t in tags(r))


# ---------- 반복어 ----------


def test_consecutive_same_word_is_repeat_from_second():
    r = analyze_speech(
        tr([("저희", 0.0, 0.3), ("저희", 0.3, 0.6), ("저희,", 0.6, 0.9), ("저희가", 0.9, 1.2)]),
        2.0,
        None,
    )
    assert [t for _, t in tags(r)] == [None, WordTag.REPEAT, WordTag.REPEAT, None]
    assert [h.word for h in r.repeat_hits] == ["저희", "저희"]


def test_non_consecutive_same_word_is_not_repeat():
    r = analyze_speech(tr([("발표", 0.0, 0.3), ("연습", 0.3, 0.6), ("발표", 0.6, 0.9)]), 2.0, None)
    assert all(t is None for _, t in tags(r))


def test_filler_takes_precedence_over_repeat():
    r = analyze_speech(tr([("음", 0.0, 0.2), ("음", 0.2, 0.4), ("발표", 0.4, 0.8)]), 2.0, None)
    assert [t for _, t in tags(r)] == [WordTag.FILLER, WordTag.FILLER, None]
    assert r.repeat_hits == []


# ---------- 침묵·말 속도 ----------


def test_silence_threshold_inclusive():
    assert SILENCE_SEC == 2.0
    r = analyze_speech(
        tr([("가", 0.0, 1.0), ("나", 3.0, 4.0), ("다", 5.99, 6.5), ("라", 9.8, 10.0)]), 10.0, None
    )
    # 1.0→3.0 = 2.0초(침묵), 4.0→5.99 = 1.99초(아님), 6.5→9.8 = 3.3초(침묵)
    assert r.metrics.silence_count == 2
    assert r.metrics.longest_silence_sec == 3.3
    assert [(s.start, s.end) for s in r.silences] == [(1.0, 3.0), (6.5, 9.8)]


def test_pace_excludes_silence_from_speaking_time():
    # 7음절, 발화 1+1+1초(사이 3초 침묵 제외) → 7 / (3/60) = 140
    r = analyze_speech(
        tr([("가나다", 0.0, 1.0), ("라마", 1.0, 2.0)], [("바사.", 5.0, 6.0)]), 6.0, None
    )
    assert r.syllable_count == 7
    assert r.speaking_sec == 3.0
    assert r.metrics.syllables_per_min == 140


def test_short_gaps_count_as_speaking_time():
    # 단어 사이 1.5초는 침묵이 아니라 발화 시간에 포함: 4음절 / 3초 → 80
    r = analyze_speech(tr([("가나", 0.0, 0.75), ("다라", 2.25, 3.0)]), 3.0, None)
    assert r.speaking_sec == 3.0
    assert r.metrics.syllables_per_min == 80


def test_pace_series_30s_windows():
    words = []
    # 0~30초: 1초마다 "가나" (2음절, 0.5초) → 60음절 / 30초 = 120
    words += [("가나", float(i), i + 0.5) for i in range(30)]
    # 30~60초: 1초마다 "가나다라" → 120음절 / 29.5초(59.5초에 끝남)
    words += [("가나다라", float(i), i + 0.5) for i in range(30, 60)]
    # 60~70초: 말 없음 / 70초 영상 → 마지막 구간은 빠진다
    r = analyze_speech(tr(words), 70.0, None)
    series = [(p.t, p.value) for p in r.metrics.pace_series]
    assert series == [(0, 120), (30, round(120 * 60 / 29.5))]


def test_middle_window_with_too_little_speech_is_zero():
    assert MIN_WINDOW_SPEECH_SEC == 3.0
    words = [("가나다", float(i), i + 0.9) for i in range(30)]
    words += [("가나다라마바", 40.0, 42.0)]  # 30~60초: 발화 2초뿐
    words += [("가나다", float(i), i + 0.9) for i in range(60, 75)]
    r = analyze_speech(tr(words), 75.0, None)
    assert [(p.t, p.value > 0) for p in r.metrics.pace_series] == [
        (0, True),
        (30, False),
        (60, True),
    ]


@pytest.mark.parametrize(("tail_speech", "kept"), [(2.9, False), (3.0, True)])
def test_short_last_window_is_dropped(tail_speech, kept):
    words = [("가나다", float(i), i + 0.9) for i in range(30)]
    words += [("가나다", 30.0, 30.0 + tail_speech)]
    r = analyze_speech(tr(words), 35.0, None)
    assert [p.t for p in r.metrics.pace_series] == ([0, 30] if kept else [0])


def test_tiny_total_speaking_time_gives_zero_pace():
    # 10음절이 0.01초에 몰린 타임스탬프 → 분당 60000이 되지 않게
    r = analyze_speech(tr([("가나다라마바사아자차", 1.0, 1.01)]), 5.0, None)
    assert r.syllable_count == 10
    assert r.metrics.syllables_per_min == 0
    assert r.metrics.pace_series == []


def test_word_straddling_window_splits_syllables():
    # 28~32초 단어(8음절)가 두 구간에 반씩 걸친다
    words = [("가나다라마바사아", 28.0, 32.0), ("가나다라", 32.0, 34.0)]
    r = analyze_speech(tr([("가", 25.0, 28.0), *words]), 34.0, None)
    first, second = r.metrics.pace_series
    # 두 번째 구간: 4 + 4음절 / 4초 = 120
    assert second.value == 120
    # 첫 구간: 1 + 4음절 / 5초(25~30) = 60
    assert first.value == 60


def test_zero_length_words_are_handled():
    words = [("가나다", 0.0, 1.0), ("라", 1.0, 1.0), ("마바", 1.0, 3.0)]
    r = analyze_speech(tr(words), 3.0, None)
    assert r.syllable_count == 6
    assert r.metrics.syllables_per_min == 120
    assert [p.t for p in r.metrics.pace_series] == [0]


def test_duration_shorter_than_last_word_end():
    # 영상 길이(29초)보다 단어가 늦게 끝나도 마지막 단어까지 구간에 넣는다
    words = [("가나다", float(i), i + 0.9) for i in range(31)]
    r = analyze_speech(tr(words), 29.0, None)
    assert r.metrics.pace_series[0].t == 0
    assert r.syllable_count == 93


def test_unsorted_and_overlapping_words_across_segments():
    # 문장 순서와 시각 순서가 다르고, 단어 시각이 겹친다
    transcript = tr(
        [("발표", 2.0, 2.5), ("이제", 2.6, 2.8)],
        [("저희가", 0.0, 3.0), ("음", 3.0, 3.2)],
    )
    r = analyze_speech(transcript, 4.0, None)
    # "이제" 앞 단어 중 가장 늦게 끝난 것은 "저희가"(3.0) → 쉼 없음, 뒤("음" 3.0)도 쉼 없음
    assert tag_of(r, "이제") is None
    assert tag_of(r, "음") is WordTag.FILLER
    # 태그는 원래 문장·자리로 돌아간다
    assert [w.text for w in r.transcript.segments[1].words] == ["저희가", "음"]
    assert r.filler_hits[0].segment_id == "s-002"
    assert r.metrics.silence_count == 0


def test_nested_word_does_not_create_fake_pause():
    # "그" 바로 앞 단어(1.0~1.2)는 긴 단어(0.0~2.0) 안에 있다 → 실제 쉼은 0
    r = analyze_speech(
        tr([("설명드리면", 0.0, 2.0), ("네", 1.0, 1.2), ("그", 2.0, 2.1), ("사람", 2.1, 2.5)]),
        3.0,
        None,
    )
    assert tag_of(r, "그") is None


def test_repeat_not_counted_across_long_silence():
    r = analyze_speech(tr([("발표", 0.0, 0.5), ("발표", 2.5, 3.0)]), 4.0, None)
    assert [t for _, t in tags(r)] == [None, None]
    r = analyze_speech(tr([("발표", 0.0, 0.5), ("발표", 2.49, 3.0)]), 4.0, None)
    assert [t for _, t in tags(r)] == [None, WordTag.REPEAT]


# ---------- 추임새 집계 ----------


def test_filler_stats_sorted_by_count_then_first_seen():
    words = [
        ("음", 0.0, 0.2), ("발표", 0.2, 0.5), ("어", 0.5, 0.7), ("저희가", 0.7, 1.0),
        ("어", 1.0, 1.2), ("시작", 1.2, 1.5), ("음", 1.5, 1.7), ("약간", 2.1, 2.3),
        ("끝", 2.6, 2.8),
    ]  # fmt: skip
    r = analyze_speech(tr(words), 3.0, None)
    assert r.metrics.filler_count == 5
    assert [(f.word, f.count) for f in r.metrics.fillers] == [("음", 2), ("어", 2), ("약간", 1)]
    assert [h.segment_id for h in r.filler_hits] == ["s-001"] * 5
    assert r.filler_hits[0].start == 0.0


def test_tags_written_to_new_transcript_only():
    original = tr([("음", 0.0, 0.2), ("발표", 0.2, 0.5)])
    r = analyze_speech(original, 1.0, None)
    assert original.segments[0].words[0].tag is None
    dumped = r.transcript.model_dump(by_alias=True)
    assert dumped["segments"][0]["words"][0] == {
        "text": "음",
        "start": 0.0,
        "end": 0.2,
        "tag": "FILLER",
    }
    assert dumped["segments"][0]["words"][1]["tag"] is None
    assert dumped["segments"][0]["text"] == original.segments[0].text


# ---------- 빈 대본 ----------


@pytest.mark.parametrize("transcript", [Transcript(segments=[]), tr([])])
def test_empty_transcript(transcript):
    r = analyze_speech(transcript, 320.0, 300)
    m = r.metrics
    assert m.syllables_per_min == 0
    assert m.pace_series == []
    assert m.filler_count == 0 and m.fillers == []
    assert m.silence_count == 0 and m.longest_silence_sec == 0.0
    assert m.over_time_sec == 0.0  # 발표 시간(말한 구간) 0초
    assert r.speech_span_sec == 0.0
    assert r.speaking_sec == 0.0


def test_punctuation_only_words_are_ignored():
    r = analyze_speech(tr([("...", 0.0, 0.2), ("발표", 0.2, 0.5), ("...", 0.5, 0.6)]), 1.0, None)
    assert r.syllable_count == 2
    assert all(t is None for _, t in tags(r))


# ---------- 시간 ----------


def test_no_time_limit():
    r = analyze_speech(tr([("가", 0.0, 1.0)]), 500.0, None)
    assert r.metrics.time_limit_sec is None
    assert r.metrics.over_time_sec == 0.0
    assert r.under_time_sec == 0.0 and r.time_used_ratio is None and not r.is_under_time


def span(start: float, end: float) -> Transcript:
    """start에 시작해 end에 끝나는 발표(중간은 1초 간격 단어)."""
    words = [("가", float(t), t + 0.5) for t in range(int(start), int(end))]
    words[0] = ("가", start, start + 0.5)
    words.append(("끝", end - 0.5, end))
    return tr(words)


def test_over_time_uses_first_to_last_word():
    # 영상 330초지만 말은 10초~320.44초 → 310.44초, 제한 300초 → 10.4초 초과
    r = analyze_speech(span(10.0, 320.44), 330.0, 300)
    assert r.metrics.time_limit_sec == 300
    assert r.speech_span_sec == 310.44
    assert r.metrics.over_time_sec == 10.4
    assert r.under_time_sec == 0.0 and not r.is_under_time


def test_dead_air_before_and_after_is_not_over_time():
    # 영상 305초, 앞 8초·뒤 2초 무음 → 발표 295초는 제한 300초 안
    r = analyze_speech(span(8.0, 303.0), 305.0, 300)
    assert r.metrics.over_time_sec == 0.0
    assert r.under_time_sec == 5.0


def test_time_rounding_is_half_up():
    r = analyze_speech(span(0.0, 302.25), 303.0, 300)
    assert r.metrics.over_time_sec == 2.3  # round(2.25, 1)은 2.2
    r = analyze_speech(tr([("가", 0.0, 1.0), ("나", 3.25, 4.0)]), 5.0, None)
    assert r.metrics.longest_silence_sec == 2.3


@pytest.mark.parametrize(
    ("end", "under", "is_under"),
    [(209.9, 90.1, True), (210.0, 90.0, False), (300.0, 0.0, False)],
)
def test_under_time(end, under, is_under):
    # 영상은 늘 400초(뒤에 빈 녹화), 말은 0초~end
    r = analyze_speech(span(0.0, end), 400.0, 300)
    assert r.metrics.over_time_sec == 0.0
    assert r.under_time_sec == under
    assert r.is_under_time is is_under
    assert r.time_used_ratio == round(end / 300, 3)


@pytest.mark.parametrize(("duration", "limit"), [(-1.0, None), (float("nan"), None), (10.0, 0)])
def test_invalid_input(duration, limit):
    with pytest.raises(ValueError):
        analyze_speech(tr([]), duration, limit)


# ---------- stt 결과와 연결 ----------


def test_works_on_stt_output_and_matches_spec_shape():
    stt_result = normalize(json.loads(FIXTURE.read_text(encoding="utf-8")))
    r = analyze_speech(stt_result.transcript, stt_result.duration, 300)
    assert tag_of(r, "음") is WordTag.FILLER
    assert tag_of(r, "그래서") is None
    speech = r.metrics.model_dump(by_alias=True)
    assert set(speech) == {
        "syllablesPerMin", "paceSeries", "fillerCount", "fillers", "silenceCount",
        "longestSilenceSec", "timeLimitSec", "overTimeSec",
    }  # fmt: skip
    assert speech["fillers"] == [{"word": "음", "count": 1}]
    assert isinstance(speech["syllablesPerMin"], int) and speech["syllablesPerMin"] > 0
