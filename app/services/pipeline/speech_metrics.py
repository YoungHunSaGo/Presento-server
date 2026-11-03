"""말하기 지표: 말 속도, 추임새, 반복어, 침묵, 시간 초과·미달.

입력은 stt가 정규화한 transcript다. 모든 수치는 여기서 코드로 계산한다(LLM 아님).
결과로 명세 metrics.speech(SpeechMetrics)와, words[].tag(FILLER/REPEAT)를 채운 transcript,
그리고 피드백 생성에 쓸 근거(추임새·침묵 위치, 시간 미달 여부)를 돌려준다.

규칙:
- 말 속도: 한글 음절 수(공백·문장부호·숫자·영문 제외) / 발화 시간(분).
  발화 시간은 첫 단어 시작~마지막 단어 끝에서 침묵(SILENCE_SEC 이상 단어 사이 공백)을 뺀 값.
- paceSeries: PACE_WINDOW_SEC 구간별. 구간에 걸친 단어의 음절은 겹친 시간 비율로 나눈다.
  구간 안 발화가 MIN_WINDOW_SPEECH_SEC보다 짧으면 값이 튀므로 0으로 둔다(차트에서 "말 없음").
  단, 마지막 구간이 그렇게 짧으면 차트 끝이 0으로 떨어져 보이므로 아예 넣지 않는다.
  전체 발화가 MIN_WINDOW_SPEECH_SEC보다 짧으면 syllablesPerMin도 0(타임스탬프가 몰려 값이 튐).
- 추임새: fillers_ko 사전. 단어에서 문장부호만 떼고 통째로 같을 때만 맞는다(조사·어미 붙으면 아님).
  문맥 추임새는 뒤 쉼이 CONTEXT_PAUSE_SEC 이상일 때만(앞 쉼은 보지 않는다). 마지막 단어의 뒤는
  쉼으로 보지 않는다(말을 끝낸 뒤 쉼은 자연스럽다). 같은 문맥 추임새가 연달아 나오면("그 그")
  쉼과 상관없이 모두 추임새다.
- 반복어: 바로 앞 단어와 같은 단어(문장부호 무시)는 두 번째부터 REPEAT. FILLER가 우선한다.
  사이에 침묵(SILENCE_SEC 이상)이 있으면 반복으로 보지 않는다.
- 쉼은 "앞 단어들 중 가장 늦게 끝난 시각"부터 잰다(단어 시각이 겹쳐도 가짜 쉼이 안 생기게).
- 시간 초과·미달은 영상 길이가 아니라 첫 단어 시작~마지막 단어 끝(speech_span_sec)으로 잰다.
  녹화 앞뒤의 빈 시간은 발표 시간이 아니다.
- 쉼·침묵 비교는 소수 둘째 자리로 반올림한 뒤 한다(0.7 - 0.4 = 0.2999… 같은 오차 방지).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from app.schemas.report import (
    FillerStat,
    PacePoint,
    Segment,
    SpeechMetrics,
    Transcript,
    Word,
    WordTag,
)
from app.services.pipeline.fillers_ko import ALWAYS_FILLERS, CONTEXT_FILLERS, CONTEXT_PAUSE_SEC

SILENCE_SEC = 2.0
PACE_WINDOW_SEC = 30
MIN_WINDOW_SPEECH_SEC = 3.0
# 제한 시간의 이 비율보다 짧게 끝나면 UNDER_TIME으로 본다.
UNDER_TIME_RATIO = 0.7

_GAP_DECIMALS = 2
_ELONGATION = re.compile(r"(.)\1{2,}")


@dataclass(frozen=True)
class Silence:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return round(self.end - self.start, _GAP_DECIMALS)


@dataclass(frozen=True)
class WordHit:
    """추임새·반복어가 나온 자리. 피드백 근거(타임라인)용."""

    word: str
    start: float
    end: float
    segment_id: str


@dataclass(frozen=True)
class SpeechAnalysis:
    metrics: SpeechMetrics
    transcript: Transcript  # words[].tag를 채운 새 transcript
    syllable_count: int
    speaking_sec: float
    silences: list[Silence]
    filler_hits: list[WordHit]
    repeat_hits: list[WordHit]
    speech_span_sec: float  # 첫 단어 시작~마지막 단어 끝(발표 시간). 시간 초과·미달 기준
    under_time_sec: float  # 제한 시간보다 짧은 초(제한 없으면 0)
    time_used_ratio: float | None  # 발표 시간 / 제한 시간(제한 없으면 None)
    is_under_time: bool


def word_core(text: str) -> str:
    """문장부호·공백을 뗀 단어(영문은 소문자)."""
    return "".join(ch for ch in text if ch.isalnum()).lower()


def count_syllables(text: str) -> int:
    """완성형 한글 음절 수."""
    return sum(1 for ch in text if "가" <= ch <= "힣")


def _filler_key(core: str) -> str:
    return _ELONGATION.sub(r"\1\1", core)


def _gap(a_end: float, b_start: float) -> float:
    return round(b_start - a_end, _GAP_DECIMALS)


def _round_int(x: float) -> int:
    return int(math.floor(x + 0.5))


def _round1(x: float) -> float:
    """소수 첫째 자리 반올림(사사오입). round()의 은행가 반올림·이진 오차를 피한다."""
    return math.floor(round(x * 10, 6) + 0.5) / 10


@dataclass
class _W:
    seg: int
    idx: int
    word: Word
    core: str
    tag: WordTag | None = None


def _flatten(transcript: Transcript) -> list[_W]:
    items = [
        _W(si, wi, w, word_core(w.text))
        for si, seg in enumerate(transcript.segments)
        for wi, w in enumerate(seg.words)
    ]
    items.sort(key=lambda x: (x.word.start, x.word.end))
    return items


def _tag(items: list[_W]) -> None:
    seq = [x for x in items if x.core]
    # prev_end[k]: seq[k] 앞 단어들 중 가장 늦게 끝난 시각
    prev_end: list[float | None] = []
    latest: float | None = None
    for x in seq:
        prev_end.append(latest)
        latest = x.word.end if latest is None else max(latest, x.word.end)
    for k, cur in enumerate(seq):
        prev = seq[k - 1] if k > 0 else None
        nxt = seq[k + 1] if k + 1 < len(seq) else None
        pe = prev_end[k]
        gap_before = _gap(pe, cur.word.start) if pe is not None else None
        key = _filler_key(cur.core)
        if key in ALWAYS_FILLERS:
            cur.tag = WordTag.FILLER
        elif cur.core in CONTEXT_FILLERS:
            stutter = (prev is not None and prev.core == cur.core) or (
                nxt is not None and nxt.core == cur.core
            )
            after = nxt is not None and _gap(cur.word.end, nxt.word.start) >= CONTEXT_PAUSE_SEC
            if stutter or after:
                cur.tag = WordTag.FILLER
        if (
            cur.tag is None
            and prev is not None
            and prev.core == cur.core
            and (gap_before is None or gap_before < SILENCE_SEC)
        ):
            cur.tag = WordTag.REPEAT


def _silences_and_runs(items: list[_W]) -> tuple[list[Silence], list[tuple[float, float]]]:
    """단어 사이 침묵과, 침묵으로 나뉜 발화 구간들."""
    silences: list[Silence] = []
    runs: list[tuple[float, float]] = []
    if not items:
        return silences, runs
    run_start = items[0].word.start
    run_end = items[0].word.end
    for x in items[1:]:
        if _gap(run_end, x.word.start) >= SILENCE_SEC:
            silences.append(Silence(run_end, x.word.start))
            runs.append((run_start, run_end))
            run_start = x.word.start
        run_end = max(run_end, x.word.end)
    runs.append((run_start, run_end))
    return silences, runs


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _pace_series(
    items: list[_W], runs: list[tuple[float, float]], duration: float
) -> list[PacePoint]:
    if not items:
        return []
    end = max(duration, max(x.word.end for x in items))
    n = max(1, math.ceil(end / PACE_WINDOW_SEC))
    points: list[PacePoint] = []
    last_speech = 0.0
    for i in range(n):
        t0, t1 = i * PACE_WINDOW_SEC, (i + 1) * PACE_WINDOW_SEC
        speech = sum(_overlap(r0, r1, t0, t1) for r0, r1 in runs)
        syl = 0.0
        for x in items:
            s = count_syllables(x.word.text)
            if not s:
                continue
            w0, w1 = x.word.start, x.word.end
            if w1 > w0:
                syl += s * _overlap(w0, w1, t0, t1) / (w1 - w0)
            elif t0 <= w0 < t1:
                syl += s
        value = _round_int(syl * 60 / speech) if speech >= MIN_WINDOW_SPEECH_SEC else 0
        points.append(PacePoint(t=t0, value=value))
        last_speech = speech
    if last_speech < MIN_WINDOW_SPEECH_SEC:
        points.pop()  # 짧은 마지막 구간은 차트 끝을 0으로 떨어뜨리므로 뺀다
    return points


def _hits(transcript: Transcript, items: list[_W], tag: WordTag) -> list[WordHit]:
    return [
        WordHit(
            word=_filler_key(x.core) if tag is WordTag.FILLER else x.core,
            start=x.word.start,
            end=x.word.end,
            segment_id=transcript.segments[x.seg].id,
        )
        for x in items
        if x.tag is tag
    ]


def _tagged_transcript(transcript: Transcript, items: list[_W]) -> Transcript:
    tags = {(x.seg, x.idx): x.tag for x in items}
    segments = [
        Segment(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            words=[
                Word(text=w.text, start=w.start, end=w.end, tag=tags[(si, wi)])
                for wi, w in enumerate(seg.words)
            ],
        )
        for si, seg in enumerate(transcript.segments)
    ]
    return Transcript(segments=segments)


def analyze_speech(
    transcript: Transcript, duration_sec: float, time_limit_sec: int | None
) -> SpeechAnalysis:
    """transcript와 영상 길이(초), 발표 제한 시간(초, 없으면 None)으로 말하기 지표를 낸다."""
    if not math.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("duration_sec must be a non-negative finite number")
    if time_limit_sec is not None and time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive")

    items = _flatten(transcript)
    _tag(items)
    silences, runs = _silences_and_runs(items)

    syllables = sum(count_syllables(x.word.text) for x in items)
    speaking = round(sum(r1 - r0 for r0, r1 in runs), _GAP_DECIMALS)
    spm = _round_int(syllables * 60 / speaking) if speaking >= MIN_WINDOW_SPEECH_SEC else 0

    filler_hits = _hits(transcript, items, WordTag.FILLER)
    repeat_hits = _hits(transcript, items, WordTag.REPEAT)
    counts = Counter(h.word for h in filler_hits)
    first_seen = {}
    for i, h in enumerate(filler_hits):
        first_seen.setdefault(h.word, i)
    fillers = [
        FillerStat(word=w, count=c)
        for w, c in sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    ]

    span = (
        round(max(x.word.end for x in items) - items[0].word.start, _GAP_DECIMALS) if items else 0.0
    )
    if time_limit_sec is None:
        over = under = 0.0
        ratio = None
        is_under = False
    else:
        over = _round1(max(0.0, span - time_limit_sec))
        under = _round1(max(0.0, time_limit_sec - span))
        ratio = round(span / time_limit_sec, 3)
        is_under = span < time_limit_sec * UNDER_TIME_RATIO

    metrics = SpeechMetrics(
        syllables_per_min=spm,
        pace_series=_pace_series(items, runs, duration_sec),
        filler_count=len(filler_hits),
        fillers=fillers,
        silence_count=len(silences),
        longest_silence_sec=_round1(max((s.end - s.start for s in silences), default=0.0)),
        time_limit_sec=time_limit_sec,
        over_time_sec=over,
    )
    return SpeechAnalysis(
        metrics=metrics,
        transcript=_tagged_transcript(transcript, items),
        syllable_count=syllables,
        speaking_sec=speaking,
        silences=silences,
        filler_hits=filler_hits,
        repeat_hits=repeat_hits,
        speech_span_sec=span,
        under_time_sec=under,
        time_used_ratio=ratio,
        is_under_time=is_under,
    )
