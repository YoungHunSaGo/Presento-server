"""피드백 생성: 무엇을 어디서 지적할지는 코드가, 문장은 LLM이.

1. 후보 만들기(코드): speech_metrics·nonverbal·content 결과에서 피드백 후보를 만든다.
   후보마다 type, category, start/end, severity, 자리표시자 값, 문장 ID를 코드가 정한다.
   기준값은 아래 상수.
2. 정리(코드): 영상 길이 밖은 자르고(완전히 밖이면 버림), 같은 type의 겹치거나 MERGE_GAP_SEC 안에
   가까운 구간은 합친다(값·심각도 다시 계산). 심각도 높은 순으로, type마다 MAX_PER_TYPE개까지,
   전체 MAX_FEEDBACKS개를 고른 뒤 시간순으로 놓는다.
3. 문장 쓰기(LLM 1번): 후보 ID별 title·description·suggestion, overall(summary·strengths·
   improvements), improvedScripts(문장 ID·improved·reason). LLM은 새 후보를 만들 수 없다.
   숫자는 {pace} 같은 자리표시자로만 쓰고 코드가 실제 값으로 채운다.
4. 검증(코드): 아래는 그 항목을 버리고, 하나라도 있으면 무엇이 틀렸는지 알려 주며 다시 생성한다
   (최대 MAX_RETRIES번 더). 재시도가 끝나면 가장 좋은 응답(살아남은 피드백 수 → overall 유효 →
   위반 수 순)을 쓴다.
   - 자리표시자 밖의 숫자(그 후보의 근거 문장에 있는 숫자는 인용으로 허용), 측정값을 한글 수사로
     쓴 것("사백오십 음절", "세 번"), 그 후보에 없는 자리표시자, 채운 뒤 남은 중괄호
   - 빠진 후보(LLM이 쓰지 않은 후보)
   없는 후보 ID, 없는(또는 보내지 않은) 문장 ID는 그 항목만 버리고 재시도하지 않는다.
   스키마 위반·빈 응답도 재시도한다. 출력 상한으로 잘리면 상한을 한 번 올려 재시도한다.
   재시도 중 API 실패·거절·잘림이 나도, 앞서 쓸 만한 응답이 있으면 그것을 쓴다.
   처음부터 쓸 만한 응답이 없으면 FeedbackServiceError.
5. evidence(코드가 대본에서 그대로): 대본 순서로 정렬한 문장 ID 중 처음 이어진 묶음.
   content 계열은 그 문장들, 말 속도는 그 구간과 가장 많이 겹치는 문장, 추임새는 첫 문장,
   침묵은 침묵 직전 문장, 시간 초과는 제한 시간 시점의 문장. 비언어·시간 미달은 null.

말투: 해요체. 합쇼체 문장은 경고 로그만(재시도 안 함).
overall은 절대 None이 아니다. 후보가 0개라 LLM을 부르지 않았거나, LLM의 overall이 끝내 유효하지
않으면(또는 strengths·improvements가 비면) 코드가 카테고리 점수와 상위 피드백 제목으로 해요체
기본 문구를 만든다(fallback_overall). 카테고리 점수는 scoring이 먼저 계산해 넘긴다.

에러 구분: FeedbackServiceError → AI_SERVICE_ERROR
"""

import json
import logging
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import Settings
from app.models.enums import Purpose, ScoreCategory
from app.schemas.report import (
    Feedback,
    FeedbackType,
    ImprovedScript,
    Severity,
    Transcript,
)
from app.services.pipeline import llm
from app.services.pipeline.content import (
    ContentAnalysis,
    IssueType,
    LanguageSeverity,
    build_transcript_block,
)
from app.services.pipeline.korean_style import to_haeyo
from app.services.pipeline.llm import LlmClient
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.profanity_ko import contains_profanity, detect_profanity
from app.services.pipeline.speech_metrics import SpeechAnalysis

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v2"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# ---- 정리 ----
MAX_FEEDBACKS = 12
MAX_PER_TYPE = 4  # 한 종류가 목록을 다 차지하지 않게
MERGE_GAP_SEC = 5.0

# ---- 말 속도(분당 음절, 30초 구간 paceSeries 기준) ----
PACE_WINDOW_SEC = 30
PACE_FAST_SPM = 400
PACE_FAST_CRITICAL_SPM = 450
PACE_SLOW_SPM = 250
PACE_SLOW_CRITICAL_SPM = 200

# ---- 추임새 ----
FILLER_CLUSTER_GAP_SEC = 10.0  # 이 간격 안에 이어진 추임새를 한 무리로 본다
FILLER_CLUSTER_MAX_SEC = (
    60.0  # 한 무리의 최대 길이(드문드문 이어져 발표 전체가 한 무리가 되지 않게)
)
FILLER_CLUSTER_MIN = 3
FILLER_CLUSTER_CRITICAL = 6
FILLER_SPREAD_PER_MIN = 4.0  # 무리가 없어도 전체가 분당 이 이상이면 발표 전체 후보

# ---- 침묵(초) ----
LONG_SILENCE_SEC = 3.0
LONG_SILENCE_WARNING_SEC = 5.0
LONG_SILENCE_CRITICAL_SEC = 8.0

# ---- 시간 ----
OVER_TIME_CRITICAL_RATIO = 0.1  # 제한 시간의 10% 넘게 초과하면 CRITICAL
UNDER_TIME_WARNING_RATIO = 0.5  # 제한 시간의 50% 미만이면 WARNING(그 위는 INFO)

# ---- 비언어(초) ----
GAZE_AWAY_WARNING_SEC = 8.0
GAZE_AWAY_CRITICAL_SEC = 15.0
GESTURE_NONE_WARNING_SEC = 60.0

# ---- 호출 ----
MAX_OUTPUT_TOKENS = 8000
MAX_OUTPUT_TOKENS_RETRY = 12000  # 상한으로 잘렸을 때 한 번만 올려 본다
MAX_RETRIES = 2
MAX_IMPROVED_SCRIPTS = 5
MAX_OVERALL_ITEMS = 3

_SEVERITY_RANK = {Severity.INFO: 0, Severity.WARNING: 1, Severity.CRITICAL: 2}

CATEGORY_OF: dict[FeedbackType, ScoreCategory] = {
    FeedbackType.PACE_FAST: ScoreCategory.DELIVERY,
    FeedbackType.PACE_SLOW: ScoreCategory.DELIVERY,
    FeedbackType.FILLER: ScoreCategory.DELIVERY,
    FeedbackType.LONG_SILENCE: ScoreCategory.DELIVERY,
    FeedbackType.OVER_TIME: ScoreCategory.DELIVERY,
    FeedbackType.UNDER_TIME: ScoreCategory.DELIVERY,
    FeedbackType.GAZE_AWAY: ScoreCategory.NONVERBAL,
    FeedbackType.POSTURE_UNSTABLE: ScoreCategory.NONVERBAL,
    FeedbackType.GESTURE_NONE: ScoreCategory.NONVERBAL,
    FeedbackType.GESTURE_EXCESS: ScoreCategory.NONVERBAL,
    FeedbackType.STRUCTURE: ScoreCategory.CONTENT,
    FeedbackType.CLARITY: ScoreCategory.CONTENT,
    FeedbackType.SLIDE_MISSED: ScoreCategory.CONTENT,
}

# LLM에게 보여 줄 후보 상황 설명(코드가 정한 사실만)
_SITUATION: dict[FeedbackType, str] = {
    FeedbackType.PACE_FAST: "이 구간 말 속도가 기준보다 빨라요.",
    FeedbackType.PACE_SLOW: "이 구간 말 속도가 기준보다 느려요.",
    FeedbackType.FILLER: "이 구간에 추임새가 몰려 있어요.",
    FeedbackType.LONG_SILENCE: "말이 길게 끊긴 침묵이 있어요.",
    FeedbackType.OVER_TIME: "발표가 제한 시간을 넘었어요.",
    FeedbackType.UNDER_TIME: "발표가 제한 시간보다 많이 짧아요.",
    FeedbackType.GAZE_AWAY: "이 구간 동안 카메라(청중)를 보지 않았어요.",
    FeedbackType.POSTURE_UNSTABLE: "이 구간에 몸이 많이 흔들렸어요.",
    FeedbackType.GESTURE_NONE: "이 구간에 손동작이 거의 없었어요.",
    FeedbackType.GESTURE_EXCESS: "이 구간에 손동작이 지나치게 많았어요.",
    FeedbackType.STRUCTURE: "내용 구성(순서·도입·결론)에 문제가 있어요.",
    FeedbackType.CLARITY: "표현이 모호하거나 근거가 부족해요.",
}

# 자리표시자 뜻(LLM 안내용)
_PLACEHOLDER_MEANING: dict[str, str] = {
    "pace": "이 구간 말 속도(분당 음절)",
    "avgPace": "발표 전체 평균 말 속도(분당 음절)",
    "fillerCount": "이 구간 추임새 수",
    "topFiller": "가장 많이 쓴 추임새",
    "fillerPerMin": "분당 추임새 수",
    "silenceSec": "가장 긴 침묵 길이(초)",
    "overTimeSec": "초과한 시간(초)",
    "underTimeSec": "모자란 시간(초)",
    "timeLimitMin": "제한 시간(분)",
    "awaySec": "시선을 벗어난 시간(초)",
    "noneSec": "손동작이 없던 시간(초)",
    "gestureCount": "이 구간 손동작 수",
    "gesturePerMin": "이 구간 분당 손동작 수",
    "syllablesPerMin": "발표 전체 말 속도(분당 음절)",
    "totalFillerCount": "발표 전체 추임새 수",
    "silenceCount": "발표 전체 긴 침묵 수",
    "gazeFrontPercent": "정면을 본 비율(%)",
}

_PLACEHOLDER = re.compile(r"\{([A-Za-z]+)\}")
_DIGIT = re.compile(r"\d")  # 유니코드 숫자 전부(전각 포함)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
# 측정값을 한글 수사로 쓴 것: "사백오십 음절", "세 번", "열두 초". "한 번"(조언)은 허용.
_SINO_MEASURE = re.compile(
    r"[일이삼사오육칠팔구]*[십백천만][일이삼사오육칠팔구십백천만]*\s*"
    r"(?:음절|초|분|번|개|회|배|퍼센트|프로)"
)
_NATIVE_COUNT = re.compile(
    r"(?<![가-힣])(?:(?:열|스물|서른|마흔|쉰)\s?(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉)?"
    r"|두|세|네|다섯|여섯|일곱|여덟|아홉|스무)\s*(?:번|개|회|초|분|음절)"
)


class FeedbackServiceError(Exception):
    """LLM 호출 실패, 또는 쓸 수 있는 응답을 못 받았다."""


def _round1(x: float) -> float:
    return math.floor(x * 10 + 0.5) / 10


def _round_int(x: float) -> int:
    return int(math.floor(x + 0.5))


# ---------------------------------------------------------------- 후보


@dataclass(frozen=True)
class Candidate:
    type: FeedbackType
    severity: Severity
    start: float
    end: float
    values: dict[str, int | float | str] = field(default_factory=dict)
    segment_ids: tuple[str, ...] = ()
    context: str | None = None  # content 계열: content 모듈이 쓴 이유(LLM 참고용)
    id: str = ""  # 정리 뒤 c-01, c-02 …
    priority: int = 0  # 같은 type·심각도 안에서 우선순위(언어·태도 문제는 일반 CLARITY보다 우선)

    @property
    def category(self) -> ScoreCategory:
        return CATEGORY_OF[self.type]


@dataclass(frozen=True)
class FeedbackInput:
    purpose: Purpose
    duration_sec: float
    transcript: Transcript  # speech_metrics가 태그를 채운 대본
    speech: SpeechAnalysis
    nonverbal: NonverbalAnalysis | None = None
    content: ContentAnalysis | None = None
    # scoring이 먼저 계산한 카테고리 점수(overall 기본 문구용). 없으면 점수 없이 문구를 만든다.
    category_scores: dict[ScoreCategory, int] | None = None


@dataclass(frozen=True)
class OverallText:
    summary: str
    strengths: list[str]
    improvements: list[str]


@dataclass(frozen=True)
class LlmUsage:
    model: str
    attempts: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


@dataclass(frozen=True)
class FeedbackResult:
    feedbacks: list[Feedback]
    overall: OverallText  # 절대 None이 아니다(LLM이 못 쓰면 코드 기본 문구)
    improved_scripts: list[ImprovedScript]
    candidates: list[Candidate]
    usage: LlmUsage | None = None  # LLM을 부르지 않았으면 None
    dropped_items: int = 0
    overall_fallback: bool = False  # overall의 일부라도 코드 기본 문구를 썼는지
    raw: dict[str, Any] | None = None  # 채택한 LLM 응답 원본(중간 결과 저장·재계산용)


def _seg_ids_in(transcript: Transcript, start: float, end: float) -> tuple[str, ...]:
    return tuple(s.id for s in transcript.segments if s.end > start and s.start < end)


def _silence_severity(d: float) -> Severity:
    if d >= LONG_SILENCE_CRITICAL_SEC:
        return Severity.CRITICAL
    if d >= LONG_SILENCE_WARNING_SEC:
        return Severity.WARNING
    return Severity.INFO


def _gaze_severity(d: float) -> Severity:
    if d >= GAZE_AWAY_CRITICAL_SEC:
        return Severity.CRITICAL
    if d >= GAZE_AWAY_WARNING_SEC:
        return Severity.WARNING
    return Severity.INFO


def _gesture_none_severity(d: float) -> Severity:
    return Severity.WARNING if d >= GESTURE_NONE_WARNING_SEC else Severity.INFO


def _filler_cluster_severity(n: int) -> Severity:
    return Severity.CRITICAL if n >= FILLER_CLUSTER_CRITICAL else Severity.WARNING


def _pace_candidates(data: FeedbackInput) -> list[Candidate]:
    m = data.speech.metrics
    out: list[Candidate] = []
    for p in m.pace_series:
        if p.value <= 0:
            continue
        if p.value >= PACE_FAST_SPM:
            ftype = FeedbackType.PACE_FAST
            sev = Severity.CRITICAL if p.value >= PACE_FAST_CRITICAL_SPM else Severity.WARNING
        elif p.value <= PACE_SLOW_SPM:
            ftype = FeedbackType.PACE_SLOW
            sev = Severity.CRITICAL if p.value <= PACE_SLOW_CRITICAL_SPM else Severity.WARNING
        else:
            continue
        start, end = float(p.t), float(p.t + PACE_WINDOW_SEC)
        out.append(
            Candidate(
                type=ftype,
                severity=sev,
                start=start,
                end=end,
                values={"pace": p.value, "avgPace": m.syllables_per_min},
                segment_ids=_seg_ids_in(data.transcript, start, end),
            )
        )
    return out


def _filler_candidates(data: FeedbackInput) -> list[Candidate]:
    hits = data.speech.filler_hits
    if not hits:
        return []
    clusters: list[list] = []
    for h in hits:
        if (
            clusters
            and h.start - clusters[-1][-1].end <= FILLER_CLUSTER_GAP_SEC
            and h.end - clusters[-1][0].start <= FILLER_CLUSTER_MAX_SEC
        ):
            clusters[-1].append(h)
        else:
            clusters.append([h])
    out: list[Candidate] = []
    for c in clusters:
        if len(c) < FILLER_CLUSTER_MIN:
            continue
        counts: dict[str, int] = {}
        for h in c:
            counts[h.word] = counts.get(h.word, 0) + 1
        top = max(counts, key=lambda w: counts[w])
        out.append(
            Candidate(
                type=FeedbackType.FILLER,
                severity=_filler_cluster_severity(len(c)),
                start=c[0].start,
                end=c[-1].end,
                values={"fillerCount": len(c), "topFiller": top},
                segment_ids=tuple(dict.fromkeys(h.segment_id for h in c)),
            )
        )
    speaking_min = data.speech.speaking_sec / 60
    if not out and speaking_min > 0:
        per_min = len(hits) / speaking_min
        if per_min >= FILLER_SPREAD_PER_MIN:
            stats = data.speech.metrics.fillers
            out.append(
                Candidate(
                    type=FeedbackType.FILLER,
                    severity=Severity.INFO,
                    start=hits[0].start,
                    end=hits[-1].end,
                    values={
                        "fillerCount": len(hits),
                        "topFiller": stats[0].word if stats else hits[0].word,
                        "fillerPerMin": _round1(per_min),
                    },
                    segment_ids=(hits[0].segment_id,),
                )
            )
    return out


def _silence_candidates(data: FeedbackInput) -> list[Candidate]:
    out: list[Candidate] = []
    for s in data.speech.silences:
        d = s.end - s.start
        if d < LONG_SILENCE_SEC:
            continue
        before = [seg.id for seg in data.transcript.segments if seg.start < s.start]
        out.append(
            Candidate(
                type=FeedbackType.LONG_SILENCE,
                severity=_silence_severity(d),
                start=s.start,
                end=s.end,
                values={"silenceSec": _round1(d)},
                segment_ids=(before[-1],) if before else (),
            )
        )
    return out


def _time_candidates(data: FeedbackInput) -> list[Candidate]:
    sp = data.speech
    limit = sp.metrics.time_limit_sec
    words = [w for seg in data.transcript.segments for w in seg.words]
    if limit is None or not words:
        return []
    first = min(w.start for w in words)
    last = max(w.end for w in words)
    over = sp.metrics.over_time_sec
    out: list[Candidate] = []
    if over > 0:
        sev = Severity.CRITICAL if over > limit * OVER_TIME_CRITICAL_RATIO else Severity.WARNING
        at = first + limit
        at_seg = [s.id for s in data.transcript.segments if s.end > at]
        out.append(
            Candidate(
                type=FeedbackType.OVER_TIME,
                severity=sev,
                start=at,
                end=last,
                values={"overTimeSec": _round1(over), "timeLimitMin": _round1(limit / 60)},
                segment_ids=tuple(at_seg[:1]),
            )
        )
    elif sp.is_under_time:
        ratio = sp.time_used_ratio or 0.0
        sev = Severity.WARNING if ratio < UNDER_TIME_WARNING_RATIO else Severity.INFO
        out.append(
            Candidate(
                type=FeedbackType.UNDER_TIME,
                severity=sev,
                start=first,
                end=last,
                values={
                    "underTimeSec": _round1(sp.under_time_sec),
                    "timeLimitMin": _round1(limit / 60),
                },
            )
        )
    return out


def _nonverbal_candidates(data: FeedbackInput) -> list[Candidate]:
    nv = data.nonverbal
    if nv is None:
        return []
    out: list[Candidate] = []
    for r in nv.gaze_away_ranges:
        d = r.end - r.start
        out.append(
            Candidate(
                FeedbackType.GAZE_AWAY, _gaze_severity(d), r.start, r.end, {"awaySec": _round1(d)}
            )
        )
    # 구간별 자세 흔들림 값은 없으므로 수치 없이 WARNING
    for r in nv.posture_unstable_ranges:
        out.append(Candidate(FeedbackType.POSTURE_UNSTABLE, Severity.WARNING, r.start, r.end))
    for r in nv.gesture_none_ranges:
        d = r.end - r.start
        out.append(
            Candidate(
                FeedbackType.GESTURE_NONE,
                _gesture_none_severity(d),
                r.start,
                r.end,
                {"noneSec": _round1(d)},
            )
        )
    for r in nv.gesture_excess_ranges:
        n = sum(1 for t in nv.gesture_times if r.start <= t < r.end)
        minutes = (r.end - r.start) / 60
        values: dict[str, int | float | str] = {"gestureCount": n}
        if minutes > 0:
            values["gesturePerMin"] = _round1(n / minutes)
        out.append(Candidate(FeedbackType.GESTURE_EXCESS, Severity.INFO, r.start, r.end, values))
    return out


_LANGUAGE_GUIDANCE = (
    "비난하지 말고 '발표 자리에서는 이 표현 대신 ~'처럼 대체 표현을 제안하고, "
    "욕설·비속어 단어 자체를 제목·설명·제안에 반복하지 마세요."
)


def _profanity_candidates(data: FeedbackInput) -> list[Candidate]:
    prof = (
        data.speech.profanity
        if data.speech.profanity.hits
        else detect_profanity(data.transcript)
    )
    if not prof.hits:
        return []
    seg_by_id = {s.id: s for s in data.transcript.segments}
    seen_segs: list[str] = list(dict.fromkeys(h.segment_id for h in prof.hits))
    out: list[Candidate] = []
    for sid in seen_segs:
        seg = seg_by_id.get(sid)
        if seg is None:
            continue
        out.append(
            Candidate(
                type=FeedbackType.CLARITY,
                severity=Severity.CRITICAL,
                start=seg.start,
                end=seg.end,
                segment_ids=(seg.id,),
                context=f"발표 자리에 맞지 않는 비속어·욕설 표현이 있어요. {_LANGUAGE_GUIDANCE}",
                priority=2,
            )
        )
    return out


def _content_candidates(data: FeedbackInput) -> list[Candidate]:
    if data.content is None:
        return []
    out: list[Candidate] = []
    for issue in data.content.issues:
        if issue.type is IssueType.LANGUAGE:
            ftype = FeedbackType.CLARITY
            sev = (
                Severity.CRITICAL
                if issue.severity is LanguageSeverity.SERIOUS
                else Severity.WARNING
            )
            prio = 2 if issue.severity is LanguageSeverity.SERIOUS else 1
            ctx = f"{issue.reason} ({_LANGUAGE_GUIDANCE})"
        elif issue.type is IssueType.STRUCTURE:
            ftype = FeedbackType.STRUCTURE
            sev = Severity.WARNING
            prio = 0
            ctx = issue.reason
        else:
            ftype = FeedbackType.CLARITY
            sev = Severity.INFO
            prio = 0
            ctx = issue.reason
        out.append(
            Candidate(
                type=ftype,
                severity=sev,
                start=issue.start,
                end=issue.end,
                segment_ids=tuple(issue.segment_ids),
                context=ctx,
                priority=prio,
            )
        )
    return out


def build_candidates(data: FeedbackInput) -> list[Candidate]:
    """후보를 만들고 정리(영상 길이로 자르기, 병합, 상한, 시간순, ID)한다."""
    raw = (
        _pace_candidates(data)
        + _filler_candidates(data)
        + _silence_candidates(data)
        + _time_candidates(data)
        + _nonverbal_candidates(data)
        + _profanity_candidates(data)
        + _content_candidates(data)
    )
    order = {s.id: i for i, s in enumerate(data.transcript.segments)}
    clipped = [c for c in (_clip(c, data.duration_sec) for c in raw) if c is not None]
    merged = _merge(clipped, order)
    ranked = sorted(
        merged,
        key=lambda c: (-_SEVERITY_RANK[c.severity], -c.priority, -(c.end - c.start), c.start),
    )
    chosen: list[Candidate] = []
    per_type: dict[FeedbackType, int] = {}
    for c in ranked:
        if len(chosen) >= MAX_FEEDBACKS:
            break
        if per_type.get(c.type, 0) >= MAX_PER_TYPE:
            continue
        per_type[c.type] = per_type.get(c.type, 0) + 1
        chosen.append(c)
    chosen.sort(key=lambda c: (c.start, c.end))
    return [replace(c, id=f"c-{i:02d}") for i, c in enumerate(chosen, 1)]


# 영상 길이로 잘렸을 때 다시 맞출 길이 값(구간 하나 = 그 길이인 경우)
_LENGTH_KEYS = {"awaySec", "noneSec", "silenceSec"}


def _clip(c: Candidate, duration: float) -> Candidate | None:
    limit = round(duration, 2)  # 리포트 durationSec과 같은 값(시간은 모두 소수 둘째 자리)
    start = max(0.0, round(c.start, 2))
    end = min(limit, round(c.end, 2))
    if start >= limit or end <= start:
        return None
    if (start, end) == (c.start, c.end):
        return c
    values = dict(c.values)
    for k in _LENGTH_KEYS & values.keys():
        values[k] = min(float(values[k]), _round1(end - start))
    severity = _severity_from_values(c.type, values) or c.severity
    return replace(c, start=start, end=end, values=values, severity=severity)


# 병합할 때 값 합치는 방법. 시선·손동작 없음 시간은 실제로 그랬던 시간의 합(사이 정상 구간은 빼고),
# 침묵은 가장 긴 것, 말 속도는 가장 극단적인 값, 나머지 수는 큰 값, 글자 값은 앞 것.
_SUM_KEYS = {"fillerCount", "awaySec", "noneSec", "gestureCount"}


def _merge_values(ftype: FeedbackType, a: dict, b: dict, start: float, end: float) -> dict:
    out = dict(a)
    for k, v in b.items():
        if k not in out:
            out[k] = v
        elif k in _SUM_KEYS:
            out[k] = _round1(out[k] + v) if isinstance(v, float) else out[k] + v
        elif k == "pace":
            out[k] = max(out[k], v) if ftype is FeedbackType.PACE_FAST else min(out[k], v)
        elif isinstance(v, int | float) and isinstance(out[k], int | float):
            out[k] = max(out[k], v)
    if ftype is FeedbackType.GESTURE_EXCESS and "gestureCount" in out and end > start:
        out["gesturePerMin"] = _round1(out["gestureCount"] / ((end - start) / 60))
    return out


def _severity_from_values(ftype: FeedbackType, values: dict) -> Severity | None:
    """값으로 정해지는 심각도(합치거나 자른 뒤 다시 계산). 값으로 못 정하면 None."""
    if ftype is FeedbackType.GAZE_AWAY and "awaySec" in values:
        return _gaze_severity(values["awaySec"])
    if ftype is FeedbackType.GESTURE_NONE and "noneSec" in values:
        return _gesture_none_severity(values["noneSec"])
    if ftype is FeedbackType.LONG_SILENCE and "silenceSec" in values:
        return _silence_severity(values["silenceSec"])
    if ftype is FeedbackType.FILLER and "fillerCount" in values and "fillerPerMin" not in values:
        return _filler_cluster_severity(values["fillerCount"])
    return None


def _max_severity(*sevs: Severity | None) -> Severity:
    return max((s for s in sevs if s is not None), key=_SEVERITY_RANK.__getitem__)


def _merge(cands: list[Candidate], order: dict[str, int]) -> list[Candidate]:
    by_group: dict[tuple[FeedbackType, bool], list[Candidate]] = {}
    for c in cands:
        by_group.setdefault((c.type, c.priority > 0), []).append(c)
    out: list[Candidate] = []
    for (ftype, _), items in by_group.items():
        items.sort(key=lambda c: c.start)
        cur = items[0]
        for nxt in items[1:]:
            if nxt.start - cur.end <= MERGE_GAP_SEC:
                start, end = cur.start, max(cur.end, nxt.end)
                values = _merge_values(ftype, cur.values, nxt.values, start, end)
                ids = sorted(
                    dict.fromkeys(cur.segment_ids + nxt.segment_ids),
                    key=lambda i: order.get(i, len(order)),
                )
                contexts = [x for x in (cur.context, nxt.context) if x]
                cur = Candidate(
                    type=ftype,
                    severity=_max_severity(
                        cur.severity, nxt.severity, _severity_from_values(ftype, values)
                    ),
                    start=start,
                    end=end,
                    values=values,
                    segment_ids=tuple(ids),
                    context=" / ".join(dict.fromkeys(contexts)) or None,
                    priority=max(cur.priority, nxt.priority),
                )
            else:
                out.append(cur)
                cur = nxt
        out.append(cur)
    return out


# ---------------------------------------------------------------- evidence


def _first_run(ids: Sequence[str], order: dict[str, int]) -> list[str]:
    """대본 순서로 정렬한 뒤 처음 이어진 문장 묶음(근거가 대본에 그대로 있도록)."""
    known = sorted((i for i in dict.fromkeys(ids) if i in order), key=order.__getitem__)
    run: list[str] = []
    for i in known:
        if run and order[i] != order[run[-1]] + 1:
            break
        run.append(i)
    return run


def _evidence(c: Candidate, transcript: Transcript) -> str | None:
    if c.category is ScoreCategory.NONVERBAL or c.type is FeedbackType.UNDER_TIME:
        return None
    segs = {s.id: s for s in transcript.segments}
    order = {s.id: i for i, s in enumerate(transcript.segments)}
    if c.type in (FeedbackType.STRUCTURE, FeedbackType.CLARITY):
        run = _first_run(c.segment_ids, order)
        return " ".join(segs[i].text for i in run) or None
    if c.type in (FeedbackType.PACE_FAST, FeedbackType.PACE_SLOW):
        found = [segs[i] for i in c.segment_ids if i in segs]
        if not found:
            return None
        best = max(found, key=lambda s: min(s.end, c.end) - max(s.start, c.start))
        return best.text
    run = _first_run(c.segment_ids, order)
    return segs[run[0]].text if run else None


def verify_evidence(evidence: str | None, transcript: Transcript) -> bool:
    """evidence 문장이 실제 대본에 있는지(공백 정규화 후 부분 일치) 확인한다."""
    if evidence is None:
        return True
    norm_ev = " ".join(evidence.split())
    norm_full = " ".join(s.text for s in transcript.segments)
    norm_full = " ".join(norm_full.split())
    return bool(norm_ev) and norm_ev in norm_full


# ---------------------------------------------------------------- 전체 자리표시자


def global_values(data: FeedbackInput) -> dict[str, int | float]:
    m = data.speech.metrics
    out: dict[str, int | float] = {
        "syllablesPerMin": m.syllables_per_min,
        "totalFillerCount": m.filler_count,
        "silenceCount": m.silence_count,
    }
    if data.nonverbal is not None and data.nonverbal.metrics.gaze_front_ratio is not None:
        out["gazeFrontPercent"] = _round_int(data.nonverbal.metrics.gaze_front_ratio * 100)
    return out


def _fmt(v: int | float | str) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".")
    return str(v)


# ---------------------------------------------------------------- LLM 스키마·입력


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _Item(_Strict):
    candidateId: str  # noqa: N815 - LLM 스키마 필드명
    title: str
    description: str
    suggestion: str


class _Overall(_Strict):
    summary: str
    strengths: list[str]
    improvements: list[str]


class _Improved(_Strict):
    segmentId: str  # noqa: N815
    improved: str
    reason: str


class _LlmFeedback(_Strict):
    feedbacks: list[_Item]
    overall: _Overall
    improvedScripts: list[_Improved]  # noqa: N815


def _obj(props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}

RESPONSE_SCHEMA: dict[str, Any] = _obj(
    {
        "feedbacks": {
            "type": "array",
            "items": _obj(
                {"candidateId": _STR, "title": _STR, "description": _STR, "suggestion": _STR}
            ),
        },
        "overall": _obj({"summary": _STR, "strengths": _STR_LIST, "improvements": _STR_LIST}),
        "improvedScripts": {
            "type": "array",
            "items": _obj({"segmentId": _STR, "improved": _STR, "reason": _STR}),
        },
    }
)


@lru_cache
def load_system_prompt(version: str = PROMPT_VERSION) -> str:
    return (PROMPTS_DIR / f"feedback_system_{version}.md").read_text(encoding="utf-8").strip()


def _position(c: Candidate, duration: float) -> str:
    if c.type in (FeedbackType.OVER_TIME, FeedbackType.UNDER_TIME):
        return "발표 전체"
    mid = (c.start + c.end) / 2 / duration if duration > 0 else 0
    return "앞부분" if mid < 1 / 3 else "중간" if mid < 2 / 3 else "뒷부분"


def _placeholder_lines(values: dict[str, Any]) -> str:
    return ", ".join(
        f"{{{k}}}={_fmt(v)}({_PLACEHOLDER_MEANING.get(k, k)})" for k, v in values.items()
    )


def build_user_input(data: FeedbackInput, cands: Sequence[Candidate]) -> tuple[str, set[str]]:
    transcript_text, _, sent_ids = build_transcript_block(data.transcript)
    gv = global_values(data)
    lines = [f"[발표 목적] {data.purpose.value}"]
    if data.content is not None and data.content.key_message:
        lines.append(f"[핵심 메시지] {data.content.key_message}")
    lines.append("[전체 자리표시자] " + (_placeholder_lines(gv) or "없음"))
    lines.append("[피드백 후보]")
    for c in cands:
        parts = [
            f"- {c.id}",
            f"종류: {c.type.value}({c.category.value})",
            f"심각도: {c.severity.value}",
            f"위치: {_position(c, data.duration_sec)}",
            f"상황: {_SITUATION.get(c.type, '')}",
        ]
        if c.context:
            parts.append(f"분석 메모: {c.context}")
        parts.append("자리표시자: " + (_placeholder_lines(c.values) or "없음(숫자 쓰지 말 것)"))
        ev = _evidence(c, data.transcript)
        if ev:
            parts.append(f'근거 문장: "{ev}"')
        lines.append(" | ".join(parts))
    lines.append("[대본]\n" + transcript_text)
    return "\n".join(lines), sent_ids


# ---------------------------------------------------------------- 검증


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER.findall(text)]


def _fill(text: str, values: dict[str, Any], quoted: str = "") -> str | None:
    """자리표시자를 값으로 바꾼다. 아래면 None.

    - 자리표시자 밖 숫자(quoted에 그대로 있는 숫자는 인용으로 허용)
    - 측정값을 한글 수사로 쓴 것, 모르는 자리표시자, 채운 뒤 남은 중괄호
    """
    text = text.strip()
    if not text:
        return None
    bare = _PLACEHOLDER.sub("", text)
    allowed = set(_numbers(quoted))
    if any(n not in allowed for n in _numbers(bare)):
        return None
    if _SINO_MEASURE.search(bare) or _NATIVE_COUNT.search(bare):
        return None
    if any(n not in values for n in _PLACEHOLDER.findall(text)):
        return None
    filled = _PLACEHOLDER.sub(lambda m: _fmt(values[m.group(1)]), text)
    if "{" in filled or "}" in filled:
        return None
    return filled


@dataclass
class _Outcome:
    feedbacks: list[Feedback]
    overall: OverallText | None
    improved: list[ImprovedScript]
    violations: int  # 재시도 사유가 된 문제 수
    dropped: int
    problems: set[str]  # 재시도할 때 LLM에 알려 줄 규칙 이름
    raw: dict[str, Any] | None = None  # 이 응답의 LLM 원본

    def score(self) -> tuple[int, int, int]:
        overall_ok = int(
            self.overall is not None
            and bool(self.overall.strengths)
            and bool(self.overall.improvements)
        )
        return (len(self.feedbacks), overall_ok, -self.violations)


_RULE_TEXT = {
    "missing": "모든 후보 ID에 대해 빠짐없이 하나씩 쓰세요.",
    "number": "숫자는 자리표시자로만 쓰고, 측정값을 한글 수사로도 쓰지 마세요.",
    "placeholder": "그 후보에 주어진 자리표시자만 쓰세요.",
    "script": "improvedScripts는 원문에 있던 숫자만 쓰고, reason에는 숫자를 쓰지 마세요.",
    "profanity": (
        "욕설·비속어 단어를 제목·설명·제안에 그대로 반복하지 말고, "
        "'발표 자리에서는 이 표현 대신 ~'처럼 대체 표현을 제안하세요."
    ),
}


def _fill_noting(text: str, values: dict[str, Any], quoted: str, problems: set[str]) -> str | None:
    filled = _fill(text, values, quoted)
    if filled is None:
        unknown = any(n not in values for n in _PLACEHOLDER.findall(text))
        problems.add("placeholder" if unknown else "number")
        return None
    if contains_profanity(filled):
        problems.add("profanity")
        return None
    return filled


def _check(
    raw: _LlmFeedback, data: FeedbackInput, cands: Sequence[Candidate], sent_ids: set[str]
) -> _Outcome:
    by_id = {c.id: c for c in cands}
    gv = global_values(data)
    violations = dropped = 0
    problems: set[str] = set()

    written: dict[str, Feedback] = {}
    for item in raw.feedbacks:
        c = by_id.get(item.candidateId)
        if c is None or c.id in written:
            dropped += 1  # 없는 후보(LLM이 만든 것)나 중복은 버림
            continue
        values = {**gv, **c.values}
        evidence = _evidence(c, data.transcript)
        if not verify_evidence(evidence, data.transcript):
            evidence = None
        title, desc, sugg = (
            _fill_noting(t, values, evidence or "", problems)
            for t in (item.title, item.description, item.suggestion)
        )
        if title is None or desc is None or sugg is None:
            dropped += 1
            violations += 1
            continue
        written[c.id] = Feedback(
            id="",
            category=c.category,
            type=c.type,
            severity=c.severity,
            start=c.start,
            end=c.end,
            title=title,
            description=desc,
            evidence=evidence,
            suggestion=sugg,
        )
    missing = [c.id for c in cands if c.id not in written]
    if missing:
        violations += len(missing)
        problems.add("missing")
    feedbacks = [
        f.model_copy(update={"id": f"f-{i:02d}"})
        for i, f in enumerate((written[c.id] for c in cands if c.id in written), 1)
    ]

    summary = _fill_noting(raw.overall.summary, gv, "", problems)
    if summary is None:
        violations += 1
        dropped += 1
    strengths: list[str] = []
    improvements: list[str] = []
    for src, dst in ((raw.overall.strengths, strengths), (raw.overall.improvements, improvements)):
        for t in src:
            filled = _fill_noting(t, gv, "", problems)
            if filled is None:
                violations += 1
                dropped += 1
            else:
                dst.append(filled)
    overall = (
        OverallText(
            summary=summary,
            strengths=strengths[:MAX_OVERALL_ITEMS],
            improvements=improvements[:MAX_OVERALL_ITEMS],
        )
        if summary is not None
        else None
    )

    seg_by_id = {s.id: s for s in data.transcript.segments}
    improved: list[ImprovedScript] = []
    seen: set[str] = set()
    for s in raw.improvedScripts:
        seg = seg_by_id.get(s.segmentId)
        text = " ".join(s.improved.split())
        reason = s.reason.strip()
        if seg is None or s.segmentId not in sent_ids or s.segmentId in seen:
            dropped += 1  # 없는 문장 ID: 그 항목만 버림
            continue
        allowed = set(_numbers(seg.text))
        has_prof = contains_profanity(text) or contains_profanity(reason)
        if (
            not text
            or text == seg.text.strip()
            or any(n not in allowed for n in _numbers(text))
            or "{" in text
            or "}" in text
            or not reason
            or _DIGIT.search(reason)
            or "{" in reason
            or has_prof
        ):
            dropped += 1
            violations += 1
            problems.add("profanity" if has_prof else "script")
            continue
        seen.add(s.segmentId)
        improved.append(
            ImprovedScript(
                segment_id=seg.id,
                start=seg.start,
                end=seg.end,
                original=seg.text,
                improved=text,
                reason=reason,
            )
        )
    improved = improved[:MAX_IMPROVED_SCRIPTS]
    return _Outcome(feedbacks, overall, improved, violations, dropped, problems)


# ---------------------------------------------------------------- 진입점


class _Invalid(Exception):
    pass


class _TokenLimit(Exception):
    pass


def _parse(response: Any) -> _LlmFeedback:
    if llm.is_refusal(response):
        raise FeedbackServiceError("llm refused")
    reason = llm.incomplete_reason(response)
    if reason == "max_output_tokens":
        raise _TokenLimit
    if reason is not None:
        raise FeedbackServiceError(f"llm response incomplete: {reason}")
    status = getattr(response, "status", None)
    if status not in (None, "completed"):
        raise FeedbackServiceError(f"llm response status {status}")
    text = getattr(response, "output_text", None)
    if not isinstance(text, str) or not text.strip():
        raise _Invalid("empty output")
    try:
        return _LlmFeedback.model_validate(json.loads(text))
    except (json.JSONDecodeError, ValidationError) as e:
        raise _Invalid(f"schema violation: {type(e).__name__}") from None


def _haeyo(o: _Outcome) -> _Outcome:
    """화면 문구 끝의 합쇼체를 해요체로 바꾸고, 못 바꾼 것만 경고한다(재시도 없음).

    improvedScripts.improved는 발표자가 말할 대본이라 발표 말투(합쇼체)가 자연스럽다.
    바꾸지도, 경고하지도 않는다(reason은 화면 문구라 바꾼다).
    """
    left = 0

    def fix(text: str) -> str:
        nonlocal left
        out, n = to_haeyo(text)
        left += n
        return out or text

    feedbacks = [
        f.model_copy(
            update={
                "title": fix(f.title),
                "description": fix(f.description),
                "suggestion": fix(f.suggestion),
            }
        )
        for f in o.feedbacks
    ]
    overall = (
        OverallText(
            summary=fix(o.overall.summary),
            strengths=[fix(t) for t in o.overall.strengths],
            improvements=[fix(t) for t in o.overall.improvements],
        )
        if o.overall
        else None
    )
    improved = [s.model_copy(update={"reason": fix(s.reason)}) for s in o.improved]
    if left:
        logger.warning("feedback: %d sentence(s) left not in 해요체", left)
    return replace(o, feedbacks=feedbacks, overall=overall, improved=improved)


def _retry_note(problems: set[str]) -> str:
    rules = [_RULE_TEXT[p] for p in _RULE_TEXT if p in problems]
    if not rules:
        return ""
    return "\n\n[다시 쓸 때 지킬 것] 앞 응답이 아래 규칙을 어겼어요.\n" + "\n".join(
        f"- {r}" for r in rules
    )


# ---------------------------------------------------------------- overall 기본 문구

CATEGORY_LABEL: dict[ScoreCategory, str] = {
    ScoreCategory.DELIVERY: "전달력",
    ScoreCategory.NONVERBAL: "비언어",
    ScoreCategory.CONTENT: "내용 구성",
}
FALLBACK_STRENGTH_SCORE = 80  # 이 점수 이상인 카테고리를 강점으로
FALLBACK_IMPROVEMENT_SCORE = 80  # 피드백이 없을 때 이 점수 미만인 카테고리를 개선점으로

# LLM 제목이 없을 때 쓰는 후보 종류별 기본 제목(해요체)
DEFAULT_TITLE: dict[FeedbackType, str] = {
    FeedbackType.PACE_FAST: "말이 빨라요",
    FeedbackType.PACE_SLOW: "말이 느려요",
    FeedbackType.FILLER: "추임새가 많아요",
    FeedbackType.LONG_SILENCE: "말이 길게 끊겨요",
    FeedbackType.OVER_TIME: "제한 시간을 넘었어요",
    FeedbackType.UNDER_TIME: "발표가 짧아요",
    FeedbackType.GAZE_AWAY: "시선이 화면 밖으로 벗어나요",
    FeedbackType.POSTURE_UNSTABLE: "몸이 많이 흔들려요",
    FeedbackType.GESTURE_NONE: "손동작이 거의 없어요",
    FeedbackType.GESTURE_EXCESS: "손동작이 너무 많아요",
    FeedbackType.STRUCTURE: "구성이 매끄럽지 않아요",
    FeedbackType.CLARITY: "표현이 모호해요",
}


def _has_batchim(word: str) -> bool:
    ch = word[-1]
    return "가" <= ch <= "힣" and (ord(ch) - ord("가")) % 28 != 0


def _subj(word: str) -> str:
    return word + ("이" if _has_batchim(word) else "가")


def _obj_josa(word: str) -> str:
    return word + ("을" if _has_batchim(word) else "를")


def fallback_overall(
    scores: dict[ScoreCategory, int] | None,
    feedbacks: Sequence[Feedback],
    candidates: Sequence[Candidate] = (),
) -> OverallText:
    """코드가 만드는 overall 기본 문구(해요체). 점수·피드백이 없어도 항상 채운다."""
    ranked = sorted(feedbacks, key=lambda f: (-_SEVERITY_RANK[f.severity], f.start))
    titles = list(dict.fromkeys(f.title for f in ranked))
    if not titles:
        by_sev = sorted(candidates, key=lambda c: (-_SEVERITY_RANK[c.severity], c.start))
        titles = list(
            dict.fromkeys(DEFAULT_TITLE[c.type] for c in by_sev if c.type in DEFAULT_TITLE)
        )

    if scores:
        ordered = sorted(scores.items(), key=lambda kv: (-kv[1], list(CATEGORY_LABEL).index(kv[0])))
        best_cat, best = ordered[0]
        worst_cat, worst = ordered[-1]
        best_label, worst_label = CATEGORY_LABEL[best_cat], CATEGORY_LABEL[worst_cat]
        if best == worst:
            summary = "세 영역이 고르게 나왔어요. 아래 피드백을 하나씩 연습해 보세요."
        else:
            summary = (
                f"{_subj(best_label)} 가장 좋았어요. "
                f"{_obj_josa(worst_label)} 보완하면 더 좋아질 거예요."
            )
        strengths = [
            f"{_subj(CATEGORY_LABEL[c])} 좋아요" for c, v in ordered if v >= FALLBACK_STRENGTH_SCORE
        ] or [f"{_subj(best_label)} 상대적으로 안정적이에요"]
        weak = [
            f"{_obj_josa(CATEGORY_LABEL[c])} 더 다듬어 보세요"
            for c, v in reversed(ordered)
            if v < FALLBACK_IMPROVEMENT_SCORE
        ]
    else:
        summary = "분석이 끝났어요. 아래 피드백을 하나씩 확인해 보세요."
        strengths = ["끝까지 발표를 마쳤어요"]
        weak = []
    improvements = (
        titles[:MAX_OVERALL_ITEMS] or weak[:MAX_OVERALL_ITEMS] or ["지금처럼 꾸준히 연습해 보세요"]
    )
    return OverallText(
        summary=summary,
        strengths=strengths[:MAX_OVERALL_ITEMS],
        improvements=improvements,
    )


def _complete_overall(got: OverallText | None, fallback: OverallText) -> tuple[OverallText, bool]:
    """LLM overall의 빈 부분을 기본 문구로 채운다. (결과, 기본 문구를 썼는지)"""
    if got is None:
        return fallback, True
    used = False
    strengths, improvements = got.strengths, got.improvements
    if not strengths:
        strengths, used = fallback.strengths, True
    if not improvements:
        improvements, used = fallback.improvements, True
    return OverallText(summary=got.summary, strengths=strengths, improvements=improvements), used


def generate_feedback(data: FeedbackInput, client: LlmClient, settings: Settings) -> FeedbackResult:
    cands = build_candidates(data)
    if not cands:
        return FeedbackResult(
            feedbacks=[],
            overall=fallback_overall(data.category_scores, []),
            improved_scripts=[],
            candidates=[],
            overall_fallback=True,
        )

    model = settings.openai_llm_model.strip()
    if not model:
        raise FeedbackServiceError("OPENAI_LLM_MODEL is not set")
    user_input, sent_ids = build_user_input(data, cands)
    kwargs = llm.structured_kwargs(
        model=model,
        name=f"feedback_{PROMPT_VERSION}",
        schema=RESPONSE_SCHEMA,
        instructions=load_system_prompt(),
        user_input=user_input,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    best: _Outcome | None = None
    attempts = total_in = total_out = 0
    last_problem = ""
    for attempt in range(1, MAX_RETRIES + 2):
        attempts = attempt
        error: str | None = None
        outcome: _Outcome | None = None
        try:
            response = llm.call(client, kwargs, "feedback")
            inp, out = llm.log_usage("feedback", model, attempt, response, settings)
            total_in += inp
            total_out += out
            parsed = _parse(response)
            outcome = _check(parsed, data, cands, sent_ids)
            outcome.raw = parsed.model_dump()
        except llm.LlmCallError as e:
            error = str(e)
        except FeedbackServiceError as e:
            error = str(e)
        except _TokenLimit:
            if kwargs["max_output_tokens"] >= MAX_OUTPUT_TOKENS_RETRY:
                error = "llm output hit token limit twice"
            else:
                logger.warning("feedback llm hit max_output_tokens (attempt %d)", attempt)
                kwargs = {**kwargs, "max_output_tokens": MAX_OUTPUT_TOKENS_RETRY}
                last_problem = "max_output_tokens"
                continue
        except _Invalid as e:
            last_problem = str(e)
            logger.warning("feedback llm response rejected (attempt %d): %s", attempt, e)
            continue
        if error is not None:
            if best is None:
                raise FeedbackServiceError(error)
            # 재시도 중 실패해도 앞서 받은 쓸 만한 응답을 버리지 않는다.
            logger.warning("feedback retry failed (attempt %d): %s; using best", attempt, error)
            break
        if outcome is None:
            continue
        if best is None or outcome.score() > best.score():
            best = outcome
        if outcome.violations == 0:
            break
        last_problem = f"{outcome.violations} violation(s)"
        logger.warning(
            "feedback llm response had %d violation(s) (attempt %d)", outcome.violations, attempt
        )
        kwargs = {**kwargs, "input": user_input + _retry_note(outcome.problems)}

    if best is None:
        raise FeedbackServiceError(f"no usable response after {attempts} attempts: {last_problem}")
    if best.dropped:
        logger.info("feedback: dropped %d invalid item(s)", best.dropped)
    best = _haeyo(best)
    overall, used_fallback = _complete_overall(
        best.overall, fallback_overall(data.category_scores, best.feedbacks, cands)
    )
    if used_fallback:
        logger.info("feedback: overall filled with code fallback text")
    return FeedbackResult(
        feedbacks=best.feedbacks,
        overall=overall,
        overall_fallback=used_fallback,
        improved_scripts=best.improved,
        candidates=list(cands),
        usage=LlmUsage(
            model=model,
            attempts=attempts,
            input_tokens=total_in,
            output_tokens=total_out,
            estimated_cost_usd=round(llm.estimate_cost(settings, total_in, total_out), 6),
        ),
        dropped_items=best.dropped,
        raw=best.raw,
    )
