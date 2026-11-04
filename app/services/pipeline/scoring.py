"""점수: 코드만으로 0~100점을 계산한다(LLM은 점수에 관여하지 않는다).

카테고리별 세부 지표 점수(0~100)를 가중 평균한다. 측정 불가(None) 지표는 빼고, 남은 지표의 가중치를
합이 1이 되도록 다시 나눈다. 종합 점수는 발표 목적별 카테고리 가중치로 같은 방식으로 낸다.
세부 지표가 하나도 없는 카테고리는 NEUTRAL_SCORE를 보여주되 종합 점수 계산에서는 뺀다.

- DELIVERY: 말 속도(분당 음절, 이상 구간이면 100), 추임새(분당 개수), 긴 침묵(분당 횟수),
  시간(제한 시간 대비 발표 시간, 제한이 없으면 측정 불가)
- NONVERBAL: 시선(정면 비율), 자세(안정성), 제스처(분당 손동작, 적정 구간이면 100)
- CONTENT: 구조(STRUCTURE 후보 수), 명확성(CLARITY 후보 수). 후보 하나마다 감점.
"""

import math
from dataclasses import dataclass

from app.models.enums import Purpose, ScoreCategory
from app.services.pipeline.content import ContentAnalysis, IssueType
from app.services.pipeline.nonverbal import NonverbalAnalysis
from app.services.pipeline.speech_metrics import SpeechAnalysis

# ---- 말 속도(분당 음절): IDEAL 구간 100점, ZERO에서 0점, 사이는 직선 ----
PACE_IDEAL_LOW = 300
PACE_IDEAL_HIGH = 380
PACE_ZERO_LOW = 180
PACE_ZERO_HIGH = 500

# ---- 추임새(분당 개수) ----
FILLER_FULL_PER_MIN = 1.0  # 이하 100점
FILLER_ZERO_PER_MIN = 10.0  # 이상 0점

# ---- 긴 침묵(2초 이상, 분당 횟수) ----
SILENCE_FULL_PER_MIN = 0.5
SILENCE_ZERO_PER_MIN = 3.0

# ---- 시간(발표 시간 / 제한 시간) ----
TIME_IDEAL_LOW = 0.85
TIME_IDEAL_HIGH = 1.0
TIME_ZERO_LOW = 0.4
TIME_ZERO_HIGH = 1.3

# ---- 시선(정면 비율) ----
GAZE_FULL = 0.85
GAZE_ZERO = 0.3

# ---- 제스처(분당 손동작) ----
GESTURE_IDEAL_LOW = 2.0
GESTURE_IDEAL_HIGH = 12.0
GESTURE_NONE_SCORE = 50  # 손동작이 전혀 없을 때(0 이하)
GESTURE_ZERO_HIGH = 40.0

# ---- 내용(후보 하나당 감점) ----
STRUCTURE_PENALTY = 15
CLARITY_PENALTY = 10

NEUTRAL_SCORE = 70  # 세부 지표가 하나도 없는 카테고리(종합 점수에서는 제외)

# 카테고리 안 세부 지표 가중치
DELIVERY_WEIGHTS = {"pace": 0.35, "filler": 0.35, "silence": 0.15, "time": 0.15}
NONVERBAL_WEIGHTS = {"gaze": 0.5, "posture": 0.25, "gesture": 0.25}
CONTENT_WEIGHTS = {"structure": 0.5, "clarity": 0.5}

# 발표 목적별 카테고리 가중치
PURPOSE_WEIGHTS: dict[Purpose, dict[ScoreCategory, float]] = {
    Purpose.INTERVIEW: {
        ScoreCategory.DELIVERY: 0.35,
        ScoreCategory.NONVERBAL: 0.30,
        ScoreCategory.CONTENT: 0.35,
    },
    Purpose.CLASS: {
        ScoreCategory.DELIVERY: 0.35,
        ScoreCategory.NONVERBAL: 0.25,
        ScoreCategory.CONTENT: 0.40,
    },
    Purpose.PITCH: {
        ScoreCategory.DELIVERY: 0.40,
        ScoreCategory.NONVERBAL: 0.25,
        ScoreCategory.CONTENT: 0.35,
    },
    Purpose.ETC: {
        ScoreCategory.DELIVERY: 1 / 3,
        ScoreCategory.NONVERBAL: 1 / 3,
        ScoreCategory.CONTENT: 1 / 3,
    },
}

# (하한 점수, 등급): 위에서부터 처음 만족하는 등급
GRADES: list[tuple[int, str]] = [
    (95, "A+"),
    (90, "A"),
    (80, "B+"),
    (70, "B"),
    (60, "C+"),
    (50, "C"),
    (0, "D"),
]


@dataclass(frozen=True)
class ScoreResult:
    overall: int
    grade: str
    categories: dict[ScoreCategory, int]
    # 세부 지표 점수(측정 불가는 None). 평가셋 비교·디버깅용
    details: dict[ScoreCategory, dict[str, float | None]]
    measured: dict[ScoreCategory, bool]  # 세부 지표가 하나라도 있었는지


def _round_int(x: float) -> int:
    return int(math.floor(x + 0.5))


def _clamp(x: float) -> float:
    if not math.isfinite(x):
        return math.nan  # weighted()가 측정 불가로 뺀다(min(100, nan)은 100이 되므로 먼저 거른다)
    return max(0.0, min(100.0, x))


def _band(
    x: float, zero_low: float, ideal_low: float, ideal_high: float, zero_high: float
) -> float:
    """[ideal_low, ideal_high]면 100, 바깥으로 zero_*까지 직선으로 0."""
    if ideal_low <= x <= ideal_high:
        return 100.0
    if x < ideal_low:
        return _clamp(100 * (x - zero_low) / (ideal_low - zero_low))
    return _clamp(100 * (zero_high - x) / (zero_high - ideal_high))


def _falling(x: float, full: float, zero: float) -> float:
    """x <= full이면 100, x >= zero면 0(값이 작을수록 좋은 지표)."""
    if x <= full:
        return 100.0
    return _clamp(100 * (zero - x) / (zero - full))


def _rising(x: float, zero: float, full: float) -> float:
    """x >= full이면 100, x <= zero면 0(값이 클수록 좋은 지표)."""
    if x >= full:
        return 100.0
    return _clamp(100 * (x - zero) / (full - zero))


def weighted(scores: dict[str, float | None], weights: dict[str, float]) -> float | None:
    """None·NaN은 빼고 남은 가중치를 다시 나눈 가중 평균. 남은 게 없으면 None."""
    used = {
        k: v
        for k, v in scores.items()
        if v is not None and math.isfinite(v) and weights.get(k, 0) > 0
    }
    total = sum(weights[k] for k in used)
    if total <= 0:
        return None
    return sum(weights[k] * v for k, v in used.items()) / total  # type: ignore[operator]


def delivery_details(speech: SpeechAnalysis) -> dict[str, float | None]:
    m = speech.metrics
    minutes = speech.speaking_sec / 60
    measured_speech = speech.speaking_sec > 0 and m.syllables_per_min > 0
    pace = (
        _band(m.syllables_per_min, PACE_ZERO_LOW, PACE_IDEAL_LOW, PACE_IDEAL_HIGH, PACE_ZERO_HIGH)
        if measured_speech
        else None
    )
    filler = (
        _falling(m.filler_count / minutes, FILLER_FULL_PER_MIN, FILLER_ZERO_PER_MIN)
        if minutes > 0
        else None
    )
    silence = (
        _falling(m.silence_count / minutes, SILENCE_FULL_PER_MIN, SILENCE_ZERO_PER_MIN)
        if minutes > 0
        else None
    )
    time = (
        _band(
            speech.time_used_ratio, TIME_ZERO_LOW, TIME_IDEAL_LOW, TIME_IDEAL_HIGH, TIME_ZERO_HIGH
        )
        if speech.time_used_ratio is not None
        else None
    )
    return {"pace": pace, "filler": filler, "silence": silence, "time": time}


def nonverbal_details(nv: NonverbalAnalysis | None) -> dict[str, float | None]:
    if nv is None:
        return {"gaze": None, "posture": None, "gesture": None}
    m = nv.metrics
    gaze = (
        _rising(m.gaze_front_ratio, GAZE_ZERO, GAZE_FULL)
        if m.gaze_front_ratio is not None
        else None
    )
    posture = _clamp(m.posture_stability * 100) if m.posture_stability is not None else None
    gesture: float | None = None
    if m.gesture_per_min is not None:
        g = m.gesture_per_min
        if g <= 0:
            gesture = float(GESTURE_NONE_SCORE)
        elif g < GESTURE_IDEAL_LOW:
            # 0 초과~이상 하한: NONE 점수에서 100까지
            gesture = GESTURE_NONE_SCORE + (100 - GESTURE_NONE_SCORE) * g / GESTURE_IDEAL_LOW
        else:
            gesture = _band(g, 0.0, GESTURE_IDEAL_LOW, GESTURE_IDEAL_HIGH, GESTURE_ZERO_HIGH)
    return {"gaze": gaze, "posture": posture, "gesture": gesture}


def content_details(content: ContentAnalysis | None) -> dict[str, float | None]:
    if content is None:
        return {"structure": None, "clarity": None}
    n_struct = sum(1 for i in content.issues if i.type is IssueType.STRUCTURE)
    n_clar = sum(1 for i in content.issues if i.type is IssueType.CLARITY)
    return {
        "structure": _clamp(100 - STRUCTURE_PENALTY * n_struct),
        "clarity": _clamp(100 - CLARITY_PENALTY * n_clar),
    }


def grade_for(score: int) -> str:
    for floor, grade in GRADES:
        if score >= floor:
            return grade
    return GRADES[-1][1]


def score_presentation(
    purpose: Purpose,
    speech: SpeechAnalysis,
    nonverbal: NonverbalAnalysis | None,
    content: ContentAnalysis | None,
) -> ScoreResult:
    raw_details = {
        ScoreCategory.DELIVERY: delivery_details(speech),
        ScoreCategory.NONVERBAL: nonverbal_details(nonverbal),
        ScoreCategory.CONTENT: content_details(content),
    }
    # 계산할 수 없는 값(NaN·무한대)은 측정 불가(None)로 본다.
    details = {
        c: {k: v if v is not None and math.isfinite(v) else None for k, v in d.items()}
        for c, d in raw_details.items()
    }
    weights = {
        ScoreCategory.DELIVERY: DELIVERY_WEIGHTS,
        ScoreCategory.NONVERBAL: NONVERBAL_WEIGHTS,
        ScoreCategory.CONTENT: CONTENT_WEIGHTS,
    }
    raw: dict[ScoreCategory, float | None] = {
        c: weighted(details[c], weights[c]) for c in ScoreCategory
    }
    measured = {c: raw[c] is not None for c in ScoreCategory}
    categories = {
        c: _round_int(raw[c]) if raw[c] is not None else NEUTRAL_SCORE  # type: ignore[arg-type]
        for c in ScoreCategory
    }
    overall_raw = weighted(
        {c.value: raw[c] for c in ScoreCategory},
        {c.value: w for c, w in PURPOSE_WEIGHTS[purpose].items()},
    )
    overall = _round_int(overall_raw) if overall_raw is not None else NEUTRAL_SCORE
    return ScoreResult(
        overall=overall,
        grade=grade_for(overall),
        categories=categories,
        details=details,
        measured=measured,
    )
