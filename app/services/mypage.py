"""마이페이지 집계·강점/약점 판정(코드 계산, AI 호출 없음).

원칙(docs/mypage-spec.md):
- 내 발표의 COMPLETED 회차만 집계한다(FAILED·WAITING_UPLOAD·삭제된 발표·다른 사용자 제외).
- 날짜·주간·연속일(streak) 기준은 한국 시간(Asia/Seoul, UTC+9). 주 시작은 월요일.
- 반올림은 scoring.py와 같은 사사오입(0.5 올림, half-up).
- M2는 최소 REQUIRED_ANALYSIS_COUNT(3)회 이상일 때만 READY이고, 최근 MAX_RECENT_ANALYSES(10)회로
  강점(score >= 80, sampleCount >= 2)·약점(score < 70, sampleCount >= 2)을 판정한다.
"""

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from app.models.enums import Purpose, ScoreCategory
from app.schemas.common import KST
from app.schemas.mypage import (
    FrequentFeedbackItem,
    InsightItem,
    InsightStatus,
    MetricKey,
    MyInsightsResponse,
    MyStatsResponse,
    Period,
    PurposeBreakdownItem,
    ScoreTimelineItem,
    ScoreTimelineResponse,
    Trend,
    WeeklyActivityItem,
)
from app.schemas.report import FeedbackType, FillerStat
from app.services.pipeline import scoring
from app.services.pipeline.scoring import GRADES, grade_for

# ---- 판정 상수 (docs/mypage-spec.md) ----
REQUIRED_ANALYSIS_COUNT = 3
MAX_RECENT_ANALYSES = 10
MIN_METRIC_SAMPLES = 2
MIN_TREND_SAMPLES = 4
STRENGTH_MIN_SCORE = 80
WEAKNESS_MAX_SCORE = 70  # score < 70이면 약점
TREND_MIN_DIFF = 5
RECENT_CHANGE_WINDOW = 5  # 최근 5회 vs 그 직전 5회
MIN_RECENT_CHANGE_COUNT = 6  # 완료 회차 6회 미만이면 recentChange는 null
MAX_STRENGTHS = 3
MAX_WEAKNESSES = 3
MAX_TOP_FILLERS = 5
MAX_FREQUENT_FEEDBACKS = 5
WEEKLY_ACTIVITY_WEEKS = 12
DEFAULT_TIMELINE_LIMIT = 50
MAX_TIMELINE_LIMIT = 200

GRADE_KEYS: tuple[str, ...] = tuple(g for _, g in GRADES)
CATEGORY_KEYS: tuple[ScoreCategory, ...] = (
    ScoreCategory.DELIVERY,
    ScoreCategory.NONVERBAL,
    ScoreCategory.CONTENT,
)
PURPOSE_ORDER: tuple[Purpose, ...] = (
    Purpose.INTERVIEW,
    Purpose.CLASS,
    Purpose.PITCH,
    Purpose.ETC,
)

# 명세 4장 M2 metric 표 순서·카테고리·화면 표시 이름
METRIC_SPECS: tuple[tuple[MetricKey, ScoreCategory, str], ...] = (
    (MetricKey.PACE, ScoreCategory.DELIVERY, "말 속도"),
    (MetricKey.FILLER, ScoreCategory.DELIVERY, "추임새"),
    (MetricKey.SILENCE, ScoreCategory.DELIVERY, "긴 침묵"),
    (MetricKey.TIME, ScoreCategory.DELIVERY, "시간 관리"),
    (MetricKey.GAZE, ScoreCategory.NONVERBAL, "시선"),
    (MetricKey.POSTURE, ScoreCategory.NONVERBAL, "자세"),
    (MetricKey.GESTURE, ScoreCategory.NONVERBAL, "제스처"),
    (MetricKey.STRUCTURE, ScoreCategory.CONTENT, "구조"),
    (MetricKey.CLARITY, ScoreCategory.CONTENT, "명확성"),
    (MetricKey.LANGUAGE, ScoreCategory.CONTENT, "언어 표현"),
)

FEEDBACK_TYPE_LABELS: dict[FeedbackType, str] = {
    FeedbackType.PACE_FAST: "말 속도 빠름",
    FeedbackType.PACE_SLOW: "말 속도 느림",
    FeedbackType.FILLER: "추임새",
    FeedbackType.LONG_SILENCE: "긴 침묵",
    FeedbackType.OVER_TIME: "시간 초과",
    FeedbackType.UNDER_TIME: "시간 부족",
    FeedbackType.GAZE_AWAY: "시선 이탈",
    FeedbackType.POSTURE_UNSTABLE: "자세 흔들림",
    FeedbackType.GESTURE_NONE: "제스처 부족",
    FeedbackType.GESTURE_EXCESS: "제스처 과다",
    FeedbackType.STRUCTURE: "구조 미흡",
    FeedbackType.CLARITY: "명확성 부족",
    FeedbackType.SLIDE_MISSED: "슬라이드 누락",
}

FOCUS_TIPS: dict[MetricKey, str] = {
    MetricKey.PACE: (
        "다음 연습에서는 첫 문장 뒤에 잠깐 쉬며 말 속도를 일정하게 유지하는 데 집중해 보세요."
    ),
    MetricKey.FILLER: "다음 연습에서는 추임새를 분당 1회 이하로 줄이는 데 집중해 보세요.",
    MetricKey.SILENCE: (
        "다음 연습에서는 문장과 문장 사이 연결어를 미리 정해 긴 침묵을 줄이는 데 집중해 보세요."
    ),
    MetricKey.TIME: (
        "다음 연습에서는 파트별 목표 시간을 미리 나누어 제한 시간에 맞추는 데 집중해 보세요."
    ),
    MetricKey.GAZE: "다음 연습에서는 문장을 끝맺을 때마다 카메라 렌즈를 바라보는 데 집중해 보세요.",
    MetricKey.POSTURE: (
        "다음 연습에서는 양발 위치를 고정하고 상체 흔들림을 줄이는 데 집중해 보세요."
    ),
    MetricKey.GESTURE: (
        "다음 연습에서는 핵심 키워드를 강조할 때만 자연스럽게 손동작을 쓰는 데 집중해 보세요."
    ),
    MetricKey.STRUCTURE: (
        "다음 연습에서는 도입·본론·결론 순서를 먼저 한 줄씩 안내하는 데 집중해 보세요."
    ),
    MetricKey.CLARITY: (
        "다음 연습에서는 주장 뒤에 구체적인 근거나 예시를 한 문장씩 덧붙이는 데 집중해 보세요."
    ),
    MetricKey.LANGUAGE: (
        "다음 연습에서는 발표 자리에 맞는 차분하고 공식적인 표현으로 말하는 데 집중해 보세요."
    ),
}


@dataclass(frozen=True)
class CompletedRecord:
    """DB에서 읽어온 완료 회차 1건(경량 요약, 큰 report_json 불필요)."""

    analysis_id: str
    presentation_id: str
    presentation_title: str
    purpose: Purpose
    attempt_no: int
    created_at: datetime
    completed_at: datetime
    duration_sec: float
    overall_score: int
    grade: str
    scores: dict[str, int]
    metric_details: dict[str, dict[str, float | None] | None] = field(default_factory=dict)
    filler_counts: list[dict[str, Any]] = field(default_factory=list)
    feedback_counts: dict[str, int] = field(default_factory=dict)


def round_int(x: float) -> int:
    """scoring.py와 같은 사사오입 반올림(0.5 올림, 음수도 대칭)."""
    if x >= 0:
        return int(math.floor(x + 0.5))
    return -int(math.floor(-x + 0.5))


def round1(x: float) -> float:
    return math.floor(round(x * 10, 6) + 0.5) / 10


def as_kst(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(KST)


def period_start(period: Period, now: datetime) -> datetime | None:
    """period의 KST 시작 시각(포함). 오늘 포함 최근 30일(오늘-29일 00:00 KST) 또는 90일."""
    if period is Period.ALL:
        return None
    days = 30 if period is Period.DAYS_30 else 90
    today_kst = as_kst(now).date()
    start_date = today_kst - timedelta(days=days - 1)
    return datetime.combine(start_date, time.min, tzinfo=KST)


def filter_records(
    records: Sequence[CompletedRecord],
    *,
    period: Period,
    purpose: Purpose | None,
    now: datetime,
) -> list[CompletedRecord]:
    start = period_start(period, now)
    out: list[CompletedRecord] = []
    for r in records:
        if purpose is not None and r.purpose is not purpose:
            continue
        comp = as_kst(r.completed_at)
        if start is not None and comp < start:
            continue
        out.append(r)
    out.sort(key=lambda r: (as_kst(r.completed_at), as_kst(r.created_at), r.analysis_id))
    return out


# ---------------------------------------------------------------- M1: 요약 통계


def compute_streak_days(practice_dates: set[date], today: date) -> int:
    """오늘 또는 어제까지 하루도 빠짐없이 연습한 날 수. 어제도 오늘도 안 했으면 0."""
    if today in practice_dates:
        cur = today
    elif (today - timedelta(days=1)) in practice_dates:
        cur = today - timedelta(days=1)
    else:
        return 0
    streak = 0
    while cur in practice_dates:
        streak += 1
        cur -= timedelta(days=1)
    return streak


def compute_recent_change(records: Sequence[CompletedRecord]) -> int | None:
    """최근 5회 평균 - 그 직전 5회 평균(정수). 완료 회차가 6회 미만이면 None."""
    if len(records) < MIN_RECENT_CHANGE_COUNT:
        return None
    recent = records[-RECENT_CHANGE_WINDOW:]
    prev = records[-2 * RECENT_CHANGE_WINDOW : -RECENT_CHANGE_WINDOW]
    recent_avg = sum(r.overall_score for r in recent) / len(recent)
    prev_avg = sum(r.overall_score for r in prev) / len(prev)
    return round_int(recent_avg - prev_avg)


def compute_weekly_activity(
    records: Sequence[CompletedRecord], now: datetime
) -> list[WeeklyActivityItem]:
    """최근 12주(월요일 시작), 오래된 순, 0인 주 포함."""
    today_kst = as_kst(now).date()
    current_monday = today_kst - timedelta(days=today_kst.weekday())
    week_starts = [
        current_monday - timedelta(weeks=w) for w in range(WEEKLY_ACTIVITY_WEEKS - 1, -1, -1)
    ]
    buckets: dict[date, list[CompletedRecord]] = {ws: [] for ws in week_starts}
    for r in records:
        d = as_kst(r.completed_at).date()
        monday = d - timedelta(days=d.weekday())
        if monday in buckets:
            buckets[monday].append(r)

    items: list[WeeklyActivityItem] = []
    for ws in week_starts:
        in_week = buckets[ws]
        count = len(in_week)
        sec = round1(sum(r.duration_sec for r in in_week)) if in_week else 0.0
        avg = (
            round_int(sum(r.overall_score for r in in_week) / count)
            if count > 0
            else None
        )
        items.append(
            WeeklyActivityItem(
                week_start=ws.isoformat(),
                analysis_count=count,
                practice_sec=sec,
                average_score=avg,
            )
        )
    return items


def build_stats(
    records: Sequence[CompletedRecord],
    *,
    period: Period,
    purpose: Purpose | None,
    now: datetime,
) -> MyStatsResponse:
    now_kst = as_kst(now)
    from_dt = period_start(period, now_kst)
    filtered = filter_records(records, period=period, purpose=purpose, now=now_kst)

    n = len(filtered)
    presentation_count = len({r.presentation_id for r in filtered})
    total_sec = round1(sum(r.duration_sec for r in filtered)) if filtered else 0.0
    avg_score = round_int(sum(r.overall_score for r in filtered) / n) if n > 0 else None

    best_score: int | None = None
    best_id: str | None = None
    if filtered:
        # 최고 점수 동점 시 가장 최근 회차
        best = max(filtered, key=lambda r: (r.overall_score, as_kst(r.completed_at)))
        best_score = best.overall_score
        best_id = best.analysis_id

    recent_change = compute_recent_change(filtered)
    # streakDays(연속 연습일)와 weeklyActivity(최근 12주 활동)는 period(30d)로 과거 날짜가
    # 인위적으로 잘리지 않도록 purpose 조건만 적용한 전체 완료 기록을 기준으로 계산한다.
    purpose_records = filter_records(records, period=Period.ALL, purpose=purpose, now=now_kst)
    practice_dates = {as_kst(r.completed_at).date() for r in purpose_records}
    streak = compute_streak_days(practice_dates, now_kst.date())

    cat_avgs: dict[ScoreCategory, int | None] = {}
    for cat in CATEGORY_KEYS:
        vals = [
            r.scores[cat.value]
            for r in filtered
            if cat.value in r.scores and r.scores[cat.value] is not None
        ]
        cat_avgs[cat] = round_int(sum(vals) / len(vals)) if vals else None

    grade_dist: dict[str, int] = {g: 0 for g in GRADE_KEYS}
    for r in filtered:
        g = r.grade or grade_for(r.overall_score)
        if g in grade_dist:
            grade_dist[g] += 1

    by_purpose: dict[Purpose, list[int]] = {}
    for r in filtered:
        by_purpose.setdefault(r.purpose, []).append(r.overall_score)
    purpose_items = [
        PurposeBreakdownItem(
            purpose=p,
            analysis_count=len(scores),
            average_score=round_int(sum(scores) / len(scores)),
        )
        for p, scores in sorted(
            by_purpose.items(),
            key=lambda kv: (-len(kv[1]), PURPOSE_ORDER.index(kv[0])),
        )
    ]

    weekly = compute_weekly_activity(purpose_records, now_kst)

    return MyStatsResponse(
        period=period,
        from_=from_dt,
        to=now_kst,
        presentation_count=presentation_count,
        analysis_count=n,
        total_practice_sec=total_sec,
        average_score=avg_score,
        best_score=best_score,
        best_analysis_id=best_id,
        recent_change=recent_change,
        streak_days=streak,
        category_averages=cat_avgs,
        grade_distribution=grade_dist,
        purpose_breakdown=purpose_items,
        weekly_activity=weekly,
    )


# ---------------------------------------------------------------- M2: 강점·약점 (Insights)


def format_value_text(metric: MetricKey, avg_val: float) -> str:
    """항목별 실제 수치를 사람이 읽는 문구로 바꾼다(docs/mypage-spec.md 4장 표)."""
    if metric is MetricKey.PACE:
        return f"분당 {round_int(avg_val)}음절"
    if metric in (MetricKey.FILLER, MetricKey.SILENCE, MetricKey.GESTURE):
        return f"분당 {round1(avg_val):.1f}회"
    if metric is MetricKey.TIME:
        return f"제한 시간의 {round_int(avg_val * 100)}% 사용"
    if metric is MetricKey.GAZE:
        return f"정면 {round_int(avg_val * 100)}%"
    if metric is MetricKey.POSTURE:
        return f"안정성 {round_int(avg_val * 100)}%"
    if metric in (MetricKey.STRUCTURE, MetricKey.CLARITY):
        return f"회차당 지적 {round1(avg_val):.1f}건"
    return f"회차당 부적절 표현 {round1(avg_val):.1f}건"


def _has_batchim(word: str) -> bool:
    for ch in reversed(word):
        if "가" <= ch <= "힣":
            return (ord(ch) - ord("가")) % 28 != 0
    return False


def format_insight_message(
    metric: MetricKey,
    score: int,
    avg_val: float,
    top_fillers: Sequence[FillerStat],
) -> str:
    """항목별 점수·원값에 따른 한 줄 설명·조언(해요체)."""
    is_good = score >= STRENGTH_MIN_SCORE
    is_weak = score < WEAKNESS_MAX_SCORE
    if metric is MetricKey.PACE:
        if is_good:
            return "말 속도가 안정적이어서 듣기 편해요."
        if avg_val > scoring.PACE_IDEAL_HIGH:
            return "말이 조금 빨라요. 핵심 문장 앞에서 속도를 늦춰 보세요."
        if avg_val < scoring.PACE_IDEAL_LOW:
            return "말이 조금 느려요. 문장 끝을 또렷하고 경쾌하게 이어 보세요."
        return "구간마다 말 속도 차이가 있어요. 일정한 호흡으로 말해 보세요."

    if metric is MetricKey.FILLER:
        if is_good:
            return "추임새가 적어 발표가 깔끔하게 들려요."
        if not is_weak:
            return "추임새가 무난한 수준이에요. 문장 사이 짧은 쉼을 의식하면 더 깔끔해져요."
        picked = [f.word for f in top_fillers[:2] if f.word]
        if picked:
            joined = ", ".join(f"'{w}'" for w in picked)
            josa = "이" if _has_batchim(picked[-1]) else "가"
            return f"{joined}{josa} 자주 나와요. 문장 사이에 말 대신 잠깐 쉬어 보세요."
        return "추임새가 자주 나와요. 문장 사이에 말 대신 잠깐 쉬어 보세요."

    if metric is MetricKey.SILENCE:
        if is_good:
            return "긴 침묵 없이 말의 흐름이 자연스러워요."
        if not is_weak:
            return "호흡이 대체로 무난해요. 전환 구간만 미리 정리하면 더 매끄러워져요."
        return "말이 길게 끊기는 구간이 있어요. 다음 문장 첫 단어를 미리 떠올려 보세요."

    if metric is MetricKey.TIME:
        if is_good:
            return "제한 시간에 맞춰 발표 분량을 잘 조절하고 있어요."
        if avg_val > scoring.TIME_IDEAL_HIGH:
            return "제한 시간을 넘기는 경향이 있어요. 본론 예시를 조금 줄여 보세요."
        return "제한 시간보다 발표가 일찍 끝나요. 핵심 근거를 한 가지 더 덧붙여 보세요."

    if metric is MetricKey.GAZE:
        if is_good:
            return "청중을 안정적으로 바라보고 있어요."
        if not is_weak:
            return "시선 처리가 무난해요. 핵심 문장에서 카메라를 조금 더 바라봐 주세요."
        return "시선이 자주 흔들려요. 문장을 마칠 때는 카메라를 바라봐 주세요."

    if metric is MetricKey.POSTURE:
        if is_good:
            return "발표 자세가 흔들림 없이 안정적이에요."
        if not is_weak:
            return "자세가 대체로 안정적이에요. 무게중심을 일정하게 유지해 보세요."
        return "상체가 자주 흔들려요. 양발에 무게중심을 고르게 두고 서 보세요."

    if metric is MetricKey.GESTURE:
        if is_good:
            return "손동작을 자연스럽고 적절하게 활용하고 있어요."
        if avg_val < scoring.GESTURE_IDEAL_LOW:
            return "손동작이 거의 없어요. 핵심 대목에서 가벼운 손짓을 더해 보세요."
        return "손동작이 다소 잦아요. 강조할 문장에서만 손을 움직여 보세요."

    if metric is MetricKey.STRUCTURE:
        if is_good:
            return "도입·본론·결론의 구성 흐름이 탄탄해요."
        if not is_weak:
            return "전체 구성은 갖추고 있어요. 파트 사이 연결 문장을 보강하면 더 좋아요."
        return "발표 흐름 안내가 부족해요. 도입에서 전체 순서를 먼저 짚어 주세요."

    if metric is MetricKey.CLARITY:
        if is_good:
            return "핵심 메시지와 근거가 명확하게 전달돼요."
        if not is_weak:
            return "전달 내용이 대체로 분명해요. 핵심 근거를 한 문장으로 요약해 보세요."
        return "주장 뒷받침이 다소 모호해요. 구체적인 수치나 사례를 함께 말해 보세요."

    if is_good:
        return "발표 자리에 어울리는 정중하고 명확한 표현을 쓰고 있어요."
    if not is_weak:
        return "대체로 격식 있는 표현을 쓰고 있어요. 구어체 표현만 조금 더 다듬어 보세요."
    return "발표 자리에 맞지 않는 표현이 있었어요. 차분하고 공식적인 문장으로 다듬어 보세요."


def compute_metric_trend(scores: Sequence[float]) -> tuple[Trend, int | None]:
    """판정 구간 안에서 최근 절반 평균 - 이전 절반 평균.

    표본 4회 미만이면 (FLAT, None). ±5점 미만이면 FLAT, +5 이상이면 UP, -5 이하면 DOWN.
    홀수 개(예: 5개)일 때는 이전 절반(앞 2개)과 최근 절반(뒤 2개)을 같은 크기로 비교한다.
    """
    n = len(scores)
    if n < MIN_TREND_SAMPLES:
        return Trend.FLAT, None
    half = n // 2
    prev_half = scores[:half]
    recent_half = scores[-half:]
    diff = round_int(sum(recent_half) / half - sum(prev_half) / half)
    if diff >= TREND_MIN_DIFF:
        return Trend.UP, diff
    if diff <= -TREND_MIN_DIFF:
        return Trend.DOWN, diff
    return Trend.FLAT, diff


def aggregate_top_fillers(records: Sequence[CompletedRecord]) -> list[FillerStat]:
    counts: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    order = 0
    for r in records:
        for item in r.filler_counts or []:
            w = str(item.get("word") or "").strip()
            c = int(item.get("count") or 0)
            if not w or c <= 0:
                continue
            counts[w] += c
            if w not in first_seen:
                first_seen[w] = order
                order += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return [FillerStat(word=w, count=c) for w, c in ranked[:MAX_TOP_FILLERS]]


def aggregate_frequent_feedbacks(
    records: Sequence[CompletedRecord],
) -> list[FrequentFeedbackItem]:
    counts: Counter[FeedbackType] = Counter()
    first_seen: dict[FeedbackType, int] = {}
    order = 0
    for r in records:
        for raw_type, raw_cnt in (r.feedback_counts or {}).items():
            cnt = int(raw_cnt or 0)
            if cnt <= 0:
                continue
            try:
                ft = FeedbackType(raw_type)
            except ValueError:
                continue
            counts[ft] += cnt
            if ft not in first_seen:
                first_seen[ft] = order
                order += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
    return [
        FrequentFeedbackItem(type=ft, label=FEEDBACK_TYPE_LABELS.get(ft, ft.value), count=cnt)
        for ft, cnt in ranked[:MAX_FREQUENT_FEEDBACKS]
    ]


def build_insights(
    records: Sequence[CompletedRecord],
    *,
    period: Period,
    purpose: Purpose | None,
    now: datetime,
) -> MyInsightsResponse:
    now_kst = as_kst(now)
    filtered = filter_records(records, period=period, purpose=purpose, now=now_kst)
    n = len(filtered)
    recent_window = filtered[-MAX_RECENT_ANALYSES:]
    based_on = len(recent_window)

    top_fillers = aggregate_top_fillers(filtered)
    recent_fillers = aggregate_top_fillers(recent_window) or top_fillers
    frequent_fbs = aggregate_frequent_feedbacks(filtered)

    if n < REQUIRED_ANALYSIS_COUNT:
        return MyInsightsResponse(
            status=InsightStatus.INSUFFICIENT_DATA,
            analysis_count=n,
            required_count=REQUIRED_ANALYSIS_COUNT,
            based_on_recent=based_on,
            strengths=[],
            weaknesses=[],
            metrics=[],
            focus_tip=None,
            top_fillers=top_fillers,
            frequent_feedback_types=[],
        )

    metrics_list: list[InsightItem] = []
    for m_key, cat, label in METRIC_SPECS:
        samples: list[tuple[float, float]] = []
        for r in recent_window:
            entry = (r.metric_details or {}).get(m_key.value)
            if not isinstance(entry, dict):
                continue
            sc_val = entry.get("score")
            raw_val = entry.get("value")
            if sc_val is None or raw_val is None:
                continue
            samples.append((float(sc_val), float(raw_val)))
        if not samples:
            continue
        sample_scores = [s for s, _ in samples]
        sample_vals = [v for _, v in samples]
        avg_score = round_int(sum(sample_scores) / len(sample_scores))
        avg_val = sum(sample_vals) / len(sample_vals)
        trend, change = compute_metric_trend(sample_scores)
        metrics_list.append(
            InsightItem(
                metric=m_key,
                category=cat,
                label=label,
                score=avg_score,
                trend=trend,
                change=change,
                sample_count=len(samples),
                value_text=format_value_text(m_key, avg_val),
                message=format_insight_message(m_key, avg_score, avg_val, recent_fillers),
            )
        )

    spec_order = {m_key: idx for idx, (m_key, _, _) in enumerate(METRIC_SPECS)}
    strengths = sorted(
        (
            m
            for m in metrics_list
            if m.sample_count >= MIN_METRIC_SAMPLES and m.score >= STRENGTH_MIN_SCORE
        ),
        key=lambda m: (-m.score, spec_order[m.metric]),
    )[:MAX_STRENGTHS]
    weaknesses = sorted(
        (
            m
            for m in metrics_list
            if m.sample_count >= MIN_METRIC_SAMPLES and m.score < WEAKNESS_MAX_SCORE
        ),
        key=lambda m: (m.score, spec_order[m.metric]),
    )[:MAX_WEAKNESSES]

    focus_tip = FOCUS_TIPS[weaknesses[0].metric] if weaknesses else None

    return MyInsightsResponse(
        status=InsightStatus.READY,
        analysis_count=n,
        required_count=REQUIRED_ANALYSIS_COUNT,
        based_on_recent=based_on,
        strengths=strengths,
        weaknesses=weaknesses,
        metrics=metrics_list,
        focus_tip=focus_tip,
        top_fillers=top_fillers,
        frequent_feedback_types=frequent_fbs,
    )


# ---------------------------------------------------------------- M3: 점수 추이 (Score Timeline)


def build_score_timeline(
    records: Sequence[CompletedRecord],
    *,
    period: Period,
    purpose: Purpose | None,
    limit: int,
    now: datetime,
) -> ScoreTimelineResponse:
    now_kst = as_kst(now)
    filtered = filter_records(records, period=period, purpose=purpose, now=now_kst)
    total = len(filtered)
    sliced = filtered[-limit:] if limit > 0 else []
    items = [
        ScoreTimelineItem(
            analysis_id=r.analysis_id,
            presentation_id=r.presentation_id,
            presentation_title=r.presentation_title,
            purpose=r.purpose,
            attempt_no=r.attempt_no,
            created_at=r.created_at,
            overall_score=r.overall_score,
            grade=r.grade or grade_for(r.overall_score),
            scores={
                cat: int(r.scores.get(cat.value, scoring.NEUTRAL_SCORE))
                for cat in CATEGORY_KEYS
            },
        )
        for r in sliced
    ]
    return ScoreTimelineResponse(items=items, total=total)
