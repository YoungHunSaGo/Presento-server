"""마이페이지 M1~M3(stats, insights, score-timeline) 및 백필 테스트(외부 호출 없음)."""

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from app.models import Analysis, AnalysisArtifact, AnalysisStatus, Presentation, Report, User
from app.models.enums import Purpose
from app.schemas.common import KST
from app.schemas.mypage import MetricKey, Period, Trend
from app.services.mypage import (
    GRADE_KEYS,
    WEEKLY_ACTIVITY_WEEKS,
    CompletedRecord,
    build_insights,
    build_score_timeline,
    build_stats,
    compute_metric_trend,
    compute_recent_change,
    compute_streak_days,
    format_value_text,
)
from app.services.pipeline.scoring import grade_for
from app.tools import recompute as cli

API = "/api/v1"
ROOT = Path(__file__).resolve().parent.parent


def _keys_tree(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _keys_tree(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_keys_tree(v[0])] if v and isinstance(v[0], dict) else []
    return None


def _spec_json_blocks() -> list[dict[str, Any]]:
    spec = (ROOT / "docs" / "mypage-spec.md").read_text(encoding="utf-8")
    return [json.loads(m.group(1)) for m in re.finditer(r"```json\n(.*?)```", spec, re.S)]


def _rec(
    *,
    aid: str = "a-1",
    pid: str = "p-1",
    title: str = "발표 1",
    purpose: Purpose = Purpose.CLASS,
    attempt_no: int = 1,
    completed_at: datetime | None = None,
    duration_sec: float = 120.0,
    overall_score: int = 80,
    scores: dict[str, int] | None = None,
    metric_details: dict[str, Any] | None = None,
    filler_counts: list[dict[str, Any]] | None = None,
    feedback_counts: dict[str, int] | None = None,
) -> CompletedRecord:
    comp = completed_at or datetime(2026, 11, 5, 11, 0, tzinfo=UTC)
    sc = scores or {"DELIVERY": overall_score, "NONVERBAL": overall_score, "CONTENT": overall_score}
    md = (
        metric_details
        if metric_details is not None
        else {
            "PACE": {"score": 63.0, "value": 424.0},
            "FILLER": {"score": 58.0, "value": 4.8},
            "SILENCE": {"score": 85.0, "value": 0.4},
            "TIME": {"score": 92.0, "value": 0.92},
            "GAZE": {"score": 89.0, "value": 0.79},
            "POSTURE": {"score": 84.0, "value": 0.84},
            "GESTURE": None,  # 측정 불가(손이 안 보임)
            "STRUCTURE": {"score": 85.0, "value": 1.0},
            "CLARITY": {"score": 90.0, "value": 1.0},
            "LANGUAGE": {"score": 100.0, "value": 0.0},
        }
    )
    return CompletedRecord(
        analysis_id=aid,
        presentation_id=pid,
        presentation_title=title,
        purpose=purpose,
        attempt_no=attempt_no,
        created_at=comp,
        completed_at=comp,
        duration_sec=duration_sec,
        overall_score=overall_score,
        grade=grade_for(overall_score),
        scores=sc,
        metric_details=md,
        filler_counts=filler_counts
        if filler_counts is not None
        else [{"word": "약간", "count": 4}, {"word": "이제", "count": 2}],
        feedback_counts=feedback_counts
        if feedback_counts is not None
        else {"FILLER": 2, "PACE_FAST": 1},
    )


# ---------- 1. 명세 JSON 키 구조 일치 검증 ----------


def test_responses_match_mypage_spec_json_structure():
    m1_spec, m2_ready_spec, m2_insuff_spec, m3_spec = _spec_json_blocks()
    now = datetime(2026, 11, 5, 20, 0, tzinfo=KST)

    records = [
        _rec(
            aid=f"a-{i}",
            pid=f"p-{i % 2}",
            purpose=Purpose.CLASS if i % 2 == 0 else Purpose.INTERVIEW,
            attempt_no=i,
            completed_at=now - timedelta(days=12 - i),
            overall_score=70 + i,
        )
        for i in range(1, 13)
    ]

    m1 = build_stats(records, period=Period.ALL, purpose=None, now=now).model_dump(
        mode="json", by_alias=True
    )
    assert _keys_tree(m1) == _keys_tree(m1_spec)
    assert len(m1["weeklyActivity"]) == WEEKLY_ACTIVITY_WEEKS == 12
    assert list(m1["gradeDistribution"].keys()) == list(GRADE_KEYS)

    m2_ready = build_insights(records, period=Period.ALL, purpose=None, now=now).model_dump(
        mode="json", by_alias=True
    )
    assert _keys_tree(m2_ready) == _keys_tree(m2_ready_spec)

    m2_insuff = build_insights(records[:2], period=Period.ALL, purpose=None, now=now).model_dump(
        mode="json", by_alias=True
    )
    assert _keys_tree(m2_insuff) == _keys_tree(m2_insuff_spec)

    m3 = build_score_timeline(
        records, period=Period.ALL, purpose=None, limit=50, now=now
    ).model_dump(mode="json", by_alias=True)
    assert _keys_tree(m3) == _keys_tree(m3_spec)


# ---------- 2. 순수 함수 단위 테스트 (연속일, 최근 변화, 추이, 문구) ----------


def test_streak_days_today_yesterday_broken_and_kst_midnight():
    today = date(2026, 11, 5)
    # 오늘 포함 3일 연속
    days_3 = {today, today - timedelta(days=1), today - timedelta(days=2)}
    assert compute_streak_days(days_3, today) == 3
    # 오늘은 안 했지만 어제까지 2일 연속 → 유효
    assert compute_streak_days({today - timedelta(days=1), today - timedelta(days=2)}, today) == 2
    # 어제도 오늘도 안 했으면 0
    assert compute_streak_days({today - timedelta(days=2), today - timedelta(days=3)}, today) == 0
    assert compute_streak_days(set(), today) == 0

    # KST 자정 경계: UTC 14:59는 KST 23:59(전날), UTC 15:01은 KST 00:01(다음날)
    r_prev = _rec(aid="a-1", completed_at=datetime(2026, 11, 3, 14, 59, tzinfo=UTC))
    r_next = _rec(aid="a-2", completed_at=datetime(2026, 11, 3, 15, 1, tzinfo=UTC))
    now_kst = datetime(2026, 11, 5, 10, 0, tzinfo=KST)  # 오늘 11/05, 어제 11/04, 그제 11/03
    stats = build_stats([r_prev, r_next], period=Period.ALL, purpose=None, now=now_kst)
    assert stats.streak_days == 2


def test_recent_change_requires_at_least_six_and_compares_windows():
    now = datetime(2026, 11, 5, 12, 0, tzinfo=KST)
    recs_5 = [
        _rec(aid=f"a-{i}", completed_at=now - timedelta(days=10 - i), overall_score=80)
        for i in range(5)
    ]
    assert compute_recent_change(recs_5) is None

    # 6회: 직전 1회(60) vs 최근 5회(80) -> +20
    recs_6 = [_rec(aid="a-0", completed_at=now - timedelta(days=20), overall_score=60), *recs_5]
    assert compute_recent_change(recs_6) == 20

    # 10회: 직전 5회(70) vs 최근 5회(76) -> +6
    prev_5 = [
        _rec(aid=f"p-{i}", completed_at=now - timedelta(days=20 - i), overall_score=70)
        for i in range(5)
    ]
    cur_5 = [
        _rec(aid=f"c-{i}", completed_at=now - timedelta(days=5 - i), overall_score=76)
        for i in range(5)
    ]
    assert compute_recent_change(prev_5 + cur_5) == 6


@pytest.mark.parametrize(
    ("scores", "expected_trend", "expected_change"),
    [
        ([70, 80, 90], Trend.FLAT, None),  # 표본 4회 미만
        ([70, 70, 74, 74], Trend.FLAT, 4),  # +4점(< 5)은 FLAT
        ([70, 70, 75, 75], Trend.UP, 5),  # 정확히 +5점 경계는 UP
        ([80, 80, 75, 75], Trend.DOWN, -5),  # 정확히 -5점 경계는 DOWN
    ],
)
def test_metric_trend_boundaries(scores, expected_trend, expected_change):
    trend, change = compute_metric_trend(scores)
    assert trend is expected_trend
    assert change == expected_change


def test_value_text_formats_all_ten_metrics():
    assert format_value_text(MetricKey.PACE, 341.2) == "분당 341음절"
    assert format_value_text(MetricKey.FILLER, 3.44) == "분당 3.4회"
    assert format_value_text(MetricKey.SILENCE, 0.8) == "분당 0.8회"
    assert format_value_text(MetricKey.TIME, 0.92) == "제한 시간의 92% 사용"
    assert format_value_text(MetricKey.GAZE, 0.78) == "정면 78%"
    assert format_value_text(MetricKey.POSTURE, 0.86) == "안정성 86%"
    assert format_value_text(MetricKey.GESTURE, 6.2) == "분당 6.2회"
    assert format_value_text(MetricKey.STRUCTURE, 1.2) == "회차당 지적 1.2건"
    assert format_value_text(MetricKey.CLARITY, 0.8) == "회차당 지적 0.8건"
    assert format_value_text(MetricKey.LANGUAGE, 0.3) == "회차당 부적절 표현 0.3건"


def test_insights_unmeasured_excluded_and_sample_count_rule():
    now = datetime(2026, 11, 5, 12, 0, tzinfo=KST)
    # 3회 완료, TIME은 1회만 측정(sampleCount=1 < 2 -> 강점/약점 제외, metrics에는 포함)
    # GESTURE는 0회 측정(None -> metrics에서도 제외)
    r1 = _rec(
        aid="a-1",
        completed_at=now - timedelta(days=3),
        metric_details={
            "PACE": {"score": 60.0, "value": 430.0},
            "FILLER": {"score": 55.0, "value": 5.0},
            "TIME": {"score": 95.0, "value": 0.95},
            "GAZE": {"score": 90.0, "value": 0.85},
            "GESTURE": None,
        },
    )
    r2 = _rec(
        aid="a-2",
        completed_at=now - timedelta(days=2),
        metric_details={
            "PACE": {"score": 64.0, "value": 420.0},
            "FILLER": {"score": 59.0, "value": 4.6},
            "TIME": None,
            "GAZE": {"score": 88.0, "value": 0.82},
            "GESTURE": None,
        },
    )
    r3 = _rec(
        aid="a-3",
        completed_at=now - timedelta(days=1),
        metric_details={
            "PACE": {"score": 65.0, "value": 415.0},
            "FILLER": {"score": 60.0, "value": 4.5},
            "TIME": None,
            "GAZE": {"score": 92.0, "value": 0.88},
            "GESTURE": None,
        },
    )
    ins = build_insights([r1, r2, r3], period=Period.ALL, purpose=None, now=now)
    assert ins.status.value == "READY"
    metric_keys = [m.metric for m in ins.metrics]
    assert MetricKey.GESTURE not in metric_keys
    assert MetricKey.TIME in metric_keys  # 1회 측정된 항목은 metrics에는 나온다
    # TIME은 sampleCount=1이므로 strengths에는 들어가지 않는다
    assert [s.metric for s in ins.strengths] == [MetricKey.GAZE]
    # 약점은 점수 낮은 순(FILLER 58 < PACE 63)
    assert [w.metric for w in ins.weaknesses] == [MetricKey.FILLER, MetricKey.PACE]
    # FILLER 메시지에 상위 추임새 단어가 들어간다
    filler_item = next(w for w in ins.weaknesses if w.metric is MetricKey.FILLER)
    assert "'약간'" in filler_item.message and "'이제'" in filler_item.message
    assert ins.focus_tip is not None and "추임새" in ins.focus_tip


# ---------- 3. API 엔드포인트 통합 테스트 (M1~M3) ----------


def _seed_completed_analysis(
    db,
    *,
    user_id: str,
    title: str = "발표",
    purpose: str = "CLASS",
    attempt_no: int = 1,
    status: str = AnalysisStatus.COMPLETED,
    overall_score: int = 80,
    duration_sec: float = 100.0,
    completed_at: datetime | None = None,
    scores: dict[str, int] | None = None,
    metric_details: dict[str, Any] | None = None,
    filler_counts: list[dict[str, Any]] | None = None,
    feedback_counts: dict[str, int] | None = None,
) -> tuple[Presentation, Analysis]:
    comp = completed_at or datetime.now(UTC)
    p = Presentation(
        user_id=user_id, title=title, purpose=purpose, created_at=comp, updated_at=comp
    )
    db.add(p)
    db.flush()
    is_done = status == AnalysisStatus.COMPLETED
    default_scores = {
        "DELIVERY": overall_score - 2,
        "NONVERBAL": overall_score,
        "CONTENT": overall_score + 2,
    }
    a = Analysis(
        presentation_id=p.id,
        attempt_no=attempt_no,
        status=status,
        duration_sec=duration_sec if is_done else None,
        overall_score=overall_score if is_done else None,
        grade=grade_for(overall_score) if is_done else None,
        scores=(scores or default_scores) if is_done else None,
        metric_details=(
            metric_details
            if metric_details is not None
            else {
                "PACE": {"score": 62.0, "value": 420.0},
                "FILLER": {"score": 56.0, "value": 4.9},
                "GAZE": {"score": 90.0, "value": 0.85},
            }
        )
        if is_done
        else None,
        filler_counts=(
            filler_counts if filler_counts is not None else [{"word": "약간", "count": 3}]
        )
        if is_done
        else None,
        feedback_counts=(
            feedback_counts if feedback_counts is not None else {"FILLER": 2, "PACE_FAST": 1}
        )
        if is_done
        else None,
        filler_count=3 if is_done else None,
        syllables_per_min=350 if is_done else None,
        created_at=comp,
        completed_at=comp if is_done else None,
    )
    db.add(a)
    db.commit()
    return p, a


def test_mypage_endpoints_require_auth(client):
    for path in ("/users/me/stats", "/users/me/insights", "/users/me/score-timeline"):
        res = client.get(API + path)
        assert res.status_code == 401
        assert res.json()["error"]["code"] == "UNAUTHORIZED"


def test_mypage_empty_user_zero_analyses(client, auth_headers):
    s = client.get(f"{API}/users/me/stats", headers=auth_headers)
    assert s.status_code == 200
    sb = s.json()
    assert sb["period"] == "all" and sb["from"] is None
    assert sb["presentationCount"] == 0 and sb["analysisCount"] == 0
    assert sb["totalPracticeSec"] == 0
    assert sb["averageScore"] is None and sb["bestScore"] is None and sb["bestAnalysisId"] is None
    assert sb["recentChange"] is None and sb["streakDays"] == 0
    assert sb["categoryAverages"] == {"DELIVERY": None, "NONVERBAL": None, "CONTENT": None}
    assert sb["gradeDistribution"] == {g: 0 for g in GRADE_KEYS}
    assert sb["purposeBreakdown"] == []
    assert len(sb["weeklyActivity"]) == 12
    assert all(w["analysisCount"] == 0 and w["averageScore"] is None for w in sb["weeklyActivity"])

    ins = client.get(f"{API}/users/me/insights", headers=auth_headers)
    assert ins.status_code == 200
    ib = ins.json()
    assert ib == {
        "status": "INSUFFICIENT_DATA",
        "analysisCount": 0,
        "requiredCount": 3,
        "basedOnRecent": 0,
        "strengths": [],
        "weaknesses": [],
        "metrics": [],
        "focusTip": None,
        "topFillers": [],
        "frequentFeedbackTypes": [],
    }

    tl = client.get(f"{API}/users/me/score-timeline", headers=auth_headers)
    assert tl.status_code == 200
    assert tl.json() == {"items": [], "total": 0}


def test_mypage_excludes_failed_waiting_deleted_and_other_users(
    client, auth_headers, other_headers, db
):
    me_id = client.get(f"{API}/users/me", headers=auth_headers).json()["id"]
    other_id = client.get(f"{API}/users/me", headers=other_headers).json()["id"]
    now = datetime.now(UTC)

    # 내 완료 회차 2개
    _, a1 = _seed_completed_analysis(
        db,
        user_id=me_id,
        title="수업1",
        purpose="CLASS",
        overall_score=70,
        completed_at=now - timedelta(days=2),
    )
    _, a2 = _seed_completed_analysis(
        db,
        user_id=me_id,
        title="수업2",
        purpose="CLASS",
        overall_score=84,
        completed_at=now - timedelta(days=1),
    )
    # 내 실패·대기 회차 (집계 제외)
    _seed_completed_analysis(db, user_id=me_id, status=AnalysisStatus.FAILED)
    _seed_completed_analysis(db, user_id=me_id, status=AnalysisStatus.WAITING_UPLOAD)
    # 삭제할 내 발표 (집계 제외)
    p_del, _ = _seed_completed_analysis(db, user_id=me_id, overall_score=99)
    client.delete(f"{API}/presentations/{p_del.id}", headers=auth_headers)
    # 다른 사용자의 완료 회차 (집계 제외)
    _seed_completed_analysis(db, user_id=other_id, overall_score=100)

    sb = client.get(f"{API}/users/me/stats", headers=auth_headers).json()
    assert sb["analysisCount"] == 2
    assert sb["presentationCount"] == 2
    assert sb["bestScore"] == 84 and sb["bestAnalysisId"] == a2.id

    ib = client.get(f"{API}/users/me/insights", headers=auth_headers).json()
    assert ib["status"] == "INSUFFICIENT_DATA"
    assert ib["analysisCount"] == 2 and ib["basedOnRecent"] == 2
    assert ib["topFillers"] == [{"word": "약간", "count": 6}]

    tl = client.get(f"{API}/users/me/score-timeline", headers=auth_headers).json()
    assert tl["total"] == 2
    assert [it["analysisId"] for it in tl["items"]] == [a1.id, a2.id]


def test_mypage_period_and_purpose_filters(client, auth_headers, db):
    me_id = client.get(f"{API}/users/me", headers=auth_headers).json()["id"]
    now = datetime.now(UTC)

    # 1) 100일 전 CLASS
    _seed_completed_analysis(
        db, user_id=me_id, purpose="CLASS", overall_score=60, completed_at=now - timedelta(days=100)
    )
    # 2) 50일 전 INTERVIEW
    _seed_completed_analysis(
        db,
        user_id=me_id,
        purpose="INTERVIEW",
        overall_score=75,
        completed_at=now - timedelta(days=50),
    )
    # 3) 5일 전 CLASS
    _seed_completed_analysis(
        db, user_id=me_id, purpose="CLASS", overall_score=90, completed_at=now - timedelta(days=5)
    )

    all_stats = client.get(f"{API}/users/me/stats?period=all", headers=auth_headers).json()
    assert all_stats["analysisCount"] == 3 and all_stats["from"] is None

    d90_stats = client.get(f"{API}/users/me/stats?period=90d", headers=auth_headers).json()
    assert d90_stats["analysisCount"] == 2 and d90_stats["from"] is not None

    d30_stats = client.get(f"{API}/users/me/stats?period=30d", headers=auth_headers).json()
    assert d30_stats["analysisCount"] == 1 and d30_stats["averageScore"] == 90

    class_stats = client.get(
        f"{API}/users/me/stats?period=all&purpose=CLASS", headers=auth_headers
    ).json()
    assert class_stats["analysisCount"] == 2 and class_stats["averageScore"] == 75


@pytest.mark.parametrize(
    "query",
    [
        "period=7d",
        "purpose=UNKNOWN",
        "limit=0",
        "limit=201",
        "limit=-1",
        "limit=abc",
    ],
)
def test_mypage_invalid_query_params_return_400_validation_error(client, auth_headers, query):
    path = "/users/me/score-timeline" if "limit=" in query else "/users/me/stats"
    res = client.get(f"{API}{path}?{query}", headers=auth_headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_score_timeline_limit_boundaries_and_oldest_first_order(client, auth_headers, db):
    me_id = client.get(f"{API}/users/me", headers=auth_headers).json()["id"]
    now = datetime.now(UTC)
    ids = []
    for i in range(5):
        _, a = _seed_completed_analysis(
            db,
            user_id=me_id,
            title=f"발표 {i}",
            overall_score=60 + i * 5,
            completed_at=now - timedelta(days=10 - i),
        )
        ids.append(a.id)

    # limit=1 -> 가장 최근 1개, total은 전체 5개
    r1 = client.get(f"{API}/users/me/score-timeline?limit=1", headers=auth_headers)
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["total"] == 5
    assert [it["analysisId"] for it in b1["items"]] == [ids[-1]]

    # limit=3 -> 최근 3개를 오래된 순으로
    r3 = client.get(f"{API}/users/me/score-timeline?limit=3", headers=auth_headers).json()
    assert r3["total"] == 5
    assert [it["analysisId"] for it in r3["items"]] == ids[-3:]

    # limit=200(최대 허용값) 통과
    r200 = client.get(f"{API}/users/me/score-timeline?limit=200", headers=auth_headers)
    assert r200.status_code == 200
    assert len(r200.json()["items"]) == 5


# ---------- 4. 파이프라인 완료 및 --backfill 테스트 ----------


def test_backfill_populates_missing_mypage_columns_and_skips_broken(
    client, auth_headers, db, capsys
):
    from app.services.pipeline.fake import build_fake_report

    me_id = client.get(f"{API}/users/me", headers=auth_headers).json()["id"]
    now = datetime.now(UTC)
    p = Presentation(
        user_id=me_id, title="백필 발표", purpose="CLASS", created_at=now, updated_at=now
    )
    db.add(p)
    db.flush()

    # 마이그레이션 전 완료 회차(grade·metric_details·filler_counts·feedback_counts가 비어 있음)
    a_ok = Analysis(
        presentation_id=p.id,
        attempt_no=1,
        status=AnalysisStatus.COMPLETED,
        duration_sec=287.4,
        overall_score=82,
        scores={"DELIVERY": 80, "NONVERBAL": 84, "CONTENT": 81},
        filler_count=9,
        syllables_per_min=341,
        created_at=now,
        completed_at=now,
    )
    # 리포트가 없는 깨진 완료 회차(건너뛰어야 함)
    a_bad = Analysis(
        presentation_id=p.id,
        attempt_no=2,
        status=AnalysisStatus.COMPLETED,
        duration_sec=100.0,
        overall_score=70,
        scores={"DELIVERY": 70, "NONVERBAL": 70, "CONTENT": 70},
        created_at=now,
        completed_at=now,
    )
    db.add_all([a_ok, a_bad])
    db.flush()
    db.add(Report(analysis_id=a_ok.id, data=build_fake_report(a_ok, 287.4), created_at=now))
    db.commit()

    assert cli.main(["--backfill"], session_factory=lambda: db) == 0
    out = capsys.readouterr().out
    assert "backfilled=1 skipped=1" in out

    db.expire_all()
    refreshed = db.get(Analysis, a_ok.id)
    assert refreshed.grade == "B+"
    assert refreshed.metric_details is not None
    assert refreshed.metric_details["PACE"]["value"] == 341.0
    assert refreshed.filler_counts and refreshed.filler_counts[0]["word"] == "약간"
    assert refreshed.feedback_counts == {"PACE_FAST": 1, "GAZE_AWAY": 1}
    _ = (User, AnalysisArtifact)


def test_filler_josa_and_neutral_score_messages_and_period_preserved_weekly():
    from app.schemas.report import FeedbackType, FillerStat
    from app.services.mypage import FEEDBACK_TYPE_LABELS, format_insight_message

    # 모든 FeedbackType이 라벨 사전에 정의되어 있다
    assert set(FeedbackType) == set(FEEDBACK_TYPE_LABELS)

    # 받침 있는 단어('음')로 끝나면 '이', 받침 없는 단어('이제')로 끝나면 '가'
    msg_eum = format_insight_message(
        MetricKey.FILLER, 55, 5.0, [FillerStat(word="음", count=5)]
    )
    assert "'음'이 자주 나와요." in msg_eum
    msg_ije = format_insight_message(
        MetricKey.FILLER,
        55,
        5.0,
        [FillerStat(word="약간", count=5), FillerStat(word="이제", count=3)],
    )
    assert "'약간', '이제'가 자주 나와요." in msg_ije

    # 중립 구간(70 <= score < 80)에서는 약점 단정 문구 대신 중립 격려 문구가 나온다
    msg_neutral = format_insight_message(MetricKey.GAZE, 75, 0.72, [])
    assert "무난해요" in msg_neutral

    # period=30d여도 weeklyActivity(12주)는 31~84일 전 연습 기록을 0으로 자르지 않는다
    now = datetime(2026, 11, 5, 12, 0, tzinfo=KST)
    old_rec = _rec(aid="a-old", completed_at=now - timedelta(days=45), overall_score=85)
    new_rec = _rec(aid="a-new", completed_at=now - timedelta(days=2), overall_score=90)
    st = build_stats([old_rec, new_rec], period=Period.DAYS_30, purpose=None, now=now)
    assert st.analysis_count == 1  # 30일 안 회차는 1개
    assert sum(w.analysis_count for w in st.weekly_activity) == 2  # 12주 차트에는 둘 다 표시


def test_mypage_get_never_backfills_or_writes_db_and_treats_none_as_unmeasured(
    client, auth_headers, other_headers, db, monkeypatch
):
    from sqlalchemy import event
    from sqlalchemy.orm import Session

    from app.services.pipeline import artifacts as art_mod

    me_id = client.get(f"{API}/users/me", headers=auth_headers).json()["id"]
    other_id = client.get(f"{API}/users/me", headers=other_headers).json()["id"]
    now = datetime.now(UTC)

    # 1) 세부 항목이 있는 완료 회차 2개 (점수 90, 80)
    _seed_completed_analysis(
        db,
        user_id=me_id,
        title="정상1",
        overall_score=90,
        completed_at=now - timedelta(days=3),
    )
    _seed_completed_analysis(
        db,
        user_id=me_id,
        title="정상2",
        overall_score=80,
        completed_at=now - timedelta(days=2),
    )

    # 2) 마이그레이션 이전 완료 회차: metric_details·filler_counts·feedback_counts·grade 모두 None
    p_legacy = Presentation(
        user_id=me_id, title="구버전", purpose="CLASS", created_at=now, updated_at=now
    )
    p_other = Presentation(
        user_id=other_id, title="타인구버전", purpose="CLASS", created_at=now, updated_at=now
    )
    db.add_all([p_legacy, p_other])
    db.flush()
    a_legacy = Analysis(
        presentation_id=p_legacy.id,
        attempt_no=1,
        status=AnalysisStatus.COMPLETED,
        duration_sec=60.0,
        overall_score=70,
        grade=None,
        scores={"DELIVERY": 70, "NONVERBAL": 70, "CONTENT": 70},
        metric_details=None,
        filler_counts=None,
        feedback_counts=None,
        created_at=now - timedelta(days=1),
        completed_at=now - timedelta(days=1),
    )
    a_other_legacy = Analysis(
        presentation_id=p_other.id,
        attempt_no=1,
        status=AnalysisStatus.COMPLETED,
        duration_sec=60.0,
        overall_score=60,
        grade=None,
        scores={"DELIVERY": 60, "NONVERBAL": 60, "CONTENT": 60},
        metric_details=None,
        filler_counts=None,
        feedback_counts=None,
        created_at=now - timedelta(days=1),
        completed_at=now - timedelta(days=1),
    )
    db.add_all([a_legacy, a_other_legacy])
    db.commit()

    # GET 요청 중 backfill 호출 금지 및 DB flush(쓰기) 감시
    monkeypatch.setattr(
        art_mod,
        "backfill_analysis_summary",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("backfill called during GET")),
    )
    flushes: list[int] = []

    def _on_flush(_session, _flush_context, _instances):
        flushes.append(1)

    event.listen(Session, "before_flush", _on_flush)
    try:
        m1 = client.get(f"{API}/users/me/stats", headers=auth_headers)
        m2 = client.get(f"{API}/users/me/insights", headers=auth_headers)
        m3 = client.get(f"{API}/users/me/score-timeline", headers=auth_headers)
    finally:
        event.remove(Session, "before_flush", _on_flush)

    assert m1.status_code == 200 and m2.status_code == 200 and m3.status_code == 200
    assert flushes == []  # GET 요청 중 어떤 DB 쓰기도 발생하지 않는다

    # (b) metric_details=None인 회차도 M1·M3 전체/카테고리 점수에는 포함된다(총 3회)
    m1_body = m1.json()
    assert m1_body["analysisCount"] == 3
    assert m1_body["averageScore"] == 80  # (90 + 80 + 70) / 3
    assert m1_body["gradeDistribution"]["B"] == 1  # grade=None이어도 70점 -> B로 집계

    m3_body = m3.json()
    assert m3_body["total"] == 3 and len(m3_body["items"]) == 3
    assert m3_body["items"][-1]["analysisId"] == a_legacy.id
    assert m3_body["items"][-1]["grade"] == "B"

    # (b) M2에서는 전체 완료 회차 수(3회)는 인정하되 세부 항목 평균·추임새·피드백 집계에서는 빠진다
    m2_body = m2.json()
    assert m2_body["status"] == "READY"
    assert m2_body["analysisCount"] == 3 and m2_body["basedOnRecent"] == 3
    assert all(m["sampleCount"] == 2 for m in m2_body["metrics"])
    assert m2_body["topFillers"] == [{"word": "약간", "count": 6}]  # 정상 2회(3+3)만 합산
    assert m2_body["frequentFeedbackTypes"] == [
        {"type": "FILLER", "label": "추임새", "count": 4},
        {"type": "PACE_FAST", "label": "말 속도 빠름", "count": 2},
    ]

    # (a) & (c) 내 구버전 회차와 다른 사용자의 구버전 회차 모두 DB에서 바뀌지 않았다
    db.expire_all()
    assert db.get(Analysis, a_legacy.id).metric_details is None
    assert db.get(Analysis, a_other_legacy.id).metric_details is None
