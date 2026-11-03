"""하루 분석 횟수: 차감은 N2(분석 시작)에서, N1은 이미 찬 경우 미리 429."""

from datetime import UTC, datetime, timedelta

import pytest

from app.models import AnalysisUsage

API = "/api/v1"
VIDEO = {"fileName": "t.webm", "contentType": "video/webm", "fileSize": 1000, "source": "UPLOAD"}


@pytest.fixture
def make_presentation(client):
    def _make(headers) -> str:
        res = client.post(
            f"{API}/presentations", json={"title": "t", "purpose": "CLASS"}, headers=headers
        )
        return res.json()["id"]

    return _make


@pytest.fixture
def n1(client):
    def _n1(pid: str, headers):
        return client.post(f"{API}/presentations/{pid}/analyses", json=VIDEO, headers=headers)

    return _n1


@pytest.fixture
def start(client, n1, storage):
    """N1 → 업로드 완료 → N2. N2 응답을 돌려준다."""

    def _start(pid: str, headers):
        res = n1(pid, headers)
        assert res.status_code == 201, res.text
        aid = res.json()["analysisId"]
        storage.objects[f"videos/{aid}"] = 1000
        return client.post(f"{API}/analyses/{aid}/start", headers=headers)

    return _start


def _err(res) -> str:
    return res.json()["error"]["code"]


def test_n1_does_not_consume_quota(make_presentation, n1, auth_headers, settings, db):
    settings.daily_analysis_limit = 1
    pid = make_presentation(auth_headers)
    for _ in range(3):
        assert n1(pid, auth_headers).status_code == 201
    assert db.query(AnalysisUsage).count() == 0


def test_n2_consumes_quota_then_429(
    client, make_presentation, n1, start, auth_headers, settings, storage, queue
):
    settings.daily_analysis_limit = 2
    pid = make_presentation(auth_headers)
    assert start(pid, auth_headers).status_code == 202
    assert start(pid, auth_headers).status_code == 202

    # 한도가 찼으면 N1에서 미리 막아 업로드 전에 알 수 있다.
    early = n1(pid, auth_headers)
    assert early.status_code == 429
    assert _err(early) == "RATE_LIMITED"
    assert "2회" in early.json()["error"]["message"]
    assert len(queue.analyses) == 2


def test_n2_rechecks_limit(client, make_presentation, n1, auth_headers, settings, storage, queue):
    """N1 두 개를 먼저 만들어 두고 N2를 차례로 부르면 한도를 넘는 N2가 429."""
    settings.daily_analysis_limit = 1
    pid = make_presentation(auth_headers)
    ids = [n1(pid, auth_headers).json()["analysisId"] for _ in range(2)]
    for aid in ids:
        storage.objects[f"videos/{aid}"] = 1000

    assert client.post(f"{API}/analyses/{ids[0]}/start", headers=auth_headers).status_code == 202
    res = client.post(f"{API}/analyses/{ids[1]}/start", headers=auth_headers)
    assert res.status_code == 429
    assert _err(res) == "RATE_LIMITED"
    status = client.get(f"{API}/analyses/{ids[1]}", headers=auth_headers).json()["status"]
    assert status == "WAITING_UPLOAD"  # 내일 다시 시작할 수 있다
    assert queue.analyses == [ids[0]]


def test_repeated_n2_consumes_once(
    client, make_presentation, n1, auth_headers, settings, storage, db
):
    settings.daily_analysis_limit = 5
    pid = make_presentation(auth_headers)
    aid = n1(pid, auth_headers).json()["analysisId"]
    storage.objects[f"videos/{aid}"] = 1000
    for _ in range(3):
        client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert db.query(AnalysisUsage).count() == 1


@pytest.mark.parametrize("failure", ["upload_missing", "too_long", "too_large"])
def test_rejected_n2_does_not_consume(
    client, make_presentation, n1, auth_headers, storage, probe, db, failure
):
    pid = make_presentation(auth_headers)
    aid = n1(pid, auth_headers).json()["analysisId"]
    if failure != "upload_missing":
        storage.objects[f"videos/{aid}"] = 600 * 1024 * 1024 if failure == "too_large" else 1000
    if failure == "too_long":
        probe.duration = 700.0
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code in (409, 413, 422)
    assert db.query(AnalysisUsage).count() == 0


def test_queue_failure_refunds_quota(
    client, make_presentation, n1, auth_headers, storage, queue, db
):
    pid = make_presentation(auth_headers)
    aid = n1(pid, auth_headers).json()["analysisId"]
    storage.objects[f"videos/{aid}"] = 1000
    queue.fail = True
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 500
    assert db.query(AnalysisUsage).count() == 0


def test_deleting_does_not_reset_quota(
    client, make_presentation, n1, start, auth_headers, settings
):
    settings.daily_analysis_limit = 1
    pid = make_presentation(auth_headers)
    assert start(pid, auth_headers).status_code == 202
    client.delete(f"{API}/presentations/{pid}", headers=auth_headers)
    new_pid = make_presentation(auth_headers)
    assert n1(new_pid, auth_headers).status_code == 429


def test_only_today_kst_counts(client, make_presentation, n1, start, auth_headers, settings, db):
    settings.daily_analysis_limit = 1
    pid = make_presentation(auth_headers)
    assert start(pid, auth_headers).status_code == 202
    usage = db.query(AnalysisUsage).one()
    usage.created_at = datetime.now(UTC) - timedelta(days=1, hours=1)
    db.commit()
    assert start(pid, auth_headers).status_code == 202


def test_limit_is_per_user(make_presentation, start, n1, auth_headers, other_headers, settings):
    settings.daily_analysis_limit = 1
    assert start(make_presentation(auth_headers), auth_headers).status_code == 202
    assert start(make_presentation(other_headers), other_headers).status_code == 202


# 11/4 00:00 KST = 11/3 15:00 UTC. now는 11/4 KST 00:30, 03:30, 08:30, 09:30, 15:30.
@pytest.mark.parametrize("now_kst_hour", [0, 3, 8, 9, 15])
def test_used_today_counts_by_kst_day_at_any_hour(db, now_kst_hour):
    """KST 0~9시(UTC로는 전날)에도 오늘 사용량을 센다. SQLite 문자열 비교 회귀 방지."""
    from app.api.v1.analyses import _used_today
    from app.models import User

    user = User(email="q@ycu.ac.kr", password_hash="x", nickname="q")
    db.add(user)
    db.commit()
    day_start = datetime(2026, 11, 3, 15, 0, tzinfo=UTC)
    now = day_start + timedelta(hours=now_kst_hour, minutes=30)
    db.add_all(
        [
            AnalysisUsage(
                user_id=user.id, analysis_id="a0", created_at=day_start - timedelta(seconds=1)
            ),
            AnalysisUsage(user_id=user.id, analysis_id="a1", created_at=day_start),
            AnalysisUsage(
                user_id=user.id, analysis_id="a2", created_at=now - timedelta(minutes=10)
            ),
        ]
    )
    db.commit()
    assert _used_today(db, user.id, now) == 2
