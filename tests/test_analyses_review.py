"""마일스톤 4 리뷰에서 나온 시나리오 회귀 테스트."""

import logging
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from app.models import Analysis
from app.services import media
from app.services.media import MediaError, ProbeUnavailable, probe_duration
from app.services.pipeline.fake import run_fake_pipeline
from tests.fakes import FakeProbe

API = "/api/v1"
VIDEO = {"fileName": "t.webm", "contentType": "video/webm", "fileSize": 1000, "source": "UPLOAD"}
SIGNED = "http://minio:9000/presento/videos/a-1?X-Amz-Credential=AKIA/x&X-Amz-Signature=deadbeef"


@pytest.fixture
def uploaded(client, auth_headers, storage):
    pid = client.post(
        f"{API}/presentations", json={"title": "t", "purpose": "CLASS"}, headers=auth_headers
    ).json()["id"]

    def _make() -> str:
        aid = client.post(
            f"{API}/presentations/{pid}/analyses", json=VIDEO, headers=auth_headers
        ).json()["analysisId"]
        storage.objects[f"videos/{aid}"] = 1000
        return aid

    return _make


# ---------- MEDIUM-1: 멈춘 회차는 N3에서 FAILED로 끝난다 ----------


@pytest.mark.parametrize(
    ("status", "field", "age", "expected"),
    [
        ("PROCESSING", "started_at", timedelta(minutes=21), "FAILED"),
        ("PROCESSING", "started_at", timedelta(minutes=5), "PROCESSING"),
        ("QUEUED", "queued_at", timedelta(minutes=31), "FAILED"),
        ("QUEUED", "queued_at", timedelta(minutes=10), "QUEUED"),
    ],
)
def test_n3_ends_stuck_analyses(client, uploaded, auth_headers, db, status, field, age, expected):
    aid = uploaded()
    a = db.get(Analysis, aid)
    a.status = status
    a.stage = "TRANSCRIBING" if status == "PROCESSING" else None
    setattr(a, field, datetime.now(UTC) - age)
    db.commit()

    body = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert body["status"] == expected
    if expected == "FAILED":
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["message"] == "분석이 제시간에 끝나지 않았어요. 다시 시도해 주세요."
        assert body["stage"] is None


def test_late_worker_cannot_revive_expired_analysis(
    client, app, uploaded, auth_headers, db, storage, settings
):
    aid = uploaded()
    a = db.get(Analysis, aid)
    a.status, a.queued_at = "QUEUED", datetime.now(UTC) - timedelta(hours=1)
    db.commit()
    client.get(f"{API}/analyses/{aid}", headers=auth_headers)  # FAILED로 정리
    result = run_fake_pipeline(
        aid,
        session_factory=app.state.session_factory,
        storage=storage,
        settings=settings,
        probe=FakeProbe(10.0),
        sleep=lambda _s: None,
    )
    assert result == "skipped"


# ---------- MEDIUM-2: 서버 문제는 CORRUPTED_VIDEO가 아니다 ----------


def test_worker_maps_probe_unavailable_to_internal_error(app, uploaded, db, storage, settings):
    aid = uploaded()
    a = db.get(Analysis, aid)
    a.status = "QUEUED"
    db.commit()
    probe = FakeProbe()
    probe.error = ProbeUnavailable("Connection refused")
    run_fake_pipeline(
        aid,
        session_factory=app.state.session_factory,
        storage=storage,
        settings=settings,
        probe=probe,
        sleep=lambda _s: None,
    )
    db.expire_all()
    assert db.get(Analysis, aid).error_code == "INTERNAL_ERROR"


def test_n2_proceeds_when_probe_unavailable(client, uploaded, auth_headers, probe, queue):
    probe.error = ProbeUnavailable("timeout")
    aid = uploaded()
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 202
    assert queue.analyses == [aid]


def _install(monkeypatch, result):
    def fake_run(args, timeout):
        if isinstance(result, Exception):
            raise result
        return subprocess.CompletedProcess(args, *result)

    monkeypatch.setattr(media, "_run", fake_run)


@pytest.mark.parametrize(
    "stderr",
    [
        f"{SIGNED}: Connection refused",
        "Failed to resolve hostname minio: Name or service not known",
        f"{SIGNED}: Server returned 403 Forbidden (access denied)",
        "Connection timed out",
    ],
)
def test_network_errors_are_probe_unavailable(monkeypatch, stderr):
    _install(monkeypatch, (1, "", stderr))
    with pytest.raises(ProbeUnavailable) as exc:
        probe_duration(SIGNED, "video/mp4")
    assert "X-Amz-Signature" not in str(exc.value)


def test_real_corruption_is_media_error_without_url(monkeypatch):
    _install(monkeypatch, (1, "", f"{SIGNED}: Invalid data found when processing input"))
    with pytest.raises(MediaError) as exc:
        probe_duration(SIGNED, "video/mp4")
    assert "X-Amz-Signature" not in str(exc.value)
    assert "<url>" in str(exc.value)


@pytest.mark.parametrize(
    "exc", [FileNotFoundError("ffprobe"), subprocess.TimeoutExpired(["ffprobe", SIGNED], 5)]
)
def test_infra_failures_raise_and_never_log_signed_url(monkeypatch, caplog, exc):
    _install(monkeypatch, exc)
    with caplog.at_level(logging.DEBUG), pytest.raises(ProbeUnavailable):
        probe_duration(SIGNED, "video/mp4")
    assert "X-Amz-Signature" not in caplog.text
    assert all(r.exc_info is None for r in caplog.records)


# ---------- LOW: VIDEO_TOO_LONG 정리와 일관성 ----------


def test_video_too_long_deletes_video_and_repeats_422(client, uploaded, auth_headers, probe, queue):
    probe.duration = 700.0
    aid = uploaded()
    first = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert first.status_code == 422
    assert queue.deleted_keys == {f"videos/{aid}"}

    second = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert second.status_code == 422
    assert second.json()["error"]["code"] == "VIDEO_TOO_LONG"
    assert queue.analyses == []


def test_oversized_upload_cleanup_goes_through_worker(
    client, uploaded, auth_headers, storage, queue
):
    aid = uploaded()
    storage.objects[f"videos/{aid}"] = 500 * 1024 * 1024 + 1
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 413
    assert queue.deleted_keys == {f"videos/{aid}"}
    assert storage.deleted == []
