import re
from datetime import UTC, datetime, timedelta

import pytest

from app.models import Analysis, AnalysisUsage, Report
from app.services.media import MediaError
from app.services.pipeline.fake import run_fake_pipeline

API = "/api/v1"
KST_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$")
VIDEO = {
    "fileName": "take3.webm",
    "contentType": "video/webm",
    "fileSize": 84213760,
    "source": "RECORDING",
}
MB500 = 500 * 1024 * 1024


def _err(res) -> str:
    return res.json()["error"]["code"]


@pytest.fixture
def presentation(client, auth_headers) -> str:
    res = client.post(
        f"{API}/presentations",
        json={"title": "캡스톤 중간 발표", "purpose": "CLASS", "timeLimitSec": 300},
        headers=auth_headers,
    )
    return res.json()["id"]


@pytest.fixture
def new_analysis(client, auth_headers, presentation):
    def _new(**overrides) -> dict:
        res = client.post(
            f"{API}/presentations/{presentation}/analyses",
            json={**VIDEO, **overrides},
            headers=auth_headers,
        )
        assert res.status_code == 201, res.text
        return res.json()

    return _new


@pytest.fixture
def uploaded(new_analysis, storage):
    """N1 + 브라우저 PUT 완료 상태."""

    def _uploaded(size: int = 1_000_000) -> str:
        aid = new_analysis()["analysisId"]
        storage.objects[f"videos/{aid}"] = size
        return aid

    return _uploaded


def _run_worker(app, storage, settings, analysis_id, probe=None):
    from tests.fakes import FakeProbe

    return run_fake_pipeline(
        analysis_id,
        session_factory=app.state.session_factory,
        storage=storage,
        settings=settings,
        probe=probe or FakeProbe(30.0),
        sleep=lambda _s: None,
    )


# ---------- 인증·권한: 모든 엔드포인트 ----------

ANALYSIS_ENDPOINTS = [
    ("post", "/analyses/{id}/start"),
    ("get", "/analyses/{id}"),
    ("get", "/analyses/{id}/report"),
    ("get", "/analyses/{id}/video-url"),
    ("delete", "/analyses/{id}"),
]


@pytest.mark.parametrize(("method", "path"), ANALYSIS_ENDPOINTS)
def test_other_users_analysis_is_forbidden(
    client, uploaded, other_headers, auth_headers, queue, method, path
):
    aid = uploaded()
    res = client.request(method.upper(), API + path.format(id=aid), headers=other_headers)
    assert res.status_code == 403
    assert _err(res) == "FORBIDDEN"
    assert res.json()["error"]["message"] == "이 분석 회차에 접근할 권한이 없어요."
    # 아무것도 바뀌지 않았다.
    after = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert after["status"] == "WAITING_UPLOAD"
    assert queue.analyses == [] and queue.deletes == []


def test_n1_on_other_users_presentation_is_forbidden(client, presentation, other_headers, db):
    res = client.post(
        f"{API}/presentations/{presentation}/analyses", json=VIDEO, headers=other_headers
    )
    assert res.status_code == 403
    assert _err(res) == "FORBIDDEN"
    assert db.query(Analysis).count() == 0
    assert db.query(AnalysisUsage).count() == 0


@pytest.mark.parametrize(("method", "path"), ANALYSIS_ENDPOINTS)
def test_missing_analysis_is_not_found(client, auth_headers, method, path):
    res = client.request(method.upper(), API + path.format(id="a-nope"), headers=auth_headers)
    assert res.status_code == 404
    assert _err(res) == "NOT_FOUND"


@pytest.mark.parametrize(
    ("method", "path"), [("post", "/presentations/{pid}/analyses"), *ANALYSIS_ENDPOINTS]
)
def test_all_analysis_endpoints_require_auth(client, uploaded, presentation, method, path):
    aid = uploaded()
    url = API + path.format(id=aid, pid=presentation)
    res = client.request(method.upper(), url, json=VIDEO if "pid" in path else None)
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


# ---------- N1 ----------


def test_n1_spec_shape(new_analysis, storage):
    body = new_analysis()
    assert set(body) == {"analysisId", "attemptNo", "status", "upload"}
    assert body["analysisId"].startswith("a-")
    assert body["attemptNo"] == 1
    assert body["status"] == "WAITING_UPLOAD"
    assert body["upload"]["method"] == "PUT"
    assert body["upload"]["headers"] == {"Content-Type": "video/webm"}
    assert body["upload"]["expiresInSec"] == 900
    assert storage.presigned[-1] == ("PUT", f"videos/{body['analysisId']}", "video/webm", 900)


def test_n1_attempt_numbers_increase_and_are_not_reused(client, new_analysis, auth_headers):
    first = new_analysis()
    second = new_analysis()
    assert (first["attemptNo"], second["attemptNo"]) == (1, 2)
    client.delete(f"{API}/analyses/{second['analysisId']}", headers=auth_headers)
    assert new_analysis()["attemptNo"] == 3


@pytest.mark.parametrize(
    ("content_type", "signed"),
    [
        ("video/mp4", "video/mp4"),
        ("video/quicktime", "video/quicktime"),
        ("video/webm;codecs=vp9,opus", "video/webm"),
        ("VIDEO/MP4", "video/mp4"),
    ],
)
def test_n1_accepts_allowed_types(new_analysis, content_type, signed):
    body = new_analysis(contentType=content_type)
    assert body["upload"]["headers"] == {"Content-Type": signed}


@pytest.mark.parametrize("content_type", ["video/x-msvideo", "image/png", "application/pdf"])
def test_n1_rejects_other_types(client, presentation, auth_headers, content_type):
    res = client.post(
        f"{API}/presentations/{presentation}/analyses",
        json={**VIDEO, "contentType": content_type},
        headers=auth_headers,
    )
    assert res.status_code == 415
    assert _err(res) == "UNSUPPORTED_FILE_TYPE"


def test_n1_file_size_limit(client, presentation, auth_headers, new_analysis):
    assert new_analysis(fileSize=MB500)["status"] == "WAITING_UPLOAD"
    res = client.post(
        f"{API}/presentations/{presentation}/analyses",
        json={**VIDEO, "fileSize": MB500 + 1},
        headers=auth_headers,
    )
    assert res.status_code == 413
    assert _err(res) == "FILE_TOO_LARGE"
    assert res.json()["error"]["message"] == "영상은 500MB 이하만 업로드할 수 있어요."


@pytest.mark.parametrize(
    "override", [{"source": "CAMERA"}, {"fileSize": "100"}, {"fileSize": 0}, {"fileName": ""}]
)
def test_n1_validation(client, presentation, auth_headers, override):
    res = client.post(
        f"{API}/presentations/{presentation}/analyses",
        json={**VIDEO, **override},
        headers=auth_headers,
    )
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"


def test_n1_daily_limit(client, presentation, auth_headers, new_analysis, settings, db):
    settings.daily_analysis_limit = 2
    new_analysis()
    new_analysis()
    res = client.post(
        f"{API}/presentations/{presentation}/analyses", json=VIDEO, headers=auth_headers
    )
    assert res.status_code == 429
    assert _err(res) == "RATE_LIMITED"
    assert "2회" in res.json()["error"]["message"]
    assert db.query(Analysis).count() == 2


def test_n1_daily_limit_not_reset_by_deleting(
    client, presentation, auth_headers, new_analysis, settings
):
    settings.daily_analysis_limit = 1
    new_analysis()
    client.delete(f"{API}/presentations/{presentation}", headers=auth_headers)
    pid = client.post(
        f"{API}/presentations", json={"title": "새 발표", "purpose": "ETC"}, headers=auth_headers
    ).json()["id"]
    res = client.post(f"{API}/presentations/{pid}/analyses", json=VIDEO, headers=auth_headers)
    assert res.status_code == 429


def test_n1_daily_limit_counts_only_today_kst(
    client, presentation, auth_headers, new_analysis, settings, db, signup
):
    settings.daily_analysis_limit = 1
    aid = new_analysis()["analysisId"]
    usage = db.query(AnalysisUsage).filter_by(analysis_id=aid).one()
    usage.created_at = datetime.now(UTC) - timedelta(days=1, hours=1)
    db.commit()
    assert new_analysis()["attemptNo"] == 2


def test_n1_limit_is_per_user(client, presentation, new_analysis, settings, other_headers):
    settings.daily_analysis_limit = 1
    new_analysis()
    pid = client.post(
        f"{API}/presentations", json={"title": "남", "purpose": "ETC"}, headers=other_headers
    ).json()["id"]
    res = client.post(f"{API}/presentations/{pid}/analyses", json=VIDEO, headers=other_headers)
    assert res.status_code == 201


def test_n1_moves_presentation_to_top(client, auth_headers, presentation, new_analysis):
    client.post(f"{API}/presentations", json={"title": "B", "purpose": "ETC"}, headers=auth_headers)
    new_analysis()
    items = client.get(f"{API}/presentations", headers=auth_headers).json()["items"]
    assert items[0]["id"] == presentation
    assert items[0]["attemptCount"] == 1


# ---------- N2 ----------


def test_n2_upload_not_found(client, new_analysis, auth_headers, queue):
    aid = new_analysis()["analysisId"]
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 409
    assert _err(res) == "UPLOAD_NOT_FOUND"
    assert client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()["status"] == (
        "WAITING_UPLOAD"
    )
    assert queue.analyses == []


def test_n2_starts_once(client, uploaded, auth_headers, queue, probe):
    aid = uploaded()
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 202
    assert res.json() == {"analysisId": aid, "status": "QUEUED"}
    assert queue.analyses == [aid]
    # 길이 확인은 서버 내부 주소로, 빠른 모드(full_scan=False)로 한다.
    source, content_type, _timeout, full_scan = probe.calls[0]
    assert source.startswith("http://fake-minio:9000/") and content_type == "video/webm"
    assert full_scan is False

    again = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert again.status_code == 202
    assert again.json() == {"analysisId": aid, "status": "QUEUED"}
    assert queue.analyses == [aid]  # 작업이 두 번 들어가지 않는다


def test_n2_after_completion_returns_current_status(
    client, app, uploaded, auth_headers, queue, storage, settings
):
    aid = uploaded()
    client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    _run_worker(app, storage, settings, aid)
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.json()["status"] == "COMPLETED"
    assert queue.analyses == [aid]


def test_n2_uploaded_file_too_large(client, uploaded, auth_headers, storage, queue):
    aid = uploaded(size=MB500 + 1)
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 413
    assert _err(res) == "FILE_TOO_LARGE"
    assert queue.deleted_keys == {f"videos/{aid}"}  # 정리는 워커가 한다
    assert queue.analyses == []


def test_n2_video_too_long(client, uploaded, auth_headers, probe, queue):
    probe.duration = 600.01
    aid = uploaded()
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 422
    assert _err(res) == "VIDEO_TOO_LONG"
    status = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert status["status"] == "FAILED"
    assert status["error"]["code"] == "VIDEO_TOO_LONG"
    assert queue.analyses == []


def test_n2_exactly_ten_minutes_ok(client, uploaded, auth_headers, probe):
    probe.duration = 600.0
    aid = uploaded()
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 202


@pytest.mark.parametrize("outcome", ["unknown", "error"])
def test_n2_unknown_duration_defers_to_worker(client, uploaded, auth_headers, probe, outcome):
    if outcome == "unknown":
        probe.duration = None  # 헤더에 길이가 없는 webm
    else:
        probe.error = MediaError("bad")
    aid = uploaded()
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 202


def test_n2_queue_failure_rolls_back(client, uploaded, auth_headers, queue):
    aid = uploaded()
    queue.fail = True
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 500
    assert _err(res) == "INTERNAL_ERROR"
    assert client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()["status"] == (
        "WAITING_UPLOAD"
    )
    queue.fail = False
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 202
    assert queue.analyses == [aid]


def test_n2_storage_failure(client, uploaded, auth_headers, storage):
    aid = uploaded()
    storage.fail_head = True
    res = client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    assert res.status_code == 500
    assert _err(res) == "INTERNAL_ERROR"


# ---------- N3 ----------


def test_n3_shape_waiting(client, new_analysis, auth_headers, presentation):
    aid = new_analysis()["analysisId"]
    body = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert set(body) == {
        "analysisId",
        "presentationId",
        "attemptNo",
        "status",
        "stage",
        "progress",
        "error",
        "createdAt",
        "completedAt",
    }
    assert body["presentationId"] == presentation
    assert (body["status"], body["stage"], body["progress"]) == ("WAITING_UPLOAD", None, 0)
    assert body["error"] is None and body["completedAt"] is None
    assert KST_ISO.match(body["createdAt"])


def test_n3_processing_shows_stage(client, uploaded, auth_headers, db):
    aid = uploaded()
    a = db.get(Analysis, aid)
    a.status, a.stage, a.progress = "PROCESSING", "TRANSCRIBING", 35
    db.commit()
    body = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert (body["status"], body["stage"], body["progress"]) == ("PROCESSING", "TRANSCRIBING", 35)


def test_n3_failed_shows_error(client, uploaded, auth_headers, db):
    aid = uploaded()
    a = db.get(Analysis, aid)
    a.status, a.error_code, a.error_message = (
        "FAILED",
        "NO_SPEECH_DETECTED",
        "영상에서 목소리를 찾지 못했어요.",
    )
    db.commit()
    body = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert body["error"] == {
        "code": "NO_SPEECH_DETECTED",
        "message": "영상에서 목소리를 찾지 못했어요.",
    }
    assert body["stage"] is None


# ---------- N4 ----------


def test_n4_not_ready(client, uploaded, auth_headers):
    aid = uploaded()
    res = client.get(f"{API}/analyses/{aid}/report", headers=auth_headers)
    assert res.status_code == 409
    assert _err(res) == "ANALYSIS_NOT_READY"


def test_full_flow_n1_to_n5(client, app, uploaded, auth_headers, storage, settings, presentation):
    aid = uploaded()
    assert client.post(f"{API}/analyses/{aid}/start", headers=auth_headers).status_code == 202
    assert _run_worker(app, storage, settings, aid) == "completed"

    status = client.get(f"{API}/analyses/{aid}", headers=auth_headers).json()
    assert (status["status"], status["progress"], status["stage"]) == ("COMPLETED", 100, None)
    assert KST_ISO.match(status["completedAt"])

    report = client.get(f"{API}/analyses/{aid}/report", headers=auth_headers)
    assert report.status_code == 200
    body = report.json()
    assert body["analysisId"] == aid
    assert body["presentationId"] == presentation
    assert body["attemptNo"] == 1
    assert body["durationSec"] == 30.0

    video = client.get(f"{API}/analyses/{aid}/video-url", headers=auth_headers)
    assert video.status_code == 200
    assert set(video.json()) == {"videoUrl", "expiresInSec"}
    assert video.json()["expiresInSec"] == 3600

    # P3·P7에 점수 요약이 반영된다.
    detail = client.get(f"{API}/presentations/{presentation}", headers=auth_headers).json()
    assert detail["latestScore"] == 82
    assert detail["attempts"][0]["durationSec"] == 30.0
    history = client.get(f"{API}/presentations/{presentation}/history", headers=auth_headers)
    assert history.json()["items"][0]["scores"] == {"DELIVERY": 80, "NONVERBAL": 84, "CONTENT": 81}


# ---------- N5 ----------


def test_n5_before_upload(client, new_analysis, auth_headers):
    aid = new_analysis()["analysisId"]
    res = client.get(f"{API}/analyses/{aid}/video-url", headers=auth_headers)
    assert res.status_code == 409
    assert _err(res) == "UPLOAD_NOT_FOUND"


# ---------- N6 ----------


def test_n6_deletes_analysis_report_and_video(
    client, app, uploaded, auth_headers, storage, settings, queue, db
):
    aid = uploaded()
    client.post(f"{API}/analyses/{aid}/start", headers=auth_headers)
    _run_worker(app, storage, settings, aid)

    res = client.delete(f"{API}/analyses/{aid}", headers=auth_headers)
    assert res.status_code == 204
    assert res.content == b""
    assert client.get(f"{API}/analyses/{aid}", headers=auth_headers).status_code == 404
    db.expire_all()
    assert db.get(Report, aid) is None
    assert queue.deleted_keys == {f"videos/{aid}"}
    assert storage.deleted == []  # 삭제는 워커 몫
