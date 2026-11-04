import re
from datetime import UTC, datetime, timedelta

import pytest

from app.models import Analysis, Presentation

BASE = "/api/v1/presentations"
KST_ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00$")
P1_KEYS = {
    "id",
    "title",
    "purpose",
    "timeLimitSec",
    "hasSlides",
    "attemptCount",
    "latestScore",
    "createdAt",
}
PDF = {"fileName": "slides.pdf", "contentType": "application/pdf", "fileSize": 3145728}
MB20 = 20 * 1024 * 1024


def _err(res) -> str:
    return res.json()["error"]["code"]


@pytest.fixture
def create(client, auth_headers):
    def _create(headers=None, **overrides) -> dict:
        body = {"title": "캡스톤 중간 발표", "purpose": "CLASS", "timeLimitSec": 300, **overrides}
        res = client.post(BASE, json=body, headers=headers or auth_headers)
        assert res.status_code == 201, res.text
        return res.json()

    return _create


def _add_analysis(db, presentation_id: str, attempt_no: int, status: str, **kw) -> Analysis:
    a = Analysis(presentation_id=presentation_id, attempt_no=attempt_no, status=status, **kw)
    db.add(a)
    db.commit()
    return a


def _completed(db, pid, n, score, **kw):
    return _add_analysis(
        db,
        pid,
        n,
        "COMPLETED",
        overall_score=score,
        scores={"DELIVERY": score - 2, "NONVERBAL": score + 2, "CONTENT": score},
        filler_count=10,
        syllables_per_min=340,
        duration_sec=287.4,
        **kw,
    )


# ---------- 인증·권한: 모든 엔드포인트 ----------

OWNED_ENDPOINTS = [
    ("get", "/{id}", None),
    ("patch", "/{id}", {"title": "남의 발표 수정"}),
    ("delete", "/{id}", None),
    ("post", "/{id}/slides/upload-url", PDF),
    ("get", "/{id}/history", None),
]


@pytest.mark.parametrize(("method", "path", "body"), OWNED_ENDPOINTS)
def test_other_users_presentation_is_forbidden(
    client, create, other_headers, auth_headers, storage, method, path, body
):
    pid = create()["id"]
    kwargs = {"headers": other_headers}
    if body is not None:
        kwargs["json"] = body
    res = client.request(method.upper(), BASE + path.format(id=pid), **kwargs)
    assert res.status_code == 403
    assert _err(res) == "FORBIDDEN"
    assert res.json()["error"]["message"] == "이 발표에 접근할 권한이 없어요."

    # 원래 주인의 발표는 그대로다.
    after = client.get(f"{BASE}/{pid}", headers=auth_headers).json()
    assert after["title"] == "캡스톤 중간 발표"
    assert storage.presigned == []


@pytest.mark.parametrize(("method", "path", "body"), OWNED_ENDPOINTS)
def test_missing_presentation_is_not_found(client, auth_headers, method, path, body):
    kwargs = {"headers": auth_headers}
    if body is not None:
        kwargs["json"] = body
    res = client.request(method.upper(), BASE + path.format(id="p-doesnotexist"), **kwargs)
    assert res.status_code == 404
    assert _err(res) == "NOT_FOUND"


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [("post", "", {"title": "t", "purpose": "CLASS"}), ("get", "", None), *OWNED_ENDPOINTS],
)
def test_all_endpoints_require_auth(client, create, method, path, body):
    pid = create()["id"]
    kwargs = {"json": body} if body is not None else {}
    res = client.request(method.upper(), BASE + path.format(id=pid), **kwargs)
    assert res.status_code == 401
    assert _err(res) == "UNAUTHORIZED"


def test_list_never_includes_other_users_presentations(client, create, other_headers):
    create(title="내 발표")
    create(headers=other_headers, title="남의 발표")
    res = client.get(BASE, headers=other_headers).json()
    assert [p["title"] for p in res["items"]] == ["남의 발표"]
    assert res["total"] == 1


# ---------- P1 ----------


def test_create_returns_spec_shape(create):
    body = create()
    assert set(body) == P1_KEYS
    assert body["id"].startswith("p-")
    assert body["title"] == "캡스톤 중간 발표"
    assert body["purpose"] == "CLASS"
    assert body["timeLimitSec"] == 300
    assert body["hasSlides"] is False
    assert body["attemptCount"] == 0
    assert body["latestScore"] is None
    assert KST_ISO.match(body["createdAt"])


def test_create_without_time_limit(client, auth_headers):
    res = client.post(BASE, json={"title": "면접", "purpose": "INTERVIEW"}, headers=auth_headers)
    assert res.status_code == 201
    assert res.json()["timeLimitSec"] is None


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"title": "t", "purpose": "SPEECH"}, "purpose"),
        ({"title": "   ", "purpose": "CLASS"}, "title"),
        ({"title": "가" * 101, "purpose": "CLASS"}, "title"),
        ({"purpose": "CLASS"}, "title"),
        ({"title": "t"}, "purpose"),
        ({"title": "t", "purpose": "CLASS", "timeLimitSec": 0}, "timeLimitSec"),
        ({"title": "t", "purpose": "CLASS", "timeLimitSec": 601}, "timeLimitSec"),
        ({"title": "t", "purpose": "CLASS", "timeLimitSec": "abc"}, "timeLimitSec"),
    ],
)
def test_create_validation(client, auth_headers, body, field):
    res = client.post(BASE, json=body, headers=auth_headers)
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"
    assert field in {d["field"] for d in res.json()["error"]["details"]}


# ---------- P2 ----------


def test_list_pagination_and_recent_update_order(client, create, auth_headers):
    ids = [create(title=f"발표 {i}")["id"] for i in range(3)]
    # 가장 먼저 만든 발표를 수정하면 맨 앞으로 온다.
    client.patch(f"{BASE}/{ids[0]}", json={"title": "수정됨"}, headers=auth_headers)

    page1 = client.get(BASE, params={"page": 1, "size": 2}, headers=auth_headers).json()
    assert set(page1) == {"items", "page", "size", "total"}
    assert (page1["page"], page1["size"], page1["total"]) == (1, 2, 3)
    assert [p["id"] for p in page1["items"]] == [ids[0], ids[2]]
    assert set(page1["items"][0]) == P1_KEYS

    page2 = client.get(BASE, params={"page": 2, "size": 2}, headers=auth_headers).json()
    assert [p["id"] for p in page2["items"]] == [ids[1]]


def test_list_defaults(client, auth_headers):
    res = client.get(BASE, headers=auth_headers).json()
    assert res == {"items": [], "page": 1, "size": 20, "total": 0}


@pytest.mark.parametrize("params", [{"page": 0}, {"size": 0}, {"size": 101}, {"page": "x"}])
def test_list_invalid_paging(client, auth_headers, params):
    res = client.get(BASE, params=params, headers=auth_headers)
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"


def test_list_attempt_count_and_latest_score(client, create, auth_headers, db):
    pid = create()["id"]
    _completed(db, pid, 1, 64)
    _add_analysis(db, pid, 2, "FAILED")
    _completed(db, pid, 3, 82)
    _add_analysis(db, pid, 4, "PROCESSING")
    item = client.get(BASE, headers=auth_headers).json()["items"][0]
    assert item["attemptCount"] == 4
    assert item["latestScore"] == 82


# ---------- P3 ----------


def test_detail_with_attempts_latest_first(client, create, auth_headers, db):
    pid = create()["id"]
    _completed(db, pid, 1, 64)
    _add_analysis(db, pid, 2, "FAILED")
    res = client.get(f"{BASE}/{pid}", headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == P1_KEYS | {"attempts"}
    assert body["attemptCount"] == 2
    assert body["latestScore"] == 64
    attempts = body["attempts"]
    assert [a["attemptNo"] for a in attempts] == [2, 1]
    assert set(attempts[0]) == {
        "analysisId",
        "attemptNo",
        "status",
        "overallScore",
        "durationSec",
        "createdAt",
    }
    assert attempts[0]["status"] == "FAILED"
    assert attempts[0]["overallScore"] is None
    assert attempts[0]["durationSec"] is None
    assert attempts[1] | {"createdAt": None} == {
        "analysisId": attempts[1]["analysisId"],
        "attemptNo": 1,
        "status": "COMPLETED",
        "overallScore": 64,
        "durationSec": 287.4,
        "createdAt": None,
    }
    assert KST_ISO.match(attempts[1]["createdAt"])


# ---------- P4 ----------


def test_patch_only_sent_fields(client, create, auth_headers):
    pid = create()["id"]
    res = client.patch(f"{BASE}/{pid}", json={"title": "캡스톤 최종 발표"}, headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == P1_KEYS
    assert body["title"] == "캡스톤 최종 발표"
    assert body["purpose"] == "CLASS"
    assert body["timeLimitSec"] == 300


def test_patch_null_time_limit_clears_it(client, create, auth_headers):
    pid = create()["id"]
    res = client.patch(f"{BASE}/{pid}", json={"timeLimitSec": None}, headers=auth_headers)
    assert res.json()["timeLimitSec"] is None


def test_patch_empty_body_is_noop(client, create, auth_headers):
    created = create()
    res = client.patch(f"{BASE}/{created['id']}", json={}, headers=auth_headers)
    assert res.status_code == 200
    assert res.json() == created


@pytest.mark.parametrize(
    "body", [{"title": None}, {"purpose": None}, {"purpose": "X"}, {"title": ""}]
)
def test_patch_validation(client, create, auth_headers, body):
    pid = create()["id"]
    res = client.patch(f"{BASE}/{pid}", json=body, headers=auth_headers)
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"


# ---------- P5 ----------


def test_delete_removes_presentation_analyses_and_files(
    client, create, auth_headers, db, storage, queue
):
    pid = create()["id"]
    a1_id = _completed(db, pid, 1, 70, video_key=f"videos/{pid}-1").id
    _add_analysis(db, pid, 2, "FAILED", video_key=f"videos/{pid}-2")
    p = db.get(Presentation, pid)
    p.slides_key = f"slides/{pid}/x.pdf"
    db.commit()

    res = client.delete(f"{BASE}/{pid}", headers=auth_headers)
    assert res.status_code == 204
    assert res.content == b""
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).status_code == 404

    db.expire_all()
    assert db.get(Analysis, a1_id) is None
    assert db.query(Analysis).count() == 0
    # S3 정리는 워커에게 맡긴다(요청 안에서 지우지 않음).
    assert queue.deleted_keys == {f"videos/{pid}-1", f"videos/{pid}-2", f"slides/{pid}/x.pdf"}
    assert storage.deleted == []


def test_delete_falls_back_to_inline_when_queue_down(
    client, create, auth_headers, db, storage, queue
):
    pid = create()["id"]
    _completed(db, pid, 1, 70, video_key="videos/x")
    queue.fail = True
    res = client.delete(f"{BASE}/{pid}", headers=auth_headers)
    assert res.status_code == 204
    assert storage.deleted == ["videos/x"]


def test_delete_succeeds_even_if_queue_and_storage_fail(
    client, create, auth_headers, db, storage, queue
):
    pid = create()["id"]
    _completed(db, pid, 1, 70, video_key="videos/x")
    queue.fail = True
    storage.fail_delete = True
    res = client.delete(f"{BASE}/{pid}", headers=auth_headers)
    assert res.status_code == 204


# ---------- P6 ----------


def test_slides_upload_url_shape(client, create, auth_headers, storage):
    pid = create()["id"]
    res = client.post(f"{BASE}/{pid}/slides/upload-url", json=PDF, headers=auth_headers)
    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"uploadUrl", "method", "headers", "expiresInSec"}
    assert body["method"] == "PUT"
    assert body["headers"] == {"Content-Type": "application/pdf"}
    assert body["expiresInSec"] == 600
    assert body["uploadUrl"].startswith("https://")

    method, key, content_type, expires = storage.presigned[0]
    assert (method, content_type, expires) == ("PUT", "application/pdf", 600)
    assert key.startswith(f"slides/{pid}/") and key.endswith(".pdf")


def test_slides_exactly_20mb_allowed(client, create, auth_headers):
    pid = create()["id"]
    res = client.post(
        f"{BASE}/{pid}/slides/upload-url", json={**PDF, "fileSize": MB20}, headers=auth_headers
    )
    assert res.status_code == 200


def test_slides_too_large(client, create, auth_headers, storage):
    pid = create()["id"]
    res = client.post(
        f"{BASE}/{pid}/slides/upload-url", json={**PDF, "fileSize": MB20 + 1}, headers=auth_headers
    )
    assert res.status_code == 413
    assert _err(res) == "FILE_TOO_LARGE"
    assert "20MB" in res.json()["error"]["message"]
    assert storage.presigned == []


@pytest.mark.parametrize("content_type", ["image/png", "video/mp4", "application/x-pdf"])
def test_slides_unsupported_type(client, create, auth_headers, content_type):
    pid = create()["id"]
    res = client.post(
        f"{BASE}/{pid}/slides/upload-url",
        json={**PDF, "contentType": content_type},
        headers=auth_headers,
    )
    assert res.status_code == 415
    assert _err(res) == "UNSUPPORTED_FILE_TYPE"


@pytest.mark.parametrize(
    "body",
    [
        {"contentType": "application/pdf", "fileSize": 10},
        {**PDF, "fileSize": 0},
        {**PDF, "fileSize": -1},
        {**PDF, "fileName": ""},
    ],
)
def test_slides_validation(client, create, auth_headers, body):
    pid = create()["id"]
    res = client.post(f"{BASE}/{pid}/slides/upload-url", json=body, headers=auth_headers)
    assert res.status_code == 400
    assert _err(res) == "VALIDATION_ERROR"


def _issue_slides(client, pid, headers, storage) -> str:
    client.post(f"{BASE}/{pid}/slides/upload-url", json=PDF, headers=headers)
    return storage.presigned[-1][1]


def test_has_slides_after_upload_confirmed(client, create, auth_headers, storage):
    pid = create()["id"]
    key = _issue_slides(client, pid, auth_headers, storage)
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).json()["hasSlides"] is False

    storage.objects[key] = 3145728  # 브라우저가 PUT 완료
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).json()["hasSlides"] is True
    assert client.get(BASE, headers=auth_headers).json()["items"][0]["hasSlides"] is True


def test_has_slides_confirmed_from_list(client, create, auth_headers, storage):
    pid = create()["id"]
    key = _issue_slides(client, pid, auth_headers, storage)
    storage.objects[key] = 100
    assert client.get(BASE, headers=auth_headers).json()["items"][0]["hasSlides"] is True


def test_oversized_upload_is_discarded(client, create, auth_headers, storage):
    pid = create()["id"]
    key = _issue_slides(client, pid, auth_headers, storage)
    storage.objects[key] = MB20 + 1  # presigned PUT은 크기를 막지 못한다
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).json()["hasSlides"] is False
    assert key in storage.deleted


def test_reupload_overwrites_same_key(client, create, auth_headers, storage):
    pid = create()["id"]
    first = _issue_slides(client, pid, auth_headers, storage)
    storage.objects[first] = 100
    client.get(f"{BASE}/{pid}", headers=auth_headers)  # hasSlides 확정

    second = _issue_slides(client, pid, auth_headers, storage)
    assert second == first  # 같은 키로 덮어쓰는 URL
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).json()["hasSlides"] is True
    assert storage.deleted == []


def test_no_head_after_has_slides_is_true(client, create, auth_headers, storage):
    pid = create()["id"]
    key = _issue_slides(client, pid, auth_headers, storage)
    storage.objects[key] = 100
    assert client.get(f"{BASE}/{pid}", headers=auth_headers).json()["hasSlides"] is True
    storage.head_calls.clear()

    client.get(f"{BASE}/{pid}", headers=auth_headers)
    client.get(BASE, headers=auth_headers)
    client.patch(f"{BASE}/{pid}", json={"title": "바뀜"}, headers=auth_headers)
    _issue_slides(client, pid, auth_headers, storage)  # 다시 올리기
    client.get(f"{BASE}/{pid}", headers=auth_headers)
    assert storage.head_calls == []


@pytest.mark.parametrize(("value", "status_code"), [(600, 201), (601, 400), (1, 201)])
def test_time_limit_range_1_to_600(client, auth_headers, value, status_code):
    res = client.post(
        BASE, json={"title": "t", "purpose": "CLASS", "timeLimitSec": value}, headers=auth_headers
    )
    assert res.status_code == status_code


def test_storage_failure_does_not_break_reads(client, create, auth_headers, storage):
    pid = create()["id"]
    _issue_slides(client, pid, auth_headers, storage)
    storage.fail_head = True
    res = client.get(f"{BASE}/{pid}", headers=auth_headers)
    assert res.status_code == 200
    assert res.json()["hasSlides"] is False
    assert client.get(BASE, headers=auth_headers).status_code == 200


def test_stale_pending_upload_is_dropped(client, create, auth_headers, storage, db):
    pid = create()["id"]
    _issue_slides(client, pid, auth_headers, storage)
    p = db.get(Presentation, pid)
    p.slides_pending_at = datetime.now(UTC) - timedelta(hours=2)
    db.commit()

    client.get(f"{BASE}/{pid}", headers=auth_headers)
    db.expire_all()
    assert db.get(Presentation, pid).slides_pending_key is None


# ---------- P7 ----------


def test_history_completed_only_oldest_first(client, create, auth_headers, db):
    pid = create()["id"]
    _completed(db, pid, 3, 82)
    _completed(db, pid, 1, 64)
    _add_analysis(db, pid, 2, "FAILED")
    _add_analysis(db, pid, 4, "PROCESSING")

    res = client.get(f"{BASE}/{pid}/history", headers=auth_headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert [i["attemptNo"] for i in items] == [1, 3]
    first = items[0]
    assert set(first) == {
        "attemptNo",
        "analysisId",
        "createdAt",
        "overallScore",
        "scores",
        "fillerCount",
        "syllablesPerMin",
    }
    assert first["overallScore"] == 64
    assert first["scores"] == {"DELIVERY": 62, "NONVERBAL": 66, "CONTENT": 64}
    assert first["fillerCount"] == 10
    assert first["syllablesPerMin"] == 340
    assert KST_ISO.match(first["createdAt"])


def test_history_empty(client, create, auth_headers):
    pid = create()["id"]
    res = client.get(f"{BASE}/{pid}/history", headers=auth_headers)
    assert res.json() == {"items": []}


@pytest.mark.parametrize("limit", [1, 600])
def test_create_time_limit_bounds_ok(client, auth_headers, limit):
    # 명세 P1: timeLimitSec은 1~600
    res = client.post(
        BASE, json={"title": "t", "purpose": "CLASS", "timeLimitSec": limit}, headers=auth_headers
    )
    assert res.status_code == 201
    assert res.json()["timeLimitSec"] == limit
