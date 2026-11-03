"""마일스톤 3 리뷰에서 나온 시나리오 회귀 테스트."""

from datetime import UTC, datetime, timedelta

import pytest

from app.models import Analysis, Presentation

BASE = "/api/v1/presentations"
PDF = {"fileName": "slides.pdf", "contentType": "application/pdf", "fileSize": 1024}


@pytest.fixture
def create(client, auth_headers):
    def _create(title: str = "발표") -> str:
        res = client.post(BASE, json={"title": title, "purpose": "CLASS"}, headers=auth_headers)
        assert res.status_code == 201
        return res.json()["id"]

    return _create


def _order(client, headers) -> list[str]:
    return [p["title"] for p in client.get(BASE, headers=headers).json()["items"]]


def _issue(client, pid, headers, storage) -> str:
    assert (
        client.post(f"{BASE}/{pid}/slides/upload-url", json=PDF, headers=headers).status_code == 200
    )
    return storage.presigned[-1][1]


def test_huge_page_is_validation_error_not_500(client, auth_headers):
    res = client.get(BASE, params={"page": 10**20}, headers=auth_headers)
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"
    res = client.get(BASE, params={"page": 10_001}, headers=auth_headers)
    assert res.status_code == 400


def test_reissuing_url_keeps_already_uploaded_slides(client, create, auth_headers, storage):
    pid = create()
    first = _issue(client, pid, auth_headers, storage)
    storage.objects[first] = 1024  # 업로드는 끝났지만 아직 조회로 확인되지 않은 상태
    _issue(client, pid, auth_headers, storage)  # 두 번째 URL을 받고 업로드하지 않음

    body = client.get(f"{BASE}/{pid}", headers=auth_headers).json()
    assert body["hasSlides"] is True
    assert first not in storage.deleted


def test_issuing_url_does_not_reorder_list(client, create, auth_headers, storage):
    a = create("A")
    create("B")
    assert _order(client, auth_headers) == ["B", "A"]
    _issue(client, a, auth_headers, storage)
    assert _order(client, auth_headers) == ["B", "A"]


def test_dropping_stale_pending_does_not_reorder_list(client, create, auth_headers, storage, db):
    a = create("A")
    create("B")
    _issue(client, a, auth_headers, storage)
    p = db.get(Presentation, a)
    p.slides_pending_at = datetime.now(UTC) - timedelta(hours=2)
    db.commit()

    assert _order(client, auth_headers) == ["B", "A"]  # 이 GET에서 대기 키가 정리된다
    db.expire_all()
    assert db.get(Presentation, a).slides_pending_key is None
    assert _order(client, auth_headers) == ["B", "A"]


def test_confirmed_slides_upload_moves_to_top(client, create, auth_headers, storage):
    a = create("A")
    create("B")
    key = _issue(client, a, auth_headers, storage)
    storage.objects[key] = 1024
    client.get(f"{BASE}/{a}", headers=auth_headers)
    assert _order(client, auth_headers) == ["A", "B"]


def test_noop_patch_does_not_reorder(client, create, auth_headers):
    a = create("A")
    create("B")
    client.patch(f"{BASE}/{a}", json={"title": "A", "purpose": "CLASS"}, headers=auth_headers)
    assert _order(client, auth_headers) == ["B", "A"]


def test_head_checks_are_capped_per_request(client, create, auth_headers, storage):
    for i in range(8):
        pid = create(f"P{i}")
        _issue(client, pid, auth_headers, storage)
    calls = []
    original = storage.head
    storage.head = lambda key: calls.append(key) or original(key)
    client.get(BASE, headers=auth_headers)
    assert len(calls) == 5


@pytest.mark.parametrize("field", ["purpose", "title"])
def test_patch_null_points_to_field(client, create, auth_headers, field):
    pid = create()
    res = client.patch(f"{BASE}/{pid}", json={field: None}, headers=auth_headers)
    assert res.status_code == 400
    assert res.json()["error"]["details"] == [{"field": field, "reason": "값을 비울 수 없어요."}]


@pytest.mark.parametrize("value", [True, "300", 300.5])
def test_time_limit_must_be_real_integer(client, auth_headers, value):
    res = client.post(
        BASE, json={"title": "t", "purpose": "CLASS", "timeLimitSec": value}, headers=auth_headers
    )
    assert res.status_code == 400
    assert res.json()["error"]["details"][0]["field"] == "timeLimitSec"


def test_file_size_must_be_real_integer(client, create, auth_headers):
    pid = create()
    res = client.post(
        f"{BASE}/{pid}/slides/upload-url", json={**PDF, "fileSize": "1024"}, headers=auth_headers
    )
    assert res.status_code == 400


def test_history_skips_completed_without_summary(client, create, auth_headers, db):
    pid = create()
    db.add(Analysis(presentation_id=pid, attempt_no=1, status="COMPLETED", overall_score=70))
    db.commit()
    res = client.get(f"{BASE}/{pid}/history", headers=auth_headers)
    assert res.json() == {"items": []}
