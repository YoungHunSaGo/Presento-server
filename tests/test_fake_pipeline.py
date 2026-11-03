import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.models import Analysis, Presentation, Report, User
from app.schemas.report import ReportResponse
from app.services.media import MediaError
from app.services.pipeline.fake import SAMPLE_REPORT_PATH, run_fake_pipeline
from tests.fakes import FakeProbe

ROOT = Path(__file__).resolve().parent.parent
REPLACED = {"analysisId", "presentationId", "attemptNo", "durationSec"}


def _spec_sample() -> dict:
    spec = (ROOT / "docs" / "api-spec.md").read_text(encoding="utf-8")
    section = spec.split("## 5. 리포트 데이터 구조")[1]
    return json.loads(re.search(r"```json\n(.*?)```", section, re.S).group(1))


@pytest.fixture
def make_analysis(db):
    def _make(status: str = "QUEUED", **kw) -> str:
        user = User(email=f"w{datetime.now().timestamp()}@b.com", password_hash="x", nickname="n")
        db.add(user)
        db.flush()
        p = Presentation(user_id=user.id, title="t", purpose="CLASS")
        db.add(p)
        db.flush()
        a = Analysis(
            presentation_id=p.id,
            attempt_no=3,
            status=status,
            content_type="video/webm",
            video_key="videos/x",
            **kw,
        )
        db.add(a)
        db.commit()
        return a.id

    return _make


def _run(app, storage, settings, aid, probe=None, sleep=None):
    return run_fake_pipeline(
        aid,
        session_factory=app.state.session_factory,
        storage=storage,
        settings=settings,
        probe=probe or FakeProbe(287.43),
        sleep=sleep or (lambda _s: None),
    )


def test_sample_report_file_matches_spec_section_5():
    assert json.loads(SAMPLE_REPORT_PATH.read_text(encoding="utf-8")) == _spec_sample()


def test_spec_sample_passes_report_schema():
    ReportResponse.model_validate(_spec_sample())


def test_stages_follow_spec_order_and_progress(app, storage, settings, make_analysis, db):
    aid = make_analysis()
    seen: list[tuple[str, int]] = []

    def sleep(_s):
        db.expire_all()
        a = db.get(Analysis, aid)
        seen.append((a.stage, a.progress))

    assert _run(app, storage, settings, aid, sleep=sleep) == "completed"
    assert seen == [
        ("EXTRACTING_AUDIO", 0),
        ("EXTRACTING_AUDIO", 5),
        ("TRANSCRIBING", 10),
        ("TRANSCRIBING", 25),
        ("ANALYZING_NONVERBAL", 40),
        ("ANALYZING_NONVERBAL", 52),
        ("ANALYZING_CONTENT", 65),
        ("ANALYZING_CONTENT", 72),
        ("GENERATING_FEEDBACK", 80),
        ("GENERATING_FEEDBACK", 90),
    ]
    progresses = [p for _, p in seen]
    assert progresses == sorted(progresses)


def test_each_stage_takes_configured_time(app, storage, settings, make_analysis):
    settings.fake_stage_seconds = 1.5
    slept: list[float] = []
    _run(app, storage, settings, make_analysis(), sleep=slept.append)
    assert len(slept) == 10
    assert sum(slept) == pytest.approx(7.5)  # 단계당 1.5초 x 5단계


def test_report_is_spec_sample_with_four_real_fields(app, storage, settings, make_analysis, db):
    aid = make_analysis()
    _run(app, storage, settings, aid)
    db.expire_all()
    a = db.get(Analysis, aid)
    report = db.get(Report, aid).data

    assert report["analysisId"] == aid
    assert report["presentationId"] == a.presentation_id
    assert report["attemptNo"] == 3
    assert report["durationSec"] == 287.43
    sample = _spec_sample()
    normalized = ReportResponse.model_validate(sample).model_dump(mode="json", by_alias=True)
    for key in normalized.keys() - REPLACED:
        assert report[key] == normalized[key], key

    assert (a.status, a.progress, a.stage) == ("COMPLETED", 100, None)
    assert a.duration_sec == 287.43
    assert a.overall_score == 82
    assert a.scores == {"DELIVERY": 80, "NONVERBAL": 84, "CONTENT": 81}
    assert (a.filler_count, a.syllables_per_min) == (9, 341)
    assert a.completed_at is not None


@pytest.mark.parametrize("status", ["WAITING_UPLOAD", "COMPLETED", "FAILED", "PROCESSING"])
def test_only_queued_is_claimed(app, storage, settings, make_analysis, status):
    aid = make_analysis(status=status, started_at=datetime.now(UTC))
    assert _run(app, storage, settings, aid) == "skipped"


def test_duplicate_delivery_runs_once(app, storage, settings, make_analysis, db):
    aid = make_analysis()
    assert _run(app, storage, settings, aid) == "completed"
    assert _run(app, storage, settings, aid) == "skipped"
    assert db.query(Report).count() == 1


def test_stale_processing_is_reclaimed(app, storage, settings, make_analysis):
    aid = make_analysis(status="PROCESSING", started_at=datetime.now(UTC) - timedelta(hours=1))
    assert _run(app, storage, settings, aid) == "completed"


@pytest.mark.parametrize(
    ("duration", "error", "code"),
    [
        (600.5, None, "VIDEO_TOO_LONG"),
        (None, None, "CORRUPTED_VIDEO"),
        (30.0, MediaError("moov atom not found"), "CORRUPTED_VIDEO"),
    ],
)
def test_media_failures(app, storage, settings, make_analysis, db, duration, error, code):
    aid = make_analysis()
    probe = FakeProbe(duration)
    probe.error = error
    assert _run(app, storage, settings, aid, probe=probe) == "failed"
    db.expire_all()
    a = db.get(Analysis, aid)
    assert (a.status, a.error_code, a.stage) == ("FAILED", code, None)
    assert a.error_message
    assert db.get(Report, aid) is None


def test_probe_uses_internal_url_and_full_scan(app, storage, settings, make_analysis):
    probe = FakeProbe(10.0)
    _run(app, storage, settings, make_analysis(), probe=probe)
    source, content_type, _timeout, full_scan = probe.calls[0]
    assert source.startswith("http://fake-minio:9000/presento/videos/x")
    assert content_type == "video/webm"
    assert full_scan is True


def test_unexpected_crash_marks_failed(app, storage, settings, make_analysis, db):
    aid = make_analysis()

    def boom(_s):
        raise RuntimeError("disk full")

    assert _run(app, storage, settings, aid, sleep=boom) == "failed"
    db.expire_all()
    assert db.get(Analysis, aid).error_code == "INTERNAL_ERROR"


def test_deleted_mid_run_stops_quietly(app, storage, settings, make_analysis, db):
    aid = make_analysis()
    calls = {"n": 0}

    def delete_on_third_sleep(_s):
        calls["n"] += 1
        if calls["n"] == 3:
            db.query(Analysis).filter_by(id=aid).delete()
            db.commit()

    assert _run(app, storage, settings, aid, sleep=delete_on_third_sleep) == "cancelled"
    assert db.query(Report).count() == 0


def test_pipeline_never_creates_openai_client(app, storage, settings, make_analysis, monkeypatch):
    # 다른 테스트(stt)가 openai를 import하므로 sys.modules 대신 클라이언트 생성을 직접 감시한다.
    import openai

    created: list[str] = []
    for cls in (openai.OpenAI, openai.AsyncOpenAI):
        monkeypatch.setattr(cls, "__init__", lambda *a, _n=cls.__name__, **k: created.append(_n))
    _run(app, storage, settings, make_analysis())
    assert created == []


def test_worker_tasks_are_registered():
    from app.services.queue import TASK_DELETE_OBJECTS, TASK_RUN_ANALYSIS
    from app.workers import tasks  # noqa: F401
    from app.workers.celery_app import celery_app

    assert {TASK_RUN_ANALYSIS, TASK_DELETE_OBJECTS} <= set(celery_app.tasks.keys())
