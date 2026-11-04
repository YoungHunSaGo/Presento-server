"""실제 파이프라인(워커). 외부 호출(OpenAI·ffmpeg·MediaPipe·S3)은 전부 가짜로 바꾼다."""

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.models import Analysis, Presentation, Report, User
from app.models.enums import ScoreCategory
from app.schemas.report import Feedback, FeedbackType, Severity
from app.services.media import MediaError
from app.services.pipeline import audio, runner
from app.services.pipeline.content import (
    ContentAnalysis,
    ContentIssue,
    ContentServiceError,
    IssueType,
)
from app.services.pipeline.feedback import FeedbackResult, FeedbackServiceError, fallback_overall
from app.services.pipeline.nonverbal import FrameObs, NonverbalUnavailable, PoseObs
from app.services.pipeline.report import validate_report
from app.services.pipeline.stt import NoSpeechDetected, SttServiceError, normalize
from tests.fakes import FakeProbe

WHISPER = json.loads((Path(__file__).parent / "fixtures" / "whisper_verbose_ko.json").read_text())
VIDEO_KEY = "videos/x"


def pose() -> PoseObs:
    return PoseObs(
        nose=(0.5, 0.29), left_shoulder=(0.6, 0.45), right_shoulder=(0.4, 0.45),
        left_wrist=(0.62, 0.65), right_wrist=(0.38, 0.65),
    )  # fmt: skip


class Steps:
    """단계 함수 가짜. 호출 기록과, 호출될 때의 stage/progress를 남긴다."""

    def __init__(self, session_factory, aid_ref):
        self.sf = session_factory
        self.aid_ref = aid_ref
        self.calls: list[str] = []
        self.stages: list[tuple[str, int]] = []
        self.raise_at: dict[str, Exception] = {}
        self.content_inputs = []
        self.feedback_inputs = []
        self.slides_paths: list[Path] = []
        self.tmp_dirs: set[Path] = set()

    def _record(self, name):
        self.calls.append(name)
        with self.sf() as db:
            a = db.get(Analysis, self.aid_ref[0])
            if a is not None:
                self.stages.append((a.stage, a.progress))
        if name in self.raise_at:
            raise self.raise_at[name]

    def extract_audio(self, video, dest, ctype, **kw):
        self.tmp_dirs.add(dest.parent)
        self._record("extract_audio")
        dest.write_bytes(b"mp3")
        return dest

    def transcribe(self, path, client):
        self._record("transcribe")
        return normalize(WHISPER)

    def observe_video(self, video, ctype, face, pose_path, **kw):
        self.video_kwargs = kw
        self._record("observe_video")
        return [FrameObs(t=float(i), yaw=0.0, pitch=0.0, pose=pose()) for i in range(15)]

    def extract_slides(self, path):
        self.slides_paths.append(path)
        self._record("extract_slides")
        return ["표지", "결론"]

    def analyze_content(self, data, client, settings):
        self.content_inputs.append(data)
        self._record("analyze_content")
        return ContentAnalysis(
            sections=[], key_message="핵심이에요.",
            issues=[ContentIssue(IssueType.CLARITY, ["s-001"], 0.0, 6.4, "모호해요.")],
            slide_coverage=None if data.slides is None else [],
            expected_questions=[], usage=None,
        )  # fmt: skip

    def generate_feedback(self, data, client, settings):
        self.feedback_inputs.append(data)
        self._record("generate_feedback")
        fb = Feedback(
            id="f-01", category=ScoreCategory.CONTENT, type=FeedbackType.CLARITY,
            severity=Severity.INFO, start=0.0, end=6.4, title="표현이 모호해요",
            description="모호해요.", evidence=data.transcript.segments[0].text,
            suggestion="구체적으로 말해 보세요.",
        )  # fmt: skip
        return FeedbackResult(
            feedbacks=[fb],
            overall=fallback_overall(data.category_scores, [fb]),
            improved_scripts=[],
            candidates=[],
        )


@pytest.fixture
def world(app, db, storage, settings, tmp_path):
    settings.work_dir = str(tmp_path / "work")
    Path(settings.work_dir).mkdir()
    aid_ref = [""]
    steps = Steps(app.state.session_factory, aid_ref)
    clients = SimpleNamespace(stt=0, llm=0)

    def stt_client():
        clients.stt += 1
        return object()

    def llm_client():
        clients.llm += 1
        return object()

    probe = FakeProbe(30.0)
    deps = runner.PipelineDeps(
        storage=storage, settings=settings, probe=probe, stt_client=stt_client,
        llm_client=llm_client, extract_audio=steps.extract_audio,
        transcribe=steps.transcribe,
        observe_video=steps.observe_video, extract_slides=steps.extract_slides,
        analyze_content=steps.analyze_content, generate_feedback=steps.generate_feedback,
    )  # fmt: skip

    users: dict[str, User] = {}

    def make(
        *, user="u1", purpose="CLASS", video=b"video-A", slides_key=None, attempt=1, limit=None
    ):
        if user not in users:
            u = User(email=f"{user}@b.com", password_hash="x", nickname=user)
            db.add(u)
            db.flush()
            users[user] = u
        p = Presentation(
            user_id=users[user].id, title="t", purpose=purpose, slides_key=slides_key,
            time_limit_sec=limit,
        )  # fmt: skip
        db.add(p)
        db.flush()
        key = f"{VIDEO_KEY}-{p.id}"
        a = Analysis(
            presentation_id=p.id, attempt_no=attempt, status="QUEUED",
            content_type="video/webm", video_key=key, created_at=datetime(2026, 11, 5, tzinfo=UTC),
        )  # fmt: skip
        db.add(a)
        db.commit()
        storage.objects[key] = len(video)
        storage.files[key] = video
        aid_ref[0] = a.id
        return a.id

    def run(aid):
        aid_ref[0] = aid
        return runner.run_pipeline(aid, session_factory=app.state.session_factory, deps=deps)

    return SimpleNamespace(
        make=make, run=run, steps=steps, deps=deps, probe=probe, clients=clients,
        work=Path(settings.work_dir), storage=storage,
    )  # fmt: skip


def _analysis(db, aid) -> Analysis:
    db.expire_all()
    return db.get(Analysis, aid)


# ---------- 정상 흐름 ----------


def test_completes_with_spec_stages_and_valid_report(world, db):
    aid = world.make()
    assert world.run(aid) == "completed"
    a = _analysis(db, aid)
    assert (a.status, a.stage, a.progress) == ("COMPLETED", None, 100)
    # 단계 함수가 불릴 때의 stage/progress가 명세 순서·범위 안
    ranges = {
        "EXTRACTING_AUDIO": (0, 10), "TRANSCRIBING": (10, 40), "ANALYZING_NONVERBAL": (40, 65),
        "ANALYZING_CONTENT": (65, 80), "GENERATING_FEEDBACK": (80, 100),
    }  # fmt: skip
    order = list(ranges)
    seen = [s for s, _ in world.steps.stages]
    assert seen == sorted(seen, key=order.index)
    assert set(seen) == set(order)
    for stage, progress in world.steps.stages:
        lo, hi = ranges[stage]
        assert lo <= progress < hi
    report = db.get(Report, aid).data
    validate_report(report)
    assert report["analysisId"] == aid and report["durationSec"] == 30.0
    assert report["overall"]["summary"]  # 절대 null 아님
    # P7 요약값이 리포트와 같음
    assert a.overall_score == report["overall"]["score"]
    assert a.scores == {s["category"]: s["score"] for s in report["scores"]}
    assert a.filler_count == report["metrics"]["speech"]["fillerCount"] == 1
    assert a.syllables_per_min == report["metrics"]["speech"]["syllablesPerMin"]
    assert a.video_sha256 and a.cache_key and a.pipeline_version == runner.PIPELINE_VERSION


def test_feedback_gets_scores_and_tagged_transcript(world):
    world.run(world.make())
    (fi,) = world.steps.feedback_inputs
    assert set(fi.category_scores) == set(ScoreCategory)
    words = [w for s in fi.transcript.segments for w in s.words]
    assert any(w.tag == "FILLER" for w in words)  # speech_metrics가 태그를 채운 대본
    (ci,) = world.steps.content_inputs
    assert ci.slides is None
    assert ci.metrics_summary["syllablesPerMin"] > 0


def test_slides_are_downloaded_and_passed(world):
    aid = world.make(slides_key="slides/p.pdf")
    world.storage.objects["slides/p.pdf"] = 10
    assert world.run(aid) == "completed"
    assert world.steps.content_inputs[0].slides == ["표지", "결론"]
    assert world.steps.slides_paths[0].name == "slides.pdf"


def test_slides_download_failure_continues_without_slides(world):
    aid = world.make(slides_key="slides/missing.pdf")
    assert world.run(aid) == "completed"
    assert world.steps.content_inputs[0].slides is None


# ---------- 실패 코드 ----------


@pytest.mark.parametrize(
    ("step", "exc", "code"),
    [
        ("extract_audio", audio.NoAudioStream("x"), "NO_SPEECH_DETECTED"),
        ("transcribe", NoSpeechDetected("x"), "NO_SPEECH_DETECTED"),
        ("transcribe", SttServiceError("x"), "AI_SERVICE_ERROR"),
        ("observe_video", audio.NoVideoStream("x"), "NO_FACE_DETECTED"),
        ("observe_video", MediaError("x"), "CORRUPTED_VIDEO"),
        ("extract_audio", MediaError("x"), "CORRUPTED_VIDEO"),
        ("extract_audio", audio.FFmpegUnavailable("x"), "INTERNAL_ERROR"),
        ("observe_video", NonverbalUnavailable("x"), "INTERNAL_ERROR"),
        ("analyze_content", ContentServiceError("x"), "AI_SERVICE_ERROR"),
        ("generate_feedback", FeedbackServiceError("x"), "AI_SERVICE_ERROR"),
        ("generate_feedback", RuntimeError("boom"), "INTERNAL_ERROR"),
    ],
)
def test_module_errors_map_to_failure_codes(world, db, step, exc, code):
    world.steps.raise_at[step] = exc
    aid = world.make()
    assert world.run(aid) == "failed"
    a = _analysis(db, aid)
    assert (a.status, a.error_code, a.stage) == ("FAILED", code, None)
    assert a.error_message and a.error_message != "x"
    assert db.get(Report, aid) is None


def test_no_face_frames_fail_with_no_face(world, db):
    world.deps.observe_video = lambda video, ctype, f, p, **kw: [
        FrameObs(t=float(i), yaw=None, pitch=None, pose=None) for i in range(15)
    ]
    aid = world.make()
    assert world.run(aid) == "failed"
    assert _analysis(db, aid).error_code == "NO_FACE_DETECTED"


def test_video_too_long(world, db):
    world.probe.duration = 601.0
    aid = world.make()
    assert world.run(aid) == "failed"
    assert _analysis(db, aid).error_code == "VIDEO_TOO_LONG"
    assert world.steps.calls == []


def test_storage_download_failure_is_internal(world, db):
    world.storage.fail_download = True
    aid = world.make()
    assert world.run(aid) == "failed"
    assert _analysis(db, aid).error_code == "INTERNAL_ERROR"


def test_error_messages_are_user_facing(world, db):
    world.steps.raise_at["transcribe"] = NoSpeechDetected("internal detail")
    aid = world.make()
    world.run(aid)
    assert _analysis(db, aid).error_message == "영상에서 목소리를 찾지 못했어요."


# ---------- 임시 파일 정리 ----------


def test_temp_files_removed_on_success(world):
    world.run(world.make())
    assert list(world.work.iterdir()) == []
    assert world.steps.tmp_dirs and not any(d.exists() for d in world.steps.tmp_dirs)


def test_temp_files_removed_on_failure(world):
    world.steps.raise_at["observe_video"] = RuntimeError("boom")
    world.run(world.make())
    assert list(world.work.iterdir()) == []


def test_temp_files_removed_when_cancelled(world, db):
    aid = world.make()

    def transcribe_then_delete(path, client):
        db.query(Analysis).filter_by(id=aid).delete()
        db.commit()
        return normalize(WHISPER)

    world.deps.transcribe = transcribe_then_delete
    assert world.run(aid) == "cancelled"
    assert list(world.work.iterdir()) == []


# ---------- 결과 재사용 ----------


def test_same_video_and_settings_reuses_whole_report(world, db):
    first = world.make(video=b"same")
    world.run(first)
    calls_before = list(world.steps.calls)
    world.clients.stt = world.clients.llm = 0
    second = world.make(video=b"same", attempt=2)
    assert world.run(second) == "completed"
    assert world.steps.calls == calls_before  # 단계 함수 하나도 안 부름
    assert world.clients.stt == 0 and world.clients.llm == 0  # OpenAI 클라이언트도 안 만듦
    a1, a2 = db.get(Report, first).data, db.get(Report, second).data
    assert a2["analysisId"] == second and a2["attemptNo"] == 2
    assert a2["presentationId"] != a1["presentationId"]
    assert {
        k: v
        for k, v in a2.items()
        if k not in ("analysisId", "presentationId", "attemptNo", "createdAt")
    } == {
        k: v
        for k, v in a1.items()
        if k not in ("analysisId", "presentationId", "attemptNo", "createdAt")
    }
    validate_report(a2)
    assert _analysis(db, second).overall_score == _analysis(db, first).overall_score


def test_same_video_different_purpose_reuses_only_transcript(world):
    world.run(world.make(video=b"same", purpose="CLASS"))
    world.steps.calls.clear()
    world.clients.stt = 0
    assert world.run(world.make(video=b"same", purpose="PITCH")) == "completed"
    assert "extract_audio" not in world.steps.calls and "transcribe" not in world.steps.calls
    assert world.clients.stt == 0
    assert "analyze_content" in world.steps.calls and "generate_feedback" in world.steps.calls


def test_different_video_runs_fully(world):
    world.run(world.make(video=b"A"))
    world.steps.calls.clear()
    world.run(world.make(video=b"B"))
    assert "transcribe" in world.steps.calls


def test_other_users_result_is_not_reused(world):
    world.run(world.make(user="u1", video=b"same"))
    world.steps.calls.clear()
    world.run(world.make(user="u2", video=b"same"))
    assert "transcribe" in world.steps.calls


def test_failed_or_old_version_results_are_not_reused(world, db, monkeypatch):
    world.steps.raise_at["generate_feedback"] = RuntimeError("boom")
    world.run(world.make(video=b"same"))
    del world.steps.raise_at["generate_feedback"]
    world.steps.calls.clear()
    world.run(world.make(video=b"same"))
    assert "transcribe" in world.steps.calls  # 실패한 회차는 재사용 안 함
    monkeypatch.setattr(runner, "PIPELINE_VERSION", "999")
    world.steps.calls.clear()
    world.run(world.make(video=b"same"))
    assert "transcribe" in world.steps.calls  # 버전이 바뀌면 재사용 안 함


def test_skipped_when_not_claimable(world, db):
    aid = world.make()
    db.query(Analysis).filter_by(id=aid).update({"status": "COMPLETED"})
    db.commit()
    assert world.run(aid) == "skipped"


# ---------- 실제 content·feedback 모듈 + 가짜 LLM(끝-끝) ----------


def _llm_response(payload):
    return SimpleNamespace(
        status="completed",
        output_text=json.dumps(payload, ensure_ascii=False),
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


class RoutingLlm:
    def __init__(self):
        self.names: list[str] = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        name = kw["text"]["format"]["name"]
        self.names.append(name)
        if name.startswith("content_analysis"):
            return _llm_response({
                "sections": {"intro": {"startId": "s-001", "endId": "s-001"}, "body": None,
                             "conclusion": {"startId": "s-002", "endId": "s-002"}},
                "keyMessage": "발표 연습 서비스를 소개해요.",
                "issues": [
                    {"type": "CLARITY", "segmentIds": ["s-001"], "reason": "주제가 늦게 나와요."}
                ],
                "slides": [],
                "expectedQuestions": [
                    {"question": f"질문 {c}?", "intent": "의도예요.", "answerTip": "예시로 답해요."}
                    for c in "가나다라마"
                ],
            })  # fmt: skip
        return _llm_response({
            "feedbacks": [
                {"candidateId": f"c-{i:02d}", "title": "제목이에요", "description": "설명이에요.",
                 "suggestion": "연습해 보세요."}
                for i in range(1, 13)
            ],
            "overall": {"summary": "전체적으로 좋아요.", "strengths": ["시선이 안정적이에요"],
                        "improvements": ["주제를 먼저 말해요"]},
            "improvedScripts": [],
        })  # fmt: skip


def test_end_to_end_with_real_llm_modules(world, db):
    from app.services.pipeline import content, feedback

    llm = RoutingLlm()
    world.deps.analyze_content = content.analyze_content
    world.deps.generate_feedback = feedback.generate_feedback
    world.deps.llm_client = lambda: llm
    aid = world.make(limit=60)
    assert world.run(aid) == "completed"
    assert llm.names == ["content_analysis_v2", "feedback_v2"]  # 각 1번
    report = db.get(Report, aid).data
    validate_report(report)
    assert report["overall"]["summary"] == "전체적으로 좋아요."
    assert len(report["expectedQuestions"]) == 5
    types = {f["type"] for f in report["feedbacks"]}
    assert "CLARITY" in types and "UNDER_TIME" in types  # 60초 제한에 약 14초 발표
    for f in report["feedbacks"]:
        assert 0 <= f["start"] < f["end"] <= report["durationSec"]


# ---------- Celery 작업: 가짜/실제 전환 ----------


@pytest.mark.parametrize(("fake", "expected"), [(True, "fake"), (False, "real")])
def test_task_switches_on_use_fake_pipeline(monkeypatch, settings, fake, expected):
    from app.workers import tasks

    settings.use_fake_pipeline = fake
    monkeypatch.setattr(tasks, "_resources", lambda: (settings, None, None))
    monkeypatch.setattr(tasks, "_deps", lambda: None)
    monkeypatch.setattr(tasks, "run_fake_pipeline", lambda aid, **kw: "fake")
    monkeypatch.setattr(tasks, "run_pipeline", lambda aid, **kw: "real")
    assert tasks.run_analysis("a-1") == expected


def test_use_fake_pipeline_defaults_false(settings):
    assert settings.use_fake_pipeline is False


# ---------- 리뷰 반영: 슬라이드 재업로드, 깨진 저장 리포트, 시간 제한, 임시 폴더 ----------


def test_slides_reuploaded_under_same_key_runs_fully(world):
    world.storage.objects["slides/p.pdf"] = 1
    world.storage.files["slides/p.pdf"] = b"slides A"
    world.run(world.make(video=b"same", slides_key="slides/p.pdf"))
    world.steps.calls.clear()
    world.storage.files["slides/p.pdf"] = b"slides B"  # P6 재업로드: 키는 같고 내용만 바뀜
    assert world.run(world.make(video=b"same", slides_key="slides/p.pdf")) == "completed"
    assert "analyze_content" in world.steps.calls  # 옛 리포트를 재사용하지 않음


def test_same_slides_content_reuses_report(world):
    world.storage.objects["slides/p.pdf"] = 1
    world.storage.files["slides/p.pdf"] = b"slides A"
    world.run(world.make(video=b"same", slides_key="slides/p.pdf"))
    world.steps.calls.clear()
    world.run(world.make(video=b"same", slides_key="slides/p.pdf"))
    assert "analyze_content" not in world.steps.calls


def test_run_that_lost_its_slides_is_not_reused(world, db):
    first = world.make(video=b"same", slides_key="slides/gone.pdf")  # 다운로드 실패
    world.run(first)
    assert _analysis(db, first).cache_key is None
    world.storage.objects["slides/gone.pdf"] = 1
    world.storage.files["slides/gone.pdf"] = b"now here"
    world.steps.calls.clear()
    world.run(world.make(video=b"same", slides_key="slides/gone.pdf"))
    assert "analyze_content" in world.steps.calls
    assert world.steps.content_inputs[-1].slides == ["표지", "결론"]


def test_broken_stored_report_is_recomputed_not_failed(world, db, caplog):
    first = world.make(video=b"same")
    world.run(first)
    rep = db.get(Report, first)
    rep.data = {**rep.data, "overall": {"score": "broken"}}  # 지금 명세와 안 맞음
    db.commit()
    world.steps.calls.clear()
    second = world.make(video=b"same")
    assert world.run(second) == "completed"
    assert "generate_feedback" in world.steps.calls  # 전체 재사용 대신 다시 계산
    assert "transcribe" not in world.steps.calls  # 대본은 멀쩡해서 재사용
    assert "안녕하세요" not in caplog.text  # 대본 내용을 로그에 남기지 않음


def test_broken_stored_transcript_is_transcribed_again(world, db):
    first = world.make(video=b"same", purpose="CLASS")
    world.run(first)
    rep = db.get(Report, first)
    rep.data = {**rep.data, "transcript": {"segments": "broken"}}
    db.commit()
    world.steps.calls.clear()
    assert world.run(world.make(video=b"same", purpose="PITCH")) == "completed"
    assert "transcribe" in world.steps.calls


def test_soft_time_limit_maps_to_internal_error(world, db):
    from celery.exceptions import SoftTimeLimitExceeded

    world.steps.raise_at["observe_video"] = SoftTimeLimitExceeded()
    aid = world.make()
    assert world.run(aid) == "failed"
    assert _analysis(db, aid).error_code == "INTERNAL_ERROR"
    assert list(world.work.iterdir()) == []


def test_missing_llm_model_fails_before_any_work(world, db, settings):
    settings.openai_llm_model = " "
    aid = world.make()
    assert world.run(aid) == "failed"
    assert _analysis(db, aid).error_code == "INTERNAL_ERROR"
    assert world.storage.downloads == [] and world.steps.calls == []


def test_stale_work_dirs_are_cleaned(tmp_path):
    import os

    old = tmp_path / f"{runner.WORKDIR_PREFIX}old"
    new = tmp_path / f"{runner.WORKDIR_PREFIX}new"
    other = tmp_path / "keep-me"
    for d in (old, new, other):
        d.mkdir()
        (d / "video").write_bytes(b"x")
    past = 10_000.0
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    now = past + runner.STALE_WORKDIR_SEC + 1
    os.utime(new, (now, now))
    assert runner.cleanup_stale_workdirs(tmp_path, now=now) == 1
    assert not old.exists() and new.exists() and other.exists()


def test_missing_work_dir_is_created(world, settings, tmp_path):
    settings.work_dir = str(tmp_path / "not" / "yet")
    assert world.run(world.make()) == "completed"
    assert Path(settings.work_dir).is_dir()


def test_progress_moves_after_probe(world):
    world.run(world.make())
    first_stage = world.steps.stages[0]
    assert first_stage == ("EXTRACTING_AUDIO", 5)  # 2(다운로드 전) → 5(재사용 확인 후)


def test_observe_video_gets_video_and_limits(world):
    world.run(world.make())
    assert world.steps.video_kwargs == {
        "fps": runner.FRAME_FPS,
        "max_seconds": float(600),
        "expected_duration": 30.0,
    }
