"""중간 결과 저장·재계산·삭제. 외부 호출 없음(LLM·Whisper는 가짜, 재계산은 호출 자체가 없음)."""

import json

import pytest

from app.models import AnalysisArtifact, Report
from app.models.enums import ScoreCategory
from app.schemas.report import Feedback, FeedbackType, Severity, SlideCoverage
from app.services.pipeline import artifacts as art
from app.services.pipeline import runner, scoring
from app.services.pipeline.content import (
    ContentAnalysis,
    ContentIssue,
    IssueType,
    Section,
    SectionKind,
)
from app.services.pipeline.feedback import Candidate, FeedbackResult, OverallText
from app.services.pipeline.lifecycle import json_size
from app.services.pipeline.nonverbal import FrameObs, HeadTurn, PoseObs
from app.services.pipeline.report import validate_report
from app.tools import recompute as cli
from tests.test_runner import RoutingLlm, _analysis, world  # noqa: F401 - pytest 픽스처

API = "/api/v1"


# ---------- 직렬화 왕복 ----------


def test_frame_roundtrip():
    pose = PoseObs((0.5, 0.2), (0.6, 0.4), (0.4, 0.4), None, (0.3, 0.6))
    frames = [
        FrameObs(t=0.0, yaw=1.5, pitch=-2.0, pose=pose, head_turn=HeadTurn.FRONT),
        FrameObs(t=1.0, yaw=None, pitch=None, pose=None, head_turn=HeadTurn.AWAY),
        FrameObs(t=2.0, yaw=None, pitch=None, pose=None),
    ]
    back = [art._frame_from_json(json.loads(json.dumps(art._frame_to_json(f)))) for f in frames]
    assert back == frames


def test_content_roundtrip():
    c = ContentAnalysis(
        sections=[Section(SectionKind.INTRO, "s-001", "s-002", 0.0, 9.5)],
        key_message="핵심이에요.",
        issues=[ContentIssue(IssueType.CLARITY, ["s-003"], 10.0, 14.5, "모호해요.")],
        slide_coverage=[SlideCoverage(page=1, covered=False, note="빠졌어요.")],
        expected_questions=[],
        usage=None,
        slides_missing_pages=[2],
    )
    back = art.content_from_json(json.loads(json.dumps(art.content_to_json(c))))
    assert back == c


def test_feedback_roundtrip():
    fb = FeedbackResult(
        feedbacks=[
            Feedback(
                id="f-01", category=ScoreCategory.DELIVERY, type=FeedbackType.PACE_SLOW,
                severity=Severity.WARNING, start=0.0, end=44.52, title="느려요",
                description="분당 244음절이에요.", evidence="문장", suggestion="빠르게 해 보세요.",
            )
        ],
        overall=OverallText("요약이에요.", ["강점"], ["개선"]),
        improved_scripts=[],
        candidates=[Candidate(FeedbackType.PACE_SLOW, Severity.WARNING, 0.0, 44.52,
                              {"pace": 244, "avgPace": 244}, ("s-001",), None, "c-01")],
    )  # fmt: skip
    back = art.feedback_from_json(json.loads(json.dumps(art.feedback_to_json(fb))))
    assert back == fb


# ---------- 워커가 저장 ----------


def test_runner_saves_artifacts(world, db):  # noqa: F811
    aid = world.make(limit=60)
    assert world.run(aid) == "completed"
    a = db.get(AnalysisArtifact, aid)
    assert a is not None
    d = a.data
    assert set(d) == {"formatVersion", "input", "stt", "speech", "nonverbal", "content", "feedback"}
    assert d["input"] == {
        "pipelineVersion": runner.PIPELINE_VERSION,
        "purpose": "CLASS",
        "durationSec": 30.0,
        "timeLimitSec": 60,
    }
    words = [w for s in d["stt"]["transcript"]["segments"] for w in s["words"]]
    assert words and all(w["tag"] is None for w in words)  # 태그 없는 원래 받아쓰기
    assert len(d["nonverbal"]["frames"]) == 15
    assert d["speech"]["syllablesPerMin"] > 0
    assert a.size_bytes == json_size(d) > 0


def test_reused_analysis_copies_artifacts(world, db):  # noqa: F811
    first = world.make(video=b"same")
    world.run(first)
    second = world.make(video=b"same", attempt=2)
    world.run(second)
    a1, a2 = db.get(AnalysisArtifact, first), db.get(AnalysisArtifact, second)
    assert a2 is not None and a2.data == a1.data


# ---------- 재계산 ----------


def test_recompute_reproduces_stored_report(world, db):  # noqa: F811
    aid = world.make(limit=60)
    world.run(aid)
    old, new, drift = art.recompute(db, aid, write=False)
    assert new == old and drift is False
    validate_report(new)


def test_recompute_with_real_llm_modules_calls_no_llm(world, db):  # noqa: F811
    from app.services.pipeline import content, feedback

    llm = RoutingLlm()
    world.deps.analyze_content = content.analyze_content
    world.deps.generate_feedback = feedback.generate_feedback
    world.deps.llm_client = lambda: llm
    aid = world.make(limit=60)
    world.run(aid)
    calls = list(llm.names)
    old, new, drift = art.recompute(db, aid, write=False)
    assert llm.names == calls and drift is False  # 재계산은 LLM을 부르지 않는다
    assert new == old
    stored = db.get(AnalysisArtifact, aid).data
    assert stored["content"]["raw"]["keyMessage"]  # LLM 원본도 저장
    assert len(stored["feedback"]["raw"]["feedbacks"]) == 12


def test_recompute_applies_changed_scoring(world, db, monkeypatch):  # noqa: F811
    aid = world.make(limit=60)
    world.run(aid)
    before = db.get(Report, aid).data
    monkeypatch.setattr(scoring, "CLARITY_PENALTY", 60)  # 기준값을 바꿨다고 가정
    old, new, drift = art.recompute(db, aid, write=True)
    assert drift is False  # 점수 기준만 바뀌고 피드백 후보는 그대로
    assert old == before
    content_score = {s["category"]: s["score"] for s in new["scores"]}["CONTENT"]
    assert content_score < {s["category"]: s["score"] for s in old["scores"]}["CONTENT"]
    assert [f["title"] for f in new["feedbacks"]] == [f["title"] for f in old["feedbacks"]]
    db.expire_all()
    assert db.get(Report, aid).data == new
    assert _analysis(db, aid).scores["CONTENT"] == content_score  # P7 요약값도 갱신


def test_recompute_without_artifact(world, db):  # noqa: F811
    aid = world.make()
    world.run(aid)
    db.query(AnalysisArtifact).filter_by(analysis_id=aid).delete()
    db.commit()
    with pytest.raises(art.ArtifactMissing):
        art.recompute(db, aid)
    with pytest.raises(art.ArtifactMissing):
        art.recompute(db, "a-nope")


# ---------- 명령 ----------


def test_cli_dry_run_does_not_write(world, db, app, capsys, monkeypatch):  # noqa: F811
    aid = world.make(limit=60)
    world.run(aid)
    before = db.get(Report, aid).data
    monkeypatch.setattr(scoring, "CLARITY_PENALTY", 60)
    assert cli.main([aid, "--dry-run"], session_factory=app.state.session_factory) == 0
    out = capsys.readouterr().out
    assert aid in out and "(저장 안 함)" in out
    db.expire_all()
    assert db.get(Report, aid).data == before


def test_cli_all_and_stats(world, db, app, capsys):  # noqa: F811
    ids = [world.make(video=b"a"), world.make(video=b"b")]
    for aid in ids:
        world.run(aid)
    assert cli.main(["--all"], session_factory=app.state.session_factory) == 0
    out = capsys.readouterr().out
    assert all(aid in out for aid in ids) and "저장함" in out
    assert cli.main(["--stats"], session_factory=app.state.session_factory) == 0
    assert "artifacts=2" in capsys.readouterr().out


def test_cli_missing_returns_error(app, capsys):
    assert cli.main(["a-nope"], session_factory=app.state.session_factory) == 1
    assert "실패" in capsys.readouterr().out


# ---------- 삭제(N6·P5) 때 함께 지움 ----------


def _make_analysis(client, auth_headers):
    p = client.post(
        f"{API}/presentations", json={"title": "t", "purpose": "CLASS"}, headers=auth_headers
    ).json()["id"]
    a = client.post(
        f"{API}/presentations/{p}/analyses",
        json={
            "fileName": "t.webm",
            "contentType": "video/webm",
            "fileSize": 1000,
            "source": "UPLOAD",
        },
        headers=auth_headers,
    ).json()["analysisId"]
    return p, a


def _attach_artifact(db, aid):
    db.add(AnalysisArtifact(analysis_id=aid, data={"stt": {"transcript": "대본"}}, size_bytes=10))
    db.commit()


def test_n6_delete_removes_artifact(client, auth_headers, db):
    _, aid = _make_analysis(client, auth_headers)
    _attach_artifact(db, aid)
    res = client.delete(f"{API}/analyses/{aid}", headers=auth_headers)
    assert res.status_code == 204
    db.expire_all()
    assert db.get(AnalysisArtifact, aid) is None


def test_p5_delete_removes_artifacts(client, auth_headers, db):
    pid, aid = _make_analysis(client, auth_headers)
    _attach_artifact(db, aid)
    res = client.delete(f"{API}/presentations/{pid}", headers=auth_headers)
    assert res.status_code == 204
    db.expire_all()
    assert db.get(AnalysisArtifact, aid) is None


def test_recompute_flags_feedback_drift(world, db, app, capsys, monkeypatch):  # noqa: F811
    from app.services.pipeline import feedback

    aid = world.make(limit=20)  # 약 14초 발표 → 제한 20초의 70% 미만 → UNDER_TIME 후보
    world.run(aid)
    monkeypatch.setattr(feedback, "UNDER_TIME_WARNING_RATIO", 0.9)  # 후보 심각도가 바뀜
    _, _, drift = art.recompute(db, aid, write=False)
    assert drift is True
    assert cli.main([aid, "--dry-run"], session_factory=app.state.session_factory) == 0
    assert "피드백 근거 변경됨" in capsys.readouterr().out


def test_unsupported_format_version(world, db):  # noqa: F811
    aid = world.make()
    world.run(aid)
    a = db.get(AnalysisArtifact, aid)
    a.data = {**a.data, "formatVersion": 99}
    db.commit()
    with pytest.raises(art.ArtifactMissing):
        art.recompute(db, aid)


def test_cli_continues_after_unexpected_error(world, db, app, capsys, monkeypatch):  # noqa: F811
    ids = [world.make(video=b"a"), world.make(video=b"b")]
    for aid in ids:
        world.run(aid)
    bad = db.get(AnalysisArtifact, ids[0])
    # durationSec 등이 없어 KeyError(버전 확인은 통과)
    bad.data = {
        **bad.data,
        "input": {"pipelineVersion": runner.PIPELINE_VERSION, "secret": "대본 일부"},
    }
    db.commit()
    assert cli.main(["--all"], session_factory=app.state.session_factory) == 1
    out = capsys.readouterr().out
    assert f"{ids[0]}: 실패 KeyError" in out and ids[1] in out and "저장함" in out
    assert "대본" not in out  # 알 수 없는 예외 내용은 출력하지 않음


def test_cli_rejects_ids_with_all(app):
    with pytest.raises(SystemExit):
        cli.main(["a-1", "--all"], session_factory=app.state.session_factory)
    with pytest.raises(SystemExit):
        cli.main(["--all", "--stats"], session_factory=app.state.session_factory)


def test_unstorable_artifacts_are_skipped_not_failing(world, db, monkeypatch):  # noqa: F811
    from app.services.pipeline import runner

    real = runner.build_artifacts
    monkeypatch.setattr(runner, "build_artifacts", lambda **kw: {**real(**kw), "bad": float("nan")})
    aid = world.make()
    assert world.run(aid) == "completed"
    assert db.get(Report, aid) is not None
    assert db.get(AnalysisArtifact, aid) is None


def test_cli_all_with_no_artifacts(app, capsys):
    assert cli.main(["--all"], session_factory=app.state.session_factory) == 0
    assert "없어요" in capsys.readouterr().out


def test_recompute_rejects_other_pipeline_version(world, db, app, capsys):  # noqa: F811
    aid = world.make()
    world.run(aid)
    a = db.get(AnalysisArtifact, aid)
    a.data = {**a.data, "input": {**a.data["input"], "pipelineVersion": "3"}}
    db.commit()
    with pytest.raises(art.ArtifactMissing, match="pipeline version"):
        art.recompute(db, aid, write=False)
    _, _, drift = art.recompute(db, aid, write=False, allow_old_version=True)
    assert drift is False
    assert cli.main([aid, "--dry-run"], session_factory=app.state.session_factory) == 1
    assert "pipeline version" in capsys.readouterr().out
    assert (
        cli.main(
            [aid, "--dry-run", "--allow-old-version"], session_factory=app.state.session_factory
        )
        == 0
    )
