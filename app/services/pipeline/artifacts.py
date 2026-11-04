"""회차별 파이프라인 중간 결과 저장과 재계산(LLM 재호출 없음).

저장(analysis_artifacts.data, JSON):
- input: 영상 길이(원래 측정값), 제한 시간, 목적, 파이프라인 버전
- stt: 정규화한 대본(태그 없음)
- speech: 말하기 지표(참고용. 재계산 때는 대본으로 다시 계산)
- nonverbal: 프레임별 관측값(얼굴 각도·자세 랜드마크·고개 방향)
- content: LLM 응답 원본(raw)과 후처리한 결과(result)
- feedback: LLM 응답 원본(raw), 후처리한 결과(result), 그때의 피드백 후보(candidates)

재계산(recompute): 대본·관측값으로 말하기·비언어 지표를 다시 계산하고, 저장된
content·feedback 결과로 점수(scoring)와 리포트(report)를 다시 조립한다.
피드백 문장은 저장된 것을 그대로 쓴다(문장 안 숫자는 처음 분석 때 값이다.
지표 기준을 바꿔 후보가 달라져도 문장을 새로 쓰지 않는다).

사용자 대본이 들어 있으므로 회차·발표를 지우면 함께 지운다(모델의 cascade).
"""

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models import Analysis, AnalysisArtifact, Report
from app.models.enums import Purpose
from app.schemas.report import (
    ExpectedQuestion,
    Feedback,
    FeedbackType,
    ImprovedScript,
    Severity,
    SlideCoverage,
    Transcript,
)
from app.services.pipeline import content as content_mod
from app.services.pipeline import feedback as feedback_mod
from app.services.pipeline.nonverbal import FrameObs, HeadTurn, PoseObs, analyze_nonverbal
from app.services.pipeline.report import ReportParts, build_report, history_values
from app.services.pipeline.scoring import score_presentation
from app.services.pipeline.speech_metrics import analyze_speech

FORMAT_VERSION = 1


class ArtifactMissing(Exception):
    """재계산할 중간 결과가 없다(이 기능 전에 분석했거나 재사용 회차)."""


# ---------------------------------------------------------------- 직렬화


def _pt(p):
    return list(p) if p is not None else None


def _frame_to_json(f: FrameObs) -> dict[str, Any]:
    pose = None
    if f.pose is not None:
        pose = {
            "nose": _pt(f.pose.nose),
            "leftShoulder": _pt(f.pose.left_shoulder),
            "rightShoulder": _pt(f.pose.right_shoulder),
            "leftWrist": _pt(f.pose.left_wrist),
            "rightWrist": _pt(f.pose.right_wrist),
        }
    return {
        "t": f.t,
        "yaw": f.yaw,
        "pitch": f.pitch,
        "pose": pose,
        "headTurn": f.head_turn.value if f.head_turn else None,
    }


def _tuple(v) -> tuple[float, float] | None:
    return (float(v[0]), float(v[1])) if v is not None else None


def _frame_from_json(d: dict[str, Any]) -> FrameObs:
    pose = None
    if d.get("pose"):
        p = d["pose"]
        pose = PoseObs(
            nose=_tuple(p["nose"]),
            left_shoulder=_tuple(p["leftShoulder"]),
            right_shoulder=_tuple(p["rightShoulder"]),
            left_wrist=_tuple(p["leftWrist"]),
            right_wrist=_tuple(p["rightWrist"]),
        )
    return FrameObs(
        t=d["t"],
        yaw=d["yaw"],
        pitch=d["pitch"],
        pose=pose,
        head_turn=HeadTurn(d["headTurn"]) if d.get("headTurn") else None,
    )


def _usage(u) -> dict[str, Any] | None:
    if u is None:
        return None
    return {
        "model": u.model,
        "attempts": u.attempts,
        "inputTokens": u.input_tokens,
        "outputTokens": u.output_tokens,
        "estimatedCostUsd": u.estimated_cost_usd,
    }


def content_to_json(c: content_mod.ContentAnalysis) -> dict[str, Any]:
    return {
        "sections": [
            {
                "kind": s.kind.value,
                "startId": s.start_id,
                "endId": s.end_id,
                "start": s.start,
                "end": s.end,
            }
            for s in c.sections
        ],  # fmt: skip
        "keyMessage": c.key_message,
        "issues": [
            {
                "type": i.type.value,
                "segmentIds": i.segment_ids,
                "start": i.start,
                "end": i.end,
                "reason": i.reason,
            }
            for i in c.issues
        ],  # fmt: skip
        "slideCoverage": (
            [s.model_dump(by_alias=True) for s in c.slide_coverage]
            if c.slide_coverage is not None
            else None
        ),
        "expectedQuestions": [q.model_dump(by_alias=True) for q in c.expected_questions],
        "usage": _usage(c.usage),
        "transcriptTruncated": c.transcript_truncated,
        "droppedItems": c.dropped_items,
        "slidesMissingPages": c.slides_missing_pages,
        "slidesTruncated": c.slides_truncated,
    }


def content_from_json(d: dict[str, Any]) -> content_mod.ContentAnalysis:
    return content_mod.ContentAnalysis(
        sections=[
            content_mod.Section(
                kind=content_mod.SectionKind(s["kind"]),
                start_id=s["startId"],
                end_id=s["endId"],
                start=s["start"],
                end=s["end"],
            )
            for s in d["sections"]
        ],  # fmt: skip
        key_message=d["keyMessage"],
        issues=[
            content_mod.ContentIssue(
                type=content_mod.IssueType(i["type"]),
                segment_ids=list(i["segmentIds"]),
                start=i["start"],
                end=i["end"],
                reason=i["reason"],
            )
            for i in d["issues"]
        ],  # fmt: skip
        slide_coverage=(
            [SlideCoverage.model_validate(s) for s in d["slideCoverage"]]
            if d["slideCoverage"] is not None
            else None
        ),
        expected_questions=[ExpectedQuestion.model_validate(q) for q in d["expectedQuestions"]],
        usage=None,
        transcript_truncated=d.get("transcriptTruncated", False),
        dropped_items=d.get("droppedItems", 0),
        slides_missing_pages=list(d.get("slidesMissingPages", [])),
        slides_truncated=d.get("slidesTruncated", False),
    )


def _candidate_to_json(c: feedback_mod.Candidate) -> dict[str, Any]:
    return {
        "id": c.id,
        "type": c.type.value,
        "severity": c.severity.value,
        "start": c.start,
        "end": c.end,
        "values": c.values,
        "segmentIds": list(c.segment_ids),
        "context": c.context,
    }


def _candidate_from_json(d: dict[str, Any]) -> feedback_mod.Candidate:
    return feedback_mod.Candidate(
        type=FeedbackType(d["type"]),
        severity=Severity(d["severity"]),
        start=d["start"],
        end=d["end"],
        values=dict(d["values"]),
        segment_ids=tuple(d["segmentIds"]),
        context=d.get("context"),
        id=d["id"],
    )


def feedback_to_json(r: feedback_mod.FeedbackResult) -> dict[str, Any]:
    return {
        "feedbacks": [f.model_dump(mode="json", by_alias=True) for f in r.feedbacks],
        "overall": {
            "summary": r.overall.summary,
            "strengths": r.overall.strengths,
            "improvements": r.overall.improvements,
        },
        "improvedScripts": [s.model_dump(mode="json", by_alias=True) for s in r.improved_scripts],
        "candidates": [_candidate_to_json(c) for c in r.candidates],
        "usage": _usage(r.usage),
        "droppedItems": r.dropped_items,
        "overallFallback": r.overall_fallback,
    }


def feedback_from_json(d: dict[str, Any]) -> feedback_mod.FeedbackResult:
    o = d["overall"]
    return feedback_mod.FeedbackResult(
        feedbacks=[Feedback.model_validate(f) for f in d["feedbacks"]],
        overall=feedback_mod.OverallText(
            o["summary"], list(o["strengths"]), list(o["improvements"])
        ),
        improved_scripts=[ImprovedScript.model_validate(s) for s in d["improvedScripts"]],
        candidates=[_candidate_from_json(c) for c in d["candidates"]],
        dropped_items=d.get("droppedItems", 0),
        overall_fallback=d.get("overallFallback", False),
    )


def build_artifacts(
    *,
    pipeline_version: str,
    purpose: Purpose,
    duration_sec: float,
    time_limit_sec: int | None,
    transcript: Transcript,
    speech_metrics: dict[str, Any],
    frames: list[FrameObs],
    content: content_mod.ContentAnalysis,
    feedback: feedback_mod.FeedbackResult,
) -> dict[str, Any]:
    return {
        "formatVersion": FORMAT_VERSION,
        "input": {
            "pipelineVersion": pipeline_version,
            "purpose": purpose.value,
            "durationSec": duration_sec,  # 반올림 전 측정값
            "timeLimitSec": time_limit_sec,
        },
        "stt": {"transcript": transcript.model_dump(mode="json", by_alias=True)},
        "speech": speech_metrics,
        "nonverbal": {"frames": [_frame_to_json(f) for f in frames]},
        "content": {"raw": content.raw, "result": content_to_json(content)},
        "feedback": {
            "raw": feedback.raw,
            "result": feedback_to_json(feedback),
        },
    }


# ---------------------------------------------------------------- 재계산


def _candidate_key(c: feedback_mod.Candidate) -> tuple:
    return (c.type.value, c.severity.value, c.start, c.end, json.dumps(c.values, sort_keys=True))


def rebuild_report(
    data: dict[str, Any],
    *,
    analysis_id: str,
    presentation_id: str,
    attempt_no: int,
    created_at: datetime,
) -> tuple[dict[str, Any], bool]:
    """중간 결과로 리포트를 다시 조립한다(LLM 없음). 제한 시간·목적은 저장된 값을 쓴다.

    (리포트, drift). drift는 지금 코드로 만든 피드백 후보(구간·수치)가 저장된 후보와 달라
    피드백 문장 속 숫자·구간이 새 지표와 어긋날 수 있다는 표시다(문장은 LLM 없이 바꿀 수 없음).
    """
    if data.get("formatVersion") != FORMAT_VERSION:
        raise ArtifactMissing(f"unsupported artifact format {data.get('formatVersion')!r}")
    inp = data["input"]
    duration = float(inp["durationSec"])
    limit = inp["timeLimitSec"]
    purpose = Purpose(inp["purpose"])
    transcript = Transcript.model_validate(data["stt"]["transcript"])
    speech = analyze_speech(transcript, duration, limit)
    frames = [_frame_from_json(f) for f in data["nonverbal"]["frames"]]
    nv = analyze_nonverbal(frames, duration_sec=duration)
    content = content_from_json(data["content"]["result"])
    fb = feedback_from_json(data["feedback"]["result"])
    scores = score_presentation(purpose, speech, nv, content)
    current = feedback_mod.build_candidates(
        feedback_mod.FeedbackInput(
            purpose=purpose,
            duration_sec=duration,
            transcript=speech.transcript,
            speech=speech,
            nonverbal=nv,
            content=content,
            category_scores=scores.categories,
        )
    )
    drift = [_candidate_key(c) for c in current] != [_candidate_key(c) for c in fb.candidates]
    report = build_report(
        ReportParts(
            analysis_id=analysis_id,
            presentation_id=presentation_id,
            attempt_no=attempt_no,
            purpose=purpose,
            duration_sec=duration,
            created_at=created_at,
            scores=scores,
            speech=speech,
            nonverbal=nv,
            content=content,
            feedback=fb,
        )
    )
    return report, drift


def recompute(
    db: Session, analysis_id: str, *, write: bool = True, allow_old_version: bool = False
) -> tuple[dict, dict, bool]:
    """저장된 중간 결과로 리포트를 다시 만든다(write면 저장). (이전, 새 리포트, drift)

    중간 결과를 만든 파이프라인 버전이 지금과 다르면 거부한다(allow_old_version이 아니면).
    예: 버전 3은 초당 1장·가려진 손목 포함으로 뽑은 프레임이라, 새 제스처 계산을 얹으면 예전
    결과도 새로 분석한 결과도 아닌 숫자가 나온다.
    """
    from app.services.pipeline.runner import PIPELINE_VERSION  # runner가 이 모듈을 import한다

    a = db.get(Analysis, analysis_id)
    if a is None:
        raise ArtifactMissing(f"analysis {analysis_id} not found")
    art = db.get(AnalysisArtifact, analysis_id)
    rep = db.get(Report, analysis_id)
    if art is None or rep is None:
        raise ArtifactMissing(f"analysis {analysis_id} has no stored intermediate results")
    stored = (art.data.get("input") or {}).get("pipelineVersion")
    if stored != PIPELINE_VERSION and not allow_old_version:
        raise ArtifactMissing(
            f"pipeline version {stored!r} != current {PIPELINE_VERSION!r}"
            " (re-analyze, or use --allow-old-version)"
        )
    created = a.created_at if a.created_at.tzinfo else a.created_at.replace(tzinfo=UTC)
    # 제한 시간·목적은 처음 분석 때 값(저장값)을 쓴다(나중에 발표를 고쳐도 같은 결과가 나오게).
    new, drift = rebuild_report(
        art.data,
        analysis_id=a.id,
        presentation_id=a.presentation_id,
        attempt_no=a.attempt_no,
        created_at=created,
    )
    old = rep.data
    if write:
        rep.data = new
        for k, v in history_values(new).items():
            setattr(a, k, v)
        db.commit()
    return old, new, drift
