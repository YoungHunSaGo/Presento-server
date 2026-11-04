"""실제 분석 파이프라인(워커).

단계(명세 6장 stage·progress):
- EXTRACTING_AUDIO 0~10: 영상 길이 확인(10분 초과 VIDEO_TOO_LONG), 영상 내려받기·해시, 음성 추출
- TRANSCRIBING 10~40: whisper-1 받아쓰기 → 말하기 지표
- ANALYZING_NONVERBAL 40~65: 원본 프레임(초당 2장, 실제 프레임 시각) → MediaPipe
  (자세 640px, 얼굴은 머리 영역) → 지표
- ANALYZING_CONTENT 65~80: 슬라이드 텍스트(있으면) → LLM 내용 분석
- GENERATING_FEEDBACK 80~100: 점수(코드) → LLM 피드백 → 리포트 조립·검증·저장

결과 재사용(같은 사용자, 같은 영상 해시):
- cache_key(영상 해시 + 목적 + 제한 시간 + 슬라이드 파일 해시 + PIPELINE_VERSION)가 같은 완료 회차가
  있으면 그 리포트를 그대로 쓴다(ID·회차 번호·생성 시각만 바꿈). OpenAI를 부르지 않는다.
  슬라이드는 다시 올려도 S3 키가 같으므로 키가 아니라 파일 내용의 해시를 쓴다.
- 영상 해시만 같으면(목적·슬라이드가 다름) 받아쓰기 결과만 재사용하고 나머지는 다시 계산한다.
- 슬라이드가 있어야 하는데 못 받은 회차는 cache_key를 남기지 않는다(슬라이드 없는 결과가 재사용되지
  않게). 저장된 리포트가 지금 명세와 안 맞으면 재사용하지 않고 다시 계산한다.
- 다른 사용자의 결과는 재사용하지 않는다. 같은 영상을 동시에 두 번 돌리면 둘 다 계산한다.
  재사용해도 하루 분석 횟수는 N2에서 이미 차감됐다.

예외 → 분석 실패 코드:
- NO_SPEECH_DETECTED: 음성 트랙 없음, 받아쓴 말 없음
- NO_FACE_DETECTED: 영상 트랙 없음, 얼굴이 거의 안 보임
- CORRUPTED_VIDEO: 영상을 열 수 없음(형식 다름·깨짐)
- AI_SERVICE_ERROR: OpenAI 호출 실패(받아쓰기·내용 분석·피드백)
- INTERNAL_ERROR: ffmpeg·MediaPipe·스토리지 등 서버 문제, 그 밖의 예외
- VIDEO_TOO_LONG: 10분 초과

임시 파일(영상·음성·프레임·슬라이드)은 회차마다 임시 폴더에 두고 성공·실패 모두 지운다.
작업이 강제 종료(Celery hard time limit)되면 지우지 못하므로, 다음 작업이 시작할 때
STALE_WORKDIR_SEC보다 오래된 임시 폴더를 지운다.
"""

import copy
import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import (
    Analysis,
    AnalysisArtifact,
    AnalysisStage,
    AnalysisStatus,
    Presentation,
    Report,
)
from app.models.enums import Purpose
from app.schemas.common import to_kst_iso
from app.schemas.report import Segment, Transcript, Word
from app.services.media import MediaError, ProbeDuration
from app.services.pipeline import audio, content, feedback, nonverbal, stt
from app.services.pipeline.artifacts import build_artifacts
from app.services.pipeline.content import ContentInput, ContentServiceError
from app.services.pipeline.feedback import FeedbackInput, FeedbackServiceError
from app.services.pipeline.lifecycle import (
    FAILURE_MESSAGES,
    AnalysisFailure,
    Cancelled,
    claim,
    complete,
    fail,
    measure_duration,
    set_progress,
)
from app.services.pipeline.llm import LlmClient
from app.services.pipeline.nonverbal import NoFaceDetected, NonverbalUnavailable
from app.services.pipeline.report import ReportInvalid, ReportParts, build_report, validate_report
from app.services.pipeline.scoring import score_presentation
from app.services.pipeline.speech_metrics import analyze_speech
from app.services.pipeline.stt import NoSpeechDetected, SttClient, SttServiceError
from app.services.storage import Storage, StorageError

logger = logging.getLogger(__name__)

# 알고리즘·프롬프트·기준값을 바꾸면 올린다. 바뀐 버전의 결과는 재사용하지 않는다.
# 2: 얼굴 머리 영역 자르기, 3: stt 프롬프트 v2, 4: 실제 프레임 시각·2fps·제스처 안정화
PIPELINE_VERSION = "4"
FRAME_FPS = 2.0  # 비언어(자세·제스처·시선) 프레임. 1fps와 처리 시간이 거의 같다(디코딩이 대부분)
_HASH_CHUNK = 1024 * 1024
WORKDIR_PREFIX = "presento-"
STALE_WORKDIR_SEC = 30 * 60  # Celery hard limit(16분)보다 넉넉히

# 예외 종류 → 실패 코드(위에서부터 처음 맞는 것)
_FAILURE_MAP: list[tuple[type[BaseException], str]] = [
    (audio.NoAudioStream, "NO_SPEECH_DETECTED"),
    (NoSpeechDetected, "NO_SPEECH_DETECTED"),
    (audio.NoVideoStream, "NO_FACE_DETECTED"),
    (NoFaceDetected, "NO_FACE_DETECTED"),
    (MediaError, "CORRUPTED_VIDEO"),
    (SttServiceError, "AI_SERVICE_ERROR"),
    (ContentServiceError, "AI_SERVICE_ERROR"),
    (FeedbackServiceError, "AI_SERVICE_ERROR"),
    (audio.FFmpegUnavailable, "INTERNAL_ERROR"),
    (NonverbalUnavailable, "INTERNAL_ERROR"),
    (StorageError, "INTERNAL_ERROR"),
    (ReportInvalid, "INTERNAL_ERROR"),
    (SoftTimeLimitExceeded, "INTERNAL_ERROR"),  # 작업 시간 제한(15분) 초과
]


def failure_code(exc: BaseException) -> str:
    for kind, code in _FAILURE_MAP:
        if isinstance(exc, kind):
            return code
    return "INTERNAL_ERROR"


@dataclass
class PipelineDeps:
    """외부 자원과 단계 함수. 테스트에서 바꿔 끼운다."""

    storage: Storage
    settings: Settings
    probe: ProbeDuration
    stt_client: Callable[[], SttClient]
    llm_client: Callable[[], LlmClient]
    extract_audio: Callable[..., Path] = audio.extract_audio
    transcribe: Callable[..., stt.SttResult] = stt.transcribe
    observe_video: Callable[..., list[nonverbal.FrameObs]] = nonverbal.observe_video
    extract_slides: Callable[[Path], list[str] | None] = content.extract_slide_texts
    analyze_content: Callable[..., content.ContentAnalysis] = content.analyze_content
    generate_feedback: Callable[..., feedback.FeedbackResult] = feedback.generate_feedback
    now: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))


# ---------------------------------------------------------------- 재사용


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def make_cache_key(video_sha: str, p: Presentation, slides_sha: str | None) -> str:
    parts = [video_sha, p.purpose, str(p.time_limit_sec), slides_sha or "", PIPELINE_VERSION]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _previous(db: Session, user_id: str, current_id: str, **where: Any) -> Report | None:
    stmt = (
        select(Report)
        .join(Analysis, Analysis.id == Report.analysis_id)
        .join(Presentation, Presentation.id == Analysis.presentation_id)
        .where(
            Presentation.user_id == user_id,
            Analysis.id != current_id,
            Analysis.status == AnalysisStatus.COMPLETED,
            Analysis.pipeline_version == PIPELINE_VERSION,
            *(getattr(Analysis, k) == v for k, v in where.items()),
        )
        .order_by(Analysis.completed_at.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def rebase_report(data: dict[str, Any], analysis: Analysis) -> dict[str, Any]:
    """다른 회차 리포트를 이 회차 것으로 바꾼다(ID·회차 번호·생성 시각)."""
    out = copy.deepcopy(data)
    out["analysisId"] = analysis.id
    out["presentationId"] = analysis.presentation_id
    out["attemptNo"] = analysis.attempt_no
    out["createdAt"] = to_kst_iso(analysis.created_at)
    validate_report(out)
    return out


def transcript_from_report(data: dict[str, Any]) -> Transcript:
    """저장된 리포트의 대본(태그는 말하기 지표가 다시 붙이므로 지운다)."""
    t = Transcript.model_validate(data["transcript"])
    return Transcript(
        segments=[
            Segment(
                id=s.id,
                start=s.start,
                end=s.end,
                text=s.text,
                words=[Word(text=w.text, start=w.start, end=w.end, tag=None) for w in s.words],
            )
            for s in t.segments
        ]
    )


# ---------------------------------------------------------------- 실행


def _metrics_summary(speech, nv) -> dict[str, Any]:
    m = speech.metrics
    summary: dict[str, Any] = {
        "syllablesPerMin": m.syllables_per_min,
        "fillerCount": m.filler_count,
        "silenceCount": m.silence_count,
        "overTimeSec": m.over_time_sec if m.time_limit_sec is not None else None,
    }
    if nv is not None:
        summary["gazeFrontRatio"] = nv.metrics.gaze_front_ratio
        summary["postureStability"] = nv.metrics.posture_stability
        summary["gesturePerMin"] = nv.metrics.gesture_per_min
    return summary


@dataclass(frozen=True)
class _Slides:
    texts: list[str] | None
    sha: str | None  # 파일 내용 해시(재사용 키)
    missing: bool  # 슬라이드가 있어야 하는데 못 받았다


def _load_slides(p: Presentation, tmp: Path, deps: PipelineDeps) -> _Slides:
    if not p.slides_key:
        return _Slides(None, None, False)
    dest = tmp / "slides.pdf"
    try:
        deps.storage.download(p.slides_key, dest)
    except StorageError:
        # 슬라이드는 부가 기능이라 못 받아도 분석은 계속한다.
        logger.warning("slides could not be downloaded; analyzing without slides")
        return _Slides(None, None, True)
    return _Slides(deps.extract_slides(dest), file_sha256(dest), False)


def _try_rebase(data: dict[str, Any], analysis: Analysis) -> dict[str, Any] | None:
    try:
        return rebase_report(data, analysis)
    except (ReportInvalid, ValidationError, KeyError, TypeError) as e:
        # 내용(대본 등)이 섞일 수 있어 예외 종류만 남긴다.
        logger.warning("stored report not reusable (%s); recomputing", type(e).__name__)
        return None


def _try_transcript(data: dict[str, Any]) -> Transcript | None:
    try:
        return transcript_from_report(data)
    except (ValidationError, KeyError, TypeError) as e:
        logger.warning("stored transcript not reusable (%s); transcribing", type(e).__name__)
        return None


def cleanup_stale_workdirs(base: Path, now: float | None = None) -> int:
    """강제 종료로 남은 오래된 임시 폴더를 지운다. 지운 개수."""
    now = time.time() if now is None else now
    removed = 0
    for d in base.glob(f"{WORKDIR_PREFIX}*"):
        try:
            if d.is_dir() and now - d.stat().st_mtime > STALE_WORKDIR_SEC:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        logger.info("removed %d stale work dir(s)", removed)
    return removed


def _execute(
    db: Session, analysis: Analysis, p: Presentation, tmp: Path, deps: PipelineDeps
) -> None:
    settings, aid = deps.settings, analysis.id
    purpose = Purpose(p.purpose)
    ctype = analysis.content_type or ""
    if not settings.openai_llm_model.strip():
        raise RuntimeError("OPENAI_LLM_MODEL is not set")  # 설정 문제 → INTERNAL_ERROR

    # ---- EXTRACTING_AUDIO 0~10
    duration = measure_duration(analysis, deps.storage, settings, deps.probe)
    set_progress(db, aid, AnalysisStage.EXTRACTING_AUDIO, 2)
    video = tmp / "video"
    deps.storage.download(analysis.video_key or "", video)
    video_sha = file_sha256(video)
    slides = _load_slides(p, tmp, deps)
    cache_key = None if slides.missing else make_cache_key(video_sha, p, slides.sha)
    extra = {
        "video_sha256": video_sha,
        "cache_key": cache_key,
        "pipeline_version": PIPELINE_VERSION,
    }

    same = _previous(db, p.user_id, aid, cache_key=cache_key) if cache_key else None
    same_art = db.get(AnalysisArtifact, same.analysis_id) if same is not None else None
    same_art_data = copy.deepcopy(same_art.data) if same_art is not None else None
    db.commit()  # 읽기 트랜잭션을 길게 열어 두지 않는다
    reused = _try_rebase(same.data, analysis) if same is not None else None
    if reused is not None:
        logger.info("Analysis %s reuses a previous report (same video and settings)", aid)
        # 중간 결과도 함께 복사해 이 회차도 재계산할 수 있게 한다(원본 회차를 지워도 남는다).
        complete(db, analysis, reused, duration, deps.now(), artifacts=same_art_data, **extra)
        return
    set_progress(db, aid, AnalysisStage.EXTRACTING_AUDIO, 5)

    prev = _previous(db, p.user_id, aid, video_sha256=video_sha)
    db.commit()
    transcript: Transcript | None = None
    if prev is not None:
        transcript = _try_transcript(prev.data)
        if transcript is not None:
            logger.info("Analysis %s reuses a previous transcript (same video)", aid)
    max_sec = float(settings.max_video_seconds)
    audio_path: Path | None = None
    if transcript is None:
        audio_path = deps.extract_audio(
            video, tmp / "audio.mp3", ctype, max_seconds=max_sec, expected_duration=duration
        )

    # ---- TRANSCRIBING 10~40
    set_progress(db, aid, AnalysisStage.TRANSCRIBING, 10)
    if audio_path is not None:  # 재사용할 대본이 없을 때만 받아쓴다
        transcript = deps.transcribe(audio_path, deps.stt_client()).transcript
    if transcript is None:
        raise RuntimeError("no transcript")
    set_progress(db, aid, AnalysisStage.TRANSCRIBING, 35)
    speech = analyze_speech(transcript, duration, p.time_limit_sec)

    # ---- ANALYZING_NONVERBAL 40~65
    set_progress(db, aid, AnalysisStage.ANALYZING_NONVERBAL, 40)
    # 원본 해상도 프레임을 한 장씩 메모리로 받아 자세(640px)·얼굴(머리 영역 잘라 냄)을 본다.
    obs = deps.observe_video(
        video,
        ctype,
        Path(settings.mediapipe_face_model_path),
        Path(settings.mediapipe_pose_model_path),
        fps=FRAME_FPS,
        max_seconds=max_sec,
        expected_duration=duration,
    )
    nv = nonverbal.analyze_nonverbal(obs, duration_sec=duration)

    # ---- ANALYZING_CONTENT 65~80
    set_progress(db, aid, AnalysisStage.ANALYZING_CONTENT, 65)
    llm_client = deps.llm_client()
    ct = deps.analyze_content(
        ContentInput(
            transcript=speech.transcript,
            purpose=purpose,
            metrics_summary=_metrics_summary(speech, nv),
            slides=slides.texts,
        ),
        llm_client,
        settings,
    )

    # ---- GENERATING_FEEDBACK 80~100
    set_progress(db, aid, AnalysisStage.GENERATING_FEEDBACK, 80)
    scores = score_presentation(purpose, speech, nv, ct)
    fb = deps.generate_feedback(
        FeedbackInput(
            purpose=purpose,
            duration_sec=duration,
            transcript=speech.transcript,
            speech=speech,
            nonverbal=nv,
            content=ct,
            category_scores=scores.categories,
        ),
        llm_client,
        settings,
    )
    set_progress(db, aid, AnalysisStage.GENERATING_FEEDBACK, 95)
    report = build_report(
        ReportParts(
            analysis_id=aid,
            presentation_id=analysis.presentation_id,
            attempt_no=analysis.attempt_no,
            purpose=purpose,
            duration_sec=duration,
            created_at=analysis.created_at,
            scores=scores,
            speech=speech,
            nonverbal=nv,
            content=ct,
            feedback=fb,
        )
    )
    artifacts = build_artifacts(
        pipeline_version=PIPELINE_VERSION,
        purpose=purpose,
        duration_sec=duration,
        time_limit_sec=p.time_limit_sec,
        transcript=transcript,
        speech_metrics=speech.metrics.model_dump(mode="json", by_alias=True),
        frames=obs,
        content=ct,
        feedback=fb,
    )
    complete(db, analysis, report, duration, deps.now(), artifacts=_storable(artifacts), **extra)


def _storable(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    """JSON으로 저장할 수 없는 값(numpy 스칼라·NaN 등)이 있으면 중간 결과는 저장하지 않는다.

    중간 결과는 재계산용 부가 데이터라, 이 때문에 이미 비용을 쓴 분석을 실패시키지 않는다.
    """
    try:
        json.dumps(artifacts, allow_nan=False)
    except (TypeError, ValueError) as e:
        logger.warning("artifacts not storable (%s); skipping", type(e).__name__)
        return None
    return artifacts


def run_pipeline(
    analysis_id: str, *, session_factory: sessionmaker[Session], deps: PipelineDeps
) -> str:
    """결과: "completed" | "failed" | "cancelled" | "skipped"."""
    with session_factory() as db:
        if not claim(db, analysis_id, deps.now()):
            logger.info("Analysis %s not claimable; skipped", analysis_id)
            return "skipped"
        analysis = db.get(Analysis, analysis_id)
        p = db.get(Presentation, analysis.presentation_id) if analysis else None
        db.commit()  # 긴 작업 동안 읽기 트랜잭션을 열어 두지 않는다(expire_on_commit=False)
        if analysis is None or p is None:
            return "cancelled"
        try:
            base = Path(deps.settings.work_dir or tempfile.gettempdir())
            base.mkdir(parents=True, exist_ok=True)
            cleanup_stale_workdirs(base)
            with tempfile.TemporaryDirectory(prefix=WORKDIR_PREFIX, dir=base) as tmp:
                _execute(db, analysis, p, Path(tmp), deps)
            return "completed"
        except Cancelled:
            db.rollback()
            logger.info("Analysis %s was deleted or changed; stopped", analysis_id)
            return "cancelled"
        except AnalysisFailure as f:
            logger.info("Analysis %s failed: %s", analysis_id, f.code)
            fail(db, analysis_id, f.code, f.message)
            return "failed"
        except Exception as e:
            code = failure_code(e)
            if code == "INTERNAL_ERROR":
                logger.exception("Analysis %s crashed", analysis_id)
            else:
                # 예외 메시지에는 경로·응답 본문이 없도록 각 모듈이 정리해 두었다.
                logger.info("Analysis %s failed: %s (%s)", analysis_id, code, type(e).__name__)
            fail(db, analysis_id, code, FAILURE_MESSAGES[code])
            return "failed"
