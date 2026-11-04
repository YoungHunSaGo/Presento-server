"""분석 회차 상태 갱신(가짜·실제 파이프라인 공용).

모든 상태 변경은 조건부 UPDATE다. 회차가 지워졌거나 다른 워커가 가져갔으면 Cancelled로 멈춘다.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import or_, update
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.models import Analysis, AnalysisStage, AnalysisStatus, Report
from app.services.analysis_state import STALE_PROCESSING
from app.services.media import MediaError, ProbeDuration, ProbeUnavailable
from app.services.pipeline.report import history_values
from app.services.storage import Storage

# N3 error.message (사용자에게 보이는 해요체)
FAILURE_MESSAGES = {
    "NO_SPEECH_DETECTED": "영상에서 목소리를 찾지 못했어요.",
    "NO_FACE_DETECTED": "영상에서 얼굴이 거의 보이지 않았어요. 얼굴이 잘 보이게 다시 찍어 주세요.",
    "CORRUPTED_VIDEO": "영상 파일을 열 수 없어요. 다른 파일로 다시 올려 주세요.",
    "VIDEO_TOO_LONG": "영상은 10분 이하만 분석할 수 있어요.",
    "AI_SERVICE_ERROR": "AI 분석 서비스에 일시적인 문제가 생겼어요. 잠시 후 다시 시도해 주세요.",
    "INTERNAL_ERROR": "분석 중 문제가 생겼어요. 다시 시도해 주세요.",
}


class AnalysisFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = FAILURE_MESSAGES[code]


class Cancelled(Exception):
    """회차가 지워졌거나 더 이상 PROCESSING이 아니다."""


def claim(db: Session, analysis_id: str, now: datetime) -> bool:
    stmt = (
        update(Analysis)
        .where(
            Analysis.id == analysis_id,
            or_(
                Analysis.status == AnalysisStatus.QUEUED,
                (Analysis.status == AnalysisStatus.PROCESSING)
                & (Analysis.started_at < now - STALE_PROCESSING),
            ),
        )
        .values(
            status=AnalysisStatus.PROCESSING,
            stage=AnalysisStage.EXTRACTING_AUDIO,
            progress=0,
            started_at=now,
        )
        .execution_options(synchronize_session=False)
    )
    claimed = db.execute(stmt).rowcount == 1
    db.commit()
    return claimed


def set_progress(db: Session, analysis_id: str, stage: AnalysisStage, progress: int) -> None:
    result = db.execute(
        update(Analysis)
        .where(Analysis.id == analysis_id, Analysis.status == AnalysisStatus.PROCESSING)
        .values(stage=stage, progress=progress)
    )
    db.commit()
    if result.rowcount != 1:
        raise Cancelled


def fail(db: Session, analysis_id: str, code: str, message: str) -> None:
    db.rollback()
    db.execute(
        update(Analysis)
        .where(
            Analysis.id == analysis_id,
            Analysis.status.in_([AnalysisStatus.QUEUED, AnalysisStatus.PROCESSING]),
        )
        .values(status=AnalysisStatus.FAILED, stage=None, error_code=code, error_message=message)
    )
    db.commit()


def measure_duration(
    analysis: Analysis, storage: Storage, settings: Settings, probe: ProbeDuration
) -> float:
    if not analysis.video_key or not analysis.content_type:
        raise AnalysisFailure("CORRUPTED_VIDEO")
    url = storage.presign_get_internal(analysis.video_key, 600)
    try:
        duration = probe(url, analysis.content_type, 120.0, True)
    except ProbeUnavailable:
        # 서버 쪽 문제(ffprobe 없음·시간 초과·스토리지 연결 실패): 사용자 파일 탓이 아니다.
        raise AnalysisFailure("INTERNAL_ERROR") from None
    except MediaError:
        raise AnalysisFailure("CORRUPTED_VIDEO") from None
    if duration is None:
        raise AnalysisFailure("CORRUPTED_VIDEO")
    if duration > settings.max_video_seconds:
        raise AnalysisFailure("VIDEO_TOO_LONG")
    return duration


def complete(
    db: Session,
    analysis: Analysis,
    report: dict[str, Any],
    duration: float,
    now: datetime,
    **extra: Any,
) -> None:
    """리포트 저장 + 회차 요약값(P2·P3·P7) 갱신. extra는 Analysis 컬럼(해시 등)."""
    result = db.execute(
        update(Analysis)
        .where(Analysis.id == analysis.id, Analysis.status == AnalysisStatus.PROCESSING)
        .values(
            status=AnalysisStatus.COMPLETED,
            stage=None,
            progress=100,
            duration_sec=round(duration, 2),
            completed_at=now,
            **history_values(report),
            **extra,
        )
    )
    if result.rowcount != 1:
        db.rollback()
        raise Cancelled
    db.add(Report(analysis_id=analysis.id, data=report))
    db.commit()
