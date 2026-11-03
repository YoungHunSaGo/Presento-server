"""마일스톤 4 가짜 파이프라인.

외부 AI(OpenAI)는 호출하지 않는다. 명세 6장의 stage 순서·progress 범위대로 단계를 진행하고,
명세 5장 예시 리포트에서 analysisId·presentationId·attemptNo·durationSec만 실제 값으로
바꿔 저장한다.
durationSec는 ffprobe로 실제 영상 길이를 잰다(로컬 실행, API 비용 없음).

모든 상태 변경은 조건부 UPDATE다. 회차가 지워졌거나 다른 워커가 가져갔으면 조용히 멈춘다.
"""

import copy
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import or_, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import STAGE_PROGRESS, Analysis, AnalysisStage, AnalysisStatus, Report
from app.schemas.report import ReportResponse
from app.services.analysis_state import STALE_PROCESSING
from app.services.media import MediaError, ProbeDuration, ProbeUnavailable
from app.services.storage import Storage

logger = logging.getLogger(__name__)

SAMPLE_REPORT_PATH = Path(__file__).with_name("sample_report.json")

FAILURE_MESSAGES = {
    "CORRUPTED_VIDEO": "영상 파일을 열 수 없어요. 다른 파일로 다시 올려 주세요.",
    "VIDEO_TOO_LONG": "영상은 10분 이하만 분석할 수 있어요.",
    "INTERNAL_ERROR": "분석 중 문제가 생겼어요. 다시 시도해 주세요.",
}


class AnalysisFailure(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.message = FAILURE_MESSAGES[code]


class Cancelled(Exception):
    """회차가 지워졌거나 더 이상 PROCESSING이 아니다."""


@lru_cache
def _sample_report() -> dict[str, Any]:
    return json.loads(SAMPLE_REPORT_PATH.read_text(encoding="utf-8"))


def build_fake_report(analysis: Analysis, duration_sec: float) -> dict[str, Any]:
    data = copy.deepcopy(_sample_report())
    data["analysisId"] = analysis.id
    data["presentationId"] = analysis.presentation_id
    data["attemptNo"] = analysis.attempt_no
    data["durationSec"] = round(duration_sec, 2)
    # 저장 전에 명세 스키마로 검증한다.
    return ReportResponse.model_validate(data).model_dump(mode="json", by_alias=True)


def _claim(db: Session, analysis_id: str, now: datetime) -> bool:
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


def _set_progress(db: Session, analysis_id: str, stage: AnalysisStage, progress: int) -> None:
    result = db.execute(
        update(Analysis)
        .where(Analysis.id == analysis_id, Analysis.status == AnalysisStatus.PROCESSING)
        .values(stage=stage, progress=progress)
    )
    db.commit()
    if result.rowcount != 1:
        raise Cancelled


def _fail(db: Session, analysis_id: str, code: str, message: str) -> None:
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


def _measure_duration(
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


def _complete(db: Session, analysis: Analysis, report: dict[str, Any], duration: float, now):
    result = db.execute(
        update(Analysis)
        .where(Analysis.id == analysis.id, Analysis.status == AnalysisStatus.PROCESSING)
        .values(
            status=AnalysisStatus.COMPLETED,
            stage=None,
            progress=100,
            duration_sec=round(duration, 2),
            overall_score=report["overall"]["score"],
            scores={s["category"]: s["score"] for s in report["scores"]},
            filler_count=report["metrics"]["speech"]["fillerCount"],
            syllables_per_min=report["metrics"]["speech"]["syllablesPerMin"],
            completed_at=now,
        )
    )
    if result.rowcount != 1:
        db.rollback()
        raise Cancelled
    db.add(Report(analysis_id=analysis.id, data=report))
    db.commit()


def run_fake_pipeline(
    analysis_id: str,
    *,
    session_factory: sessionmaker[Session],
    storage: Storage,
    settings: Settings,
    probe: ProbeDuration,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """결과: "completed" | "failed" | "cancelled" | "skipped"."""
    half = settings.fake_stage_seconds / 2
    with session_factory() as db:
        if not _claim(db, analysis_id, datetime.now(UTC)):
            logger.info("Analysis %s not claimable; skipped", analysis_id)
            return "skipped"
        analysis = db.get(Analysis, analysis_id)
        if analysis is None:
            return "cancelled"

        try:
            duration = 0.0
            for stage, (start, end) in STAGE_PROGRESS.items():
                _set_progress(db, analysis_id, stage, start)
                if stage == AnalysisStage.EXTRACTING_AUDIO:
                    duration = _measure_duration(analysis, storage, settings, probe)
                sleep(half)
                _set_progress(db, analysis_id, stage, (start + end) // 2)
                sleep(half)
            report = build_fake_report(analysis, duration)
            _complete(db, analysis, report, duration, datetime.now(UTC))
            return "completed"
        except Cancelled:
            db.rollback()
            logger.info("Analysis %s was deleted or changed; stopped", analysis_id)
            return "cancelled"
        except AnalysisFailure as f:
            logger.info("Analysis %s failed: %s", analysis_id, f.code)
            _fail(db, analysis_id, f.code, f.message)
            return "failed"
        except Exception:
            logger.exception("Analysis %s crashed", analysis_id)
            _fail(db, analysis_id, "INTERNAL_ERROR", FAILURE_MESSAGES["INTERNAL_ERROR"])
            return "failed"
