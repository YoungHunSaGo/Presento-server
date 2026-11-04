"""마일스톤 4 가짜 파이프라인.

외부 AI(OpenAI)는 호출하지 않는다. 명세 6장의 stage 순서·progress 범위대로 단계를 진행하고,
명세 5장 예시 리포트에서 analysisId·presentationId·attemptNo·durationSec만 실제 값으로
바꿔 저장한다.
durationSec는 ffprobe로 실제 영상 길이를 잰다(로컬 실행, API 비용 없음).

상태 변경은 lifecycle.py의 조건부 UPDATE. 회차가 지워졌거나 다른 워커가 가져갔으면 조용히 멈춘다.
설정 USE_FAKE_PIPELINE=true일 때만 쓴다(프론트 개발용).
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

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.models import STAGE_PROGRESS, Analysis, AnalysisStage
from app.schemas.report import ReportResponse
from app.services.media import ProbeDuration
from app.services.pipeline.lifecycle import (
    FAILURE_MESSAGES,
    AnalysisFailure,
    Cancelled,
)
from app.services.pipeline.lifecycle import claim as _claim
from app.services.pipeline.lifecycle import complete as _complete
from app.services.pipeline.lifecycle import fail as _fail
from app.services.pipeline.lifecycle import measure_duration as _measure_duration
from app.services.pipeline.lifecycle import set_progress as _set_progress
from app.services.storage import Storage

logger = logging.getLogger(__name__)

SAMPLE_REPORT_PATH = Path(__file__).with_name("sample_report.json")


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
