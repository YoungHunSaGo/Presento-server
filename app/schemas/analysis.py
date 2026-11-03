from typing import Annotated

from pydantic import Field, StringConstraints

from app.models.enums import AnalysisStage, AnalysisStatus, VideoSource
from app.schemas.common import CamelModel, KstDatetime
from app.schemas.presentation import UploadUrlResponse


class AnalysisCreateRequest(CamelModel):
    file_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
    ]
    content_type: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    file_size: Annotated[int, Field(ge=1, strict=True)]
    source: VideoSource


class AnalysisCreateResponse(CamelModel):
    analysis_id: str
    attempt_no: int
    status: AnalysisStatus
    upload: UploadUrlResponse


class AnalysisStartResponse(CamelModel):
    analysis_id: str
    status: AnalysisStatus


class AnalysisError(CamelModel):
    code: str
    message: str


class AnalysisStatusResponse(CamelModel):
    analysis_id: str
    presentation_id: str
    attempt_no: int
    status: AnalysisStatus
    stage: AnalysisStage | None
    progress: int
    error: AnalysisError | None
    created_at: KstDatetime
    completed_at: KstDatetime | None


class VideoUrlResponse(CamelModel):
    video_url: str
    expires_in_sec: int
