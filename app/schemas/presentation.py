from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator
from pydantic_core import PydanticCustomError

from app.models.enums import AnalysisStatus, Purpose
from app.schemas.common import CamelModel, KstDatetime

Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
# strict: JSON의 true나 "300"을 숫자로 바꿔 받지 않는다.
TimeLimitSec = Annotated[int, Field(ge=1, le=600, strict=True)]  # 영상 최대 10분과 같게


class PresentationCreate(CamelModel):
    title: Title
    purpose: Purpose
    time_limit_sec: TimeLimitSec | None = None


class PresentationUpdate(CamelModel):
    """바꿀 필드만 보낸다. timeLimitSec에 null을 보내면 제한 시간을 없앤다."""

    title: Title | None = None
    purpose: Purpose | None = None
    time_limit_sec: TimeLimitSec | None = None

    # 기본값(필드 생략)에는 실행되지 않고, null을 명시적으로 보낸 경우에만 실행된다.
    @field_validator("title", "purpose")
    @classmethod
    def _no_null(cls, v: object) -> object:
        if v is None:
            raise PydanticCustomError("null_not_allowed", "value cannot be null")
        return v


class PresentationResponse(CamelModel):
    id: str
    title: str
    purpose: Purpose
    time_limit_sec: int | None
    has_slides: bool
    attempt_count: int
    latest_score: int | None
    created_at: KstDatetime


class AttemptSummary(CamelModel):
    analysis_id: str
    attempt_no: int
    status: AnalysisStatus
    overall_score: int | None
    duration_sec: float | None
    created_at: KstDatetime


class PresentationDetailResponse(PresentationResponse):
    attempts: list[AttemptSummary]


class PresentationListResponse(CamelModel):
    items: list[PresentationResponse]
    page: int
    size: int
    total: int


class HistoryItem(CamelModel):
    attempt_no: int
    analysis_id: str
    created_at: KstDatetime
    overall_score: int
    scores: dict[str, int]
    filler_count: int
    syllables_per_min: int


class HistoryResponse(CamelModel):
    items: list[HistoryItem]


class SlidesUploadRequest(CamelModel):
    file_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=255)
    ]
    content_type: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    file_size: Annotated[int, Field(ge=1, strict=True)]


class UploadUrlResponse(CamelModel):
    upload_url: str
    method: Literal["PUT"] = "PUT"
    headers: dict[str, str]
    expires_in_sec: int
