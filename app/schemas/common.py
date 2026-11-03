from datetime import UTC, datetime, timedelta, timezone
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, PlainSerializer
from pydantic.alias_generators import to_camel

# 한국은 서머타임이 없어 고정 오프셋으로 충분하다(컨테이너 tzdata 의존 없음).
KST = timezone(timedelta(hours=9), "KST")


def to_kst_iso(dt: datetime) -> str:
    """명세 날짜 형식: 2026-11-03T17:04:00+09:00. naive 값(SQLite)은 UTC로 본다."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(KST).isoformat(timespec="seconds")


KstDatetime = Annotated[datetime, PlainSerializer(to_kst_iso, return_type=str, when_used="json")]


class CamelModel(BaseModel):
    """요청/응답 모델 공통 베이스. JSON 필드명은 camelCase."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorDetail(CamelModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(CamelModel):
    error: ErrorDetail
