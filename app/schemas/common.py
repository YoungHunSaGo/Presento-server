from typing import Any

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    """요청/응답 모델 공통 베이스. JSON 필드명은 camelCase."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ErrorDetail(CamelModel):
    code: str
    message: str
    details: Any = None


class ErrorResponse(CamelModel):
    error: ErrorDetail
