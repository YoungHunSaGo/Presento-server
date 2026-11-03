from typing import Literal

from fastapi import APIRouter

from app.schemas.common import CamelModel

router = APIRouter(tags=["health"])


class HealthResponse(CamelModel):
    status: Literal["ok"]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")
