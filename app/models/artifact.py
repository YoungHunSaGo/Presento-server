from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, utcnow

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class AnalysisArtifact(Base):
    """회차별 파이프라인 중간 결과(재계산용). 사용자 대본이 들어 있다.

    회차(N6)·발표(P5)를 지우면 함께 지워진다(ORM cascade + FK ON DELETE CASCADE).
    형식은 app/services/pipeline/artifacts.py 참고.
    """

    __tablename__ = "analysis_artifacts"

    analysis_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSON)
    size_bytes: Mapped[int] = mapped_column(Integer)  # 저장한 JSON 크기(용량 확인용)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analysis: Mapped[Analysis] = relationship(back_populates="artifact")
