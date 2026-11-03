from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, utcnow

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class Report(Base):
    """N4 응답 전체(camelCase JSON). 저장 전에 ReportResponse 스키마로 검증한다."""

    __tablename__ = "reports"

    analysis_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("analyses.id", ondelete="CASCADE"), primary_key=True
    )
    data: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analysis: Mapped[Analysis] = relationship(back_populates="report")
