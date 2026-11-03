from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, new_id, utcnow
from app.models.enums import AnalysisStatus

if TYPE_CHECKING:
    from app.models.presentation import Presentation


class Analysis(Base):
    """발표 한 회차. 상태·진행률·업로드 정보는 마일스톤 4에서 확장한다."""

    __tablename__ = "analyses"
    __table_args__ = (UniqueConstraint("presentation_id", "attempt_no"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("a"))
    presentation_id: Mapped[str] = mapped_column(
        ForeignKey("presentations.id", ondelete="CASCADE"), index=True
    )
    attempt_no: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20), default=AnalysisStatus.WAITING_UPLOAD)
    video_key: Mapped[str | None] = mapped_column(String(255))
    duration_sec: Mapped[float | None] = mapped_column(Float)

    # 완료 시 채우는 점수 요약(P2·P3·P7용). 전체 리포트는 마일스톤 4의 Report에 둔다.
    overall_score: Mapped[int | None] = mapped_column(Integer)
    scores: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    filler_count: Mapped[int | None] = mapped_column(Integer)
    syllables_per_min: Mapped[int | None] = mapped_column(Integer)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    presentation: Mapped[Presentation] = relationship(back_populates="analyses")
