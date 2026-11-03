from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, new_id, utcnow


class AnalysisUsage(Base):
    """하루 분석 횟수 제한용 기록. 발표·회차를 지워도 남아서 제한을 우회할 수 없다."""

    __tablename__ = "analysis_usage"
    __table_args__ = (Index("ix_analysis_usage_user_id_created_at", "user_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("q"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    analysis_id: Mapped[str] = mapped_column(String(32))  # FK 없음: 회차가 지워져도 남긴다
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
