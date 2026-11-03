from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, new_id, utcnow

if TYPE_CHECKING:
    from app.models.analysis import Analysis


class Presentation(Base):
    __tablename__ = "presentations"
    __table_args__ = (Index("ix_presentations_user_id_updated_at", "user_id", "updated_at"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: new_id("p"))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(100))
    purpose: Mapped[str] = mapped_column(String(20))
    time_limit_sec: Mapped[int | None] = mapped_column(Integer)
    # 마지막으로 발급한 회차 번호. 회차를 지워도 번호를 다시 쓰지 않도록 따로 센다.
    attempt_seq: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    # 업로드가 확인된 슬라이드 PDF 키. 이 값이 있으면 hasSlides = true.
    slides_key: Mapped[str | None] = mapped_column(String(255))
    # P6에서 URL만 발급하고 아직 업로드가 확인되지 않은 키.
    slides_pending_key: Mapped[str | None] = mapped_column(String(255))
    slides_pending_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    # P2 "최근 수정순" 기준. 사용자에게 보이는 변경(P4 수정, 슬라이드 반영, 새 회차)에서만
    # 직접 갱신한다. onupdate를 쓰지 않는 이유: 내부 정리용 쓰기까지 순서를 바꾸기 때문.
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="presentation",
        cascade="all, delete-orphan",
        passive_deletes=True,  # DB의 ON DELETE CASCADE에 맡겨 행마다 DELETE하지 않는다
        order_by="Analysis.attempt_no",
    )
