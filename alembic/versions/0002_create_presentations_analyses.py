"""create presentations and analyses

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "presentations",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=100), nullable=False),
        sa.Column("purpose", sa.String(length=20), nullable=False),
        sa.Column("time_limit_sec", sa.Integer(), nullable=True),
        sa.Column("slides_key", sa.String(length=255), nullable=True),
        sa.Column("slides_pending_key", sa.String(length=255), nullable=True),
        sa.Column("slides_pending_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_presentations_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_presentations")),
    )
    op.create_index(
        "ix_presentations_user_id_updated_at",
        "presentations",
        ["user_id", "updated_at"],
        unique=False,
    )

    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("presentation_id", sa.String(length=32), nullable=False),
        sa.Column("attempt_no", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("video_key", sa.String(length=255), nullable=True),
        sa.Column("duration_sec", sa.Float(), nullable=True),
        sa.Column("overall_score", sa.Integer(), nullable=True),
        sa.Column("scores", sa.JSON(), nullable=True),
        sa.Column("filler_count", sa.Integer(), nullable=True),
        sa.Column("syllables_per_min", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["presentation_id"],
            ["presentations.id"],
            name=op.f("fk_analyses_presentation_id_presentations"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
        sa.UniqueConstraint(
            "presentation_id", "attempt_no", name=op.f("uq_analyses_presentation_id")
        ),
    )
    op.create_index(
        op.f("ix_analyses_presentation_id"), "analyses", ["presentation_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_analyses_presentation_id"), table_name="analyses")
    op.drop_table("analyses")
    op.drop_index("ix_presentations_user_id_updated_at", table_name="presentations")
    op.drop_table("presentations")
