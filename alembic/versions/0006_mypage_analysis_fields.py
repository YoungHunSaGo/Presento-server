"""mypage summary columns and composite index on analyses

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.add_column(sa.Column("grade", sa.String(length=4), nullable=True))
        batch.add_column(sa.Column("metric_details", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("filler_counts", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("feedback_counts", sa.JSON(), nullable=True))
        batch.create_index(
            "ix_analyses_presentation_id_status_completed_at",
            ["presentation_id", "status", "completed_at"],
        )


def downgrade() -> None:
    with op.batch_alter_table("analyses") as batch:
        batch.drop_index("ix_analyses_presentation_id_status_completed_at")
        batch.drop_column("feedback_counts")
        batch.drop_column("filler_counts")
        batch.drop_column("metric_details")
        batch.drop_column("grade")
