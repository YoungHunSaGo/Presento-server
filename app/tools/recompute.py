"""저장된 중간 결과로 점수·리포트를 다시 계산한다(LLM·Whisper 재호출 없음).

평가와 시연은 Docker 안에서만 실행한다(README). 예:

    docker compose exec worker python -m app.tools.recompute a-1234 a-5678
    docker compose exec worker python -m app.tools.recompute --all --dry-run
    docker compose exec worker python -m app.tools.recompute --stats

--dry-run은 저장하지 않고 이전/새 점수만 보여준다. 대본 내용은 출력하지 않는다.
"""

import argparse
import sys

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.database import make_engine, make_session_factory
from app.models import AnalysisArtifact
from app.services.pipeline.artifacts import ArtifactMissing, recompute
from app.services.pipeline.report import ReportInvalid


def _summary(r: dict) -> str:
    cats = " ".join(f"{s['category'][:3]}={s['score']}" for s in r["scores"])
    return f"{r['overall']['score']}({r['overall']['grade']}) {cats}"


def main(argv: list[str] | None = None, session_factory=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m app.tools.recompute", description=__doc__)
    ap.add_argument("analysis_ids", nargs="*", help="재계산할 회차 ID")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="중간 결과가 있는 모든 회차")
    mode.add_argument("--stats", action="store_true", help="저장 용량 통계만 보기")
    ap.add_argument("--dry-run", action="store_true", help="저장하지 않고 결과만 보기")
    args = ap.parse_args(argv)
    if (args.all or args.stats) and args.analysis_ids:
        ap.error("회차 ID와 --all/--stats는 함께 쓸 수 없어요")
    if session_factory is None:
        session_factory = make_session_factory(make_engine(get_settings().database_url))

    with session_factory() as db:
        if args.stats:
            n, total, avg, mx = db.execute(
                select(
                    func.count(AnalysisArtifact.analysis_id),
                    func.coalesce(func.sum(AnalysisArtifact.size_bytes), 0),
                    func.coalesce(func.avg(AnalysisArtifact.size_bytes), 0),
                    func.coalesce(func.max(AnalysisArtifact.size_bytes), 0),
                )
            ).one()
            kb = [v / 1024 for v in (total, avg, mx)]
            print(f"artifacts={n} total={kb[0]:.1f}KB avg={kb[1]:.1f}KB max={kb[2]:.1f}KB")
            return 0
        ids = list(args.analysis_ids)
        if args.all:
            ids = list(db.scalars(select(AnalysisArtifact.analysis_id)))
        if not ids:
            ap.error("회차 ID나 --all을 주세요")
        failed = 0
        for aid in ids:
            try:
                old, new, drift = recompute(db, aid, write=not args.dry_run)
            except Exception as e:  # noqa: BLE001 - 한 회차가 실패해도 나머지는 계속
                db.rollback()
                failed += 1
                # 알 수 없는 예외 메시지에는 대본이 섞일 수 있어 종류만 출력한다.
                detail = f": {e}" if isinstance(e, ArtifactMissing | ReportInvalid) else ""
                print(f"{aid}: 실패 {type(e).__name__}{detail}")
                continue
            mark = "(저장 안 함)" if args.dry_run else "저장함"
            warn = (
                " [피드백 근거 변경됨: 문장 속 수치·구간이 새 지표와 다를 수 있음]" if drift else ""
            )
            print(f"{aid}: {_summary(old)} -> {_summary(new)} {mark}{warn}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
