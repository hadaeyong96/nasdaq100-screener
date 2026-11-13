"""실적급등 매도 조건 비교(pead_exits) 성적 보기 — 가상 장부 DB를 읽기만 한다 (docs/prereg/pead_exits.md 3~7장).

  python scripts/pead_exits_report.py            # 로컬 virtual_books.db (드라이브 원본을 받은 것)
  python scripts/pead_exits_report.py --db x.db
중간 확인용 숫자다. 규칙은 바꾸지 않는다. 이긴 조건으로 편지 기준을 바꿀지는 사용자가 정한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    import pandas as pd

    from core import pead_exits as px
    from core import pead_validation as pv
    from engine import pead_exits as ex
    from store import pead_exits as xst
    from store import pead_watch as st
    from store import virtual_books as vbstore

    ap = argparse.ArgumentParser(description="실적급등 매도 조건 비교 성적(읽기만)")
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    p = px.params_from(ex.verify_prereg())
    conn = vbstore.connect(Path(a.db or vbstore.VBOOKS_DB_PATH))
    st.ensure_schema(conn)
    xst.ensure_schema(conn)
    start = st.get_meta(conn, xst.META_START)
    if not start:
        print("[pead_exits] 시작일 없음 — 아직 첫 정규 실행 전")
        conn.close()
        return 0
    books = {"main": [t for t in st.load_trades(conn) if t["ledger"] == "main"],
             **{r: xst.load_trades(conn, r) for r in xst.TABLES}}
    rep = px.report(books, st.load_signals(conn), xst.load_plans(conn), start, p)
    rep["portfolio"] = {}
    for b in xst.BOOKS:  # 15칸 누적: 평가액·QQQ 100% 대비·최대 낙폭 (pv.pf_summary)
        rows = xst.load_portfolio(conn, b)
        fx = next((r.get("fx") for r in reversed(rows) if r.get("fx")), None)
        rep["portfolio"][b] = pv.pf_summary(pd.DataFrame(rows), fx) if rows and fx else None
    conn.close()
    print(f"[pead_exits] 시작일 {start} · 끝난 매매 {rep['counts']} · 단계 {rep['milestone']}")
    print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
