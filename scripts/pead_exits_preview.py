"""실적급등 매도 조건 A·B 미리보기(dry-run, DB에 쓰지 않음) — 실제 신호로 A 매도 예정일·출처, B 목표가·도달 여부를 표로.

  python scripts/pead_exits_preview.py --from 2026-10-01 --to 2026-11-10
빈 임시 DB에서 --from부터 V1과 A·B를 함께 진행한다(A·B 시작일도 --from). 기준일 = 직전 확정 미국 종가일.
주의: A 예정일은 실제 기록과 달리 '지금' yfinance 값이다(실제 기록은 신호를 처음 기록한 실행의 값으로 고정).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def table(res: dict, d_to: str) -> list[dict]:
    """미리보기 결과 → 신호별 줄(반응일 ≤ d_to)."""
    ex = res["exits"]
    a = {(t["ticker"], t["d0"]): t for t in ex["trades"]["exit_a"]}
    b = {(t["ticker"], t["d0"]): t for t in ex["trades"]["exit_b"]}
    out = []
    for j in sorted((j for j in res["judged"] if j["signal"] and j["d0"] <= d_to), key=lambda j: (j["d0"], j["ticker"])):
        k = (j["ticker"], j["d0"])
        ta, tb, pl = a.get(k) or {}, b.get(k) or {}, ex["plans"].get(k) or {}
        out.append({"ticker": k[0], "d0": k[1], "buy_day": ta.get("buy_day") or tb.get("buy_day"),
                    "buy_px": tb.get("buy_px"), "next_date": pl.get("next_date"), "source": pl.get("source"),
                    "a_plan": ta.get("plan_day"), "a_status": ta.get("status"), "a_skip": ta.get("skip"),
                    "b_target": tb.get("target_px"), "b_hit": tb.get("hit_day"), "b_hit_days": tb.get("hit_days"),
                    "b_status": tb.get("status"), "b_sell_day": tb.get("sell_day"), "b_sell_px": tb.get("sell_px"),
                    "guid": pl.get("guid_raise"), "prev_below": pl.get("prev_below_sma200"), "errors": pl.get("errors")})
    return out


def main(argv=None) -> int:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    from data.market_calendar import latest_closed_trading_day
    from engine import pead_watch as ew
    from engine.daily import US_EASTERN, load_config

    ap = argparse.ArgumentParser(description="실적급등 매도 조건 A·B 미리보기(DB에 쓰지 않음)")
    ap.add_argument("--from", dest="start", default="2026-10-01")
    ap.add_argument("--to", default="2026-11-10", help="표에 넣을 마지막 반응일")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args(argv)
    cfg = load_config()
    as_of = a.as_of or latest_closed_trading_day(datetime.now(US_EASTERN))
    with tempfile.TemporaryDirectory() as tmp:
        res = ew.run(cfg, as_of, db_path=Path(tmp) / "preview.db", dry_run=True, start=a.start, exits_start=a.start)
    assert res["status"] == "preview", res["status"]
    if res.get("exits") is None:
        print("[pead_exits] A·B 미리보기 실패:", [w for w in res["warnings"] if "매도 조건" in w])
        return 1
    rows = table(res, a.to)
    print(f"[pead_exits] 미리보기 {a.start} ~ 기준일 {res['as_of']} · 반응일 ≤ {a.to} 신호 {len(rows)}건")
    print("| 종목 | 반응일 | 매수일 | 매수가 | 다음 발표 예정일 | 출처 | A 매도 예정일 | B 목표가 | B 도달 | 가이던스 | 전날 200일선 아래 |")
    print("| --- | --- | --- | ---: | --- | --- | --- | ---: | --- | --- | --- |")
    for r in rows:
        hit = f"예 {r['b_hit']}({r['b_hit_days']}일) → {r['b_sell_day']} 시가 ${r['b_sell_px']:.2f}" if r["b_hit"] else (
            "아니오" if r["b_status"] in ("open", "closed") else (r["b_status"] or ""))
        a_plan = r["a_plan"] or f"({r['a_status']}·{r['a_skip']})"
        print(f"| {r['ticker']} | {r['d0']} | {r['buy_day'] or ''} | {r['buy_px'] or 0:.2f} | {r['next_date'] or '-'} | "
              f"{r['source']} | {a_plan} | {r['b_target'] or 0:.2f} | {hit} | "
              f"{'' if r['guid'] is None else ('예' if r['guid'] else '아니오')} | "
              f"{'' if r['prev_below'] is None else ('예' if r['prev_below'] else '아니오')} |")
    for w in res["warnings"]:
        if "매도 조건" in w or "15칸" in w:
            print("  경고:", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
