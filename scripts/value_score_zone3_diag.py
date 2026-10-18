"""가치평가 전략 구역 ③ 사후 점검 (판정 아님) — 결과 본 뒤 점검 2가지.

    python scripts/value_score_zone3_diag.py price   # 17종목 검산: 복원한 실제 주가 vs Tiingo 원시 종가
    python scripts/value_score_zone3_diag.py cost    # 비용 하락 원인: 연도별 매매·회전율, 비용 종류별·사유별

가격은 value_score_zone3_data(구역 가드)로만 읽고, Tiingo 원시 종가 캐시는 2021-12-31 넘는 줄에서 읽기를 멈춘다.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
import value_score_zone3_data as vd  # noqa: E402

RAW_DIRS = [ROOT / "data" / "cache" / p for p in ("sp500/tiingo", "smallmid/tiingo", "sw1/tiingo", "sw1/ohlc_tiingo")]


def read_raw(path: Path) -> pd.Series:
    """Tiingo 원시 종가(raw_close)를 2011-06-01 ~ 2021-12-31만 읽는다."""
    out = {}
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = [h.strip().lower() for h in next(rd)]
        i_d = head.index("date") if "date" in head else 0
        i_c = head.index("raw_close")
        for row in rd:
            d = pd.Timestamp(row[i_d][:10])
            if d > z.ZONE_LAST:
                break
            if d >= z.ZONE_FIRST and row[i_c]:
                out[d] = float(row[i_c])
    return pd.Series(out, dtype=float).sort_index()


def step_price() -> None:
    data = vd.Data()
    gap = pd.read_csv(z.SP500 / "final_gap.csv", encoding="utf-8-sig").set_index("ticker")
    yf_t = [t for t in data.raw if str(gap.at[t, "price_source"]).startswith("yfinance")]
    rows = []
    for t in yf_t:
        src = next((d / f"{t}.csv" for d in RAW_DIRS if (d / f"{t}.csv").exists()), None)
        if src is None:
            continue
        raw = read_raw(src)
        mine = data.raw[t]
        both = mine.index.intersection(raw.index)
        if not len(both):
            continue
        err = (mine[both] / raw[both] - 1.0).abs() * 100.0
        rows.append({"ticker": t, "source": src.parent.relative_to(ROOT).as_posix(), "days": len(both),
                     "first": both.min().date(), "last": both.max().date(),
                     "median_err_pct": float(err.median()), "max_err_pct": float(err.max()),
                     "max_err_date": err.idxmax().date(), "share_days_over_1pct": float((err > 1).mean() * 100)})
    df = pd.DataFrame(rows).sort_values("max_err_pct", ascending=False)
    df.to_csv(vd.c.OUT / "diag_price.csv", index=False, encoding="utf-8-sig")
    allerr = []
    for t in df["ticker"]:
        pass
    print(f"[검산] 종목 {len(df)} · 종목별 오차 중앙값의 중앙값 {df['median_err_pct'].median():.4f}% · "
          f"종목별 최대 오차의 최대 {df['max_err_pct'].max():.3f}% ({df.iloc[0]['ticker']} {df.iloc[0]['max_err_date']})")
    print(df.to_string(index=False))


def step_cost() -> None:
    import value_score_zone3_run as r

    data = vd.Data()
    sc = vd.load_scores(data)  # 판정 실행이 잠근 코드로 다시 만든 캐시
    m = r.Market(data)
    board = r.Board(sc, sc["total"], data.sic2)
    out = {}
    for name, cost in (("비용 전", r.ZERO), ("비용 뒤", r.NO_TAX)):
        res = r.run_value(m, board, cost, record=True)
        b = res["book"]
        out[name] = res
        print(f"[{name}] XIRR {r.xirr(b.flows):.2f}% · 매매 {len(b.log)}건 · 갈아타기 {res['stats']['swaps']}")
    a, c = out["비용 전"]["book"], out["비용 뒤"]["book"]

    def table(b, cost):
        rows = []
        for (i, kind, j, q, usd, why) in b.log:
            why0 = why.split(" ")[0] if why else ""
            gross = q * m.ffill[i, j]
            rows.append({"year": m.days[i].year, "kind": kind, "why": why0, "ticker": m.tickers[j], "usd": usd,
                         "comm": gross * cost.comm, "slip": gross * cost.slip})
        return pd.DataFrame(rows)

    tc = table(c, r.NO_TAX)
    print("\n[연도별 매매 건수 (비용 뒤)]\n" + tc.groupby(["year", "kind"]).size().unstack(fill_value=0).to_string())
    print("\n[사유별 건수·금액·수수료·슬리피지 (비용 뒤, 달러)]\n" +
          tc.groupby("why")[["usd", "comm", "slip"]].agg(["count", "sum"]).round(0).to_string())
    print(f"\n[종류별 비용 합 (달러)] 수수료 {tc['comm'].sum():,.0f} · 슬리피지 {tc['slip'].sum():,.0f} · "
          f"환전(납입 {len(m.month_first)}회 + 시작 + 끝) 약 {(100_000_000 + 1_000_000 * len(m.month_first)) / 1150 * 0.001:,.0f}+끝 청산분")
    # 두 장부가 처음 갈라지는 곳
    la = [(m.days[i].date(), k, m.tickers[j], w) for (i, k, j, q, u, w) in a.log if not w.startswith("공제")]
    lc = [(m.days[i].date(), k, m.tickers[j], w) for (i, k, j, q, u, w) in c.log if not w.startswith("공제")]
    print(f"\n[공제 소진 뺀 매매] 비용 전 {len(la)} · 비용 뒤 {len(lc)}")
    for x, y in zip(la, lc):
        if x != y:
            print(f"처음 갈라지는 매매: 비용 전 {x} / 비용 뒤 {y}")
            break
    sa = {k: v for k, v in out["비용 전"]["stats"].items() if k in ("init", "fills", "delist", "harvest_trades")}
    sb = {k: v for k, v in out["비용 뒤"]["stats"].items() if k in ("init", "fills", "delist", "harvest_trades")}
    print(f"[사건 수] 비용 전 {sa} · 비용 뒤 {sb}")
    print(f"[수시 매도] 비용 전 {out['비용 전']['stats']['sell_events']}\n           비용 뒤 {out['비용 뒤']['stats']['sell_events']}")
    hv = pd.Series(c.value_usd_daily, index=m.days)
    print("[회전율 % (비용 뒤)] " + ", ".join(f"{y} {c.sells_usd.get(y, 0) / hv[hv.index.year == y].mean() * 100:.0f}"
                                         for y in sorted(set(m.days.year))))
    tc.to_csv(vd.c.OUT / "diag_cost_trades.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    {"price": step_price, "cost": step_cost}[sys.argv[1]]()
