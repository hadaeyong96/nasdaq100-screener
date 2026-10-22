"""실패 구간 분석기 1·5단계 — 「RSI·MACD·일목 신호 전략」(6번 장부 규칙) 2012~2021 다시 뽑기 (main 코드 그대로).

규칙: core.picks_book.process_day + core.state.ICHIMOKU_RULES(6번 「추천 종목 따라하기」와 같음 — 신규 진입 A1·B만, 자리 10,
총자산 ÷ 10, t일 신호 → t+1 종가, 손절·E1·E2·E3·A1 만료 때 전량, 남는 돈 SGOV, 수수료 0.07%, 연말 세금 22%·원천징수 15%).
과거에는 6번 장부가 없었으므로 새로 계산한 곡선이다(기존 숫자 없음). 다른 점(분석 선택, 계획 문서 구현 메모):
- 명단: 시점별 나스닥100(전 달 말, scripts/sw1_data_check.ndx_members — 3단 거르기·탐색 B와 같음), 가격은 3단 거르기 캐시
  (data/cache/triple_screen/px, 2021-12-31에서 멈추는 읽기). 출처가 없는 명단 종목은 뺀다.
- 실적 임박 필터: 과거 시점별 실적 발표일 무료 자료가 없어 항상 통과(P5 백테스트와 같은 처리).
- SGOV 자리(남는 돈)는 BIL(2007~), 분할은 가격에 이미 반영돼 0.
출력(<출력 폴더>): RSI_MACD_일목__QQQ기준.csv(date, strategy, benchmark=QQQ 분배금 재투자), rsi_trades_raw.csv, rsi_signals_raw.csv
실행: python <이 파일> <출력 폴더>   (작업 폴더 = research/failure-analysis 루트)
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else None  # deep_rsi.py가 도구만 가져갈 때는 없음
PX = Path(os.environ.get("FA_PX", r"C:\projects\nasdaq100-screener\data\cache\triple_screen\px"))
LAST = pd.Timestamp("2021-12-31")
START = pd.Timestamp("2012-01-03")

from core import picks_book as pb  # noqa: E402
from core import state as st  # noqa: E402
from core.indicators import compute_indicators  # noqa: E402
from engine.daily import load_config  # noqa: E402


def read_guarded(path: Path) -> pd.DataFrame:
    """OHLCV CSV → DataFrame. 2021-12-31 넘는 줄에서 멈춘다(봉인 ⓑ)."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = next(rd)
        for row in rd:
            d = pd.Timestamp(row[0][:10])
            if d > LAST:
                break
            rows.append([d] + [float(x) if x not in ("", "nan") else np.nan for x in row[1:]])
    df = pd.DataFrame(rows, columns=["date"] + head[1:]).set_index("date")
    assert df.index.max() <= LAST
    return df


def members() -> dict[pd.Timestamp, set[str]]:
    import sw1_data_check as sw1

    m = sw1.ndx_members()
    return {k: v for k, v in m.items() if pd.Timestamp("2011-12-31") <= k <= pd.Timestamp("2021-11-30")}


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    mem = members()
    files = {p.stem: p for p in PX.glob("*.csv")}
    raw = {t: read_guarded(files[t]) for t in sorted(set().union(*mem.values())) if t in files}
    missing = sorted(set().union(*mem.values()) - set(raw))
    qqq, bil = read_guarded(files["QQQ"]), read_guarded(files["BIL"])
    frames = {t: compute_indicators(df[["open", "high", "low", "close", "volume"]], cfg) for t, df in raw.items()}
    days = qqq.index[(qqq.index >= START) & (qqq.index <= LAST)]
    me_keys = sorted(mem)
    year_ends = set(pd.Series(days, index=days).groupby(days.year).max())
    params = pb.PicksParams(name="RSI·MACD·일목", slots=int(cfg["virtual_books"]["picks"]["slots"]),
                            commission_pct=float(cfg["virtual_books"]["commission_pct"]), slippage_pct=0.0,
                            tax_rate_pct=float(cfg["virtual_books"]["capital_gains_rate_pct"]),
                            dividend_withholding_pct=float(cfg["virtual_books"]["dividend_withholding_pct"]),
                            cash_ticker="SGOV", progress_on_skipped_adds=bool(cfg["virtual_books"]["picks"]["progress_on_skipped_adds"]))
    book = pb.new_book(params.name, days[0], 100_000.0)
    rows, trades, signals = [], [], []
    for d in days:
        prev_me = max(k for k in me_keys if k < d.replace(day=1)) if any(k < d.replace(day=1) for k in me_keys) else me_keys[0]
        uni = sorted(t for t in mem[prev_me] if t in frames)
        closes = {t: float(df.at[d, "close"]) for t, df in raw.items() if d in df.index and not np.isnan(df.at[d, "close"])}
        divs = {t: float(df.at[d, "div"]) for t, df in raw.items() if d in df.index and df.at[d, "div"] > 0}
        if d in bil.index and bil.at[d, "div"] > 0:
            divs["SGOV"] = float(bil.at[d, "div"])
        scan = list(dict.fromkeys(uni + sorted(book["positions"]) + [o["ticker"] for o in book["pending_sells"]]))
        book, row, tr, sg = pb.process_day(book, d, frames=frames, closes=closes, sgov_close=float(bil.at[d, "close"]),
                                           dividends=divs, splits={}, scan_tickers=scan, earnings_map={}, rules=st.ICHIMOKU_RULES,
                                           params=params, cfg=cfg, year_end=d in year_ends, briefing=None)
        rows.append(row)
        trades.extend(tr)
        signals.extend(sg)
    daily = pd.DataFrame(rows).set_index("date")
    qtr = (qqq["close"] + qqq["div"]) / qqq["close"].shift(1)
    qidx = qtr.loc[days].fillna(1.0)
    qidx.iloc[0] = 1.0
    bench = qidx.cumprod() * 100_000.0
    pd.DataFrame({"strategy": daily["value"], "benchmark": bench, "sgov_value": daily["sgov_value"], "cash": daily["cash"],
                  "n_positions": daily["qty"]}, index=pd.DatetimeIndex(days, name="date")).to_csv(OUT / "RSI_MACD_일목__QQQ기준.csv")
    pd.DataFrame(trades).to_csv(OUT / "rsi_trades_raw.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(signals).to_csv(OUT / "rsi_signals_raw.csv", index=False, encoding="utf-8-sig")
    yrs = (days[-1] - days[0]).days / 365.25
    info = {"cagr_strategy": (float(daily["value"].iat[-1]) / 100_000.0) ** (1 / yrs) - 1,
            "cagr_qqq_tr": float(bench.iat[-1] / 100_000.0) ** (1 / yrs) - 1, "trades": len(trades),
            "members_without_price": missing, "seconds": time.time() - t0,
            "note": "세전(연말 세금은 장부 현금에서 냄)·실적 필터 없음, QQQ는 분배금 재투자 세전"}
    (OUT / "rsi_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", {k: v for k, v in info.items() if k != "members_without_price"}, "가격 없는 명단 종목", len(missing))


if __name__ == "__main__":
    main()
