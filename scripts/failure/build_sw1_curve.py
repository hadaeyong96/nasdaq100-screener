"""실패 구간 분석기 1단계 — 실적급등 다음날 매수(SW1, 2012~2021) 일간 곡선 (기존 매매 601건 그대로, 곡선만 새로 만듦).

SW1은 매매 단위로 판정했고 장부 곡선이 없었다. 기존 매매 목록(outputs/sw1_trades.csv — sw1-prereg 실행 결과)의
매수일 시가 → 매도일 종가를 그대로 쓰고, 날마다 열려 있는 매매를 같은 비중으로 묶은 달력 시간 포트폴리오를 만든다
(열린 매매가 없는 날은 QQQ 수익). 기준 = QQQ(분배금 재투자). 비용·세금 전(곡선 모양 비교용).
출력: 실적급등SW1__QQQ기준.csv, sw1_check.json(매매당 평균 초과 다시 계산)
실행: python scripts/failure/build_sw1_curve.py <출력 폴더>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r"C:\projects\nasdaq100-screener")
OUT = Path(sys.argv[1])
LAST = pd.Timestamp("2021-12-31")


def load_close(t: str) -> pd.Series:
    """매매 종목 종가(분할 반영): sw1 가격 캐시 → sw1 ohlc → 3단 거르기 px → 이 분석에서 받은 extra_px(WMT)."""
    for p in (MAIN / "data/cache/sw1/prices" / f"{t}.csv", MAIN / "data/cache/sw1/ohlc" / f"{t}.csv",
              MAIN / "data/cache/triple_screen/px" / f"{t}.csv", Path.cwd() / "data/cache/failure/extra_px" / f"{t}.csv"):
        if p.exists():
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            return df["close"]
    raise FileNotFoundError(t)


def main() -> None:
    tr = pd.read_csv(MAIN / "outputs" / "sw1_trades.csv", parse_dates=["buy_day", "sell_day"])
    q = pd.read_csv(Path.cwd() / "data" / "cache" / "failure" / "ind_raw" / "yf_QQQ.csv", index_col=0, parse_dates=True)["Adj Close"]
    q = q[q.index <= LAST]
    days = q.index[(q.index >= tr["buy_day"].min()) & (q.index <= LAST)]
    qret = q.pct_change().reindex(days).fillna(0.0)
    rets = pd.DataFrame(np.nan, index=days, columns=range(len(tr)))
    for k, t in tr.iterrows():
        p = load_close(t["ticker"])
        p = p[p.index <= LAST]
        seg = p[(p.index >= t["buy_day"]) & (p.index <= t["sell_day"])]
        if seg.empty:
            continue
        r = seg.pct_change()
        r.iloc[0] = seg.iloc[0] / t["buy_open"] - 1.0  # 매수일: 시가 → 종가
        r.iloc[-1] = (t["sell_close"] / seg.iloc[-2] - 1.0) if len(seg) > 1 else (t["sell_close"] / t["buy_open"] - 1.0)
        rets.loc[r.index.intersection(days), k] = r.reindex(r.index.intersection(days)).to_numpy()
    port = rets.mean(axis=1, skipna=True)
    n_open = rets.notna().sum(axis=1)
    port = port.where(n_open > 0, qret)
    strat = (1 + port).cumprod() * 100.0
    bench = (1 + qret).cumprod() * 100.0
    pd.DataFrame({"strategy": strat, "benchmark": bench, "open_trades": n_open}, index=pd.DatetimeIndex(days, name="date")).to_csv(
        OUT / "실적급등SW1__QQQ기준.csv")
    chk = {"전략": "실적급등 다음날 매수 SW1", "매매 수": int(len(tr)), "매매당 평균 초과(비용·세금 뒤, 기존 값 그대로)": float(tr["ex_after"].mean()),
           "곡선 연복리(비용 전)": float((strat.iat[-1] / 100) ** (365.25 / (days[-1] - days[0]).days) - 1),
           "QQQ 연복리": float((bench.iat[-1] / 100) ** (365.25 / (days[-1] - days[0]).days) - 1),
           "열린 매매 없는 날 비율": float((n_open == 0).mean())}
    (OUT / "sw1_check.json").write_text(json.dumps(chk, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", chk)


if __name__ == "__main__":
    main()
