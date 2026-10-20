"""3단 거르기 탐색 데이터 층 — 시점별 나스닥100 명단·가격 출처·OHLCV 받기·가드 읽기 (수익 계산 없음).

사용:
  python scripts/triple_screen_data.py --step fetch     명단 종목 OHLCV + QQQ·XLY·XLP·BIL 받기 (2021-12-31까지만 저장)
  python scripts/triple_screen_data.py --step coverage  확보율·가격 공백·옛 종가 캐시와 대조 보고 (허용 조회)

명단: 매달 말 나스닥100 시점별 명단(scripts/sw1_data_check.ndx_members, data.universe_history). 신호일이 속한 달의 전 달 말 명단.
가격 출처: 탐색 B가 CIK로 확인한 출처 표(data/cache/sector_pullback/price_sources.csv)를 그대로 쓴다.
  yfinance 출처 → 그 파일 이름의 티커로 yfinance auto_adjust=False Open·High·Low·Close·Volume·Dividends (분할 반영, 배당 미반영)
  Tiingo 출처 → 그 파일 이름의 티커로 Tiingo 원시 open·high·low·close·volume·divCash·splitFactor → 2021-12-31까지의 분할 곱으로 맞춤
가격 파일은 read_guarded로만 읽는다: 2021-12-31 넘는 줄에서 멈춘다(2022년 이후 데이터 금지 가드).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

CACHE = ROOT / "data" / "cache" / "triple_screen"
PX = CACHE / "px"
SOURCES_IN = ROOT / "data" / "cache" / "sector_pullback" / "price_sources.csv"
TABLE = CACHE / "tickers.csv"
LAST = pd.Timestamp("2021-12-31")
STOCK_FROM = "2011-01-01"
ETF_FROM = {"QQQ": "1998-12-01", "XLY": "1998-12-01", "XLP": "1998-12-01", "BIL": "2007-05-01"}
FIRST_USE, LAST_USE = pd.Timestamp("2011-12-31"), pd.Timestamp("2021-11-30")  # 2012-01 ~ 2021-12 신호 달의 전 달 말
LISTING = {"GOOGL": pd.Timestamp("2014-04-03")}  # 탐색 B와 같음: 실제 상장일 전 날짜는 쓰지 않는다


class GuardError(RuntimeError):
    """2022년 이후 데이터에 닿으려 할 때."""


def read_guarded(path: Path, first: str | None = None) -> pd.DataFrame:
    """OHLCV CSV(date, open, high, low, close, volume, div) → DataFrame(index 날짜). 2021-12-31 넘는 줄에서 멈춘다."""
    rows, prev = [], None
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        head = next(rd)
        for row in rd:
            d = pd.Timestamp(row[0][:10])
            if prev is not None and d < prev:
                raise GuardError(f"{path.name}: 날짜가 오름차순이 아님")
            prev = d
            if d > LAST:
                break
            rows.append([d] + [float(x) if x not in ("", "nan") else np.nan for x in row[1:]])
    df = pd.DataFrame(rows, columns=["date"] + head[1:]).set_index("date")
    if first is not None:
        df = df[df.index >= pd.Timestamp(first)]
    if len(df) and df.index.max() > LAST:
        raise GuardError(f"{path.name}: 2021-12-31 뒤 날짜")
    return df


def members() -> dict[pd.Timestamp, set[str]]:
    """{전 달 말: 나스닥100 티커} — 2011-12-31 ~ 2021-11-30."""
    import sw1_data_check as sw1

    m = sw1.ndx_members()
    return {k: v for k, v in m.items() if FIRST_USE <= k <= LAST_USE}


def build_table() -> pd.DataFrame:
    """명단 티커 → (cik, 출처 종류, 받을 티커). 출처 없는 티커는 kind='없음'."""
    mem = members()
    months = pd.Series([t for ts in mem.values() for t in ts]).value_counts()
    src = pd.read_csv(SOURCES_IN, encoding="utf-8-sig").drop_duplicates("ticker").set_index("ticker")
    rows = []
    for t in sorted(months.index):
        if t not in src.index:
            rows.append({"ticker": t, "cik": None, "kind": "없음", "symbol": "", "months": int(months[t]), "note": "탐색 B 제외(CIK 없음)"})
            continue
        r = src.loc[t]
        s = str(r["source"])
        path = str(r["path"]) if isinstance(r["path"], str) else ""
        kind = "yfinance" if s.startswith("yfinance") else "tiingo" if s.startswith("tiingo") else "없음"
        sym = Path(path.replace("\\", "/")).stem if path else ""
        rows.append({"ticker": t, "cik": int(r["cik"]), "kind": kind, "symbol": sym, "months": int(months[t]), "note": s})
    return pd.DataFrame(rows)


def fetch_yf(sym: str, start: str) -> int:
    import yfinance as yf

    p = PX / f"{sym}.csv"
    if p.exists():
        return -1
    h = yf.Ticker(sym).history(start=start, end="2022-01-01", auto_adjust=False)
    time.sleep(0.2)
    if h is None or h.empty:
        pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "div"]).to_csv(p, index=False)
        return 0
    h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
    h = h[h.index <= LAST]
    pd.DataFrame({"date": h.index.strftime("%Y-%m-%d"), "open": h["Open"].values, "high": h["High"].values,
                  "low": h["Low"].values, "close": h["Close"].values, "volume": h["Volume"].values,
                  "div": h["Dividends"].values}).to_csv(p, index=False)
    return len(h)


def fetch_tiingo(sym: str, key: str) -> int:
    import requests

    p = PX / f"{sym}.csv"
    if p.exists():
        return -1
    try:  # URL에 토큰이 들어 있어 예외 메시지를 그대로 내보내지 않는다
        resp = requests.get(f"https://api.tiingo.com/tiingo/daily/{sym}/prices",
                            params={"startDate": STOCK_FROM, "endDate": "2021-12-31", "token": key, "format": "json"}, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Tiingo 연결 실패({type(exc).__name__})") from None
    time.sleep(0.5)
    if resp.status_code != 200 or not isinstance(resp.json(), list) or not resp.json():
        print(f"  Tiingo {sym}: HTTP {resp.status_code} 자료 없음")
        return 0
    d = pd.DataFrame(resp.json())
    d["date"] = d["date"].str[:10]
    d = d[d["date"] <= "2021-12-31"].sort_values("date")
    cum = d["splitFactor"].astype(float).replace(0, 1.0).cumprod()
    k = (cum / cum.iloc[-1]).to_numpy()
    out = pd.DataFrame({"date": d["date"], "open": d["open"] * k, "high": d["high"] * k, "low": d["low"] * k,
                        "close": d["close"] * k, "volume": d["volume"] / k, "div": d["divCash"].astype(float) * k})
    out.to_csv(p, index=False)
    return len(out)


def step_fetch() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    print(f"[fetch] TIINGO_API_KEY 있음: {bool(key)}")
    PX.mkdir(parents=True, exist_ok=True)
    tab = build_table()
    tab.to_csv(TABLE, index=False, encoding="utf-8-sig")
    for sym, start in ETF_FROM.items():
        print(f"[fetch] {sym}: {fetch_yf(sym, start)}")
    fail = []
    for r in tab.itertuples():
        if r.kind == "없음":
            continue
        n = fetch_yf(r.symbol, STOCK_FROM) if r.kind == "yfinance" else fetch_tiingo(r.symbol, key)
        if n == 0:
            fail.append(f"{r.ticker}({r.kind} {r.symbol})")
    print(f"[fetch] 명단 티커 {len(tab)} · 출처 없음 {int((tab['kind'] == '없음').sum())} · 받기 실패 {len(fail)}: {fail}")


def companies(tab: pd.DataFrame | None = None) -> pd.DataFrame:
    """회사(CIK)당 대표 주식: 2011-01 ~ 2021-12 가격 줄 수가 가장 긴 주식(같으면 티커 순). 출력: ticker, cik, symbol, rep(bool)."""
    tab = pd.read_csv(TABLE, encoding="utf-8-sig") if tab is None else tab
    tab = tab[tab["kind"] != "없음"].copy()
    n = []
    for r in tab.itertuples():
        p = PX / f"{r.symbol}.csv"
        df = read_guarded(p) if p.exists() else pd.DataFrame()
        if r.ticker in LISTING and len(df):
            df = df[df.index >= LISTING[r.ticker]]
        n.append(int(df["close"].notna().sum()) if len(df) else 0)
    tab["n_days"] = n
    tab = tab[tab["n_days"] > 0]
    tab = tab.sort_values(["cik", "n_days", "ticker"], ascending=[True, False, True])
    tab["rep"] = ~tab.duplicated("cik")
    return tab


def step_coverage() -> None:
    """허용 조회만: 명단 확보율·가격 공백·OHLC 결측·옛 종가 캐시(탐색 B)와 종가 대조."""
    tab = pd.read_csv(TABLE, encoding="utf-8-sig")
    comp = companies(tab)
    rep = comp[comp["rep"]]
    mem = members()
    sym_of = dict(zip(comp["ticker"], comp["cik"]))
    rep_cik = set(rep["cik"])
    tot = got = 0
    for me, ts in mem.items():
        ciks = {sym_of.get(t) for t in ts}
        tot += len(ts)
        got += sum(1 for t in ts if sym_of.get(t) in rep_cik)
    print(f"[coverage] 명단 티커 {len(tab)} · 출처 없음 {int((tab['kind'] == '없음').sum())} · 회사(대표 주식) {len(rep)}")
    print(f"[coverage] 명단 티커·달 {tot} 중 가격 있는 회사 소속 {got} ({got / tot * 100:.1f}%)")
    print("[coverage] 출처 없음:", ", ".join(f"{r.ticker}({r.months}달)" for r in tab[tab['kind'] == '없음'].itertuples()))
    print("[coverage] 같은 회사 여러 주식(대표 아님):", ", ".join(comp[~comp['rep']]["ticker"]))
    bad, dev = [], []
    old = ROOT / "data" / "cache" / "sector_pullback" / "yf"
    for r in rep.itertuples():
        df = read_guarded(PX / f"{r.symbol}.csv")
        miss = df[["open", "high", "low", "close", "volume"]].isna().any(axis=1) | (df["open"] <= 0) | (df["high"] < df["low"])
        if miss.any():
            bad.append(f"{r.ticker}:{int(miss.sum())}")
        o = old / f"{r.symbol}.csv"
        if r.kind == "yfinance" and o.exists():
            oc = pd.read_csv(o, index_col=0, parse_dates=True)["close"]
            oc = oc[oc.index <= LAST]
            j = df["close"].reindex(oc.index).dropna()
            if len(j):
                dev.append((r.ticker, float((j / oc.reindex(j.index) - 1).abs().max() * 100)))
    print(f"[coverage] OHLCV 결측·이상 줄이 있는 종목 {len(bad)}: {bad[:30]}")
    dv = sorted(dev, key=lambda x: -x[1])
    print(f"[coverage] 탐색 B 종가 캐시와 최대 차이(%) 상위: {[(t, round(x, 4)) for t, x in dv[:8]]}")
    for s in ETF_FROM:
        df = read_guarded(PX / f"{s}.csv")
        print(f"[coverage] {s} {df.index.min().date()} ~ {df.index.max().date()} ({len(df)}일)")


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="3단 거르기 데이터")
    ap.add_argument("--step", choices=["fetch", "coverage"], required=True)
    a = ap.parse_args(argv)
    {"fetch": step_fetch, "coverage": step_coverage}[a.step]()


if __name__ == "__main__":
    main()
