"""F2 자금 흐름 관찰 — 지표·TOP 10·섹터 표·주간 기록 행 (docs/design/flow_watch.md, configs/f2_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·DB·현재 시각 접근 없음). 미래 데이터 금지: 각 종목 일봉은 기준일 이하 행만
쓰고, 음수 shift를 쓰지 않는다. 참고 표시·기록 전용이며 매수·매도 판정에는 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INFLOW, DUMP, NORMAL = "유입", "투매", "보통"
UNIVERSE_LIST = "대상"
DEFAULTS = {
    "share_ratio_min": 1.5, "up_volume_ratio_min": 0.5, "recent_days": 5, "base_days": 60, "top_n": 10,
    "exclude_tickers": ["QQQ", "QQQM", "TQQQ", "QLD", "SQQQ", "PSQ"],
}
UNCLASSIFIED = "미분류"


def settings(cfg: dict | None) -> dict:
    """config.yaml의 flow_watch 구역(없으면 기본값)을 읽는다. 입력: 전체 cfg / 출력: dict"""
    return {**DEFAULTS, **((cfg or {}).get("flow_watch") or {})}


def classify(share_ratio: float, weekly_return: float, up_ratio: float, s: dict) -> str:
    """상태 판정(구현 해석 10번). 결측이 있으면 보통.

    유입: 배율 ≥ 기준 & 주간 > 0 & 상승일 비중 > 기준 / 투매: 배율 ≥ 기준 & 주간 < 0
    """
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (share_ratio, weekly_return)):
        return NORMAL
    if share_ratio >= s["share_ratio_min"]:
        if weekly_return > 0 and up_ratio is not None and not np.isnan(up_ratio) and up_ratio > s["up_volume_ratio_min"]:
            return INFLOW
        if weekly_return < 0:
            return DUMP
    return NORMAL


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den and den > 0 else float("nan")


def compute_flow(prices: dict[str, pd.DataFrame], sector_map: dict[str, str], as_of, cfg: dict | None = None) -> dict:
    """그날 대상 종목의 자금 흐름 지표·TOP 10·섹터 표를 만든다.

    입력: prices {ticker: DataFrame(날짜 인덱스, close, volume)}, sector_map {ticker: 섹터}, as_of(기준일), cfg
    출력: dict(
        metrics: DataFrame(인덱스 ticker; share_ratio, volume_ratio, weekly_return, up_volume_ratio,
                 dollar_volume_5d, high_52w, close, sector, status) — 제외 안 된 전 종목,
        top: list[dict] (TOP n, rank 포함), sectors: list[dict], excluded: list[str],
        window: (첫 날짜, 기준일) 또는 None, as_of: Timestamp)
    """
    s = settings(cfg)
    rd, bd = int(s["recent_days"]), int(s["base_days"])
    n_win = rd + bd
    as_of = pd.Timestamp(as_of).normalize()
    skip = set(s["exclude_tickers"])
    cut = {t: df.loc[df.index <= as_of, ["close", "volume"]] for t, df in prices.items() if t not in skip}
    all_dates = sorted(set().union(*[set(df.index) for df in cut.values()])) if cut else []
    empty = {"metrics": pd.DataFrame(), "top": [], "sectors": [], "excluded": sorted(cut), "window": None, "as_of": as_of}
    if len(all_dates) < n_win or all_dates[-1] != as_of:
        return empty
    win = pd.DatetimeIndex(all_dates[-n_win:])

    close = pd.DataFrame({t: df["close"].reindex(win) for t, df in cut.items()})
    vol = pd.DataFrame({t: df["volume"].reindex(win) for t, df in cut.items()})
    bad = close.isna().any() | vol.isna().any() | (close <= 0).any()
    excluded = sorted(close.columns[bad])
    keep = [t for t in close.columns if not bad[t]]
    if not keep:
        return {**empty, "excluded": excluded}
    close, vol = close[keep].astype(float), vol[keep].astype(float)

    dv = close * vol
    share = dv.div(dv.sum(axis=1), axis=0)
    recent, base = slice(n_win - rd, n_win), slice(0, n_win - rd)
    share_ratio = share.iloc[recent].mean() / share.iloc[base].mean().replace(0, np.nan)
    vol_ratio = vol.iloc[recent].mean() / vol.iloc[base].mean().replace(0, np.nan)
    weekly = close.iloc[-1] / close.iloc[-1 - rd] - 1.0
    up = close.diff().iloc[recent] > 0  # 양수 방향 차분(전일 대비)만
    v5 = vol.iloc[recent]
    up_ratio = (v5.where(up, 0.0).sum()) / v5.sum().replace(0, np.nan)
    high = {t: float(cut[t]["close"].iloc[-252:].max()) for t in keep}

    rows = []
    for t in keep:
        sr, wr, ur = float(share_ratio[t]), float(weekly[t]), float(up_ratio[t])
        rows.append({
            "ticker": t, "share_ratio": sr, "volume_ratio": float(vol_ratio[t]), "weekly_return": wr,
            "up_volume_ratio": ur, "dollar_volume_5d": float(dv[t].iloc[recent].sum()),
            "high_52w": float(close[t].iloc[-1] / high[t] - 1.0), "close": float(close[t].iloc[-1]),
            "sector": sector_map.get(t) or UNCLASSIFIED, "status": classify(sr, wr, ur, s),
        })
    metrics = pd.DataFrame(rows).set_index("ticker")

    ranked = metrics.dropna(subset=["share_ratio"]).reset_index().sort_values(["share_ratio", "ticker"], ascending=[False, True])
    top = [{**r, "rank": i + 1} for i, r in enumerate(ranked.head(int(s["top_n"])).to_dict("records"))]

    sec = dv.T.groupby(metrics["sector"]).sum().T  # 날짜 × 섹터 거래대금
    tot_recent, tot_base = dv.iloc[recent].values.sum(), dv.iloc[base].values.sum()
    sectors = []
    for name in sec.columns:
        r_share = float(sec[name].iloc[recent].sum() / tot_recent * 100) if tot_recent else float("nan")
        b_share = float(sec[name].iloc[base].sum() / tot_base * 100) if tot_base else float("nan")
        sectors.append({"sector": name, "recent_share_pct": r_share, "base_share_pct": b_share,
                        "change_pp": r_share - b_share, "count": int((metrics["sector"] == name).sum())})
    sectors.sort(key=lambda r: (-r["change_pp"], -r["recent_share_pct"]))
    return {"metrics": metrics, "top": top, "sectors": sectors, "excluded": excluded,
            "window": (win[0], win[-1]), "as_of": as_of}


# ── 주간 기록 ────────────────────────────────────────────────────────────────


def is_last_trading_day_of_week(as_of, next_trading_day) -> bool:
    """기준일 다음 거래일의 ISO 주(연, 주)가 기준일과 다르면 True (구현 해석 14번).

    입력: as_of, next_trading_day(NYSE 달력에서 구한 다음 거래일) / 출력: bool
    """
    a, n = pd.Timestamp(as_of).isocalendar(), pd.Timestamp(next_trading_day).isocalendar()
    return (a[0], a[1]) != (n[0], n[1])


def weekly_record_rows(flow: dict, qqq_close: float, cfg: dict | None = None) -> list[dict]:
    """compute_flow 결과 → flow_watch 표에 넣을 행(유입·투매 상위 n, 대상 전부). 구현 해석 15·16번.

    입력: flow, qqq_close(기준일 QQQ 종가), cfg / 출력: list[dict(base_date, list, rank, ticker, share_ratio, close, qqq_close)]
    """
    s = settings(cfg)
    m = flow["metrics"]
    if m.empty:
        return []
    base_date = flow["as_of"].date().isoformat()
    out = []
    ranked = m.dropna(subset=["share_ratio"]).reset_index().sort_values(["share_ratio", "ticker"], ascending=[False, True])
    for status in (INFLOW, DUMP):
        sel = ranked[ranked["status"] == status].head(int(s["top_n"]))
        for i, r in enumerate(sel.itertuples()):
            out.append({"base_date": base_date, "list": status, "rank": i + 1, "ticker": r.ticker,
                        "share_ratio": float(r.share_ratio), "close": float(r.close), "qqq_close": float(qqq_close)})
    for t, r in m.sort_index().iterrows():
        out.append({"base_date": base_date, "list": UNIVERSE_LIST, "rank": None, "ticker": t,
                    "share_ratio": None if np.isnan(r["share_ratio"]) else float(r["share_ratio"]),
                    "close": float(r["close"]), "qqq_close": float(qqq_close)})
    return out


def forward_return(series: pd.Series, base_date, horizon: int, as_of) -> float | None:
    """종가 시리즈에서 base_date 행 뒤 horizon번째 행(기준일 이하)이 있으면 수익, 없으면 None.

    두 종가 모두 같은 시리즈에서 읽는다(구현 해석 19번). 입력: series(날짜 인덱스 종가) / 출력: float|None
    """
    s = series.loc[series.index <= pd.Timestamp(as_of)].dropna()
    base = pd.Timestamp(base_date)
    if base not in s.index:
        return None
    pos = s.index.get_loc(base) + horizon
    if pos >= len(s):
        return None
    b = float(s.iloc[pos - horizon])
    return float(s.iloc[pos]) / b - 1.0 if b > 0 else None


def fill_results(pending: list[dict], closes: dict[str, pd.Series], qqq: pd.Series, as_of) -> list[dict]:
    """결과가 빈 기록 행에 5·20일 결과를 채울 값을 만든다 (구현 해석 19번).

    입력: pending [{id, base_date, ticker, need_5d, need_20d}], closes {ticker: 종가 Series}, qqq(QQQ 종가), as_of
    출력: [{id, horizon(5|20), ret, qqq_ret, excess}] — 채울 수 있는 것만
    """
    out = []
    for row in pending:
        for h in (5, 20):
            if not row.get(f"need_{h}d"):
                continue
            series = closes.get(row["ticker"])
            if series is None:
                continue
            r = forward_return(series, row["base_date"], h, as_of)
            q = forward_return(qqq, row["base_date"], h, as_of)
            if r is None or q is None:
                continue
            out.append({"id": row["id"], "horizon": h, "ret": r, "qqq_ret": q, "excess": r - q})
    return out
