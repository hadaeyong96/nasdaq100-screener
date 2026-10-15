"""이번 주 QQQ보다 강한 종목·업종 — 브리핑 관찰 목록 (작업 5-1, docs/design/weekly_rs.md).

순수 함수만 둔다(네트워크·파일·DB·현재 시각 접근 없음). 미래 데이터 금지: 기준일 이하 행만 쓰고
음수 shift를 쓰지 않는다. 관찰 표시 전용이며 매수·매도 판정·기존 계산에는 쓰지 않는다.

정의(사전 등록, 2026-10-15 사용자 지시):
- 5거래일 수익률 = 기준일 종가 ÷ QQQ 거래일 기준 5거래일 전 종가 − 1 (종목·QQQ 같은 두 날짜)
- 초과(%p) = 종목 5거래일 수익률 − QQQ 5거래일 수익률
- 표 1: 초과 상위 10 / 표 2: 업종 묶음별 평균 초과 상위 3·하위 3 (명단에 있고 계산된 종목 2개 미만 묶음은 뺀다)
"""

from __future__ import annotations

import pandas as pd

DEFAULTS = {"days": 5, "top_n": 10, "group_top": 3, "group_bottom": 3, "group_min": 2, "groups": {}}
NO_GROUP = "-"


def settings(cfg: dict | None) -> dict:
    """config.yaml의 weekly_rs 구역(없으면 기본값). 입력: 전체 cfg / 출력: dict"""
    return {**DEFAULTS, **((cfg or {}).get("weekly_rs") or {})}


def group_of(groups: dict[str, list[str]]) -> dict[str, str]:
    """{묶음: [티커]} -> {티커: 묶음}. 한 티커가 두 묶음에 있으면 먼저 적힌 묶음."""
    out: dict[str, str] = {}
    for name, tickers in groups.items():
        for t in tickers:
            out.setdefault(str(t), str(name))
    return out


def compute_weekly_rs(prices: dict[str, pd.DataFrame], qqq_close: pd.Series, as_of, cfg: dict | None = None) -> dict:
    """명단 종목의 이번 주(5거래일) QQQ 대비 초과 수익과 업종 묶음 평균.

    입력: prices {ticker: DataFrame(날짜 인덱스, close)} — 오늘 명단 종목만, qqq_close(QQQ 확정 종가 Series),
         as_of(기준일), cfg
    출력: dict(
        status "ok" | "no_qqq"(QQQ에 기준일·5거래일 전 종가가 없음),
        window (시작일, 기준일) 또는 None, qqq_return(소수),
        rows: list[dict] 계산된 전 종목 (ticker, group, ret, excess_pp) — 초과 큰 순,
        top: rows 상위 top_n,
        groups_top / groups_bottom: list[dict] (group, count, avg_excess_pp, tickers),
        excluded: list[str] 두 날짜 종가가 없는 종목)
    """
    s = settings(cfg)
    days = int(s["days"])
    as_of = pd.Timestamp(as_of).normalize()
    q = qqq_close.copy()
    q.index = pd.DatetimeIndex(q.index).normalize()
    q = q[q.index <= as_of].dropna()
    out = {"status": "no_qqq", "window": None, "qqq_return": None, "rows": [], "top": [],
           "groups_top": [], "groups_bottom": [], "excluded": []}
    if len(q) <= days or q.index[-1] != as_of:
        return out
    start = q.index[-1 - days]
    q_ret = float(q.iloc[-1] / q.iloc[-1 - days] - 1.0)
    out.update(status="ok", window=(start, as_of), qqq_return=q_ret)

    gmap = group_of(s["groups"])
    rows, excluded = [], []
    for t in sorted(prices):
        df = prices[t]
        close = df["close"] if isinstance(df, pd.DataFrame) else df
        close = close[close.index <= as_of]
        c0 = close.get(start)
        c1 = close.get(as_of)
        if c0 is None or c1 is None or pd.isna(c0) or pd.isna(c1) or float(c0) <= 0:
            excluded.append(t)
            continue
        ret = float(c1) / float(c0) - 1.0
        rows.append({"ticker": t, "group": gmap.get(t, NO_GROUP), "ret": ret, "excess_pp": (ret - q_ret) * 100.0})
    rows.sort(key=lambda r: (-r["excess_pp"], r["ticker"]))
    out["rows"], out["excluded"] = rows, excluded
    out["top"] = rows[: int(s["top_n"])]

    by_group: dict[str, list[dict]] = {}
    for r in rows:
        if r["group"] != NO_GROUP:
            by_group.setdefault(r["group"], []).append(r)
    stats = [{"group": g, "count": len(rs), "avg_excess_pp": sum(r["excess_pp"] for r in rs) / len(rs),
              "tickers": [r["ticker"] for r in rs]}
             for g, rs in by_group.items() if len(rs) >= int(s["group_min"])]
    stats.sort(key=lambda x: (-x["avg_excess_pp"], x["group"]))
    n_top = int(s["group_top"])
    out["groups_top"] = stats[:n_top]
    rest = stats[n_top:]  # 묶음이 적으면 상위에 나온 묶음은 하위에 다시 쓰지 않는다
    out["groups_bottom"] = list(reversed(rest[-int(s["group_bottom"]):])) if rest else []
    return out
