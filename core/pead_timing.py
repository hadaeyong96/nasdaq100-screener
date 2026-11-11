"""실적급등 타이밍 탐색 — 순수 함수 (docs/research/pead_timing_plan.md, 태그 pead-timing-plan).

SW1(core/sw1.py)의 신호·비용·세금을 그대로 쓰고, 매수 시점(B1~B5)과 보유 기간만 바꾼다.
가격은 거래일 달력에 맞춘 배열(없는 날 NaN)로 받는다. 네트워크·파일 접근 없음.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import sw1

BUY_MODES = ("B1", "B2", "B3", "B4", "B5")
HOLDS = (20, 40, 60, 90, 120)
PULLBACK = 0.03  # B5: 반응일 종가보다 3% 이상 낮은 종가
PULLBACK_WINDOW = 10  # B5: 반응일 뒤 10거래일 안


@dataclass(frozen=True)
class Aligned:
    """달력에 맞춘 한 종목 가격. open·close: 길이 = 달력, 없는 날 NaN. lastv[j]: j 이하 마지막 종가 위치(없으면 −1)."""

    open: np.ndarray
    close: np.ndarray
    lastv: np.ndarray


def align(px: pd.DataFrame, days: pd.DatetimeIndex) -> Aligned:
    """open·close 일봉 → 달력 배열. 달력에 없는 날 행은 버린다."""
    d = px.reindex(days)
    o = d["open"].to_numpy(dtype=float)
    c = d["close"].to_numpy(dtype=float)
    pos = np.where(np.isfinite(c), np.arange(len(c)), -1)
    return Aligned(o, c, np.maximum.accumulate(pos))


def uses_open(mode: str) -> bool:
    """B1만 시가에 산다."""
    return mode == "B1"


def buy_point(a: Aligned, i0: int, mode: str) -> tuple[int | None, str]:
    """반응일 위치 i0 → (매수 위치, '') 또는 (None, 사유 '봉인 경계'|'가격 공백').

    B1 i0+1 시가 · B2 i0+1 종가 · B3 i0+3 종가 · B4 i0+5 종가 ·
    B5 i0+1~i0+10 중 종가 ≤ 반응일 종가 × 0.97인 첫날 종가, 없으면 i0+10 종가. 행 없는 날은 안 닿음으로 지나간다.
    """
    n = len(a.close)
    if mode == "B5":
        base = a.close[i0]
        if not (np.isfinite(base) and base > 0):
            return None, "가격 공백"
        ib = None
        for k in range(1, PULLBACK_WINDOW + 1):
            if i0 + k >= n:
                return None, "봉인 경계"
            c = a.close[i0 + k]
            if np.isfinite(c) and c <= base * (1 - PULLBACK):
                ib = i0 + k
                break
        ib = i0 + PULLBACK_WINDOW if ib is None else ib
    else:
        ib = i0 + {"B1": 1, "B2": 1, "B3": 3, "B4": 5}[mode]
        if ib >= n:
            return None, "봉인 경계"
    p = (a.open if uses_open(mode) else a.close)[ib]
    if not (np.isfinite(p) and p > 0):
        return None, "가격 공백"
    return ib, ""


def exit_point(a: Aligned, ib: int, j: int) -> int | None:
    """매수 위치 ib, 예정 매도 위치 j → 실제 매도 위치(j 이하 마지막 종가, ib 이상). 없으면 None (sw1.realized_exit와 같은 규칙)."""
    e = int(a.lastv[j])
    return e if e >= ib else None


def make_trades(signals: pd.DataFrame, prices: dict[str, Aligned], days: pd.DatetimeIndex, mode: str,
                hold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신호 표(열 ticker, d0 + 옮길 열) → (매매 표, 매매 안 된 신호 표).

    매매 열: 원래 열 + i0, ib, plan, ie, buy_day, sell_day, buy_px, sell_px.
    겹침: 앞 신호 반응일 ≤ 새 반응일 < 앞 매매 실제 매도일이면 무시('보유 중'). 매도 예정 위치가 달력 밖이면 '봉인 경계'.
    """
    s = signals.sort_values(["d0", "ticker"]).reset_index(drop=True)
    trades, skipped = [], []
    busy: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    n = len(days)
    for _, r in s.iterrows():
        t, d0 = r["ticker"], pd.Timestamp(r["d0"])
        rec = r.to_dict()
        h = busy.get(t)
        if h is not None and h[0] <= d0 < h[1]:
            skipped.append({**rec, "skip": "보유 중"})
            continue
        a = prices.get(t)
        i0 = int(days.searchsorted(d0))
        if a is None:
            skipped.append({**rec, "skip": "가격 공백"})
            continue
        ib, why = buy_point(a, i0, mode)
        if ib is None:
            skipped.append({**rec, "skip": why})
            continue
        j = ib + hold
        if j >= n:
            skipped.append({**rec, "skip": "봉인 경계"})
            continue
        ie = exit_point(a, ib, j)
        if ie is None:
            skipped.append({**rec, "skip": "가격 공백"})
            continue
        bp = float((a.open if uses_open(mode) else a.close)[ib])
        trades.append({**rec, "i0": i0, "ib": ib, "plan": j, "ie": ie, "buy_day": days[ib], "sell_day": days[ie],
                       "buy_px": bp, "sell_px": float(a.close[ie])})
        busy[t] = (d0, days[ie])
    cols = list(s.columns)
    tr = pd.DataFrame(trades, columns=cols + ["i0", "ib", "plan", "ie", "buy_day", "sell_day", "buy_px", "sell_px"])
    return tr, pd.DataFrame(skipped, columns=cols + ["skip"])


def price_trades(tr: pd.DataFrame, bench: Aligned, fx: np.ndarray, mode: str, costs: sw1.Costs,
                 rate: float) -> pd.DataFrame:
    """매매 표 → 수익 열(달러 비용 전, 원화 비용 뒤, 세금 뒤; 신호·QQQ 각각)과 초과 열 ex_pre·ex_cost·ex_after, year(반응일).

    QQQ는 같은 매수 위치·같은 가격 종류에 사고 같은 실제 매도 위치 종가에 판다. fx: 달력에 맞춘 원/달러.
    """
    out = tr.copy()
    ib, ie = out["ib"].to_numpy(int), out["ie"].to_numpy(int)
    bb = (bench.open if uses_open(mode) else bench.close)[ib]
    out["ret_usd"] = out["sell_px"] / out["buy_px"] - 1
    out["bench_usd"] = bench.close[ie] / bb - 1
    if not np.isfinite(out["bench_usd"]).all():
        raise ValueError("비교 지수 가격 결측")
    fb, fs = fx[ib], fx[ie]
    out["ret_krw"] = sw1.krw_after_cost(out["ret_usd"].to_numpy(), fb, fs, costs)
    out["bench_krw"] = sw1.krw_after_cost(out["bench_usd"].to_numpy(), fb, fs, costs)
    yr = pd.DatetimeIndex(out["sell_day"]).year.to_numpy()
    out["ret_at"] = out["ret_krw"] - sw1.ledger_tax(out["ret_krw"].to_numpy(), yr, rate)
    out["bench_at"] = out["bench_krw"] - sw1.ledger_tax(out["bench_krw"].to_numpy(), yr, rate)
    out["ex_pre"] = out["ret_usd"] - out["bench_usd"]
    out["ex_cost"] = out["ret_krw"] - out["bench_krw"]
    out["ex_after"] = out["ret_at"] - out["bench_at"]
    out["year"] = pd.DatetimeIndex(out["d0"]).year
    return out


def candidates(tr: pd.DataFrame, pools: list[np.ndarray], O: np.ndarray, C: np.ndarray, L: np.ndarray,
               bench: Aligned, fx: np.ndarray, years: np.ndarray, mode: str, costs: sw1.Costs) -> list[dict]:
    """매매마다 무작위 후보들의 원화 비용 뒤 수익·매도 연도·QQQ 원화 비용 뒤 수익.

    입력: pools[k](그 매매 후보 종목의 행 번호), O·C·L(종목 × 달력 시가·종가·lastv 행렬), years(달력 위치 → 연도).
    후보 = 매수 위치 매수가가 있고 실제 매도 위치(예정 위치 이하 마지막 종가, 매수 위치 이상)가 있는 종목.
    """
    P = O if uses_open(mode) else C
    bb_all = bench.open if uses_open(mode) else bench.close
    out = []
    for k, (ib, j) in enumerate(zip(tr["ib"].to_numpy(int), tr["plan"].to_numpy(int))):
        rows = pools[k]
        bp = P[rows, ib]
        e = L[rows, j]
        ok = np.isfinite(bp) & (bp > 0) & (e >= ib)
        rows, bp, e = rows[ok], bp[ok], e[ok]
        r = C[rows, e] / bp - 1
        bu = bench.close[e] / bb_all[ib] - 1
        fb, fs = fx[ib], fx[e]
        out.append({"ret": sw1.krw_after_cost(r, fb, fs, costs), "year": years[e],
                    "bench": sw1.krw_after_cost(bu, fb, fs, costs)})
    return out


def cumulative_excess(signals: pd.DataFrame, prices: dict[str, Aligned], bench_close: np.ndarray,
                      days: pd.DatetimeIndex, horizon: int = 120) -> tuple[np.ndarray, int]:
    """반응일 종가 기준 k = 1..horizon 거래일 뒤 QQQ 대비 누적 초과(비용 전)의 신호 평균.

    대상: i0 + horizon이 달력 안이고 반응일 종가가 있는 신호. 종목 행이 끊기면 마지막 종가를 그대로 쓴다.
    출력: (길이 horizon 배열, 신호 수)
    """
    rows = []
    for t, d0 in zip(signals["ticker"], signals["d0"]):
        a = prices.get(t)
        i0 = int(days.searchsorted(pd.Timestamp(d0)))
        if a is None or i0 + horizon >= len(days) or not np.isfinite(a.close[i0]):
            continue
        idx = a.lastv[i0 + 1:i0 + horizon + 1]
        s = a.close[idx] / a.close[i0] - 1
        q = bench_close[i0 + 1:i0 + horizon + 1] / bench_close[i0] - 1
        rows.append(s - q)
    if not rows:
        return np.full(horizon, np.nan), 0
    return np.vstack(rows).mean(axis=0), len(rows)


def select_cell(grid: pd.DataFrame) -> tuple[tuple[str, int] | None, list[dict]]:
    """고르는 규칙(계획 7절). 입력: 행 = (mode, hold) 25개, 열 mean·p1·p2·pct. 출력: (고른 칸 또는 None, 칸별 판단 목록).

    1) p1 > 0 그리고 p2 > 0 · 2) 상하좌우 이웃 칸(있는 만큼) mean > 0 · 3) pct 최고 → mean 큰 쪽 → 보유 짧은 쪽.
    """
    g = grid.set_index(["mode", "hold"])
    notes = []
    for m in BUY_MODES:
        for h in HOLDS:
            r = g.loc[(m, h)]
            mi, hi = BUY_MODES.index(m), HOLDS.index(h)
            nb = [(BUY_MODES[mi + dm], HOLDS[hi + dh]) for dm, dh in ((-1, 0), (1, 0), (0, -1), (0, 1))
                  if 0 <= mi + dm < len(BUY_MODES) and 0 <= hi + dh < len(HOLDS)]
            both = bool(r["p1"] > 0 and r["p2"] > 0)
            nbr = bool(all(g.loc[x, "mean"] > 0 for x in nb))
            notes.append({"mode": m, "hold": h, "both_periods": both, "neighbors": nbr, "ok": both and nbr,
                          "pct": float(r["pct"]), "mean": float(r["mean"])})
    ok = [x for x in notes if x["ok"]]
    if not ok:
        return None, notes
    best = sorted(ok, key=lambda x: (-x["pct"], -x["mean"], x["hold"]))[0]
    return (best["mode"], best["hold"]), notes
