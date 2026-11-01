"""마이크론 사이클 2단계 — 신호·진입 계획·무작위 대조·두 자산 장부 (docs/prereg/mu_cycle.md, 태그 mu-cycle-prereg).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.
장부: 자산 A(QQQ) + 자산 B(MU 또는 동료). 세금은 core/tax.py(공용 세금 규칙): 연간 원화 실현손익 − 250만 원 → 22%,
다음 해 5월 첫 거래일 납부. 배당은 15% 원천징수 뒤 같은 자산에 재투자.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import mu_cycle as mc
from core import tax

TRANCHES = (0.025, 0.05, 0.075, 0.10)  # 4개월 분할 목표 비중 — 상한 10%
MAX_WEIGHT = TRANCHES[-1]


# ── 신호 ────────────────────────────────────────────────────────────────────


def change_6m(values: pd.Series) -> tuple[pd.Timestamp, float] | None:
    """한 발표본(또는 그때 알려진 값들)의 마지막 관측월 L과 6개월 변화(%) = v[L] / v[L−6개월] − 1. 못 구하면 None."""
    v = values.dropna()
    if v.empty:
        return None
    last = v.index[-1]
    prev = last - pd.DateOffset(months=6)
    if prev not in v.index or v[prev] == 0:
        return None
    return last, float((v[last] / v[prev] - 1) * 100)


def realtime_change(month_ends: pd.DatetimeIndex, asof: dict[pd.Timestamp, pd.Series], current: pd.Series) -> pd.DataFrame:
    """월말 M마다 그때 알 수 있던 6개월 변화.

    M 당시 발표본(asof[M], 비어 있지 않음)이 있으면 그 발표본 안에서 계산(발표본 사용), 없으면 현재 값 중
    관측월 + 2개월째 1일 ≤ M인 값으로 계산(1단계 공개 시차). 출력: DataFrame[value, vintage(bool), obs_month], 인덱스 = month_ends.
    """
    rows = []
    for m in month_ends:
        s = asof.get(m)
        used = s is not None and len(s.dropna()) > 0
        if not used:
            cur = current.dropna()
            s = cur[cur.index + pd.DateOffset(months=2) <= m]
        r = change_6m(s)
        rows.append({"value": r[1] if r else np.nan, "vintage": bool(used), "obs_month": r[0] if r else pd.NaT})
    return pd.DataFrame(rows, index=month_ends)


def signal_months(x: pd.Series) -> list[pd.Timestamp]:
    """1단계와 같은 신호: 최근 12개월(그달 포함) 최저보다 +2%p 이상 높아진 첫 달(core.mu_cycle.signal_dates)."""
    return mc.signal_dates(x, rebound=2.0, lookback=12)


# ── 진입 계획 ────────────────────────────────────────────────────────────────


def month_add(m: pd.Timestamp, k: int) -> pd.Timestamp:
    """월말 m에서 k개월 뒤 월말."""
    return (m + pd.offsets.MonthEnd(k)) if k else m


def exit_month(s: pd.Timestamp, hold: int) -> pd.Timestamp:
    """신호 달 s → 복귀 달: 분할 S+1~S+4, 마지막 분할 뒤 hold개월 = S+4+hold 달 첫 거래일."""
    return month_add(s, 4 + hold)


def accept_signals(signals: list[pd.Timestamp], first: pd.Timestamp, last: pd.Timestamp, hold: int) -> list[pd.Timestamp]:
    """[first, last] 안 신호 중 보유(첫 분할 ~ 복귀 달) 중인 신호는 무시. 다음 신호는 앞 복귀 달 이후(같은 달 포함)부터."""
    out, free_from = [], first
    for s in sorted(signals):
        if s < free_from or s < first or s > last:
            continue
        out.append(s)
        free_from = exit_month(s, hold)
    return out


def random_signal_months(n: int, months: pd.DatetimeIndex, hold: int, rng: np.random.Generator) -> list[pd.Timestamp]:
    """months(가능한 신호 달, 연속 월말) 중 n개를 겹치지 않게(다음 ≥ 앞 복귀 달) 모든 배치가 같은 확률로 뽑는다.

    간격 = 4 + hold개월. y_i = s_i − (간격 − 1)·i 로 바꾸면 서로 다른 n개를 고르는 문제가 된다.
    """
    if n == 0:
        return []
    gap = 4 + hold
    span = len(months) - (gap - 1) * (n - 1)
    if span < n:
        raise ValueError("진입 횟수가 기간에 비해 너무 많습니다")
    y = np.sort(rng.choice(span, size=n, replace=False))
    return [months[int(v) + (gap - 1) * i] for i, v in enumerate(y)]


def first_trading_day(days: pd.DatetimeIndex, month_end: pd.Timestamp) -> int | None:
    """그달 첫 거래일 위치(없으면 None)."""
    i = int(np.searchsorted(days, pd.Timestamp(month_end.year, month_end.month, 1)))
    if i < len(days) and days[i].month == month_end.month and days[i].year == month_end.year:
        return i
    return None


def entry_events(entries: list[pd.Timestamp], days: pd.DatetimeIndex, hold: int) -> dict[int, tuple[float, bool]]:
    """진입 신호 달들 → 장부 이벤트 {위치: (B 목표 비중, 사기만 함)}. 분할 4번은 사기만, 복귀는 B 0%로 전량 매도. 데이터 밖은 뺀다."""
    ev: dict[int, tuple[float, bool]] = {}
    for s in entries:
        for k, w in enumerate(TRANCHES, 1):
            p = first_trading_day(days, month_add(s, k))
            if p is not None:
                ev[p] = (w, True)
        p = first_trading_day(days, exit_month(s, hold))
        if p is not None:
            ev[p] = (0.0, False)
    return ev


def annual_rebalance_events(days: pd.DatetimeIndex, w: float = MAX_WEIGHT) -> dict[int, tuple[float, bool]]:
    """항상 B w: 첫날 w로 맞추고, 해마다 첫 거래일에 정확히 w로 다시 맞춘다(사고팔기 모두)."""
    ev = {0: (w, False)}
    for i in range(1, len(days)):
        if days[i].year != days[i - 1].year:
            ev[i] = (w, False)
    return ev


# ── 장부 ────────────────────────────────────────────────────────────────────


@dataclass
class Book:
    nav: np.ndarray  # 일별 가치(달러, 세금 납부 반영)
    end_after_tax_krw: float  # 끝날 전부 판 뒤 세후 원화(tax=False면 nan)
    max_b_weight_after_buy: float  # 사기만 하는 이벤트 직후 B 비중 최댓값(상한 점검용)
    taxes_krw: float
    costs: float


def run_book(pa: np.ndarray, pb: np.ndarray, da: np.ndarray, db: np.ndarray, events: dict[int, tuple[float, bool]], *,
             cost: float = 0.001, fx: np.ndarray | None = None, tax_on: bool = True, start_krw: float = 100_000_000.0,
             dates: pd.DatetimeIndex | None = None) -> Book:
    """A 100%로 시작하는 두 자산 장부(위치 0 종가 시작, 시작 비용 없음).

    입력: pa·pb(종가, 분할 반영), da·db(그날 주당 배당, 없으면 0), events({위치: (B 목표 비중, 사기만)}),
    fx(원/달러), tax_on(세금 계산), dates(연도·5월 판단용). 출력: Book.
    배당: 보유 수량 × 주당 배당 × (1 − 15%)를 그날 종가에 같은 자산으로 재투자(비용 포함).
    """
    n = len(pa)
    f = fx if fx is not None else np.ones(n)
    years = dates.year.to_numpy() if dates is not None else np.zeros(n, dtype=int)
    may = {}
    if dates is not None:
        for p in range(1, n):
            if dates[p].month == 5 and dates[p - 1].month != 5:
                may[p] = int(dates[p].year) - 1
    ua, ub = start_krw / f[0] / pa[0], 0.0
    ba, bb = start_krw, 0.0
    realized: dict[int, float] = {}
    costs = taxes = 0.0
    nav = np.empty(n)
    max_w = 0.0

    def sell(which: str, value: float, p: int) -> float:
        nonlocal ua, ub, ba, bb, costs
        price, units = (pa[p], ua) if which == "a" else (pb[p], ub)
        q = min(value / price, units)
        if q <= 0:
            return 0.0
        basis = (ba if which == "a" else bb) * q / units
        gross = q * price
        c = gross * cost
        costs += c
        realized[years[p]] = realized.get(years[p], 0.0) + (gross - c) * f[p] - basis
        if which == "a":
            ua -= q
            ba -= basis
        else:
            ub -= q
            bb -= basis
        return gross - c

    def buy(which: str, cash: float, p: int) -> None:
        nonlocal ua, ub, ba, bb, costs
        if cash <= 0:
            return
        spend = cash / (1.0 + cost)
        costs += cash - spend
        if which == "a":
            ua += spend / pa[p]
            ba += cash * f[p]
        else:
            ub += spend / pb[p]
            bb += cash * f[p]

    for p in range(n):
        if tax_on and p in may:
            owed = tax.capital_gains_tax(realized.get(may[p], 0.0))
            if owed > 0:
                need = owed / f[p]
                got = sell("a", need / (1.0 - cost), p)
                if got < need:
                    got += sell("b", (need - got) / (1.0 - cost), p)
                taxes += owed
        if da[p] > 0 and ua > 0:
            buy("a", tax.dividend_after_withholding(ua * da[p]), p)
        if db[p] > 0 and ub > 0:
            buy("b", tax.dividend_after_withholding(ub * db[p]), p)
        if p in events:
            w, buy_only = events[p]
            va, vb = ua * pa[p], ub * pb[p]
            tot = va + vb
            if vb < w * tot - 1e-9:
                buy("b", sell("a", w * tot - vb, p), p)
            elif vb > w * tot + 1e-9 and not buy_only:
                buy("a", sell("b", vb - w * tot, p), p)
            if buy_only:
                max_w = max(max_w, ub * pb[p] / (ua * pa[p] + ub * pb[p]))
        nav[p] = ua * pa[p] + ub * pb[p]
    after = np.nan
    if tax_on:
        last = n - 1
        cash = sell("a", ua * pa[last], last) + (sell("b", ub * pb[last], last) if ub > 0 else 0.0)
        paid = set(may.values())
        owed = sum(tax.capital_gains_tax(g) for y, g in realized.items() if y == years[last] or y not in paid)
        taxes += owed
        after = cash * f[last] - owed
    return Book(nav, after, max_w, taxes, costs)


def cagr(start: float, end: float, days: float) -> float:
    """연복리(달력일 365.25)."""
    if days <= 0 or start <= 0 or end <= 0:
        return np.nan
    return (end / start) ** (365.25 / days) - 1.0


def max_drawdown(nav: np.ndarray) -> float:
    """최대 낙폭(음수 비율)."""
    return float((nav / np.maximum.accumulate(nav) - 1.0).min())
