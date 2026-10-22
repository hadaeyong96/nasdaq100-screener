"""여러 자산 장부 — 순수 함수 (변동성 맞춤·갈아타기·2배 균형 시험 공용, 2026-10-22).

core/shock_dip_perf.run_book(자산 2개)과 같은 비용·세금 규칙을 자산 K개로 넓힌 것:
- 판단일(pos)의 목표 비중 벡터와 그날 실제 비중을 should(현재, 목표)로 비교해 참이면 다음 거래일(pos+1) 종가에 목표로 맞춘다.
  목표보다 많은 자산을 먼저 팔고 그 돈으로 모자란 자산을 산다. 비용은 거래액의 편도 cost. 사이에는 수량 그대로.
- 세금(tax=True): 자산별 원화 평균 원가, 과세연도 원화 실현손익 통산 − 공제 → 세율, 다음 해 5월 첫 거래일 납부
  (sell_order 순서로 판다), 끝날 전부 판 것으로 보고 세후 원화 값.
네트워크·파일·현재 시각에 접근하지 않는다. 음수 shift를 쓰지 않는다(판단은 pos까지의 값만).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MultiResult:
    nav: np.ndarray  # 일별 가치(원래 통화, 세금 납부 반영)
    weights: np.ndarray  # 일별 종가 비중 (n, K)
    end_value: float
    end_after_tax_krw: float  # tax=False면 nan
    trades: int  # 다시 맞춘 횟수
    costs: float
    taxes_krw: float


def run_multi(prices: np.ndarray, start_w, decisions: dict[int, np.ndarray], should, *, cost: float = 0.001,
              tax: bool = False, fx: np.ndarray | None = None, start_krw: float = 100_000_000.0,
              deduction_krw: float = 2_500_000.0, rate: float = 0.22, may_pay: dict[int, int] | None = None,
              years: np.ndarray | None = None, sell_order: list[int] | None = None) -> MultiResult:
    """장부 하나를 위치 0(start_w 비중)부터 끝까지.

    입력: prices(n×K 가격, 위치 0이 시작일), start_w(시작 비중 K), decisions({판단 위치: 목표 비중 K}),
    should(현재 비중, 목표 비중) → bool(참이면 다음 날 맞춤), tax·fx·start_krw·may_pay·years(세금, run_book과 같음),
    sell_order(세금 낼 때 파는 자산 순서, 기본 0..K−1).
    """
    n, k = prices.shape
    f = fx if fx is not None else np.ones(n)
    v0 = start_krw / f[0]
    w0 = np.asarray(start_w, dtype=float)
    units = v0 * w0 / prices[0]
    basis = start_krw * w0  # 원화 원가(시작 비용 없음 — 1배로 시작하는 이전 시험과 같음)
    realized: dict[int, float] = {}
    yrs = years if years is not None else np.zeros(n, dtype=int)
    may_pay = may_pay or {}
    order = sell_order if sell_order is not None else list(range(k))
    costs = taxes = 0.0
    trades = 0
    nav = np.empty(n)
    wts = np.empty((n, k))
    pending: np.ndarray | None = None

    def sell(i: int, value: float, p: int) -> float:
        nonlocal costs
        q = min(value / prices[p, i], units[i])
        if q <= 0:
            return 0.0
        b = basis[i] * q / units[i]
        gross = q * prices[p, i]
        c = gross * cost
        costs += c
        realized[yrs[p]] = realized.get(yrs[p], 0.0) + (gross - c) * f[p] - b
        units[i] -= q
        basis[i] -= b
        return gross - c

    def buy(i: int, cash: float, p: int) -> None:
        nonlocal costs
        spend = cash / (1.0 + cost)
        costs += cash - spend
        units[i] += spend / prices[p, i]
        basis[i] += cash * f[p]

    for p in range(n):
        if tax and p in may_pay:
            owed = max(realized.get(may_pay[p], 0.0) - deduction_krw, 0.0) * rate
            if owed > 0:
                need, got = owed / f[p], 0.0
                for i in order:
                    if got >= need - 1e-12:
                        break
                    got += sell(i, (need - got) / (1.0 - cost), p)
                taxes += owed
        if pending is not None:
            vals = units * prices[p]
            tot = vals.sum()
            cash = 0.0
            for i in range(k):
                if vals[i] > pending[i] * tot + 1e-12:
                    cash += sell(i, vals[i] - pending[i] * tot, p)
            short = np.maximum(pending * tot - units * prices[p], 0.0)
            if short.sum() > 0:
                for i in range(k):
                    if short[i] > 0:
                        buy(i, cash * short[i] / short.sum(), p)
            trades += 1
            pending = None
        vals = units * prices[p]
        nav[p] = vals.sum()
        wts[p] = vals / nav[p] if nav[p] > 0 else 0.0
        if p in decisions and p + 1 < n:
            tgt = np.asarray(decisions[p], dtype=float)
            if should(wts[p], tgt):
                pending = tgt
    end_value = float(nav[-1])
    after = np.nan
    if tax:
        last = n - 1
        cash = sum(sell(i, units[i] * prices[last, i], last) for i in range(k) if units[i] > 0)
        paid = set(may_pay.values())
        owed = sum(max(g - deduction_krw, 0.0) * rate for y, g in realized.items()
                   if y == yrs[last] or (y not in paid and y >= yrs[0]))
        taxes += owed
        after = cash * f[last] - owed
    return MultiResult(nav, wts, end_value, after, trades, costs, taxes)


def cagr(start_value: float, end_value: float, days: float) -> float:
    """연복리(소수)."""
    return (end_value / start_value) ** (365.25 / days) - 1.0 if days > 0 and start_value > 0 else float("nan")


def max_drawdown(nav: np.ndarray) -> float:
    """최대 낙폭(소수, 음수)."""
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())
