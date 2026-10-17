"""폭락 연구 관문 3 — 전략 성적 계산 (docs/prereg/shock_dip.md A~K + v2 + v3).

순수 함수만 둔다(네트워크·파일·현재 시각 접근 없음). 음수 shift를 쓰지 않는다.

장부 모형:
- 자산 두 개: X(1배: 지수 또는 QQQ), L(2배: QLD 실제 수정주가 또는 합성 2배). 가격 배열로 받는다.
- 이벤트(체결일 종가)에서 목표 비중(wx, wl)으로 맞춘다. 남는 쪽을 팔고 그 돈으로 모자란 쪽을 산다. 비용은 편도 cost(거래액 기준).
  사이에는 수량이 그대로라 비중이 움직인다(매일 다시 맞추지 않음).
- 세금(tax=True): 과세연도 원화 실현손익 통산 − 공제 → 세율, 다음 해 5월 첫 거래일에 X부터 팔아 낸다(그 매도 손익은 그해).
  원가는 자산별 평균 원가(원화). 끝날 세후 가치 = 그날 전부 팔았다고 보고(매도 비용 포함) 그해 세금·밀린 세금을 뺀 원화 값.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def leveraged_returns(r1: pd.Series, rate_pct: pd.Series, fee_annual: float = 0.0095) -> pd.Series:
    """합성 2배 일간 수익률 = 2 × 1배 일간 수익률 − 연 보수/252 − 차입금리(전날 값, 연 %)/252 (사전 등록 E).

    입력: r1(1배 일간 수익률), rate_pct(DTB3 등 연 %, r1과 같은 인덱스로 as-of 정렬된 값). 출력: Series.
    """
    rate_prev = rate_pct.shift(1).fillna(rate_pct)
    return 2.0 * r1 - fee_annual / TRADING_DAYS - rate_prev / 100.0 / TRADING_DAYS


def splice_leveraged(r1: pd.Series, real_l: pd.Series | None, rate_pct: pd.Series, fee_annual: float = 0.0095) -> tuple[pd.Series, pd.Timestamp | None]:
    """2배 자산 가격(첫날 1.0): 실제 2배 상품이 있는 날(첫 수익률이 나오는 날부터)은 실제 수익률, 그 전은 합성.

    출력: (가격 Series, 실제 수익률을 처음 쓴 날 또는 None)
    """
    syn = leveraged_returns(r1, rate_pct, fee_annual)
    r = syn.copy()
    first_real = None
    if real_l is not None and len(real_l.dropna()) > 1:
        rr = real_l.reindex(r1.index).pct_change()
        ok = rr.notna() & real_l.reindex(r1.index).shift(1).notna()
        if ok.any():
            first_real = ok.idxmax()
            r[ok] = rr[ok]
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod(), first_real


def may_payment_positions(dates: pd.DatetimeIndex) -> dict[int, int]:
    """해마다 5월 첫 거래일 위치 → 직전 과세연도. 출력: {pos: year}."""
    out = {}
    s = pd.Series(range(len(dates)), index=dates)
    for y in sorted(set(dates.year)):
        may = s[(s.index.year == y) & (s.index.month == 5)]
        if len(may):
            out[int(may.iloc[0])] = y - 1
    return out


@dataclass
class BookResult:
    nav: np.ndarray | None  # 일별 가치(원래 통화, 세금 납부 반영) — daily=False면 None
    end_value: float  # 끝날 가치(원래 통화, 청산 전)
    end_after_tax_krw: float  # 끝날 세후 청산 가치(원화) — tax=False면 nan
    costs: float  # 거래 비용 합(원래 통화)
    taxes_krw: float  # 낸 세금 합(원화, 청산 세금 포함)
    trades: int = 0
    log: list = field(default_factory=list)


def run_book(px: np.ndarray, pl: np.ndarray, events: list[tuple[int, float]], *, cost: float = 0.001,
             tax: bool = False, fx: np.ndarray | None = None, start_krw: float = 100_000_000.0,
             deduction_krw: float = 2_500_000.0, rate: float = 0.22, may_pay: dict[int, int] | None = None,
             years: np.ndarray | None = None, daily: bool = True) -> BookResult:
    """한 장부를 처음(위치 0, X 100%)부터 끝까지 돌린다.

    입력: px·pl(같은 길이 가격 배열, 위치 0이 시작일), events([(pos, wx)] — pos 종가에 X 비중 wx, L 비중 1−wx로),
    tax(세금 계산 여부), fx(원/달러 등 원화 환산 배열, None이면 1), start_krw(시작 원화 — 원래 통화 시작값 = start_krw / fx[0]),
    may_pay(5월 첫 거래일 위치 → 과세연도), years(위치별 연도 배열), daily(일별 가치 배열을 만들지).
    출력: BookResult.
    """
    n = len(px)
    f = fx if fx is not None else np.ones(n)
    v0 = start_krw / f[0]
    ux, ul = v0 / px[0], 0.0
    bx, bl = start_krw, 0.0  # 원화 원가
    realized: dict[int, float] = {}
    taxes = 0.0
    costs = 0.0
    nav = np.empty(n) if daily else None
    ev = dict(events)
    may_pay = may_pay or {}
    yrs = years if years is not None else np.zeros(n, dtype=int)
    ntr = 0

    def sell(asset: str, value: float, p: int) -> float:
        nonlocal ux, ul, bx, bl, costs
        price = px[p] if asset == "x" else pl[p]
        units = ux if asset == "x" else ul
        q = min(value / price, units)
        if q <= 0:
            return 0.0
        basis = (bx if asset == "x" else bl) * q / units
        gross = q * price
        c = gross * cost
        costs += c
        realized[yrs[p]] = realized.get(yrs[p], 0.0) + (gross - c) * f[p] - basis
        if asset == "x":
            ux -= q
            bx -= basis
        else:
            ul -= q
            bl -= basis
        return gross - c

    def buy(asset: str, cash: float, p: int) -> None:
        nonlocal ux, ul, bx, bl, costs
        price = px[p] if asset == "x" else pl[p]
        spend = cash / (1.0 + cost)
        costs += cash - spend
        if asset == "x":
            ux += spend / price
            bx += cash * f[p]
        else:
            ul += spend / price
            bl += cash * f[p]

    for p in range(n):
        if tax and p in may_pay:
            y = may_pay[p]
            owed_krw = max(realized.get(y, 0.0) - deduction_krw, 0.0) * rate
            if owed_krw > 0:
                need = owed_krw / f[p]
                got = sell("x", need / (1.0 - cost), p)
                if got < need:
                    got += sell("l", (need - got) / (1.0 - cost), p)
                taxes += owed_krw
        if p in ev:
            wx = ev[p]
            vx, vl = ux * px[p], ul * pl[p]
            tot = vx + vl
            if vx > wx * tot + 1e-12:
                buy("l", sell("x", vx - wx * tot, p), p)
            elif vl > (1 - wx) * tot + 1e-12:
                buy("x", sell("l", vl - (1 - wx) * tot, p), p)
            ntr += 1
        if daily:
            nav[p] = ux * px[p] + ul * pl[p]
    end_value = ux * px[-1] + ul * pl[-1]
    after = np.nan
    if tax:
        last = n - 1
        cash = 0.0
        if ux > 0:
            cash += sell("x", ux * px[last], last)
        if ul > 0:
            cash += sell("l", ul * pl[last], last)
        y_end = yrs[last]
        paid_years = set(may_pay.values())
        owed = 0.0
        for y, g in realized.items():  # 끝까지 5월이 오지 않은 해(끝 해, 그리고 끝이 5월 전이면 전 해)
            if y == y_end or (y not in paid_years and y >= yrs[0]):
                owed += max(g - deduction_krw, 0.0) * rate
        taxes += owed
        after = cash * f[last] - owed
    return BookResult(nav, end_value, after, costs, taxes, ntr)


def strategy_events(trades: list[tuple[int, int | None]], n: int, w_in: float = 0.5) -> list[tuple[int, float]]:
    """거래 [(체결 위치, 청산 위치 또는 None)] → 이벤트 [(pos, X 비중)]. 진입 = X 50%·L 50%, 청산 = X 100%."""
    ev = []
    for e, x in trades:
        if e < n:
            ev.append((e, w_in))
        if x is not None and x < n:
            ev.append((x, 1.0))
    return ev


def switch_events(above: np.ndarray) -> list[tuple[int, float]]:
    """200일선 2배 전환(사전 등록 G): t일 종가 > 200일선이면 t+1 종가에 L 100%, 아니면 X 100%. 바뀔 때만 이벤트."""
    ev, state = [], 1.0
    for t in range(len(above) - 1):
        want = 0.0 if above[t] else 1.0
        if want != state:
            ev.append((t + 1, want))
            state = want
    return ev


def cagr(start_value: float, end_value: float, days: float) -> float:
    """연복리(달력일 기준 365.25일)."""
    if days <= 0 or start_value <= 0 or end_value <= 0:
        return np.nan
    return (end_value / start_value) ** (365.25 / days) - 1.0


def max_drawdown(nav: np.ndarray) -> float:
    """최대 낙폭(음수 비율)."""
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def random_placements(holds: list[int], first_pos: int, last_pos: int, rng: np.random.Generator) -> list[tuple[int, int]] | None:
    """보유 일수 목록을 그대로 쓰고 시작일만 겹치지 않게 무작위로 놓는다(v2-6).

    체결 위치 e_i, 청산 위치 e_i + h_i ≤ last_pos, 다음 체결 ≥ 앞 청산 + 1. 순서도 섞는다.
    출력: [(체결, 청산)] 또는 None(자리가 모자람).
    """
    h = list(rng.permutation(holds))
    k = len(h)
    slack = last_pos - first_pos - sum(h) - (k - 1)
    if k == 0:
        return []
    if slack < 0:
        return None
    gaps = np.sort(rng.integers(0, slack + 1, size=k))
    out, offset = [], first_pos
    for i in range(k):
        e = offset + int(gaps[i]) - (int(gaps[i - 1]) if i else 0)
        out.append((e, e + int(h[i])))
        offset = e + int(h[i]) + 1
    return out
