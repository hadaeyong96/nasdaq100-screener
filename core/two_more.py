"""전략 2개 더(갈아타기·2배 균형) — 자산 만들기·신호 순수 함수 (docs/prereg/two_more.md, 태그 two-more-prereg).

네트워크·파일·현재 시각에 접근하지 않는다. 음수 shift 없음 — 판단일 값은 그날까지의 가격만 쓴다.
자산 순서(가격 행렬 열): 0 현금, 1 주식 1배, 2 장기채 1배, 3 금, 4 주식 2배, 5 장기채 2배.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CASH, STOCK, BOND, GOLD, STOCK2, BOND2 = range(6)
NAMES = {CASH: "현금", STOCK: "주식", BOND: "장기채", GOLD: "금"}


def par_bond_price(y_new: float, coupon: float, years: float = 15.0) -> float:
    """연 2회 이표 액면 채권(쿠폰 coupon, 만기 years년)을 금리 y_new(소수)로 평가한 가격(액면 1)."""
    n = int(round(years * 2))
    i = y_new / 2.0
    c = coupon / 2.0
    if abs(i) < 1e-12:
        return c * n + 1.0
    return c * (1 - (1 + i) ** -n) / i + (1 + i) ** -n


def bond_returns(yield_pct: pd.Series, years: float = 15.0) -> pd.Series:
    """금리(연 %, 날짜 인덱스)로 합성한 장기채 일간 총수익률: 전날 금리로 산 액면 채권을 오늘 금리로 평가 + 전날 금리 ÷ 252 이자.
    첫날은 0."""
    y = yield_pct.astype(float) / 100.0
    prev = y.shift(1)
    out = pd.Series([par_bond_price(a, b, years) - 1.0 + b / 252.0 if not (np.isnan(a) or np.isnan(b)) else 0.0
                     for a, b in zip(y.to_numpy(), prev.to_numpy())], index=y.index)
    out.iloc[0] = 0.0
    return out


def splice_real(synth_r: pd.Series, real_px: pd.Series | None) -> pd.Series:
    """합성 일간 수익률에 실제 상품 수익률을 잇는다(실제 첫 수익률이 나오는 날부터) → 가격(첫날 1.0)."""
    r = synth_r.copy()
    if real_px is not None and len(real_px.dropna()) > 1:
        rp = real_px.reindex(r.index)
        rr = rp.pct_change()
        ok = rr.notna() & rp.shift(1).notna()
        r[ok] = rr[ok]
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod()


def total_return_with_monthly_dividend(price: pd.Series, monthly_div: pd.Series) -> pd.Series:
    """일별 가격지수 + 실러 월별 배당(연율 D, 월 시작일 인덱스) → 총수익 가격(첫날 1.0).
    그달 거래일마다 (D ÷ 12 ÷ 그달 거래일 수)를 배당으로 더한다: r_t = (P_t + div_t) / P_{t−1} − 1."""
    p = price.astype(float)
    month = p.index.to_period("M")
    d = monthly_div.copy()
    d.index = d.index.to_period("M")
    per_month = pd.Series(month, index=p.index).map(d).astype(float) / 12.0
    ndays = pd.Series(month, index=p.index).map(pd.Series(month).value_counts()).astype(float)
    div = (per_month / ndays).fillna(0.0)
    r = (p + div) / p.shift(1) - 1.0
    r.iloc[0] = 0.0
    return (1.0 + r.fillna(0.0)).cumprod()


def month_end_positions(dates: pd.DatetimeIndex, w0: int) -> list[int]:
    """w0 이상 위치 중 그달 마지막 거래일."""
    m = dates.to_period("M")
    return [p for p in range(w0, len(dates)) if p == len(dates) - 1 or m[p + 1] != m[p]]


def quarter_end_positions(dates: pd.DatetimeIndex, w0: int) -> list[int]:
    """w0 이상 위치 중 분기(3·6·9·12월) 마지막 거래일."""
    return [p for p in month_end_positions(dates, w0) if dates[p].month in (3, 6, 9, 12)]


def lookback_returns(prices: np.ndarray, dates: pd.DatetimeIndex, p: int, months: int, first_valid: dict[int, pd.Timestamp],
                     assets=(STOCK, BOND, GOLD, CASH)) -> dict[int, float]:
    """판단 위치 p에서 자산별 months개월 수익률 = p 값 ÷ (months개월 전 같은 달 마지막 거래일 값) − 1.
    그 기준일이 없거나 자산 가격이 그 기준일 전에 시작하지 않았으면 그 자산은 뺀다."""
    target = dates[p].to_period("M") - months
    m = dates.to_period("M")
    prior = np.where(m == target)[0]
    if len(prior) == 0:
        return {}
    q = int(prior[-1])
    out = {}
    for a in assets:
        if first_valid.get(a) is not None and dates[q] < first_valid[a]:
            continue
        out[a] = float(prices[p, a] / prices[q, a] - 1.0)
    return out


def switch_choice(rets: dict[int, float]) -> int | None:
    """1등(주식·장기채·금 중 가장 큰 수익률)이 현금보다 낮으면 현금, 아니면 1등. 고를 수 없으면 None."""
    risky = {a: r for a, r in rets.items() if a != CASH}
    if not risky or CASH not in rets:
        return None
    best = max(sorted(risky), key=lambda a: risky[a])
    return CASH if risky[best] < rets[CASH] else best


def one_hot(a: int, k: int = 6) -> np.ndarray:
    w = np.zeros(k)
    w[a] = 1.0
    return w


def sharpe(nav: np.ndarray, cash_px: np.ndarray) -> float:
    """일별 가치 → (초과 수익률 평균 × 252) ÷ (수익률 표준편차 × √252). 초과 = 일간 수익률 − 현금 일간 수익률."""
    r = nav[1:] / nav[:-1] - 1.0
    c = cash_px[1:] / cash_px[:-1] - 1.0
    sd = r.std(ddof=1)
    return float((r - c).mean() * 252 / (sd * np.sqrt(252))) if sd > 0 else float("nan")
