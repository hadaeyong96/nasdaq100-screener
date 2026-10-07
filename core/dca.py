"""D1 적립식 매수 방식 비교 — 신호·매수 규칙·10년 장부 (docs/d1_plan.md, configs/d1_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·DB·현재 시각 접근 없음). 미래 데이터 금지: 매수일 규칙은 매수일 종가까지의 값만 쓰고,
음수 shift를 쓰지 않는다. 200일선은 L1b와 같은 200행 단순 평균(같으면 '위' — D1 구현 해석 16번).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CANDIDATES = ["D0", "D1", "D2", "D3", "D4", "D5"]
HIGH_SEASON = (11, 12, 1, 2, 3, 4)


# ── 신호 ─────────────────────────────────────────────────────────────────────


def signal_facts(close: pd.Series, sma_days: int = 200) -> pd.DataFrame:
    """날마다 D1·D2·D4 판단값 (그날 종가까지만 사용).

    입력: close(신호 지수 종가, 날짜 인덱스, 데이터 첫 행부터) / 출력: DataFrame(last_month_ret, dd, below)
    - last_month_ret: 그날이 속한 달의 전 달 수익 = 전 달 마지막 거래일 종가 ÷ 전전 달 마지막 거래일 종가 − 1 (구현 해석 13번)
    - dd: 그날 종가 ÷ 데이터 첫 행부터 그날까지 최고 종가 − 1 (14번)
    - below: 종가 < 200일 SMA면 1.0, 같거나 위면 0.0, SMA 미산출 NaN (16번 — L1b above_sma와 평균은 같고, 같을 때만 반대로 '위')
    """
    close = close.sort_index()
    periods = close.index.to_period("M")
    month_end = close.groupby(periods).last()
    month_ret = month_end / month_end.shift(1) - 1.0  # M월 수익(M월 말 확정)
    prev_month_ret = month_ret.shift(1)  # M월에 쓰는 값 = M−1월 수익
    sma = close.rolling(sma_days).mean()  # L1b above_sma와 같은 200행 단순 평균
    below = (close < sma).astype(float).where(sma.notna())  # 같으면 위(0.0) — 구현 해석 16번
    return pd.DataFrame({
        "last_month_ret": prev_month_ret.reindex(periods).to_numpy(),
        "dd": (close / close.cummax() - 1.0).to_numpy(),
        "below": below.to_numpy(),
    }, index=close.index)


def buy_dates(days: pd.DatetimeIndex, months: list[pd.Period], rule: str = "first") -> list[pd.Timestamp]:
    """달마다 매수일: first = 첫 거래일, mid = 15일 이후 첫 거래일, last = 마지막 거래일. 없는 달은 ValueError."""
    days = pd.DatetimeIndex(days)
    per = days.to_period("M")
    out = []
    for m in months:
        in_m = days[per == m]
        if not len(in_m):
            raise ValueError(f"{m}에 거래일이 없습니다")
        if rule == "first":
            out.append(in_m[0])
        elif rule == "mid":
            later = in_m[in_m.day >= 15]
            if not len(later):
                raise ValueError(f"{m}에 15일 이후 거래일이 없습니다")
            out.append(later[0])
        elif rule == "last":
            out.append(in_m[-1])
        else:
            raise ValueError(f"알 수 없는 매수일 규칙: {rule}")
    return out


# ── 규칙 ─────────────────────────────────────────────────────────────────────


def d2_multiple(dd: float) -> float:
    """고점 대비 하락폭 → 배수 (경계값은 더 깊은 칸, 구현 해석 14번)."""
    if dd > -0.10:
        return 0.7
    if dd > -0.20:
        return 1.5
    if dd > -0.30:
        return 2.0
    return 3.0


def rule_amount(cand: str, base: float, *, month: int, t: int, last_month_ret: float | None = None,
                dd: float | None = None, below: float | None = None, equity_value: float = 0.0) -> float:
    """그달 규칙이 사라고 하는 금액(기준 통화, 현금 한도 적용 전).

    입력: cand(D0~D5), base(B), month(매수일의 달 1~12), t(구간 안 몇 번째 달, 0부터),
         last_month_ret·dd·below(signal_facts 값), equity_value(매수 직전 주식 평가액, D3용, 기준 통화)
    """
    if cand == "D0":
        return base
    if cand == "D1":
        if last_month_ret is None or np.isnan(last_month_ret):
            raise ValueError("D1: 지난달 수익을 구할 수 없습니다")
        return 1.5 * base if last_month_ret < 0 else 0.5 * base
    if cand == "D2":
        if dd is None or np.isnan(dd):
            raise ValueError("D2: 하락폭을 구할 수 없습니다")
        return d2_multiple(dd) * base
    if cand == "D3":
        return max(0.0, base * (t + 1) - equity_value)
    if cand == "D4":
        if below is None or np.isnan(below):
            raise ValueError("D4: 200일선을 구할 수 없습니다")
        return 2.0 * base if below == 1.0 else 0.5 * base
    if cand == "D5":
        return 1.5 * base if month in HIGH_SEASON else 0.5 * base
    raise ValueError(f"알 수 없는 후보: {cand}")


# ── 장부 ─────────────────────────────────────────────────────────────────────


@dataclass
class Step:
    """매수일 하나(또는 마지막 평가일)의 자료. growth·div는 직전 매수일 기준 누적."""

    date: pd.Timestamp
    price: float
    fx: float  # QQQ: 그날 원/달러 환율, 달러 데이터: 1.0
    growth: float  # 현금 이자 배수: 직전 매수일 다음 거래일 ~ 이날(포함)
    div_per_share: float  # 직전 매수일 다음 날 ~ 이날 전날 배당락일의 주당 배당 합
    month: int
    last_month_ret: float
    dd: float
    below: float


def build_steps(days: pd.DatetimeIndex, price: pd.Series, fx: pd.Series, rate: pd.Series, div: pd.Series,
                facts: pd.DataFrame, dates: list[pd.Timestamp], valuation: pd.Timestamp) -> list[Step]:
    """매수일 목록 + 평가일 → Step 목록(마지막 원소 = 평가일). 입력 Series는 모두 날짜 인덱스."""
    days = pd.DatetimeIndex(days)
    pts = list(dates) + [valuation]
    log_g = np.log1p(rate.reindex(days).fillna(0.0)).cumsum()
    div_cum = div.reindex(days).fillna(0.0).cumsum()
    steps = []
    prev = None
    for d in pts:
        if prev is None:
            g, dv = 1.0, 0.0
        else:
            g = float(np.exp(log_g.loc[d] - log_g.loc[prev]))  # (prev, d]
            before = days[days < d][-1]
            dv = float(div_cum.loc[before] - div_cum.loc[prev]) if before > prev else 0.0  # (prev, d)
        f = facts.loc[d] if d in facts.index else None
        steps.append(Step(
            date=d, price=float(price.loc[d]), fx=float(fx.loc[d]) if fx is not None else 1.0, growth=g, div_per_share=dv,
            month=d.month,
            last_month_ret=float(f["last_month_ret"]) if f is not None else np.nan,
            dd=float(f["dd"]) if f is not None else np.nan,
            below=float(f["below"]) if f is not None else np.nan,
        ))
        prev = d
    return steps


def simulate(steps: list[Step], cand: str, base: float, *, buy_commission_pct: float = 0.07,
             krw: bool = False, fx_spread_pct: float = 0.1, sell_commission_pct: float = 0.07,
             dividend_withholding_pct: float = 15.0, tax_rate: float = 0.22, deduction: float = 2_500_000.0) -> dict:
    """한 구간 장부(구현 해석 8~12번). steps[-1]은 평가일(납입·매수 없음).

    krw=True(QQQ): 납입 base(원)를 그날 환율로 달러 현금에(환전 비용), 규칙 금액은 원화 → 달러 환산,
      평가일에 전량 매도(매도 수수료)·원화 환전·양도세(22%, 공제) → 최종 자산(원).
    krw=False(달러 데이터): 세전, 최종 자산 = 주식 평가액 + 현금(달러).
    출력: dict(final, contributed, cash_ratio_avg, min_cash, spent_total, tax)
    """
    c_buy = buy_commission_pct / 100.0
    shares = cash = basis_krw = spent = 0.0
    ratios = []
    min_cash = 0.0
    for t, s in enumerate(steps[:-1]):
        cash *= s.growth
        if s.div_per_share and shares:
            cash += shares * s.div_per_share * (1 - dividend_withholding_pct / 100.0)
        if krw:
            cash += base / s.fx * (1 - fx_spread_pct / 100.0)
            equity_base = shares * s.price * s.fx
            want = rule_amount(cand, base, month=s.month, t=t, last_month_ret=s.last_month_ret, dd=s.dd, below=s.below,
                               equity_value=equity_base) / s.fx
        else:
            cash += base
            want = rule_amount(cand, base, month=s.month, t=t, last_month_ret=s.last_month_ret, dd=s.dd, below=s.below,
                               equity_value=shares * s.price)
        pay = min(max(want, 0.0), cash)
        if pay > 0:
            shares += pay / (1 + c_buy) / s.price
            cash -= pay
            spent += pay
            basis_krw += pay * s.fx
        min_cash = min(min_cash, cash)
        total = cash + shares * s.price
        ratios.append(cash / total if total > 0 else 0.0)
    v = steps[-1]
    cash *= v.growth
    if v.div_per_share and shares:
        cash += shares * v.div_per_share * (1 - dividend_withholding_pct / 100.0)
    contributed = base * (len(steps) - 1)
    if krw:
        sale = shares * v.price * (1 - sell_commission_pct / 100.0)
        gain = sale * v.fx - basis_krw
        tax = tax_rate * max(gain - deduction, 0.0)
        final = (sale + cash) * v.fx * (1 - fx_spread_pct / 100.0) - tax
    else:
        tax = 0.0
        final = shares * v.price + cash
    return {"final": final, "contributed": contributed, "cash_ratio_avg": float(np.mean(ratios)) if ratios else 0.0,
            "min_cash": min_cash, "spent_total": spent, "tax": tax}
