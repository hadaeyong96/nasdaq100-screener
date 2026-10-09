"""SW1 실적 발표 후 흐름(PEAD) 매매 계산 — 순수 함수.

기준: docs/sw1_plan.md(잠금본 v2), configs/sw1_preregistration.yaml, docs/sw1_interpretations.md.
네트워크·파일 접근 없음. 입력은 거래일 달력(DatetimeIndex)과 종목별 일봉(open·close), 출력은 표·숫자.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Costs:
    """비용 (퍼센트). D1 설정 그대로: 매수·매도 수수료 0.07%, 환전 0.1%(원→달러, 달러→원 각각)."""

    buy_commission_pct: float = 0.07
    sell_commission_pct: float = 0.07
    fx_spread_pct: float = 0.1


def buy_index(days: pd.DatetimeIndex, d0: pd.Timestamp) -> int | None:
    """반응일 d0 → 매수일(다음 거래일)의 달력 위치. 달력 끝이면 None (해석 6)."""
    i = days.searchsorted(pd.Timestamp(d0), side="right")
    return int(i) if i < len(days) else None


def plan_exit(days: pd.DatetimeIndex, ib: int, hold: int) -> int | None:
    """매수 위치 ib → 예정 매도 위치 ib + hold. 달력(봉인일) 밖이면 None → 매매 제외 (H1)."""
    j = ib + hold
    return j if j < len(days) else None


def realized_exit(close: pd.Series, buy_day: pd.Timestamp, planned: pd.Timestamp,
                  stop_level: float | None = None) -> tuple[pd.Timestamp, float, str] | None:
    """실제 매도일·매도가·사유.

    입력: close(그 종목 종가, 날짜 오름차순), buy_day, planned(예정 매도일), stop_level(종가 손절선, 없으면 None)
    출력: (매도일, 종가, '만기'|'손절'|'가격 끝'|'행 없음') 또는 None(매수일 뒤 종가가 하나도 없음)
    - 손절: 매수 다음 날부터 예정 매도일까지 종가 ≤ stop_level인 첫날 (해석 11)
    - 예정 매도일 행이 없으면 그 전 마지막 종가 (해석 8·9: 상장폐지·인수·거래 정지)
    """
    w = close.loc[(close.index >= buy_day) & (close.index <= planned)].dropna()
    if w.empty:
        return None
    if stop_level is not None:
        after = w.iloc[1:]
        hit = after[after <= stop_level]
        if len(hit):
            return hit.index[0], float(hit.iloc[0]), "손절"
    last_day = w.index[-1]
    if last_day == planned:
        return last_day, float(w.iloc[-1]), "만기"
    reason = "가격 끝" if close.dropna().index[-1] < planned else "행 없음"
    return last_day, float(w.iloc[-1]), reason


def make_trades(signals: pd.DataFrame, prices: dict[str, pd.DataFrame], days: pd.DatetimeIndex, hold: int,
                stop_pct: float | None = None, buy_after_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """신호 표 → 매매 표와 매매가 안 된 신호 표.

    입력: signals(열 ticker, d0 + 그 밖의 열은 그대로 옮김), prices(ticker → open·close 일봉), days(거래일 달력),
          hold(보유 거래일), stop_pct(예: 8.0 → 매수가 × 0.92 종가 손절), buy_after_col(있으면 매수일 =
          max(d0 다음 거래일, 그 열 날짜의 다음 거래일) — EPS 확인판)
    출력: (trades, skipped). trades 열: 원래 열 + buy_day, buy_open, plan_sell_day, sell_day, sell_close, exit_reason.
          skipped 열: 원래 열 + skip ('봉인 경계'|'가격 공백'|'보유 중')
    같은 종목 보유 중(매수일 ≤ 새 반응일 < 매도일) 새 신호는 무시한다 (해석 10).
    """
    s = signals.sort_values(["d0", "ticker"]).reset_index(drop=True)
    trades, skipped = [], []
    holding_until: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for _, r in s.iterrows():
        t, d0 = r["ticker"], pd.Timestamp(r["d0"])
        rec = r.to_dict()
        h = holding_until.get(t)
        if h is not None and h[0] <= d0 < h[1]:
            skipped.append({**rec, "skip": "보유 중"})
            continue
        ib = buy_index(days, d0)
        if ib is not None and buy_after_col is not None:
            ib2 = buy_index(days, pd.Timestamp(r[buy_after_col]))
            ib = None if ib2 is None else max(ib, ib2)
        if ib is None or plan_exit(days, ib, hold) is None:
            skipped.append({**rec, "skip": "봉인 경계"})
            continue
        buy_day, planned = days[ib], days[plan_exit(days, ib, hold)]
        px = prices.get(t)
        if px is None or buy_day not in px.index or not np.isfinite(px.at[buy_day, "open"]) or px.at[buy_day, "open"] <= 0:
            skipped.append({**rec, "skip": "가격 공백"})
            continue
        bo = float(px.at[buy_day, "open"])
        ex = realized_exit(px["close"], buy_day, planned, bo * (1 - stop_pct / 100.0) if stop_pct else None)
        if ex is None:
            skipped.append({**rec, "skip": "가격 공백"})
            continue
        trades.append({**rec, "buy_day": buy_day, "buy_open": bo, "plan_sell_day": planned,
                       "sell_day": ex[0], "sell_close": ex[1], "exit_reason": ex[2]})
        holding_until[t] = (buy_day, ex[0])
    cols = list(s.columns)
    tr = pd.DataFrame(trades, columns=cols + ["buy_day", "buy_open", "plan_sell_day", "sell_day", "sell_close", "exit_reason"])
    sk = pd.DataFrame(skipped, columns=cols + ["skip"])
    return tr, sk


def leg_returns(px: pd.DataFrame, buy_day: pd.Timestamp, sell_day: pd.Timestamp) -> float | None:
    """같은 매수일 시가 → 같은 매도일 종가의 달러 수익 (비교 지수·무작위 종목). 그날 행이 없으면 None."""
    if buy_day not in px.index or sell_day not in px.index:
        return None
    o, c = float(px.at[buy_day, "open"]), float(px.at[sell_day, "close"])
    if not (o > 0) or not np.isfinite(c):
        return None
    return c / o - 1.0


def krw_after_cost(usd_ret: np.ndarray | float, fx_buy: np.ndarray | float, fx_sell: np.ndarray | float,
                   costs: Costs = Costs()) -> np.ndarray | float:
    """원화 1을 넣었을 때 비용·환전 뒤 원화 수익 (세전).

    원→달러 환전(스프레드) → 매수 수수료(shares = 달러 / (1+c) / 시가) → 매도 수수료 → 달러→원 환전.
    입력: usd_ret(종가/시가 − 1), fx_buy·fx_sell(원/달러). 출력: 원화 수익률.
    """
    f = costs.fx_spread_pct / 100.0
    cb, cs = costs.buy_commission_pct / 100.0, costs.sell_commission_pct / 100.0
    usd_in = (1.0 - f) / np.asarray(fx_buy, dtype=float)
    out = usd_in / (1.0 + cb) * (1.0 + np.asarray(usd_ret, dtype=float)) * (1.0 - cs) * np.asarray(fx_sell, dtype=float) * (1.0 - f)
    return out - 1.0


def ledger_tax(ret: np.ndarray, year: np.ndarray, rate: float = 0.22) -> np.ndarray:
    """한 장부의 매매별 세금 (원화 1 기준, H2).

    매도 연도마다 순손익 합이 양수면 그 rate가 그해 세금, 그해 이익 난 매매에 이익 크기 비례로 나눈다. 공제 없음.
    입력: ret(매매별 원화 수익, 비용 뒤), year(매도 연도). 출력: 매매별 세금 (≥ 0).
    """
    ret = np.asarray(ret, dtype=float)
    year = np.asarray(year)
    tax = np.zeros_like(ret)
    for y in np.unique(year):
        m = year == y
        net = ret[m].sum()
        if net <= 0:
            continue
        pos = np.where(ret[m] > 0, ret[m], 0.0)
        tax[m] = rate * net * pos / pos.sum()
    return tax


def ledger_tax_many(ret: np.ndarray, year_codes: np.ndarray, n_years: int, rate: float = 0.22) -> np.ndarray:
    """ledger_tax를 여러 회차에 한 번에 (무작위 1,000회).

    입력: ret[회차, 매매], year_codes[회차, 매매](0..n_years-1 정수). 출력: tax[회차, 매매]. 결과는 ledger_tax와 같다.
    """
    ret = np.asarray(ret, dtype=float)
    k, n = ret.shape
    rows = np.repeat(np.arange(k), n)
    cols = year_codes.ravel()
    net = np.zeros((k, n_years))
    posum = np.zeros((k, n_years))
    np.add.at(net, (rows, cols), ret.ravel())
    pos = np.where(ret > 0, ret, 0.0)
    np.add.at(posum, (rows, cols), pos.ravel())
    yr_tax = rate * np.maximum(net, 0.0)
    share = np.divide(yr_tax, posum, out=np.zeros_like(yr_tax), where=posum > 0)
    return pos * share[np.arange(k)[:, None], year_codes]


def percentile_of(value: float, draws: np.ndarray) -> float:
    """무작위 값들 중 value보다 작은 비율 × 100 (같은 값은 절반, 해석 14)."""
    d = np.asarray(draws, dtype=float)
    return float(((d < value).sum() + 0.5 * (d == value).sum()) / len(d) * 100.0)


def verdict(excess_after: pd.Series, reaction_year: pd.Series, random_pct: float, min_trades: int = 300,
            periods: tuple[tuple[int, int], ...] = ((2012, 2016), (2017, 2021)), min_pct: float = 75.0,
            min_years: int = 6) -> dict:
    """판정 기준 1~5 (계획서 6장).

    입력: excess_after(매매별 비용·세금 뒤 초과 수익), reaction_year(반응일 연도), random_pct(기준 2 백분위)
    출력: {'c1'..'c5': bool, 'pass': bool|None(판정 불가), 값들}
    """
    m = float(excess_after.mean()) if len(excess_after) else float("nan")
    per = [float(excess_after[(reaction_year >= a) & (reaction_year <= b)].mean()) for a, b in periods]
    by_year = excess_after.groupby(reaction_year).mean()
    pos_years = int((by_year > 0).sum())
    out = {"mean": m, "random_pct": random_pct, "period_means": per, "pos_years": pos_years, "n_years": len(by_year),
           "n": int(len(excess_after)),
           "c1": bool(m > 0), "c2": bool(random_pct >= min_pct), "c3": bool(all(p > 0 for p in per)),
           "c4": bool(pos_years >= min_years), "c5": bool(len(excess_after) >= min_trades)}
    out["pass"] = None if not out["c5"] else bool(out["c1"] and out["c2"] and out["c3"] and out["c4"])
    return out
