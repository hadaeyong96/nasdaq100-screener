"""E1 이익 성장주 적립 — 분기 EPS·TTM·성장 판정·월별 매매 장부 (docs/e1_plan.md, configs/e1_preregistration.yaml).

순수 함수만 둔다(네트워크·파일·DB·현재 시각 접근 없음). 미래 데이터 금지:
- 공시 사실은 filed ≤ 판단일만 쓴다. 분할 보정도 판단일까지의 분할만 쓴다.
- 장부는 매수일 종가로 체결하고, 판단(대상·순위·매도)은 그 전 거래일(판단일) 정보로만 한다.
- 음수 shift를 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from core.tax import capital_gains_tax

FORMS = ("10-Q", "10-K", "10-Q/A", "10-K/A")
QUARTER_DAYS = (80, 100)
NINE_MONTH_DAYS = (260, 290)
ANNUAL_DAYS = (350, 380)
GAP_DAYS = (80, 100)


# ── 분기 값 ─────────────────────────────────────────────────────────────────


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def _split_factor(splits: pd.Series | None, filed: str, decision: pd.Timestamp) -> float:
    """filed 다음 날 ~ 판단일 분할 비율의 곱 (구현 해석 8번). splits: 날짜 → 비율(예 4.0)."""
    if splits is None or not len(splits):
        return 1.0
    f = pd.Timestamp(filed)
    s = splits[(splits.index > f) & (splits.index <= decision)]
    return float(np.prod(s.to_numpy(dtype=float))) if len(s) else 1.0


def latest_facts(entries: list[dict], decision: pd.Timestamp) -> dict[tuple[str, str], dict]:
    """filed ≤ 판단일·허용 form·start/end/filed 있는 사실 중 (start, end)마다 가장 최근 filed 하나 (구현 해석 4·5번)."""
    cut = decision.date().isoformat()
    out: dict[tuple[str, str], dict] = {}
    for i, e in enumerate(entries):
        if e.get("form") not in FORMS or not e.get("start") or not e.get("end") or not e.get("filed"):
            continue
        if e["filed"] > cut:
            continue
        key = (e["start"], e["end"])
        prev = out.get(key)
        if prev is None or (e["filed"], i) >= (prev["filed"], prev["_i"]):
            out[key] = {**e, "_i": i}
    return out


def quarterly_values(entries: list[dict], decision, splits: pd.Series | None = None) -> pd.Series:
    """판단일까지 공시된 분기 값(분할 보정 후). 인덱스 = 분기말 Timestamp, 오름차순 (구현 해석 6·8번).

    입력: entries(edgar.extract_fact_entries 결과), decision(판단일), splits(None이면 보정 안 함 — 매출)
    """
    decision = pd.Timestamp(decision)
    facts = latest_facts(entries, decision)

    def adj(e):
        return float(e["val"]) / _split_factor(splits, e["filed"], decision)

    q3, nine, annual = {}, [], []
    for (s, e), f in facts.items():
        d = _days(s, e)
        if QUARTER_DAYS[0] <= d <= QUARTER_DAYS[1]:
            q3[e] = (s, adj(f))
        elif NINE_MONTH_DAYS[0] <= d <= NINE_MONTH_DAYS[1]:
            nine.append((s, e, adj(f)))
        elif ANNUAL_DAYS[0] <= d <= ANNUAL_DAYS[1]:
            annual.append((s, e, adj(f)))
    out = {e: v for e, (s, v) in q3.items()}
    for s_a, e_a, v_a in annual:
        if e_a in out:
            continue
        nine_match = [v for s9, e9, v in nine if abs(_days(s_a, s9)) <= 7 and e9 < e_a]
        if nine_match:
            # 9개월 누계가 여럿이면(드묾) 끝이 가장 늦은 것
            best = max(((e9, v) for s9, e9, v in nine if abs(_days(s_a, s9)) <= 7 and e9 < e_a))
            out[e_a] = v_a - best[1]
            continue
        inside = sorted((e, v) for e, (s, v) in q3.items() if s >= s_a and e < e_a)
        if len(inside) == 3:
            out[e_a] = v_a - sum(v for _, v in inside)
    if not out:
        return pd.Series(dtype=float)
    ser = pd.Series(out)
    ser.index = pd.to_datetime(ser.index)
    return ser.sort_index()


def consecutive_tail(q: pd.Series, n: int) -> pd.Series | None:
    """가장 최근 분기말부터 거꾸로 n개가 이웃 간격 80~100일로 이어지면 그 n개, 아니면 None (구현 해석 7번)."""
    if len(q) < n:
        return None
    tail = q.iloc[-n:]
    gaps = np.diff(tail.index.values).astype("timedelta64[D]").astype(int)
    if ((gaps < GAP_DAYS[0]) | (gaps > GAP_DAYS[1])).any():
        return None
    return tail


@dataclass
class GrowthInfo:
    ttm: list[float] | None  # [TTM(t), TTM(t−4), TTM(t−8), TTM(t−12)] (16분기 있을 때)
    growth: float | None  # TTM(t)/TTM(t−4) − 1 (8분기 있고 TTM(t−4) > 0일 때)
    eligible: bool  # 3년 연속 증가 + TTM(t−4) > 0
    pool_ok: bool  # 분기 16개 연속 (성장 조건 전 대상 자격)
    last_quarter: pd.Timestamp | None


def growth_info(q: pd.Series) -> GrowthInfo:
    """분기 값 → TTM·증가율·성장 조건 (구현 해석 7·9·10번)."""
    last = q.index[-1] if len(q) else None
    t8 = consecutive_tail(q, 8)
    growth = None
    if t8 is not None:
        cur, base = float(t8.iloc[4:].sum()), float(t8.iloc[:4].sum())
        growth = cur / base - 1.0 if base > 0 else None
    t16 = consecutive_tail(q, 16)
    if t16 is None:
        return GrowthInfo(None, growth, False, False, last)
    v = t16.to_numpy(dtype=float)
    ttm = [float(v[12:16].sum()), float(v[8:12].sum()), float(v[4:8].sum()), float(v[0:4].sum())]
    eligible = ttm[0] > ttm[1] > ttm[2] > ttm[3] and ttm[1] > 0
    return GrowthInfo(ttm, growth, eligible, True, last)


def should_sell(info: GrowthInfo | None) -> bool:
    """보유 종목 매도 판정: 증가율 < 0 또는 계산 불가면 True (구현 해석 10번)."""
    return info is None or info.growth is None or info.growth < 0


def rank_growth(infos: dict[str, GrowthInfo], pool: list[str]) -> list[str]:
    """pool 중 성장 조건 통과 종목을 증가율 높은 순, 같으면 티커 오름차순으로 (구현 해석 9번)."""
    ok = [(t, infos[t].growth) for t in pool if infos.get(t) is not None and infos[t].eligible and infos[t].growth is not None]
    return [t for t, _ in sorted(ok, key=lambda x: (-x[1], x[0]))]


def merge_revenue(entries_by_tag: list[list[dict]], decision, splits=None) -> pd.Series:
    """매출: 태그 우선순위대로 분기말마다 값이 있는 첫 태그 (구현 해석 24번)."""
    out = pd.Series(dtype=float)
    for entries in entries_by_tag:
        q = quarterly_values(entries, decision, splits)
        if len(q):
            out = pd.concat([out, q[~q.index.isin(out.index)]]) if len(out) else q
    return out.sort_index()


# ── 장부 ─────────────────────────────────────────────────────────────────────


@dataclass
class Market:
    """장부 입력(모두 같은 거래일 인덱스).

    close: DataFrame(거래일 × 티커, 그날 종가 없으면 NaN), div_cum: DataFrame(주당 배당 누적합),
    last_date: {티커: 마지막 종가 날짜}, fx: Series(원/달러), buy_days: 매수일 목록
    """

    days: pd.DatetimeIndex
    close: pd.DataFrame
    div_cum: pd.DataFrame
    last_date: dict
    fx: pd.Series
    buy_days: list
    ffill: pd.DataFrame = field(default=None)

    def __post_init__(self):
        if self.ffill is None:
            self.ffill = self.close.ffill()


@dataclass
class Costs:
    buy_pct: float = 0.07
    sell_pct: float = 0.07
    fx_spread_pct: float = 0.1
    withholding_pct: float = 15.0
    tax_rate: float = 0.22
    deduction_krw: float = 2_500_000.0


def simulate(market: Market, picks: list[list[str]], sell_flags: list[dict], valuation: pd.Timestamp, *,
             monthly_krw: float = 300_000.0, costs: Costs = Costs(), apply_tax: bool = True, daily: bool = True) -> dict:
    """월별 매매 장부 (구현 해석 13~20번).

    입력: picks[i] = i번째 매수일에 살 티커 목록(현금 전부를 똑같이 나눔, 빈 목록이면 현금 보유),
         sell_flags[i] = {티커: True면 매도} — 보유 종목 중 여기 없는 종목은 매도하지 않는다,
         valuation = 평가일(전량 매도·마지막 해 세금)
    출력: dict(final(원화, 세후 또는 apply_tax=False면 세전), units, unit_value(일별 Series, daily일 때),
              mdd_pct, sells, holding_days(닫힌 보유 구간), taxes{연도: 원}, contributed, min_cash, log)
    """
    c_buy, c_sell, spread = costs.buy_pct / 100, costs.sell_pct / 100, costs.fx_spread_pct / 100
    keep = 1 - costs.withholding_pct / 100
    shares: dict[str, float] = {}
    basis: dict[str, float] = {}
    opened: dict[str, pd.Timestamp] = {}
    cash = 0.0
    realized: dict[int, float] = {}
    taxes: dict[int, float] = {}
    pending_tax: tuple[int, float] | None = None
    sells = 0
    holding_days: list[int] = []
    min_cash = 0.0
    units = 0.0
    snapshots = []  # (날짜, shares 복사, cash) — 일별 평가용
    log = []
    events = list(market.buy_days) + [valuation]
    prev = None

    def price_on(t, d):
        v = market.close.at[d, t] if t in market.close.columns else np.nan
        return None if pd.isna(v) else float(v)

    def sell(t, frac, px, d, fx):
        nonlocal cash, sells
        n = shares[t] * frac
        proceeds = n * px * (1 - c_sell)
        gain = proceeds * fx - basis[t] * frac
        realized[d.year] = realized.get(d.year, 0.0) + gain
        cash += proceeds
        shares[t] -= n
        basis[t] -= basis[t] * frac
        if frac >= 1 - 1e-12:
            del shares[t], basis[t]
            holding_days.append((d - opened.pop(t)).days)
            sells += 1

    def value_usd(d):
        return cash + sum(n * float(market.ffill.at[d, t]) for t, n in shares.items())

    for k, d in enumerate(events):
        fx = float(market.fx.loc[d])
        # 배당: (직전 이벤트, d] 배당락분 — 그 사이 보유 주식 기준
        if prev is not None and shares:
            for t, n in shares.items():
                cash += n * float(market.div_cum.at[d, t] - market.div_cum.at[prev, t]) * keep
        is_val = k == len(events) - 1
        if not is_val:
            # ① 전년 세금 (그해 첫 매수일)
            if apply_tax and pending_tax is not None and pending_tax[0] < d.year:
                usd = pending_tax[1] / (fx * (1 - spread))
                if usd > cash and shares:
                    short = usd - cash
                    tot = sum(n * float(market.ffill.at[d, t]) for t, n in shares.items())
                    for t in list(shares):
                        px = float(market.ffill.at[d, t])
                        frac = min(1.0, short * (shares[t] * px / tot) / (shares[t] * px * (1 - c_sell))) if tot > 0 else 0.0
                        sell(t, frac, px, d, fx)
                cash -= usd
                pending_tax = None
            # ② 매도: 규칙 매도 + 가격 끊김
            for t in sorted(list(shares)):
                px = price_on(t, d)
                gone = px is None and market.last_date.get(t) is not None and market.last_date[t] < d
                if gone:
                    sell(t, 1.0, float(market.ffill.at[d, t]), d, fx)
                elif px is not None and sell_flags[k].get(t, False):
                    sell(t, 1.0, px, d, fx)
            # 단위: 납입 직전 계좌 가치로
            v_before = value_usd(d) * fx
            if units == 0:
                units = monthly_krw / 1.0
            else:
                units += monthly_krw / (v_before / units) if v_before > 0 else 0.0
            # ③ 납입 ④ 매수
            cash += monthly_krw / fx * (1 - spread)
            buy = [t for t in picks[k] if price_on(t, d) is not None]
            if buy and cash > 0:
                each = cash / len(buy)
                for t in buy:
                    px = price_on(t, d)
                    if t not in shares:
                        shares[t], basis[t], opened[t] = 0.0, 0.0, d
                    shares[t] += each / (1 + c_buy) / px
                    basis[t] += each * fx
                cash = 0.0
            log.append({"date": d, "bought": buy, "held": sorted(shares)})
            min_cash = min(min_cash, cash)
            # 해가 바뀌기 전 마지막 매수일이면 그해 세금 확정(다음 해 첫 매수일 납부)
            nxt = events[k + 1]
            if nxt.year != d.year and k + 1 < len(events) - 1:
                tax = costs.tax_rate * max(realized.get(d.year, 0.0) - costs.deduction_krw, 0.0) if apply_tax else 0.0
                taxes[d.year] = tax
                pending_tax = (d.year, tax) if tax > 0 else None
        else:
            for t in sorted(list(shares)):
                sell(t, 1.0, float(market.ffill.at[d, t]), d, fx)
            # 그 사이 해가 바뀌었는데 확정 안 된 해(마지막 매수일과 평가일 연도 다름)는 없다(평가일 = 12-31)
            tax = costs.tax_rate * max(realized.get(d.year, 0.0) - costs.deduction_krw, 0.0) if apply_tax else 0.0
            taxes[d.year] = taxes.get(d.year, 0.0) + tax
            final = cash * fx * (1 - spread) - tax
        snapshots.append((d, dict(shares), cash, units))
        prev = d

    out = {"final": final, "units": units, "sells": sells, "holding_days": holding_days, "taxes": taxes,
           "contributed": monthly_krw * len(market.buy_days), "min_cash": min_cash, "log": log}
    if daily:
        out["unit_value"] = unit_values(market, snapshots, valuation, final, units, keep, monthly_krw)
        uv = out["unit_value"]
        out["mdd_pct"] = float((uv / uv.cummax() - 1).min() * 100)
    return out


def unit_values(market: Market, snapshots, valuation, final, units_end, keep, monthly_krw=None) -> pd.Series:
    """일별 단위 가치(구현 해석 20번). snapshots = [(이벤트 날짜, 보유, 현금, 그때 단위 수)].

    이벤트 사이에는 보유를 고정하고 그 사이 배당 누적분(원천 후)을 더한다. 평가일 = 세후 최종 자산 ÷ 단위.
    """
    days = market.days[(market.days >= snapshots[0][0]) & (market.days <= valuation)]
    snap_days = [s[0] for s in snapshots]
    vals = []
    j = 0
    for d in days:
        while j + 1 < len(snap_days) and snap_days[j + 1] <= d:
            j += 1
        sd, sh, cs, u = snapshots[j]
        if d == valuation:
            vals.append(final / units_end)
            continue
        v = cs
        for t, n in sh.items():
            v += n * float(market.ffill.at[d, t])
            if d > sd:
                v += n * float(market.div_cum.at[d, t] - market.div_cum.at[sd, t]) * keep
        vals.append(v * float(market.fx.loc[d]) / u)
    return pd.Series(vals, index=days)
