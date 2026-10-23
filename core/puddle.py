"""상승장 대형주 웅덩이 갈아타기 — 순수 함수 (docs/prereg/puddle_megacap.md, 2026-10-23).

기본 QQQ 100%. 매달 첫 거래일 그 시점 나스닥100 중 20일 평균 거래대금 상위 10(같은 회사 여러 종류 주식은 하나로)에서,
자격(종가 > 200일선, 200일선 > 20거래일 전 200일선)·지수 조건(QQQ 종가 > 200일선)을 갖춘 종목이 단기 이동평균 아래로
내려오면 QQQ를 팔아 단계적으로 산다(5%씩 최대 15%, 동시 2종목). 메워짐·200일선 이탈·250거래일이면 전량 QQQ로.

네트워크·파일·현재 시각에 접근하지 않는다. 음수 shift를 쓰지 않는다(판단은 그날까지의 값만, 체결은 다음 거래일 종가).
세금: 자산별 원화 평균 원가, 과세연도 원화 실현손익 − 250만 원 → 22%, 다음 해 5월 첫 거래일 납부,
12월 공제 소진(그해 마지막 거래일 2거래일 전 종가에 이익 큰 자산부터 남은 공제만큼 팔고 같은 날 다시 삼), 끝날 전부 판 것으로 보고 세후 원화.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REASONS = ("메워짐", "200일선 이탈", "250거래일", "가격 끝")


@dataclass(frozen=True)
class Params:
    stages: tuple[int, int, int] = (20, 60, 120)  # 1·2·3차 이동평균
    tranche: float = 0.05  # 한 차수 비율(신호 날 전체 자산 기준)
    cap: float = 0.15  # 종목당 최대
    max_names: int = 2  # 동시 최대 종목 수
    cooldown: int = 20  # 매도 체결 뒤 다시 안 사는 거래일
    max_hold: int = 250  # 첫 체결부터 최대 보유 거래일
    high_lookback: int = 60  # 메워짐: 직전 N거래일 최고 종가
    top_n: int = 10
    dv_window: int = 20
    cost: float = 0.001  # 편도
    mode: str = "swap"  # swap(QQQ를 팔아 삼) | cash30(원형: 현금 30% 따로, 현금으로 삼, 판 돈은 현금으로)
    tax: bool = True
    deduction_krw: float = 2_500_000.0
    rate: float = 0.22
    start_krw: float = 100_000_000.0


@dataclass
class Inputs:
    """한 구간 입력 (위치 0 = 구간 첫 거래일, 지표용 앞 데이터는 warm 배열에 이미 반영).

    dates(n), qqq_sig·qqq_px(n: 신호용 분할 반영 종가·평가용 총수익 가격), qqq_sma200(n),
    sig·px(n×K: 종목 신호용 종가·평가용 가격, 없으면 NaN), sma(dict 기간 → n×K), sma200_prev20(n×K: 20거래일 전 200일선),
    hi_prev(n×K: 직전 high_lookback거래일 최고 종가), top(dict 위치 → 그달 상위 10 열 번호 목록, 그달 첫 거래일 위치 키),
    fx(n: 원/달러), names(K)."""

    dates: pd.DatetimeIndex
    qqq_sig: np.ndarray
    qqq_px: np.ndarray
    qqq_sma200: np.ndarray
    sig: np.ndarray
    px: np.ndarray
    sma: dict
    sma200_prev20: np.ndarray
    hi_prev: np.ndarray
    top: dict
    fx: np.ndarray
    names: list


@dataclass
class Result:
    nav: np.ndarray  # 일별 세전 가치(달러, 세금 납부 반영)
    stock_w: np.ndarray  # 일별 종목 비중 합
    end_after_tax_krw: float
    costs_usd: float
    costs_krw: float
    taxes_krw: float
    fills: list = field(default_factory=list)  # 차수 체결 [{pos, k, tranche, usd}]
    trades: list = field(default_factory=list)  # 종목 한 번 보유 [{k, fills, exit_pos, reason, ret, qqq_ret, excess}]
    entry_signal_days: list = field(default_factory=list)  # 새 종목 1차 신호 위치(무작위 대조용)
    harvests: int = 0


# ── 지표 (순수) ───────────────────────────────────────────────────────────────


def sma(a: pd.DataFrame | pd.Series, n: int):
    """단순이동평균(창이 다 차야 값)."""
    return a.rolling(n, min_periods=n).mean()


def avg_dollar_volume(close: pd.DataFrame, volume: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """20일 평균 거래대금 = (종가 × 거래량)의 n일 평균(창의 n일 모두 값이 있어야)."""
    return sma(close * volume, n)


def prior_high(close: pd.DataFrame, n: int = 60) -> pd.DataFrame:
    """직전 n거래일(그날 제외) 최고 종가."""
    return close.shift(1).rolling(n, min_periods=n).max()


def month_first_positions(dates: pd.DatetimeIndex) -> list[int]:
    """달마다 첫 거래일 위치."""
    m = dates.to_period("M")
    return [i for i in range(len(dates)) if i == 0 or m[i] != m[i - 1]]


def top_by_dollar_volume(dv_row: np.ndarray, member_row: np.ndarray, company: np.ndarray, n: int = 10) -> list[int]:
    """그날 거래대금·구성 여부 → 상위 n 열 번호. 같은 회사(company 같은 값) 여러 열은 거래대금 큰 열 하나만."""
    best: dict[int, tuple[float, int]] = {}
    for k in np.where(member_row & np.isfinite(dv_row))[0]:
        c = int(company[k])
        if c not in best or dv_row[k] > best[c][0]:
            best[c] = (float(dv_row[k]), int(k))
    return [k for _, k in sorted(best.values(), key=lambda x: -x[0])[:n]]


def may_payment_positions(dates: pd.DatetimeIndex) -> dict[int, int]:
    """해마다 5월 첫 거래일 위치 → 직전 과세연도."""
    out, seen = {}, set()
    for i, d in enumerate(dates):
        if d.month == 5 and d.year not in seen:
            seen.add(d.year)
            out[i] = d.year - 1
    return out


def harvest_positions(dates: pd.DatetimeIndex) -> set[int]:
    """해마다 마지막 거래일 2거래일 전 위치(12월 공제 소진, core.tax_harvest.deadline과 같은 날). 구간 마지막 해는 끝날이 그해 마지막 거래일일 때만."""
    out = set()
    yrs = dates.year
    for y in sorted(set(yrs)):
        idx = np.where(yrs == y)[0]
        if len(idx) >= 3 and dates[idx[-1]].month == 12 and dates[idx[-1]].day >= 24:
            out.add(int(idx[-3]))
    return out


# ── 장부 ─────────────────────────────────────────────────────────────────────


def simulate(inp: Inputs, p: Params = Params(), *, random_days: list[int] | None = None,
             rng: np.random.Generator | None = None, no_stocks: bool = False) -> Result:
    """한 구간을 위치 0(QQQ 100%, 원형은 QQQ 70%·현금 30%)부터 끝까지 돌린다.

    random_days: 주면 무작위 대조 — 전략의 새 종목 1차 신호 날(같은 날 여러 번이면 그 수만큼)에만,
    그날 상위 10 중 자격(+지수 조건·보유 아님·재매수 금지 아님) 종목에서 무작위로 골라 1차를 산다(20일선 하향 조건 없음).
    이후 차수·매도는 같은 규칙. no_stocks: QQQ 보유 비교(같은 세금 규칙).
    """
    n, K = inp.px.shape
    f = inp.fx
    c = p.cost
    s1, s2, s3 = p.stages
    pxf = pd.DataFrame(inp.px).ffill().to_numpy()  # 평가용(중간 빈 날은 직전 값)
    last_valid = np.array([np.where(np.isfinite(inp.px[:, k]))[0].max() if np.isfinite(inp.px[:, k]).any() else -1
                           for k in range(K)])
    years = inp.dates.year.to_numpy()
    may = may_payment_positions(inp.dates)
    harv = harvest_positions(inp.dates)
    rand_ct = Counter(random_days or [])

    q_units = 0.0
    cash = 0.0
    v0 = p.start_krw / f[0]
    if p.mode == "cash30":
        q_units, cash = 0.7 * v0 / inp.qqq_px[0], 0.3 * v0
        q_basis, cash_basis = 0.7 * p.start_krw, 0.3 * p.start_krw
    else:
        q_units, q_basis, cash_basis = v0 / inp.qqq_px[0], p.start_krw, 0.0
    units = np.zeros(K)
    basis = np.zeros(K)
    realized: dict[int, float] = {}
    costs = costs_krw = taxes = 0.0
    held: dict[int, dict] = {}  # k → {entry_pos, tranches, fills:[(pos, usd)]}
    last_sell: dict[int, int] = {}
    pend_sell: dict[int, str] = {}
    pend_buy: list[tuple[int, int, float]] = []  # (k, tranche, usd)
    fills, trades, entry_days = [], [], []
    nav = np.empty(n)
    stock_w = np.empty(n)
    harvests = 0
    top_now: list[int] = []

    def qprice(t):
        return inp.qqq_px[t]

    def _sell_qqq(value: float, t: int) -> float:
        nonlocal q_units, q_basis, costs, costs_krw
        q = min(value / qprice(t), q_units)
        if q <= 0:
            return 0.0
        b = q_basis * q / q_units
        gross = q * qprice(t)
        costs += gross * c
        costs_krw += gross * c * f[t]
        realized[years[t]] = realized.get(years[t], 0.0) + gross * (1 - c) * f[t] - b
        q_units -= q
        q_basis -= b
        return gross * (1 - c)

    def _buy_qqq(usd: float, t: int) -> None:
        nonlocal q_units, q_basis, costs, costs_krw
        if usd <= 0:
            return
        spend = usd / (1 + c)
        costs += usd - spend
        costs_krw += (usd - spend) * f[t]
        q_units += spend / qprice(t)
        q_basis += usd * f[t]

    def _sell_stock(k: int, value: float, t: int) -> float:
        nonlocal costs, costs_krw
        q = min(value / pxf[t, k], units[k])
        if q <= 0:
            return 0.0
        b = basis[k] * q / units[k]
        gross = q * pxf[t, k]
        costs += gross * c
        costs_krw += gross * c * f[t]
        realized[years[t]] = realized.get(years[t], 0.0) + gross * (1 - c) * f[t] - b
        units[k] -= q
        basis[k] -= b
        if units[k] < 1e-12:
            units[k] = basis[k] = 0.0
        return gross * (1 - c)

    def _buy_stock(k: int, usd: float, t: int) -> None:
        nonlocal costs, costs_krw
        spend = usd / (1 + c)
        costs += usd - spend
        costs_krw += (usd - spend) * f[t]
        units[k] += spend / pxf[t, k]
        basis[k] += usd * f[t]

    def _cash_out(usd: float, t: int) -> float:
        """현금(원형)에서 꺼낸다 — 원화 원가는 비율대로 줄인다(현금은 손익 없음)."""
        nonlocal cash, cash_basis
        take = min(usd, cash)
        if cash > 0:
            cash_basis -= cash_basis * take / cash
        cash -= take
        return take

    def _cash_in(usd: float, t: int) -> None:
        nonlocal cash, cash_basis
        cash += usd
        cash_basis += usd * f[t]

    def _funding(usd: float, t: int) -> float:
        """종목 살 돈: swap은 QQQ를 팔아(판 금액 = usd), 원형은 현금에서."""
        if p.mode == "cash30":
            return _cash_out(usd, t)
        return _sell_qqq(usd, t)

    def _proceeds(usd: float, t: int) -> None:
        if p.mode == "cash30":
            _cash_in(usd, t)
        else:
            _buy_qqq(usd, t)

    def _close_trade(k: int, t: int, reason: str, got_usd: float) -> None:
        h = held.pop(k)
        paid = sum(u for _, u in h["fills"])
        qqq_equiv = sum(u * qprice(t) / qprice(pp) for pp, u in h["fills"])
        trades.append({"k": k, "fills": h["fills"], "tranches": h["tranches"], "entry_pos": h["entry_pos"], "exit_pos": t,
                       "reason": reason, "ret": got_usd / paid - 1 if paid > 0 else np.nan,
                       "qqq_ret": qqq_equiv / paid - 1 if paid > 0 else np.nan,
                       "excess": got_usd / qqq_equiv - 1 if qqq_equiv > 0 else np.nan})
        last_sell[k] = t

    for t in range(n):
        # 1) 5월 첫 거래일 세금 납부 (원형: 현금 → QQQ → 종목, swap: QQQ → 종목)
        if p.tax and t in may:
            owed = max(realized.get(may[t], 0.0) - p.deduction_krw, 0.0) * p.rate
            if owed > 0:
                need = owed / f[t]
                got = _cash_out(need, t) if p.mode == "cash30" else 0.0
                if got < need - 1e-9:
                    got += _sell_qqq((need - got) / (1 - c), t)
                for k in list(held):
                    if got >= need - 1e-9:
                        break
                    got += _sell_stock(k, (need - got) / (1 - c), t)
                taxes += owed
        # 2) 어제 신호 체결: 가격 끝 → 매도 → 매수
        for k in list(held):
            if t > last_valid[k] and k not in pend_sell:
                pend_sell[k] = "가격 끝"
        for k, reason in list(pend_sell.items()):
            if k in held:
                got = _sell_stock(k, units[k] * pxf[t, k] + 1.0, t) if units[k] > 0 else 0.0
                _proceeds(got, t)
                _close_trade(k, t, reason, got)
        pend_sell = {}
        for k, tr, usd in pend_buy:
            if not np.isfinite(inp.px[t, k]):
                continue  # 체결일 가격 없음 → 체결 안 함
            got = _funding(usd, t)
            if got <= 0:
                continue
            _buy_stock(k, got, t)
            h = held.setdefault(k, {"entry_pos": t, "tranches": 0, "fills": []})
            h["tranches"] = tr
            h["fills"].append((t, got))
            fills.append({"pos": t, "k": k, "tranche": tr, "usd": got})
        pend_buy = []
        # 3) 12월 공제 소진
        if p.tax and t in harv:
            assets = [("q", -1)] + [("s", k) for k in held if units[k] > 0]

            def unreal(a):
                if a[0] == "q":
                    return q_units * qprice(t) * f[t] - q_basis
                return units[a[1]] * pxf[t, a[1]] * f[t] - basis[a[1]]

            for a in sorted(assets, key=lambda a: -unreal(a)):
                left = p.deduction_krw - realized.get(years[t], 0.0)
                if left <= 1.0:
                    break
                if a[0] == "q":
                    if q_units <= 0:
                        continue
                    g = qprice(t) * (1 - c) * f[t] - q_basis / q_units
                    if g <= 0:
                        continue
                    _buy_qqq(_sell_qqq(min(q_units, left / g) * qprice(t), t), t)
                else:
                    k = a[1]
                    g = pxf[t, k] * (1 - c) * f[t] - basis[k] / units[k]
                    if g <= 0:
                        continue
                    _buy_stock(k, _sell_stock(k, min(units[k], left / g) * pxf[t, k], t), t)
                harvests += 1
        # 4) 오늘 종가 가치
        sv = float(np.nansum(units * np.nan_to_num(pxf[t])))
        nav[t] = q_units * qprice(t) + sv + cash
        stock_w[t] = sv / nav[t] if nav[t] > 0 else 0.0
        if t + 1 >= n or no_stocks:
            continue
        # 5) 오늘 신호 → 내일 종가 체결
        if t in inp.top:
            top_now = list(inp.top[t])
        for k, h in held.items():
            x = inp.sig[t, k]
            if not np.isfinite(x):
                continue
            if np.isfinite(inp.hi_prev[t, k]) and x > inp.hi_prev[t, k]:
                pend_sell[k] = REASONS[0]
            elif np.isfinite(inp.sma[200][t, k]) and x < inp.sma[200][t, k]:
                pend_sell[k] = REASONS[1]
            elif t - h["entry_pos"] >= p.max_hold:
                pend_sell[k] = REASONS[2]
        index_ok = np.isfinite(inp.qqq_sma200[t]) and inp.qqq_sig[t] > inp.qqq_sma200[t]
        if not index_ok:
            continue

        def qualified(k):
            x, m200, prev = inp.sig[t, k], inp.sma[200][t, k], inp.sma200_prev20[t, k]
            return np.isfinite(x) and np.isfinite(m200) and np.isfinite(prev) and x > m200 and m200 > prev

        total = nav[t]
        # 다음 차수 (보유 중, 오늘 매도 신호 없음)
        for k, h in held.items():
            if k in pend_sell or not qualified(k):
                continue
            x = inp.sig[t, k]
            want = None
            if h["tranches"] == 1 and np.isfinite(inp.sma[s2][t, k]) and x < inp.sma[s2][t, k]:
                want = 2
            elif h["tranches"] == 2 and np.isfinite(inp.sma[s3][t, k]) and x < inp.sma[s3][t, k] and x > inp.sma[200][t, k]:
                want = 3
            if want:
                room = p.cap * total - units[k] * pxf[t, k]
                usd = min(p.tranche * total, room)
                if usd > 0:
                    pend_buy.append((k, want, usd))
        # 새 종목 1차
        slots = p.max_names - len(held)

        def free(k):
            return k not in held and not (k in last_sell and t - last_sell[k] <= p.cooldown) and qualified(k)

        if random_days is None:
            for k in top_now:  # 거래대금 순
                if slots <= 0:
                    break
                if not free(k):
                    continue
                x0, x1 = inp.sig[t - 1, k] if t > 0 else np.nan, inp.sig[t, k]
                m0, m1 = inp.sma[s1][t - 1, k] if t > 0 else np.nan, inp.sma[s1][t, k]
                if np.isfinite(x0) and np.isfinite(m0) and np.isfinite(m1) and x1 < m1 and x0 >= m0:
                    pend_buy.append((k, 1, p.tranche * total))
                    entry_days.append(t)
                    slots -= 1
        elif rand_ct.get(t):
            pool = [k for k in top_now if free(k)]
            m = min(rand_ct[t], slots, len(pool))
            if m > 0:
                for k in rng.choice(pool, size=m, replace=False):
                    pend_buy.append((int(k), 1, p.tranche * total))
    # 끝: 전부 판 것으로 보고 세후 원화
    last = n - 1
    after = np.nan
    if p.tax:
        got = _sell_qqq(q_units * qprice(last), last) if q_units > 0 else 0.0
        for k in range(K):
            if units[k] > 0:
                got += _sell_stock(k, units[k] * pxf[last, k], last)
        got += cash
        paid_years = set(may.values())
        owed = sum(max(g - p.deduction_krw, 0.0) * p.rate for y, g in realized.items()
                   if y == years[last] or (y not in paid_years and y >= years[0]))
        taxes += owed
        after = got * f[last] - owed
    for k in list(held):  # 끝날 열린 보유는 끝날 가치로 기록(이유 "구간 끝")
        h = held[k]
        paid = sum(u for _, u in h["fills"])
        val = units[k] * pxf[last, k] if units[k] > 0 else sum(u * pxf[last, k] / pxf[pp, k] for pp, u in h["fills"])
        qe = sum(u * qprice(last) / qprice(pp) for pp, u in h["fills"])
        trades.append({"k": k, "fills": h["fills"], "tranches": h["tranches"], "entry_pos": h["entry_pos"], "exit_pos": last,
                       "reason": "구간 끝", "ret": val / paid - 1, "qqq_ret": qe / paid - 1, "excess": val / qe - 1})
    return Result(nav, stock_w, after, costs, costs_krw, taxes, fills, trades, entry_days, harvests)


def cagr(start: float, end: float, days: float) -> float:
    """연복리(달력일 365.25)."""
    return (end / start) ** (365.25 / days) - 1.0 if days > 0 and start > 0 and end > 0 else float("nan")


def max_drawdown(nav: np.ndarray) -> float:
    """최대 낙폭(음수 소수)."""
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def build_inputs(qqq_close: pd.Series, qqq_adj: pd.Series, close: pd.DataFrame, adj: pd.DataFrame, volume: pd.DataFrame,
                 members: dict, company: dict, fx: pd.Series, start, end, p: Params = Params()) -> tuple[Inputs, np.ndarray]:
    """전체 이력 → 한 구간 Inputs (지표는 구간 앞 데이터로 미리 계산, 구간 밖 뒤 데이터는 쓰지 않는다).

    qqq_close·qqq_adj: QQQ 분할 반영 종가·총수익 가격. close·adj·volume: 종목(열) DataFrame.
    members: {달 첫 거래일 Timestamp: 그날 구성 열 이름 집합}(그달 첫 거래일 키 — 없으면 그 전 가장 가까운 키).
    company: {열 이름: 회사 키}(같은 회사 여러 종류 주식 → 같은 키). fx: 원/달러(빈 날은 직전 값).
    출력: (Inputs, company 배열)."""
    end = pd.Timestamp(end)
    qc = qqq_close[qqq_close.index <= end].dropna()
    cal = qc.index
    cl = close.reindex(cal)
    ad = adj.reindex(cal)
    vo = volume.reindex(cal)
    smas = {m: sma(cl, m) for m in set(p.stages) | {200}}
    s200_prev = smas[200].shift(20)
    hi = prior_high(cl, p.high_lookback)
    dv = avg_dollar_volume(cl, vo, p.dv_window)
    q200 = sma(qc, 200)
    win = (cal >= pd.Timestamp(start)) & (cal <= end)
    dates = cal[win]
    cols = list(close.columns)
    ids: dict = {}
    comp = np.array([ids.setdefault(company.get(c, c), len(ids)) for c in cols])
    keys = sorted(members)
    top = {}
    for pos in month_first_positions(dates):
        d = dates[pos]
        ks = [k for k in keys if k <= d]
        mem = members[ks[-1]] if ks else set()
        mrow = np.array([c in mem for c in cols])
        top[pos] = top_by_dollar_volume(dv.loc[d].to_numpy(), mrow, comp, p.top_n)
    fxa = fx.reindex(cal).ffill().bfill()[win].to_numpy()
    inp = Inputs(dates=dates, qqq_sig=qc[win].to_numpy(), qqq_px=qqq_adj.reindex(cal)[win].to_numpy(),
                 qqq_sma200=q200[win].to_numpy(), sig=cl[win].to_numpy(), px=ad[win].to_numpy(),
                 sma={m: v[win].to_numpy() for m, v in smas.items()}, sma200_prev20=s200_prev[win].to_numpy(),
                 hi_prev=hi[win].to_numpy(), top=top, fx=fxa, names=cols)
    return inp, comp
