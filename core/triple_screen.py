"""3단 거르기 탐색 — 지표·국면·신호·장부 (순수 함수, docs/prereg/triple_screen.md).

네트워크·파일·현재 시각에 접근하지 않는다. 배열(거래일 × 회사)과 설정만 받는다. 음수 shift 없음.
1단 = 시장 국면(QQQ·XLY/XLP), 2단 = 주봉 MACD 히스토그램 상승 + 종목 200일선(B 회복장 매수는 100일선),
3단 = 일봉 RSI(14) 40 재돌파 또는 슬로우 스토캐스틱(14,3,3) %K 20 재돌파.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from factory import regimes as rg

UP, REC, DOWN, CRASH, UNKNOWN = "상승장", "회복장", "하락장", "폭락장", "판단불가"
IDLE = "__QQQ__"  # Market 마지막 열(대상 아님): 빈자리 돈을 담는 QQQ
KEEP_DIV = 0.85
DEDUCT = 2_500_000.0
TAX = 0.22
INITIAL_KRW = 100_000_000.0


# ── 지표 ────────────────────────────────────────────────────────────────


def sma(x: pd.Series, n: int) -> pd.Series:
    """단순 이동평균 (n개가 차야 값)."""
    return x.rolling(n, min_periods=n).mean()


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder RSI (ewm alpha=1/n, adjust=False) — core/indicators.py와 같은 식."""
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = 100 - 100 / (1 + gain / loss)
    return r.where(loss != 0, 100.0).where(close.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> tuple[pd.Series, pd.Series]:
    """MACD선·시그널선 (ewm span, adjust=False)."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return line, line.ewm(span=sig, adjust=False).mean()


def slow_stoch_k(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14, k: int = 3) -> pd.Series:
    """슬로우 스토캐스틱 %K = (100 × (종가 − 14일 최저) ÷ (14일 최고 − 14일 최저))의 3일 단순 평균."""
    hh, ll = high.rolling(n, min_periods=n).max(), low.rolling(n, min_periods=n).min()
    rng_ = (hh - ll).where(hh > ll)
    fast = 100 * (close - ll) / rng_
    return fast.rolling(k, min_periods=k).mean()


def cross_up(x: pd.Series, level: float) -> pd.Series:
    """전날 < level 이고 오늘 ≥ level."""
    return (x.shift(1) < level) & (x >= level)


def week_ends(days: pd.DatetimeIndex) -> np.ndarray:
    """거래일마다 그 주(월~금)의 마지막 거래일이면 True. 거래 달력은 미리 정해진 것이라 미래 가격 정보가 아니다."""
    wk = days.to_period("W-FRI")
    return np.r_[wk[1:] != wk[:-1], True]


def completed_week_index(days: pd.DatetimeIndex) -> np.ndarray:
    """거래일 i의 '직전 완료 주' 번호(0부터, 그날 종가로 끝난 주 포함). 아직 완료 주가 없으면 −1."""
    return np.cumsum(week_ends(days)) - 1


def weekly_macd(close: pd.Series, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """주봉 종가(주 마지막 거래일 종가, 빈 날은 그 주 앞선 종가) → 주별 MACD선·시그널선 배열(완료 주 번호 순)."""
    we = week_ends(days)
    c = close.reindex(days)
    wk = days.to_period("W-FRI")
    wc = c.groupby(wk).last()  # 그 주 마지막 유효 종가
    wc = wc.reindex(pd.PeriodIndex(wk[we]))
    first = wc.first_valid_index()
    line = pd.Series(np.nan, index=wc.index)
    sig = pd.Series(np.nan, index=wc.index)
    if first is not None:
        s = wc.loc[first:].ffill()
        l_, g_ = macd(s)
        line.loc[first:], sig.loc[first:] = l_.to_numpy(), g_.to_numpy()
    return line.to_numpy(), sig.to_numpy()


# ── 1단: 시장 국면 ───────────────────────────────────────────────────────


def classify(qqq: pd.Series, xly: pd.Series, xlp: pd.Series, mid: int = 100, use_ratio: bool = True) -> pd.Series:
    """국면(그날 종가까지): 폭락장(factory/regimes 정의) > 상승장(QQQ ≥ 200일선) > 회복장 > 하락장.
    회복장 = QQQ < 200일선 & QQQ > mid일선 & mid일선 > 20거래일 전 mid일선 & (XLY/XLP > 그 비율의 50일 평균, use_ratio일 때).
    200일선이 없으면 판단불가. 입력은 같은 날짜 인덱스(QQQ 거래일)."""
    q = qqq.astype(float)
    base = rg.classify(q)
    s200, sm = sma(q, 200), sma(q, mid)
    ratio = (xly / xlp).reindex(q.index)
    rec = (q < s200) & (q > sm) & (sm > sm.shift(20))
    if use_ratio:
        rec &= ratio > sma(ratio, 50)
    out = pd.Series(DOWN, index=q.index, dtype=object)
    out[rec] = REC
    out[q >= s200] = UP
    out[base == rg.CRASH] = CRASH
    out[base == rg.UNKNOWN] = UNKNOWN
    return out


def recovery_episodes(reg: pd.Series) -> dict:
    """회복장 연속 구간(신호) 수와 그 뒤 처음 다른 국면이 상승장인 비율. 끝까지 회복장인 마지막 구간은 '미정'."""
    key = (reg != reg.shift()).cumsum()
    seg = [(g.iloc[0], k) for k, g in reg.groupby(key)]
    n = up = open_ = 0
    for idx, (r, _) in enumerate(seg):
        if r != REC:
            continue
        n += 1
        if idx + 1 >= len(seg):
            open_ += 1
        elif seg[idx + 1][0] == UP:
            up += 1
    done = n - open_
    return {"episodes": n, "to_up": up, "undecided": open_, "to_up_pct": (up / done * 100.0) if done else None}


# ── 시장 묶음 ────────────────────────────────────────────────────────────


@dataclass
class Market:
    """거래일 × 회사 배열. 열 순서 = names."""
    days: pd.DatetimeIndex
    names: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    div: np.ndarray
    member: np.ndarray          # 그날 대상(전 달 말 나스닥100)
    qqq: pd.DataFrame           # open, close, div (days 인덱스)
    bil_ret: np.ndarray         # 그날 현금(BIL) 수익률 (종가 + 분배금 85%) ÷ 전날 종가 − 1, 없는 날 0
    fx: np.ndarray              # 원/달러
    reg: dict = field(default_factory=dict)  # (mid, use_ratio) → 국면 배열
    ind: dict = field(default_factory=dict)

    @property
    def close_ff(self) -> np.ndarray:
        if getattr(self, "_ff", None) is None or self._ff.shape != self.close.shape:
            self._ff = pd.DataFrame(self.close).ffill().to_numpy()
        return self._ff

    @property
    def last_idx(self) -> np.ndarray:
        has = ~np.isnan(self.close)
        n = len(self.days)
        return np.where(has.any(axis=0), n - 1 - np.argmax(has[::-1], axis=0), -1)


def build_indicators(mk: Market, mids: tuple[int, ...] = (100,)) -> None:
    """회사별 지표 → mk.ind (D×C 배열). 주봉 MACD는 완료 주 번호로 펼친다."""
    D, C = mk.close.shape
    cw = completed_week_index(mk.days)
    out = {k: np.full((D, C), np.nan) for k in ("s200", "rsi", "stk", "adv20", "wl", "ws", "wl_prev", "ws_prev",
                                                 "wh", "wh_prev")}
    for m in mids:
        out[f"s{m}"] = np.full((D, C), np.nan)
    for j in range(C):
        c = pd.Series(mk.close[:, j], index=mk.days)
        h = pd.Series(mk.high[:, j], index=mk.days)
        lo = pd.Series(mk.low[:, j], index=mk.days)
        v = pd.Series(mk.volume[:, j], index=mk.days)
        cv = c.dropna()
        if cv.empty:
            continue
        out["s200"][:, j] = sma(cv, 200).reindex(mk.days).to_numpy()
        for m in mids:
            out[f"s{m}"][:, j] = sma(cv, m).reindex(mk.days).to_numpy()
        out["rsi"][:, j] = rsi_wilder(cv).reindex(mk.days).to_numpy()
        ok = c.notna() & h.notna() & lo.notna()
        out["stk"][:, j] = slow_stoch_k(h[ok], lo[ok], c[ok]).reindex(mk.days).to_numpy()
        out["adv20"][:, j] = (cv * v.reindex(cv.index)).rolling(20, min_periods=20).mean().reindex(mk.days).to_numpy()
        wl, ws = weekly_macd(c, mk.days)
        valid = cw >= 0
        idx = np.where(valid, cw, 0)
        prev = np.where(cw >= 1, cw - 1, 0)
        out["wl"][:, j] = np.where(valid, wl[idx], np.nan)
        out["ws"][:, j] = np.where(valid, ws[idx], np.nan)
        out["wl_prev"][:, j] = np.where(cw >= 1, wl[prev], np.nan)
        out["ws_prev"][:, j] = np.where(cw >= 1, ws[prev], np.nan)
    out["wh"] = out["wl"] - out["ws"]
    out["wh_prev"] = out["wl_prev"] - out["ws_prev"]
    out["week_end"] = week_ends(mk.days)
    with np.errstate(invalid="ignore"):
        r = out["rsi"]
        out["trig_rsi"] = np.r_[np.zeros((1, C), bool), (r[:-1] < 40) & (r[1:] >= 40)]
        k = out["stk"]
        out["trig_stoch"] = np.r_[np.zeros((1, C), bool), (k[:-1] < 20) & (k[1:] >= 20)]
        out["tide"] = out["wh"] > out["wh_prev"]
        out["below_now"] = out["wl"] < out["ws"]
        out["cross_down"] = out["week_end"][:, None] & (out["wl"] < out["ws"]) & (out["wl_prev"] >= out["ws_prev"])
        out["above_now"] = out["wl"] >= out["ws"]
    mk.ind = out


# ── 판 ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Variant:
    name: str
    trigger: str = "rsi"          # rsi | stoch
    recovery: bool = False        # B: 회복장 새 매수
    rec_cap: float = 0.30         # 회복장 새 매수 한도(합계 비중)
    rec_max: int = 3              # 회복장 최대 종목
    mid: int = 100                # B의 100일선(국면·2단 완화·청산 ②)
    use_ratio: bool = True        # 회복장 XLY/XLP 조건
    slots: int = 10
    weight: float = 0.10
    idle_qqq: bool = True         # 빈자리 돈: 그날 국면이 상승장이면 QQQ, 아니면 BIL(현금) — 태그 전 수정 2026-10-21 07:19


A1 = Variant("A1")
A2 = Variant("A2", trigger="stoch")
B = Variant("B", recovery=True)
B0 = Variant("B0", recovery=True, use_ratio=False)


@dataclass(frozen=True)
class Cost:
    comm: float = 0.0007
    slip: float = 0.0005
    fx: float = 0.001
    tax: bool = True


FULL = Cost()
NO_TAX = Cost(tax=False)
ZERO = Cost(0.0, 0.0, 0.0, False)


def regime_of(mk: Market, v: Variant) -> np.ndarray:
    return mk.reg[(v.mid, v.use_ratio)]


# ── 장부 ────────────────────────────────────────────────────────────────


class Book:
    """원화 기준 공용 세금 규칙 장부. 현금(달러)은 매일 BIL 수익을 받는다."""

    def __init__(self, mk: Market, cost: Cost, harvest_days: dict[int, int], may_first: set[int], year_last: set[int]):
        self.mk, self.cost = mk, cost
        self.hold: dict[int, dict] = {}  # 열 → {q, basis(원), fill, signal, rec, below, weeks_after, ok_after}
        self.cash = 0.0
        self.realized: dict[int, float] = {}
        self.pending_tax: tuple | None = None
        self.taxes: dict[int, float] = {}
        self.flows: list = []
        self.trades: list = []  # 끝난 매매
        self.sells_usd: dict[int, float] = {}
        self.curve_pre: list = []
        self.curve_stock: list = []
        self.harvest_days, self.may_first, self.year_last = harvest_days, may_first, year_last
        self.ff = mk.close_ff
        self.idle: int | None = None  # 빈자리 돈을 담는 QQQ 열(A판). 매매 통계·회전율에서 뺀다

    # 가격
    def buy_unit(self, px):
        return px * (1 + self.cost.slip) * (1 + self.cost.comm)

    def sell_unit(self, px):
        return px * (1 - self.cost.slip) * (1 - self.cost.comm)

    def last_px(self, i, j):
        """그날까지 마지막 종가(앞으로 채운 배열, 미래 값 없음)."""
        return float(self.ff[i, j])

    def stock_value(self, i):
        return sum(h["q"] * self.last_px(i, j) for j, h in self.hold.items())

    def equity_usd(self, i):
        return self.cash + self.stock_value(i)

    # 매매
    def buy(self, i, j, usd, px, meta: dict):
        usd = min(usd, self.cash)
        if usd <= 0 or not np.isfinite(px) or px <= 0:
            return False
        q = usd / self.buy_unit(px)
        krw = usd * self.mk.fx[i]
        h = self.hold.get(j)
        if h is None:
            self.hold[j] = dict(q=q, basis=krw, fill=i, cost_krw=krw, **meta)
        else:
            h["q"] += q
            h["basis"] += krw
        self.cash -= usd
        return True

    def sell(self, i, j, px, why: str, q: float | None = None):
        h = self.hold[j]
        q = h["q"] if q is None else min(q, h["q"])
        usd = q * self.sell_unit(px)
        avg = h["basis"] / h["q"]
        gain = usd * self.mk.fx[i] - avg * q
        y = self.mk.days[i].year
        self.realized[y] = self.realized.get(y, 0.0) + gain
        if j != self.idle:
            self.sells_usd[y] = self.sells_usd.get(y, 0.0) + usd
        self.cash += usd
        h["pnl"] = h.get("pnl", 0.0) + gain
        h["q"] -= q
        h["basis"] -= avg * q
        if h["q"] <= 1e-12:
            del self.hold[j]
            if j == self.idle:
                return usd
            self.trades.append({"col": j, "fill": h["fill"], "exit": i, "why": why, "pnl_krw": h["pnl"] + h.get("divk", 0.0),
                                "cost_krw": h["cost_krw"], "rec": h.get("rec", False)})
        return usd

    # 하루 공통
    def start(self, i):
        self.cash = INITIAL_KRW / self.mk.fx[i] * (1 - self.cost.fx)
        self.flows.append((self.mk.days[i], -INITIAL_KRW))

    def accrue(self, i):
        """분배금(85%) → 현금, 현금 → BIL 하루 수익."""
        self.cash *= 1.0 + self.mk.bil_ret[i]
        for j, h in self.hold.items():
            dv = self.mk.div[i, j]
            if dv > 0:
                usd = h["q"] * dv * KEEP_DIV
                self.cash += usd
                h["divk"] = h.get("divk", 0.0) + usd * self.mk.fx[i]

    def delist(self, i):
        for j in [j for j in self.hold if self.mk.last_idx[j] < i]:
            self.sell(i, j, self.last_px(i, j), "가격 끝(마지막 종가)")

    def raise_cash(self, i, need):
        while self.cash < need - 1e-9 and self.hold:
            j = max(self.hold, key=lambda k: self.hold[k]["q"] * self.last_px(i, k))
            px = self.last_px(i, j)
            self.sell(i, j, px, "세금 마련", q=(need - self.cash) / self.sell_unit(px))

    def pay_tax(self, i):
        if self.pending_tax is None:
            return
        _, krw = self.pending_tax
        usd = krw / (self.mk.fx[i] * (1 - self.cost.fx))
        self.raise_cash(i, usd)
        self.cash -= usd
        self.pending_tax = None

    def close_year(self, i):
        y = self.mk.days[i].year
        tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.cost.tax else 0.0
        self.taxes[y] = tax
        if tax > 0:
            self.pending_tax = (y, tax)

    def harvest(self, i):
        """12월 공제 소진: 1주당 이익이 + 인 종목을 평가 이익 큰 순으로 팔고 같은 날 같은 수량 다시 산다(그날 종가)."""
        y = self.mk.days[i].year
        left = DEDUCT - self.realized.get(y, 0.0)
        fx = self.mk.fx[i]
        order = sorted(self.hold, key=lambda k: -(self.hold[k]["q"] * self.sell_unit(self.last_px(i, k)) * fx - self.hold[k]["basis"]))
        for j in order:
            if left <= 1.0:
                break
            h = self.hold[j]
            px = self.last_px(i, j)
            g = self.sell_unit(px) * fx - h["basis"] / h["q"]
            if g <= 0:
                continue
            q = min(left / g, h["q"])
            keep = {k: v for k, v in h.items() if k not in ("q", "basis")}
            whole = q >= h["q"] - 1e-12
            self.sell(i, j, px, "공제 소진", q=q)
            if whole and j != self.idle:  # 전량을 팔았다 사도 같은 보유로 본다(진입일·조건 유지) — 끝난 매매 기록은 되돌린다
                t = self.trades.pop()
                keep["pnl"] = t["pnl_krw"] - keep.get("divk", 0.0)
            usd = min(q * self.buy_unit(px), max(self.cash, 0.0))
            qq = usd / self.buy_unit(px)
            if j in self.hold:
                self.hold[j]["q"] += qq
                self.hold[j]["basis"] += usd * fx
            else:
                self.hold[j] = dict(q=qq, basis=usd * fx, **{k: v for k, v in keep.items()})
            self.cash -= usd
            left -= q * g

    def snapshot(self, i):
        self.curve_pre.append(self.equity_usd(i) * self.mk.fx[i])
        self.curve_stock.append(sum(h["q"] * self.last_px(i, j) for j, h in self.hold.items() if j != self.idle))

    def finish(self, i):
        for j in list(self.hold):
            self.sell(i, j, self.last_px(i, j), "끝 청산")
        y = self.mk.days[i].year
        tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.cost.tax else 0.0
        self.taxes[y] = tax
        final = self.cash * self.mk.fx[i] * (1 - self.cost.fx) - tax
        if self.pending_tax is not None:
            final -= self.pending_tax[1]
            self.pending_tax = None
        self.flows.append((self.mk.days[i], final))
        return final


@dataclass
class Calendar:
    harvest_days: dict
    may_first: set
    year_last: set


def calendar(days: pd.DatetimeIndex, deadline_fn) -> Calendar:
    """12월 공제 소진일(deadline_fn(days, y)), 5월 첫 거래일, 해 마지막 거래일 위치."""
    s = pd.Series(np.arange(len(days)), index=days)
    di = {d: i for i, d in enumerate(days)}
    years = sorted(set(days.year))
    hv = {int(di[deadline_fn(days, y)]): y for y in years if (days.year == y).sum() >= 3 and days[days.year == y][-1].month == 12}
    mf = set(s.groupby([days.year, days.month]).min())
    may = {i for i in mf if days[i].month == 5 and i != 0}
    yl = set(s.groupby(days.year).max()) - {len(days) - 1}
    return Calendar(hv, may, yl)


def run_strategy(mk: Market, v: Variant, cost: Cost, cal: Calendar, *, rng: np.random.Generator | None = None,
                 buys_per_day: dict | None = None, log: bool = False) -> dict:
    """판 하나. rng가 있으면 무작위 판: 실제 판이 산 날(buys_per_day: 신호일 → 개수)에 같은 개수를
    그날 같은 국면·2단 통과 종목 중 무작위로 고르고, 같은 청산 규칙을 쓴다."""
    ind = mk.ind
    reg = regime_of(mk, v)
    D, C = mk.close.shape
    b = Book(mk, cost, cal.harvest_days, cal.may_first, cal.year_last)
    qj = mk.names.index(IDLE) if (v.idle_qqq and IDLE in mk.names) else None
    b.idle = qj
    trig = ind["trig_rsi"] if v.trigger == "rsi" else ind["trig_stoch"]
    smid = ind[f"s{v.mid}"]
    s200 = ind["s200"]
    valid = ~np.isnan(mk.close)
    with np.errstate(invalid="ignore"):
        st2_200 = mk.member & valid & ind["tide"] & (mk.close > s200)
        st2_mid = mk.member & valid & ind["tide"] & (mk.close > smid)
    pend_buy: list = []
    pend_sell: dict = {}
    buys_log: dict = {}
    stats = {"signals": 0, "no_slot": 0, "no_fill": 0, "rec_signals": 0, "rec_buys": 0}
    for i in range(D):
        if i == 0:
            b.start(i)
        else:
            b.accrue(i)
            b.delist(i)
            if i in cal.may_first:
                b.pay_tax(i)
            # 시가 체결: 매도 먼저, 그다음 매수
            for j, why in list(pend_sell.items()):
                if j not in b.hold:
                    pend_sell.pop(j)
                    continue
                px = mk.open[i, j]
                if np.isfinite(px) and px > 0:
                    b.sell(i, j, px, why)
                    pend_sell.pop(j)
            qpx = mk.open[i, qj] if qj is not None else np.nan
            need = sum(usd for _, usd, _ in pend_buy)
            if qj is not None and qj in b.hold and b.cash < need and np.isfinite(qpx) and qpx > 0:
                b.sell(i, qj, qpx, "종목 매수 자금", q=(need - b.cash) / b.sell_unit(qpx))  # 모자란 만큼 QQQ를 판다
            for (j, usd, meta) in pend_buy:
                px = mk.open[i, j]
                if j in b.hold:
                    continue
                if not (np.isfinite(px) and px > 0):
                    stats["no_fill"] += 1
                    continue
                b.buy(i, j, usd, px, meta)
            pend_buy = []
            if qj is not None and np.isfinite(qpx) and qpx > 0:  # 빈자리 돈: 전날 국면이 상승장이면 QQQ, 아니면 BIL
                if reg[i - 1] == UP:
                    if b.cash > 1e-6:
                        b.buy(i, qj, b.cash, qpx, {})
                elif qj in b.hold:
                    b.sell(i, qj, qpx, "국면(빈자리 BIL)")
            if i in cal.harvest_days:
                b.harvest(i)
        # 종가 판정 (마지막 날은 다음 날이 없으므로 판정하지 않음)
        if i < D - 1:
            r = reg[i]
            # 보유 종목 상태 갱신·청산 판정
            for j, h in b.hold.items():
                if j in pend_sell or j == qj:
                    continue
                if r == UP:
                    h["rec"] = False  # 상승장이 되면 그 뒤로는 200일선
                why = None
                if ind["week_end"][i] and i >= h["fill"]:
                    h["weeks_after"] = h.get("weeks_after", 0) + 1
                    if ind["above_now"][i, j]:
                        h["ok_after"] = True
                    if ind["cross_down"][i, j]:
                        why = "① 주봉 하향 교차"
                    elif h.get("below") and not h.get("ok_after") and h["weeks_after"] >= 4:
                        why = "①-보완 4주 안 회복 없음"
                line = smid[i, j] if h.get("rec") else s200[i, j]
                c = mk.close[i, j]
                if why is None and np.isfinite(c) and np.isfinite(line) and c < line:
                    why = f"② 종가 < {v.mid if h.get('rec') else 200}일선"
                if why is None and ((not v.recovery and r != UP) or (v.recovery and r in (DOWN, CRASH, UNKNOWN))):
                    why = "③ 국면"
                if why:
                    pend_sell[j] = why
            # 새 매수
            if r == UP or (v.recovery and r == REC):
                cand_mask = (st2_200[i] if r == UP else st2_mid[i]) & trig[i]
                held = set(b.hold) - set(pend_sell) - {qj}
                eq = b.equity_usd(i)
                cands = [j for j in np.flatnonzero(cand_mask) if j not in held]
                if rng is None:
                    cands.sort(key=lambda j: (-np.nan_to_num(ind["adv20"][i, j], nan=-1.0), mk.names[j]))
                    stats["signals"] += len(cands)
                    if r == REC:
                        stats["rec_signals"] += len(cands)
                else:
                    k = (buys_per_day or {}).get(i, 0)
                    pool = [j for j in np.flatnonzero((st2_200[i] if r == UP else st2_mid[i])) if j not in held]
                    cands = list(rng.choice(pool, size=min(k, len(pool)), replace=False)) if k and pool else []
                free = v.slots - len(held)
                stock_now = sum(b.hold[j]["q"] * b.last_px(i, j) for j in held)
                expo = stock_now / eq if eq > 0 else 1.0
                n_held = len(held)
                chosen = []
                for j in cands:
                    if free <= 0:
                        break
                    if r == REC and (expo >= v.rec_cap - 1e-12 or n_held >= v.rec_max):
                        break
                    chosen.append(j)
                    free -= 1
                    n_held += 1
                    expo += v.weight
                if rng is None:
                    stats["no_slot"] += len(cands) - len(chosen)
                for j in chosen:
                    meta = {"signal": i, "rec": r == REC, "below": bool(ind["below_now"][i, j]), "weeks_after": 0,
                            "ok_after": False}
                    pend_buy.append((int(j), eq * v.weight, meta))
                if chosen:
                    buys_log[i] = len(chosen)
                    if r == REC:
                        stats["rec_buys"] += len(chosen)
        if i in cal.year_last:
            b.close_year(i)
        b.snapshot(i)
    final = b.finish(D - 1)
    return {"book": b, "final": final, "stats": stats, "buys_per_day": buys_log}


def run_timing(mk: Market, weights: dict[str, float], reg: np.ndarray, cost: Cost, cal: Calendar) -> dict:
    """1단만 비교: 그날 국면 → 다음 거래일 시가에 QQQ 비중을 weights[국면]으로 맞춘다(국면이 바뀐 날만), 나머지 현금(BIL)."""
    q = mk.qqq
    mk2 = _qqq_market(mk)
    b = Book(mk2, cost, cal.harvest_days, cal.may_first, cal.year_last)
    target = None
    for i in range(len(mk.days)):
        if i == 0:
            b.start(i)
        else:
            b.accrue(i)
            if i in cal.may_first:
                b.pay_tax(i)
            if target is not None:
                px = float(q["open"].iloc[i])
                cur = b.hold[0]["q"] * px if 0 in b.hold else 0.0
                eq = b.cash + cur
                want = eq * target
                if want < cur - 1e-6:
                    b.sell(i, 0, px, "국면", q=(cur - want) / px)
                elif want > cur + 1e-6:
                    b.buy(i, 0, want - cur, px, {})
                target = None
            if i in cal.harvest_days:
                b.harvest(i)
        if i < len(mk.days) - 1 and (i == 0 or reg[i] != reg[i - 1]):
            target = weights.get(reg[i], 0.0)
        if i in cal.year_last:
            b.close_year(i)
        b.snapshot(i)
    return {"book": b, "final": b.finish(len(mk.days) - 1)}


def run_qqq_hold(mk: Market, cost: Cost, cal: Calendar) -> dict:
    """QQQ 보유: 첫날 종가에 전부 사서 끝까지(분배금 85%는 현금 → BIL)."""
    mk2 = _qqq_market(mk)
    b = Book(mk2, cost, cal.harvest_days, cal.may_first, cal.year_last)
    for i in range(len(mk.days)):
        if i == 0:
            b.start(i)
            b.buy(i, 0, b.cash, float(mk.qqq["close"].iloc[0]), {})
        else:
            b.accrue(i)
            if i in cal.may_first:
                b.pay_tax(i)
            if i in cal.harvest_days:
                b.harvest(i)
        if i in cal.year_last:
            b.close_year(i)
        b.snapshot(i)
    return {"book": b, "final": b.finish(len(mk.days) - 1)}


def _qqq_market(mk: Market) -> Market:
    q = mk.qqq
    one = lambda s: q[s].to_numpy(float)[:, None]  # noqa: E731
    return Market(mk.days, ["QQQ"], one("open"), one("close"), one("close"), one("close"), np.ones((len(mk.days), 1)),
                  q["div"].fillna(0.0).to_numpy(float)[:, None], np.ones((len(mk.days), 1), bool), q, mk.bil_ret, mk.fx)


def run_equal_weight(mk: Market, cost: Cost, cal: Calendar) -> dict:
    """시점별 나스닥100 동일 비중: 매달 첫 거래일 시가에 그날 대상(시가 있음) 전체를 같은 비중으로 맞춘다."""
    D = len(mk.days)
    b = Book(mk, cost, cal.harvest_days, cal.may_first, cal.year_last)
    m = mk.days.to_period("M")
    first = np.r_[True, m[1:] != m[:-1]]
    for i in range(D):
        if i == 0:
            b.start(i)
        else:
            b.accrue(i)
            b.delist(i)
            if i in cal.may_first:
                b.pay_tax(i)
            if first[i]:
                px = mk.open[i]
                cols = [j for j in np.flatnonzero(mk.member[i]) if np.isfinite(px[j]) and px[j] > 0]
                for j in [j for j in b.hold if j not in cols]:
                    p = px[j] if np.isfinite(px[j]) and px[j] > 0 else b.last_px(i, j)
                    b.sell(i, j, p, "재조정(명단 밖)")
                eq = b.cash + sum(h["q"] * px[j] for j, h in b.hold.items())
                tgt = eq / len(cols) * (1 - cost.comm - cost.slip)
                for j in list(b.hold):
                    vv = b.hold[j]["q"] * px[j]
                    if vv > tgt:
                        b.sell(i, j, px[j], "재조정", q=(vv - tgt) / px[j])
                need = {j: tgt - (b.hold[j]["q"] * px[j] if j in b.hold else 0.0) for j in cols}
                need = {j: x for j, x in need.items() if x > 0}
                tot, cash = sum(need.values()), b.cash
                for j, x in need.items():
                    b.buy(i, j, cash * x / tot, px[j], {})
            if i in cal.harvest_days:
                b.harvest(i)
        if i in cal.year_last:
            b.close_year(i)
        b.snapshot(i)
    return {"book": b, "final": b.finish(D - 1)}


# ── 지표 계산 ───────────────────────────────────────────────────────────


def xirr(flows: list) -> float:
    """원화 현금흐름 내부수익률(%/년), 이분법."""
    t0 = flows[0][0]
    ts = np.array([(d - t0).total_seconds() / 86400.0 / 365.25 for d, _ in flows])
    cf = np.array([v for _, v in flows])
    lo, hi = -0.99, 5.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if float((cf / (1.0 + mid) ** ts).sum()) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 * 100.0


def mdd(c: pd.Series) -> float:
    return float((c / c.cummax() - 1.0).min() * 100.0)


def regime_table(curve: pd.Series, reg: pd.Series, min_days: int = 60) -> dict:
    """국면별: 그날 국면에 그날 → 다음 날 수익. 일수·누적·연환산·구간 안 최대 낙폭(그 국면 연속 구간마다, 가장 깊은 값)."""
    r = curve.pct_change().iloc[1:]
    g = reg.reindex(curve.index).iloc[:-1]
    g.index = r.index
    out = {}
    key = (reg != reg.shift()).cumsum().reindex(curve.index)
    for k in (UP, REC, DOWN, CRASH):
        m = g == k
        n = int(m.sum())
        cum = float(np.prod(1 + r[m]) - 1) * 100 if n else 0.0
        worst = 0.0
        for _, seg in curve[reg.reindex(curve.index) == k].groupby(key[reg.reindex(curve.index) == k]):
            # 구간 시작 전날 값부터(그 구간 첫날 수익 포함)
            i0 = curve.index.get_loc(seg.index[0])
            s = curve.iloc[max(i0 - 1, 0): curve.index.get_loc(seg.index[-1]) + 2]
            worst = min(worst, mdd(s))
        out[k] = {"days": n, "cum_pct": cum, "ann_pct": ((1 + cum / 100) ** (252 / n) - 1) * 100 if n else None,
                  "mdd_pct": worst, "inconclusive": n < min_days}
    return out


def down_crash_mdd(curve: pd.Series, reg: pd.Series) -> float:
    """하락장·폭락장이 이어진 구간마다 구간 안 최대 낙폭(구간 전날 값부터), 가장 깊은 값."""
    r = reg.reindex(curve.index)
    bad = r.isin([DOWN, CRASH])
    key = (bad != bad.shift()).cumsum()
    worst = 0.0
    for _, seg in curve[bad].groupby(key[bad]):
        i0 = curve.index.get_loc(seg.index[0])
        s = curve.iloc[max(i0 - 1, 0): curve.index.get_loc(seg.index[-1]) + 2]
        worst = min(worst, mdd(s))
    return worst
