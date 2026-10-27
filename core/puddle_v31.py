"""상승장 대형주 웅덩이 v3.1 — 가상 1억 장부 순수 함수 (docs/prereg/puddle_v31_forward.md, 태그 puddle-v31-prereg).

네트워크·파일·DB·현재 시각에 접근하지 않는다. 음수 shift를 쓰지 않는다(그날까지의 값만, 체결은 다음 거래일 종가).
장부 상태는 JSON으로 저장할 수 있는 dict다(engine/puddle_v31.py가 DB에 저장해 다음 날 이어 쓴다).

하루 순서(장부마다): 5월 첫 거래일 세금 → 어제 신호 체결(매도 → 매수) → 12월 공제 소진 → 오늘 종가 가치 → 오늘 신호(내일 체결).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REASONS = {"max_hold": "보유 250거래일", "high": "직전 120거래일 최고가 돌파", "rank": "월초 상위 10 밖", "gone": "가격 끝"}
COMPANY = {"GOOG": "GOOGL", "FOX": "FOXA", "NWS": "NWSA", "LBTYK": "LBTYA", "DISCK": "DISCA"}  # 같은 회사 여러 종류 주식


@dataclass(frozen=True)
class Params:
    tranche: float = 0.05
    cap: float = 0.10
    max_names: int = 2
    min_hold: int = 60
    max_hold: int = 250
    cooldown: int = 20
    above_days: int = 20  # 1차: 직전 N거래일 종가 ≥ 60일선
    high_lookback: int = 120
    top_n: int = 10
    dv_window: int = 20
    filing_window: int = 10  # 실적 필터: 신호 날 포함 N거래일
    cost: float = 0.001
    deduction_krw: float = 2_500_000.0
    rate: float = 0.22
    start_krw: float = 100_000_000.0
    n_random: int = 200
    seed: int = 20261023


# ── 지표 ──────────────────────────────────────────────────────────────────────


def indicators(close: pd.DataFrame, volume: pd.DataFrame, p: Params = Params()) -> dict[str, pd.DataFrame]:
    """종목 종가·거래량(열 = 티커) → 지표 DataFrame 묶음. 모두 그날까지의 값만(창이 다 차야 값).

    sma60·sma120·sma200, sma200_prev20(20거래일 전 200일선), hi_prev(직전 120거래일 최고 종가, 그날 제외),
    above_prev(직전 20거래일 모두 종가 ≥ 60일선이면 1.0, 아니면 0.0, 모르면 NaN), dv(20일 평균 거래대금)."""
    sma = {n: close.rolling(n, min_periods=n).mean() for n in (60, 120, 200)}
    ok = (close >= sma[60]).astype(float).where(sma[60].notna())
    above_prev = ok.shift(1).rolling(p.above_days, min_periods=p.above_days).min()
    return {"sma60": sma[60], "sma120": sma[120], "sma200": sma[200], "sma200_prev20": sma[200].shift(20),
            "hi_prev": close.shift(1).rolling(p.high_lookback, min_periods=p.high_lookback).max(),
            "above_prev": above_prev, "dv": (close * volume).rolling(p.dv_window, min_periods=p.dv_window).mean()}


def top_by_dollar_volume(dv: dict[str, float], members: list[str], n: int = 10) -> list[str]:
    """그날 거래대금 → 상위 n 티커(같은 회사는 거래대금 큰 티커 하나)."""
    best: dict[str, tuple[float, str]] = {}
    for t in members:
        v = dv.get(t)
        if v is None or not np.isfinite(v):
            continue
        c = COMPANY.get(t, t)
        if c not in best or v > best[c][0]:
            best[c] = (float(v), t)
    return [t for _, t in sorted(best.values(), key=lambda x: (-x[0], x[1]))[:n]]


def top_by_market_cap(closes: dict[str, float], shares: dict[str, float | None], dv: dict[str, float], members: list[str],
                      n: int = 10, whole: set[str] | frozenset = frozenset()) -> tuple[list[str], dict[str, float], list[str]]:
    """웅덩이 v3.2·반등 확인 v2 대상(docs/prereg/puddle_v32_forward.md 2-2): 그날 종가 × 발행 주식 수 → 시가총액 상위 n.

    입력: closes·dv(그날 종가·20일 평균 거래대금), shares(티커 → 주식 수, 못 받으면 None), members(구성 종목),
          whole(주식 수가 회사 전체 보통주인 티커 — EDGAR 값).
    같은 회사(COMPANY)는 시가총액을 합쳐 한 회사로, 거래 티커는 거래대금 큰 쪽(모르면 알파벳 앞). 한 종류라도 주식 수가 없으면 회사 전체 제외.
    whole 티커가 있는 회사는 그 주식 수 × 거래 티커 종가 하나로 센다(이중 계산 방지).
    출력: (상위 n 거래 티커 — 시가총액 큰 순·같으면 티커 순, {거래 티커: 회사 시가총액(달러)}, 주식 수 없는 티커 목록)."""
    groups: dict[str, list[str]] = {}
    for t in dict.fromkeys(members):
        c = closes.get(t)
        if c is None or not np.isfinite(c):
            continue
        groups.setdefault(COMPANY.get(t, t), []).append(t)
    caps: dict[str, float] = {}
    missing: list[str] = []
    for tks in groups.values():
        lack = [t for t in tks if not (shares.get(t) is not None and np.isfinite(shares[t]) and shares[t] > 0)]
        if lack:
            missing += lack
            continue

        def dvv(t):
            v = dv.get(t)
            return float(v) if v is not None and np.isfinite(v) else -1.0

        trade = sorted(tks, key=lambda t: (-dvv(t), t))[0]
        w = sorted(t for t in tks if t in whole)
        caps[trade] = float(shares[w[0]]) * float(closes[trade]) if w else sum(float(shares[t]) * float(closes[t]) for t in tks)
    top = [t for t, _ in sorted(caps.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]
    return top, {t: caps[t] for t in top}, sorted(missing)


def has_earnings_filing(filings: list[dict], window_start, signal_date) -> bool:
    """접수 목록 [{form, filingDate, items}] 중 10-Q·10-K·8-K(Item 2.02)의 접수일이 [window_start, signal_date] 안에 있는가."""
    a, b = pd.Timestamp(window_start).normalize(), pd.Timestamp(signal_date).normalize()
    for f in filings:
        form = str(f.get("form") or "").strip()
        items = str(f.get("items") or "")
        hit = form in ("10-Q", "10-K") or (form == "8-K" and "2.02" in [x.strip() for x in items.split(",")])
        if hit:
            d = pd.Timestamp(f.get("filingDate")).normalize()
            if a <= d <= b:
                return True
    return False


# ── 하루 입력 ─────────────────────────────────────────────────────────────────


@dataclass
class Day:
    """하루 입력. closes·ind는 그날 값(티커 → 값), top10은 그날 쓰는 그달 상위 10."""

    date: pd.Timestamp
    idx: int  # 시작일 0부터 센 거래일 번호
    fx: float
    qqq: float
    qqq_sma200: float
    qqq_ret20: float  # QQQ 20거래일 수익률(t−20 → t)
    closes: dict
    ind: dict  # 티커 → {sma60, sma120, sma200, sma200_prev20, hi_prev, above_prev}
    top10: list
    may_first: bool = False
    harvest: bool = False


# ── 장부 ─────────────────────────────────────────────────────────────────────


def new_ledger(name: str, day: Day, p: Params = Params(), *, earnings_filter: bool, stocks: bool = True,
               random: bool = False) -> dict:
    """시작일 종가에 QQQ 100%(비용 없음). random=True면 무작위 경로(1차는 random_n이 있는 날만)."""
    usd = p.start_krw / day.fx
    return {"name": name, "filter": earnings_filter, "stocks": stocks, "random": random, "q_units": usd / day.qqq, "q_basis": p.start_krw,
            "pos": {}, "realized": {}, "paid": [], "last_sell": {}, "pending": [], "costs_usd": 0.0, "costs_krw": 0.0,
            "taxes_krw": 0.0, "harvests": 0, "peak": p.start_krw, "mdd": 0.0, "trades": []}


def _sell_q(L, value, d: Day, p):
    q = min(value / d.qqq, L["q_units"])
    if q <= 0:
        return 0.0
    b = L["q_basis"] * q / L["q_units"]
    gross = q * d.qqq
    L["costs_usd"] += gross * p.cost
    L["costs_krw"] += gross * p.cost * d.fx
    y = str(d.date.year)
    L["realized"][y] = L["realized"].get(y, 0.0) + gross * (1 - p.cost) * d.fx - b
    L["q_units"] -= q
    L["q_basis"] -= b
    return gross * (1 - p.cost)


def _buy_q(L, usd, d: Day, p):
    if usd <= 0:
        return
    spend = usd / (1 + p.cost)
    L["costs_usd"] += usd - spend
    L["costs_krw"] += (usd - spend) * d.fx
    L["q_units"] += spend / d.qqq
    L["q_basis"] += usd * d.fx


def _price(L, tk, d: Day):
    px = d.closes.get(tk)
    if px is not None and np.isfinite(px):
        return float(px)
    return L["pos"][tk].get("last_close")


def _sell_s(L, tk, value, d: Day, p):
    pos = L["pos"][tk]
    px = _price(L, tk, d)
    q = min(value / px, pos["units"])
    if q <= 0:
        return 0.0
    b = pos["basis"] * q / pos["units"]
    gross = q * px
    L["costs_usd"] += gross * p.cost
    L["costs_krw"] += gross * p.cost * d.fx
    y = str(d.date.year)
    L["realized"][y] = L["realized"].get(y, 0.0) + gross * (1 - p.cost) * d.fx - b
    pos["units"] -= q
    pos["basis"] -= b
    return gross * (1 - p.cost)


def _buy_s(L, tk, usd, d: Day, p):
    pos = L["pos"][tk]
    spend = usd / (1 + p.cost)
    L["costs_usd"] += usd - spend
    L["costs_krw"] += (usd - spend) * d.fx
    pos["units"] += spend / d.closes[tk]
    pos["basis"] += usd * d.fx


def nav_usd(L, d: Day) -> float:
    return L["q_units"] * d.qqq + sum(pos["units"] * (_price(L, tk, d) or 0.0) for tk, pos in L["pos"].items())


def weights(L, d: Day) -> dict:
    """종목 비중(티커 → 비중)."""
    tot = nav_usd(L, d)
    return {tk: pos["units"] * (_price(L, tk, d) or 0.0) / tot for tk, pos in L["pos"].items()} if tot > 0 else {}


def after_tax_krw(L, d: Day, p: Params = Params()) -> float:
    """그날 종가로 전부 판다고 본 세후 원화(매도 비용·그해 세금·아직 안 낸 전년 세금 차감)."""
    proceeds_q = L["q_units"] * d.qqq * (1 - p.cost)
    gain = proceeds_q * d.fx - L["q_basis"]
    proceeds = proceeds_q
    for tk, pos in L["pos"].items():
        v = pos["units"] * (_price(L, tk, d) or 0.0) * (1 - p.cost)
        proceeds += v
        gain += v * d.fx - pos["basis"]
    y = d.date.year
    tax = max(L["realized"].get(str(y), 0.0) + gain - p.deduction_krw, 0.0) * p.rate
    prev = str(y - 1)
    if prev in L["realized"] and prev not in L["paid"]:
        tax += max(L["realized"][prev] - p.deduction_krw, 0.0) * p.rate
    return proceeds * d.fx - tax


def _qualified(ind: dict | None, close) -> bool:
    if not ind or close is None or not np.isfinite(close):
        return False
    m200, prev = ind.get("sma200"), ind.get("sma200_prev20")
    return all(x is not None and np.isfinite(x) for x in (m200, prev)) and close > m200 and m200 > prev


def step(L: dict, d: Day, p: Params = Params(), *, filing=None, random_n: int = 0, rng=None) -> dict:
    """장부 하루 진행. filing(티커) → True(접수 있음 = 건너뜀)·False(없음)·None(조회 실패). 실적 필터가 없는 장부는 부르지 않는다.

    random_n > 0이면 무작위 경로: 1차는 상위 10의 자격·지수 조건·필터 통과 후보에서 rng로 random_n개까지(20일선 하향 조건 없음).
    출력: {fills, sells, skipped(실적 필터로 건너뛴 신호), failed(조회 실패 티커), entries(1차 주문 수), signals_1(1차 신호 정보)}."""
    ev = {"fills": [], "sells": [], "skipped": [], "failed": [], "entries": 0, "signals_1": []}
    y = d.date.year
    # 1) 5월 첫 거래일: 전년 세금
    if d.may_first and str(y - 1) not in L["paid"]:
        owed = max(L["realized"].get(str(y - 1), 0.0) - p.deduction_krw, 0.0) * p.rate
        if owed > 0:
            need = owed / d.fx
            got = _sell_q(L, need / (1 - p.cost), d, p)
            for tk in list(L["pos"]):
                if got >= need - 1e-9:
                    break
                got += _sell_s(L, tk, (need - got) / (1 - p.cost), d, p)
            L["taxes_krw"] += owed
        L["paid"].append(str(y - 1))
    # 2) 어제 신호 체결: 가격 끝 → 매도 → 매수
    for tk, pos in L["pos"].items():
        px = d.closes.get(tk)
        if (px is None or not np.isfinite(px)) and not any(o["tk"] == tk and o["type"] == "sell" for o in L["pending"]):
            L["pending"].append({"type": "sell", "tk": tk, "reason": "gone", "signal_date": str(d.date.date())})
    for o in [o for o in L["pending"] if o["type"] == "sell"]:
        tk = o["tk"]
        if tk not in L["pos"]:
            continue
        pos = L["pos"][tk]
        got = _sell_s(L, tk, pos["units"] * (_price(L, tk, d) or 0.0) + 1.0, d, p)
        _buy_q(L, got, d, p)
        paid = sum(u for _, u in pos["fills"])
        qqq_eq = sum(u * d.qqq / q0 for (_, u), q0 in zip(pos["fills"], pos["fill_qqq"]))
        tr = {"tk": tk, "entry_date": pos["entry_date"], "exit_date": str(d.date.date()), "reason": REASONS[o["reason"]],
              "tranches": pos["tranches"], "ret": got / paid - 1 if paid else None,
              "excess": got / qqq_eq - 1 if qqq_eq else None, "stock_only_puddle": pos.get("stock_only_puddle")}
        L["trades"].append(tr)
        ev["sells"].append(tr)
        del L["pos"][tk]
        L["last_sell"][tk] = d.idx
    for o in [o for o in L["pending"] if o["type"] == "buy"]:
        tk = o["tk"]
        px = d.closes.get(tk)
        if px is None or not np.isfinite(px):
            continue
        got = _sell_q(L, o["usd"], d, p)
        if got <= 0:
            continue
        if tk not in L["pos"]:
            L["pos"][tk] = {"units": 0.0, "basis": 0.0, "entry_idx": d.idx, "entry_date": str(d.date.date()), "tranches": 0,
                            "fills": [], "fill_qqq": [], "stock_only_puddle": o.get("stock_only_puddle")}
        pos = L["pos"][tk]
        _buy_s(L, tk, got, d, p)
        pos["tranches"] = o["tranche"]
        pos["fills"].append([str(d.date.date()), got])
        pos["fill_qqq"].append(d.qqq)
        ev["fills"].append({"tk": tk, "tranche": o["tranche"], "usd": got, "date": str(d.date.date())})
    L["pending"] = []
    # 3) 12월 공제 소진
    if d.harvest:
        def unreal(a):
            if a == "__Q__":
                return L["q_units"] * d.qqq * d.fx - L["q_basis"]
            pos = L["pos"][a]
            return pos["units"] * (_price(L, a, d) or 0.0) * d.fx - pos["basis"]

        for a in sorted(["__Q__"] + list(L["pos"]), key=lambda a: -unreal(a)):
            left = p.deduction_krw - L["realized"].get(str(y), 0.0)
            if left <= 1.0:
                break
            if a == "__Q__":
                if L["q_units"] <= 0:
                    continue
                g = d.qqq * (1 - p.cost) * d.fx - L["q_basis"] / L["q_units"]
                if g <= 0:
                    continue
                _buy_q(L, _sell_q(L, min(L["q_units"], left / g) * d.qqq, d, p), d, p)
            else:
                pos = L["pos"][a]
                px = _price(L, a, d)
                if not px or pos["units"] <= 0:
                    continue
                g = px * (1 - p.cost) * d.fx - pos["basis"] / pos["units"]
                if g <= 0:
                    continue
                got = _sell_s(L, a, min(pos["units"], left / g) * px, d, p)
                spend = got / (1 + p.cost)
                L["costs_usd"] += got - spend
                L["costs_krw"] += (got - spend) * d.fx
                pos["units"] += spend / px
                pos["basis"] += got * d.fx
            L["harvests"] += 1
    # 4) 가치·낙폭
    for tk, pos in L["pos"].items():
        px = d.closes.get(tk)
        if px is not None and np.isfinite(px):
            pos["last_close"] = float(px)
    val = after_tax_krw(L, d, p)
    L["peak"] = max(L["peak"], val)
    L["mdd"] = min(L["mdd"], val / L["peak"] - 1)
    L["last_value"] = val
    if not L["stocks"]:
        return ev
    # 5) 오늘 신호
    selling = set()
    for tk, pos in L["pos"].items():
        x, ind = d.closes.get(tk), d.ind.get(tk) or {}
        if x is None or not np.isfinite(x):
            continue
        held = d.idx - pos["entry_idx"]
        reason = None
        if held >= p.max_hold:
            reason = "max_hold"
        elif held >= p.min_hold:
            hi = ind.get("hi_prev")
            if hi is not None and np.isfinite(hi) and x > hi:
                reason = "high"
            elif tk not in d.top10:
                reason = "rank"
        if reason:
            L["pending"].append({"type": "sell", "tk": tk, "reason": reason, "signal_date": str(d.date.date())})
            selling.add(tk)
    if not (np.isfinite(d.qqq_sma200) and d.qqq > d.qqq_sma200):
        return ev
    total = nav_usd(L, d)

    def filter_ok(tk, tranche):
        if not L["filter"]:
            return True
        r = filing(tk)
        if r is None:
            ev["failed"].append(tk)
            return False
        if r:
            ev["skipped"].append({"tk": tk, "tranche": tranche})
            return False
        return True

    # 2차
    for tk, pos in L["pos"].items():
        if tk in selling or pos["tranches"] != 1:
            continue
        x, ind = d.closes.get(tk), d.ind.get(tk) or {}
        if not _qualified(ind, x):
            continue
        m120 = ind.get("sma120")
        if m120 is None or not np.isfinite(m120) or not (x < m120 and x > ind["sma200"]):
            continue
        usd = min(p.tranche * total, p.cap * total - pos["units"] * x)
        if usd <= 0 or not filter_ok(tk, 2):
            continue
        L["pending"].append({"type": "buy", "tk": tk, "tranche": 2, "usd": usd, "signal_date": str(d.date.date())})
    # 1차
    slots = p.max_names - len(L["pos"])

    def free(tk):
        return tk not in L["pos"] and not (tk in L["last_sell"] and d.idx - L["last_sell"][tk] <= p.cooldown)

    def order_1(tk):
        L["pending"].append({"type": "buy", "tk": tk, "tranche": 1, "usd": p.tranche * total, "signal_date": str(d.date.date()),
                             "stock_only_puddle": bool(d.qqq_ret20 > -0.03) if np.isfinite(d.qqq_ret20) else None})
        ev["entries"] += 1

    if L.get("random"):
        if random_n <= 0:
            return ev
        pool = []
        for tk in d.top10:
            x, ind = d.closes.get(tk), d.ind.get(tk) or {}
            if free(tk) and _qualified(ind, x):
                pool.append(tk)
        pool = [tk for tk in pool if filter_ok(tk, 1)]
        m = min(random_n, slots, len(pool))
        if m > 0:
            for tk in rng.choice(pool, size=m, replace=False):
                order_1(str(tk))
        return ev
    for tk in d.top10:
        if slots <= 0:
            break
        x, ind = d.closes.get(tk), d.ind.get(tk) or {}
        if not free(tk) or not _qualified(ind, x):
            continue
        m60, ab = ind.get("sma60"), ind.get("above_prev")
        if m60 is None or not np.isfinite(m60) or ab is None or not np.isfinite(ab) or not (x < m60 and ab >= 1.0):
            continue
        ev["signals_1"].append(tk)
        if not filter_ok(tk, 1):
            continue
        order_1(tk)
        slots -= 1
    return ev


def percentiles(values) -> dict:
    """무작위 분포 요약(10·20·50·80·90백분위)."""
    v = np.asarray(values, dtype=float)
    return {str(q): float(np.percentile(v, q)) for q in (10, 20, 50, 80, 90)}


def judge(v31: float, qqq: float, v3: float, rand_p80: float, v31_mdd: float, qqq_mdd: float) -> dict:
    """12개월 판정(사전 등록 2-8): 세후 값·최대 낙폭(음수 소수) → 기준별 O/X와 통과 여부."""
    c = {"1_beats_qqq": v31 > qqq, "2_beats_random_p80": v31 > rand_p80, "3_mdd_within_2pp": v31_mdd >= qqq_mdd - 0.02,
         "4_beats_v3": v31 > v3}
    return {**c, "pass": all(c.values())}
