"""웅덩이 반등 확인 전략 — 가상 1억 장부 순수 함수 (docs/prereg/puddle_rebound_forward.md, 태그 puddle-rebound-prereg).

웅덩이 v3.1(core/puddle_v31.py)과 같은 틀(상위 10·자격·지수 조건·실적 필터·한도·매도·세금)을 **고치지 않고** 불러 쓰고,
매수 신호만 「반등 확인」으로 바꾼다: 웅덩이가 열리면(60일선 아래 처음) 20거래일 대기,
그 안에서 RSI(14)가 40 이하를 거쳐 40 위로 다시 올라온 날 + 최근 3거래일 안 MACD 골든크로스 → 다음 거래일 종가에 5% 매수.

하루 순서(장부마다): pv.step(장부의 stocks=False — 세금·어제 신호 체결·12월 공제·가치까지만) → signals(오늘 신호, 내일 체결).
네트워크·파일·DB·현재 시각에 접근하지 않는다. 음수 shift 없음.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import puddle_v31 as pv
from core.indicators import macd_lines, rsi_wilder


@dataclass(frozen=True)
class Params(pv.Params):
    seed: int = 20261024
    wait_days: int = 20  # 반등 확인 대기 거래일(열린 날 = 1일째)
    rsi_period: int = 14
    rsi_level: float = 40.0
    gc_window: int = 3  # 골든크로스: 그날 포함 최근 3거래일


# ── 지표 ──────────────────────────────────────────────────────────────────────


def macd_cross(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """한 종목 종가(NaN 뺀 것) → (MACD 선, 시그널 선, 골든크로스 날 1.0·아니면 0.0, 첫날 NaN). 장부·후보 차트가 같이 쓴다."""
    m, sig = macd_lines(s)
    cross = ((m.shift(1) <= sig.shift(1)) & (m > sig)).astype(float)
    cross.iloc[:1] = np.nan
    return m, sig, cross


def rebound_indicators(close: pd.DataFrame, p: Params = Params()) -> dict[str, pd.DataFrame]:
    """종가(열 = 티커) → rsi·rsi_prev(전날 RSI)·gc3(그날 포함 최근 3거래일 안 골든크로스면 1.0, 아니면 0.0). 그날까지의 값만.

    종목마다 상장 첫날부터 계산한다(앞쪽 NaN은 건너뛰고 EWM 시작)."""
    rsi, gc = {}, {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < 2:
            rsi[t] = pd.Series(np.nan, index=close.index)
            gc[t] = pd.Series(np.nan, index=close.index)
            continue
        r = rsi_wilder(s, p.rsi_period)
        r.iloc[: p.rsi_period] = np.nan  # 처음 14일은 평활이 덜 돼 쓰지 않는다
        _, _, cross = macd_cross(s)
        rsi[t] = r.reindex(close.index)
        gc[t] = cross.rolling(p.gc_window, min_periods=p.gc_window).max().reindex(close.index)
    rsi_df = pd.DataFrame(rsi, index=close.index)
    return {"rsi": rsi_df, "rsi_prev": rsi_df.shift(1), "gc3": pd.DataFrame(gc, index=close.index)}


def _f(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


def rsi_cross_up(rsi_prev, rsi, level: float = 40.0) -> bool:
    """RSI 재돌파: 전날 < level, 그날 ≥ level."""
    a, b = _f(rsi_prev), _f(rsi)
    return a is not None and b is not None and a < level <= b


def confirmed(wait: dict, ind: dict, p: Params = Params()) -> bool:
    """반등 확인 ①②: 대기 중 RSI ≤ 40인 날이 있었고(wait["dipped"]) 그날 RSI 재돌파, 그리고 최근 3거래일 안 골든크로스."""
    return bool(wait.get("dipped")) and rsi_cross_up(ind.get("rsi_prev"), ind.get("rsi"), p.rsi_level) and _f(ind.get("gc3")) == 1.0


def opens_puddle(ind: dict, close) -> bool:
    """웅덩이 열림(1차): 자격 + Close < SMA60 + 직전 20거래일 모두 Close ≥ SMA60."""
    if not pv._qualified(ind, close):
        return False
    m60, ab = _f(ind.get("sma60")), _f(ind.get("above_prev"))
    return m60 is not None and ab is not None and close < m60 and ab >= 1.0


def opens_second(ind: dict, close) -> bool:
    """2차 대기 열림: Close < SMA120 그리고 Close > SMA200."""
    m120, m200 = _f(ind.get("sma120")), _f(ind.get("sma200"))
    x = _f(close)
    return None not in (m120, m200, x) and x < m120 and x > m200


# ── 장부 ─────────────────────────────────────────────────────────────────────


def new_ledger(name: str, day: pv.Day, p: Params = Params(), *, random: bool = False) -> dict:
    """시작일 종가에 QQQ 100%. 실적 필터 있음. stocks=False — pv.step은 신호를 내지 않고 이 모듈의 signals가 낸다."""
    L = pv.new_ledger(name, day, p, earnings_filter=True, stocks=False)
    L.update({"rb_random": random, "waits": {}})
    return L


def new_qqq_ledger(name: str, day: pv.Day, p: Params = Params()) -> dict:
    return pv.new_ledger(name, day, p, earnings_filter=False, stocks=False)


def _wait_day(w: dict, d: pv.Day) -> int:
    return d.idx - w["open_idx"] + 1


def signals(L: dict, d: pv.Day, p: Params = Params(), *, filing=None, random_n: int = 0, rng=None) -> dict:
    """pv.step 다음에 부른다 — 오늘 매도·2차·1차 신호를 L["pending"]에 넣는다(내일 체결).

    filing(티커) → True(접수 있음 = 건너뜀)·False·None(조회 실패). random 장부는 random_n개까지 무작위 1차(반등 확인 없음).
    출력(후보 칸용): {buys[{tk, tranche, rsi}], waiting[{tk, tranche, day, rsi}], opened[tk], skipped[{tk, tranche}],
                     failed[tk], unqualified[tk], entries}"""
    ev = {"buys": [], "waiting": [], "opened": [], "skipped": [], "failed": [], "unqualified": [], "entries": 0}
    # 1) 매도 — v3.1과 같은 규칙(보유 250 → 최고가 돌파 → 상위 10 밖, 최소 보유 60)
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
    for t in d.top10:
        if not pv._qualified(d.ind.get(t), d.closes.get(t)):
            ev["unqualified"].append(t)
    index_ok = bool(np.isfinite(d.qqq_sma200) and d.qqq > d.qqq_sma200)
    total = pv.nav_usd(L, d)

    def filter_ok(tk, tranche):
        r = filing(tk)
        if r is None:
            ev["failed"].append(tk)
            return False
        if r:
            ev["skipped"].append({"tk": tk, "tranche": tranche})
            return False
        return True

    def track(w: dict, ind: dict) -> None:
        r = _f(ind.get("rsi"))
        if r is not None and r <= p.rsi_level:
            w["dipped"] = True

    # 2) 2차 — 1차만 산 보유 종목: 120일선 아래 처음 → 20거래일 대기 → 반등 확인
    for tk, pos in L["pos"].items():
        x, ind = d.closes.get(tk), d.ind.get(tk) or {}
        if tk in selling:
            if isinstance(pos.get("w2"), dict):
                pos["w2"] = "done"
            continue
        if pos["tranches"] != 1 or pos.get("w2") == "done":
            continue
        if pos.get("w2") is None:
            if not opens_second(ind, x):
                continue
            pos["w2"] = {"open_idx": d.idx, "dipped": False}
        w = pos["w2"]
        if _wait_day(w, d) > p.wait_days:
            pos["w2"] = "done"
            continue
        track(w, ind)
        if confirmed(w, ind, p) and index_ok and pv._qualified(ind, x):
            usd = min(p.tranche * total, p.cap * total - pos["units"] * x)
            if usd > 0 and filter_ok(tk, 2):
                L["pending"].append({"type": "buy", "tk": tk, "tranche": 2, "usd": usd, "signal_date": str(d.date.date())})
                ev["buys"].append({"tk": tk, "tranche": 2, "rsi": _f(ind.get("rsi"))})
                pos["w2"] = "done"
                continue
        if isinstance(pos.get("w2"), dict):
            ev["waiting"].append({"tk": tk, "tranche": 2, "day": _wait_day(w, d), "rsi": _f(ind.get("rsi"))})

    # 3) 1차
    slots = p.max_names - len(L["pos"])

    def free(tk):
        return tk not in L["pos"] and not (tk in L["last_sell"] and d.idx - L["last_sell"][tk] <= p.cooldown)

    def order_1(tk):
        L["pending"].append({"type": "buy", "tk": tk, "tranche": 1, "usd": p.tranche * total, "signal_date": str(d.date.date()),
                             "stock_only_puddle": bool(d.qqq_ret20 > -0.03) if np.isfinite(d.qqq_ret20) else None})
        ev["entries"] += 1

    if L.get("rb_random"):
        if random_n <= 0 or not index_ok:
            return ev
        pool = [tk for tk in d.top10 if free(tk) and pv._qualified(d.ind.get(tk), d.closes.get(tk))]
        pool = [tk for tk in pool if filter_ok(tk, 1)]
        m = min(random_n, slots, len(pool))
        if m > 0:
            for tk in rng.choice(pool, size=m, replace=False):
                order_1(str(tk))
        return ev

    waits = L.setdefault("waits", {})
    for tk in list(waits):  # 기간 끝난 대기 닫기, 이미 보유한 종목 대기 닫기
        if tk in L["pos"] or _wait_day(waits[tk], d) > p.wait_days:
            del waits[tk]
    for tk in d.top10:  # 오늘 열림
        if tk in waits or tk in L["pos"]:
            continue
        if opens_puddle(d.ind.get(tk) or {}, d.closes.get(tk)):
            waits[tk] = {"open_idx": d.idx, "dipped": False}
            ev["opened"].append(tk)
    rank = {t: i for i, t in enumerate(d.top10)}
    for tk in sorted(waits, key=lambda t: (rank.get(t, len(rank)), t)):
        w, x, ind = waits[tk], d.closes.get(tk), d.ind.get(tk) or {}
        track(w, ind)
        if (confirmed(w, ind, p) and slots > 0 and tk in rank and index_ok and free(tk) and pv._qualified(ind, x)
                and filter_ok(tk, 1)):
            order_1(tk)
            ev["buys"].append({"tk": tk, "tranche": 1, "rsi": _f(ind.get("rsi"))})
            slots -= 1
            del waits[tk]
            continue
        ev["waiting"].append({"tk": tk, "tranche": 1, "day": _wait_day(w, d), "rsi": _f(ind.get("rsi"))})
    return ev


def step(L: dict, d: pv.Day, p: Params = Params(), *, filing=None, random_n: int = 0, rng=None) -> dict:
    """하루 진행: pv.step(세금·체결·12월 공제·가치) → signals. 출력: pv.step의 fills·sells와 signals 결과를 합친 dict."""
    base = pv.step(L, d, p)
    out = {"fills": base["fills"], "sells": base["sells"]}
    if L.get("stocks") is False and "waits" in L:
        out.update(signals(L, d, p, filing=filing, random_n=random_n, rng=rng))
    return out


def judge(value: float, qqq: float, v31: float | None, rand_p80: float, mdd: float, qqq_mdd: float) -> dict:
    """12개월 판정(사전 등록 2-6): ④ v3.1 값이 없으면 판정 불가(False)."""
    c = {"1_beats_qqq": value > qqq, "2_beats_random_p80": value > rand_p80, "3_mdd_within_2pp": mdd >= qqq_mdd - 0.02,
         "4_beats_v31": v31 is not None and value > v31}
    return {**c, "pass": all(c.values()), "v31_missing": v31 is None}
