"""내부자 집단 매수 — 원화 세후 장부·무작위·판정 (docs/prereg/insider_cluster.md 4~6절). 순수 함수.

Market: 거래일 T개 × 종목 N개 배열. 가격은 분할 반영·배당 미반영 종가, 첫 값 앞·마지막 거래일 뒤는 NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.tax import capital_gains_tax
from scripts.insider_cluster.gate import assert_cutoff

COST = 0.001
WEIGHT = 0.02
MAX_POSITIONS = 10
HOLD_DAYS = 252
START_KRW = 100_000_000.0
SEED = 20261107
N_RANDOM = 1000
P_LIMIT = 0.01
MDD_LIMIT = -0.03
MIN_ENTRIES = 30


@dataclass
class Market:
    """시장 배열. days(T), qqq(T), fx(T), px(T×N, NaN=가격 없음), last_idx(N, 마지막 거래일 위치), ciks(N),
    eligible(T×N, 그날 구역 ①·유동성 통과), basket(T, 그날 수익 = 전날 대상 동일 가중)."""

    days: pd.DatetimeIndex
    qqq: np.ndarray
    fx: np.ndarray
    px: np.ndarray
    last_idx: np.ndarray
    ciks: list
    eligible: np.ndarray
    basket: np.ndarray

    def __post_init__(self) -> None:
        assert_cutoff(self.days)
        self.col = {c: i for i, c in enumerate(self.ciks)}
        y, m = self.days.year.values, self.days.month.values
        self.new_year = np.concatenate([[False], y[1:] != y[:-1]])
        self.may_first = (m == 5) & np.concatenate([[False], m[1:] != m[:-1]])
        self.month_end = np.concatenate([m[1:] != m[:-1], [True]])


def basket_returns(px: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """바스켓 일 수익(길이 T). d일 수익 = d−1일 대상 종목의 d−1→d 수익 평균. 대상이 없으면 0."""
    out = np.zeros(px.shape[0])
    for d in range(1, px.shape[0]):
        s = eligible[d - 1]
        if not s.any():
            continue
        a, b = px[d - 1, s], px[d, s]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.any():
            out[d] = float(np.mean(b[ok] / a[ok] - 1.0))
    return out


@dataclass
class Entry:
    """신호 하나. confirm_idx(신호 확인일 위치 — 휴장일이면 다음 거래일), exec_idx(체결일 위치), col(종목 열), key(보고용)."""

    confirm_idx: int
    exec_idx: int
    col: int
    key: object = None


def run_strategy(entries: list[Entry], m: Market, want_path: bool = False) -> dict:
    """전략 장부. 입력: 체결일 오름차순 신호, 시장. 출력: end_krw, cagr, curve(월말 그날 전부 팔면 세후 값), mdd,
    trades, counts, want_path면 sat_weight(T)·cost_frac(T)."""
    T = len(m.days)
    fx, qq, px = m.fx, m.qqq, m.px
    q_units = START_KRW / fx[0] / qq[0]
    q_basis = START_KRW
    realized, owed = 0.0, 0.0
    pos: dict[int, dict] = {}
    exits: dict[int, list[int]] = {}
    last_exit: dict[int, int] = {}
    by_exec: dict[int, list[Entry]] = {}
    for e in entries:
        by_exec.setdefault(e.exec_idx, []).append(e)
    trades: list[dict] = []
    counts = {"ignored_holding": 0, "skipped_full": 0, "no_price_at_exec": 0, "delisted": 0, "forced_end": 0, "taxes_paid_krw": 0.0, "costs_krw": 0.0}
    curve, curve_days = [], []
    sat = np.zeros(T) if want_path else None
    cost_frac = np.zeros(T) if want_path else None

    def nav_usd(d: int) -> float:
        return q_units * qq[d] + sum(p["units"] * px[d, c] for c, p in pos.items())

    def liquidation(d: int) -> float:
        krw = nav_usd(d) * fx[d]
        sell_cost = COST * krw
        gain = realized + (q_units * qq[d] * fx[d] - q_basis) + sum(p["units"] * px[d, c] * fx[d] - p["basis"] for c, p in pos.items()) - sell_cost
        return krw - sell_cost - capital_gains_tax(gain) - owed

    for d in range(T):
        if m.new_year[d]:
            owed += capital_gains_tax(realized)
            realized = 0.0
        day_cost = 0.0
        if d < T - 1 and d in exits:
            for c in exits.pop(d):
                p = pos.pop(c)
                sale = p["units"] * px[d, c]
                sell_cost = COST * sale
                realized += (sale - sell_cost) * fx[d] - p["basis"]
                buy = (sale - sell_cost) / (1.0 + COST)
                realized -= COST * buy * fx[d]
                day_cost += sell_cost + COST * buy
                q_units += buy / qq[d]
                q_basis += buy * fx[d]
                last_exit[c] = d
                delisted = d == m.last_idx[c] and d < p["exec"] + HOLD_DAYS
                counts["delisted"] += int(delisted)
                trades.append({"key": p["key"], "col": c, "confirm_idx": p["confirm"], "exec_idx": p["exec"], "exit_idx": d, "reason": "가격 끊김" if delisted else "252거래일",
                               "ret": px[d, c] / p["entry_px"] - 1.0})
        if m.may_first[d] and owed > 0.0:
            need = owed / fx[d] / (1.0 - COST)
            q_val = q_units * qq[d]
            if need > q_val:
                raise ValueError("납부액이 QQQ 평가액보다 큽니다")
            frac = need / q_val
            realized += need * (1.0 - COST) * fx[d] - q_basis * frac
            q_basis *= 1.0 - frac
            q_units *= 1.0 - frac
            counts["taxes_paid_krw"] += owed
            counts["costs_krw"] += COST * need * fx[d]
            owed = 0.0
        if d in by_exec and d < T:
            for e in by_exec[d]:
                c = e.col
                # 신호 확인일 종가에 보유 중이면 무시 (해석 11): 지금 보유 중이거나 확인일 뒤에 청산됨
                if c in pos or last_exit.get(c, -1) > e.confirm_idx:
                    counts["ignored_holding"] += 1
                    continue
                if not np.isfinite(px[d, c]) or d > m.last_idx[c] or (d == m.last_idx[c] and d < T - 1):
                    counts["no_price_at_exec"] += 1  # 체결일이 마지막 거래일이면 보유할 다음 날이 없다
                    continue
                if len(pos) >= MAX_POSITIONS:
                    counts["skipped_full"] += 1
                    continue
                x = WEIGHT * nav_usd(d)
                q_val = q_units * qq[d]
                frac = x / q_val
                sell_cost = COST * x
                realized += (x - sell_cost) * fx[d] - q_basis * frac
                q_basis *= 1.0 - frac
                q_units *= 1.0 - frac
                buy = (x - sell_cost) / (1.0 + COST)
                realized -= COST * buy * fx[d]
                day_cost += sell_cost + COST * buy
                pos[c] = {"units": buy / px[d, c], "basis": buy * fx[d], "exec": d, "confirm": e.confirm_idx, "entry_px": px[d, c], "key": e.key}
                exits.setdefault(min(d + HOLD_DAYS, int(m.last_idx[c])), []).append(c)
        counts["costs_krw"] += day_cost * fx[d]
        if want_path:
            nav = nav_usd(d)
            sat[d] = sum(p["units"] * px[d, c] for c, p in pos.items()) / nav
            cost_frac[d] = day_cost / nav
        if m.month_end[d]:
            curve.append(liquidation(d))
            curve_days.append(d)
    end = liquidation(T - 1)
    for c, p in pos.items():
        counts["forced_end"] += 1
        trades.append({"key": p["key"], "col": c, "confirm_idx": p["confirm"], "exec_idx": p["exec"], "exit_idx": T - 1, "reason": "끝날 강제 청산",
                       "ret": px[T - 1, c] / p["entry_px"] - 1.0})
    out = {"end_krw": end, "cagr": cagr(end, m.days), "trades": trades, "counts": counts}
    full = np.array([START_KRW] + curve)
    out.update({"curve": full, "curve_days": curve_days, "mdd": mdd(full)})
    if want_path:
        out.update({"sat_weight": sat, "cost_frac": cost_frac})
    return out


def run_basket(sat_weight: np.ndarray, cost_frac: np.ndarray, m: Market) -> dict:
    """같은 노출 단순 보유. 입력: 전략의 매일 위성 비중·비용 비율. 바스켓은 원가 계산상 자산 1개(해석 21). 출력: end_krw, cagr, curve, mdd."""
    T = len(m.days)
    q = START_KRW / m.fx[0]
    b = 0.0
    bq, bb = START_KRW, 0.0
    realized, owed = 0.0, 0.0
    curve = []

    def liquidation(d: int) -> float:
        krw = (q + b) * m.fx[d]
        sell_cost = COST * krw
        gain = realized + (q * m.fx[d] - bq) + (b * m.fx[d] - bb) - sell_cost
        return krw - sell_cost - capital_gains_tax(gain) - owed

    for d in range(T):
        if d > 0:
            q *= m.qqq[d] / m.qqq[d - 1]
            b *= 1.0 + m.basket[d]
        if m.new_year[d]:
            owed += capital_gains_tax(realized)
            realized = 0.0
        nav = q + b
        if m.may_first[d] and owed > 0.0:
            need = owed / m.fx[d] / (1.0 - COST)
            frac = need / q
            realized += need * (1.0 - COST) * m.fx[d] - bq * frac
            bq *= 1.0 - frac
            q *= 1.0 - frac
            nav = q + b
            owed = 0.0
        cost = cost_frac[d] * nav
        realized -= cost * m.fx[d]
        nav -= cost
        tb = sat_weight[d] * nav
        tq = nav - tb
        for name, target in (("q", tq), ("b", tb)):
            held, basis = (q, bq) if name == "q" else (b, bb)
            if target < held and held > 0.0:
                frac = (held - target) / held
                realized += (held - target) * m.fx[d] - basis * frac
                basis *= 1.0 - frac
            else:
                basis += (target - held) * m.fx[d]
            if name == "q":
                q, bq = target, basis
            else:
                b, bb = target, basis
        if m.month_end[d]:
            curve.append(liquidation(d))
    end = liquidation(T - 1)
    full = np.array([START_KRW] + curve)
    return {"end_krw": end, "cagr": cagr(end, m.days), "curve": full, "mdd": mdd(full)}


def cagr(end_krw: float, days: pd.DatetimeIndex) -> float:
    """연복리 = (끝/1억)^(365.25/달력일) − 1."""
    return float((end_krw / START_KRW) ** (365.25 / (days[-1] - days[0]).days) - 1.0)


def mdd(curve: np.ndarray) -> float:
    """곡선의 최대 낙폭(음수)."""
    return float((curve / np.maximum.accumulate(curve) - 1.0).min())


def mdd_ok(mdd_strategy: float, mdd_qqq: float) -> bool:
    """기준 4. 전략 MDD − QQQ MDD > −3%p면 True(정확히 3%p 깊으면 불합격)."""
    return bool(round(mdd_strategy - mdd_qqq, 12) > MDD_LIMIT)


def random_control(entries: list[Entry], actual_cols_by_confirm: dict[int, set], m: Market, n: int = N_RANDOM, seed: int = SEED) -> np.ndarray:
    """무작위 대조. 입력: 실제 체결된 진입, 확인일별 실제 신호 종목 열, 시장. 출력: 회차별 세후 원화 연복리(길이 n).

    후보 = 그날 대상, 그날 실제 신호 종목 아님, 체결일 가격 있음, 그 회차에서 보유 중 아님(해석 22).
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(n)
    plan = sorted(entries, key=lambda e: (e.exec_idx, e.confirm_idx))
    cand_base = []
    for e in plan:
        T = len(m.days)
        ok = m.eligible[e.confirm_idx] & np.isfinite(m.px[e.exec_idx]) & ((m.last_idx > e.exec_idx) | (m.last_idx == T - 1))
        for c in actual_cols_by_confirm.get(e.confirm_idx, ()):
            ok[c] = False
        cand_base.append(np.flatnonzero(ok))
    for i in range(n):
        open_until: dict[int, int] = {}
        chosen = []
        for e, cands in zip(plan, cand_base):
            held = {c for c, u in open_until.items() if u > e.confirm_idx}
            pick = cands if not held else np.array([c for c in cands if c not in held], dtype=int)
            if len(pick) == 0:
                continue
            c = int(rng.choice(pick))
            ex = min(e.exec_idx + HOLD_DAYS, int(m.last_idx[c]))
            open_until[c] = ex
            chosen.append(Entry(e.confirm_idx, e.exec_idx, c))
        out[i] = run_strategy(chosen, m)["cagr"]
    return out


def p_value(strategy_cagr: float, random_cagr: np.ndarray) -> tuple[float, int]:
    """p = (r+1)/(n+1), r = 전략 이상인 회차 수. 출력: (p, r)."""
    r = int((random_cagr >= strategy_cagr - 1e-15).sum())
    return (r + 1) / (len(random_cagr) + 1), r


def judge(main: dict, qqq: dict, basket: dict, p: float, cons: dict, cons_basket: dict, var2: dict, var2_basket: dict, var4: dict, var4_basket: dict, entries: int) -> dict:
    """6절 판정. 출력: c1~c6, enough(진입 ≥ 30), result(PASS/FAIL)."""
    c = {"c1": main["cagr"] > qqq["cagr"], "c2": main["cagr"] > basket["cagr"], "c3": p <= P_LIMIT, "c4": mdd_ok(main["mdd"], qqq["mdd"]),
         "c5": cons["cagr"] > qqq["cagr"] and cons["cagr"] > cons_basket["cagr"],
         "c6": var2["cagr"] > var2_basket["cagr"] and var4["cagr"] > var4_basket["cagr"], "enough": entries >= MIN_ENTRIES}
    c = {k: bool(v) for k, v in c.items()}
    c["result"] = "PASS" if all(c.values()) else "FAIL"
    return c
