"""사이클 실험실 2단계 — 순수 계산 (docs/prereg/cycle_lab_confirm.md 3~7절).

네트워크·파일 접근 없음. 모든 월간 Series·DataFrame은 월말 Timestamp 인덱스, 수익률은 소수.
신호·비중 함수는 1단계 lab.py와 같은 식이고 날짜 차단만 confirm_gate로 다르다(테스트로 일치 확인).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.tax import capital_gains_tax
from scripts.cycle_lab.confirm_gate import CUTOFF, assert_cutoff
from scripts.cycle_lab.lab import HOLD_MONTHS, LAG, _window_ok, add_months, ramp, random_weights

SEED = 20261103
N_RANDOM = 1000
COST_ONE_WAY = 0.001
START_KRW = 100_000_000.0
EVAL_START = pd.Timestamp("2013-01-31")
EVAL_END = CUTOFF
FIRST_JUDGE = EVAL_START  # 2012-12 판정 신호는 쓰지 않는다
EVAL_MONTHS = pd.date_range(EVAL_START, EVAL_END, freq="ME")
BOOK_MONTHS = pd.date_range("2012-12-31", EVAL_END, freq="ME")
CYCLE_17 = ["BldMt", "Cnstr", "Steel", "FabPr", "Mach", "ElcEq", "Autos", "Aero", "Ships", "Mines", "PerSv",
            "BusSv", "Hardw", "Chips", "RlEst", "Fin", "Other"]
CELLS = {1: ("Autos", "M2"), 2: ("RlEst", "M6"), 3: ("PerSv", "M5")}
EVENT_METHODS = {"M2", "M5"}
HOLM_ALPHA = 0.05
MDD_LIMIT = -0.03


class CellRejected(ValueError):
    """사전 등록한 3칸이 아닌 칸을 실행하려 할 때."""


# ── 당시 발표본 ───────────────────────────────────────────────────────────────


def realtime_values(rows: pd.DataFrame, current: pd.Series | None, months: pd.DatetimeIndex,
                    lags: tuple[int, ...] = (LAG,)) -> pd.DataFrame:
    """판정 달마다 그 달 말 시점에 알 수 있던 관측값.

    입력: rows(ALFRED 행 — 열 obs, realtime_start, realtime_end, value; realtime_end ≤ 2021-12-31),
          current(3순위 현재값 또는 None), months(판정 달 월말), lags(관측 = t − lag).
    출력: months 인덱스 DataFrame — 열 v{lag}, level(1=그 달 발표본, 2=2021-12-31 vintage, 3=현재값, NaN=없음).
    """
    assert_cutoff(months, current)
    if rows is not None and len(rows):
        assert_cutoff(pd.DatetimeIndex(rows["obs"]), pd.DatetimeIndex(rows["realtime_start"]),
                      pd.DatetimeIndex(rows["realtime_end"]))
    by_obs: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    last_vintage: dict[pd.Timestamp, float] = {}
    if rows is not None and len(rows):
        for obs, g in rows.groupby("obs"):
            rs = g["realtime_start"].to_numpy(dtype="datetime64[ns]")
            re_ = g["realtime_end"].to_numpy(dtype="datetime64[ns]")
            v = g["value"].to_numpy(dtype=float)
            by_obs[pd.Timestamp(obs)] = (rs, re_, v)
            last = g[g["realtime_end"] == CUTOFF]
            if len(last) and np.isfinite(last["value"].iloc[-1]):
                last_vintage[pd.Timestamp(obs)] = float(last["value"].iloc[-1])
    cur = {} if current is None else {pd.Timestamp(k): float(x) for k, x in current.dropna().items()}

    def at(obs: pd.Timestamp, d: np.datetime64) -> float:
        if obs not in by_obs:
            return np.nan
        rs, re_, v = by_obs[obs]
        hit = (rs <= d) & (d <= re_)
        if not hit.any():
            return np.nan
        return float(v[np.flatnonzero(hit)[-1]])

    out = {f"v{lag}": [] for lag in lags}
    out["level"] = []
    for t in months:
        d = np.datetime64(pd.Timestamp(t), "ns")
        obs_list = [add_months(t, -lag) for lag in lags]
        cands = [
            [at(o, d) for o in obs_list],
            [last_vintage.get(o, np.nan) for o in obs_list],
            [cur.get(o, np.nan) for o in obs_list],
        ]
        chosen, lvl = [np.nan] * len(lags), np.nan
        for k, vals in enumerate(cands, start=1):
            if all(np.isfinite(vals)):
                chosen, lvl = vals, float(k)
                break
        for lag, v in zip(lags, chosen):
            out[f"v{lag}"].append(v)
        out["level"].append(lvl)
    return pd.DataFrame(out, index=months)


def has_gap(level: pd.Series) -> bool:
    """첫 값이 나온 뒤 빈 달이 있으면 True(3순위 현재값이 필요). 입력: realtime_values의 level 열."""
    first = level.first_valid_index()
    return first is None or bool(level.loc[first:].isna().any())


# ── 신호 ──────────────────────────────────────────────────────────────────────


def m2_signal(s: pd.Series) -> pd.DataFrame:
    """M2. 입력: S = BAA − GS10(판정 달 인덱스, %p). 출력: sig, ok."""
    assert_cutoff(s)
    v = s.to_numpy(dtype=float)
    sig, ok = np.zeros(len(v), bool), np.zeros(len(v), bool)
    armed = False
    for i in range(len(v)):
        if not _window_ok(v, i, 24):
            continue
        ok[i] = True
        mx = v[i - 24:i].max()
        if armed and v[i] <= mx - 0.5:
            sig[i], armed = True, False
        elif v[i] > mx:
            armed = True
    return pd.DataFrame({"sig": sig, "ok": ok}, index=s.index)


def m5_signal(ret: pd.Series) -> pd.DataFrame:
    """M5. 입력: 산업 월간 수익률(예열 포함). 출력: sig, ok — 누적지수 60개월 고점 대비 −40% 뒤 10개월 평균 위 회복."""
    assert_cutoff(ret)
    level = (1.0 + ret).cumprod()
    lv = level.to_numpy(dtype=float)
    sig, ok = np.zeros(len(lv), bool), np.zeros(len(lv), bool)
    armed, arm_i = False, -1
    for i in range(len(lv)):
        if not _window_ok(lv, i, 59):
            continue
        ok[i] = True
        dd = lv[i] / lv[i - 59:i + 1].max() - 1.0
        if armed and i > arm_i and lv[i] > lv[i - 9:i + 1].mean():
            sig[i], armed = True, False
            continue
        if not armed and dd <= -0.40:
            armed, arm_i = True, i
    return pd.DataFrame({"sig": sig, "ok": ok}, index=ret.index)


def m6_state(ret: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """M6. 입력: 사이클 산업 월간 수익률(예열 포함, 열 순서 = 표 순서). 출력: 같은 모양 bool(그 달 12-1 수익 상위 top)."""
    assert_cutoff(ret)
    gross = (1.0 + ret)
    mom = gross.shift(1).rolling(11, min_periods=11).apply(np.prod, raw=True) - 1.0
    out = pd.DataFrame(False, index=ret.index, columns=ret.columns)
    cols = list(ret.columns)
    for t, row in mom.iterrows():
        avail = row.dropna()
        if len(avail) < top:
            continue
        order = sorted(avail.index, key=lambda c: (-avail[c], cols.index(c)))
        out.loc[t, order[:top]] = True
    return out


def cell_signal(industry: str, method: str, ind: pd.DataFrame, spread: pd.Series | None) -> pd.Series:
    """사전 등록 3칸의 판정 달 신호(bool). 입력: 칸, 산업 수익률(예열 포함), S(M2용). 3칸이 아니면 CellRejected."""
    if (industry, method) not in CELLS.values():
        raise CellRejected(f"사전 등록에 없는 칸: {industry}×{method}")
    assert_cutoff(ind, spread)
    if method == "M2":
        return m2_signal(spread)["sig"]
    if method == "M6":
        return m6_state(ind[CYCLE_17])[industry]
    r = ind[industry].loc[ind[industry].first_valid_index():]
    if r.isna().any():
        last_na = r[r.isna()].index.max()
        if last_na >= EVAL_START:
            raise ValueError(f"{industry}: 평가 창에 수익률 결측")
        r = r.loc[last_na + pd.offsets.MonthEnd(1):]
    return m5_signal(r)["sig"]


# ── 비중 ──────────────────────────────────────────────────────────────────────


def event_weights(sig: pd.Series, months: pd.DatetimeIndex) -> tuple[pd.Series, list[dict]]:
    """사건 신호 → 수익 평가 달의 위성 비중. 보유 중 신호 무시, 24개월째 말 복귀. 출력: (비중, 진입 목록)."""
    assert_cutoff(sig, months)
    w = np.zeros(len(months))
    eps: list[dict] = []
    start = None
    for j, m in enumerate(months):
        prev = bool(sig.get(add_months(m, -1), False))
        if start is not None and j - start >= HOLD_MONTHS:
            start = None
        if start is None and prev:
            start = j
            eps.append({"signal": add_months(m, -1), "start": m, "months": 0})
        if start is not None:
            w[j] = ramp(j - start)
            eps[-1]["months"] += 1
    return pd.Series(w, index=months), eps


def state_weights(state: pd.Series, months: pd.DatetimeIndex) -> tuple[pd.Series, list[dict]]:
    """상태 신호 → 위성 비중. 조건 꺼진 판정 달 다음 달 또는 24개월째 말 복귀(빠른 쪽). 24개월로 나왔으면 꺼졌다 켜져야 재진입."""
    assert_cutoff(state, months)
    w = np.zeros(len(months))
    eps: list[dict] = []
    start, need_off = None, False
    for j, m in enumerate(months):
        v = state.get(add_months(m, -1), False)
        s = bool(v) if pd.notna(v) else False
        if start is not None:
            if not s:
                start = None
            elif j - start >= HOLD_MONTHS:
                start, need_off = None, True
        if start is None:
            if not s:
                need_off = False
            elif not need_off:
                start = j
                eps.append({"signal": add_months(m, -1), "start": m, "months": 0})
        if start is not None:
            w[j] = ramp(j - start)
            eps[-1]["months"] += 1
    return pd.Series(w, index=months), eps


def cell_weights(sig: pd.Series, method: str, months: pd.DatetimeIndex = EVAL_MONTHS) -> tuple[pd.Series, list[dict]]:
    """2013-01 이후 판정 신호만으로 비중을 만든다(2012년 이전 신호 미승계). 출력: (비중, 진입 목록)."""
    live = sig.astype(bool) & (sig.index >= FIRST_JUDGE)
    return (event_weights if method in EVENT_METHODS else state_weights)(live, months)


# ── 원화 세후 장부 ────────────────────────────────────────────────────────────


def run_book(w: np.ndarray, qqq: np.ndarray, sat: np.ndarray, fx: np.ndarray, book_months: pd.DatetimeIndex,
             start_krw: float = START_KRW, cost: float = COST_ONE_WAY) -> dict:
    """월간 원화 세후 장부.

    입력: w(달 t의 위성 비중, 길이 T), qqq·sat(달 t 달러 수익률, 길이 T), fx(월말 원/달러, 길이 T+1),
          book_months(월말, 길이 T+1, 첫 값이 시작 시점).
    출력: end_after_tax_krw, cagr, curve(달마다 그날 전부 팔면 세후 값, 길이 T+1), mdd, taxes_paid_krw, costs_krw.
    세금은 core.tax.capital_gains_tax. 평균 원가(원화), 5월 말 납부, 끝날 전량 매도 비용 포함.
    """
    T = len(w)
    q, s = start_krw / fx[0], 0.0
    bq, bs = start_krw, 0.0
    realized, owed = 0.0, 0.0
    taxes_paid, costs_krw = 0.0, 0.0
    prev_w = 0.0
    curve = np.empty(T + 1)
    years, mons = book_months.year, book_months.month
    for t in range(T + 1):
        if t > 0:
            q *= 1.0 + qqq[t - 1]
            s *= 1.0 + sat[t - 1]
            if years[t] != years[t - 1]:
                owed += capital_gains_tax(realized)
                realized = 0.0
        krw_now = (q + s) * fx[t]
        sell_cost = cost * krw_now
        gain = realized + (q * fx[t] - bq) + (s * fx[t] - bs) - sell_cost
        curve[t] = krw_now - sell_cost - capital_gains_tax(gain) - owed
        if t == T:
            break
        nav = q + s
        if mons[t] == 5 and owed > 0.0:
            nav -= owed / fx[t]
            taxes_paid += owed
            owed = 0.0
        trade_cost = cost * 2.0 * abs(w[t] - prev_w) * nav
        nav -= trade_cost
        realized -= trade_cost * fx[t]
        costs_krw += trade_cost * fx[t]
        prev_w = float(w[t])
        tq, ts = (1.0 - w[t]) * nav, w[t] * nav
        for name, target in (("q", tq), ("s", ts)):
            held, basis = (q, bq) if name == "q" else (s, bs)
            if target < held and held > 0.0:
                frac = (held - target) / held
                realized += (held - target) * fx[t] - basis * frac
                basis *= 1.0 - frac
            else:
                basis += (target - held) * fx[t]
            if name == "q":
                q, bq = target, basis
            else:
                s, bs = target, basis
    level = curve / np.maximum.accumulate(curve) - 1.0
    return {"end_after_tax_krw": float(curve[T]), "cagr": float((curve[T] / start_krw) ** (12.0 / T) - 1.0),
            "curve": curve, "mdd": float(level.min()), "taxes_paid_krw": taxes_paid, "costs_krw": costs_krw}


# ── 칸 평가와 판정 ────────────────────────────────────────────────────────────


def evaluate_cell(w: pd.Series, eps: list[dict], qqq: pd.Series, sat: pd.Series, fx: pd.Series,
                  n_random: int = N_RANDOM, seed: int = SEED) -> dict:
    """칸 하나. 입력: 비중(평가 달), 진입 목록, QQQ·산업 월 수익, 월말 환율(시작 시점 포함).
    출력: 전략·QQQ·상시 보유 세후 값, 무작위 p, 기준 1·2·4 결과, 곡선."""
    assert_cutoff(w, qqq, sat, fx)
    months = w.index
    book_months = pd.DatetimeIndex([add_months(months[0], -1)]).append(months)
    q, s = qqq.reindex(months).to_numpy(float), sat.reindex(months).to_numpy(float)
    f = fx.reindex(book_months).to_numpy(float)
    if not (np.isfinite(q).all() and np.isfinite(s).all() and np.isfinite(f).all()):
        raise ValueError("평가 창에 QQQ·산업·환율 결측")
    wv = w.to_numpy(float)
    strat = run_book(wv, q, s, f, book_months)
    base = run_book(np.zeros(len(wv)), q, s, f, book_months)
    wbar = float(wv.mean())
    const = run_book(np.full(len(wv), wbar), q, s, f, book_months)
    durs = [e["months"] for e in eps]
    if eps:
        W = random_weights(durs, len(wv), n_random, seed)
        rand = np.array([run_book(W[i], q, s, f, book_months)["cagr"] for i in range(n_random)])
        r = int((rand >= strat["cagr"] - 1e-15).sum())
        p, pctile = (r + 1) / (n_random + 1), 100.0 * float((rand < strat["cagr"] - 1e-15).sum()) / n_random
    else:
        p, pctile = 1.0, np.nan
    mdd_diff = strat["mdd"] - base["mdd"]
    return {"entries": len(eps), "avg_weight": wbar, "avg_hold": float(np.mean(durs)) if durs else np.nan,
            "cagr": strat["cagr"], "qqq_cagr": base["cagr"], "const_cagr": const["cagr"],
            "end_krw": strat["end_after_tax_krw"], "qqq_end_krw": base["end_after_tax_krw"],
            "const_end_krw": const["end_after_tax_krw"], "mdd": strat["mdd"], "qqq_mdd": base["mdd"], "mdd_diff": mdd_diff,
            "taxes_krw": strat["taxes_paid_krw"], "costs_krw": strat["costs_krw"], "p": p, "pctile": pctile,
            "c1_beats_qqq": bool(strat["cagr"] > base["cagr"]), "c2_beats_const": bool(strat["cagr"] > const["cagr"]),
            "c4_mdd_ok": mdd_ok(strat["mdd"], base["mdd"]),
            "curves": {"strategy": strat["curve"], "qqq": base["curve"], "constant": const["curve"]},
            "book_months": book_months}


def mdd_ok(mdd_strategy: float, mdd_qqq: float) -> bool:
    """기준 4. 전략 MDD − QQQ MDD > −3%p면 True(음수끼리의 차, 정확히 3%p 깊으면 불합격)."""
    return bool(round(mdd_strategy - mdd_qqq, 12) > MDD_LIMIT)


def holm(p_by_cell: dict[int, float], alpha: float = HOLM_ALPHA) -> dict[int, dict]:
    """Holm. 입력: {칸 번호: p}. 출력: {칸 번호: {rank, threshold, passed}}. 처음 실패하면 그 뒤 칸도 불합격."""
    order = sorted(p_by_cell, key=lambda c: (p_by_cell[c], c))
    m = len(order)
    out, alive = {}, True
    for i, c in enumerate(order):
        thr = alpha / (m - i)
        alive = alive and p_by_cell[c] <= thr
        out[c] = {"rank": i + 1, "threshold": thr, "passed": alive}
    return out


def verdict(passed: dict[int, bool]) -> dict:
    """7절 판정. 입력: {칸 번호: 합격}. 출력: result(PASS/FAIL), kind, catalog(탈락 보관 추가 여부)."""
    if passed.get(1) or passed.get(3):
        return {"result": "PASS", "kind": "칸 1 또는 칸 3 합격 — 3단계 후보", "catalog": False}
    if passed.get(2):
        return {"result": "FAIL", "kind": "칸 2(대조군)만 합격 — 상대 강도 결과로만 기록, 3단계 안 감", "catalog": False}
    return {"result": "FAIL", "kind": "합격 칸 없음 — 탈락 보관", "catalog": True}


def compute(ind: pd.DataFrame, spread: pd.Series, qqq: pd.Series, fx: pd.Series, n_random: int = N_RANDOM,
            seed: int = SEED) -> dict:
    """3칸 전체. 입력: 산업 수익률(예열 포함), S(판정 달), QQQ 월 수익, 월말 환율(2012-12 포함).
    출력: {cells: {칸 번호: 결과 + industry·method·entries_list·criteria·passed}, holm, verdict}."""
    assert_cutoff(ind, spread, qqq, fx)
    res: dict[int, dict] = {}
    for no, (industry, method) in CELLS.items():
        sig = cell_signal(industry, method, ind, spread)
        w, eps = cell_weights(sig, method)
        r = evaluate_cell(w, eps, qqq, ind[industry], fx, n_random, seed)
        res[no] = {"industry": industry, "method": method, "weights": w, "entries_list": eps, **r}
    h = holm({no: r["p"] for no, r in res.items()})
    for no, r in res.items():
        r["c3_holm"] = h[no]["passed"]
        r["passed"] = bool(r["entries"] > 0 and r["c1_beats_qqq"] and r["c2_beats_const"] and r["c3_holm"] and r["c4_mdd_ok"])
    return {"cells": res, "holm": h, "verdict": verdict({no: r["passed"] for no, r in res.items()})}


def qqq_monthly_total(qqq: pd.DataFrame, ndx: pd.DataFrame, max_diff: float | None = 0.02) -> tuple[pd.Series, pd.DataFrame]:
    """QQQ 월간 총수익. 일간 가격 수익률이 ^NDX와 max_diff 넘게 다르면 ^NDX 수익률로 바꾸고 배당을 더한다(1단계와 같은 식).
    max_diff=None이면 바꾸지 않는다(참고용). 출력: (월말 인덱스 월간 총수익 — 첫 거래일이 있는 달 제외, 바꾼 날 표 qqq·ndx)."""
    assert_cutoff(qqq, ndx)
    c = qqq["Close"]
    pr = c.pct_change()
    rr = ndx["Close"].reindex(c.index).pct_change()
    bad = ((pr - rr).abs() > (0.02 if max_diff is None else max_diff)) & rr.notna()
    table = pd.DataFrame({"qqq": pr[bad], "ndx": rr[bad]})
    if max_diff is not None:
        pr = pr.where(~bad, rr)
    tr = pr + qqq["Dividends"] / c.shift(1)
    tr = tr.iloc[1:]
    m = (1.0 + tr).groupby(tr.index.to_period("M")).prod() - 1.0
    m.index = m.index.to_timestamp(how="end").normalize()
    first_month = c.index[0].to_period("M").to_timestamp(how="end").normalize()
    return m[m.index > first_month], table


def month_end_fx(daily: pd.Series, months: pd.DatetimeIndex = BOOK_MONTHS) -> pd.Series:
    """일별 환율 → 달마다 말일 이하 최근 값. 출력: months 인덱스 Series."""
    assert_cutoff(daily, months)
    d = daily.dropna().sort_index()
    return pd.Series([d.loc[:m].iloc[-1] if len(d.loc[:m]) else np.nan for m in months], index=months, dtype=float)
