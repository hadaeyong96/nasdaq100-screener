"""사이클 실험실 1단계 — 순수 계산 (docs/prereg/cycle_lab_discovery.md 3~7절).

네트워크·파일 접근 없음. 모든 월간 Series·DataFrame은 월말 Timestamp 인덱스, 수익률은 소수(0.01 = 1%).
날짜가 들어오는 입구는 gate.assert_cutoff()로 2012-12-31 뒤 날짜를 막는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.cycle_lab.gate import CUTOFF, assert_cutoff

EVAL_START = pd.Timestamp("1963-07-31")
EVAL_END = pd.Timestamp("2012-12-31")
M4_FIX_FIRST, M4_FIX_LAST = pd.Timestamp("1963-07-31"), pd.Timestamp("1987-11-30")
M4_SIG_FIRST = pd.Timestamp("1988-01-31")
LAG = 2
HOLD_MONTHS = 24
RAMP = (0.05, 0.10, 0.15, 0.20)
COST_ONE_WAY = 0.001
SEED = 20261102
N_RANDOM = 1000
PHASES = ["(+,+)", "(+,-)", "(-,+)", "(-,-)"]


def month_end(ts) -> pd.Timestamp:
    """날짜 → 그 달 말일 Timestamp."""
    return pd.Timestamp(ts) + pd.offsets.MonthEnd(0)


def add_months(ts: pd.Timestamp, k: int) -> pd.Timestamp:
    """월말 Timestamp에 k달을 더한 월말."""
    return month_end(pd.Timestamp(ts) + pd.DateOffset(months=k))


def mnum(ts) -> int:
    """월말 Timestamp → 정수 달 번호(연×12+월). 달 차이 계산용."""
    ts = pd.Timestamp(ts)
    return ts.year * 12 + ts.month - 1


# ── 당시 발표본 (3절) ─────────────────────────────────────────────────────────


def realtime_values(rows: pd.DataFrame, current: pd.Series | None, months: pd.DatetimeIndex,
                    lags: tuple[int, ...] = (LAG,)) -> pd.DataFrame:
    """판정 달마다 그 달 말 시점에 알 수 있던 관측값.

    입력: rows(ALFRED 실시간 구간 행 — 열 obs(관측 달 월말), realtime_start, realtime_end, value; realtime_end ≤ 2012-12-31),
          current(현재 발표본, 관측 달 월말 인덱스 — 3순위), months(판정 달 월말), lags(관측 = t − lag).
    출력: months 인덱스 DataFrame — 열 v{lag}(값), level(1=그 달 발표본, 2=2012-12-31 vintage, 3=현재값, NaN=값 없음).
          한 달의 lags는 모두 같은 순위에서 읽는다(모두 있는 가장 높은 순위).
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


def pct_change_rt(rt: pd.DataFrame, a: int = 2, b: int = 8) -> pd.Series:
    """실시간 값 표에서 (관측 t−a / 관측 t−b − 1) × 100. 입력: realtime_values(lags=(a, b)) 결과."""
    return (rt[f"v{a}"] / rt[f"v{b}"] - 1.0) * 100.0


# ── 사이클 산업 선정 (4절) ────────────────────────────────────────────────────


def select_cycle(ind: pd.DataFrame, mkt_rf: pd.Series, rf: pd.Series, usrec: pd.Series,
                 start: pd.Timestamp = EVAL_START, end: pd.Timestamp = EVAL_END) -> pd.DataFrame:
    """사이클 산업 선정 표.

    입력: ind(산업 월간 수익률, 열 순서 = French 표 순서, 결측 산업은 이미 제외), mkt_rf·rf(월간), usrec(0/1), 구간.
    출력: 산업 인덱스 DataFrame — beta, std, rec_rel, rank_std(1=가장 큼), rank_rec(1=가장 낮음), A, B, C, cycle.
    """
    assert_cutoff(ind, mkt_rf, rf, usrec)
    sl = slice(start, end)
    r, m, f = ind.loc[sl], mkt_rf.loc[sl], rf.loc[sl]
    if r.isna().any().any():
        raise ValueError("선정 구간에 결측이 있는 산업이 남아 있습니다")
    mkt = m + f
    rec = usrec.reindex(r.index).fillna(0) == 1
    x = m.to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    rows = []
    for c in r.columns:
        y = (r[c] - f).to_numpy()
        beta = float(np.linalg.lstsq(X, y, rcond=None)[0][1])
        rows.append({"industry": c, "beta": beta, "std": float(r[c].std(ddof=1)),
                     "rec_rel": float((r[c] - mkt)[rec].mean())})
    df = pd.DataFrame(rows).set_index("industry")
    n = len(df)
    order = {c: i for i, c in enumerate(df.index)}
    df["_o"] = [order[c] for c in df.index]
    df["rank_std"] = df.sort_values(["std", "_o"], ascending=[False, True]).assign(k=range(1, n + 1))["k"].reindex(df.index)
    df["rank_rec"] = df.sort_values(["rec_rel", "_o"], ascending=[True, True]).assign(k=range(1, n + 1))["k"].reindex(df.index)
    df["A"] = df["beta"] > 1.1
    df["B"] = df["rank_std"] <= n // 3
    df["C"] = df["rank_rec"] <= n // 2
    df["cycle"] = (df["A"] | df["B"]) & df["C"]
    df["market_std_ratio"] = df["std"] / float(mkt.std(ddof=1))
    return df.drop(columns="_o")


# ── 예측법 신호 (5절) ─────────────────────────────────────────────────────────


def _window_ok(v: np.ndarray, i: int, k: int) -> bool:
    return i >= k and np.isfinite(v[i - k:i + 1]).all()


def m1_signal(x: pd.Series) -> pd.DataFrame:
    """M1. 입력: X(판정 달 인덱스, %). 출력: DataFrame sig(bool), ok(정의됨) — 무장/재무장 규칙 포함."""
    assert_cutoff(x)
    v = x.to_numpy(dtype=float)
    sig, ok = np.zeros(len(v), bool), np.zeros(len(v), bool)
    armed = False
    for i in range(len(v)):
        if not _window_ok(v, i, 12):
            continue
        ok[i] = True
        mn = v[i - 12:i].min()
        if armed and v[i] - mn >= 2.0:
            sig[i], armed = True, False
        elif v[i] < mn:
            armed = True
    return pd.DataFrame({"sig": sig, "ok": ok}, index=x.index)


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


def m3_signal(c: pd.Series) -> pd.DataFrame:
    """M3. 입력: C = GS10 − TB3MS(판정 달 인덱스). 출력: sig, ok."""
    assert_cutoff(c)
    v = c.to_numpy(dtype=float)
    sig, ok = np.zeros(len(v), bool), np.isfinite(v)
    armed = False
    for i in range(len(v)):
        if not ok[i]:
            continue
        if v[i] < 0:
            armed = True
        elif armed and v[i] > 0:
            sig[i], armed = True, False
    return pd.DataFrame({"sig": sig, "ok": ok}, index=c.index)


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


def m4_phase(indpro_a: pd.Series, indpro_b: pd.Series, c_a: pd.Series, c_b: pd.Series) -> pd.Series:
    """M4 국면 번호(0=(+,+), 1=(+,−), 2=(−,+), 3=(−,−), NaN=정의 안 됨). 입력: 관측 t−2·t−8 값(판정 달 인덱스)."""
    assert_cutoff(indpro_a, c_a)
    g = (indpro_a / indpro_b - 1.0)
    h = (c_a - c_b)
    code = np.where(g > 0, 0, 2) + np.where(h > 0, 0, 1)
    out = pd.Series(code.astype(float), index=indpro_a.index)
    return out.where(g.notna() & h.notna())


def m4_fix(phase: pd.Series, rel: pd.Series, first: pd.Timestamp = M4_FIX_FIRST, last: pd.Timestamp = M4_FIX_LAST) -> tuple[int | None, pd.Series]:
    """M4 국면 고정. 입력: phase(판정 달), rel(산업 − 시장 월간), 고정 구간(판정 달). 판정 달 t 국면 ↔ t+1 수익.
    출력: (최고 국면 번호 또는 None, 국면별 평균 Series)."""
    assert_cutoff(phase, rel)
    # 판정 달 t의 짝 = t+1 달 수익. 수익 쪽 인덱스를 한 달 앞 판정 달 이름으로 붙인다(고정 구간 안에서만 씀).
    nxt = pd.Series(rel.to_numpy(), index=pd.DatetimeIndex([add_months(t, -1) for t in rel.index]))
    df = pd.DataFrame({"p": phase, "r": nxt.reindex(phase.index)}).loc[first:last].dropna()
    means = df.groupby("p")["r"].mean().reindex(range(4)).astype(float)
    means.index = PHASES
    if means.notna().sum() == 0:
        return None, means
    return int(np.nanargmax(means.to_numpy())), means


def m6_state(ret: pd.DataFrame, top: int = 3) -> pd.DataFrame:
    """M6. 입력: 사이클 산업 월간 수익률(예열 포함, 열 순서 = 표 순서). 출력: 같은 모양 bool(그 달 12-1 수익 상위 top),
    정의 안 된 칸은 False. 12-1 = t−11 ~ t−1 누적."""
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


def m6_ok(ret: pd.DataFrame) -> pd.DataFrame:
    """M6 정의 여부(12-1 수익이 있는 칸)."""
    return ((1.0 + ret).shift(1).rolling(11, min_periods=11).apply(np.prod, raw=True)).notna()


# ── 전략 비중 (6절) ───────────────────────────────────────────────────────────


def ramp(k: int) -> float:
    """매수 첫 달(k=0)부터 목표 비중."""
    return RAMP[min(k, len(RAMP) - 1)]


def episode_path(d: int) -> np.ndarray:
    """보유 d개월의 비중 경로."""
    return np.array([ramp(k) for k in range(d)])


def event_weights(sig: pd.Series, months: pd.DatetimeIndex) -> tuple[pd.Series, list[dict]]:
    """사건 신호 → 수익 평가 달의 위성 비중.

    입력: sig(판정 달 bool), months(수익 평가 달 월말, 연속). 판정 달 t 신호 → t+1 매수. 보유 중 신호 무시, 24개월째 말 복귀.
    출력: (비중 Series, 진입 목록 [{signal, start, months}]).
    """
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
    """상태 신호 → 위성 비중. 조건 꺼진 판정 달 다음 달 또는 24개월째 말 복귀(빠른 쪽). 24개월로 나왔으면 꺼졌다 켜져야 재진입.
    입력·출력은 event_weights와 같다(state = 판정 달 bool, NaN은 꺼짐)."""
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


# ── 성적 (6·7절) ──────────────────────────────────────────────────────────────


def strat_returns(w: np.ndarray, core: np.ndarray, ind: np.ndarray) -> np.ndarray:
    """월 수익 = (1−w)·핵심 + w·산업 − 0.1%×|Δw|×2. w는 (T,) 또는 (n, T). 첫 달 직전 비중은 0."""
    w = np.asarray(w, dtype=float)
    prev = np.concatenate([np.zeros(w.shape[:-1] + (1,)), w[..., :-1]], axis=-1)
    cost = COST_ONE_WAY * 2.0 * np.abs(w - prev)
    return (1.0 - w) * core + w * ind - cost


def cagr(r: np.ndarray) -> np.ndarray:
    """월 수익(마지막 축) → 연복리."""
    r = np.asarray(r, dtype=float)
    n = r.shape[-1]
    return np.exp(np.log1p(r).sum(axis=-1) * 12.0 / n) - 1.0


def mdd(r: np.ndarray) -> float:
    """월 수익 → 최대 낙폭(음수)."""
    lv = np.cumprod(1.0 + np.asarray(r, dtype=float))
    lv = np.concatenate([[1.0], lv])
    return float((lv / np.maximum.accumulate(lv) - 1.0).min())


def random_weights(durations: list[int], T: int, n: int = N_RANDOM, seed: int = SEED) -> np.ndarray:
    """무작위 대조 비중 행렬 (n, T). 같은 보유 기간 목록, 겹치지 않게 시작 달만 무작위(순서 섞기 + 별과 막대 간격)."""
    rng = np.random.default_rng(seed)
    K, L = len(durations), int(sum(durations))
    W = np.zeros((n, T))
    if K == 0:
        return W
    free = T - L
    if free < 0:
        raise ValueError("보유 합이 창 길이보다 깁니다")
    paths = {d: episode_path(d) for d in set(durations)}
    for i in range(n):
        order = rng.permutation(np.asarray(durations))
        cuts = np.sort(rng.choice(free + K, size=K, replace=False))
        gaps = np.diff(np.concatenate([[-1], cuts])) - 1  # 각 구간 앞 간격 K개
        pos = 0
        for d, g in zip(order, gaps):
            pos += int(g)
            W[i, pos:pos + d] = paths[int(d)]
            pos += int(d)
    return W


def bh_q(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q값."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


def zigzag_troughs(level: pd.Series, down: float = 0.30, up: float = 0.40) -> list[pd.Timestamp]:
    """월말 지그재그 바닥(사후 표시). 입력: 누적지수. 출력: 확정된 바닥 달 목록."""
    assert_cutoff(level)
    v, idx = level.to_numpy(dtype=float), level.index
    troughs: list[pd.Timestamp] = []
    mode = None
    hi = lo = v[0]
    lo_i = 0
    for i in range(1, len(v)):
        x = v[i]
        if mode is None:
            if x > hi:
                hi = x
            if x < lo:
                lo, lo_i = x, i
            if x <= hi * (1 - down):
                mode, lo, lo_i = "down", x, i
            elif x >= lo * (1 + up):
                troughs.append(idx[lo_i])
                mode, hi = "up", x
        elif mode == "up":
            if x > hi:
                hi = x
            elif x <= hi * (1 - down):
                mode, lo, lo_i = "down", x, i
        else:
            if x < lo:
                lo, lo_i = x, i
            elif x >= lo * (1 + up):
                troughs.append(idx[lo_i])
                mode, hi = "up", x
    return troughs


def lead_stats(signal_months: list[pd.Timestamp], troughs: list[pd.Timestamp], judge_months: pd.DatetimeIndex) -> dict:
    """선행성: 신호 s가 바닥 B의 [B−12, B+3] 안이면 적중. 입력: 진입 신호 판정 달, 창 안 바닥, 판정 가능 달.
    출력: hits, hit_rate, base_rate, lead_median, false."""
    tb = [mnum(b) for b in troughs]
    hits, leads = 0, []
    for s in signal_months:
        ms = mnum(s)
        near = [b for b in tb if b - 12 <= ms <= b + 3]
        if near:
            hits += 1
            b = min(near, key=lambda b: abs(b - ms))
            leads.append(b - ms)
    inside = [any(b - 12 <= mnum(t) <= b + 3 for b in tb) for t in judge_months]
    n = len(signal_months)
    return {"hits": hits, "hit_rate": hits / n if n else np.nan,
            "base_rate": float(np.mean(inside)) if len(inside) else np.nan,
            "lead_median": float(np.median(leads)) if leads else np.nan, "false": n - hits}


def evaluate_cell(w: pd.Series, eps: list[dict], core: pd.Series, ind: pd.Series, n_random: int = N_RANDOM,
                  seed: int = SEED) -> dict:
    """칸 하나 성적. 입력: 비중(수익 평가 달), 진입 목록, 핵심·산업 월 수익(같은 달 포함). 출력: 7절 지표 dict."""
    assert_cutoff(w, core, ind)
    months = w.index
    c, s = core.reindex(months).to_numpy(float), ind.reindex(months).to_numpy(float)
    if not (np.isfinite(c).all() and np.isfinite(s).all()):
        raise ValueError("평가 창에 핵심·산업 수익 결측")
    wv = w.to_numpy(float)
    r = strat_returns(wv, c, s)
    base = c
    ex = float(cagr(r) - cagr(base))
    wbar = float(wv.mean())
    const = (1.0 - wbar) * c + wbar * s
    years = len(months) / 12.0
    dw = np.abs(np.diff(np.concatenate([[0.0], wv])))
    durs = [e["months"] for e in eps]
    out = {"eval_start": months[0], "eval_end": months[-1], "entries": len(eps), "excess_cagr": ex,
           "cagr": float(cagr(r)), "core_cagr": float(cagr(base)), "mdd_diff": mdd(r) - mdd(base),
           "avg_weight": wbar, "turnover": float(dw.sum() / years),
           "avg_hold": float(np.mean(durs)) if durs else np.nan,
           "vs_constant": float(cagr(r) - cagr(const)),
           "min_hold": int(min(durs)) if durs else np.nan}
    if eps:
        W = random_weights(durs, len(months), n_random, seed)
        rex = cagr(strat_returns(W, c, s)) - cagr(base)
        rr = int((rex >= ex - 1e-15).sum())
        out.update({"pctile": 100.0 * float((rex < ex - 1e-15).sum()) / n_random, "p": (rr + 1) / (n_random + 1)})
    else:
        out.update({"pctile": np.nan, "p": 1.0})
    return out


def pick_top(cells: pd.DataFrame, k: int = 3, exclude=None) -> pd.DataFrame:
    """상위 선정. 입력: cells(열 industry, method, pctile, excess_cagr, entries, ind_order, method_order).
    자격: pctile ≥ 90, excess > 0, entries ≥ 8. 순위: pctile↓ → excess↓ → 표 순서. 같은 예측법 최대 1개.
    exclude: 뺄 (industry, method) 목록. 출력: 뽑힌 행(최대 k개)."""
    df = cells.copy()
    if exclude:
        df = df[~df.apply(lambda r: (r["industry"], r["method"]) in set(exclude), axis=1)]
    elig = df[(df["pctile"] >= 90) & (df["excess_cagr"] > 0) & (df["entries"] >= 8)]
    elig = elig.sort_values(["pctile", "excess_cagr", "ind_order", "method_order"], ascending=[False, False, True, True])
    picked, used = [], set()
    for _, row in elig.iterrows():
        if row["method"] in used:
            continue
        picked.append(row)
        used.add(row["method"])
        if len(picked) == k:
            break
    return pd.DataFrame(picked)
