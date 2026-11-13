"""실적급등 매도 조건 비교(pead_exits) — 순수 함수 (사전 등록: docs/prereg/pead_exits.md, configs/pead_exits.yaml).

같은 신호·같은 매수(반응일 다음 거래일 시가)에 파는 날만 다른 가상 장부 A·B와 15칸 포트폴리오, 가이던스 꼬리표, 비교 통계.
V1.0(core/pead_watch) 규칙은 바꾸지 않는다 — 겹침 규칙·매수 체결·끝난 매매 초과는 그쪽 함수를 그대로 쓴다.
네트워크·파일·DB·현재 시각 접근 없음.
"""

from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core import pead_tags as ptag
from core import pead_timing as pt
from core import pead_validation as pv
from core import pead_watch as pw

SRC_YF = "yfinance"
SRC_GAP = "8-K 간격 추정"
SRC_DEFAULT = "기본 60일"
KIND = {"a": "발표 전", "target": "목표", "max": "90일"}


@dataclass(frozen=True)
class Params:
    """configs/pead_exits.yaml 수치."""

    min_days_after_d0: int = 30
    max_gaps: int = 4
    default_hold: int = 60
    a_offset: int = 2
    min_hold: int = 5
    max_hold: int = 90
    target_mult: float = 1.20
    hit_last: int = 89
    slots: int = 15
    iterations: int = 10_000
    seed: int = 20261113
    alpha: float = 0.05
    fdr: float = 0.10
    min_bin_n: int = 5


def params_from(pre: dict) -> Params:
    """configs/pead_exits.yaml 내용 → Params."""
    a, b = pre["exit_a"], pre["exit_b"]
    return Params(min_days_after_d0=int(a["next_date"]["min_days_after_d0"]), max_gaps=int(a["fallback"]["max_gaps"]),
                  default_hold=int(a["default_hold"]), a_offset=int(a["sell"]["offset_trading_days_before"]),
                  min_hold=int(a["min_hold"]), max_hold=int(a["max_hold"]), target_mult=float(b["target_mult"]),
                  hit_last=int(b["hit_window"][1]), slots=int(pre["portfolio"]["slots"]),
                  iterations=int(pre["compare"]["iterations"]), seed=int(pre["compare"]["seed"]),
                  alpha=float(pre["compare"]["alpha"]), fdr=float(pre["analysis"]["tags"]["fdr"]),
                  min_bin_n=int(pre["analysis"]["tags"]["min_bin_n"]))


# ── 조건 A: 다음 발표 예정일 → 매도 위치 ─────────────────────────────────────


def pick_next_date(dates: list, d0, min_days: int = 30) -> pd.Timestamp | None:
    """yfinance 실적 날짜들 → 반응일 + min_days 달력일 이후의 첫 날짜(시각 버림). 없으면 None."""
    lo = pd.Timestamp(d0).normalize() + pd.Timedelta(days=min_days)
    xs = sorted({pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo else pd.Timestamp(x).normalize()
                 for x in dates if x is not None and not pd.isna(x)})
    xs = [x for x in xs if x >= lo]
    return xs[0] if xs else None


def estimate_from_gaps(reaction_days: list, max_gaps: int = 4) -> tuple[pd.Timestamp | None, int]:
    """이번 발표 반응일과 그 앞 발표 반응일들(순서 무관) → (이번 반응일 + 이웃 간격 평균(달력일, 반올림), 쓴 간격 수).

    마지막(가장 늦은) 반응일이 이번 발표다. 간격은 가장 최근 max_gaps개. 반응일이 2개 미만이면 (None, 0).
    """
    xs = sorted({pd.Timestamp(x).normalize() for x in reaction_days})
    if len(xs) < 2:
        return None, 0
    xs = xs[-(max_gaps + 1):]
    gaps = [(b - a).days for a, b in zip(xs[:-1], xs[1:])]
    return xs[-1] + pd.Timedelta(days=int(round(float(np.mean(gaps))))), len(gaps)


def a_plan_index(days: pd.DatetimeIndex, ib: int, next_date, p: Params = Params()) -> int:
    """매수 위치 ib, 예정일(None이면 기본 보유) → A 매도 예정 위치. E 이상 첫 거래일 − a_offset, [ib+min, ib+max]로 자름."""
    if next_date is None:
        return ib + p.default_hold
    j = int(days.searchsorted(pd.Timestamp(next_date), side="left")) - p.a_offset
    return min(max(j, ib + p.min_hold), ib + p.max_hold)


# ── 조건 B: 종가 목표 → 다음 거래일 시가 ─────────────────────────────────────


def b_exit(a: pt.Aligned, bench: pt.Aligned, ib: int, i_now: int, buy_px: float, p: Params = Params()) -> dict | None:
    """B 매도 판정(i_now까지의 값만 본다). 아직 안 팔면 None.

    목표 = buy_px × target_mult. 매수일 ~ hit_last 거래일째 실제 종가가 목표 이상인 첫날 k →
    k 다음 ~ max_hold 거래일째 중 시가가 있는 첫날 시가(QQQ도 그날 시가). 그런 날이 없으면 max_hold째까지 마지막 종가.
    못 닿으면 max_hold째 종가(V1과 같음, 행이 없으면 그 전 마지막 종가).
    출력: {sell_idx, sell_px, q_sell, kind('target'|'max'), sell_at('open'|'close'), hit_idx}
    """
    target = buy_px * p.target_mult
    jmax = ib + p.max_hold
    hit = None
    for k in range(ib, min(i_now, ib + p.hit_last) + 1):
        c = a.close[k]
        if np.isfinite(c) and c >= target:
            hit = k
            break
    if hit is not None:
        for m in range(hit + 1, min(i_now, jmax) + 1):
            if np.isfinite(a.open[m]) and a.open[m] > 0:
                q = bench.open[m] if np.isfinite(bench.open[m]) else bench.close[m]
                return {"sell_idx": m, "sell_px": float(a.open[m]), "q_sell": float(q), "kind": "target", "sell_at": "open",
                        "hit_idx": hit}
    if i_now < jmax:
        return None
    e = pt.exit_point(a, ib, jmax)
    if e is None:
        return {"sell_idx": None, "sell_px": None, "q_sell": None, "kind": "max", "sell_at": "close", "hit_idx": hit}
    return {"sell_idx": e, "sell_px": float(a.close[e]), "q_sell": float(bench.close[e]),
            "kind": "target" if hit is not None else "max", "sell_at": "close", "hit_idx": hit}


# ── 장부 하루 진행 (V1 step_day와 같은 틀, 파는 날만 다름) ───────────────────


def step_exit_day(trades: list[dict], d, new_signals: list[dict], prices: dict[str, pt.Aligned], bench: pt.Aligned,
                  days: pd.DatetimeIndex, rule: str, plans: dict | None = None, p: Params = Params()) -> tuple[list[dict], list[str]]:
    """A('exit_a') 또는 B('exit_b') 장부를 거래일 d 하루 진행: ① 매수 체결 ② 매도 ③ d가 반응일인 새 신호.

    입력: plans(A만) — (종목, 반응일) → {next_date('YYYY-MM-DD'|None), source}. 없거나 날짜가 없으면 기본 보유.
    매매 dict = V1 칸 + next_date, next_source(A) / target_px, hit_day, hit_days(B) + exit_kind, sell_at.
    매수 체결·겹침·가격 공백은 V1(pw.step_day)과 같다. 출력: (새 매매 목록, 경고).
    """
    i = int(days.get_loc(pd.Timestamp(d)))
    ds = pw._d(d)
    out, warn = [], []
    for t in trades:
        t = dict(t)
        a = prices.get(t["ticker"])
        if t["status"] == "pending" and t["buy_day"] <= ds:
            ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
            if a is None:
                warn.append(f"{t['ticker']} 가격 없음 — 매수 체결을 다음 실행으로 미룸({rule})")
            else:
                px = a.open[ib]
                if np.isfinite(px) and px > 0 and np.isfinite(bench.open[ib]):
                    t.update(status="open", buy_px=float(px), q_buy=float(bench.open[ib]))
                    if rule == "exit_b":
                        t["target_px"] = float(px) * p.target_mult
                else:
                    t.update(status="skipped", skip="가격 공백")
        if t["status"] == "open":
            ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
            if a is None:
                if t["plan_day"] <= ds:
                    warn.append(f"{t['ticker']} 가격 없음 — 매도를 다음 실행으로 미룸({rule})")
            elif rule == "exit_a":
                if t["plan_day"] <= ds:
                    e = pt.exit_point(a, ib, int(days.get_loc(pd.Timestamp(t["plan_day"]))))
                    if e is None:
                        t.update(status="skipped", skip="가격 공백")
                    else:
                        t.update(status="closed", sell_day=pw._d(days[e]), sell_px=float(a.close[e]),
                                 q_sell=float(bench.close[e]), exit_kind="a", sell_at="close")
            else:
                x = b_exit(a, bench, ib, i, float(t["buy_px"]), p)
                if x is not None:
                    if x["hit_idx"] is not None:
                        t.update(hit_day=pw._d(days[x["hit_idx"]]), hit_days=int(x["hit_idx"] - ib))
                    if x["sell_idx"] is None:
                        t.update(status="skipped", skip="가격 공백")
                    else:
                        t.update(status="closed", sell_day=pw._d(days[x["sell_idx"]]), sell_px=x["sell_px"],
                                 q_sell=x["q_sell"], exit_kind=x["kind"], sell_at=x["sell_at"])
        out.append(t)
    for s in sorted(new_signals, key=lambda x: x["ticker"]):
        d0 = pw._d(s["d0"])
        base = {"ledger": rule, "ticker": s["ticker"], "d0": d0, "label": s.get("label"), "buy_day": None, "plan_day": None,
                "buy_px": None, "q_buy": None, "sell_day": None, "sell_px": None, "q_sell": None, "skip": None,
                "exit_kind": None, "sell_at": None}
        if rule == "exit_a":
            pl = (plans or {}).get((s["ticker"], d0)) or {}
            base.update(next_date=pl.get("next_date"), next_source=pl.get("source") or SRC_DEFAULT)
        else:
            base.update(target_px=None, hit_day=None, hit_days=None)
        busy = [t for t in out if t["ticker"] == s["ticker"] and pw._busy_until(t) is not None
                and t["d0"] <= d0 < pw._busy_until(t)]
        if busy:
            out.append({**base, "status": "skipped", "skip": "보유 중"})
            continue
        ib = i + 1
        j = a_plan_index(days, ib, base.get("next_date"), p) if rule == "exit_a" else ib + p.max_hold
        if j >= len(days) or ib + p.max_hold >= len(days):
            out.append({**base, "status": "skipped", "skip": "달력 끝"})
            continue
        out.append({**base, "status": "pending", "buy_day": pw._d(days[ib]), "plan_day": pw._d(days[j])})
    return out, warn


# ── 15칸 포트폴리오 ─────────────────────────────────────────────────────────


def pf_sell_day(t: dict) -> str | None:
    """포트폴리오 매도 처리일: 끝난 매매의 실제 매도일(B 시가 매도·A 조기 매도 포함)."""
    return t.get("sell_day") if t.get("status") == "closed" else None


def pf_step_slots(state: dict, d, trades: list[dict], q_open: float, q_close: float, marks: dict[str, float],
                  p: pv.PfParams = pv.PfParams(), slots: int = 15) -> tuple[dict, list[dict], dict]:
    """15칸 포트폴리오 하루 진행. 순서: 시가 매도 → 시가 매수 → 종가 매도 → 종가 평가. state는 바꾸지 않는다.

    매수 = status 'open'|'closed'이고 buy_day ≤ d인 처리 안 한 매매. 예산 = 남은 현금(QQQ 시가) ÷ 빈 칸 수, 칸이 다 차면 건너뜀.
    매도 = status 'closed'이고 sell_day ≤ d인 보유 매매(sell_at 'open'이면 QQQ 시가로, 아니면 종가로 되돌림).
    출력: (새 상태, 사건, 하루 줄) — pv.pf_step과 같은 모양.
    """
    ds = pd.Timestamp(d).date().isoformat()
    st = {**state, "pos": {k: dict(v) for k, v in state["pos"].items()}, "done": list(state["done"])}
    done = set(st["done"])
    c = p.commission_pct / 100
    ev = []

    def sell(open_side: bool):
        for t in sorted(trades, key=lambda t: (str(t.get("sell_day")), t["ticker"])):
            k = pv.pf_key(t)
            sd = pf_sell_day(t)
            if k not in st["pos"] or sd is None or sd > ds or (t.get("sell_at") == "open") != open_side:
                continue
            h = st["pos"].pop(k)
            got = h["shares"] * float(t["sell_px"]) * (1 - c)
            st["q_units"] += got / (q_open if open_side else q_close)
            ev.append({"date": ds, "action": "sell", "ticker": t["ticker"], "d0": str(t["d0"])[:10], "shares": h["shares"],
                       "px": float(t["sell_px"]), "amount_usd": got, "note": f"손익 {got - h['cost']:+.0f}달러"})

    sell(True)
    buys = sorted((t for t in trades if t["status"] in ("open", "closed") and t.get("buy_px") and t["buy_day"] <= ds
                   and pv.pf_key(t) not in done), key=lambda t: (t["buy_day"], t["ticker"], str(t["d0"])))
    for t in buys:
        k = pv.pf_key(t)
        qo = float(t.get("q_buy") or q_open)
        px = float(t["buy_px"])
        base = {"date": ds, "ticker": t["ticker"], "d0": str(t["d0"])[:10], "px": px}
        free = slots - len(st["pos"])
        cash = st["q_units"] * qo
        if free <= 0:
            ev.append({**base, "action": "skip", "shares": 0, "amount_usd": 0.0, "note": f"{slots}칸 다 참"})
            done.add(k)
            continue
        budget = cash / free
        shares = math.floor(budget / (px * (1 + c))) if px > 0 else 0
        if budget < st["min_usd"] or shares <= 0:
            why = "최소 금액 미만" if budget < st["min_usd"] else "1주 값보다 예산이 작음"
            ev.append({**base, "action": "skip", "shares": 0, "amount_usd": budget, "note": why})
        else:
            cost = shares * px * (1 + c)
            st["q_units"] -= cost / qo
            st["pos"][k] = {"ticker": t["ticker"], "shares": shares, "cost": cost}
            ev.append({**base, "action": "buy", "shares": shares, "amount_usd": cost, "note": ""})
        done.add(k)
    sell(False)
    st["done"] = sorted(done)
    stock = sum(h["shares"] * float(marks.get(h["ticker"], h["cost"] / h["shares"])) for h in st["pos"].values())
    cash = st["q_units"] * q_close
    value = cash + stock
    q100 = st["q100_units"] * q_close
    row = {"date": ds, "value_usd": value, "cash_usd": cash, "stock_usd": stock, "qqq100_usd": q100,
           "vs_qqq_pct": (value / q100 - 1) * 100, "n_pos": len(st["pos"])}
    return st, ev, row


# ── 꼬리표: 가이던스 상향 · 전날 200일선 ─────────────────────────────────────

GUID_VERBS = {"raise", "raises", "raised", "raising", "increase", "increases", "increased", "increasing"}
GUID_N1 = {"guidance", "outlook", "outlooks", "forecast", "forecasts"}
GUID_N2 = GUID_N1 | {"expectations", "target", "targets", "range"}
GUID_NEG = {"not", "no", "never", "without"}


def _plain_lower(doc: str) -> str:
    t = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", doc)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", _html.unescape(t)).lower()


def _period_at(w: list[str], j: int) -> int:
    """w[j]부터 기간 낱말이면 그 길이(full year·fiscal year = 2, annual·fy = 1), 아니면 0."""
    if j < len(w) and w[j] in ("annual", "fy"):
        return 1
    if j + 1 < len(w) and w[j] in ("full", "fiscal") and w[j + 1] == "year":
        return 2
    return 0


def guidance_raise(doc: str) -> tuple[int | None, str | None]:
    """보도자료 HTML/글 → (1 상향 문장 있음 | 0 없음 | None 단어 0개, 처음 맞은 문장 앞 200자). 정의: 사전 등록 1-3절."""
    text = _plain_lower(doc)
    sents = re.split(r"(?<=[.!?;])\s+", text)
    if not any(re.search(r"[a-z]", s) for s in sents):
        return None, None
    for s in sents:
        w = re.findall(r"[a-z]+", s)
        for i, x in enumerate(w):
            if x not in GUID_VERBS or GUID_NEG & set(w[max(0, i - 3):i]):
                continue
            if GUID_N1 & set(w[i + 1:i + 9]):
                return 1, s[:200]
            for j in range(i + 1, min(i + 5, len(w))):
                n = _period_at(w, j)
                if n and GUID_N2 & set(w[j + n:j + n + 6]):
                    return 1, s[:200]
    return 0, None


def prev_below_sma200(close: pd.Series, d0) -> int | None:
    """반응일 전날 종가 < 전날까지 종가 200개 평균이면 1, 아니면 0. 자료 부족이면 None. d0 당일·뒤 값은 안 본다."""
    h = close.loc[:pd.Timestamp(d0)].dropna()
    if not len(h) or h.index[-1] != pd.Timestamp(d0) or len(h) < 2:
        return None
    above, _ = ptag.sma200(h.iloc[:-1], h.index[-2])
    return None if above is None else int(not above)


# ── 비교 통계 (보고용) ─────────────────────────────────────────────────────


def clusters(tickers: list[str], d0s: list[str]) -> np.ndarray:
    """같은 종목이거나 같은 반응일이면 한 묶음(연결 요소). 출력: 묶음 번호 배열(0부터)."""
    n = len(tickers)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in (tickers, d0s):
        first: dict = {}
        for i, k in enumerate(key):
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    roots = [find(i) for i in range(n)]
    ids = {r: j for j, r in enumerate(dict.fromkeys(roots))}
    return np.array([ids[r] for r in roots], dtype=int)


def sign_flip_p(d: np.ndarray, groups: np.ndarray, iters: int, seed: int) -> float:
    """묶음 단위 부호 뒤집기 양쪽 p: |뒤집은 평균| ≥ |실제 평균|인 비율."""
    d = np.asarray(d, dtype=float)
    if not len(d):
        return float("nan")
    g = np.asarray(groups)
    k = int(g.max()) + 1
    sums = np.bincount(g, weights=d, minlength=k)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(iters, k))
    m = np.abs(flips @ sums) / len(d)
    return float((m >= abs(d.mean()) - 1e-15).mean())


def cluster_boot_ci(x: np.ndarray, groups: np.ndarray, iters: int, seed: int) -> tuple[float, float]:
    """묶음 단위 복원 추출 평균의 2.5·97.5 백분위수."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(groups)
    k = int(g.max()) + 1
    sums = np.bincount(g, weights=x, minlength=k)
    cnts = np.bincount(g, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, k, size=(iters, k))
    m = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def holm(ps: list[float]) -> list[float]:
    """Holm 보정 p (단조, 1 이하)."""
    order = np.argsort(ps)
    m = len(ps)
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i]))
        adj[i] = run
    return adj


def bh_reject(ps: list[float], q: float) -> list[bool]:
    """Benjamini-Hochberg: FDR q에서 기각되는 검정."""
    m = len(ps)
    if not m:
        return []
    order = np.argsort(ps)
    kmax = 0
    for r, i in enumerate(order, start=1):
        if ps[i] <= q * r / m:
            kmax = r
    rej = [False] * m
    for r, i in enumerate(order, start=1):
        rej[i] = r <= kmax
    return rej


def perm_p(x: np.ndarray, mask: np.ndarray, iters: int, seed: int) -> float:
    """매매 단위 꼬리표 섞기 양쪽 p: (그룹 평균 − 나머지 평균)의 |섞은 값| ≥ |실제 값| 비율."""
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n1, n = int(mask.sum()), len(x)
    if n1 == 0 or n1 == n:
        return float("nan")
    tot = x.sum()

    def diff(s1):
        return s1 / n1 - (tot - s1) / (n - n1)

    obs = abs(diff(x[mask].sum()))
    rng = np.random.default_rng(seed)
    s = np.array([x[rng.permutation(n)[:n1]].sum() for _ in range(iters)])
    return float((np.abs(diff(s)) >= obs - 1e-15).mean())


def metrics(x: np.ndarray, groups: np.ndarray | None = None, iters: int = 10_000, seed: int = 20261113) -> dict:
    """매매 초과 배열 → 평균·중앙값·승률·상위 10% 뺀 평균·최대 손실·95% 구간(묶음 복원 추출, % 단위)."""
    x = np.asarray(x, dtype=float)
    if not len(x):
        return {"n": 0}
    top = math.ceil(len(x) * 0.1)
    rest = np.sort(x)[:len(x) - top] if len(x) > top else np.array([])
    g = groups if groups is not None else np.arange(len(x))
    lo, hi = cluster_boot_ci(x, g, iters, seed)
    return {"n": int(len(x)), "mean": float(x.mean() * 100), "median": float(np.median(x) * 100),
            "win": float((x > 0).mean() * 100), "mean_ex_top10": float(rest.mean() * 100) if len(rest) else None,
            "max_loss": float(x.min() * 100), "ci_lo": lo * 100, "ci_hi": hi * 100}


def in_bin(x, edge: str) -> bool:
    """구간 글('x<=-10', '-10<x<=0', '2<=x<=3', '1', '월', 'x>median' 제외)에 x가 드는가."""
    e = edge.strip()
    if "x" not in e:
        return str(x) == e if not isinstance(x, (int, float)) else float(x) == float(e)
    parts = re.findall(r"(-?[\d.]+)?\s*(<=|<|>=|>)?\s*x\s*(<=|<|>=|>)?\s*(-?[\d.]+)?", e)[0]
    lo, op1, op2, hi = parts
    ok = True
    if lo:
        ok &= (float(lo) <= x) if op1 == "<=" else (float(lo) < x)
    if hi:
        ok &= {"<=": x <= float(hi), "<": x < float(hi), ">=": x >= float(hi), ">": x > float(hi)}[op2]
    return bool(ok)


def monotone(means: list[float]) -> bool:
    """구간 평균(순서대로)이 단조 증가 또는 단조 감소면 True."""
    a = np.asarray(means, dtype=float)
    dif = np.diff(a)
    return bool((dif >= 0).all() or (dif <= 0).all())


# ── 보고 (docs/prereg/pead_exits.md 3~5장, 판정·매매에 안 씀) ───────────────

TAG_BINS = {  # configs/pead_exits.yaml analysis.tags.bins 와 같은 순서·기준. (연속형인가, 구간)
    "tag_sma200_dist": (True, ["x<=-10", "-10<x<=0", "0<x<=10", "x>10"]),
    "tag_hi52": (True, ["-5<x<=0", "-15<x<=-5", "-30<x<=-15", "x<=-30"]),
    "tag_close_pos": (True, ["x<0.25", "0.25<=x<=0.75", "x>0.75"]),
    "tag_n202_d0": (True, ["x<=1", "2<=x<=3", "x>=4"]),
    "tag_q_before_close": (False, ["1", "0"]),
    "tag_weekday": (False, ["월", "화", "수", "목", "금"]),
    "tone_net": (False, ["x>median", "x<=median"]),
    "guid_raise": (False, ["1", "0"]),
}


def trade_table(trades: list[dict], rate: float = 0.22) -> pd.DataFrame:
    """끝난 매매(환율 있음) → 표: ticker, d0, ex_cost(원화 비용 뒤 세전 초과), ex_after(장부 세금 뒤 — 종목·QQQ 각각), hit_days."""
    from core import sw1

    rows = [t for t in trades if t["status"] == "closed" and t.get("fx_buy") and t.get("fx_sell")]
    if not rows:
        return pd.DataFrame(columns=["ticker", "d0", "ex_cost", "ex_after", "hit_days"])
    r = np.array([t["sell_px"] / t["buy_px"] - 1 for t in rows])
    q = np.array([t["q_sell"] / t["q_buy"] - 1 for t in rows])
    fb = np.array([t["fx_buy"] for t in rows], dtype=float)
    fs = np.array([t["fx_sell"] for t in rows], dtype=float)
    rk, qk = sw1.krw_after_cost(r, fb, fs), sw1.krw_after_cost(q, fb, fs)
    yr = np.array([int(t["sell_day"][:4]) for t in rows])
    ra, qa = rk - sw1.ledger_tax(rk, yr, rate), qk - sw1.ledger_tax(qk, yr, rate)
    return pd.DataFrame({"ticker": [t["ticker"] for t in rows], "d0": [str(t["d0"])[:10] for t in rows],
                         "ex_cost": rk - qk, "ex_after": ra - qa, "hit_days": [t.get("hit_days") for t in rows]})


def tag_frame(signals: list[dict], plans: dict) -> pd.DataFrame:
    """신호 꼬리표(pead_watch_signals) + 매도 조건 기록(pead_exits_plan) → (ticker, d0)별 분석 칸."""
    out = []
    for s in signals:
        k = (s["ticker"], str(s["d0"])[:10])
        pl = plans.get(k) or {}
        tone = (s["tag_lm_pos"] - s["tag_lm_neg"]) if s.get("tag_lm_pos") is not None and s.get("tag_lm_neg") is not None else None
        out.append({"ticker": k[0], "d0": k[1], **{c: s.get(c) for c in TAG_BINS if c.startswith("tag_")},
                    "tone_net": tone, "guid_raise": pl.get("guid_raise"), "prev_below_sma200": pl.get("prev_below_sma200")})
    cols = ["ticker", "d0", *[c for c in TAG_BINS if c.startswith("tag_")], "tone_net", "guid_raise", "prev_below_sma200"]
    return pd.DataFrame(out, columns=cols)


def _tag_tests(x: np.ndarray, frame: pd.DataFrame, p: Params) -> list[dict]:
    """탐색 꼬리표: 구간 vs 나머지 섞기 p → BH(FDR p.fdr), 연속형은 단조가 아니면 우연."""
    rows, tests = [], []
    for col, (cont, edges) in TAG_BINS.items():
        have = frame[col].notna().to_numpy()
        xs, vs = x[have], frame[col][have].to_numpy()
        if col == "tone_net":
            fv = vs.astype(float)
            med = float(np.median(fv)) if len(fv) else 0.0
            masks = [fv > med, fv <= med]
        else:
            masks = [np.array([in_bin(y if col == "tag_weekday" else float(y), e) for y in vs], dtype=bool) for e in edges]
        means = [float(xs[m].mean() * 100) if m.sum() else None for m in masks]
        row = {"tag": col, "bins": edges, "n": [int(m.sum()) for m in masks], "means": means, "tests": []}
        for e, m in zip(edges, masks[:1] if len(masks) == 2 else masks):
            if m.sum() < p.min_bin_n or len(xs) - m.sum() < p.min_bin_n:
                row["tests"].append({"bin": e, "p": None, "bh": False, "note": "표본 부족"})
                continue
            row["tests"].append({"bin": e, "p": perm_p(xs, m, p.iterations, p.seed), "bh": False, "note": ""})
            tests.append((len(rows), len(row["tests"]) - 1))
        filled = [v for v in means if v is not None]
        row["monotone"] = monotone(filled) if cont and len(filled) >= 2 else None
        rows.append(row)
    rej = bh_reject([rows[i]["tests"][j]["p"] for i, j in tests], p.fdr)
    for (i, j), rj in zip(tests, rej):
        chance = TAG_BINS[rows[i]["tag"]][0] and rows[i]["monotone"] is False
        rows[i]["tests"][j].update(bh=bool(rj) and not chance, note="우연(단조 아님)" if rj and chance else ("BH 통과" if rj else ""))
    return rows


def report(books: dict[str, list[dict]], signals: list[dict], plans: dict, start: str, p: Params = Params()) -> dict:
    """장부별 지표(세전·세후) → 짝 비교 ①A ②B vs V1(Holm) → 분석(전날 200일선, 꼬리표 BH). books = {'main','exit_a','exit_b'}."""
    T = {b: trade_table([t for t in tr if str(t["d0"])[:10] >= start]) for b, tr in books.items()}
    out: dict = {"books": {}, "pairs": {}, "analysis": {}}
    for b, d in T.items():
        if not len(d):
            out["books"][b] = {"n": 0}
            continue
        g = clusters(list(d["ticker"]), list(d["d0"]))
        m = {"pre_tax": metrics(d["ex_cost"].to_numpy(), g, p.iterations, p.seed),
             "after_tax": metrics(d["ex_after"].to_numpy(), g, p.iterations, p.seed)}
        if b == "exit_b":
            h = pd.to_numeric(d["hit_days"], errors="coerce").dropna()
            m.update(hit_rate=float(len(h) / len(d) * 100), hit_days_median=float(h.median()) if len(h) else None,
                     hit_days_mean=float(h.mean()) if len(h) else None)
        out["books"][b] = m
    ps = {}
    for b in ("exit_a", "exit_b"):
        j = T[b].merge(T["main"], on=["ticker", "d0"], suffixes=("", "_v1"))
        if not len(j):
            out["pairs"][b] = {"n": 0}
            continue
        dd = (j["ex_cost"] - j["ex_cost_v1"]).to_numpy(dtype=float)
        da = (j["ex_after"] - j["ex_after_v1"]).to_numpy(dtype=float)
        g = clusters(list(j["ticker"]), list(j["d0"]))
        lo, hi = cluster_boot_ci(dd, g, p.iterations, p.seed)
        ps[b] = sign_flip_p(dd, g, p.iterations, p.seed)
        out["pairs"][b] = {"n": int(len(dd)), "mean": float(dd.mean() * 100), "median": float(np.median(dd) * 100),
                           "ci_lo": lo * 100, "ci_hi": hi * 100, "p": ps[b], "mean_after_tax": float(da.mean() * 100)}
    for b, adj in zip(ps, holm(list(ps.values()))):
        pr = out["pairs"][b]
        pr["p_holm"] = adj
        pr["verdict"] = ("낫다" if pr["mean"] > 0 else "못하다") if adj < p.alpha else "차이 없음"
    v1 = T["main"].merge(tag_frame(signals, plans), on=["ticker", "d0"], how="left")
    if len(v1):
        x = v1["ex_cost"].to_numpy(dtype=float)
        have = v1["prev_below_sma200"].notna().to_numpy()
        below = pd.to_numeric(v1["prev_below_sma200"], errors="coerce").to_numpy(dtype=float) == 1
        xs, mk = x[have], below[have]
        out["analysis"]["sma200_prev"] = {
            "n_below": int(mk.sum()), "n_rest": int((~mk).sum()),
            "diff": float((xs[mk].mean() - xs[~mk].mean()) * 100) if 0 < mk.sum() < len(xs) else None,
            "p": perm_p(xs, mk, p.iterations, p.seed)}
        out["analysis"]["tags"] = _tag_tests(x, v1, p)
    out["counts"] = {b: int(len(d)) for b, d in T.items()}
    k = min(out["counts"].get("exit_a", 0), out["counts"].get("exit_b", 0))
    out["milestone"] = "확정 판단" if k >= 120 else ("첫 성적" if k >= 50 else "기록 중")
    return out
