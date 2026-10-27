"""실적 동반 시총 성장 전략(mcap-growth) 검증 구역 ③ 판정 실행 — 사전 등록 태그 뒤에만 수익을 계산한다.

    python scripts/mcap_growth_run.py inputs    # 분기별 입력표(점수·매출 성장·주식 수 변화·섹터) → data/cache/mcap_growth/inputs.parquet
    python scripts/mcap_growth_run.py run       # 판정 + 참고 보고 + 엑셀 (무작위 1,000회)
    python scripts/mcap_growth_run.py run --quick   # 무작위 20회만(실행기 점검용, 판정 아님)

시작할 때 (1) 태그 mcap-growth-prereg가 있는지, (2) 잠근 파일이 태그 내용과 같은지 확인하고 아니면 멈춘다.
명단·가격·재무는 scripts/mcap_growth_check.Inputs(구역 가드)로만 읽는다. 규칙·해석은 docs/prereg/mcap_growth.md 1장·3장 그대로다.
결과 숫자는 data/cache/mcap_growth/result.json(git 제외), 엑셀은 exports/mcap/(git 제외).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mcap_growth_check as mc  # noqa: E402
import rs26_zone3_check as z  # noqa: E402
from core import mcap_growth as mg  # noqa: E402
from core import tax_harvest as th  # noqa: E402

TAG = "mcap-growth-prereg"
LOCKED = ["docs/prereg/mcap_growth.md", "core/mcap_growth.py", "scripts/mcap_growth_check.py", "scripts/value_score_zone3_data.py",
          "scripts/value_score_zone3_check.py", "scripts/rs26_zone3_check.py"]
START = pd.Timestamp("2012-07-03")
END = pd.Timestamp("2021-12-31")
MID = pd.Timestamp("2016-12-30")
INITIAL_KRW = 100_000_000.0
DEDUCT, TAX, KEEP_DIV, COST = 2_500_000.0, 0.22, 0.85, 0.001
SEED, N_RANDOM = 20261027, 1000
FX_CSV = ROOT / "data" / "cache" / "shock_dip" / "fred_DEXKOUS.csv"
INPUTS = mc.OUT / "inputs.parquet"
RESULT = mc.OUT / "result.json"
EXPORT = ROOT / "exports" / "mcap"
QQQ = "__QQQ__"
LOOKS = {"12m": 252, "9m": 189, "6m": 126}


def verify_prereg() -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", f"{TAG}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"태그 {TAG}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}가 태그 {TAG} 뒤에 바뀌었습니다 — 멈춥니다")
    return r.stdout.strip()


# ── 입력표 ───────────────────────────────────────────────────────────────────


def build_inputs() -> None:
    """판정 가능 분기마다 가격 파일 있는 명단 종목의 입력(사전 등록 3-1). 수익은 계산하지 않는다."""
    verify_prereg()
    inp = mc.Inputs()
    cov = pd.read_csv(mc.COVER, encoding="utf-8-sig", parse_dates=["quarter"]).set_index("quarter")
    rows = []
    for q in inp.qs:
        priced, _ = inp.members(q)
        for t in sorted(priced):
            r = inp.row(q, t)
            for name, n in (("9m", 189), ("6m", 126)):
                d = inp.back(q, n)
                px = inp.data.price_on(t, d) if d is not None else None
                f = inp.facts.get(t)
                sh = mg.shares_asof(f[f["tag"].isin([mg.SHARES_DEI, mg.SHARES_WA])], d, inp.data.splits.get(t)) if (f is not None and d is not None) else None
                r[f"mcap_{name}"] = px * sh if (px is not None and sh is not None) else None
            rows.append({"quarter": q, "judgeable": bool(cov.at[q, "judgeable"]), **r})
        print(f"[inputs] {q.date()} {len(priced)}종목", flush=True)
    pd.DataFrame(rows).to_parquet(INPUTS, index=False)


@dataclass
class Quarter:
    q: pd.Timestamp
    scores: dict
    ok: dict
    ok_norev: dict
    sector: dict
    rev_g: dict
    med: dict
    share_g: dict
    pool: list


def quarters(look: str = "12m", rank_rise: bool = False) -> list[Quarter]:
    """판정 가능 분기마다 대상·점수·통과 여부 (사전 등록 3-1·3-2)."""
    f = pd.read_parquet(INPUTS)
    out = []
    for q, g in f[f["judgeable"]].groupby("quarter"):
        g = g.set_index("ticker")
        need = ["mcap_t", "mcap_s", "mcap_l", "share_g", "rev_g", "sector"]
        u = g.dropna(subset=need)
        if look != "12m":
            u = u.dropna(subset=[f"mcap_{look}"])
        base = u[f"mcap_{look}"] if look != "12m" else u["mcap_l"]
        if rank_rise:
            r_now = mg.mcap_ranks(u["mcap_s"].to_dict())
            r_then = mg.mcap_ranks(base.to_dict())
            sc = {t: float(r_then[t] - r_now[t]) for t in u.index}
        else:
            sc = {t: mg.score(u.at[t, "mcap_s"], base[t]) for t in u.index}
        sector = u["sector"].to_dict()
        rev = u["rev_g"].to_dict()
        med = mg.sector_medians(rev, sector)
        shg = u["share_g"].to_dict()
        ok = {t: mg.passes(rev[t], med.get(sector[t]), shg[t]) for t in u.index}
        ok2 = {t: mg.passes(rev[t], med.get(sector[t]), shg[t], use_rev=False) for t in u.index}
        out.append(Quarter(pd.Timestamp(q), sc, ok, ok2, sector, rev, med, shg, sorted(u.index)))
    return out


# ── 시장 ─────────────────────────────────────────────────────────────────────


def load_fx(days: pd.DatetimeIndex) -> np.ndarray:
    rows = []
    with open(FX_CSV, encoding="utf-8-sig") as f:
        next(f)
        for line in f:
            d, v = line.strip().split(",")[:2]
            ts = pd.Timestamp(d)
            if ts > END:
                break
            if ts >= pd.Timestamp("2011-01-01") and v not in ("", "."):
                rows.append((ts, float(v)))
    s = pd.Series(dict(rows)).sort_index()
    out = s.reindex(s.index.union(days)).ffill().reindex(days)
    if out.isna().any():
        raise RuntimeError("환율 빈 날을 채우지 못했습니다")
    return out.to_numpy()


class Market:
    """(거래일 × 종목) 종가·분배금 배열. 열 마지막은 QQQ."""

    def __init__(self, data, qs_all: list[pd.Timestamp]):
        import value_score_zone3_data as vd

        days = vd.nyse_days()
        self.days = days[(days >= START) & (days <= END)]
        qc = pd.read_csv(z.QQQ_CSV, index_col=0, parse_dates=True)
        if qc.index.max() > END:
            raise z.ZoneGuardError("QQQ 파일에 2021-12-31 뒤 줄이 있습니다")
        cl = {t: data.close[t] for t in data.close}
        cl[QQQ] = qc["close"].astype(float)
        dv = {t: data.div[t] for t in data.div}
        dv[QQQ] = qc["div"].fillna(0.0).astype(float)
        self.tickers = sorted(t for t in cl if t != QQQ) + [QQQ]
        self.ti = {t: j for j, t in enumerate(self.tickers)}
        cdf = pd.DataFrame(cl).reindex(columns=self.tickers)
        self.close = cdf.reindex(self.days).to_numpy()
        self.ffill = cdf.ffill().reindex(self.days).to_numpy()
        self.div = pd.DataFrame(dv).reindex(columns=self.tickers).reindex(self.days).fillna(0.0).to_numpy()
        has = ~np.isnan(self.close)
        self.last_idx = np.where(has.any(axis=0), len(self.days) - 1 - np.argmax(has[::-1], axis=0), -1)
        self.fx = load_fx(self.days)
        d = self.days
        s = pd.Series(np.arange(len(d)), index=d)
        ym = [d.year, d.month]
        self.may_first = {int(i) for i in s.groupby(ym).min() if d[i].month == 5}
        self.year_last = {int(i) for i in s.groupby(d.year).max()}
        self.dec_exec = {int(np.searchsorted(d, th.deadline(d, y))): y for y in sorted(set(d.year)) if d[d.year == y][-1].month == 12}
        # 신호일 → 체결일(다음 거래일) 위치
        import value_score_zone3_data as vd2

        alld = vd2.nyse_days()
        self.exec_of = {}
        for q in qs_all:
            nxt = alld[alld > q][0]
            if nxt in s.index:
                self.exec_of[q] = int(s[nxt])
        # 동일비중: 분기마다 신호일 명단 중 그날 종가 있는 종목
        self.ew_members = {}
        for q in qs_all:
            prior = [m for m in data.mes if m < q]
            me = max(prior) if prior else min(data.mes)
            self.ew_members[q] = sorted(t for t in data.mem[me] if t in data.close and data.price_on(t, q) is not None)


# ── 장부 ─────────────────────────────────────────────────────────────────────


@dataclass
class Book:
    m: Market
    cost: float = COST
    tax: bool = True
    record: bool = False
    hold: dict = field(default_factory=dict)  # 열 → [수량, 원화 취득가]
    cash: float = 0.0
    realized: dict = field(default_factory=dict)
    pending: tuple | None = None
    taxes: dict = field(default_factory=dict)
    pnl: dict = field(default_factory=dict)
    curve: list = field(default_factory=list)
    costs_usd: float = 0.0

    def px(self, i, j):
        return float(self.m.ffill[i, j])

    def buy(self, i, j, usd):
        if usd <= 1e-9:
            return
        q = usd / (self.px(i, j) * (1 + self.cost))
        self.costs_usd += usd - usd / (1 + self.cost)
        h = self.hold.setdefault(j, [0.0, 0.0])
        h[0] += q
        h[1] += usd * self.m.fx[i]
        self.cash -= usd

    def sell(self, i, j, q):
        h = self.hold[j]
        q = min(q, h[0])
        if q <= 0:
            return 0.0
        gross = q * self.px(i, j)
        usd = gross * (1 - self.cost)
        self.costs_usd += gross * self.cost
        avg = h[1] / h[0]
        gain = usd * self.m.fx[i] - avg * q
        y = self.m.days[i].year
        self.realized[y] = self.realized.get(y, 0.0) + gain
        self.pnl[j] = self.pnl.get(j, 0.0) + gain
        self.cash += usd
        h[0] -= q
        h[1] -= avg * q
        if h[0] <= 1e-12:
            del self.hold[j]
        return usd

    def value_usd(self, i):
        return self.cash + sum(h[0] * self.px(i, j) for j, h in self.hold.items())

    def value_liq(self, i):
        fx = self.m.fx[i]
        usd, gain = self.cash, 0.0
        for j, h in self.hold.items():
            u = h[0] * self.px(i, j) * (1 - self.cost)
            usd += u
            gain += u * fx - h[1]
        out = usd * fx
        if self.tax:
            out -= max(self.realized.get(self.m.days[i].year, 0.0) + gain - DEDUCT, 0.0) * TAX
            if self.pending is not None:
                out -= self.pending[1]
        return out

    def rebalance(self, i, cols: list[int]):
        """목표 = 평가액 ÷ n. 빠지는 종목·초과분 먼저 팔고, 모자란 종목을 남은 현금 안에서 산다."""
        if not cols:
            return
        tot = self.value_usd(i)
        tgt = tot / len(cols)
        keep = set(cols)
        for j in [j for j in self.hold if j not in keep]:
            self.sell(i, j, self.hold[j][0])
        for j in cols:
            if j in self.hold:
                v = self.hold[j][0] * self.px(i, j)
                if v > tgt:
                    self.sell(i, j, (v - tgt) / self.px(i, j))
        need = {j: tgt - (self.hold[j][0] * self.px(i, j) if j in self.hold else 0.0) for j in cols}
        need = {j: v for j, v in need.items() if v > 1e-9}
        tot_need = sum(need.values())
        if tot_need <= 0:
            return
        k = min(1.0, max(self.cash, 0.0) / tot_need)
        for j, v in need.items():
            self.buy(i, j, v * k)

    def day(self, i):
        """분배금 → 가격 끝 → 5월 세금."""
        for j, h in list(self.hold.items()):
            dv = self.m.div[i, j]
            if dv > 0:
                usd = h[0] * dv * KEEP_DIV
                self.cash += usd
                self.pnl[j] = self.pnl.get(j, 0.0) + usd * self.m.fx[i]
        for j in [j for j in self.hold if self.m.last_idx[j] < i]:
            self.sell(i, j, self.hold[j][0])
        if i in self.m.may_first and self.pending is not None:
            krw = self.pending[1]
            usd = krw / self.m.fx[i]
            while self.cash < usd - 1e-9 and self.hold:
                j = max(self.hold, key=lambda k: self.hold[k][0] * self.px(i, k))
                self.sell(i, j, min(self.hold[j][0], (usd - self.cash) / (self.px(i, j) * (1 - self.cost))))
            self.cash -= usd
            self.pending = None

    def harvest(self, i):
        y = self.m.days[i].year
        left = DEDUCT - self.realized.get(y, 0.0)
        fx = self.m.fx[i]
        order = sorted(self.hold, key=lambda k: -(self.hold[k][0] * self.px(i, k) * (1 - self.cost) * fx - self.hold[k][1]))
        for j in order:
            if left <= 1.0:
                break
            h = self.hold[j]
            g = self.px(i, j) * (1 - self.cost) * fx - h[1] / h[0]
            if g <= 0:
                continue
            q = min(left / g, h[0])
            self.sell(i, j, q)
            self.buy(i, j, min(q * self.px(i, j) * (1 + self.cost), max(self.cash, 0.0)))
            left -= q * g

    def end_day(self, i):
        if i in self.m.year_last and i != len(self.m.days) - 1:
            y = self.m.days[i].year
            tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.tax else 0.0
            self.taxes[y] = tax
            if tax > 0:
                self.pending = (y, tax)
        if self.record:
            self.curve.append(self.value_liq(i))

    def finish(self, i):
        for j in list(self.hold):
            self.sell(i, j, self.hold[j][0])
        y = self.m.days[i].year
        tax = max(self.realized.get(y, 0.0) - DEDUCT, 0.0) * TAX if self.tax else 0.0
        self.taxes[y] = tax
        final = self.cash * self.m.fx[i] - tax - (self.pending[1] if self.pending else 0.0)
        if self.record:
            self.curve[-1] = final
        return final


def run_book(m: Market, plan: dict[int, list[str]], cost=COST, tax=True, record=False, qqq=False) -> dict:
    """plan = {체결일 위치: 종목 목록}. qqq=True면 QQQ 100%(분배금은 체결일마다 QQQ로)."""
    b = Book(m, cost, tax, record)
    n = len(m.days)
    for i in range(n):
        if i == 0:
            b.cash = INITIAL_KRW / m.fx[0]
        b.day(i)
        if i == 0 or i in plan:
            if qqq:
                b.buy(i, m.ti[QQQ], max(b.cash, 0.0))
            else:
                cols = [m.ti[t] for t in plan.get(i, []) if t in m.ti and not np.isnan(m.close[i, m.ti[t]])]
                b.rebalance(i, cols)
        if i in m.dec_exec and b.tax:
            b.harvest(i)
        b.end_day(i)
    final = b.finish(n - 1)
    years = (END - START).days / 365.25
    return {"final": final, "cagr": (final / INITIAL_KRW) ** (1 / years) - 1, "book": b,
            "curve": pd.Series(b.curve, index=m.days) if record else None}


# ── 계획(분기별 종목) ─────────────────────────────────────────────────────────


def plan_strategy(m: Market, qs: list[Quarter], mode: str = "main", p: mg.Params = mg.Params()) -> tuple[dict, list[dict]]:
    """전략 계획 {체결일: 종목} + 엑셀용 선택 기록. mode: main | norev(c 없음)."""
    held: list[str] = []
    plan, log = {}, []
    for Q in qs:
        i = m.exec_of.get(Q.q)
        if i is None:
            continue
        ok = Q.ok if mode == "main" else Q.ok_norev
        sel = mg.select(Q.scores, ok, Q.sector, held, p)
        rank = {t: k + 1 for k, t in enumerate(mg.rank_order(Q.scores))}
        for t in sel:
            log.append({"분기": Q.q.date(), "종목": t, "섹터": Q.sector[t], "점수": Q.scores[t], "점수 순위": rank[t],
                        "매출 성장": Q.rev_g[t], "섹터 중앙값": Q.med.get(Q.sector[t]), "주식 수 변화(12개월)": Q.share_g[t],
                        "유지/새로": "유지" if t in held else "새로", "대상 수": len(Q.pool)})
        plan[i] = sel
        held = sel
    return plan, log


def plan_ew(m: Market) -> dict:
    return {i: m.ew_members[q] for q, i in m.exec_of.items()}


def plan_random(m: Market, qs: list[Quarter], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return {m.exec_of[Q.q]: mg.select_random(Q.pool, Q.sector, rng) for Q in qs if Q.q in m.exec_of}


_POOL: dict = {}


def _init(m, qs):
    _POOL.update(m=m, qs=qs)


def _rand(i: int) -> float:
    return run_book(_POOL["m"], plan_random(_POOL["m"], _POOL["qs"], SEED + i))["cagr"]


# ── 성적 ─────────────────────────────────────────────────────────────────────


def mdd(c: pd.Series) -> float:
    return float((c / c.cummax() - 1).min())


def seg_cagr(c: pd.Series, a, b) -> float:
    a, b = c.index[c.index <= pd.Timestamp(a)][-1], c.index[c.index <= pd.Timestamp(b)][-1]
    return (c[b] / c[a]) ** (365.25 / (b - a).days) - 1


def seg_ret(c: pd.Series, a, b) -> float:
    a, b = c.index[c.index <= pd.Timestamp(a)][-1], c.index[c.index <= pd.Timestamp(b)][-1]
    return c[b] / c[a] - 1


def main(quick: bool = False) -> None:
    import value_score_zone3_data as vd

    tag_commit = verify_prereg()
    t0 = time.time()
    data = vd.Data()
    qs_all = mc.quarter_starts()
    m = Market(data, qs_all)
    judge_q = {k: quarters(k) for k in LOOKS}
    qs = judge_q["12m"]
    first = min(Q.q for Q in qs)
    if m.exec_of[first] != 0:
        raise SystemExit(f"첫 판정 가능 분기 체결일이 시작일과 다릅니다: {first.date()}")
    res = {}
    plan_main, log = plan_strategy(m, qs)
    res["strategy"] = run_book(m, plan_main, record=True)
    res["qqq"] = run_book(m, {i: [] for i in plan_main}, record=True, qqq=True)
    res["ew"] = run_book(m, plan_ew(m), record=True)
    for k in ("9m", "6m"):
        res[f"strategy_{k}"] = run_book(m, plan_strategy(m, judge_q[k])[0], record=True)
    res["ref_norev"] = run_book(m, plan_strategy(m, qs, "norev")[0], record=True)
    res["ref_rankrise"] = run_book(m, plan_strategy(m, quarters("12m", rank_rise=True))[0], record=True)
    res["strategy_zero"] = run_book(m, plan_main, cost=0.0, tax=False)
    res["strategy_notax"] = run_book(m, plan_main, tax=False)
    res["qqq_zero"] = run_book(m, {i: [] for i in plan_main}, cost=0.0, tax=False, qqq=True)
    res["qqq_notax"] = run_book(m, {i: [] for i in plan_main}, tax=False, qqq=True)
    res["ew_zero"] = run_book(m, plan_ew(m), cost=0.0, tax=False)
    res["ew_notax"] = run_book(m, plan_ew(m), tax=False)
    print(f"[run] 장부 계산 {time.time() - t0:.0f}초", flush=True)
    n_rand = 20 if quick else N_RANDOM
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(m, qs)) as ex:
        rand = list(ex.map(_rand, range(n_rand), chunksize=10))
    print(f"[run] 무작위 {n_rand}회 {time.time() - t0:.0f}초", flush=True)

    S, Qq, E = res["strategy"], res["qqq"], res["ew"]
    r = np.array(rand)
    pct = 100.0 * (float((r < S["cagr"]).sum()) + 0.5 * float((r == S["cagr"]).sum())) / len(r)
    ms, mq = mdd(S["curve"]), mdd(Qq["curve"])
    segs = {"2012~16": (seg_cagr(S["curve"], START, MID), seg_cagr(Qq["curve"], START, MID)),
            "2017~21": (seg_cagr(S["curve"], MID, END), seg_cagr(Qq["curve"], MID, END))}
    rob = {k: (res[f"strategy_{k}"]["cagr"], res[f"strategy_{k}"]["cagr"] > Qq["cagr"] and res[f"strategy_{k}"]["cagr"] > E["cagr"])
           for k in ("6m", "9m")}
    crit = {
        "1": S["cagr"] > Qq["cagr"] and S["cagr"] > E["cagr"],
        "2": pct >= 95.0,
        "3": ms >= mq - 0.05,
        "4": all(v[1] for v in rob.values()),
        "5": all(a > b for a, b in segs.values()),
        "6": pct >= 99.0,
    }
    verdict = "PASS" if all(crit.values()) else "FAIL"
    # 표준 보고
    windows = {"2018-10-01~12-24 하락": ("2018-10-01", "2018-12-24"), "2020-02-19~03-23 급락": ("2020-02-19", "2020-03-23"),
               "2020-03-23~06-30 반등": ("2020-03-23", "2020-06-30")}
    win = {k: (seg_ret(S["curve"], a, b), seg_ret(Qq["curve"], a, b)) for k, (a, b) in windows.items()}
    sec_max = []
    for Q in qs:
        i = m.exec_of[Q.q]
        sel = plan_main[i]
        if sel:
            sec_max.append(pd.Series([Q.sector[t] for t in sel]).value_counts().iloc[0] / len(sel))
    pnl = pd.Series({m.tickers[j]: v for j, v in S["book"].pnl.items()})
    top5 = pnl.sort_values(ascending=False).head(5)
    out = {
        "tag_commit": tag_commit, "verdict": verdict, "criteria": crit, "n_random": n_rand,
        "cagr": {k: v["cagr"] for k, v in res.items()}, "final": {k: v["final"] for k, v in res.items()},
        "mdd": {k: mdd(v["curve"]) for k, v in res.items() if v.get("curve") is not None},
        "random": {"pct": pct, "p50": float(np.percentile(r, 50)), "p95": float(np.percentile(r, 95)), "p99": float(np.percentile(r, 99))},
        "segments": segs, "robust": rob, "windows": win,
        "concentration": {"sector_max_mean": float(np.mean(sec_max)), "sector_max_max": float(np.max(sec_max)),
                          "top5": {k: float(v) for k, v in top5.items()}, "top5_share": float(top5.sum() / pnl.sum()) if pnl.sum() else None,
                          "pnl_total": float(pnl.sum())},
        "taxes": {int(k): v for k, v in S["book"].taxes.items()}, "costs_usd": S["book"].costs_usd,
        "n_names": {str(Q.q.date()): len(plan_main[m.exec_of[Q.q]]) for Q in qs},
        "pool_size": {str(Q.q.date()): len(Q.pool) for Q in qs},
        "yearly": {int(y): (float(S["curve"][S["curve"].index.year == y].iloc[-1] / S["curve"][S["curve"].index <= f"{y - 1}-12-31"].iloc[-1] - 1)
                            if (S["curve"].index <= f"{y - 1}-12-31").any() else None,
                            float(Qq["curve"][Qq["curve"].index.year == y].iloc[-1] / Qq["curve"][Qq["curve"].index <= f"{y - 1}-12-31"].iloc[-1] - 1)
                            if (Qq["curve"].index <= f"{y - 1}-12-31").any() else None) for y in range(2013, 2022)},
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not quick:
        EXPORT.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(EXPORT / "mcap_growth_selections.xlsx") as w:
            pd.DataFrame(log).to_excel(w, sheet_name="분기별 15종목", index=False)
            pd.DataFrame([{"판": k, "세후 연복리": v["cagr"], "세후 청산액(원)": v["final"]} for k, v in res.items()]).to_excel(w, sheet_name="요약", index=False)
    print(json.dumps({k: out[k] for k in ("verdict", "criteria", "cagr", "mdd", "random", "segments", "robust")}, ensure_ascii=False,
                     indent=1, default=float))


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "inputs":
        build_inputs()
    else:
        main(quick="--quick" in sys.argv)
