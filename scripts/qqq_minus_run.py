"""QQQ 실수 빼기 전략(qqq-minus) 탐색 1회 실행 — 사전 등록 태그 뒤에만 수익을 계산한다.

    python scripts/qqq_minus_run.py inputs          # 매달 입력표(시가총액·빼기 이유) → data/cache/qqq_minus/inputs.parquet (수익 없음)
    python scripts/qqq_minus_run.py run             # 판정 + 참고 보고 + 엑셀 (무작위 1,000회)
    python scripts/qqq_minus_run.py run --quick     # 무작위 20회만(실행기 점검용)

시작할 때 (1) 태그 qqq-minus-prereg가 있는지, (2) 잠근 파일이 태그 내용과 같은지, (3) 잠근 캐시(확보율 표·K 표)의 SHA-256이 사전 등록 문서와 같은지 확인한다.
장부(원화 계좌·비용·세금·12월 공제 소진)는 mcap-growth 실행기 Book을 그대로 쓰고, 비중 맞추기(rebalance_w)만 더한다.
결과 숫자는 data/cache/qqq_minus/result.json(git 제외), 엑셀은 exports/qqq_minus/(git 제외).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mcap_growth_run as mr  # noqa: E402
import qqq_minus_check as qc  # noqa: E402
import rs26_zone3_check as z  # noqa: E402
from core import mcap_growth as mg  # noqa: E402
from core import qqq_minus as qm  # noqa: E402
from core import tax_harvest as th  # noqa: E402

TAG = "qqq-minus-prereg"
PREREG = ROOT / "docs" / "prereg" / "qqq_minus.md"
LOCKED = ["docs/prereg/qqq_minus.md", "core/qqq_minus.py", "scripts/qqq_minus_check.py", "scripts/mcap_growth_run.py",
          "core/mcap_growth.py", "core/tax_harvest.py"]
LOCKED_CACHE = [qc.COVER, qc.KFACTOR, qc.MEMBERS, qc.PRICE_META]
START = pd.Timestamp("2012-01-03")
END = pd.Timestamp("2021-12-31")
MID = pd.Timestamp("2016-12-30")
SEED, N_RANDOM = 20261030, 1000
INPUTS = qc.OUT / "inputs.parquet"
RESULT = qc.OUT / "result.json"
EXPORT = ROOT / "exports" / "qqq_minus"
QQQ = mr.QQQ
SMA_DAYS = 200


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_prereg() -> str:
    r = subprocess.run(["git", "rev-parse", "--verify", f"{TAG}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"태그 {TAG}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in LOCKED:
        if subprocess.run(["git", "diff", "--quiet", TAG, "--", f], cwd=ROOT).returncode != 0:
            raise SystemExit(f"{f}가 태그 {TAG} 뒤에 바뀌었습니다 — 멈춥니다")
    doc = PREREG.read_text(encoding="utf-8")
    for p in LOCKED_CACHE:
        if sha256(p) not in doc:
            raise SystemExit(f"{p.name}의 SHA-256이 사전 등록 문서와 다릅니다 — 멈춥니다")
    return r.stdout.strip()


# ── 입력표 (수익 없음) ────────────────────────────────────────────────────────


def build_inputs() -> None:
    """2012-01 ~ 2021-12 매달 첫 거래일: 명단 종목마다 시가총액·A(첫 거래일 504거래일)·B(주식 수 12개월 변화)·C(200일선·매출 성장)."""
    verify_prereg()
    inp = qc.Inputs()
    rows = []
    for d, mem in sorted(inp.members.items()):
        if d < START:
            continue
        caps = inp.mcaps(d)
        for t in mem:
            x = inp.px.get(t)
            below = None
            if x is not None and d in x.index:
                c = x.loc[:d, "close"].dropna()
                if len(c) >= SMA_DAYS:
                    below = bool(c.iloc[-1] < c.iloc[-SMA_DAYS:].mean())
            f = inp.facts.get(t)
            rev = f[f["tag"].isin(mg.REV_TAGS) & (f["unit"] == "USD")] if f is not None else None
            rows.append({"date": d, "ticker": t, "mcap": caps.get(t), "age_ok": inp.age_ok(t, d), "share_g": inp.share_growth(t, d),
                         "below_sma": below, "rev_g": mg.revenue_growth(rev, d) if rev is not None else None})
        print(f"[inputs] {d.date()} 명단 {len(mem)} · 시총 {len(caps)}", flush=True)
    df = pd.DataFrame(rows)
    df["age_ok"] = df["age_ok"].astype(object)
    df["below_sma"] = df["below_sma"].astype(object)
    df.to_parquet(INPUTS, index=False)


# ── 시장·장부 ────────────────────────────────────────────────────────────────


class Market:
    """mcap_growth_run.Market과 같은 속성(장부가 읽는 것)을 나스닥100 가격 캐시로 만든다. 열 마지막은 QQQ."""

    def __init__(self, px: dict[str, pd.DataFrame]):
        alld = qc.trading_days()
        self.days = alld[(alld >= START) & (alld <= END)]
        qq = pd.read_csv(z.QQQ_CSV, index_col=0, parse_dates=True)
        if qq.index.max() > END:
            raise z.ZoneGuardError("QQQ 파일에 2021-12-31 뒤 줄이 있습니다")
        cl = {t: d["close"] for t, d in px.items()}
        dv = {t: d["div"] for t, d in px.items()}
        cl[QQQ], dv[QQQ] = qq["close"].astype(float), qq["div"].fillna(0.0).astype(float)
        self.tickers = sorted(t for t in cl if t != QQQ) + [QQQ]
        self.ti = {t: j for j, t in enumerate(self.tickers)}
        cdf = pd.DataFrame(cl).reindex(columns=self.tickers)
        self.close = cdf.reindex(self.days).to_numpy()
        self.ffill = cdf.ffill().reindex(self.days).to_numpy()
        self.div = pd.DataFrame(dv).reindex(columns=self.tickers).reindex(self.days).fillna(0.0).to_numpy()
        has = ~np.isnan(self.close)
        self.last_idx = np.where(has.any(axis=0), len(self.days) - 1 - np.argmax(has[::-1], axis=0), -1)
        self.fx = mr.load_fx(self.days)
        d = self.days
        s = pd.Series(np.arange(len(d)), index=d)
        self.may_first = {int(i) for i in s.groupby([d.year, d.month]).min() if d[i].month == 5}
        self.year_last = {int(i) for i in s.groupby(d.year).max()}
        self.dec_exec = {int(np.searchsorted(d, th.deadline(d, y))): y for y in sorted(set(d.year)) if d[d.year == y][-1].month == 12}
        self.pos = {dd: int(i) for dd, i in s.items()}


class WBook(mr.Book):
    def rebalance_w(self, i: int, w: dict[int, float]) -> None:
        """목표 = 평가액(현금 포함) × 비중. 빠지는 종목·초과분 먼저 팔고, 모자란 종목을 남은 현금 안에서 비례해 산다(편도 비용)."""
        if not w:
            return
        tot = self.value_usd(i)
        for j in [j for j in self.hold if j not in w]:
            self.sell(i, j, self.hold[j][0])
        for j, wt in w.items():
            if j in self.hold:
                v = self.hold[j][0] * self.px(i, j)
                if v > tot * wt:
                    self.sell(i, j, (v - tot * wt) / self.px(i, j))
        need = {j: tot * wt - (self.hold[j][0] * self.px(i, j) if j in self.hold else 0.0) for j, wt in w.items()}
        need = {j: v for j, v in need.items() if v > 1e-9}
        tn = sum(need.values())
        if tn <= 0:
            return
        k = min(1.0, max(self.cash, 0.0) / tn)
        for j, v in need.items():
            self.buy(i, j, v * k)


def run_w(m: Market, plan: dict[int, dict[str, float]], cost=mr.COST, tax=True, record=False) -> dict:
    """plan = {체결일 위치: {종목: 비중}} (매달 첫 거래일). 시작일에 1억 원을 달러로 바꿔 첫 비중대로."""
    b = WBook(m, cost, tax, record)
    n = len(m.days)
    for i in range(n):
        if i == 0:
            b.cash = mr.INITIAL_KRW / m.fx[0]
        b.day(i)
        if i in plan:
            w = {m.ti[t]: v for t, v in plan[i].items() if t in m.ti and not np.isnan(m.close[i, m.ti[t]])}
            s = sum(w.values())
            b.rebalance_w(i, {j: v / s for j, v in w.items()} if s > 0 else {})
        if i in m.dec_exec and b.tax:
            b.harvest(i)
        b.end_day(i)
    final = b.finish(n - 1)
    years = (END - START).days / 365.25
    return {"final": final, "cagr": (final / mr.INITIAL_KRW) ** (1 / years) - 1, "book": b,
            "curve": pd.Series(b.curve, index=m.days) if record else None}


# ── 계획 ─────────────────────────────────────────────────────────────────────


def month_tables(m: Market) -> dict[int, pd.DataFrame]:
    f = pd.read_parquet(INPUTS)
    out = {}
    for d, g in f.groupby("date"):
        g = g[g["mcap"].notna()].set_index("ticker")
        out[m.pos[pd.Timestamp(d)]] = g
    if 0 not in out:
        raise SystemExit("시작일 입력이 없습니다")
    return out


def reasons(g: pd.DataFrame) -> dict[str, set[str]]:
    return {t: qm.drop_reasons(None if pd.isna(r.age_ok) else bool(r.age_ok), None if pd.isna(r.share_g) else float(r.share_g),
                               None if pd.isna(r.below_sma) else bool(r.below_sma), None if pd.isna(r.rev_g) else float(r.rev_g))
            for t, r in g.iterrows()}


def plans(tabs: dict[int, pd.DataFrame]) -> tuple[dict, list[dict]]:
    """판 이름 → {체결일: 비중}, 그리고 매달 뺀 종목 기록(빼기판 A∪B와 참고 C)."""
    use = {"replica": set(), "minus": {"A", "B"}, "a_only": {"A"}, "b_only": {"B"}, "ref_c": {"C"}, "ref_abc": {"A", "B", "C"}}
    out = {k: {} for k in use}
    log = []
    for i, g in sorted(tabs.items()):
        caps = g["mcap"].astype(float).to_dict()
        rs = reasons(g)
        base = qm.cap_weights(caps)
        for k, rules in use.items():
            drop = {t for t, r in rs.items() if r & rules}
            out[k][i] = qm.cap_weights(caps, drop)
        for t, r in rs.items():
            if r:
                log.append({"i": i, "종목": t, "이유": "".join(sorted(r)), "복제판 비중": base[t], "빼기판(A∪B)에서 뺌": bool(r & {"A", "B"}),
                            "첫 거래일 504일 넘음": g.at[t, "age_ok"], "주식 수 12개월 변화": g.at[t, "share_g"],
                            "200일선 아래": g.at[t, "below_sma"], "매출 4분기 성장": g.at[t, "rev_g"]})
    return out, log


def plan_random(tabs: dict[int, pd.DataFrame], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for i, g in sorted(tabs.items()):
        caps = g["mcap"].astype(float).to_dict()
        base = qm.cap_weights(caps)
        drop = {t for t, r in reasons(g).items() if r & {"A", "B"}}
        rd = qm.random_drop(base, len(drop), sum(base[t] for t in drop), rng)
        out[i] = qm.cap_weights(caps, rd)
    return out


_POOL: dict = {}


def _init(m, tabs):
    mr.START, mr.END = START, END
    _POOL.update(m=m, tabs=tabs)


def _rand(i: int) -> float:
    return run_w(_POOL["m"], plan_random(_POOL["tabs"], SEED + i))["cagr"]


# ── 실행 ─────────────────────────────────────────────────────────────────────


def main(quick: bool = False) -> None:
    tag_commit = verify_prereg()
    t0 = time.time()
    mr.START, mr.END = START, END
    px = qc.load_prices()
    m = Market(px)
    if m.days[0] != START or m.days[-1] != END:
        raise SystemExit("거래일 범위가 시작·끝과 다릅니다")
    tabs = month_tables(m)
    P, log = plans(tabs)
    res = {k: run_w(m, p, record=True) for k, p in P.items()}
    res["qqq"] = mr.run_book(m, {i: [] for i in tabs}, record=True, qqq=True)
    for k in ("replica", "minus"):
        res[f"{k}_zero"] = run_w(m, P[k], cost=0.0, tax=False, record=True)
        res[f"{k}_notax"] = run_w(m, P[k], tax=False, record=True)
    res["qqq_zero"] = mr.run_book(m, {i: [] for i in tabs}, cost=0.0, tax=False, record=True, qqq=True)
    res["qqq_notax"] = mr.run_book(m, {i: [] for i in tabs}, tax=False, record=True, qqq=True)
    print(f"[run] 장부 계산 {time.time() - t0:.0f}초", flush=True)
    n_rand = 20 if quick else N_RANDOM
    with ProcessPoolExecutor(max_workers=8, initializer=_init, initargs=(m, tabs)) as ex:
        rand = list(ex.map(_rand, range(n_rand), chunksize=10))
    print(f"[run] 무작위 {n_rand}회 {time.time() - t0:.0f}초", flush=True)

    M, R, Q = res["minus"], res["replica"], res["qqq"]
    r = np.array(rand)
    pct = 100.0 * (float((r < M["cagr"]).sum()) + 0.5 * float((r == M["cagr"]).sum())) / len(r)
    segs = {"2012~16": (mr.seg_cagr(M["curve"], START, MID), mr.seg_cagr(R["curve"], START, MID)),
            "2017~21": (mr.seg_cagr(M["curve"], MID, END), mr.seg_cagr(R["curve"], MID, END))}
    crit = {
        "1": bool(M["cagr"] > R["cagr"] and M["cagr"] > Q["cagr"]),
        "2": bool(pct >= 90.0),
        "3": bool(all(a > b for a, b in segs.values())),
        "4": bool(res["a_only"]["cagr"] >= R["cagr"] - 0.002 and res["b_only"]["cagr"] >= R["cagr"] - 0.002),
    }
    verdict = "기준 충족" if all(crit.values()) else "기준 미충족"
    # 추적 오차(세금 없음·비용 있음 곡선의 일별 수익 차 표준편차 × √252)
    rd = res["replica_notax"]["curve"].pct_change().dropna()
    qd = res["qqq_notax"]["curve"].pct_change().dropna()
    te = float((rd - qd).std() * np.sqrt(252))
    # 매달 뺀 종목 수·비중, 다음 달 수익
    lg = pd.DataFrame(log)
    keys = sorted(tabs)
    nxt = {a: b for a, b in zip(keys, keys[1:] + [len(m.days) - 1])}
    if len(lg):
        lg["날짜"] = [m.days[i].date() for i in lg["i"]]
        nr = []
        for i, t in zip(lg["i"], lg["종목"]):
            j, b = m.ti.get(t), nxt[i]
            if j is None:
                nr.append(None)
                continue
            p0, p1 = m.ffill[i, j], m.ffill[b, j]
            dvs = float(m.div[i + 1:b + 1, j].sum())
            nr.append(float((p1 + dvs) / p0 - 1) if p0 and not np.isnan(p0) else None)
        lg["다음 달 수익(배당 포함)"] = nr
    ab = lg[lg["빼기판(A∪B)에서 뺌"]] if len(lg) else lg
    per_month = ab.groupby("i").agg(n=("종목", "size"), w=("복제판 비중", "sum")).reindex(keys, fill_value=0)
    qret = {i: float(m.ffill[nxt[i], m.ti[QQQ]] / m.ffill[i, m.ti[QQQ]] - 1) for i in keys}
    out = {
        "tag_commit": tag_commit, "verdict": verdict, "criteria": crit, "n_random": n_rand,
        "cagr": {k: v["cagr"] for k, v in res.items()}, "final": {k: v["final"] for k, v in res.items()},
        "mdd": {k: mr.mdd(v["curve"]) for k, v in res.items() if v.get("curve") is not None},
        "random": {"pct": pct, "p50": float(np.percentile(r, 50)), "p90": float(np.percentile(r, 90)), "p95": float(np.percentile(r, 95))},
        "segments": segs, "tracking_error": te,
        "dropped": {"n_mean": float(per_month["n"].mean()), "n_min": int(per_month["n"].min()), "n_max": int(per_month["n"].max()),
                    "w_mean": float(per_month["w"].mean()), "w_max": float(per_month["w"].max()),
                    "months_with_drop": int((per_month["n"] > 0).sum()), "months": len(keys),
                    "by_reason": {k: int(v) for k, v in ab["이유"].value_counts().items()} if len(ab) else {}},
        "next_month_return": {"dropped_mean": float(ab["다음 달 수익(배당 포함)"].mean()) if len(ab) else None,
                              "qqq_mean_same_months": float(np.mean([qret[i] for i in ab["i"]])) if len(ab) else None},
        "taxes": {"minus": {int(k): v for k, v in M["book"].taxes.items()}, "replica": {int(k): v for k, v in R["book"].taxes.items()}},
        "n_names": {str(m.days[i].date()): len(P["replica"][i]) for i in keys},
        "yearly": {int(y): tuple(float(c[c.index.year == y].iloc[-1] / c[c.index <= f"{y - 1}-12-31"].iloc[-1] - 1)
                                 if (c.index <= f"{y - 1}-12-31").any() else None for c in (M["curve"], R["curve"], Q["curve"]))
                   for y in range(2013, 2022)},
    }
    RESULT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if not quick:
        EXPORT.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(EXPORT / "qqq_minus_dropped.xlsx") as w:
            lg.drop(columns=["i"]).to_excel(w, sheet_name="매달 뺀 종목", index=False)
            pm = per_month.reset_index()
            pm["날짜"] = [m.days[i].date() for i in pm["i"]]
            pm[["날짜", "n", "w"]].rename(columns={"n": "뺀 종목 수(A∪B)", "w": "뺀 비중 합"}).to_excel(w, sheet_name="매달 요약", index=False)
            pd.DataFrame([{"판": k, "세후 연복리": v["cagr"], "세후 청산액(원)": v["final"]} for k, v in res.items()]).to_excel(
                w, sheet_name="요약", index=False)
    print(json.dumps({k: out[k] for k in ("verdict", "criteria", "cagr", "mdd", "random", "segments", "tracking_error", "dropped")},
                     ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "inputs":
        build_inputs()
    else:
        main(quick="--quick" in sys.argv)
