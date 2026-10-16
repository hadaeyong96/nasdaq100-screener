"""검증 구역 ③ 판정 실행 — RS26(6개월 강세, 5-3)·ACCEL(한 달 가속, 6-3). 사전 등록 태그 뒤에만 수익을 계산한다.

    python scripts/rs26_zone3_run.py rs26     # 태그 rs26-prereg, docs/prereg/rs26_zone3.md
    python scripts/rs26_zone3_run.py accel    # 태그 accel-prereg, docs/prereg/accel_zone3.md

시작할 때 (1) 해당 태그가 있는지, (2) 잠근 파일(사전 등록 문서·이 스크립트·데이터 스크립트·core 함수)이 태그 내용과 같은지
확인하고 아니면 멈춘다. 가격은 scripts/rs26_zone3_check.read_prices_guarded(2021-12-31 넘는 줄에서 읽기 멈춤, 구역 밖 종목
거부)로만 읽는다. 체결·재조정·비용·세금·분배금은 가상 장부와 같은 core.momentum_book.process_day를 쓴다.
결과 숫자는 data/cache/rs26/result_{kind}.json(git 제외)에 남기고 사람이 docs/results/에 정리한다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import rs26_zone3_check as z  # noqa: E402
from core import accel  # noqa: E402
from core import momentum_book as mb  # noqa: E402
from data.market_calendar import trading_days_between  # noqa: E402

END = pd.Timestamp("2021-12-31")
INITIAL = 100_000.0
TAX = 22.0
WITHHOLD = 15.0
SEED = 20261015
N_RANDOM = 1000
EXCLUDED = {"HSH", "WYND", "EP"}  # 결정 3 점프 점검 제외 (docs/prereg/rs26_zone3.md 1장)
SPECS = {
    "rs26": {"tag": "rs26-prereg", "prereg": "docs/prereg/rs26_zone3.md",
             "locked": ["docs/prereg/rs26_zone3.md", "scripts/rs26_zone3_run.py", "scripts/rs26_zone3_check.py",
                        "core/momentum_book.py"]},
    "accel": {"tag": "accel-prereg", "prereg": "docs/prereg/accel_zone3.md",
              "locked": ["docs/prereg/accel_zone3.md", "scripts/rs26_zone3_run.py", "scripts/rs26_zone3_check.py",
                         "core/momentum_book.py", "core/accel.py"]},
}


# ── 사전 등록 확인 ─────────────────────────────────────────────────────


def verify_prereg(kind: str) -> str:
    """태그가 있고 잠근 파일이 태그 내용과 같은지. 출력: 태그 커밋. 아니면 SystemExit."""
    spec = SPECS[kind]
    tag = spec["tag"]
    r = subprocess.run(["git", "rev-parse", "--verify", f"{tag}^{{commit}}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"[{kind}] 태그 {tag}가 없습니다 — 사전 등록 전에는 수익을 계산하지 않습니다")
    for f in spec["locked"]:
        d = subprocess.run(["git", "diff", "--quiet", tag, "--", f], cwd=ROOT)
        if d.returncode != 0:
            raise SystemExit(f"[{kind}] {f}가 태그 {tag} 뒤에 바뀌었습니다 — 멈춥니다")
    return r.stdout.strip()


# ── 데이터 ─────────────────────────────────────────────────────────────


def nyse_days() -> pd.DatetimeIndex:
    return trading_days_between(date(2011, 6, 1), END.date())


def monthly_signals() -> list[pd.Timestamp]:
    """RS26: 2012-01 ~ 2021-11 매월 마지막 거래일 (2021-12 신호는 체결일이 2022라 체결하지 않음, 결정 1)."""
    return [d for d in z.signal_days() if d < pd.Timestamp("2021-12-01")]


def cycle_signals(step: int = 20) -> list[pd.Timestamp]:
    """ACCEL: 2012-01-03부터 20거래일마다, 체결일(다음 거래일)이 2021-12-31 이하인 신호까지 (결정 2)."""
    days = trading_days_between(date(2012, 1, 3), END.date())
    return [d for d in days[::step] if d < END]


class Zone:
    """구역 ③ 명단·가격 (가드 읽기). members(d) — 신호일 d 기준 구역 ③ 종목."""

    def __init__(self, signal_dates: list[pd.Timestamp], monthly: bool):
        self.fja = z.load_fja()
        self.sp100_plain, _, self.sp100 = z.load_sp100()
        self.cur = {z.norm(t) for t in pd.read_csv(z.ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"]}
        self.aliases = z.alias_map()
        self.gap = pd.read_csv(z.SP500 / "final_gap.csv", encoding="utf-8-sig").set_index("ticker")
        self.renames = {z.norm(r["old"]): z.norm(r["final_ticker"])
                        for _, r in pd.read_csv(z.SP500 / "renames.csv", encoding="utf-8-sig").iterrows()}
        self.monthly = monthly
        self.mem = {d: self._members(d) for d in signal_dates}
        allowed = set().union(*self.mem.values())
        self.px, self.dv = {}, {}
        for t in sorted(allowed):
            if t in EXCLUDED:
                continue
            p = z.price_path(t, self.gap, self.renames) if t in self.gap.index else None
            if p is None or not p.exists():
                continue
            df = z.read_prices_guarded(t, p, allowed)
            if df.empty:
                continue
            self.px[t], self.dv[t] = df["close"], df["div"]
        self.last_date = {t: s.index.max() for t, s in self.px.items()}

    def _sp100_for(self, d: pd.Timestamp) -> set[str]:
        key = d.strftime("%Y-%m")
        out = set(self.sp100.get(key) or set())
        if not self.monthly:  # 달 중간 신호: 전달 말 명단도 함께 뺀다(더 많이 빼는 쪽)
            prev = (d.replace(day=1) - pd.Timedelta(days=1)).strftime("%Y-%m")
            out |= set(self.sp100_plain.get(prev) or set())
        return out

    def _members(self, d: pd.Timestamp) -> set[str]:
        sp = self.fja.loc[:d].iloc[-1]["tickers"]
        excl = z.ndx_on(d, self.cur) | self._sp100_for(d)
        return {t for t in sp if t not in excl and not (self.aliases.get(t, {t}) & excl)}


def load_qqq() -> tuple[pd.Series, pd.Series]:
    q = pd.read_csv(z.QQQ_CSV, index_col=0, parse_dates=True)
    if q.index.max() > END:
        raise z.ZoneGuardError("QQQ 파일에 2021-12-31 뒤 줄이 있습니다")
    q = q[q.index >= z.ZONE_FIRST]
    return q["close"].astype(float), q["div"].fillna(0.0).astype(float)


# ── 장부 하나 돌리기 ─────────────────────────────────────────────────────


class Mat:
    """빠른 조회용 (날짜 × 종목) 종가·분배금 배열."""

    def __init__(self, zone: Zone, days: pd.DatetimeIndex):
        self.days = days
        self.di = {d: i for i, d in enumerate(days)}
        self.tickers = sorted(zone.px)
        self.ti = {t: j for j, t in enumerate(self.tickers)}
        self.close = pd.DataFrame({t: zone.px[t] for t in self.tickers}).reindex(days).to_numpy()
        self.div = pd.DataFrame({t: zone.dv[t] for t in self.tickers}).reindex(days).fillna(0.0).to_numpy()
        self.last = {t: zone.last_date[t] for t in self.tickers}
        self.last_close = {t: float(zone.px[t].iloc[-1]) for t in self.tickers}


def simulate(mat: Mat, signals: dict[pd.Timestamp, list[dict]], cost_pct: float, tax_pct: float = TAX) -> dict:
    """신호일마다 targets로 다시 맞추는 장부 (core.momentum_book.process_day 그대로).

    입력: signals({신호일: [{ticker, score_pp}] — 이미 고른 목록, 비중은 1/len}), cost_pct(편도 %)
    출력: {"curve": Series(일별 총자산, 첫 신호일부터), "final": 끝 청산·세금 뒤 값, "tax": 세금 합, "rebalances": n}
    가격이 끝난 보유 종목(그 뒤 종가 없음)은 마지막 종가로 평가하고 다음 교체일에 그 값에 판다(사전 등록 해석).
    """
    first = min(signals)
    days = [d for d in mat.days if first <= d <= END]
    year_ends = {d for d in pd.Series(days, index=days).groupby(pd.DatetimeIndex(days).year).max() if d < END}
    book = mb.start_book(mb.new_book("zone3", first), first, INITIAL)
    vals = []
    for d in days:
        i = mat.di[d]
        need = set(book["positions"]) | {o["ticker"] for o in book.get("carry_sells", [])}
        if book.get("pending"):
            need |= {x["ticker"] for x in book["pending"]["targets"]}
        closes, divs = {}, {}
        for t in need:
            j = mat.ti.get(t)
            if j is None:
                continue
            c = mat.close[i, j]
            if not np.isnan(c):
                closes[t] = float(c)
            elif d > mat.last[t]:
                closes[t] = mat.last_close[t]  # 가격이 끝난 종목: 마지막 종가로 평가·처분
            if t in book["positions"] and mat.div[i, j] > 0:
                divs[t] = float(mat.div[i, j])
        sig = signals.get(d)
        top_n = len(book["pending"]["targets"]) if book.get("pending") else (len(sig) if sig else 10)
        params = mb.MomentumParams(name="zone3", top_n=max(top_n, 1), min_price=0.0, commission_pct=cost_pct,
                                   tax_rate_pct=tax_pct, dividend_withholding_pct=WITHHOLD)
        signal = {"ok": True, "targets": sig} if sig is not None else None
        book, row, _ = mb.process_day(book, d, closes=closes, dividends=divs, splits={}, signal=signal,
                                      universe=list(need | {x["ticker"] for x in (sig or [])}), params=params,
                                      year_end=d in year_ends)
        vals.append(row["value"])
    curve = pd.Series(vals, index=pd.DatetimeIndex(days))
    # 끝: 2021-12-31 종가에 전량 청산(수수료) → 그해 실현 순이익 × 세율
    c = cost_pct / 100.0
    i = mat.di[END]
    for t in list(book["positions"]):
        j = mat.ti[t]
        px = mat.close[i, j] if not np.isnan(mat.close[i, j]) else mat.last_close[t]
        mb._sell(book, END, t, book["positions"][t]["qty"], float(px), 0.0, c, "끝 청산", END, "END", [])
    tax = max(book["realized_ytd"], 0.0) * tax_pct / 100.0
    return {"curve": curve, "final": book["cash"] - tax, "tax": book["tax_cum"] + tax, "rebalances": book["rebalances"]}


def simulate_qqq(qc: pd.Series, qd: pd.Series, first_signal: pd.Timestamp, cost_pct: float, tax_pct: float = TAX) -> dict:
    """QQQ 그냥 보유: 첫 체결일(첫 신호 다음 거래일) 종가에 사서 끝까지, 분배금 85% 현금, 끝 청산·세금."""
    c = cost_pct / 100.0
    days = qc.index[(qc.index >= first_signal) & (qc.index <= END)]
    fill = days[1]
    cash, qty, basis, vals = INITIAL, 0, 0.0, []
    for d in days:
        if d == fill:
            qty = int(cash // (qc[d] * (1 + c)))
            basis = qty * qc[d] * (1 + c)
            cash -= basis
        if qty and qd.get(d, 0.0) > 0 and d > fill:
            cash += qty * qd[d] * (1 - WITHHOLD / 100.0)
        vals.append(cash + qty * qc[d])
    proceeds = qty * qc[END] * (1 - c)
    tax = max(proceeds - basis, 0.0) * tax_pct / 100.0
    return {"curve": pd.Series(vals, index=days), "final": cash + proceeds - tax, "tax": tax}


# ── 지표·판정 ───────────────────────────────────────────────────────────


def cagr(v0: float, v1: float, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    return ((v1 / v0) ** (365.25 / (d1 - d0).days) - 1.0) * 100.0


def mdd(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1.0).min() * 100.0)


def slice_cagr(curve: pd.Series, d0: pd.Timestamp, d1: pd.Timestamp) -> float:
    """곡선을 날짜로 잘라 연환산 (결정 3: 다시 시작하지 않음). 시작값 = d0 이하 마지막 값."""
    a = float(curve[curve.index <= d0].iloc[-1])
    b = float(curve[curve.index <= d1].iloc[-1])
    return cagr(a, b, curve.index[curve.index <= d0][-1], curve.index[curve.index <= d1][-1])


def yearly(curve: pd.Series, start: pd.Timestamp) -> dict[int, float]:
    """달력 연도 수익률(%) — 그해 마지막 값 ÷ 전해 마지막 값(2012는 첫 신호일 값) − 1."""
    out, prev = {}, float(curve.iloc[0])
    for y in range(start.year, END.year + 1):
        yc = curve[curve.index.year == y]
        if yc.empty:
            continue
        out[y] = (float(yc.iloc[-1]) / prev - 1.0) * 100.0
        prev = float(yc.iloc[-1])
    return out


def judge(s: dict, q: dict, ew: dict, rand_cagrs: list[float], first: pd.Timestamp) -> dict:
    """사전 등록 합격 기준 6개 (반올림 전 값)."""
    cs, cq, ce = (cagr(INITIAL, x["final"], first, END) for x in (s, q, ew))
    r = np.array(rand_cagrs)
    pct = 100.0 * (float((r < cs).sum()) + 0.5 * float((r == cs).sum())) / len(r)
    mid = (pd.Timestamp("2016-12-30"), pd.Timestamp("2016-12-30"))
    sub = {"2012~2016": (slice_cagr(s["curve"], first, mid[0]), slice_cagr(q["curve"], first, mid[0])),
           "2017~2021": (slice_cagr(s["curve"], mid[1], END), slice_cagr(q["curve"], mid[1], END))}
    ys, yq = yearly(s["curve"], first), yearly(q["curve"], first)
    years_beat = sum(1 for y in ys if ys[y] - yq[y] > 0)
    ms, mq = mdd(s["curve"]), mdd(q["curve"])
    crit = [
        {"id": 1, "name": "세후 연복리 > QQQ", "ok": cs > cq, "value": cs - cq},
        {"id": 2, "name": "세후 연복리 > 구역 EW", "ok": cs > ce, "value": cs - ce},
        {"id": 3, "name": "무작위 99백분위 이상", "ok": pct >= 99.0, "value": pct},
        {"id": 4, "name": "두 구간 모두 QQQ 초과 > 0", "ok": all(a - b > 0 for a, b in sub.values()),
         "value": {k: a - b for k, (a, b) in sub.items()}},
        {"id": 5, "name": "10년 중 6년 이상 QQQ 초과 > 0", "ok": years_beat >= 6, "value": years_beat},
        {"id": 6, "name": "최대 낙폭 − QQQ ≥ −15%p", "ok": ms - mq >= -15.0, "value": ms - mq},
    ]
    return {"cagr": {"strategy": cs, "qqq": cq, "ew": ce}, "random_pct": pct,
            "random_cagr_p50": float(np.median(r)), "random_cagr_p99": float(np.percentile(r, 99)),
            "sub": sub, "years": {y: (ys[y], yq[y]) for y in ys}, "years_beat": years_beat,
            "mdd": {"strategy": ms, "qqq": mq}, "criteria": crit, "pass": all(c["ok"] for c in crit)}


# ── 신호 ────────────────────────────────────────────────────────────────


def build_signals(kind: str, zone: Zone, qc: pd.Series, sig_days: list[pd.Timestamp]) -> tuple[dict, dict]:
    """출력: (전략 신호 {d: 상위 10}, 대상 {d: 그날 대상 전체 [{ticker, score_pp}]})."""
    hist = {t: s for t, s in zone.px.items()}
    strat, elig = {}, {}
    for d in sig_days:
        uni = sorted(t for t in zone.mem[d] if t in hist)
        if kind == "rs26":
            r = mb.rank(hist, qc, d, uni, mb.MomentumParams(name="rs26", top_n=10, lookback_days=126, min_price=0.0))
            ranked = r["ranked"]
        else:
            r = accel.rank(hist, qc, d, uni, accel.AccelParams(min_price=0.0))
            ranked = [{**x, "score_pp": x["score"] * 100.0} for x in r["ranked"]]
        if not r["ok"]:
            raise RuntimeError(f"{d.date()}: QQQ 가격이 모자라 신호를 계산하지 못했습니다")
        elig[d] = [{"ticker": x["ticker"], "score_pp": x["score_pp"]} for x in ranked]
        strat[d] = elig[d][:10]
    return strat, elig


def coverage(kind: str, zone: Zone, mat: Mat, qc: pd.Series, sig_days: list[pd.Timestamp]) -> dict:
    """종목×신호 확보율: 구역 ③ 소속 중 점수 계산에 필요한 날짜·체결일 종가가 모두 있는 비율 (결정 2, 제외 3종목 반영)."""
    qd = qc.index
    rows = []
    for d in sig_days:
        pos = qd.get_loc(d)
        need = [qd[pos - 126], d] if kind == "rs26" else [qd[pos - 59], qd[pos - 40], qd[pos - 20], d]
        need.append(qd[pos + 1])
        ok = 0
        for t in zone.mem[d]:
            j = mat.ti.get(t)
            if j is not None and all(not np.isnan(mat.close[mat.di[x], j]) for x in need):
                ok += 1
        rows.append((d, len(zone.mem[d]), ok))
    df = pd.DataFrame(rows, columns=["d", "n", "ok"])
    early = df[df["d"] <= pd.Timestamp("2013-06-30")]
    low = df.loc[(df["ok"] / df["n"]).idxmin()]
    return {"overall_pct": df["ok"].sum() / df["n"].sum() * 100.0, "lowest_pct": float(low["ok"] / low["n"] * 100.0),
            "lowest_date": str(low["d"].date()), "early_pct": early["ok"].sum() / early["n"].sum() * 100.0,
            "signals": len(df), "members_min": int(df["n"].min()), "members_max": int(df["n"].max()),
            "eligible_min": int(df["ok"].min()), "eligible_max": int(df["ok"].max())}


_POOL: dict = {}


def _pool_init(elig: dict, mat: Mat, cost: float) -> None:
    _POOL.update(elig=elig, mat=mat, cost=cost)


def _random_run(i: int) -> float:
    """무작위 회차 i: 신호일마다 그날 대상에서 10종목을 고르게 뽑는다(시드 SEED + i)."""
    elig, mat, cost = _POOL["elig"], _POOL["mat"], _POOL["cost"]
    rng = np.random.default_rng(SEED + i)
    sig = {d: [lst[k] for k in sorted(rng.choice(len(lst), size=min(10, len(lst)), replace=False))] for d, lst in elig.items()}
    return cagr(INITIAL, simulate(mat, sig, cost)["final"], min(elig), END)


def main(kind: str) -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    tag_commit = verify_prereg(kind)
    t0 = time.time()
    sig_days = monthly_signals() if kind == "rs26" else cycle_signals()
    zone = Zone(sig_days, monthly=(kind == "rs26"))
    qc, qd = load_qqq()
    mat = Mat(zone, nyse_days())
    cov = coverage(kind, zone, mat, qc, sig_days)
    print(f"[{kind}] 태그 {SPECS[kind]['tag']} {tag_commit[:7]} 확인 · 신호 {len(sig_days)}개 {sig_days[0].date()} ~ {sig_days[-1].date()}")
    print(f"[{kind}] 확보율 전체 {cov['overall_pct']:.2f}% · 최저 {cov['lowest_pct']:.1f}% ({cov['lowest_date']}) · 2012-01~2013-06 {cov['early_pct']:.2f}%")
    status = "INCONCLUSIVE" if cov["overall_pct"] < 90.0 else None
    strat, elig = build_signals(kind, zone, qc, sig_days)
    first = sig_days[0]
    out = {"kind": kind, "tag_commit": tag_commit, "coverage": cov, "signals": len(sig_days)}
    for cost in (0.1, 0.3):
        s = simulate(mat, strat, cost)
        q = simulate_qqq(qc, qd, first, cost)
        ew = simulate(mat, elig, cost)
        out[f"cost_{cost}"] = {"strategy_final": s["final"], "qqq_final": q["final"], "ew_final": ew["final"],
                               "strategy_tax": s["tax"], "rebalances": s["rebalances"],
                               "cagr": {"strategy": cagr(INITIAL, s["final"], first, END), "qqq": cagr(INITIAL, q["final"], first, END),
                                        "ew": cagr(INITIAL, ew["final"], first, END)},
                               "mdd": {"strategy": mdd(s["curve"]), "qqq": mdd(q["curve"]), "ew": mdd(ew["curve"])}}
        if cost == 0.1:
            base = (s, q, ew)
        print(f"[{kind}] 비용 {cost}%: 세후 연복리 전략 {out[f'cost_{cost}']['cagr']['strategy']:.2f}% · "
              f"QQQ {out[f'cost_{cost}']['cagr']['qqq']:.2f}% · EW {out[f'cost_{cost}']['cagr']['ew']:.2f}%")
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(initializer=_pool_init, initargs=(elig, mat, 0.1)) as ex:
        rand = list(ex.map(_random_run, range(N_RANDOM), chunksize=10))
    j = judge(base[0], base[1], base[2], rand, first)
    out["judge"] = {**j, "years": {str(k): v for k, v in j["years"].items()}}
    verdict = status or ("PASS" if j["pass"] else "FAIL")
    out["verdict"] = verdict
    for c in j["criteria"]:
        print(f"[{kind}] 기준 {c['id']} {c['name']}: {'충족' if c['ok'] else '미달'} ({c['value']})")
    print(f"[{kind}] 무작위 연복리 중앙값 {j['random_cagr_p50']:.2f}% · 99백분위 {j['random_cagr_p99']:.2f}% · 전략 백분위 {j['random_pct']:.1f}")
    print(f"[{kind}] 연도별(전략, QQQ): " + ", ".join(f"{y} {a:+.1f}/{b:+.1f}" for y, (a, b) in j["years"].items()))
    print(f"[{kind}] 판정: {verdict} · {time.time() - t0:.0f}초")
    (z.OUT / f"result_{kind}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
