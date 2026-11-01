"""마이크론 사이클 2단계 — 판정 (docs/prereg/mu_cycle.md, 태그 mu-cycle-prereg).

판정 구간 2013-01-01 ~ 2021-12-31. 모든 데이터는 2021-12-31에서 끊어 받고 읽을 때 다시 막는다(data/mu_cycle.guard_s2 — 봉인 ⓑ).
결과를 본 뒤 규칙을 바꾸지 않는다.

받기(태그 push 뒤 한 번): python scripts/mu_cycle_stage2_run.py --fetch
실행: python scripts/mu_cycle_stage2_run.py
출력: docs/results/mu_cycle/stage2/v{N}/ (report.md, books.csv, summary.json) — 이미 있는 버전은 덮어쓰지 않는다.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import mu_cycle as mc  # noqa: E402
from core import mu_strategy as ms  # noqa: E402
from data import mu_cycle as dm  # noqa: E402

OUT_BASE = ROOT / "docs" / "results" / "mu_cycle" / "stage2"
START, END = pd.Timestamp("2013-01-01"), dm.S2_CUTOFF
SIG_FIRST, SIG_LAST = pd.Timestamp("2012-12-31"), pd.Timestamp("2021-11-30")  # 첫 분할이 판정 구간 안에 오는 신호 달
X_FIRST = pd.Timestamp("2011-01-31")  # 6개월 변화 12개월 창이 2012-12에 다 차도록
HOLD, ROBUST = 18, (12, 24)
PEERS = ["AMAT", "LRCX", "KLAC", "TXN", "MCHP"]
N_RANDOM = 1000
SEED = 20261101
COST = 0.001


def fetch() -> None:
    """네트워크: 2단계 시세·환율·A34SNO 현재 값·월말별 ALFRED 발표본. 실패 목록을 모아 보고한다."""
    fail = []
    for t in ["MU", "QQQ", *PEERS]:
        try:
            dm.fetch_yf_s2(t)
        except Exception as e:  # noqa: BLE001 — 모아서 보고
            fail.append(f"{t}: {type(e).__name__}")
    for sid, start in (("DEXKOUS", "2012-01-01"), ("A34SNO", "2008-01-01")):
        try:
            dm.fetch_fred_s2(sid, start)
        except Exception as e:  # noqa: BLE001
            fail.append(f"{sid}: {type(e).__name__}")
    for m in pd.date_range(X_FIRST, END, freq="ME"):
        if (dm.S2_ALFRED / f"A34SNO_{m.date().isoformat()}.csv").exists():
            continue
        try:
            dm.fetch_alfred_asof("A34SNO", m)
        except Exception as e:  # noqa: BLE001
            fail.append(f"ALFRED {m.date()}: {type(e).__name__}")
    print("받기 끝. 실패:", fail or "없음")


def prices(ticker: str, days: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """판정 구간 거래일(QQQ 기준)에 맞춘 종가·주당 배당."""
    df = dm.load_yf_s2(ticker)
    c = df["Close"].reindex(days).ffill().to_numpy()
    d = df["Dividends"].reindex(days).fillna(0.0).to_numpy()
    if np.isnan(c).any():
        raise ValueError(f"{ticker}: 판정 구간 첫날 가격 없음")
    return c, d


def signal_series() -> pd.DataFrame:
    me = pd.date_range(X_FIRST, END, freq="ME")
    asof = {m: dm.load_alfred_asof("A34SNO", m) for m in me}
    return ms.realtime_change(me, asof, dm.load_fred_s2("A34SNO"))


def evaluate(pa, da, pb, db, ev, fx, days) -> dict:
    t = ms.run_book(pa, pb, da, db, ev, cost=COST, fx=fx, tax_on=True, dates=days)
    pre = ms.run_book(pa, pb, da, db, ev, cost=COST, tax_on=False, dates=days)
    span = (days[-1] - days[0]).days
    return {"cagr": ms.cagr(100_000_000.0, t.end_after_tax_krw, span), "mdd": ms.max_drawdown(pre.nav),
            "pre_cagr": ms.cagr(pre.nav[0], pre.nav[-1], span), "max_w": t.max_b_weight_after_buy}


def random_cagrs(n: int, pa, da, pb, db, fx, days, rng) -> np.ndarray:
    months = pd.date_range(SIG_FIRST, SIG_LAST, freq="ME")
    span = (days[-1] - days[0]).days
    out = []
    for _ in range(N_RANDOM):
        ev = ms.entry_events(ms.random_signal_months(n, months, HOLD, rng), days, HOLD)
        b = ms.run_book(pa, pb, da, db, ev, cost=COST, fx=fx, tax_on=True, dates=days)
        out.append(ms.cagr(100_000_000.0, b.end_after_tax_krw, span))
    return np.array(out)


def main() -> None:
    qqq = dm.load_yf_s2("QQQ")
    days = qqq.index[(qqq.index >= START) & (qqq.index <= END)]
    fx = dm.load_fred_s2("DEXKOUS")
    fx = fx.reindex(fx.index.union(days)).ffill().reindex(days).to_numpy()
    pa, da = prices("QQQ", days)
    x = signal_series()
    sigs = ms.signal_months(x["value"])
    entries = {h: ms.accept_signals(sigs, SIG_FIRST, SIG_LAST, h) for h in (HOLD, *ROBUST)}
    n = len(entries[HOLD])
    rows, peer_rows = [], []
    for i, tk in enumerate(["MU", *PEERS]):
        pb, db = prices(tk, days)
        strat = evaluate(pa, da, pb, db, ms.entry_events(entries[HOLD], days, HOLD), fx, days)
        fakes = random_cagrs(n, pa, da, pb, db, fx, days, np.random.default_rng([SEED, i]))
        rec = {"종목": tk, "전략 연복리": strat["cagr"], "전략 최대 낙폭": strat["mdd"], "전략 세전 연복리": strat["pre_cagr"],
               "무작위 백분위": float((fakes < strat["cagr"]).mean() * 100), "무작위 95백분위": float(np.percentile(fakes, 95)),
               "무작위 중앙값": float(np.median(fakes)), "분할 직후 최대 비중": strat["max_w"]}
        if tk == "MU":
            base_q = evaluate(pa, da, pb, db, {}, fx, days)
            base_10 = evaluate(pa, da, pb, db, ms.annual_rebalance_events(days), fx, days)
            rob = {h: evaluate(pa, da, pb, db, ms.entry_events(entries[h], days, h), fx, days) for h in ROBUST}
            rows = [{"장부": "전략(18개월)", "연복리": strat["cagr"], "최대 낙폭": strat["mdd"], "무작위 백분위": rec["무작위 백분위"],
                     "세전 연복리": strat["pre_cagr"]},
                    {"장부": "QQQ 100%", "연복리": base_q["cagr"], "최대 낙폭": base_q["mdd"], "무작위 백분위": np.nan, "세전 연복리": base_q["pre_cagr"]},
                    {"장부": "항상 MU 10%", "연복리": base_10["cagr"], "최대 낙폭": base_10["mdd"], "무작위 백분위": np.nan, "세전 연복리": base_10["pre_cagr"]}]
            rows += [{"장부": f"견고성 보유 {h}개월", "연복리": rob[h]["cagr"], "최대 낙폭": rob[h]["mdd"], "무작위 백분위": np.nan,
                      "세전 연복리": rob[h]["pre_cagr"]} for h in ROBUST]
            mu_rec, mu_fakes = rec, fakes
        else:
            peer_rows.append(rec)
    books, peers = pd.DataFrame(rows).set_index("장부"), pd.DataFrame(peer_rows)
    verdict = judge(books, mu_rec, peers)
    refs = references(x, sigs, entries[HOLD], days)
    write(books, mu_rec, peers, verdict, refs, x, sigs, entries, n)


def judge(books: pd.DataFrame, mu: dict, peers: pd.DataFrame) -> dict:
    s, q, a = books.loc["전략(18개월)"], books.loc["QQQ 100%"], books.loc["항상 MU 10%"]
    c1 = bool(s["연복리"] > q["연복리"] and s["연복리"] > a["연복리"])
    c2 = bool(s["연복리"] >= mu["무작위 95백분위"])
    c3 = bool(s["최대 낙폭"] > q["최대 낙폭"] - 0.03)
    rob = {h: bool(books.loc[f"견고성 보유 {h}개월", "연복리"] > q["연복리"] and books.loc[f"견고성 보유 {h}개월", "연복리"] > a["연복리"])
           for h in ROBUST}
    c4 = all(rob.values())
    wins = peers[peers["전략 연복리"] > peers["무작위 중앙값"]]["종목"].tolist()
    c5 = len(wins) >= 3
    return {"1": c1, "2": c2, "3": c3, "4": c4, "4_detail": rob, "5": c5, "5_wins": wins,
            "판정": "PASS" if all((c1, c2, c3, c4, c5)) else "FAIL"}


def references(x: pd.DataFrame, sigs, entries, days) -> dict:
    """참고(판정 제외): 진입별 MU 18개월 수익, 2013~21 적중률·기준선, 발표본 사용 비율."""
    mu = dm.load_yf_s2("MU")
    qq = dm.load_yf_s2("QQQ")
    rows = []
    for s in [g for g in sigs if SIG_FIRST <= g <= SIG_LAST]:
        r = {"신호 달": s.strftime("%Y-%m"), "6개월 변화(%)": round(float(x.loc[s, "value"]), 2), "진입": "진입" if s in entries else "무시(보유 중)"}
        if s in entries:
            p_last = ms.first_trading_day(days, ms.month_add(s, 4))
            p_exit = ms.first_trading_day(days, ms.exit_month(s, HOLD))
            r["마지막 분할일"] = days[p_last].date() if p_last is not None else None
            r["복귀일"] = days[p_exit].date() if p_exit is not None else "구간 끝까지 보유"
            if p_last is not None:
                d0, d1 = days[p_last], days[p_exit] if p_exit is not None else days[-1]
                r["MU 수익(마지막 분할 → 복귀, 가격)"] = float(mu["Close"][d1] / mu["Close"][d0] - 1)
                r["QQQ 같은 기간"] = float(qq["Close"][d1] / qq["Close"][d0] - 1)
        rows.append(r)
    m = mc.month_end_close(mu["Close"])["close"]
    m = m[m.index >= pd.Timestamp("1990-01-31")]
    pts = mc.zigzag(m)
    pts = pts[pts["date"] != m.index[0]]
    troughs = [d for d in pts.loc[pts["kind"] == "trough", "date"] if d >= START]
    xv = x["value"]
    st = mc.signal_stats(xv, m.reindex(xv.index), troughs, pd.Timestamp("2013-01-31"))
    base = mc.base_hit_rate(xv.index, troughs, pd.Timestamp("2013-01-31"))
    used = x.loc[(x.index >= pd.Timestamp("2011-12-31")) & (x.index <= SIG_LAST)]
    return {"signals": pd.DataFrame(rows), "hit": st, "base": base, "troughs": [d.strftime("%Y-%m") for d in troughs],
            "points": pts[pts["date"] >= START], "vintage_ratio": float(used["vintage"].mean()), "vintage_months": len(used)}


def p(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.{d}f}%"


def md(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("—" if (isinstance(v, float) and np.isnan(v)) or v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def ox(b: bool) -> str:
    return "O" if b else "X"


def write(books, mu, peers, verdict, refs, x, sigs, entries, n) -> None:
    out = OUT_BASE / "v1"
    k = 1
    while out.exists():
        k += 1
        out = OUT_BASE / f"v{k}"
    out.mkdir(parents=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    books.to_csv(out / "books.csv", encoding="utf-8-sig")
    peers.to_csv(out / "peers.csv", index=False, encoding="utf-8-sig")
    b = books.reset_index()
    show = pd.DataFrame({"장부": b["장부"], "연복리(원화 세후)": b["연복리"].map(p), "최대 낙폭(세전 일별)": b["최대 낙폭"].map(p),
                         "무작위 백분위": b["무작위 백분위"].map(lambda v: "—" if np.isnan(v) else f"{v:.1f}"),
                         "세전 연복리(달러)": b["세전 연복리"].map(p)})
    pr = pd.DataFrame({"종목": peers["종목"], "전략 연복리": peers["전략 연복리"].map(p), "무작위 중앙값": peers["무작위 중앙값"].map(p),
                       "무작위 백분위": peers["무작위 백분위"].map(lambda v: f"{v:.1f}"),
                       "중앙값 초과": [ox(a > c) for a, c in zip(peers["전략 연복리"], peers["무작위 중앙값"])],
                       "최대 낙폭": peers["전략 최대 낙폭"].map(p)})
    s, q, a = books.loc["전략(18개월)"], books.loc["QQQ 100%"], books.loc["항상 MU 10%"]
    crit = pd.DataFrame([
        {"기준": "1. 연복리 > QQQ 100% 그리고 > 항상 MU 10%", "충족": ox(verdict["1"]), "값": f"{p(s['연복리'])} vs {p(q['연복리'])} · {p(a['연복리'])}"},
        {"기준": "2. 무작위 대조 95백분위 이상", "충족": ox(verdict["2"]), "값": f"{mu['무작위 백분위']:.1f}백분위 (95백분위 {p(mu['무작위 95백분위'])})"},
        {"기준": "3. 최대 낙폭이 QQQ보다 3%p 이상 깊지 않음", "충족": ox(verdict["3"]), "값": f"{p(s['최대 낙폭'])} vs QQQ {p(q['최대 낙폭'])}"},
        {"기준": "4. 보유 12·24개월 모두 1번 충족", "충족": ox(verdict["4"]),
         "값": ", ".join(f"{h}개월 {ox(v)} ({p(books.loc[f'견고성 보유 {h}개월', '연복리'])})" for h, v in verdict["4_detail"].items())},
        {"기준": "5. 동료 5곳 중 3곳 이상 자기 무작위 중앙값 초과", "충족": ox(verdict["5"]), "값": f"{len(verdict['5_wins'])}/5: {', '.join(verdict['5_wins']) or '없음'}"},
    ])
    sg = refs["signals"].copy()
    for c in ("MU 수익(마지막 분할 → 복귀, 가격)", "QQQ 같은 기간"):
        if c in sg:
            sg[c] = sg[c].map(lambda v: p(v, 1) if isinstance(v, float) else "—")
    h = refs["hit"]
    lines = [
        "# 마이크론 사이클 2단계 — 판정 (A34SNO 6개월 변화 신호 → MU 최대 10%)", "",
        f"> 규칙 `docs/prereg/mu_cycle.md`(태그 `mu-cycle-prereg`). 코드 커밋 `{commit}`. 실행 `python scripts/mu_cycle_stage2_run.py` → `{out.relative_to(ROOT).as_posix()}`.",
        f"> 판정: **{verdict['판정']}**. 판정 구간 {START.date()} ~ {END.date()}, 원화 세후(core/tax.py), 비용 편도 0.1%. 2022년 이후 데이터 미사용(받을 때·읽을 때 차단).", "",
        "## 합격 기준", "", md(crit), "",
        f"## 장부 (진입 {n}번)", "", md(show), "",
        "## 동료 5곳 (같은 신호·규칙, MU 자리에 그 종목, 자기 무작위 1,000회)", "", md(pr), "",
        "## 참고 (판정 제외)", "",
        "### 신호와 진입별 MU 수익", "", md(sg) if len(sg) else "판정 구간에 신호 없음", "",
        f"### 2013~2021 적중률 (1단계와 같은 방식)", "",
        f"- MU 바닥(지그재그 −40%/+60%, 1990~2021 이어서 계산): {', '.join(refs['troughs']) or '없음'}",
        f"- 평가 신호 {h['신호 수']}번 중 12개월 안 바닥 {h['적중']}번 → 적중률 {p(h['적중률'], 0)}, 헛신호 {h['헛신호']}번, 놓침 {h['놓침']}/{h['평가 바닥']}",
        f"- 기준선(아무 달 신호 시 12개월 안 바닥 비율): {p(refs['base'], 0)} (1단계 1990~2012는 41%)", "",
        "### ALFRED 발표본 사용 비율", "",
        f"- 신호 판단에 들어간 달(2011-12 ~ 2021-11, {refs['vintage_months']}개월) 중 당시 발표본 사용 {p(refs['vintage_ratio'], 1)}, "
        f"나머지는 현재 값 + 2개월 지연. A34SNO 첫 발표본 2011-06-24.", "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"name": "마이크론 사이클 2단계 (A34SNO 6개월 → MU 10%)", "verdict": verdict["판정"], "commit": commit, "version": out.name,
               "entries": n, "seed": SEED, "n_random": N_RANDOM,
               "strategy_cagr": round(float(s["연복리"]) * 100, 2), "qqq_cagr": round(float(q["연복리"]) * 100, 2),
               "mu10_cagr": round(float(a["연복리"]) * 100, 2), "random_pct": round(mu["무작위 백분위"], 1),
               "criteria": {k: verdict[k] for k in ("1", "2", "3", "4", "5")}, "peers_win": verdict["5_wins"],
               "vintage_ratio": round(refs["vintage_ratio"], 3)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("완료", out.relative_to(ROOT).as_posix(), verdict["판정"])


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    else:
        main()
