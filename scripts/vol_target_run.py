"""변동성 맞춤 레버리지 전략 — 성적 계산 (docs/prereg/vol_target.md, 태그 vol-target-prereg).

데이터·합성 2배·^NDX 보정·세금은 research/leverage-timing(scripts/shock_dip_gate3)과 같다. 모든 데이터는 2021-12-31에서 자른다
(scripts/sealed_guard). 결과를 본 뒤 규칙을 바꾸지 않는다.

실행: python scripts/vol_target_run.py   → docs/research/vol_target_result.md, _summary.json
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

from scripts import sealed_guard  # noqa: E402

sealed_guard.install()

from core import multi_book as mb  # noqa: E402
from core import shock_dip as sd  # noqa: E402
from core import shock_dip_perf as sp  # noqa: E402
from core import vol_target as vt  # noqa: E402
from scripts.shock_dip_gate01 import md_table  # noqa: E402
from scripts.shock_dip_gate2 import load_close  # noqa: E402
from scripts.shock_dip_gate3 import START_KRW, asof_series, assets  # noqa: E402

OUT = ROOT / "docs" / "research"
COST = 0.001
N_RANDOM = int(__import__("os").environ.get("VT_N_RANDOM", "1000"))  # 점검용으로만 줄인다(판정은 1000)
SEED = 20261022
US = [("S&P500", "^GSPC", "1977-01-01", "1998-12-31"), ("나스닥 종합", "^IXIC", "1977-01-01", "1998-12-31"),
      ("QQQ(보정)", "QQQ", "2000-01-01", "2021-12-31")]
FOREIGN = [("닛케이225", "^N225"), ("DAX", "^GDAXI"), ("FTSE100", "^FTSE"), ("항셍", "^HSI"), ("코스피", "^KS11")]
VARIANTS = {"V1": False, "V2": True}


def market(name, tk, start, end, dtb3):
    """(dates, 1배 신호 종가, 가격 n×3[현금,1배,2배], 창 시작 위치 w0, 2배 설명)."""
    close = load_close(tk)
    if tk == "QQQ":
        close, _ = sd.correct_with_reference(load_close("QQQ"), load_close("^NDX"), 0.02)
    close = close[close.index <= pd.Timestamp(end)]
    d = close.index
    x, lev, _, desc = assets(name, tk, close, end, dtb3)
    cash = (1 + asof_series(dtb3, d).shift(1).fillna(0) / 100 / 252).cumprod()
    prices = np.column_stack([cash.to_numpy(), x.to_numpy(), lev.to_numpy()])
    if start is None:  # 해외: σ(20)·200일선이 모두 나오는 첫날
        w0 = 200
    else:
        w0 = int(np.searchsorted(d, pd.Timestamp(start)))
    return d, close, prices, w0, desc


def book(prices, d, w0, decisions, should, fx, start_w=(0.0, 1.0, 0.0)):
    px = prices[w0:]
    dd = d[w0:]
    dec = {p - w0: w for p, w in decisions.items() if p >= w0}
    kw = dict(start_krw=1.0)
    if fx is not None:
        kw = dict(tax=True, fx=fx[w0:], start_krw=START_KRW, may_pay=sp.may_payment_positions(dd), years=dd.year.to_numpy(),
                  sell_order=[vt.CASH, vt.ONE, vt.TWO])
    r = mb.run_multi(px, list(start_w), dec, should, cost=COST, **kw)
    days = (dd[-1] - dd[0]).days
    pre = mb.cagr(r.nav[0], r.end_value, days)
    val = mb.cagr(START_KRW, r.end_after_tax_krw, days) if fx is not None else pre
    return r, pre, val


def run_vt(close, prices, d, w0, fx, target_vol, cap, rng=None):
    pos = vt.week_end_positions(d, w0)
    es = vt.target_exposures(close, pos, target_vol, cap)
    r, pre, val = book(prices, d, w0, {p: vt.weights_of(e) for p, e in zip(pos, es)}, vt.should_rebalance(0.2), fx)
    fakes = None
    if rng is not None:
        fakes = []
        for _ in range(N_RANDOM):
            sh = vt.block_shuffle(es, rng)
            fakes.append(book(prices, d, w0, {p: vt.weights_of(e) for p, e in zip(pos, sh)}, vt.should_rebalance(0.2), fx)[2])
        fakes = np.array(fakes)
    expo = r.weights @ vt.BETA
    years = (d[-1] - d[w0]).days / 365.25
    return {"r": r, "pre": pre, "val": val, "mdd": mb.max_drawdown(r.nav), "avg_expo": float(expo.mean()),
            "two_share": float((r.weights[:, vt.TWO] > 1e-9).mean()), "rebal_per_year": r.trades / years, "fakes": fakes}


def fixed(prices, d, w0, fx, w):
    r, pre, val = book(prices, d, w0, {}, lambda a, b: False, fx, start_w=w)
    return {"pre": pre, "val": val, "mdd": mb.max_drawdown(r.nav)}


def sma_switch(close, prices, d, w0, fx):
    above = (close > close.rolling(200).mean()).to_numpy()
    dec = {p: (np.array([0.0, 0.0, 1.0]) if above[p] else np.array([1.0, 0.0, 0.0])) for p in range(w0, len(d))}
    r, pre, val = book(prices, d, w0, dec, lambda a, b: np.abs(a - b).max() > 0.5, fx,
                       start_w=(0.0, 0.0, 1.0) if above[w0 - 1] else (1.0, 0.0, 0.0))
    return {"pre": pre, "val": val, "mdd": mb.max_drawdown(r.nav)}


def main() -> None:
    dtb3 = sealed_guard.load_fred("DTB3")
    fx_all = sealed_guard.load_fred("DEXKOUS")
    rng = np.random.default_rng(SEED)
    rows, refs, alt = [], [], []
    mk = {}
    for name, tk, start, end in US:
        mk[name] = market(name, tk, start, end, dtb3)
    for v, cap in VARIANTS.items():
        for name, tk, start, end in US:
            d, close, prices, w0, desc = mk[name]
            fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
            res = run_vt(close, prices, d, w0, fx, 0.20, cap, rng)
            base = fixed(prices, d, w0, fx, (0.0, 1.0, 0.0))
            rows.append({"변형": v, "구간": name, "창": f"{d[w0].date()} ~ {d[-1].date()}", "기준": "세후 원화" if fx is not None else "세전",
                         "연복리": res["val"], "1배 연복리": base["val"], "세전 연복리": res["pre"], "최대 낙폭": res["mdd"],
                         "1배 최대 낙폭": base["mdd"], "무작위 95백분위": float(np.percentile(res["fakes"], 95)),
                         "무작위 백분위": float((res["fakes"] < res["val"]).mean() * 100), "평균 노출": res["avg_expo"],
                         "2배 시간 비율": res["two_share"], "연간 다시 맞춤": res["rebal_per_year"], "2배 자산": desc})
            for tv in (0.15, 0.25):
                a = run_vt(close, prices, d, w0, fx, tv, cap)
                alt.append({"변형": v, "목표변동성": f"{tv:.0%}", "구간": name, "연복리": a["val"], "1배 연복리": base["val"]})
    for name, tk, start, end in US:
        d, close, prices, w0, desc = mk[name]
        fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
        for lab, r in (("항상 1배", fixed(prices, d, w0, fx, (0, 1, 0))), ("항상 2배", fixed(prices, d, w0, fx, (0, 0, 1))),
                       ("200일선 2배 전환(아래 현금)", sma_switch(close, prices, d, w0, fx))):
            refs.append({"구간": name, "방식": lab, "연복리(판정 기준과 같은 기준)": r["val"], "세전 연복리": r["pre"], "최대 낙폭": r["mdd"]})
    foreign = []
    for name, tk in FOREIGN:
        d, close, prices, w0, desc = market(name, tk, None, "2021-12-31", dtb3)
        base = fixed(prices, d, w0, None, (0, 1, 0))
        for v, cap in VARIANTS.items():
            r = run_vt(close, prices, d, w0, None, 0.20, cap)
            foreign.append({"변형": v, "지수": name, "창": f"{d[w0].date()} ~ {d[-1].date()}", "세전 연복리": r["pre"],
                            "1배 연복리": base["pre"], "최대 낙폭": r["mdd"], "1배 최대 낙폭": base["mdd"], "평균 노출": r["avg_expo"]})
    res, al, fr, rf = pd.DataFrame(rows), pd.DataFrame(alt), pd.DataFrame(foreign), pd.DataFrame(refs)
    verdict = judge(res, al, fr)
    if N_RANDOM != 1000:  # 점검 실행 — 결과 파일을 쓰지 않는다
        print("점검 실행(무작위", N_RANDOM, "회) — 파일 안 씀")
        return
    write(res, al, fr, rf, verdict)


def judge(res, al, fr) -> pd.DataFrame:
    out = []
    for v in VARIANTS:
        r, a, f = res[res["변형"] == v], al[al["변형"] == v], fr[fr["변형"] == v]
        c1 = bool((r["연복리"] > r["1배 연복리"]).all())
        c2 = bool((r["연복리"] >= r["무작위 95백분위"]).all())
        c3 = bool((r["최대 낙폭"] >= r["1배 최대 낙폭"]).all())
        c4 = bool((a["연복리"] > a["1배 연복리"]).all())
        wins = f[f["세전 연복리"] > f["1배 연복리"]]["지수"].tolist()
        c5 = len(wins) >= 3
        ok = all((c1, c2, c3, c4, c5))
        out.append({
            "변형": v,
            "① 세 구간 > 1배": ("O" if c1 else "X") + " (" + ", ".join(f"{g}:{'O' if x > y else 'X'}" for g, x, y in zip(r["구간"], r["연복리"], r["1배 연복리"])) + ")",
            "② 무작위 95백분위 이상": ("O" if c2 else "X") + " (" + ", ".join(f"{g}:{p:.1f}" for g, p in zip(r["구간"], r["무작위 백분위"])) + ")",
            "③ 낙폭 1배보다 깊지 않음": ("O" if c3 else "X") + " (" + ", ".join(f"{g}:{'O' if x >= y else 'X'}" for g, x, y in zip(r["구간"], r["최대 낙폭"], r["1배 최대 낙폭"])) + ")",
            "④ 목표 15%·25%도 ①": ("O" if c4 else "X") + " (" + ", ".join(f"{t} {g}:{'O' if x > y else 'X'}" for t, g, x, y in zip(a["목표변동성"], a["구간"], a["연복리"], a["1배 연복리"])) + ")",
            "⑤ 해외 3곳 이상": f"{'O' if c5 else 'X'} ({len(wins)}/5: {', '.join(wins) or '없음'})",
            "판정": "PASS" if ok else "FAIL"})
    return pd.DataFrame(out)


def _p(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def write(res, al, fr, rf, verdict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    show = res.copy()
    for c in ("연복리", "1배 연복리", "세전 연복리", "최대 낙폭", "1배 최대 낙폭", "무작위 95백분위", "2배 시간 비율"):
        show[c] = show[c].map(_p)
    show["무작위 백분위"] = show["무작위 백분위"].map(lambda v: f"{v:.1f}")
    show["평균 노출"] = show["평균 노출"].map(lambda v: f"{v:.2f}")
    show["연간 다시 맞춤"] = show["연간 다시 맞춤"].map(lambda v: f"{v:.1f}")
    a = al.copy()
    for c in ("연복리", "1배 연복리"):
        a[c] = a[c].map(_p)
    f = fr.copy()
    for c in ("세전 연복리", "1배 연복리", "최대 낙폭", "1배 최대 낙폭"):
        f[c] = f[c].map(_p)
    f["평균 노출"] = f["평균 노출"].map(lambda v: f"{v:.2f}")
    r = rf.copy()
    for c in ("연복리(판정 기준과 같은 기준)", "세전 연복리", "최대 낙폭"):
        r[c] = r[c].map(_p)
    passed = verdict[verdict["판정"] == "PASS"]["변형"].tolist()
    cols = ["변형", "구간", "창", "기준", "연복리", "1배 연복리", "무작위 95백분위", "무작위 백분위", "최대 낙폭", "1배 최대 낙폭",
            "평균 노출", "2배 시간 비율", "연간 다시 맞춤", "세전 연복리"]
    md = ["# 변동성 맞춤 레버리지 전략 — 결과 (V1·V2 판정)", "",
          f"> 규칙 `docs/prereg/vol_target.md`(태그 `vol-target-prereg`). 코드 커밋 `{commit}`. 실행 `python scripts/vol_target_run.py`.",
          "> 모든 데이터 2021-12-31에서 자름(봉인 ⓑ 열람·계산 없음).",
          f"> 종합: **{'PASS: ' + ', '.join(passed) if passed else 'FAIL (두 변형 모두 불합격)'}**", "",
          "## 판정", "", md_table(verdict), "",
          "## 미국 세 구간 (QQQ는 세후 원화, 나머지 세전)", "", md_table(show[cols]), "",
          "## 목표변동성 15%·25% (④)", "", md_table(a), "",
          "## 해외 5곳 (세전, 합성 2배, σ·200일선이 나오는 첫날부터)", "", md_table(f), "",
          "## 참고 보고 (판정 제외)", "", md_table(r), ""]
    (OUT / "vol_target_result.md").write_text("\n".join(md), encoding="utf-8")
    q = res[res["구간"] == "QQQ(보정)"].set_index("변형")
    summary = {"verdict": "PASS: " + ", ".join(passed) if passed else "FAIL", "commit": commit,
               "qqq": {v: {"cagr": round(float(q.loc[v, "연복리"]) * 100, 2), "base": round(float(q.loc[v, "1배 연복리"]) * 100, 2),
                           "mdd": round(float(q.loc[v, "최대 낙폭"]) * 100, 1), "random_pct": round(float(q.loc[v, "무작위 백분위"]), 1)}
                       for v in VARIANTS},
               "verdict_table": verdict.to_dict(orient="records")}
    (OUT / "vol_target_result_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("done", summary["verdict"])


if __name__ == "__main__":
    main()
