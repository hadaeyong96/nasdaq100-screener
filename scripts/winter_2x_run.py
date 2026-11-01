"""겨울엔 2배 — 성적 계산 (docs/prereg/winter_2x.md, 태그 winter2x-prereg).

데이터·합성 2배·^NDX 보정·세금·비용은 레버리지 비율 조절 1·변동성 맞춤과 같다(scripts/shock_dip_gate3, core/shock_dip_perf).
모든 데이터는 2021-12-31에서 자른다(scripts/sealed_guard — 봉인 ⓑ). 결과를 본 뒤 규칙을 바꾸지 않는다.

실행: python scripts/winter_2x_run.py
출력: docs/results/winter_2x/v{N}/ (result.md, results.csv, summary.json) — 이미 있는 버전은 덮어쓰지 않고 다음 번호로 쓴다.
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

from core import leverage_timing as lt  # noqa: E402
from core import shock_dip as sd  # noqa: E402
from core import shock_dip_perf as sp  # noqa: E402
from core import winter_2x as w2  # noqa: E402
from scripts.shock_dip_gate01 import load_fred, md_table  # noqa: E402
from scripts.shock_dip_gate2 import load_close  # noqa: E402
from scripts.shock_dip_gate3 import START_KRW, asof_series, assets, run_variant  # noqa: E402

OUT_BASE = ROOT / "docs" / "results" / "winter_2x"
N_RANDOM = 1000
SEED = 20261101
US = [("S&P500", "^GSPC", "1977-01-01", "1998-12-31"), ("나스닥 종합", "^IXIC", "1977-01-01", "1998-12-31"),
      ("QQQ(보정)", "QQQ", "2000-01-01", "2021-12-31")]
FOREIGN = [("닛케이225", "^N225"), ("DAX", "^GDAXI"), ("FTSE100", "^FTSE"), ("항셍", "^HSI"), ("코스피", "^KS11")]
VARIANTS = ["H1", "H2"]
NAMES = {"H1": "순수(11~4월 2배)", "H2": "H1 + 200일선 아래면 1배"}
BASES = {"항상 1배": None, "항상 1.5배": lt.W_15X, "항상 2배": lt.W_2X}


def market(name: str, tk: str, start: str | None, end: str, dtb3: pd.Series):
    """(종가, 1배 가격, 2배 가격, 창 시작 위치 w0, 2배 설명). 해외(start None)는 200일 이평이 나오는 첫날(위치 199)부터."""
    close = load_close(tk)
    if tk == "QQQ":
        close, _ = sd.correct_with_reference(load_close("QQQ"), load_close("^NDX"), 0.02)
    close = close[close.index <= pd.Timestamp(end)]
    if close.index.max() > sealed_guard.CUTOFF:
        raise sealed_guard.SealedDataError(f"{tk}: 봉인 ⓑ 값")
    x, lev, _, desc = assets(name, tk, close, end, dtb3)
    w0 = w2.SMA_DAYS - 1 if start is None else int(np.searchsorted(close.index, pd.Timestamp(start)))
    return close, x, lev, w0, desc


def targets_for(v: str, close: pd.Series, pos: list[int], winter: frozenset[int]) -> list[float]:
    return w2.h1_targets(close.index, pos, winter) if v == "H1" else w2.h2_targets(close, pos, winter)


def evaluate(x, lev, d, w0, ev, fx) -> dict:
    """판정 값(QQQ는 세후 원화, 나머지 세전)·세전 연복리·세전 최대 낙폭·일별 가치."""
    pre = run_variant(x, lev, d, w0, ev)
    days = (d[-1] - d[w0]).days
    out = {"pre": sp.cagr(pre.nav[0], pre.end_value, days), "mdd": sp.max_drawdown(pre.nav), "nav": pre.nav, "trades": pre.trades}
    out["val"] = out["pre"]
    if fx is not None:
        out["val"] = sp.cagr(START_KRW, run_variant(x, lev, d, w0, ev, tax_fx=fx).end_after_tax_krw, days)
    return out


def random_values(x, lev, d, w0, pos, fx, rng) -> tuple[np.ndarray, float]:
    """무작위 경로 N_RANDOM개의 판정 값과 경로들의 평균 배율(일 기준)."""
    days = (d[-1] - d[w0]).days
    vals, expo = [], []
    for _ in range(N_RANDOM):
        ev = w2.events(pos, w2.random_targets(d, pos, rng))
        if fx is not None:
            r = run_variant(x, lev, d, w0, ev, tax_fx=fx, daily=False)
            vals.append(sp.cagr(START_KRW, r.end_after_tax_krw, days))
        else:
            r = run_variant(x, lev, d, w0, ev, daily=False)
            vals.append(sp.cagr(1.0, r.end_value, days))
        expo.append(float(w2.leverage_of(lt.exposure(ev, w0, len(d))[w0 + 1:]).mean()))
    return np.array(vals), float(np.mean(expo))


def main() -> None:
    dtb3 = load_fred("DTB3")
    fx_all = load_fred("DEXKOUS")
    for s in (dtb3, fx_all):
        if s.index.max() > sealed_guard.CUTOFF:
            raise sealed_guard.SealedDataError("FRED 봉인 ⓑ 값")
    rng = np.random.default_rng(SEED)
    rows, robust, foreign, seasons, losses = [], [], [], [], []
    for name, tk, start, end in US:
        close, x, lev, w0, desc = market(name, tk, start, end, dtb3)
        d = close.index
        fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
        pos = w2.decision_positions(d, w0)
        base = {k: evaluate(x, lev, d, w0, [] if w is None else [(w0, w)], fx) for k, w in BASES.items()}
        fakes, fake_expo = random_values(x, lev, d, w0, pos, fx, rng)
        for v in VARIANTS:
            ev = w2.events(pos, targets_for(v, close, pos, w2.WINTER))
            r = evaluate(x, lev, d, w0, ev, fx)
            wts = lt.exposure(ev, w0, len(d))
            rows.append({"변형": v, "구간": name, "창": f"{d[w0].date()} ~ {d[-1].date()}",
                         "기준": "세후 원화" if fx is not None else "세전", "연복리": r["val"], "세전 연복리": r["pre"],
                         "최대 낙폭": r["mdd"], **{f"{k} 연복리": b["val"] for k, b in base.items()},
                         **{f"{k} 최대 낙폭": b["mdd"] for k, b in base.items()},
                         "무작위 95백분위": float(np.percentile(fakes, 95)), "무작위 백분위": float((fakes < r["val"]).mean() * 100),
                         "무작위 중앙값": float(np.median(fakes)), "무작위 평균 배율": fake_expo,
                         "평균 배율": float(w2.leverage_of(wts[w0 + 1:]).mean()), "전환 횟수": len(ev), "2배 자산": desc})
            sp_df = w2.lever_spells(r["nav"], wts[w0:], d[w0:]).sort_values("구간 수익률").head(3)
            for _, s in sp_df.iterrows():
                losses.append({"변형": v, "구간": name, **s.to_dict()})
            for lab, win in w2.ROBUST.items():
                rr = evaluate(x, lev, d, w0, w2.events(pos, targets_for(v, close, pos, win)), fx)
                robust.append({"변형": v, "겨울": lab, "구간": name, "연복리": rr["val"],
                               "항상 1배 연복리": base["항상 1배"]["val"], "항상 1.5배 연복리": base["항상 1.5배"]["val"]})
        hy = w2.half_year_returns(x.iloc[w0:])
        seasons.append({"구간": name, "겨울 평균": hy["겨울"].mean(), "여름 평균": hy["여름"].mean(),
                        "겨울 > 여름 비율": float((hy["겨울−여름"].dropna() > 0).mean()), "비교 연도 수": int(hy["겨울−여름"].notna().sum())})
    for name, tk in FOREIGN:
        close, x, lev, w0, _ = market(name, tk, None, "2021-12-31", dtb3)
        d = close.index
        pos = w2.decision_positions(d, w0)
        b15 = evaluate(x, lev, d, w0, [(w0, lt.W_15X)], None)
        b1 = evaluate(x, lev, d, w0, [], None)
        for v in VARIANTS:
            r = evaluate(x, lev, d, w0, w2.events(pos, targets_for(v, close, pos, w2.WINTER)), None)
            foreign.append({"변형": v, "지수": name, "창": f"{d[w0].date()} ~ {d[-1].date()}", "세전 연복리": r["pre"],
                            "항상 1.5배 연복리": b15["pre"], "항상 1배 연복리": b1["pre"], "최대 낙폭": r["mdd"],
                            "항상 1.5배 최대 낙폭": b15["mdd"]})
        hy = w2.half_year_returns(x.iloc[w0:])
        seasons.append({"구간": name, "겨울 평균": hy["겨울"].mean(), "여름 평균": hy["여름"].mean(),
                        "겨울 > 여름 비율": float((hy["겨울−여름"].dropna() > 0).mean()), "비교 연도 수": int(hy["겨울−여름"].notna().sum())})
    res, rob, fr = pd.DataFrame(rows), pd.DataFrame(robust), pd.DataFrame(foreign)
    verdict = judge(res, rob, fr)
    write(res, rob, fr, pd.DataFrame(seasons), pd.DataFrame(losses), verdict)


def c1_ok(r: pd.DataFrame) -> bool:
    """합격 기준 1: 세 구간 모두 연복리 > 항상 1배 그리고 > 항상 1.5배."""
    return bool(((r["연복리"] > r["항상 1배 연복리"]) & (r["연복리"] > r["항상 1.5배 연복리"])).all()) and len(r) == len(US)


def judge(res: pd.DataFrame, rob: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for v in VARIANTS:
        r = res[res["변형"] == v]
        c1 = c1_ok(r)
        c2 = bool((r["연복리"] >= r["무작위 95백분위"]).all())
        c3 = bool((r["최대 낙폭"] >= r["항상 1.5배 최대 낙폭"]).all())
        rb = {lab: c1_ok(rob[(rob["변형"] == v) & (rob["겨울"] == lab)]) for lab in w2.ROBUST}
        c4 = sum(rb.values()) >= 3
        f = fr[fr["변형"] == v]
        wins = f[f["세전 연복리"] > f["항상 1.5배 연복리"]]["지수"].tolist()
        c5 = len(wins) >= 3
        g = list(r["구간"])
        out.append({
            "변형": v, "이름": NAMES[v],
            "1. 세 구간 > 1배·1.5배": ("O" if c1 else "X") + " (" + ", ".join(
                f"{k}:{'O' if a > b and a > c else 'X'}" for k, a, b, c in zip(g, r["연복리"], r["항상 1배 연복리"], r["항상 1.5배 연복리"])) + ")",
            "2. 무작위 95백분위 이상": ("O" if c2 else "X") + " (" + ", ".join(f"{k}:{p:.1f}" for k, p in zip(g, r["무작위 백분위"])) + ")",
            "3. 낙폭 ≤ 1.5배": ("O" if c3 else "X") + " (" + ", ".join(
                f"{k}:{'O' if a >= b else 'X'}" for k, a, b in zip(g, r["최대 낙폭"], r["항상 1.5배 최대 낙폭"])) + ")",
            "4. 견고성 4개 중 3개": ("O" if c4 else "X") + f" ({sum(rb.values())}/4: " + ", ".join(f"{k} {'O' if ok else 'X'}" for k, ok in rb.items()) + ")",
            "5. 해외 5곳 중 3곳 > 1.5배": ("O" if c5 else "X") + f" ({len(wins)}/5: {', '.join(wins) or '없음'})",
            "종합": "PASS" if (c1 and c2 and c3 and c4 and c5) else "FAIL",
        })
    return pd.DataFrame(out)


def _p(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def next_version_dir(base: Path) -> Path:
    """base/v1, v2, … 중 아직 없는 첫 번호(덮어쓰기 금지)."""
    n = 1
    while (base / f"v{n}").exists():
        n += 1
    return base / f"v{n}"


def write(res, rob, fr, seasons, losses, verdict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    out = next_version_dir(OUT_BASE)
    out.mkdir(parents=True)
    res.to_csv(out / "results.csv", index=False, encoding="utf-8-sig")
    show = res.copy()
    pct_cols = ["연복리", "세전 연복리", "최대 낙폭", "무작위 95백분위", "무작위 중앙값"] + [c for c in res.columns if c.startswith("항상 ")]
    for c in pct_cols:
        show[c] = show[c].map(_p)
    show["무작위 백분위"] = show["무작위 백분위"].map(lambda v: f"{v:.1f}")
    for c in ("평균 배율", "무작위 평균 배율"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    main_cols = ["구간", "변형", "기준", "연복리", "최대 낙폭", "무작위 백분위", "무작위 95백분위", "항상 1배 연복리", "항상 1배 최대 낙폭",
                 "항상 1.5배 연복리", "항상 1.5배 최대 낙폭", "항상 2배 연복리", "항상 2배 최대 낙폭"]
    etc_cols = ["구간", "변형", "창", "세전 연복리", "무작위 중앙값", "평균 배율", "무작위 평균 배율", "전환 횟수", "2배 자산"]
    rb = rob.copy()
    for c in ("연복리", "항상 1배 연복리", "항상 1.5배 연복리"):
        rb[c] = rb[c].map(_p)
    f = fr.copy()
    for c in ("세전 연복리", "항상 1.5배 연복리", "항상 1배 연복리", "최대 낙폭", "항상 1.5배 최대 낙폭"):
        f[c] = f[c].map(_p)
    se = seasons.copy()
    for c in ("겨울 평균", "여름 평균"):
        se[c] = se[c].map(_p)
    se["겨울 > 여름 비율"] = se["겨울 > 여름 비율"].map(lambda v: f"{v * 100:.0f}%")
    lo = losses.copy()
    for c in ("구간 수익률", "구간 안 최대 낙폭"):
        lo[c] = lo[c].map(_p)
    passed = verdict[verdict["종합"] == "PASS"]["변형"].tolist()
    md = [
        "# 겨울엔 2배 — 성적 (변형 2개 판정)", "",
        f"> 규칙 `docs/prereg/winter_2x.md`(태그 `winter2x-prereg`). 코드 커밋 `{commit}`. 실행 `python scripts/winter_2x_run.py` → `{out.relative_to(ROOT).as_posix()}`.",
        f"> 판정: **{'PASS — ' + ', '.join(passed) if passed else 'FAIL (두 변형 모두 불합격)'}**. 봉인 ⓑ(2022~) 데이터는 읽을 때 2021-12-31에서 잘랐다.", "",
        "## 변형별 합격 기준", "", md_table(verdict), "",
        "## 미국 세 구간 (QQQ는 세후 원화, 나머지 세전 · 낙폭은 세전 일별)", "", md_table(show[main_cols]), "",
        "### 그 밖의 값", "", md_table(show[etc_cols]), "",
        "## 견고성 (겨울 구간 옮기기, 합격 기준 1로 판단)", "", md_table(rb), "",
        "## 해외 5곳 (세전, 현지 통화, 합성 2배)", "", md_table(f), "",
        "## 참고 보고 (판정 아님)", "",
        "### 겨울·여름 6개월 평균 수익 (1배 자산, 겨울 = 전년 10월 말 → 4월 말, 여름 = 4월 말 → 10월 말)", "", md_table(se), "",
        "### 2배 구간 최대 손실 3건 (구간별, 세전 일별 가치)", "", md_table(lo), "",
    ]
    (out / "result.md").write_text("\n".join(md), encoding="utf-8")
    q = res[res["구간"] == "QQQ(보정)"].set_index("변형")
    summary = {"name": "겨울엔 2배 (H1·H2)", "verdict": "PASS: " + ", ".join(passed) if passed else "FAIL", "commit": commit,
               "version": out.name, "seed": SEED, "n_random": N_RANDOM,
               "qqq": {v: {"cagr_after_tax": round(float(q.loc[v, "연복리"]) * 100, 2),
                           "random_pct": round(float(q.loc[v, "무작위 백분위"]), 1)} for v in VARIANTS},
               "verdict_table": verdict.to_dict(orient="records")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("완료", out.name, summary["verdict"])


if __name__ == "__main__":
    main()
