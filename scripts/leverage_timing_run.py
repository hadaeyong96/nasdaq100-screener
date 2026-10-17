"""레버리지 비율 조절 1 — 성적 계산 (docs/prereg/leverage_timing.md, 태그 leverage-timing-prereg).

신호 core/leverage_timing.py, 장부 core/shock_dip_perf.py(폭락 연구와 같은 비용·세금·합성 2배·^NDX 보정).
각 지수는 2021-12-31(S&P·나스닥은 1998-12-31)에서 자른다. 결과를 본 뒤 규칙을 바꾸지 않는다.

실행: python scripts/leverage_timing_run.py
출력: docs/research/leverage_timing_results.md, .csv, _summary.json
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

from core import leverage_timing as lt  # noqa: E402
from core import shock_dip as sd  # noqa: E402
from core import shock_dip_perf as sp  # noqa: E402
from factory import regimes  # noqa: E402
from scripts.shock_dip_gate01 import CACHE, load_fred, md_table  # noqa: E402
from scripts.shock_dip_gate2 import load_close  # noqa: E402
from scripts.shock_dip_gate3 import START_KRW, asof_series, assets, run_variant  # noqa: E402

OUT = ROOT / "docs" / "research"
N_RANDOM = 1000
SEED = 20261017
US = [("S&P500", "^GSPC", "1977-01-01", "1998-12-31"), ("나스닥 종합", "^IXIC", "1977-01-01", "1998-12-31"),
      ("QQQ(보정)", "QQQ", "2000-01-01", "2021-12-31")]
FOREIGN = [("닛케이225", "^N225"), ("DAX", "^GDAXI"), ("FTSE100", "^FTSE"), ("항셍", "^HSI"), ("코스피", "^KS11")]
NAMES = {"R1": "국면 1.5배", "R2": "국면 2배", "M1": "금리차 1.5배", "M2": "금융여건 1.5배", "M3": "통화량 1.5배"}


def nfci_vintages() -> pd.DatetimeIndex:
    path = CACHE / "alfred_vintages_NFCI.csv"
    if not path.exists():
        html = subprocess.run(["curl", "-s", "-m", "200", "https://alfred.stlouisfed.org/series/downloaddata?seid=NFCI"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
        import re

        dates = sorted(set(re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', html)))
        if not dates:
            raise RuntimeError("NFCI 발표본 목록 없음")
        pd.Series(dates, name="vintage").to_csv(path, index=False)
    return pd.DatetimeIndex(pd.read_csv(path)["vintage"])


def nfci_vintage_value(vintage: pd.Timestamp) -> float:
    """그 발표본의 마지막 관측값(캐시)."""
    path = CACHE / "nfci_vintage_last.csv"
    cache = pd.read_csv(path, index_col=0)["value"].to_dict() if path.exists() else {}
    key = vintage.date().isoformat()
    if key not in cache:
        url = f"https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=NFCI&vintage_date={key}"
        txt = subprocess.run(["curl", "-s", "-m", "120", url], capture_output=True, text=True).stdout.strip().splitlines()
        vals = [ln.split(",")[1] for ln in txt[1:] if "," in ln and ln.split(",")[1] not in ("", ".")]
        if not vals:
            raise RuntimeError(f"NFCI 발표본 {key} 값 없음")
        cache[key] = float(vals[-1])
        pd.Series(cache, name="value").to_csv(path)
    return float(cache[key])


def m_decisions(v: str, d: pd.DatetimeIndex, pos: list[int], data: dict) -> tuple[list[float], dict]:
    """M1·M2·M3 판단일별 목표 비중(결측은 1배)과 기록(출처·수정값 사용 수)."""
    out, info = [], {"src": {}, "revised": 0, "missing": 0}
    for p in pos:
        day = d[p]
        if v == "M1":
            w, src = lt.m1_decision(day, data["t10y2y"], data["gs10"], data["gs2"])
            info["src"][src] = info["src"].get(src, 0) + 1
        elif v == "M2":
            vint = data["nfci_v"][data["nfci_v"] <= day]
            if len(vint):
                val = nfci_vintage_value(vint[-1])
            else:
                val = lt.nfci_current_asof(data["nfci"], day)
                info["revised"] += 1
            w = lt.m2_decision(val)
        else:
            w = lt.m3_decision(day, data["m2sl"])
        if np.isnan(w):
            info["missing"] += 1
            w = lt.W_1X
        out.append(w)
    return out, info


def window(close: pd.Series, start: str | None, end: str) -> tuple[pd.Series, int]:
    c = close[close.index <= pd.Timestamp(end)]
    w0 = 0 if start is None else int(np.searchsorted(c.index, pd.Timestamp(start)))
    return c, w0


def cagr_of(res, d, w0, tax: bool) -> float:
    days = (d[-1] - d[w0]).days
    if tax:
        return sp.cagr(START_KRW, res.end_after_tax_krw, days)
    return sp.cagr(res.nav[0], res.end_value, days)


def run_one(x, lev, d, w0, events, fx):
    pre = run_variant(x, lev, d, w0, events)
    out = {"pre": cagr_of(pre, d, w0, False), "mdd": sp.max_drawdown(pre.nav), "nav": pre.nav}
    if fx is not None:
        out["tax"] = cagr_of(run_variant(x, lev, d, w0, events, tax_fx=fx), d, w0, True)
    return out


def main() -> None:
    dtb3 = load_fred("DTB3")
    data = {"t10y2y": load_fred("T10Y2Y"), "gs10": load_fred("GS10"), "gs2": load_fred("GS2"),
            "nfci": load_fred("NFCI"), "m2sl": load_fred("M2SL"), "nfci_v": nfci_vintages()}
    fx_all = load_fred("DEXKOUS")
    rng = np.random.default_rng(SEED)
    rows, refs, regime_rows = [], [], []
    for name, tk, start, end in US:
        close = load_close(tk)
        if tk == "QQQ":
            close, _ = sd.correct_with_reference(load_close("QQQ"), load_close("^NDX"), 0.02)
        close, w0 = window(close, start, end)
        d = close.index
        x, lev, _, desc = assets(name, tk, close, end, dtb3)
        fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
        key = "tax" if fx is not None else "pre"
        base = run_one(x, lev, d, w0, [], fx)
        # 참고: 항상 1.5배·2배, 전략 4 방식(200일선 위 2배·아래 현금 DTB3)
        cash = (1 + (asof_series(dtb3, d).shift(1).fillna(0) / 100 / 252)).cumprod()
        above = (close > close.rolling(200).mean()).to_numpy()
        s4_ev = lt.events_from_daily_targets(np.where(above, lt.W_2X, lt.W_1X), w0)
        s4 = sp.run_book(cash.to_numpy()[w0:], lev.to_numpy()[w0:], [(p - w0, w) for p, w in s4_ev], cost=0.001, start_krw=1.0)
        for lab, ev in (("항상 1.5배", [(w0, lt.W_15X)]), ("항상 2배", [(w0, lt.W_2X)])):
            r = run_one(x, lev, d, w0, ev, fx)
            refs.append({"구간": name, "방식": lab, "세전 연복리": r["pre"], "세후 연복리(원화)": r.get("tax"), "최대 낙폭": r["mdd"]})
        refs.append({"구간": name, "방식": "전략 4 방식(200일선 위 2배·아래 현금)", "세전 연복리": cagr_of(s4, d, w0, False),
                     "세후 연복리(원화)": None, "최대 낙폭": sp.max_drawdown(s4.nav)})
        refs.append({"구간": name, "방식": "1배 보유", "세전 연복리": base["pre"], "세후 연복리(원화)": base.get("tax"), "최대 낙폭": base["mdd"]})
        reg = regimes.classify(close)
        for v in lt.VARIANTS:
            info = {}
            if v.startswith("R"):
                ev = lt.events_from_daily_targets(lt.regime_targets(close, lt.LEVEL[v]).to_numpy(), w0)
            else:
                pos = lt.month_end_positions(d, w0, len(d))
                dec, info = m_decisions(v, d, pos, data)
                ev = lt.events_from_monthly(pos, dec)
            r = run_one(x, lev, d, w0, ev, fx)
            expo = lt.exposure(ev, w0, len(d))[w0:]
            # 무작위 대조
            fakes = []
            for _ in range(N_RANDOM):
                if v.startswith("R"):
                    fe = lt.random_daily_events(list(expo != lt.W_1X), w0, lt.LEVEL[v], rng)
                else:
                    fe = lt.random_monthly_events([w != lt.W_1X for w in dec], pos, lt.LEVEL[v], rng)
                fr = run_variant(x, lev, d, w0, fe, tax_fx=fx, daily=False)
                fakes.append(cagr_of(fr, d, w0, fx is not None) if fx is not None else sp.cagr(1.0, fr.end_value, (d[-1] - d[w0]).days))
            fakes = np.array(fakes)
            actual = r[key]
            rows.append({"변형": v, "구간": name, "창": f"{d[w0].date()} ~ {d[-1].date()}", "기준": "세후 원화" if fx is not None else "세전",
                         "연복리": actual, "1배 연복리": base[key], "세전 연복리": r["pre"], "최대 낙폭": r["mdd"], "1배 최대 낙폭": base["mdd"],
                         "전환 횟수": len(ev), "레버리지 시간 비율": float((expo != lt.W_1X).mean()),
                         "무작위 95백분위": float(np.percentile(fakes, 95)), "무작위 백분위": float((fakes < actual).mean() * 100),
                         "M1 출처": "; ".join(f"{k} {n}" for k, n in info.get("src", {}).items()),
                         "수정값 사용 판단일": info.get("revised", 0), "자료 없음 판단일": info.get("missing", 0), "2배 자산": desc})
            if tk == "QQQ":
                sr = pd.Series(r["nav"]).pct_change().to_numpy()
                br = pd.Series(base["nav"]).pct_change().to_numpy()
                lab = reg.to_numpy()[w0:]
                for g in (regimes.UP, regimes.DOWN, regimes.CRASH):
                    m = np.r_[False, lab[:-1] == g]
                    if m.any():
                        regime_rows.append({"변형": v, "국면(전날)": g, "날 수": int(m.sum()),
                                            "전략 연율 평균(%)": float(np.nanmean(sr[m]) * 252 * 100),
                                            "1배 연율 평균(%)": float(np.nanmean(br[m]) * 252 * 100)})
    # 해외 (R1·R2)
    foreign = []
    for name, tk in FOREIGN:
        close, w0 = window(load_close(tk), None, "2021-12-31")
        d = close.index
        x, lev, _, _ = assets(name, tk, close, "2021-12-31", dtb3)
        base = run_one(x, lev, d, w0, [], None)
        for v in ("R1", "R2"):
            ev = lt.events_from_daily_targets(lt.regime_targets(close, lt.LEVEL[v]).to_numpy(), w0)
            r = run_one(x, lev, d, w0, ev, None)
            foreign.append({"변형": v, "지수": name, "창": f"{d[w0].date()} ~ {d[-1].date()}", "세전 연복리": r["pre"],
                            "1배 연복리": base["pre"], "최대 낙폭": r["mdd"], "1배 최대 낙폭": base["mdd"], "전환 횟수": len(ev)})
    res, fr = pd.DataFrame(rows), pd.DataFrame(foreign)
    verdict = judge(res, fr)
    write(res, fr, pd.DataFrame(refs), pd.DataFrame(regime_rows), verdict)


def judge(res: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    out = []
    for v in lt.VARIANTS:
        r = res[res["변형"] == v]
        c1 = bool((r["연복리"] > r["1배 연복리"]).all())
        c2 = bool((r["연복리"] > r["무작위 95백분위"]).all())
        c3 = bool((r["최대 낙폭"] >= r["1배 최대 낙폭"] - 0.10).all())
        row = {"변형": v, "이름": NAMES[v],
               "① 세 구간 > 1배": ("PASS" if c1 else "FAIL") + " (" + ", ".join(f"{g}:{'O' if a > b else 'X'}" for g, a, b in zip(r["구간"], r["연복리"], r["1배 연복리"])) + ")",
               "② 세 구간 > 무작위 95백분위": ("PASS" if c2 else "FAIL") + " (" + ", ".join(f"{g}:{p:.1f}" for g, p in zip(r["구간"], r["무작위 백분위"])) + ")",
               "③ 낙폭 1배 −10%p 이내": ("PASS" if c3 else "FAIL") + " (" + ", ".join(f"{g}:{'O' if a >= b - 0.10 else 'X'}" for g, a, b in zip(r["구간"], r["최대 낙폭"], r["1배 최대 낙폭"])) + ")"}
        ok = c1 and c2 and c3
        if v in ("R1", "R2"):
            f = fr[fr["변형"] == v]
            wins = f[f["세전 연복리"] > f["1배 연복리"]]["지수"].tolist()
            c4 = len(wins) >= 3
            row["④ 해외 5곳 중 3곳"] = f"{'PASS' if c4 else 'FAIL'} ({len(wins)}/5: {', '.join(wins) or '없음'})"
            ok = ok and c4
        else:
            row["④ 해외 5곳 중 3곳"] = "해당 없음"
        verdict = "PASS" if ok else "FAIL"
        if v == "M2" and int(r["수정값 사용 판단일"].sum()) > 0:
            verdict = f"참고({verdict}) — 수정값 사용 판단일 {int(r['수정값 사용 판단일'].sum())}개"
        row["종합"] = verdict
        out.append(row)
    return pd.DataFrame(out)


def _p(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def write(res, fr, refs, regs, verdict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res.to_csv(OUT / "leverage_timing_results.csv", index=False, encoding="utf-8-sig")
    show = res.copy()
    for c in ("연복리", "1배 연복리", "세전 연복리", "최대 낙폭", "1배 최대 낙폭", "레버리지 시간 비율", "무작위 95백분위"):
        show[c] = show[c].map(_p)
    show["무작위 백분위"] = show["무작위 백분위"].map(lambda v: f"{v:.1f}")
    cols = ["변형", "구간", "창", "기준", "연복리", "1배 연복리", "무작위 95백분위", "무작위 백분위", "최대 낙폭", "1배 최대 낙폭",
            "전환 횟수", "레버리지 시간 비율", "세전 연복리"]
    qqq = show[show["구간"] == "QQQ(보정)"][[c for c in cols if c not in ("구간",)]]
    f = fr.copy()
    for c in ("세전 연복리", "1배 연복리", "최대 낙폭", "1배 최대 낙폭"):
        f[c] = f[c].map(_p)
    rf = refs.copy()
    for c in ("세전 연복리", "세후 연복리(원화)", "최대 낙폭"):
        rf[c] = rf[c].map(_p)
    rg = regs.copy()
    for c in ("전략 연율 평균(%)", "1배 연율 평균(%)"):
        rg[c] = rg[c].map(lambda v: f"{v:+.1f}")
    notes = show[["변형", "구간", "M1 출처", "수정값 사용 판단일", "자료 없음 판단일"]]
    passed = verdict[verdict["종합"] == "PASS"]["변형"].tolist()
    md = [
        "# 레버리지 비율 조절 1 — 성적 (시험 5 판정)", "",
        f"> 규칙 `docs/prereg/leverage_timing.md`(태그 `leverage-timing-prereg`). 코드 커밋 `{commit}`. 실행 `python scripts/leverage_timing_run.py`.",
        f"> 종합: **{', '.join(passed) if passed else '합격 없음'}**", "",
        "## 구현 해석 (사전 등록에 숫자로 없는 부분 — 분석 선택)", "",
        "- 장부·비용·세금·합성 2배·^NDX 보정은 폭락 연구 관문 3과 같다(`core/shock_dip_perf.py`): 비중을 바꿀 때만 1배를 팔아 2배를 사거나 반대, 그 사이 매일 다시 맞추지 않음. 편도 0.1%.",
        "- 세금(QQQ 구간만 판정): 시작 원화 1억, DEXKOUS, 과세연도 원화 실현손익 통산 − 250만 원 → 22%, 다음 해 5월 첫 거래일 납부, 끝날 전부 판 것으로 보고 세후 원화 연복리.",
        "- R1·R2: 각 지수 자기 종가로 factory/regimes 국면(판단불가 = 1배). 창 첫날은 1배로 시작해 그날 판단부터.",
        "- M1·M2·M3: 매달 마지막 거래일 판단 → 다음 거래일 종가 체결(사전 등록 D를 M2·M3에도 적용, H②의 \"M은 월 단위\").",
        "- M1: T10Y2Y는 판단일 전날까지 값(FRED 게시가 다음 날). 없으면 GS10 − GS2 월별(월 m 값은 m+1월 1일부터).",
        "- M2: 판단일 이하 마지막 ALFRED 발표본(첫 발표본 2011-05-25)의 마지막 주 값. 그 전은 현재 값 중 (관측일 + 5일) ≤ 판단일인 마지막 값(주간 공개 시차 5일 가정) — 이 판단일은 \"수정값 사용\".",
        "- M3: 판단일에 알려진 마지막 달 L(월 m 값은 m+2월 1일부터)의 M2SL 전년비 > L−6개월의 전년비. M2SL은 현재 값(발표본 아님).",
        "- 자료가 없는 판단일은 1배.",
        "- ② 무작위 대조: 그 구간 전략의 레버리지 구간 길이 목록과 1배 구간 길이 목록을 각각 섞어 번갈아 다시 놓음(총 레버리지 날 수·길이 분포 유지). R은 거래일 단위, M은 판단 달 단위(판단일 다음 날 체결). 1000회, 씨앗 20261017. 비교는 QQQ 세후 원화·나머지 세전.",
        "- ③은 미국 세 구간 각각(세전 일별 가치). ④는 세전.",
        "- 참고 \"전략 4 방식\": 종가 > 200일선이면 다음 날 2배 100%, 아니면 현금(DTB3 일할 이자). 세전.",
        "", "## 변형별 판정", "", md_table(verdict), "",
        "## QQQ 2000~2021 (세후 원화로 판정)", "", md_table(qqq), "",
        "## 세 구간 전체", "", md_table(show[cols]), "",
        "## 해외 5곳 (R1·R2, 세전, 합성 2배)", "", md_table(f), "",
        "## 참고 보고 (판정 아님)", "", md_table(rf), "",
        "### QQQ 국면별 성적 (전날 국면 기준 일간 수익률 연율 평균, 세전)", "", md_table(rg), "",
        "### 자료 출처·수정값", "", md_table(notes), "",
    ]
    (OUT / "leverage_timing_results.md").write_text("\n".join(md), encoding="utf-8")
    q = res[res["구간"] == "QQQ(보정)"].set_index("변형")
    best = q["연복리"].idxmax()
    summary = {"name": "레버리지 비율 조절 1 (R1·R2·M1·M2·M3)", "verdict": ("합격: " + ", ".join(passed)) if passed else "합격 없음",
               "cagr": round(float(q.loc[best, "연복리"]) * 100, 2), "qqq_cagr": round(float(q.loc[best, "1배 연복리"]) * 100, 2),
               "random_pct": round(float(q.loc[best, "무작위 백분위"]), 1), "best_variant": best, "commit": commit,
               "verdict_table": verdict.to_dict(orient="records")}
    (OUT / "leverage_timing_results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("done", summary["verdict"], best)


if __name__ == "__main__":
    main()
