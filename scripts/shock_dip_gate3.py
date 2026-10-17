"""폭락 연구 관문 3 — 전략 성적 계산 (docs/prereg/shock_dip.md A~K + v2 + v3, 태그 shock-dip-prereg-v3).

관문 2와 같은 신호(core/shock_dip.py)로 거래를 만들고 core/shock_dip_perf.py로 성적을 낸다.
각 지수는 판정 구간 끝에서 자른다(그 뒤 데이터 없음). 결과를 본 뒤 규칙·숫자를 바꾸지 않는다.

실행: python scripts/shock_dip_gate3.py
출력: docs/research/shock_dip_results.md, shock_dip_results.csv, shock_dip_results_summary.json
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

from core import shock_dip as sd  # noqa: E402
from core import shock_dip_perf as sp  # noqa: E402
from scripts.shock_dip_gate01 import load_fred, load_yf, md_table  # noqa: E402
from scripts.shock_dip_gate2 import PERIODS, build, load_close, macro_inputs  # noqa: E402

OUT = ROOT / "docs" / "research"
COST = 0.001  # E: 편도 0.1%
START_KRW = 100_000_000  # [분석 선택] 시작 원화 1억(가상 장부와 같음) — 250만 원 공제 때문에 규모가 결과에 약간 영향
N_RANDOM = 1000
SEED = 20261017
US = ["S&P500", "나스닥 종합", "QQQ(보정)"]
FOREIGN = ["닛케이225", "DAX", "FTSE100", "항셍", "코스피"]
EVENT_YEARS = [1987, 1998, 2000, 2008, 2011, 2018, 2020]


def asof_series(s: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """일별 값(그날 이하 가장 최근 값)."""
    s = s.dropna().sort_index()
    return pd.Series(pd.merge_asof(pd.DataFrame({"d": days}), pd.DataFrame({"d": s.index, "v": s.to_numpy()}), on="d")["v"].to_numpy(), index=days)


def assets(name: str, tk: str, close_signal: pd.Series, end: str, dtb3: pd.Series):
    """(1배 가격, 2배 가격, 실제 2배 첫날 또는 None, 설명)."""
    d = close_signal.index
    if tk == "QQQ":
        raw = load_yf("QQQ")
        adj = raw["Adj Close"].dropna()
        _, bad = sd.correct_with_reference(raw["Close"].dropna(), load_close("^NDX"), 0.02)
        ndx = load_close("^NDX")
        r = adj.pct_change()
        rn = ndx.reindex(adj.index).pct_change()
        badset = adj.index.isin(bad)
        r[badset] = rn[badset]  # v2-5: 보정한 날은 ^NDX 일간 수익률
        r.iloc[0] = 0.0
        x = (1 + r).cumprod().reindex(d)
        qld = load_yf("QLD")["Adj Close"].dropna()
        lev, first_real = sp.splice_leveraged(x.pct_change().fillna(0.0), qld[qld.index <= pd.Timestamp(end)],
                                              asof_series(dtb3, d))
        return x, lev, first_real, "QQQ 수정주가(분배금 포함, ^NDX 보정) / QLD 수정주가(2006-06-22부터), 그 전 합성 2배"
    x = close_signal / close_signal.iloc[0]
    lev, _ = sp.splice_leveraged(x.pct_change().fillna(0.0), None, asof_series(dtb3, d))
    return x, lev, None, f"{tk} 가격지수 / 합성 2배"


def run_variant(x, lev, d, w0, events, *, tax_fx=None, daily=True):
    """창 [w0, 끝]에서 장부 하나. 이벤트 위치는 전체 기준 → 창 기준으로 옮긴다."""
    px, pl = x.to_numpy()[w0:], lev.to_numpy()[w0:]
    ev = [(p - w0, w) for p, w in events if p >= w0]
    dd = d[w0:]
    kw = {}
    if tax_fx is not None:
        kw = dict(tax=True, fx=tax_fx[w0:], start_krw=START_KRW, may_pay=sp.may_payment_positions(dd),
                  years=dd.year.to_numpy())
    else:
        kw = dict(start_krw=1.0)
    return sp.run_book(px, pl, ev, cost=COST, daily=daily, **kw)


def main() -> None:
    mi = macro_inputs()
    dtb3 = load_fred("DTB3")
    fx_all = load_fred("DEXKOUS")
    rng = np.random.default_rng(SEED)
    rows, ev_rows, rand_rows = [], [], []
    windows = {}
    for name, tk, start, end, foreign in PERIODS:
        close = load_close(tk)
        if tk == "QQQ":
            close, _ = sd.correct_with_reference(load_close("QQQ"), load_close("^NDX"), 0.02)
        close = close[close.index <= pd.Timestamp(end)]
        table, trades, starts = build(name, close, start, end, foreign, mi)
        d = close.index
        x, lev, first_real, desc = assets(name, tk, close, end, dtb3)
        above = (close > close.rolling(200).mean()).to_numpy()
        tr = pd.DataFrame(trades)
        fx = asof_series(fx_all, d).to_numpy() if tk == "QQQ" else None
        for v in sd.VARIANTS:
            if starts[v] is None:
                continue
            w0 = int(d.get_loc(pd.Timestamp(starts[v])))
            t = tr[tr["변형"] == v] if len(tr) else tr
            pos = [(int(d.get_loc(pd.Timestamp(r["체결일"]))),
                    int(d.get_loc(pd.Timestamp(r["청산 체결일"]))) if r["청산 체결일"] != "보유 중(데이터 끝)" else None)
                   for _, r in t.iterrows()]
            ev = sp.strategy_events(pos, len(d))
            sw = [(p + w0, w) for p, w in sp.switch_events(above[w0:])]  # 창 첫날(1배)에서 시작, 첫날 판단부터
            books = {"전략": ev, "1배 보유": [], "200일선 2배 전환": sw}
            res = {k: run_variant(x, lev, d, w0, e) for k, e in books.items()}
            res_tax = {k: run_variant(x, lev, d, w0, e, tax_fx=fx if fx is not None else np.ones(len(d)))
                       for k, e in books.items()}
            days = (d[-1] - d[w0]).days
            holds = [(xp if xp is not None else len(d) - 1) - e for e, xp in pos]
            row = {"지수": name, "변형": v, "창": f"{d[w0].date()} ~ {d[-1].date()}", "2배 자산": desc,
                   "진입 횟수": len(pos), "평균 보유일": round(float(np.mean(holds)), 1) if holds else 0.0,
                   "실제 2배 첫날": str(first_real.date()) if first_real is not None else "합성만"}
            for k in books:
                row[f"{k} 세전 연복리"] = sp.cagr(res[k].nav[0], res[k].end_value, days)
                row[f"{k} 최대 낙폭"] = sp.max_drawdown(res[k].nav)
                if fx is not None:
                    row[f"{k} 세후 연복리(원화)"] = sp.cagr(START_KRW, res_tax[k].end_after_tax_krw, days)
                else:
                    row[f"{k} 참고 세후 연복리(현지 통화, 환산 없음)"] = sp.cagr(START_KRW, res_tax[k].end_after_tax_krw, days)
            row["거래 비용 합계(시작 자본 대비 %)"] = res["전략"].costs * 100
            rows.append(row)
            # 사건별 표
            nav = res["전략"].nav
            base = res["1배 보유"].nav
            for e, xp in pos:
                if d[e].year in EVENT_YEARS and name in US:
                    xe = xp if xp is not None else len(d) - 1
                    sig = t[t["체결일"].astype(str) == str(d[e].date())].iloc[0]
                    ev_rows.append({"지수": name, "변형": v, "신호일": sig["신호일"], "체결일": d[e].date(),
                                    "청산 체결일": d[xe].date() if xp is not None else "보유 중(구간 끝)",
                                    "청산 사유": sig["청산 사유"],
                                    "거래 구간 전략(세전)": nav[xe - w0] / nav[e - w0] - 1,
                                    "같은 구간 1배": base[xe - w0] / base[e - w0] - 1})
            # 무작위 대조 (QQQ 구간만, 세후 원화)
            if tk == "QQQ":
                fakes = []
                for _ in range(N_RANDOM):
                    place = sp.random_placements(holds, w0 + 1, len(d) - 1, rng)
                    if place is None:
                        break
                    rb = run_variant(x, lev, d, w0, sp.strategy_events(place, len(d)), tax_fx=fx, daily=False)
                    fakes.append(sp.cagr(START_KRW, rb.end_after_tax_krw, days))
                actual = row["전략 세후 연복리(원화)"]
                fakes = np.array(fakes)
                rand_rows.append({"변형": v, "진입 횟수": len(pos), "가짜 전략 수": len(fakes),
                                  "실제 세후 연복리": actual,
                                  "가짜 95백분위": float(np.percentile(fakes, 95)) if len(fakes) else np.nan,
                                  "가짜 중앙값": float(np.median(fakes)) if len(fakes) else np.nan,
                                  "실제의 백분위": float((fakes < actual).mean() * 100) if len(fakes) else np.nan})
        windows[name] = (d[0].date(), d[-1].date())

    res = pd.DataFrame(rows)
    rnd = pd.DataFrame(rand_rows)
    verdict = judge(res, rnd)
    write(res, pd.DataFrame(ev_rows), rnd, verdict)


def judge(res: pd.DataFrame, rnd: pd.DataFrame) -> pd.DataFrame:
    """합격 기준 ①~④ (H, v2-7: QQQ만 세후 원화, 나머지 세전)."""
    out = []
    for v in sd.VARIANTS:
        r = res[res["변형"] == v].set_index("지수")
        c1_parts, c2_parts = [], []
        for nm in US:
            if nm not in r.index:
                c1_parts.append((nm, None))
                continue
            row = r.loc[nm]
            if nm == "QQQ(보정)":
                s, b = row["전략 세후 연복리(원화)"], row["1배 보유 세후 연복리(원화)"]
            else:
                s, b = row["전략 세전 연복리"], row["1배 보유 세전 연복리"]
            c1_parts.append((nm, s > b))
            c2_parts.append((nm, row["전략 최대 낙폭"] >= row["1배 보유 최대 낙폭"] - 0.10))
        c1 = all(p is True for _, p in c1_parts)
        c2 = all(p for _, p in c2_parts) and len(c2_parts) == 3
        wins = [nm for nm in FOREIGN if nm in r.index and r.loc[nm, "전략 세전 연복리"] > r.loc[nm, "1배 보유 세전 연복리"]]
        c3 = len(wins) >= 3
        rr = rnd[rnd["변형"] == v]
        c4 = bool(len(rr) and rr.iloc[0]["실제 세후 연복리"] > rr.iloc[0]["가짜 95백분위"])
        out.append({"변형": v,
                    "① 미국 3구간 연복리 > 1배": ("PASS" if c1 else "FAIL") + " (" + ", ".join(f"{n}:{'O' if p else 'X'}" for n, p in c1_parts) + ")",
                    "② 낙폭 1배 −10%p 이내": ("PASS" if c2 else "FAIL") + " (" + ", ".join(f"{n}:{'O' if p else 'X'}" for n, p in c2_parts) + ")",
                    "③ 해외 5곳 중 3곳 이상": f"{'PASS' if c3 else 'FAIL'} ({len(wins)}/5: {', '.join(wins) or '없음'})",
                    "④ 무작위 95백분위 초과": "PASS" if c4 else "FAIL",
                    "종합": "PASS" if (c1 and c2 and c3 and c4) else "FAIL"})
    return pd.DataFrame(out)


def pct(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def write(res, evs, rnd, verdict) -> None:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    res.to_csv(OUT / "shock_dip_results.csv", index=False, encoding="utf-8-sig")
    show = res.copy()
    for c in show.columns:
        if "연복리" in c or "낙폭" in c:
            show[c] = show[c].map(pct)
    show["거래 비용 합계(시작 자본 대비 %)"] = show["거래 비용 합계(시작 자본 대비 %)"].map(lambda v: f"{v:.2f}")
    cols_us = ["지수", "변형", "창", "진입 횟수", "평균 보유일", "전략 세전 연복리", "1배 보유 세전 연복리", "200일선 2배 전환 세전 연복리",
               "전략 최대 낙폭", "1배 보유 최대 낙폭", "200일선 2배 전환 최대 낙폭", "거래 비용 합계(시작 자본 대비 %)"]
    qqq = show[show["지수"] == "QQQ(보정)"][["변형", "창", "진입 횟수", "평균 보유일", "전략 세후 연복리(원화)", "1배 보유 세후 연복리(원화)",
                                            "200일선 2배 전환 세후 연복리(원화)", "전략 세전 연복리", "1배 보유 세전 연복리",
                                            "전략 최대 낙폭", "1배 보유 최대 낙폭", "거래 비용 합계(시작 자본 대비 %)"]]
    ref_cols = ["지수", "변형", "전략 참고 세후 연복리(현지 통화, 환산 없음)", "1배 보유 참고 세후 연복리(현지 통화, 환산 없음)",
                "200일선 2배 전환 참고 세후 연복리(현지 통화, 환산 없음)"]
    ev = evs.copy()
    for c in ["거래 구간 전략(세전)", "같은 구간 1배"]:
        ev[c] = ev[c].map(pct)
    rn = rnd.copy()
    for c in ["실제 세후 연복리", "가짜 95백분위", "가짜 중앙값"]:
        rn[c] = rn[c].map(pct)
    rn["실제의 백분위"] = rn["실제의 백분위"].map(lambda v: f"{v:.1f}")
    passed = verdict[verdict["종합"] == "PASS"]["변형"].tolist()
    md = [
        "# 폭락 연구 관문 3 — 전략 성적 (시험 6개 판정)", "",
        f"> 규칙: `docs/prereg/shock_dip.md` A~K + v2 + v3 (태그 `shock-dip-prereg`, `-v2`, `-v3`). 코드 커밋 `{commit}`.",
        "> 실행 `python scripts/shock_dip_gate3.py`. 신호 `core/shock_dip.py`, 장부 `core/shock_dip_perf.py`. 결과를 본 뒤 규칙·숫자 변경 금지.",
        f"> 종합 판정: **{', '.join(passed) if passed else '합격 없음'}**", "",
        "## 계산 방식 (사전 등록에 숫자로 없는 부분 — 분석 선택)", "",
        "- 평소 1배 100%. 진입 체결일 종가에 1배 50% + 2배 50%로 맞추고(1배를 팔아 2배를 삼), 청산 체결일 종가에 1배 100%로. 그 사이 매일 다시 맞추지 않는다.",
        "- 비용: 편도 0.1% × 거래액(파는 쪽·사는 쪽 각각). 200일선 2배 전환도 같은 비용.",
        "- 200일선 2배 전환: t일 종가 > 그날 200일선이면 t+1 종가에 2배 100%, 아니면 1배 100%(바뀔 때만 거래). 비교 창 첫날은 1배로 시작.",
        "- 2배 자산: QQQ 구간은 QLD 수정주가(2006-06-22부터), 그 전은 합성 2배. 나머지 지수는 전부 합성 2배 = 2 × 일간 수익률 − 연 0.95%/252 − 전날 DTB3/252(사전 등록 E).",
        "- 1배 자산: QQQ는 수정주가(분배금 재투자, 원천징수 미반영) + v2-5 보정. 지수는 가격지수(배당 없음, DAX만 총수익 지수).",
        f"- 세금(QQQ 구간만 판정, v2-7): 시작 원화 {START_KRW:,}원, DEXKOUS 원/달러. 과세연도 원화 실현손익 통산 − 250만 원 → 22%, 다음 해 5월 첫 거래일에 1배부터 팔아 냄. "
        "끝날은 전부 판 것으로 보고(매도 비용 포함) 그해 세금을 뺀 원화 값으로 연복리. 1배 보유도 끝날 같은 방식으로 청산·세금.",
        "- 나머지 구간의 「참고 세후」는 같은 규칙을 현지 통화 그대로(환율 환산 없이, 공제 250만을 현지 통화 단위로) 적용한 참고값이다 — 판정에 쓰지 않음.",
        "- 최대 낙폭은 세전 일별 가치 기준. 연복리는 달력일 365.25 기준. 창 = 그 변형의 첫 판정 가능일 ~ 구간 끝(관문 2 표), 비교 대상도 같은 창.",
        "- 합격 ②는 미국 3개 구간 각각에서 적용(B3은 S&P·나스닥 1986~). ④ 무작위 대조는 QQQ 창에서 거래별 보유 일수 목록 그대로, 순서를 섞고 시작일만 겹치지 않게(다음 진입 ≥ 앞 청산 + 1거래일) 1000회, 같은 50%+50%(1.5배)·비용·세금. 씨앗 20261017.",
        "", "## 변형별 판정", "", md_table(verdict), "",
        "## QQQ 1999~2021 (세후 원화로 판정)", "", md_table(qqq), "",
        "## 무작위 날짜 대조 (QQQ, 세후 원화 연복리)", "", md_table(rn), "",
        "## 전체 구간 세전 성적", "", md_table(show[cols_us]), "",
        "## 참고 세후 (판정에 안 씀)", "", md_table(show[show["지수"] != "QQQ(보정)"][ref_cols]), "",
        "## 사건별 기록 (미국 구간, 해당 연도에 체결한 거래)", "",
        "> 거래 구간 = 체결일 종가 → 청산 체결일 종가. 전략 수익률은 장부 전체(1배 50% + 2배 50%) 세전. 「같은 구간 1배」는 1배 보유 장부.", "",
        md_table(ev) if len(ev) else "(없음)", "",
    ]
    (OUT / "shock_dip_results.md").write_text("\n".join(md), encoding="utf-8")
    q = res[(res["지수"] == "QQQ(보정)") & (res["변형"] == "A")]
    summary = {
        "name": "폭락 연구 1 충격 웅덩이 (A·B1·B2·B3·B4·D)",
        "verdict": "합격: " + ", ".join(passed) if passed else "합격 없음",
        "cagr": round(float(q["전략 세후 연복리(원화)"].iloc[0]) * 100, 2) if len(q) else None,
        "qqq_cagr": round(float(q["1배 보유 세후 연복리(원화)"].iloc[0]) * 100, 2) if len(q) else None,
        "random_pct": float(rnd[rnd["변형"] == "A"]["실제의 백분위"].iloc[0]) if len(rnd[rnd["변형"] == "A"]) else None,
        "commit": commit,
        "verdict_table": verdict.to_dict(orient="records"),
        "random": rnd.to_dict(orient="records"),
    }
    (OUT / "shock_dip_results_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("done", summary["verdict"])


if __name__ == "__main__":
    main()
