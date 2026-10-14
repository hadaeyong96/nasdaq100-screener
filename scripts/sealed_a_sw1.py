"""봉인 ⓐ 개봉 — 후보 ② 실적급등 다음날 매수·3달 보유(대형) 실행 (사전 등록: docs/sealed_a_plan.md, configs/sealed_a.yaml).

일회성 리포트 스크립트. 신호·매매·비용·세금·무작위는 SW1 코드(core/sw1.py, scripts/sw1_experiments.py) 그대로,
데이터는 scripts/sealed_a_sw1_data.py가 받은 data/cache/sealed_a/sw1/. 시작할 때 사전 등록 해시를 확인하고,
판정 불가 비율(⑦)이 10%를 넘으면 수익을 계산하지 않고 멈춘다(대기). 2026-09-30 뒤 행이 있으면 멈춘다.

실행: python -u -m scripts.sealed_a_sw1
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import sealed_a as sa  # noqa: E402
from core import sw1  # noqa: E402
from scripts import sealed_a_b1 as b1  # noqa: E402
from scripts import sealed_a_sw1_data as sd  # noqa: E402
from scripts import sw1_experiments as sx  # noqa: E402

RESULT_MD = ROOT / "docs" / "results" / "sealed_a_sw1.md"
FX_CSV = ROOT / "data" / "cache" / "sealed_a" / "USDKRW.csv"
SEED, ITERS, RATE = 20261013, 1000, 0.22
FIRST_YEAR, N_YEARS = 2022, 5
PERIODS = ((2022, 2023), (2024, 2026))
# docs/results/sw1.md (sw1-prereg 시도 1) 숫자 그대로
SW1_REF = {"n": 601, "ex_pre": "+0.73%", "ex_after": "+0.53%", "win": "48.4%", "random_pct": "99.9",
           "p": "+0.91% / +0.12%"}


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_bench() -> pd.DataFrame:
    """QQQ 시가·종가 (B1과 같은 파일, 2021-01-04 ~ 2026-09-30)."""
    d = pd.read_csv(b1.CACHE / "QQQ.csv", index_col=0, parse_dates=True)
    sa.check_seal(d, "QQQ")
    return pd.DataFrame({"open": d["Open"].astype(float), "close": d["Close"].astype(float)})


def load_fx(days: pd.DatetimeIndex) -> tuple[pd.Series, dict]:
    """USDKRW = yfinance KRW=X + FRED DEXKOUS 보완 (S-6). 거래일에 직전 값, 2021-12-01 전 값 없으면 멈춤."""
    from datetime import date

    from data import fx as fxmod

    if not FX_CSV.exists():
        prim = fxmod.fetch_usd_krw_range(date(2021, 11, 1), date(2026, 9, 30))
        fb = fxmod.fetch_fred_series_range("DEXKOUS", "2021-11-01", "2026-09-30")
        merged, _ = fxmod.merge_fx_with_fallback(prim, fb)
        s = pd.Series(merged, dtype=float)
        s.index = pd.to_datetime(s.index)
        s.sort_index().rename("usdkrw").to_csv(FX_CSV)
    s = pd.read_csv(FX_CSV, index_col=0, parse_dates=True)["usdkrw"].astype(float)
    sa.check_seal(s, "USDKRW")
    exact = int(days.isin(s.index).sum())
    fx = s.reindex(s.index.union(days)).ffill().reindex(days)
    need = days[days >= "2021-12-01"]
    if fx.loc[need].isna().any():
        raise SystemExit("[sealed-a sw1] 환율 결측 거래일 — 멈춤")
    return fx, {"rows": len(s), "trading_days": len(need), "same_day": exact, "first": s.index[0].date(),
                "last": s.index[-1].date()}


def load_prices() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """CIK(문자열) → 시가·종가·거래량 (price_check 사용 판정만). 출력: (가격 dict, price_check 표)"""
    pc = pd.read_csv(sd.CACHE / "price_check.csv")
    out = {}
    for _, r in pc[pc["use"]].iterrows():
        d = sd.fetch_yf(r["price_symbol"])
        if d is not None:
            out[str(int(r["cik"]))] = d
    return out, pc


# ── 계산 ─────────────────────────────────────────────────────────────────────


def stats(t: pd.DataFrame) -> dict:
    """매매 표 → 평균 초과(비용 전·뒤·세금 뒤), 승률, 구간·연도 (% 단위)."""
    by_year = t.groupby("year")["ex_after"].mean()
    per = [float(t.loc[(t["year"] >= a) & (t["year"] <= b), "ex_after"].mean() * 100) for a, b in PERIODS]
    return {"n": len(t), "ex_pre": t["ex_pre"].mean() * 100, "ex_cost": t["ex_cost"].mean() * 100,
            "ex_after": t["ex_after"].mean() * 100, "win": (t["ex_after"] > 0).mean() * 100,
            "ret_usd": t["ret_usd"].mean() * 100, "bench_usd": t["bench_usd"].mean() * 100,
            "median": t["ex_after"].median() * 100, "per": per, "by_year": by_year * 100,
            "pos_years": int((by_year.reindex(range(FIRST_YEAR, FIRST_YEAR + N_YEARS)) > 0).sum())}


def mean_without_top(ex: pd.Series, k: int = 5) -> float:
    """세금 뒤 초과 수익 큰 순 k건을 뺀 평균 (S-8, 세금 다시 계산 안 함)."""
    return float(ex.sort_values(ascending=False).iloc[k:].mean())


def verdict(S: dict, pct: float, top5: float, und_rate: float) -> dict:
    """판정 ①~⑦ (configs/sealed_a.yaml SW1_large.criteria)과 결과 이름."""
    ok = {1: S["ex_after"] > 0, 2: pct >= 97.5, 3: all(p > 0 for p in S["per"]), 4: S["pos_years"] >= 3,
          5: S["n"] >= 200, 6: top5 > 0, 7: und_rate <= 0.10}
    if not ok[7]:
        name = "WAIT"
    elif not ok[5]:
        name = "판정 불가"
    elif all(ok.values()):
        name = "PASS"
    else:
        name = "FAIL"
    return {"ok": ok, "name": name}


def pools_for(tr: pd.DataFrame, m: dict, cik_of: dict[str, str]) -> list[list[str]]:
    """매매마다 무작위 후보: 매수일 전 달 말 나스닥100 ∪ S&P100 명단의 티커 → 가격 있는 CIK, 신호 종목 자신 제외 (해석 14)."""
    out = []
    for b, me_t in zip(tr["buy_day"], tr["ticker"]):
        me = (pd.Timestamp(b) - pd.offsets.MonthEnd(1)).normalize()
        ts = m["ndx"].get(me, set()) | m["sp100"].get(me, set())
        out.append(sorted({cik_of[t] for t in ts if t in cik_of} - {me_t}))
    return out


def run() -> None:
    b1.verify_preregistration()
    bench = load_bench()
    days = bench.index
    k = sd.classify(days, bench["close"])
    cikt = pd.read_csv(sd.CACHE / "cik.csv")
    no_cik = cikt[cikt["cik"].isna()]
    v = k[k["status"] != "기간 밖"]
    judged = int((v["status"] == "판정").sum())
    und_ann = int(v["status"].isin(sd.UNDECIDED_STATUS).sum())
    und_est = int(sum(-(-n // 3) for n in no_cik["n_months"]))
    und = und_ann + und_est
    und_rate = und / (judged + und)
    print(f"[sealed-a sw1] 판정 {judged} · 판정 불가 {und} ({und_rate * 100:.2f}%)", flush=True)
    if und_rate > 0.10:
        raise SystemExit("[sealed-a sw1] 판정 불가 비율 > 10% — 수익 계산 안 함, 대기")

    prices, pc = load_prices()
    fx, fx_info = load_fx(days)
    sig = v[(v["status"] == "판정") & v["signal"]].copy()
    sig["d0"] = pd.to_datetime(sig["d0"])
    sig["sym"] = sig["ticker"]
    sig["ticker"] = sig["cik"].astype(int).astype(str)
    dup = int(sig.duplicated(["ticker", "d0"]).sum())
    sig = sig.drop_duplicates(["ticker", "d0"])
    costs = sw1.Costs()

    def trades(hold: int):
        tr, sk = sw1.make_trades(sig[["ticker", "d0", "sym", "in_ndx", "in_sp100"]], prices, days, hold)
        for _, r in tr.iterrows():  # 종목 × 날짜 가드
            if not (pd.Timestamp(r["d0"]) >= sd.D0_FIRST and pd.Timestamp(r["sell_day"]) <= sa.SEAL_LAST):
                raise SystemExit(f"[sealed-a sw1] 봉인 밖 매매 {r['sym']} {r['d0']} — 멈춤")
        return sx.price_trades(tr, bench, fx, costs, RATE), sk

    T, SK = trades(60)
    S = stats(T)
    m = sd.members()
    cik_of = {r["ticker"]: str(int(r["cik"])) for _, r in pc[pc["use"]].iterrows()}
    cands = sx.candidate_table(T, pools_for(T, m, cik_of), prices, fx, bench, costs)
    rmeans = sx.random_means(cands, ITERS, SEED, RATE, FIRST_YEAR, N_YEARS) * 100
    pct = sw1.percentile_of(S["ex_after"], rmeans)
    top5 = mean_without_top(T["ex_after"]) * 100
    V = verdict(S, pct, top5, und_rate)
    print(f"[sealed-a sw1] 판정 {V['name']} {V['ok']}", flush=True)
    refs = []
    for h in (20, 40):
        t2, _ = trades(h)
        refs.append((f"{h}거래일 보유", stats(t2)))
    refs.append(("나스닥100만 (반응일 명단)", stats(T[T["in_ndx"]])))
    refs.append(("S&P 100만 (반응일 명단)", stats(T[T["in_sp100"]])))
    gap_list = pd.read_csv(sd.CACHE / "gaps.csv") if (sd.CACHE / "gaps.csv").exists() else pd.DataFrame()
    info = {"judged": judged, "und": und, "und_ann": und_ann, "und_est": und_est, "und_rate": und_rate,
            "signals": int(len(sig) + dup), "dup": dup, "skipped": SK["skip"].value_counts().to_dict(),
            "exit": T["exit_reason"].value_counts().to_dict(), "fx": fx_info,
            "cands": [len(c["ret"]) for c in cands], "status": v["status"].value_counts().to_dict()}
    write_report(S, V, pct, rmeans, top5, refs, info, gap_list)


# ── 보고서 ───────────────────────────────────────────────────────────────────


def p(x: float, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


def write_report(S, V, pct, rmeans, top5, refs, info, gaps) -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    ok = lambda i: "충족" if V["ok"][i] else "미충족"  # noqa: E731
    yrs = range(FIRST_YEAR, FIRST_YEAR + N_YEARS)
    L = [
        "# 봉인 ⓐ 개봉 — 실적급등 다음날 매수·3달 보유(대형) 결과 (sealed-a)", "",
        "> 해시 기준 (LF 정규화 sha256, 태그 = 작업 폴더 = 스크립트 기록값 확인 후 실행)",
        f"> - `docs/sealed_a_plan.md` `{b1.PREREG['sealed-a-prereg']['docs/sealed_a_plan.md']}`",
        f"> - `configs/sealed_a.yaml` `{b1.PREREG['sealed-a-prereg']['configs/sealed_a.yaml']}`",
        "> - 근거 태그 `sw1-prereg`(신호·매매·비용·세금)·`l1b-prereg` 파일 해시도 같은 확인 통과",
        f"> 실행 코드 커밋 `{head}` · 개봉 2026-10-13 · 봉인 2022-01-01 ~ 2026-09-30", "",
        "- 대상: 시점별 나스닥100 ∪ S&P 100 (제출일 전 달 말 명단), 반응일 2022-01-01 ~, 매도 예정일이 2026-09-30을 넘으면 제외",
        "- 신호: 반응일 종가 +5% 이상, QQQ 대비 +5%p 이상, 거래량 직전 20일 평균 2배 이상 · 매수 다음 거래일 시가, 60거래일 뒤 종가 매도",
        f"- 비용: 수수료 0.07%·환전 0.1% (편도) · 세금 매도 연도별 장부 순이익 22% (공제 없음, 이익 비례) · 무작위 {ITERS}회, 시드 {SEED}",
        "",
        f"## 판정: **{V['name']}**", "",
        "| 번호 | 기준 | 값 | 결과 |", "| --- | --- | --- | --- |",
        f"| ① | 매매당 평균 초과(QQQ 대비, 비용·세금 뒤) > 0 | {p(S['ex_after'])} | {ok(1)} |",
        f"| ② | 무작위 1,000회 97.5백분위 이상 | {pct:.1f}백분위 (p = {1 - pct / 100:.3f}) | {ok(2)} |",
        f"| ③ | 2022~2023 · 2024~2026-09 모두 > 0 | {p(S['per'][0])} / {p(S['per'][1])} | {ok(3)} |",
        f"| ④ | 2022~2026 중 3개 연도 이상 > 0 | {S['pos_years']}/5년 | {ok(4)} |",
        f"| ⑤ | 매매 200건 이상 | {S['n']}건 | {ok(5)} |",
        f"| ⑥ | 상위 5건 제외 평균 > 0 | {p(top5)} | {ok(6)} |",
        f"| ⑦ | 판정 불가 비율 ≤ 10% | {info['und_rate'] * 100:.2f}% ({info['und']}/{info['judged'] + info['und']}) | {ok(7)} |",
        "",
        "## 핵심 숫자", "",
        "| 항목 | 값 |", "| --- | --- |",
        f"| 매매 수 | {S['n']} |",
        f"| 평균 수익: 신호 / QQQ (달러, 비용 전) | {p(S['ret_usd'])} / {p(S['bench_usd'])} |",
        f"| 평균 초과: 비용 전 | {p(S['ex_pre'])} |",
        f"| 평균 초과: 비용 뒤 (원화, 세금 전) | {p(S['ex_cost'])} |",
        f"| 평균 초과: 비용·세금 뒤 | {p(S['ex_after'])} |",
        f"| 중앙값 초과 (세금 뒤) | {p(S['median'])} |",
        f"| 승률 (세금 뒤 QQQ보다 나은 매매) | {S['win']:.1f}% |",
        f"| 무작위 평균 초과 중앙값 · 97.5백분위 | {p(float(np.median(rmeans)))} · {p(float(np.percentile(rmeans, 97.5)))} |",
        f"| 무작위 후보 수 (매매당) 중앙값 · 최소 | {int(np.median(info['cands']))} · {min(info['cands'])} |",
        "",
        "## 연도별 (반응일 연도, 세금 뒤 초과)", "",
        "| 연도 | " + " | ".join(str(y) for y in yrs) + " |", "| --- |" + " --- |" * N_YEARS,
        "| 평균 초과 | " + " | ".join(p(S["by_year"].get(y, np.nan)) for y in yrs) + " |",
        "", "- 2026은 반응일 1~7월 무렵까지 (60거래일 매도가 2026-09-30 안인 매매만)", "",
        "## 참고 보고 (판정 제외)", "",
        "| 보고 | 매매 | 초과 비용 전 | 초과 세금 뒤 | 승률 | 2022~23 / 2024~26 | 양수 연도 |", "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, r in refs:
        L.append(f"| {name} | {r['n']} | {p(r['ex_pre'])} | {p(r['ex_after'])} | {r['win']:.1f}% | "
                 f"{p(r['per'][0])} / {p(r['per'][1])} | {r['pos_years']}/5 |")
    L += [
        "", "### SW1 2012~2021 (sw1-prereg 시도 1)과 나란히", "",
        "| | SW1 2012~2021 | **봉인 ⓐ 2022~2026-09** |", "| --- | --- | --- |",
        f"| 매매 | {SW1_REF['n']} | **{S['n']}** |",
        f"| 평균 초과 비용 전 | {SW1_REF['ex_pre']} | **{p(S['ex_pre'])}** |",
        f"| 평균 초과 비용·세금 뒤 | {SW1_REF['ex_after']} | **{p(S['ex_after'])}** |",
        f"| 승률 | {SW1_REF['win']} | **{S['win']:.1f}%** |",
        f"| 무작위 백분위 | {SW1_REF['random_pct']} | **{pct:.1f}** |",
        f"| 두 구간 | {SW1_REF['p']} | **{p(S['per'][0])} / {p(S['per'][1])}** |",
        "",
        "## 판정 불가·공백", "",
        "| 항목 | 값 |", "| --- | --- |",
        f"| 대상 발표 (반응일 2022-01-01 ~ 2026-09-30) 판정 | {info['judged']} |",
        f"| 판정 불가 | {info['und']} (발표 {info['und_ann']} + CIK 없는 티커 추정 {info['und_est']}) |",
        f"| 상태별 | {', '.join(f'{k} {n}' for k, n in info['status'].items())} |",
        f"| 신호 (같은 CIK·같은 반응일 중복) | {info['signals']} ({info['dup']}) |",
        f"| 신호 중 매매 안 됨 | {', '.join(f'{k} {n}' for k, n in info['skipped'].items()) or '없음'} |",
        f"| 매도 사유 | {', '.join(f'{k} {n}' for k, n in info['exit'].items())} |",
        f"| 환율 | {info['fx']['first']} ~ {info['fx']['last']}, {info['fx']['rows']}행, 거래일 {info['fx']['trading_days']} 중 같은 날 값 {info['fx']['same_day']} (나머지 직전 값) |",
        "",
    ]
    if len(gaps):
        L += ["### 공백 종목", "", "| 티커 | 사유 | 발표 수 |", "| --- | --- | --- |"]
        L += [f"| {r['ticker']} | {r['status']} | {r['announcements']} |"
              for _, r in gaps.sort_values("announcements", ascending=False).iterrows()]
        L.append("")
    L += [
        "## 한계", "",
        "- 인수·상장폐지된 종목 중 yfinance에 가격이 남지 않은 것은 공백(Tiingo는 11월 한도 회복 뒤). 외국 기업(20-F·6-K)은 8-K 2.02가 없어 공백(SW1과 같은 처리)",
        "- 배당 미반영(신호·QQQ 같음), 슬리피지 없음(SW1 그대로)",
        "",
        "## 결과를 본 뒤 바꾼 점", "", "없음", "",
    ]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"[sealed-a sw1] 보고서 → {RESULT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    run()
