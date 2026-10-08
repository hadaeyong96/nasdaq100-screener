"""E1 이익 성장주 적립 검증 (사전 등록: docs/e1_plan.md, configs/e1_preregistration.yaml, tag e1-prereg).

일회성 리포트 스크립트. 분기 EPS·TTM·성장·매도 판정과 장부는 core/e1.py(순수 함수), 무작위 대조군은
core/moat_backtest.py(feature/moat), EDGAR 수집은 data/edgar.py(feature/moat), QQQM·환율은 D1과 같은 L1 체크포인트.
시작할 때 사전 등록 두 파일의 sha256(LF 정규화)을 확인하고 다르면 멈춘다. SEC_USER_AGENT는 존재 여부만 출력한다.

실행: python -u -m scripts.e1_experiments --check-data   (데이터 검사표만)
      python -u -m scripts.e1_experiments                (전체)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import e1  # noqa: E402
from core import moat_backtest as mb  # noqa: E402

PREREG_FILES = {
    "docs/e1_plan.md": "a9450c24be43e3eee31dfaca64a1d02bf251b1e1334c136b508d69072af1fa73",
    "configs/e1_preregistration.yaml": "5e622816cb1a5842cb6b74f74bcf471e8af696e23c1e659fd18db18c552846d3",
}
OUT_DIR = ROOT / "outputs"
RESULT_MD = ROOT / "docs" / "results" / "e1.md"
PRICE_CACHE = ROOT / "data" / "cache" / "e1" / "prices"
SEAL = pd.Timestamp("2021-12-31")
CANDS = {"E1a": 1, "E1b": 5, "E1c": 10}
QQQM = "QQQM"
REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"]


def normalized_sha256(path: Path) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256 (Windows 체크아웃의 CRLF 변환에 흔들리지 않게)."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    for rel, expected in PREREG_FILES.items():
        got = normalized_sha256(ROOT / rel)
        if got != expected:
            raise SystemExit(f"[e1] 사전 등록 파일이 바뀌었습니다: {rel} (기대 {expected[:12]}…, 실제 {got[:12]}…) — 멈춤")
    print("[e1] 사전 등록 해시 확인 통과", flush=True)
    return yaml.safe_load((ROOT / "configs" / "e1_preregistration.yaml").read_text(encoding="utf-8"))


# ── 데이터 ───────────────────────────────────────────────────────────────────


def load_qqqm():
    """D1과 같은 L1 체크포인트(engine.portfolio.prepare_data, 2021-12-31 봉인): 달력·QQQM·환율."""
    from engine import backtest as bt
    from scripts import l1_experiments as l1x

    ctx = l1x.load_all(bt.load_config(), {"periods": {"full": {"start": "1999-03-10", "end": "2021-12-31"}}})
    data = ctx["data"]
    days = data.core_df.index
    fx = pd.Series({pd.Timestamp(k): v for k, v in data.fx_by_date.items()}, dtype=float).reindex(days)
    return days, data.core_df["close"], data.core_dividends, fx


def calendar(days: pd.DatetimeIndex, first_month: str, last_month: str):
    """매수일(매달 첫 거래일)과 판단일(그 전 거래일)."""
    months = pd.period_range(first_month, last_month, freq="M")
    per = days.to_period("M")
    buys, decisions = [], []
    for m in months:
        d = days[per == m][0]
        buys.append(d)
        decisions.append(days[days.get_loc(d) - 1])
    return buys, decisions


def universe_by_decision(decisions) -> dict:
    from data import universe_history as uh

    cur = set(pd.read_csv(ROOT / "data" / "universe_fallback.csv", comment="#")["ticker"])
    return {d: sorted(uh.point_in_time_universe(d.date(), cur)) for d in decisions}


def load_edgar(tickers: list[str]) -> tuple[dict, dict, list, list]:
    """티커 → companyfacts. 출력: (facts, cik, 실패 [(티커, 이유)], 외국 기업 목록)."""
    from data import edgar

    ua_present = bool(__import__("os").environ.get("SEC_USER_AGENT", "").strip())
    print(f"[e1] SEC_USER_AGENT 있음: {ua_present}", flush=True)
    cik_map = edgar.fetch_ticker_to_cik()
    facts, ciks, failed, foreign = {}, {}, [], []
    for t in tickers:
        cik = cik_map.get(t.upper().replace("-", "."), cik_map.get(t.upper()))
        if cik is None:
            failed.append((t, "CIK 없음(현재 SEC 목록에 티커 없음)"))
            continue
        try:
            f = edgar.fetch_company_facts(cik)
        except edgar.EdgarFetchError as exc:
            failed.append((t, f"수집 실패: {exc}"))
            continue
        forms = {e.get("form") for tag in f.get("facts", {}).get("us-gaap", {}).values() for u in tag.get("units", {}).values() for e in u}
        forms |= {e.get("form") for tag in f.get("facts", {}).get("ifrs-full", {}).values() for u in tag.get("units", {}).values() for e in u}
        eps = edgar.extract_fact_entries(f, "us-gaap", "EarningsPerShareDiluted")
        if forms & {"20-F", "40-F"} or not any(e.get("form") == "10-Q" for e in eps):
            foreign.append(t)
            continue
        facts[t], ciks[t] = f, cik
    return facts, ciks, failed, foreign


def load_prices(tickers: list[str]) -> tuple[dict, list]:
    """yfinance Close·배당·분할(auto_adjust=False), 캐시. 출력: ({티커: DataFrame(close, div, split)}, 못 받은 목록)."""
    out, missing = {}, []
    PRICE_CACHE.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        path = PRICE_CACHE / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        else:
            import yfinance as yf

            try:
                h = yf.Ticker(t).history(start="2011-06-01", end="2022-01-01", auto_adjust=False, actions=True)
            except Exception:  # noqa: BLE001 — 실패 종목은 모아서 보고한다
                h = pd.DataFrame()
            if h is None or h.empty or "Close" not in h:
                df = pd.DataFrame(columns=["close", "div", "split"])
            else:
                h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
                df = pd.DataFrame({"close": h["Close"], "div": h.get("Dividends", 0.0), "split": h.get("Stock Splits", 0.0)})
            df.to_csv(path)
        if df.empty:
            missing.append(t)
            continue
        df.index = pd.DatetimeIndex(df.index)
        df = df[df.index <= SEAL]
        if (df.index > SEAL).any():
            raise SystemExit(f"[e1] {t} 2022 이후 가격 행 — 멈춤")
        df = df.dropna(subset=["close"])
        if df.empty:
            missing.append(t)
            continue
        out[t] = df
    return out, missing


def splits_of(df: pd.DataFrame) -> pd.Series:
    s = df["split"].fillna(0.0)
    return s[s > 0].astype(float)


def build_market(days, prices: dict, qqqm_close, qqqm_div, fx) -> e1.Market:
    cols = {t: df["close"].reindex(days) for t, df in prices.items()}
    cols[QQQM] = qqqm_close.reindex(days)
    close = pd.DataFrame(cols)
    divs = {t: df["div"].fillna(0.0).reindex(days).fillna(0.0) for t, df in prices.items()}
    divs[QQQM] = qqqm_div.reindex(days).fillna(0.0)
    div_cum = pd.DataFrame(divs).cumsum()
    last = {t: close[t].last_valid_index() for t in close.columns}
    return e1.Market(days=days, close=close, div_cum=div_cum, last_date=last, fx=fx, buy_days=[])


def data_check(facts, prices, failed, foreign, missing, tickers, prereg) -> tuple[list, list]:
    from data import edgar

    rows = [
        {"항목": "과거 명단 종목(2011-12~2021-12 판단일 합집합)", "값": len(tickers)},
        {"항목": "EDGAR 성공(미국 분기 공시)", "값": len(facts)},
        {"항목": "EDGAR 실패", "값": f"{len(failed)}: " + ", ".join(t for t, _ in failed)},
        {"항목": "외국 기업(20-F·40-F 또는 10-Q EPS 없음)", "값": f"{len(foreign)}: " + ", ".join(foreign)},
        {"항목": "가격 못 받음", "값": f"{len(missing)} ({len(missing) / len(tickers) * 100:.1f}%): " + ", ".join(missing)},
        {"항목": "2022 이후 가격 행", "값": int(sum((df.index > SEAL).sum() for df in prices.values()))},
    ]
    checks = []
    for kv in prereg["data_checks"]["known_values"]:
        t = kv["ticker"]
        if t not in facts:
            checks.append({"항목": f"{t} {kv['kind']}", "계산": "데이터 없음", "기대": kv.get("value"), "통과": False})
            continue
        if kv["kind"] == "annual_eps":
            ents = [e for e in edgar.extract_fact_entries(facts[t], "us-gaap", "EarningsPerShareDiluted")
                    if e.get("start") == kv["start"] and e.get("end") == kv["end"] and e.get("form") in e1.FORMS]
            first = sorted(ents, key=lambda e: e["filed"])[0] if ents else None
            got = float(first["val"]) if first else None
            checks.append({"항목": f"{t} 연간 희석 EPS {kv['start']}~{kv['end']} (최초 공시 {first['filed'] if first else '-'})",
                           "계산": got, "기대": kv["value"], "통과": got is not None and abs(got - kv["value"]) <= kv["tol"]})
        else:
            ents = edgar.extract_fact_entries(facts[t], "us-gaap", "EarningsPerShareDiluted")
            sp = splits_of(prices[t]) if t in prices else None
            # 사용자 결정(2026-10-07, 결과 계산 전): 판단일마다 그날 주식 수 기준인 TTM을 마지막 판단일 주식 수
            # 기준으로 환산해 이웃 비율을 본다(등록 문구 그대로면 분할 종목은 정의상 항상 실패). 원래 값도 같이 적는다.
            last_d = pd.Timestamp(kv["decision_days"][-1])
            raw, common = [], []
            for d in kv["decision_days"]:
                q = e1.quarterly_values(ents, pd.Timestamp(d), sp)
                tail = e1.consecutive_tail(q, 4)
                v = float(tail.sum()) if tail is not None else np.nan
                f = float(np.prod(sp[(sp.index > pd.Timestamp(d)) & (sp.index <= last_d)].to_numpy())) if sp is not None and len(sp) else 1.0
                raw.append(v)
                common.append(v / f)
            ratios = [b / a for a, b in zip(common, common[1:])]
            ok = all(kv["ratio_range"][0] <= r <= kv["ratio_range"][1] for r in ratios)
            checks.append({"항목": f"{t} 분할 보정 TTM {', '.join(kv['decision_days'])} (분할 {', '.join(str(x.date()) for x in sp.index[sp.index.year == 2021]) if sp is not None else '-'})",
                           "계산": "판단일 기준 " + " → ".join(f"{x:.2f}" for x in raw) + " / 마지막 판단일 주식 수 기준 " + " → ".join(f"{x:.2f}" for x in common),
                           "기대": f"같은 주식 수 기준 이웃 비율 {kv['ratio_range'][0]}~{kv['ratio_range'][1]}", "통과": ok})
    return rows, checks


# ── 월별 대상·순위·매도 판정 ────────────────────────────────────────────────


def monthly_decisions(decisions, buys, uni, facts, prices, metric: str = "eps") -> list[dict]:
    """판단일마다 대상(성장 조건 전)·순위·매도 판정·대상 수 (core.e1 순수 함수로)."""
    from data import edgar

    entries = {}
    for t in facts:
        if t not in prices:
            continue
        if metric == "eps":
            entries[t] = edgar.extract_fact_entries(facts[t], "us-gaap", "EarningsPerShareDiluted")
        else:
            entries[t] = [edgar.extract_fact_entries(facts[t], "us-gaap", tag) for tag in REV_TAGS]
    out = []
    for dd, bd in zip(decisions, buys):
        infos = {}
        for t, ents in entries.items():
            sp = splits_of(prices[t]) if metric == "eps" else None
            q = e1.quarterly_values(ents, dd, sp) if metric == "eps" else e1.merge_revenue(ents, dd)
            infos[t] = e1.growth_info(q) if len(q) else None
        u = uni[dd]
        us = [t for t in u if t in facts]
        priced = [t for t in us if t in prices and dd in prices[t].index and bd in prices[t].index]
        pool = sorted(t for t in priced if infos.get(t) is not None and infos[t].pool_ok)
        ranked = e1.rank_growth(infos, pool)
        out.append({
            "decision": dd, "buy": bd, "pool": pool, "ranked": ranked,
            "growth": {t: infos[t].growth for t in ranked},
            "sell": {t: e1.should_sell(infos.get(t)) for t in entries},
            "counts": {"명단": len(u), "미국 공시": len(us), "가격 있음": len(priced), "분기 16개(대상)": len(pool), "성장 조건 통과": len(ranked)},
        })
    return out


PERIODS = [("2012~2021", 0, 120, "2021-12-31"), ("2012~2015", 0, 48, "2015-12-31"), ("2016~2021", 48, 120, "2021-12-31")]


def run_strategy(market: e1.Market, dec: list[dict], lo: int, hi: int, valuation: str, picks: list[list[str]], *,
                 hold_qqqm: bool = False, apply_tax: bool = True, daily: bool = True) -> dict:
    market.buy_days = [m["buy"] for m in dec[lo:hi]]
    flags = [{} if hold_qqqm else m["sell"] for m in dec[lo:hi]]
    return e1.simulate(market, picks[lo:hi], flags, pd.Timestamp(valuation), apply_tax=apply_tax, daily=daily)


def summarize(res: dict, pre: dict, valuation: str, first_buy) -> dict:
    years = (pd.Timestamp(valuation) - pd.Timestamp(first_buy)).days / 365.25
    uv = res.get("unit_value")
    twr = (float(uv.iloc[-1]) ** (1 / years) - 1) * 100 if uv is not None else None
    return {"contributed": res["contributed"], "pretax": pre["final"], "posttax": res["final"], "mdd": res.get("mdd_pct"),
            "twr": twr, "sells": res["sells"], "avg_hold_days": float(np.mean(res["holding_days"])) if res["holding_days"] else None,
            "tax": sum(res["taxes"].values()), "min_cash": res["min_cash"], "res": res}


def run_all(prereg, days, qqqm_close, qqqm_div, fx, buys, decisions, uni, facts, prices, failed, foreign, missing, tickers, rows, checks) -> None:
    print("[e1] 판단일별 대상·순위 계산(EPS) ...", flush=True)
    dec = monthly_decisions(decisions, buys, uni, facts, prices, "eps")
    print("[e1] 판단일별 대상·순위 계산(매출, 참고) ...", flush=True)
    dec_rev = monthly_decisions(decisions, buys, uni, facts, prices, "revenue")
    market = build_market(days, {t: prices[t] for t in prices}, qqqm_close, qqqm_div, fx)
    if (market.close.index > SEAL).any():
        raise SystemExit("[e1] 2022 이후 가격 행이 장부에 들어감 — 멈춤")

    picks = {"E0": [[QQQM] for _ in dec]}
    for c, n in CANDS.items():
        picks[c] = [m["ranked"][:n] for m in dec]
    results = {}
    for name, lo, hi, val in PERIODS:
        results[name] = {}
        for c in ["E0", *CANDS]:
            dd = dec if c == "E0" else dec
            r = run_strategy(market, dd, lo, hi, val, picks[c], hold_qqqm=(c == "E0"))
            p = run_strategy(market, dd, lo, hi, val, picks[c], hold_qqqm=(c == "E0"), apply_tax=False, daily=False)
            if r["min_cash"] < -1e-6 or r["contributed"] != 300_000 * (hi - lo):
                raise SystemExit(f"[e1] 점검 실패: {name} {c} 현금 {r['min_cash']} 납입 {r['contributed']}")
            results[name][c] = summarize(r, p, val, dec[lo]["buy"])
        print(f"[e1] {name} 완료", flush=True)

    print("[e1] 무작위 대조군 3 × 1,000회 ...", flush=True)
    seed = prereg["candidates"]["random"]["seed"]
    rand = {}
    for c, n in CANDS.items():
        rng = random.Random(seed)
        vals = []
        for _ in range(prereg["candidates"]["random"]["iterations"]):
            pk = [mb.draw_random_portfolio(m["pool"], n, rng) for m in dec]
            vals.append(run_strategy(market, dec, 0, 120, "2021-12-31", pk, daily=False)["final"])
        rand[c] = vals
        print(f"[e1]   {c} 무작위 완료", flush=True)

    rev = {}
    for c, n in CANDS.items():
        pk = [m["ranked"][:n] for m in dec_rev]
        rev[c] = {}
        for name, lo, hi, val in PERIODS:
            r = run_strategy(market, dec_rev, lo, hi, val, pk)
            p = run_strategy(market, dec_rev, lo, hi, val, pk, apply_tax=False, daily=False)
            rev[c][name] = summarize(r, p, val, dec_rev[lo]["buy"])

    verdict = judge(results, rand)
    write_outputs(prereg, dec, dec_rev, results, rand, rev, verdict, rows, checks, failed, foreign, missing, tickers)


def judge(results: dict, rand: dict) -> dict:
    out = {}
    e0 = {k: results[k]["E0"] for k in results}
    for c in CANDS:
        r = {k: results[k][c] for k in results}
        pct = mb.percentile_rank(r["2012~2021"]["posttax"], rand[c])
        ok = {
            1: r["2012~2021"]["posttax"] >= e0["2012~2021"]["posttax"] * 1.10,
            2: pct >= 75,
            3: r["2012~2015"]["posttax"] >= e0["2012~2015"]["posttax"] and r["2016~2021"]["posttax"] >= e0["2016~2021"]["posttax"],
            4: r["2012~2021"]["mdd"] >= e0["2012~2021"]["mdd"] - 10.0,
        }
        out[c] = {"ok": ok, "pass": all(ok.values()), "pct": pct}
    return out


# ── 출력 ─────────────────────────────────────────────────────────────────────


def _f(x, nd=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if isinstance(x, (bool, np.bool_)):
        return "예" if x else "아니오"
    if isinstance(x, (float, np.floating)):
        return f"{x:,.{nd}f}"
    return str(x)


def md_table(rows: list[dict]) -> str:
    if not rows:
        return "(없음)\n"
    cols = list(rows[0])
    return "\n".join(["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
                     + ["| " + " | ".join(_f(r[c]) for c in cols) + " |" for r in rows]) + "\n"


def cand_rows(results: dict, rand: dict | None) -> list[dict]:
    rows = []
    for name in results:
        e0 = results[name]["E0"]["posttax"]
        for c, s in results[name].items():
            row = {"구간": name, "후보": c, "총 납입": s["contributed"], "세전 최종": s["pretax"], "세후 최종": s["posttax"],
                   "E0 대비 %": (s["posttax"] / e0 - 1) * 100, "MDD %": s["mdd"], "연 시간가중 %": s["twr"],
                   "매도 횟수": s["sells"], "평균 보유(일)": s["avg_hold_days"], "세금 합계": s["tax"]}
            if rand is not None:
                row["무작위 백분위"] = mb.percentile_rank(s["posttax"], rand[c]) if (c in rand and name == "2012~2021") else None
            rows.append(row)
    return rows


def make_plots(results: dict, rand: dict, verdict: dict) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"E0": "#555555", "E1a": "#d62728", "E1b": "#1f77b4", "E1c": "#2ca02c"}
    full = results["2012~2021"]
    paths = []
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for c, s in full.items():
        uv = s["res"]["unit_value"]
        axes[0].plot(uv.index, uv.values, color=colors[c], lw=1, label=f"{c} 세후 {s['posttax'] / 1e6:,.1f}백만 원")
        axes[1].plot(uv.index, (uv / uv.cummax() - 1) * 100, color=colors[c], lw=0.8, label=f"{c} MDD {s['mdd']:.1f}%")
    axes[0].set_yscale("log")
    axes[0].set_title("단위 가치(시간 가중, 로그) — 2012-01 ~ 2021-12, 매달 30만 원")
    axes[1].set_title("단위 가치 낙폭 %")
    for ax in axes:
        ax.legend(fontsize=8, loc="upper left" if ax is axes[0] else "lower left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT_DIR / "e1_unit_value.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for ax, c in zip(axes, CANDS):
        v = np.array(rand[c]) / 1e6
        ax.hist(v, bins=40, color="#bbbbbb")
        ax.axvline(full[c]["posttax"] / 1e6, color=colors[c], lw=2, label=f"{c} ({verdict[c]['pct']:.1f}백분위)")
        ax.axvline(full["E0"]["posttax"] / 1e6, color="#555555", lw=1.5, ls="--", label="E0")
        ax.axvline(np.percentile(v, 75), color="black", lw=0.8, ls=":", label="무작위 75백분위")
        ax.set_title(f"{c} 무작위 대조군 1,000회 세후 최종(백만 원)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = OUT_DIR / "e1_random_distribution.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    paths.append(p)
    return paths


def write_outputs(prereg, dec, dec_rev, results, rand, rev, verdict, rows, checks, failed, foreign, missing, tickers) -> None:
    import subprocess
    from collections import Counter

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    tag = subprocess.run(["git", "rev-parse", "e1-prereg"], capture_output=True, text=True, cwd=ROOT).stdout.strip()

    main_rows = cand_rows(results, rand)
    pd.DataFrame(main_rows).to_csv(OUT_DIR / "e1_summary.csv", index=False, encoding="utf-8-sig")
    rand_rows = []
    for c in CANDS:
        v = np.array(rand[c])
        rand_rows.append({"후보": c, "후보 세후": results["2012~2021"][c]["posttax"], "무작위 평균": v.mean(), "무작위 중앙": float(np.median(v)),
                          "무작위 10%": float(np.percentile(v, 10)), "무작위 90%": float(np.percentile(v, 90)),
                          "무작위 75백분위 값": float(np.percentile(v, 75)), "후보 백분위": verdict[c]["pct"],
                          "E0 세후": results["2012~2021"]["E0"]["posttax"], "E0 백분위(참고)": mb.percentile_rank(results["2012~2021"]["E0"]["posttax"], rand[c])})
    pd.DataFrame(rand_rows).to_csv(OUT_DIR / "e1_random.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({c: rand[c] for c in CANDS}).to_csv(OUT_DIR / "e1_random_runs.csv", index=False, encoding="utf-8-sig")

    top1 = [{"매수일": m["buy"].date(), "1위": (m["ranked"][0] if m["ranked"] else "없음(현금)"),
             "증가율 %": (m["growth"][m["ranked"][0]] * 100 if m["ranked"] else None), "대상 수": len(m["pool"]), "성장 통과 수": len(m["ranked"])} for m in dec]
    pd.DataFrame(top1).to_csv(OUT_DIR / "e1_top1_monthly.csv", index=False, encoding="utf-8-sig")
    turn_rows, most_rows = [], []
    for c, n in CANDS.items():
        sets = [tuple(m["ranked"][:n]) for m in dec]
        changes = sum(1 for a, b in zip(sets, sets[1:]) if set(a) != set(b))
        turn_rows.append({"후보": c, "종목 구성이 바뀐 달": changes, "비율 %": changes / (len(sets) - 1) * 100,
                          "매도 횟수(2012~2021)": results["2012~2021"][c]["sells"]})
        cnt = Counter(t for s in sets for t in s)
        for t, k in cnt.most_common(10):
            most_rows.append({"후보": c, "종목": t, "담은 달 수": k})
    pd.DataFrame(most_rows).to_csv(OUT_DIR / "e1_most_held.csv", index=False, encoding="utf-8-sig")
    counts = pd.DataFrame([{"매수일": m["buy"].date(), **m["counts"]} for m in dec])
    counts.to_csv(OUT_DIR / "e1_target_counts.csv", index=False, encoding="utf-8-sig")
    count_summary = []
    for col in ["명단", "미국 공시", "가격 있음", "분기 16개(대상)", "성장 조건 통과"]:
        s = counts[col]
        by_year = counts.assign(y=pd.to_datetime(counts["매수일"]).dt.year).groupby("y")[col].mean()
        count_summary.append({"단계": col, "최소": int(s.min()), "중앙": float(s.median()), "최대": int(s.max()),
                              "2012 평균": by_year.get(2012), "2016 평균": by_year.get(2016), "2021 평균": by_year.get(2021)})
    rev_rows = []
    for c in CANDS:
        for name in rev[c]:
            s = rev[c][name]
            e0 = results[name]["E0"]["posttax"]
            rev_rows.append({"구간": name, "후보(매출)": c, "세후 최종": s["posttax"], "E0 대비 %": (s["posttax"] / e0 - 1) * 100,
                             "MDD %": s["mdd"], "매도 횟수": s["sells"]})
    pd.DataFrame(rev_rows).to_csv(OUT_DIR / "e1_revenue_reference.csv", index=False, encoding="utf-8-sig")
    plots = make_plots(results, rand, verdict)

    v_lines = []
    for c in CANDS:
        v = verdict[c]
        m = lambda k: "✓" if v["ok"][k] else "✗"  # noqa: E731
        full, e0 = results["2012~2021"][c], results["2012~2021"]["E0"]
        v_lines.append(
            f"- **{c}: {'합격' if v['pass'] else '불합격'}** — 1 {m(1)}(E0 대비 {(full['posttax'] / e0['posttax'] - 1) * 100:+.2f}%, 기준 +10%) · "
            f"2 {m(2)}(무작위 {v['pct']:.1f}백분위, 기준 75) · 3 {m(3)}(2012~2015 {(results['2012~2015'][c]['posttax'] / results['2012~2015']['E0']['posttax'] - 1) * 100:+.2f}%, "
            f"2016~2021 {(results['2016~2021'][c]['posttax'] / results['2016~2021']['E0']['posttax'] - 1) * 100:+.2f}%) · "
            f"4 {m(4)}(MDD {full['mdd']:.2f}% vs E0 {e0['mdd']:.2f}%)")
    passed = [c for c in CANDS if verdict[c]["pass"]]
    if passed:
        best = max(passed, key=lambda c: results["2012~2021"][c]["mdd"])
        conclusion = f"**{best} 최종 후보** (합격 {len(passed)}개 중 MDD 가장 얕음) → 봉인 구간(2022~) 1회 개봉 후보로 기록(개봉하지 않음)"
    else:
        conclusion = "**이익 성장 골라 담기는 QQQM 적립을 못 이김** — E1a·E1b·E1c 모두 불합격. 같은 기간 재판정 없음."

    md = f"""# E1 이익 성장주 적립 검증 — 결과 (E1 시도 1)

- 사전 등록: `docs/e1_plan.md`, `configs/e1_preregistration.yaml` (tag `e1-prereg` = `{tag}`), 시작 시 sha256 확인 통과
  - `docs/e1_plan.md` `{PREREG_FILES['docs/e1_plan.md']}`
  - `configs/e1_preregistration.yaml` `{PREREG_FILES['configs/e1_preregistration.yaml']}`
- 실행 코드 커밋: `{head}` · 가격·공시 봉인 2021-12-31 · 매달 30만 원, 첫 거래일 종가 체결, 판단일 = 그 전 거래일
- 점검: 모든 실행에서 현금 음수 없음, 총 납입 = 개월 수 × 30만 원

## 판정

{chr(10).join(v_lines)}

결론: {conclusion}

## 데이터 검사표

{md_table(rows)}
알려진 값 대조:

{md_table(checks)}
## 후보 (세후 최종 자산 원화, MDD·시간가중은 단위 가치 기준)

{md_table(main_rows)}
## 무작위 대조군 (2012~2021, 1,000회, 시드 20261007)

{md_table(rand_rows)}
## 종목 선택

E1a 매달 1위 종목 빈도(상위):

{md_table([{"종목": t, "1위였던 달": k} for t, k in Counter(r["1위"] for r in top1).most_common(10)])}
종목 교체 빈도:

{md_table(turn_rows)}
가장 많이 담은 종목 상위 10(후보별, 매달 상위 N 포함 횟수):

{md_table(most_rows)}
매달 1위 종목 전체 120줄: `outputs/e1_top1_monthly.csv`

## 대상 수 추이 (판단일마다, 전체 표 `outputs/e1_target_counts.csv`)

{md_table(count_summary)}
외국 기업 {len(foreign)}개, 과거 명단 {len(tickers)}종목 중 가격 못 받음 {len(missing)}개({len(missing) / len(tickers) * 100:.1f}%) — 생존 편향 한계.

## 참고 보고 — 매출 버전 (판정에 쓰지 않음)

{md_table(rev_rows)}
## 결과를 본 뒤 바꾼 점

- 결과 계산 **전**(데이터 검사 단계)에 바꾼 점 1개 — 사용자 결정(2026-10-07): NVDA 분할 연속성 검사(구현 해석 25번)를 "판단일별 TTM의 이웃 비율"에서 "마지막 판단일 주식 수 기준으로 환산한 TTM의 이웃 비율"로 바꿨다. 등록 문구대로면 분할 종목은 판단일 기준 주식 수 보정(구현 해석 8번) 때문에 정의상 항상 실패한다(8.45 → 2.11 = ÷4). 검사 대상 데이터·판정 규칙은 그대로다. 또 실제 분할일은 계획에 적힌 2021-06이 아니라 2021-07-20이다.
- 결과를 본 뒤 바꾼 점: 없음.

## 한계

- **생존 편향**: 과거 명단 {len(tickers)}종목 중 {len(missing)}개({len(missing) / len(tickers) * 100:.1f}%)는 지금 무료 출처(yfinance·SEC 현재 티커 목록)에 없어(상장폐지·인수·티커 변경) 대상에서 빠졌다. 살아남은 종목 위주라 후보·무작위 대조군 모두 실제보다 좋게 나올 수 있다(E0 QQQM은 영향 없음).
- 티커 별칭 변환을 하지 않았다(예: FI ↔ FISV). GOOG·GOOGL은 같은 회사인데 따로 순위에 들 수 있다.
- XBRL 태그 누락·형식 차이로 분기가 끊기면 그 종목은 대상에서 빠지거나(보유 중이면) 매도된다.
- 분할 보정은 yfinance 분할 이력에 의존한다. 4분기 EPS = 연간 − 9개월 누계는 주식 수 차이로 실제 4분기 EPS와 조금 다를 수 있다.
- 2016~2021은 대형 기술주가 유난히 강했던 시기다. 현금 이자 없음(모든 후보 같음).

## 그래프

{chr(10).join(f'- `{p.relative_to(ROOT).as_posix()}`' for p in plots)}
"""
    RESULT_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULT_MD.write_text(md, encoding="utf-8")
    print(f"[e1] 결과: {RESULT_MD.relative_to(ROOT)}")
    print("\n".join(v_lines))
    print("결론:", conclusion)


def print_table(rows: list[dict], title: str) -> None:
    print(f"\n## {title}")
    if not rows:
        print("(없음)")
        return
    cols = list(rows[0])
    print("| " + " | ".join(cols) + " |")
    print("| " + " | ".join("---" for _ in cols) + " |")
    for r in rows:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-data", action="store_true")
    args = ap.parse_args()
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # SEC_USER_AGENT (값은 출력하지 않는다)
    prereg = verify_preregistration()
    days, qqqm_close, qqqm_div, fx = load_qqqm()
    buys, decisions = calendar(days, "2012-01", "2021-12")
    uni = universe_by_decision(decisions)
    tickers = sorted(set().union(*uni.values()))
    facts, ciks, failed, foreign = load_edgar(tickers)
    prices, missing = load_prices(tickers)
    rows, checks = data_check(facts, prices, failed, foreign, missing, tickers, prereg)
    print_table(rows, "데이터 검사표")
    print_table(checks, "알려진 값 대조")
    if not all(c["통과"] for c in checks):
        raise SystemExit("[e1] 알려진 값 대조 실패 — 결과 계산 전에 멈춤")
    if args.check_data:
        return
    run_all(prereg, days, qqqm_close, qqqm_div, fx, buys, decisions, uni, facts, prices, failed, foreign, missing, tickers, rows, checks)


if __name__ == "__main__":
    main()
