"""SW1 실적 발표 후 흐름(PEAD) 시도 1 (사전 등록: docs/sw1_plan.md, configs/sw1_preregistration.yaml, tag sw1-prereg).

구현 해석: docs/sw1_interpretations.md (결과 계산 전 고정). 계산은 core/sw1.py(순수 함수).
신호는 데이터 조사(scripts/sw1_data_check.py)의 reaction_day·signal_on을 그대로 쓴다.

단계:
  --step fetch  시가가 없는 캐시 종목의 OHLC를 같은 출처에서 2021-12-31까지 다시 받고(종가 대조), SPY를 받는다
  --step run    사전 등록 해시 확인 → 신호 재현(636건) → 매매·비교·판정 → docs/results/sw1.md

실행: python -u -m scripts.sw1_experiments --step run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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

from core import sw1  # noqa: E402
from scripts import sw1_data_check as dc  # noqa: E402

PREREG_TAG = "sw1-prereg"
PREREG_FILES = {
    "docs/sw1_plan.md": "8b864411c51104aef01888ff2b99228b6688d4e34a89fcb13f7fbeda140ebfe9",
    "configs/sw1_preregistration.yaml": "c5ad820f50993b91f14642a9ab58aa48f4adf88731dc5f8285d95246d90c730e",
}
OHLC_YF = dc.CACHE / "ohlc"
OHLC_TI = dc.CACHE / "ohlc_tiingo"
SPY_CSV = dc.CACHE / "ohlc" / "SPY.csv"
RESULT_MD = ROOT / "docs" / "results" / "sw1.md"
OUT_DIR = ROOT / "outputs"
SEAL = dc.SEAL
CLOSE_TOL = 0.005  # 다시 받은 종가와 기존 캐시 종가의 상대 차이 허용 (해석 7)


# ── 사전 등록 확인 ───────────────────────────────────────────────────────────


def lf_sha256(b: bytes) -> str:
    """줄바꿈을 LF로 맞춘 내용의 sha256."""
    return hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest()


def verify_preregistration() -> dict:
    """작업 폴더 파일·태그 sw1-prereg에 커밋된 내용·기록된 해시 셋이 모두 같아야 통과 (LF 기준). 다르면 멈춘다."""
    for rel, expected in PREREG_FILES.items():
        work = lf_sha256((ROOT / rel).read_bytes())
        try:
            committed = lf_sha256(subprocess.run(["git", "show", f"{PREREG_TAG}:{rel}"], cwd=ROOT, check=True,
                                                 capture_output=True).stdout)
        except subprocess.CalledProcessError:
            raise SystemExit(f"[sw1] 태그 {PREREG_TAG}에서 {rel}을 읽지 못함 — 멈춤")
        if not (work == committed == expected):
            raise SystemExit(f"[sw1] 사전 등록 파일 불일치: {rel} (기록 {expected[:12]}…, 커밋 {committed[:12]}…, "
                             f"작업 폴더 {work[:12]}…) — 멈춤")
    print("[sw1] 사전 등록 해시 확인 통과 (작업 폴더 = 태그 커밋 = 기록값, LF 기준)", flush=True)
    with open(ROOT / "configs" / "sw1_preregistration.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── 가격 ─────────────────────────────────────────────────────────────────────


def universe_tickers() -> list[str]:
    """2011-12 ~ 2021-12 매달 말 나스닥100 ∪ S&P100 명단에 한 번이라도 있던 티커."""
    s: set[str] = set()
    for m in (dc.ndx_members(), dc.sp100_members()):
        for v in m.values():
            s |= v
    return sorted(s)


def ohlc_source(t: str) -> tuple[str, str] | None:
    """티커 → (종류, 파일 티커). 종류: 'bt'(backtest parquet) | 'yf'(sw1 OHLC CSV) | 'ti'(Tiingo OHLC CSV). 없으면 None."""
    src = dc.load_px(t)[1]
    if src == "없음":
        return None
    if src.startswith("yfinance"):
        x = src[len("yfinance("):-1] if "(" in src else t
        return ("bt", x) if (dc.BT_CACHE / f"{x}.parquet").exists() else ("yf", x)
    return ("ti", t)


def _tiingo_raw_path(t: str) -> Path:
    p = dc.SP500_TIINGO / f"{t}.csv"
    return p if dc.load_px(t)[1] == "tiingo" else dc.TIINGO_CACHE / f"{t}.csv"


def load_ohlc(t: str) -> pd.DataFrame | None:
    """open·close 일봉 (분할 반영, 2021-12-31까지). 봉인일 뒤 행이 있으면 멈춘다 (해석 24)."""
    s = ohlc_source(t)
    if s is None:
        return None
    kind, x = s
    if kind == "bt":
        d = pd.read_parquet(dc.BT_CACHE / f"{x}.parquet")[["open", "close"]]
    elif kind == "yf":
        p = OHLC_YF / f"{x}.csv"
        if not p.exists():
            return None
        d = pd.read_csv(p, index_col=0, parse_dates=True)[["open", "close"]]
    else:
        p = OHLC_TI / f"{x}.csv"
        if not p.exists():
            return None
        raw = pd.read_csv(p, index_col=0, parse_dates=True).sort_index()
        f = raw["split_factor"].fillna(1.0).replace(0, 1.0).astype(float)
        k = f.cumprod() / f.cumprod().iloc[-1]
        d = pd.DataFrame({"open": raw["raw_open"] * k, "close": raw["raw_close"] * k}, index=raw.index)
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    d = d.dropna(subset=["close"])
    if len(d) and d.index[-1] > SEAL:
        raise SystemExit(f"[sw1] {t} 가격에 봉인일 뒤 행 — 멈춤")
    return d if len(d) else None


def close_mismatch(new_close: pd.Series, old_close: pd.Series) -> float:
    """같은 날짜에서 다시 받은 종가와 기존 종가의 최대 상대 차이 (겹치는 날 없으면 inf)."""
    j = pd.concat([new_close.rename("n"), old_close.rename("o")], axis=1, join="inner").dropna()
    if j.empty:
        return float("inf")
    return float((j["n"] / j["o"] - 1).abs().max())


def step_fetch() -> None:
    """시가가 없는 종목(sw1 yfinance CSV·Tiingo 캐시)의 OHLC를 같은 출처에서 다시 받고, SPY를 받는다. 실패는 모아 보고."""
    import requests
    import yfinance as yf
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("TIINGO_API_KEY", "").strip()
    print(f"[fetch] TIINGO_API_KEY 있음: {bool(key)}")
    OHLC_YF.mkdir(parents=True, exist_ok=True)
    OHLC_TI.mkdir(parents=True, exist_ok=True)
    srcs = {t: ohlc_source(t) for t in universe_tickers()}
    need_yf = sorted({x for s in srcs.values() if s and s[0] == "yf" for x in [s[1]]})
    need_ti = sorted(t for t, s in srcs.items() if s and s[0] == "ti")
    print(f"[fetch] 명단 티커 {len(srcs)} · backtest {sum(1 for s in srcs.values() if s and s[0] == 'bt')}"
          f" · yfinance 다시 받기 {len(need_yf)} · Tiingo 다시 받기 {len(need_ti)} · 가격 없음 {sum(1 for s in srcs.values() if s is None)}")
    failed, mismatch = [], []

    def yf_get(t: str) -> pd.DataFrame | None:
        try:
            h = yf.Ticker(t).history(start="2011-06-01", end="2022-01-01", auto_adjust=False)
        except Exception:  # noqa: BLE001 — 실패는 모아 보고
            return None
        if h is None or h.empty:
            return None
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        return pd.DataFrame({"open": h["Open"], "close": h["Close"], "volume": h["Volume"]})[lambda d: d.index <= SEAL]

    for t in need_yf + ["SPY"]:
        p = OHLC_YF / f"{t}.csv"
        if p.exists():
            continue
        d = yf_get(t)
        if d is None:
            failed.append((t, "yfinance 빈 응답"))
            continue
        if t != "SPY":
            old = pd.read_csv(dc.PRICE_CACHE / f"{t}.csv", index_col=0, parse_dates=True)["close"]
            mm = close_mismatch(d["close"], old)
            if mm > CLOSE_TOL:
                mismatch.append((t, mm))
                continue
        d.to_csv(p)
    for t in need_ti:
        p = OHLC_TI / f"{t}.csv"
        if p.exists():
            continue
        resp = requests.get(f"https://api.tiingo.com/tiingo/daily/{t}/prices",
                            params={"startDate": "2011-06-01", "endDate": "2021-12-31", "token": key, "format": "json"},
                            timeout=30)
        time.sleep(0.5)
        if resp.status_code != 200 or not isinstance(resp.json(), list) or not resp.json():
            failed.append((t, f"Tiingo HTTP {resp.status_code}"))
            if resp.status_code == 429:
                print("[fetch] Tiingo 한도(429) — 멈춤, 다시 실행하면 이어 받는다")
                break
            continue
        d = pd.DataFrame([{"date": x["date"][:10], "raw_open": x["open"], "raw_close": x["close"], "volume": x["volume"],
                           "split_factor": x.get("splitFactor", 1.0)} for x in resp.json()]).set_index("date")
        d.index = pd.to_datetime(d.index)
        old = pd.read_csv(_tiingo_raw_path(t), index_col=0, parse_dates=True)
        old.index = pd.DatetimeIndex(old.index).tz_localize(None).normalize()
        mm = close_mismatch(d["raw_close"], old["raw_close"])
        if mm > CLOSE_TOL:
            mismatch.append((t, mm))
            continue
        d.to_csv(p)
    print(f"[fetch] 실패 {len(failed)}: {failed}")
    print(f"[fetch] 종가 불일치(> {CLOSE_TOL:.1%}) {len(mismatch)}: {[(t, round(m, 4)) for t, m in mismatch]}")


# ── 신호 재현 ───────────────────────────────────────────────────────────────


def rebuild_announcements(days: pd.DatetimeIndex, qqq_close: pd.Series) -> pd.DataFrame:
    """데이터 조사와 같은 방법으로 실적 발표 → 반응일 → 신호 (해석 3). 반응일 뒤 가격은 읽지 않는다."""
    k = dc.earnings_table()
    acc = json.loads(dc.ACCEPTED_CACHE.read_text(encoding="utf-8")) if dc.ACCEPTED_CACHE.exists() else {}
    k["accepted_et"] = k["accn"].map(lambda a: pd.Timestamp(acc[a]) if a in acc else None)
    k["d0"] = k["accepted_et"].map(lambda x: dc.reaction_day(x, days) if x is not None else None)
    pxs: dict[str, pd.DataFrame | None] = {}
    res = []
    for _, r in k.iterrows():
        t = r["ticker"]
        if t not in pxs:
            pxs[t] = dc.load_px(t)[0]
        px = pxs[t]
        a = dc.signal_on(px, qqq_close, r["d0"]) if px is not None and r["d0"] is not None else None
        res.append({"judged": a is not None, "signal": bool(a and a["signal"]),
                    "ret": a["ret"] if a else np.nan, "excess": a["excess"] if a else np.nan,
                    "vol_ratio": a["vol_ratio"] if a else np.nan})
    return pd.concat([k.reset_index(drop=True), pd.DataFrame(res)], axis=1)


# ── 계산 ─────────────────────────────────────────────────────────────────────


def load_fx(days: pd.DatetimeIndex) -> pd.Series:
    """원/달러 (D1과 같은 fx_by_date: KRW=X + DEXKOUS 보완). 거래일에 결측이 있으면 멈춘다."""
    from engine import backtest as bt
    from scripts import l1_experiments as l1x

    ctx = l1x.load_all(bt.load_config(), {"periods": {"full": {"start": "1999-03-10", "end": "2021-12-31"}}})
    fx = pd.Series({pd.Timestamp(k): v for k, v in ctx["data"].fx_by_date.items()}, dtype=float).sort_index()
    fx = fx[fx.index <= SEAL].reindex(days)
    if fx.isna().any():
        raise SystemExit(f"[sw1] 환율 결측 거래일 {int(fx.isna().sum())}개 — 멈춤")
    return fx


def price_trades(tr: pd.DataFrame, bench: pd.DataFrame, fx: pd.Series, costs: sw1.Costs, rate: float) -> pd.DataFrame:
    """매매 표 → 수익 열 추가 (비용 전 달러, 비용 뒤 원화, 세금 뒤 원화; 신호와 비교 지수 각각, 장부별 세금)."""
    out = tr.copy()
    out["ret_usd"] = out["sell_close"] / out["buy_open"] - 1
    out["bench_usd"] = [sw1.leg_returns(bench, b, s) for b, s in zip(out["buy_day"], out["sell_day"])]
    if out["bench_usd"].isna().any():
        raise SystemExit("[sw1] 비교 지수 가격 결측 — 멈춤")
    fb, fs = fx.reindex(out["buy_day"]).values, fx.reindex(out["sell_day"]).values
    out["ret_krw"] = sw1.krw_after_cost(out["ret_usd"].values, fb, fs, costs)
    out["bench_krw"] = sw1.krw_after_cost(out["bench_usd"].astype(float).values, fb, fs, costs)
    yr = pd.DatetimeIndex(out["sell_day"]).year.values
    out["ret_at"] = out["ret_krw"] - sw1.ledger_tax(out["ret_krw"].values, yr, rate)
    out["bench_at"] = out["bench_krw"] - sw1.ledger_tax(out["bench_krw"].values, yr, rate)
    out["ex_pre"] = out["ret_usd"] - out["bench_usd"]
    out["ex_cost"] = out["ret_krw"] - out["bench_krw"]
    out["ex_after"] = out["ret_at"] - out["bench_at"]
    out["year"] = pd.DatetimeIndex(out["d0"]).year
    out["quarter"] = pd.PeriodIndex(pd.DatetimeIndex(out["d0"]), freq="Q").astype(str)
    return out


def summary(t: pd.DataFrame) -> dict:
    """매매 수, 평균 초과 수익(비용 전·비용 뒤·세금 뒤), 승률, 구간·연도 요약 (% 단위)."""
    by_year = t.groupby("year")["ex_after"].mean()
    return {"n": len(t), "ex_pre": t["ex_pre"].mean() * 100, "ex_cost": t["ex_cost"].mean() * 100,
            "ex_after": t["ex_after"].mean() * 100, "win": (t["ex_after"] > 0).mean() * 100,
            "p1": t.loc[t["year"] <= 2016, "ex_after"].mean() * 100, "p2": t.loc[t["year"] >= 2017, "ex_after"].mean() * 100,
            "pos_years": int((by_year > 0).sum()), "n_years": len(by_year),
            "ret_usd": t["ret_usd"].mean() * 100, "bench_usd": t["bench_usd"].mean() * 100}


def candidate_table(tr: pd.DataFrame, pools: list[list[str]], prices: dict, fx: pd.Series, bench: pd.DataFrame,
                    costs: sw1.Costs) -> list[dict]:
    """매매마다 무작위 후보들의 (원화 비용 뒤 수익, 매도 연도, 비교 지수 원화 비용 뒤 수익, 매도 연도) 배열.

    후보 = pools[j]의 종목 중 매수일 시가와 매도 가격이 있는 것 (해석 14). 실제 매도일은 sw1.realized_exit와 같은 규칙.
    """
    memo: dict[tuple, tuple | None] = {}
    out = []
    for j, (b, pl) in enumerate(zip(tr["buy_day"], tr["plan_sell_day"])):
        rows = []
        for c in pools[j]:
            key = (c, b, pl)
            if key not in memo:
                px = prices.get(c)
                v = None
                if px is not None and b in px.index and np.isfinite(px.at[b, "open"]) and px.at[b, "open"] > 0:
                    ex = sw1.realized_exit(px["close"], b, pl)
                    if ex is not None:
                        bu = sw1.leg_returns(bench, b, ex[0])
                        r = ex[1] / float(px.at[b, "open"]) - 1
                        fb, fs = float(fx[b]), float(fx[ex[0]])
                        v = (float(sw1.krw_after_cost(r, fb, fs, costs)), ex[0].year,
                             float(sw1.krw_after_cost(bu, fb, fs, costs)))
                memo[key] = v
            if memo[key] is not None:
                rows.append(memo[key])
        out.append({"ret": np.array([x[0] for x in rows]), "year": np.array([x[1] for x in rows]),
                    "bench": np.array([x[2] for x in rows])})
    return out


def random_means(cands: list[dict], iters: int, seed: int, rate: float, first_year: int, n_years: int) -> np.ndarray:
    """무작위 회차마다 매매 하나당 후보 1개 → 장부별 세금 → 평균 초과 수익(세금 뒤). 후보 없는 매매는 빠진다."""
    rng = np.random.default_rng(seed)
    use = [c for c in cands if len(c["ret"])]
    n = len(use)
    R = np.empty((iters, n))
    B = np.empty((iters, n))
    Y = np.empty((iters, n), dtype=int)
    for j, c in enumerate(use):
        idx = rng.integers(0, len(c["ret"]), size=iters)
        R[:, j], B[:, j], Y[:, j] = c["ret"][idx], c["bench"][idx], c["year"][idx] - first_year
    ra = R - sw1.ledger_tax_many(R, Y, n_years, rate)
    ba = B - sw1.ledger_tax_many(B, Y, n_years, rate)
    return (ra - ba).mean(axis=1)


def pct(x: float, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


def step_run() -> None:
    """사전 등록 확인 → 신호 재현 → 매매·비교·판정 → docs/results/sw1.md, outputs/sw1_trades.csv."""
    pre = verify_preregistration()
    tr_cfg, sig_cfg = pre["trade"], pre["signal"]["conditions"]
    if (sig_cfg["stock_return_min"], sig_cfg["excess_vs_qqq_min"], sig_cfg["volume_ratio_min"], sig_cfg["volume_window_days"]) != (
            dc.RET_MIN, dc.EXCESS_MIN, dc.VOL_MULT, dc.VOL_WINDOW):
        raise SystemExit("[sw1] 신호 조건이 사전 등록과 다름 — 멈춤")
    hold = int(tr_cfg["hold_trading_days"])
    costs = sw1.Costs(**tr_cfg["costs"])
    rate = tr_cfg["tax"]["rate_pct"] / 100.0
    rnd = pre["benchmarks"]["random"]

    qqq = pd.read_parquet(dc.BT_CACHE / "QQQ.parquet")[["open", "close"]]
    if qqq.index[-1] > SEAL:
        raise SystemExit("[sw1] QQQ 봉인일 뒤 행 — 멈춤")
    days = qqq.index
    spy = pd.read_csv(SPY_CSV, index_col=0, parse_dates=True)[["open", "close"]]
    fx = load_fx(days)

    # 1) 신호 재현 (예상 636건, 데이터 조사 표와 같아야 함)
    k = rebuild_announcements(days, qqq["close"])
    ref = pd.read_csv(dc.CACHE / "announcements.csv")
    sig = k[k["signal"]].copy()
    if len(sig) != pre["signal"]["expected_signals"]["union"] or set(sig["accn"]) != set(ref.loc[ref["signal"], "accn"]):
        raise SystemExit(f"[sw1] 신호 재현 불일치: {len(sig)}건 — 멈춤")
    sig["d0"] = pd.to_datetime(sig["d0"])
    print(f"[sw1] 신호 재현 {len(sig)}건 (나스닥100 {int(sig['in_ndx'].sum())}, S&P100 {int(sig['in_sp100'].sum())})", flush=True)

    universe = universe_tickers()
    prices = {t: load_ohlc(t) for t in universe}
    prices = {t: v for t, v in prices.items() if v is not None}

    # 2) 주 판정 매매
    tr0, sk0 = sw1.make_trades(sig[["ticker", "d0", "accn", "in_ndx", "in_sp100", "filing_date"]], prices, days, hold)
    T = price_trades(tr0, qqq, fx, costs, rate)
    S = summary(T)

    # 3) 무작위 1,000회 (같은 매수일, 매수일 전 달 말 합집합 명단, 신호 종목 자신 제외)
    ndx, sp = dc.ndx_members(), dc.sp100_members()

    def members(day: pd.Timestamp) -> set[str]:
        me = (day - pd.offsets.MonthEnd(1)).normalize()
        return ndx.get(me, set()) | sp.get(me, set())

    pools = [sorted(members(b) - {t}) for b, t in zip(T["buy_day"], T["ticker"])]
    cands = candidate_table(T, pools, prices, fx, qqq, costs)
    first_year, n_years = 2012, 11
    rmeans = random_means(cands, int(rnd["iterations"]), int(rnd["seed"]), rate, first_year, n_years)
    rpct = sw1.percentile_of(T["ex_after"].mean(), rmeans)
    V = sw1.verdict(T["ex_after"], T["year"], rpct)
    print(f"[sw1] 주 판정 매매 {S['n']} · 평균 초과(세금 뒤) {S['ex_after']:+.2f}% · 무작위 {rpct:.1f}백분위", flush=True)

    # 4) 참고 대조군: 같은 ISO 주에 반응일이 있고 판정됐지만 신호가 아닌 발표
    ns = k[k["judged"] & ~k["signal"] & k["d0"].notna()].copy()
    ns["wk"] = pd.to_datetime(ns["d0"]).dt.strftime("%G-%V")
    wk_pool = ns.groupby("wk")["ticker"].apply(lambda s: sorted(set(s))).to_dict()
    pools_c = [[c for c in wk_pool.get(pd.Timestamp(d).strftime("%G-%V"), []) if c != t] for d, t in zip(T["d0"], T["ticker"])]
    cands_c = candidate_table(T, pools_c, prices, fx, qqq, costs)
    cmeans = random_means(cands_c, int(rnd["iterations"]), int(rnd["seed"]), rate, first_year, n_years)
    c_n = sum(1 for c in cands_c if len(c["ret"]))

    # 5) 참고 보고
    refs = {}
    for h in (20, 40):
        a, _ = sw1.make_trades(sig[["ticker", "d0", "accn", "in_ndx", "in_sp100", "filing_date"]], prices, days, h)
        refs[f"{h}거래일 보유"] = summary(price_trades(a, qqq, fx, costs, rate))
    a, _ = sw1.make_trades(sig[["ticker", "d0", "accn", "in_ndx", "in_sp100", "filing_date"]], prices, days, hold, stop_pct=8.0)
    Tst = price_trades(a, qqq, fx, costs, rate)
    refs["−8% 손절 (60일)"] = summary(Tst)
    eps = pd.read_csv(dc.CACHE / "signals_eps_ref.csv")
    eps = eps[eps["eps_up"].astype(bool)].copy()
    eps["d0"] = pd.to_datetime(eps["d0"])
    a, _ = sw1.make_trades(eps[["ticker", "d0", "accn", "in_ndx", "in_sp100", "filing_date", "q_filing_date"]], prices, days, hold,
                           buy_after_col="q_filing_date")
    Teps = price_trades(a, qqq, fx, costs, rate)
    refs[f"EPS 확인판 ({len(eps)}건 중)"] = summary(Teps)
    refs["비교 지수 SPY"] = summary(price_trades(tr0, spy, fx, costs, rate))
    refs["나스닥100만"] = summary(T[T["in_ndx"]])
    refs["S&P 100만"] = summary(T[T["in_sp100"]])
    q_mean = T.groupby("quarter")["ex_after"].mean()

    # 6) 공백 비율
    union = k
    gap = {"ann": len(union), "not_judged": int((~union["judged"]).sum()), "signals": len(sig),
           "skip": sk0["skip"].value_counts().to_dict(), "exit": T["exit_reason"].value_counts().to_dict()}

    OUT_DIR.mkdir(exist_ok=True)
    T.to_csv(OUT_DIR / "sw1_trades.csv", index=False, encoding="utf-8-sig")
    write_report(pre, T, S, V, rmeans, cmeans, c_n, refs, q_mean, gap)
    print(f"[sw1] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


def write_report(pre, T, S, V, rmeans, cmeans, c_n, refs, q_mean, gap) -> None:
    """결과 문서 작성 (맨 위에 해시 기준)."""
    ok = lambda b: "충족" if b else "미충족"  # noqa: E731
    verdict_txt = "판정 불가 (매매 300건 미만)" if V["pass"] is None else ("**합격**" if V["pass"] else "**불합격**")
    L = []
    L += ["# SW1 결과 (시도 1) — 실적 발표 후 흐름(PEAD)", "",
          "> 해시 기준 (LF 정규화 sha256, 태그 `sw1-prereg` 커밋 내용 = 작업 폴더 = 스크립트 기록값 확인 후 실행)",
          *[f"> - `{k}` `{v}`" for k, v in PREREG_FILES.items()],
          "> 구현 해석: `docs/sw1_interpretations.md` (결과 계산 전 커밋 aceb621)", "",
          f"- 기간: 반응일 2012-01 ~ 2021-12, 가격 봉인 2021-12-31 · 매수 반응일 다음 거래일 시가, 매도 60거래일 뒤 종가",
          f"- 비용: 매수·매도 수수료 0.07%, 환전 0.1%씩 · 세금: 매도 연도별 순이익 22% (공제 미적용, 이익 비례 배분 — H2)",
          f"- 무작위: 1,000회, 시드 {pre['benchmarks']['random']['seed']}", "",
          f"## 판정: {verdict_txt}", "",
          "| 번호 | 기준 | 값 | 결과 |", "| --- | --- | --- | --- |",
          f"| 1 | 매매당 평균 초과 수익(비용·세금 뒤) > 0 | {pct(V['mean'] * 100)} | {ok(V['c1'])} |",
          f"| 2 | 무작위 1,000회의 75백분위 이상 | {V['random_pct']:.1f}백분위 | {ok(V['c2'])} |",
          f"| 3 | 2012~2016, 2017~2021 모두 > 0 | {pct(V['period_means'][0] * 100)} / {pct(V['period_means'][1] * 100)} | {ok(V['c3'])} |",
          f"| 4 | 10년 중 6년 이상 > 0 | {V['pos_years']}/{V['n_years']}년 | {ok(V['c4'])} |",
          f"| 5 | 매매 300건 이상 | {V['n']}건 | {ok(V['c5'])} |", "",
          "## 핵심 숫자", "",
          "| 항목 | 값 |", "| --- | --- |",
          f"| 매매 수 | {S['n']} |",
          f"| 평균 수익: 신호 / QQQ (달러, 비용 전) | {pct(S['ret_usd'])} / {pct(S['bench_usd'])} |",
          f"| 평균 초과 수익: 비용 전 | {pct(S['ex_pre'])} |",
          f"| 평균 초과 수익: 비용 뒤 (원화, 세금 전) | {pct(S['ex_cost'])} |",
          f"| 평균 초과 수익: 비용·세금 뒤 | {pct(S['ex_after'])} |",
          f"| 승률 (세금 뒤 QQQ보다 나은 매매) | {S['win']:.1f}% |",
          f"| 무작위 평균 초과(세금 뒤) 중앙값 · 75백분위 | {pct(np.median(rmeans) * 100)} · {pct(np.percentile(rmeans, 75) * 100)} |",
          f"| 실제 값의 무작위 백분위 | {V['random_pct']:.1f} |",
          f"| (참고) 분기별 평균의 평균 (기준 1 재확인, {len(q_mean)}분기) | {pct(q_mean.mean() * 100)} |",
          f"| (참고) 같은 주 발표·신호 없음 대조군 중앙값 ({c_n}건) · 실제 백분위 | {pct(np.median(cmeans) * 100)} · {sw1.percentile_of(T['ex_after'].mean(), cmeans):.1f} |",
          "", "## 연도별 (반응일 연도)", "",
          "| 연도 | 매매 | 초과 비용 전 | 초과 세금 뒤 | 승률 |", "| --- | --- | --- | --- | --- |"]
    for y, g in T.groupby("year"):
        L.append(f"| {y} | {len(g)} | {pct(g['ex_pre'].mean() * 100)} | {pct(g['ex_after'].mean() * 100)} | {(g['ex_after'] > 0).mean() * 100:.0f}% |")
    L += ["", "## 구간별", "", "| 구간 | 매매 | 초과 비용 전 | 초과 세금 뒤 | 승률 |", "| --- | --- | --- | --- | --- |"]
    for nm, g in (("2012~2016", T[T["year"] <= 2016]), ("2017~2021", T[T["year"] >= 2017])):
        L.append(f"| {nm} | {len(g)} | {pct(g['ex_pre'].mean() * 100)} | {pct(g['ex_after'].mean() * 100)} | {(g['ex_after'] > 0).mean() * 100:.0f}% |")
    L += ["", "## 참고 보고 (판정 제외)", "",
          "| 보고 | 매매 | 초과 비용 전 | 초과 세금 뒤 | 승률 | 2012~16 / 2017~21 | 양수 연도 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for nm, r in refs.items():
        L.append(f"| {nm} | {r['n']} | {pct(r['ex_pre'])} | {pct(r['ex_after'])} | {r['win']:.1f}% | {pct(r['p1'])} / {pct(r['p2'])} | {r['pos_years']}/{r['n_years']} |")
    cov = pre["universe"]["coverage_ticker_months"]
    sk = gap["skip"]
    L += ["", "- 세금은 보고마다 그 매매 집합의 장부로 다시 계산 (나스닥100만·S&P 100만은 주 판정 장부의 매매별 값을 나눠 본 것)",
          "- EPS 확인판 매수 = max(10-Q 제출일 다음 거래일, 반응일 다음 거래일) 시가 (해석 16)", "",
          "## 공백", "",
          "| 항목 | 값 |", "| --- | --- |",
          f"| 종목·달 범위 (CIK·가격 모두) | 나스닥100 {cov['nasdaq100_pct']}%, S&P 100 {cov['sp100_pct']}% (계획서 2장) |",
          f"| 실적 발표 중 판정 못 함 (가격·시각·이력 부족) | {gap['not_judged']}/{gap['ann']} ({gap['not_judged'] / gap['ann']:.1%}) |",
          f"| 신호 중 매매 안 됨 | {sum(sk.values())}/{gap['signals']} — " + ", ".join(f"{k} {v}" for k, v in sk.items()) + " |",
          "| 매도 사유 | " + ", ".join(f"{k} {v}" for k, v in gap["exit"].items()) + " |", "",
          "## 결과를 본 뒤 바꾼 점", "", "없음", ""]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["fetch", "run"], required=True)
    a = ap.parse_args()
    {"fetch": step_fetch, "run": step_run}[a.step]()


if __name__ == "__main__":
    main()
