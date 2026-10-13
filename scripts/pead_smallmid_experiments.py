"""실적급등 다음날 매수·3달 보유 — 중소형 재현 (내부 ID pead-smallmid, 태그 pead-smallmid-prereg).

사전 등록: docs/pead_smallmid_plan.md, configs/pead_smallmid_preregistration.yaml.
구현 해석: docs/pead_smallmid_interpretations.md (결과 계산 전 고정).
신호·매매 계산은 SW1과 같은 코드: core/sw1.py(make_trades·krw_after_cost·ledger_tax 등),
scripts/sw1_data_check.py(reaction_day·signal_on), scripts/smallmid_prices_check.py(편입·대형 겹침·유동성).

실행: python -u -m scripts.pead_smallmid_experiments
  1) 사전 등록 파일 해시(작업 폴더 = 태그 커밋 = 기록값) → 2) "10월 한도 시점" 캐시 해시 → 3) 신호 재현(사전 등록 건수와 같아야 함)
  → 4) 매매·비교·무작위·판정 → docs/results/pead_smallmid.md, outputs/pead_smallmid_trades.csv
"""

from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import sw1  # noqa: E402
from scripts import sw1_data_check as sw  # noqa: E402
from scripts import sw1_experiments as sx  # noqa: E402
import smallmid_prices_check as pc  # noqa: E402
import smallmid_universe_check as su  # noqa: E402

PREREG_TAG = "pead-smallmid-prereg"
PREREG_FILES = {
    "docs/pead_smallmid_plan.md": "ae3547cb7d8e9143d2aa021948c4caac1cd9f809d4f75079831c760f07e7a64f",
    "configs/pead_smallmid_preregistration.yaml": "257928601a0e08a1a31801fbbdd11e4309db8c3316eefa96d7055c8a3f6ef9c5",
}
PRE_YAML = ROOT / "configs" / "pead_smallmid_preregistration.yaml"
RESULT_MD = ROOT / "docs" / "results" / "pead_smallmid.md"
OUT_DIR = ROOT / "outputs"
SEAL = pd.Timestamp("2021-12-31")
ZONE_FIRST, ZONE_LAST = pd.Timestamp("2017-01-01"), pd.Timestamp("2021-12-31")
IJR_CSV = ROOT / "data" / "cache" / "pead_smallmid" / "IJR.csv"
MDY_CSV = ROOT / "data" / "cache" / "pead_smallmid" / "MDY.csv"
UNDECIDED = ("가격 공백·다른 회사", "가격 부족(판정 불가)")


class StopRun(SystemExit):
    """사전 등록·캐시·봉인·구역 검사 실패 — 멈춘다."""


# ── 확인 ─────────────────────────────────────────────────────────────────────


def verify_preregistration() -> dict:
    """작업 폴더 = 태그 커밋 = 기록값 (LF 정규화 sha256). 다르면 멈춘다 (SW1 해석 24와 같은 방식)."""
    for rel, expected in PREREG_FILES.items():
        work = sx.lf_sha256((ROOT / rel).read_bytes())
        try:
            committed = sx.lf_sha256(subprocess.run(["git", "show", f"{PREREG_TAG}:{rel}"], cwd=ROOT, check=True,
                                                    capture_output=True).stdout)
        except subprocess.CalledProcessError:
            raise StopRun(f"[pead] 태그 {PREREG_TAG}에서 {rel}을 읽지 못함 — 멈춤")
        if not (work == committed == expected):
            raise StopRun(f"[pead] 사전 등록 파일 불일치: {rel} (기록 {expected[:12]}…, 커밋 {committed[:12]}…, "
                          f"작업 폴더 {work[:12]}…) — 멈춤")
    print("[pead] 사전 등록 해시 확인 통과 (작업 폴더 = 태그 커밋 = 기록값, LF 기준)", flush=True)
    return yaml.safe_load(PRE_YAML.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    """파일 바이트 그대로의 sha256 (캐시 파일은 줄바꿈 정규화 안 함)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_cache(freeze: dict) -> list[str]:
    """사전 등록 yaml의 data_freeze 목록과 캐시 파일 sha256 비교. 하나라도 다르거나 없으면 멈춘다.

    입력: {'files': [{path, sha256}], 'prices': [{path, sha256}]}. 출력: 고정된 가격 파일 경로 목록.
    목록에 없는 가격 파일(11월 뒤 Tiingo로 새로 받은 것)은 읽지 않는다.
    """
    bad = []
    for rec in freeze["files"] + freeze["prices"]:
        p = ROOT / rec["path"]
        if not p.exists():
            bad.append(f"{rec['path']} 없음")
        elif file_sha256(p) != rec["sha256"]:
            bad.append(f"{rec['path']} 해시 다름")
    if bad:
        raise StopRun(f"[pead] 캐시 고정 불일치 {len(bad)}개 (앞 5개: {bad[:5]}) — 멈춤")
    print(f"[pead] 캐시 해시 확인 통과: 표 {len(freeze['files'])}개 · 가격 {len(freeze['prices'])}개", flush=True)
    return [r["path"] for r in freeze["prices"]]


def seal_check(df: pd.DataFrame, what: str) -> None:
    """2021-12-31 뒤 행이 있으면 잘라내지 않고 멈춘다."""
    if len(df) and pd.Timestamp(df.index.max()) > SEAL:
        raise StopRun(f"[pead] {what}: 봉인일 뒤 행 — 멈춤")


# ── 순수 함수 ────────────────────────────────────────────────────────────────


def liquid_flags(px: pd.DataFrame, min_price: float = pc.LIQ_PRICE, min_dollar_vol: float = pc.LIQ_DOLLAR_VOL,
                 window: int = 20) -> pd.Series:
    """날짜 d마다 pc.liquid_before(px, d)와 같은 판정 (순수, d 당일 값 미사용).

    입력: px(close·volume, 날짜 오름차순). 출력: bool Series (px 날짜). 전날 종가 ≥ 5달러,
    전날까지 20거래일 평균 거래대금 ≥ 500만 달러. 앞 자료가 20거래일 미만이면 False.
    """
    dv = (px["close"] * px["volume"]).rolling(window, min_periods=window).mean().shift(1)
    prev = px["close"].shift(1)
    return ((prev >= min_price) & (dv >= min_dollar_vol)).fillna(False).astype(bool)


def in_zone(ivs: list, big: tuple | None, d: pd.Timestamp) -> tuple[bool, str | None]:
    """그날 검증 ① 대상인지 (순수): 편입 중(보고 1번 구간 제외) & 대형 겹침(앞뒤 한 분기) 아님. 출력 (여부, 지수)."""
    m, idx = pc.member_on(ivs, d)
    if not m or pc.big_overlap(big, d):
        return False, None
    return True, idx


def classify_events(events: pd.DataFrame, days: pd.DatetimeIndex, ivs: dict, big_ivs: dict, usable: set[int],
                    prices: dict[int, pd.DataFrame], qqq_close: pd.Series, liq: dict[int, pd.Series]) -> pd.DataFrame:
    """실적 발표 → 반응일 → 상태(판정·공백 등)·신호·유동성 (smallmid_prices_check.step_signals와 같은 순서·규칙).

    반응일 2017-01-01 ~ 2021-12-31만 남긴다. 신호 판정은 반응일까지의 가격만 쓴다(sw.signal_on).
    출력 열: cik, accn, d0, year, index, status, signal, ret, excess, vol_ratio, liquid.
    """
    rows = []
    for r in events.to_dict("records"):
        cik = int(r["cik"])
        if not isinstance(r.get("accepted_et"), str):
            continue  # 발표 시각 없음 (19장: 0건)
        d0 = sw.reaction_day(pd.Timestamp(r["accepted_et"]), days)
        if d0 is None or d0 < ZONE_FIRST or d0 > ZONE_LAST:
            continue
        row = {"cik": cik, "accn": r["accn"], "d0": d0, "year": d0.year, "index": None, "signal": False, "liquid": False}
        m, idx = pc.member_on(ivs.get(cik, []), d0)
        if not m:
            rows.append({**row, "status": "그날 편입 아님"})
            continue
        if pc.big_overlap(big_ivs.get(cik), d0):
            rows.append({**row, "status": "대형 겹침(앞뒤 한 분기)"})
            continue
        row["index"] = idx
        if cik not in usable:
            rows.append({**row, "status": "가격 공백·다른 회사"})
            continue
        px = prices[cik]
        sig = sw.signal_on(px[["close", "volume"]], qqq_close, d0)
        if sig is None:
            rows.append({**row, "status": "가격 부족(판정 불가)"})
            continue
        lq = bool(liq[cik].get(d0, False))
        rows.append({**row, "status": "판정", "signal": bool(sig["signal"]), "ret": sig["ret"], "excess": sig["excess"],
                     "vol_ratio": sig["vol_ratio"], "liquid": lq})
    return pd.DataFrame(rows)


def undecided_stats(ev: pd.DataFrame) -> dict:
    """판정 불가 비율 (해석: 분모 = 판정 + 가격 공백·다른 회사 + 가격 부족, 2017~2021 편입 중·대형 겹침 아님)."""
    judged = int((ev["status"] == "판정").sum())
    und = int(ev["status"].isin(UNDECIDED).sum())
    sig = int((ev["signal"] & (ev["status"] == "판정")).sum())
    sig_liq = int((ev["signal"] & ev["liquid"] & (ev["status"] == "판정")).sum())
    return {"judged": judged, "undecided": und, "den": judged + und, "rate": und / (judged + und) if judged + und else float("nan"),
            "signals": sig, "signals_liquid": sig_liq}


def missed_trades_mean(ex_after: pd.Series, undecided: int, signal_rate: float, low_q: float = 0.10) -> dict:
    """판정 ⑦ (순수): 놓친 매매 k = ceil(판정 불가 발표 × 관찰 신호율)을 관찰 매매 하위 10% 평균 초과로 채운 평균.

    하위 10% = 초과 수익(세금 뒤) 낮은 순 ceil(n × 0.1)건.
    """
    x = np.sort(np.asarray(ex_after, dtype=float))
    n = len(x)
    k = int(math.ceil(undecided * signal_rate))
    low = x[: max(1, int(math.ceil(n * low_q)))]
    fill = float(low.mean())
    return {"k": k, "fill": fill, "mean": float((x.sum() + k * fill) / (n + k)), "n_low": len(low)}


def mean_without_top(ex_after: pd.Series, top: int = 5) -> float:
    """판정 ⑥ (순수): 초과 수익(세금 뒤) 상위 top건을 뺀 평균."""
    x = np.sort(np.asarray(ex_after, dtype=float))
    return float(x[:-top].mean()) if len(x) > top else float("nan")


def verdict(ex_after: pd.Series, year: pd.Series, random_pct: float, undecided_rate: float, missed_mean: float,
            crit: dict) -> dict:
    """판정 ①~⑧ (계획서 6장, 해석 V1~V3).

    PASS = 모두 충족. ① 미달 → FAIL (⑧과 겹쳐도 FAIL, 사용자 결정). ① 충족·⑧ 미달 → INCONCLUSIVE.
    ① 충족인데 ②~⑦ 중 미달 → INCONCLUSIVE.
    """
    m = float(ex_after.mean()) if len(ex_after) else float("nan")
    per = [float(ex_after[(year >= a) & (year <= b)].mean()) for a, b in crit["periods"]]
    by_year = ex_after.groupby(year).mean()
    pos_years = int((by_year > 0).sum())
    top_out = mean_without_top(ex_after, crit["drop_top"])
    c = {
        "c1": bool(m > 0),
        "c2": bool(random_pct >= crit["random_pct_min"]),
        "c3": bool(all(np.isfinite(p) and p > 0 for p in per)),
        "c4": bool(pos_years >= crit["min_pos_years"]),
        "c5": bool(len(ex_after) >= crit["min_trades"]),
        "c6": bool(np.isfinite(top_out) and top_out > 0),
        "c7": bool(missed_mean > 0),
        "c8": bool(undecided_rate <= crit["max_undecided"]),
    }
    if not c["c1"]:
        res = "FAIL"
    elif not c["c8"]:
        res = "INCONCLUSIVE"
    elif all(c[f"c{i}"] for i in range(2, 8)):
        res = "PASS"
    else:
        res = "INCONCLUSIVE"
    return {**c, "result": res, "mean": m, "period_means": per, "pos_years": pos_years, "n_years": len(by_year),
            "n": int(len(ex_after)), "random_pct": random_pct, "top_out": top_out, "missed_mean": missed_mean,
            "undecided_rate": undecided_rate, "by_year": by_year}


def price_trades(tr: pd.DataFrame, bench: pd.DataFrame, fx: pd.Series, c_stock: sw1.Costs, c_bench: sw1.Costs,
                 rate: float) -> pd.DataFrame:
    """매매 표 → 수익 열 (비용 전 달러, 비용 뒤 원화, 세금 뒤 원화). 종목은 슬리피지 포함, 비교 지수는 슬리피지 없음 (해석 C1)."""
    out = tr.copy()
    out["ret_usd"] = out["sell_close"] / out["buy_open"] - 1
    out["bench_usd"] = [sw1.leg_returns(bench, b, s) for b, s in zip(out["buy_day"], out["sell_day"])]
    if out["bench_usd"].isna().any():
        raise StopRun("[pead] 비교 지수 가격 결측 — 멈춤")
    fb, fs = fx.reindex(out["buy_day"]).values, fx.reindex(out["sell_day"]).values
    out["ret_krw"] = sw1.krw_after_cost(out["ret_usd"].values, fb, fs, c_stock)
    out["bench_krw"] = sw1.krw_after_cost(out["bench_usd"].astype(float).values, fb, fs, c_bench)
    yr = pd.DatetimeIndex(out["sell_day"]).year.values
    out["ret_at"] = out["ret_krw"] - sw1.ledger_tax(out["ret_krw"].values, yr, rate)
    out["bench_at"] = out["bench_krw"] - sw1.ledger_tax(out["bench_krw"].values, yr, rate)
    out["ex_pre"] = out["ret_usd"] - out["bench_usd"]
    out["ex_cost"] = out["ret_krw"] - out["bench_krw"]
    out["ex_after"] = out["ret_at"] - out["bench_at"]
    out["year"] = pd.DatetimeIndex(out["d0"]).year
    return out


def summary(t: pd.DataFrame) -> dict:
    """매매 수, 평균 초과(비용 전·뒤·세금 뒤), 승률, 구간 평균 (% 단위)."""
    if t.empty:
        return {"n": 0}
    by_year = t.groupby("year")["ex_after"].mean()
    return {"n": len(t), "ex_pre": t["ex_pre"].mean() * 100, "ex_cost": t["ex_cost"].mean() * 100,
            "ex_after": t["ex_after"].mean() * 100, "win": (t["ex_after"] > 0).mean() * 100,
            "p1": t.loc[t["year"] <= 2019, "ex_after"].mean() * 100, "p2": t.loc[t["year"] >= 2020, "ex_after"].mean() * 100,
            "pos_years": int((by_year > 0).sum()), "n_years": len(by_year),
            "ret_usd": t["ret_usd"].mean() * 100, "bench_usd": t["bench_usd"].mean() * 100}


def candidate_table(tr: pd.DataFrame, pools: list[list[int]], prices: dict, fx: pd.Series, bench: pd.DataFrame,
                    c_stock: sw1.Costs, c_bench: sw1.Costs) -> list[dict]:
    """매매마다 무작위 후보들의 (원화 비용 뒤 수익, 매도 연도, QQQ 원화 비용 뒤 수익) 배열 (sw1_experiments와 같은 규칙).

    후보 = pools[j] 중 매수일 시가와 매도 가격이 있는 것. 실제 매도일은 sw1.realized_exit 규칙.
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
                        v = (float(sw1.krw_after_cost(r, fb, fs, c_stock)), ex[0].year,
                             float(sw1.krw_after_cost(bu, fb, fs, c_bench)))
                memo[key] = v
            if memo[key] is not None:
                rows.append(memo[key])
        out.append({"ret": np.array([x[0] for x in rows]), "year": np.array([x[1] for x in rows], dtype=int),
                    "bench": np.array([x[2] for x in rows])})
    return out


def zone_guard(trades: pd.DataFrame, ivs: dict, big_ivs: dict) -> None:
    """종목 × 날짜 가드 (검증 구간 정책 2장 7): 매매마다 반응일이 2017~2021이고 그날 검증 ① 대상, 매도일 ≤ 봉인일. 아니면 멈춘다."""
    for r in trades.itertuples():
        d0 = pd.Timestamp(r.d0)
        if not (ZONE_FIRST <= d0 <= ZONE_LAST) or pd.Timestamp(r.sell_day) > SEAL or \
                not in_zone(ivs.get(int(r.ticker), []), big_ivs.get(int(r.ticker)), d0)[0]:
            raise StopRun(f"[pead] 구역 밖 매매: CIK {r.ticker} 반응일 {d0.date()} — 멈춤")


# ── 실행 ─────────────────────────────────────────────────────────────────────


def pct(x: float, d: int = 2) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}%"


def run() -> None:
    """사전 등록·캐시 확인 → 신호 재현 → 매매·비교·무작위·판정 → 결과 문서."""
    pre = verify_preregistration()
    price_paths = verify_cache(pre["data_freeze"])
    sc, tc, cr = pre["signal"]["conditions"], pre["trade"], pre["criteria_values"]
    if (sc["stock_return_min"], sc["excess_vs_qqq_min"], sc["volume_ratio_min"], sc["volume_window_days"]) != (
            sw.RET_MIN, sw.EXCESS_MIN, sw.VOL_MULT, sw.VOL_WINDOW):
        raise StopRun("[pead] 신호 조건이 사전 등록과 다름 — 멈춤")
    lq = pre["signal"]["liquidity"]
    if (lq["min_prev_close_usd"], lq["min_avg_dollar_volume_usd"]) != (pc.LIQ_PRICE, pc.LIQ_DOLLAR_VOL):
        raise StopRun("[pead] 유동성 조건이 사전 등록과 다름 — 멈춤")
    hold = int(tc["hold_trading_days"])
    co = pre["costs"]
    c_stock = sw1.Costs(co["commission_pct"], co["commission_pct"], co["fx_spread_pct"], co["slippage_pct"])
    c_bench = sw1.Costs(co["commission_pct"], co["commission_pct"], co["fx_spread_pct"], 0.0)
    rate = co["tax_rate_pct"] / 100.0
    rnd = pre["benchmarks"]["random"]

    qqq = pd.read_parquet(sw.BT_CACHE / "QQQ.parquet")[["open", "close"]]
    seal_check(qqq, "QQQ")
    days = pd.DatetimeIndex(qqq.index)
    ijr = pd.read_csv(IJR_CSV, index_col=0, parse_dates=True)[["open", "close"]]
    mdy = pd.read_csv(MDY_CSV, index_col=0, parse_dates=True)[["open", "close"]]
    seal_check(ijr, "IJR")
    seal_check(mdy, "MDY")
    fx = sx.load_fx(days)
    fx_hash = hashlib.sha256(fx.loc["2016-12-01":].round(6).to_csv().encode()).hexdigest()
    if fx_hash != pre["data_freeze"]["fx_series_sha256"]:
        raise StopRun(f"[pead] 환율 시리즈 해시 다름 ({fx_hash[:12]}…) — 멈춤")

    ivs = pc.intervals_by_cik()
    big = pd.read_csv(su.SW1_CIK, parse_dates=["first", "last"]).dropna(subset=["cik"])
    big_ivs = {int(c): (a, b) for c, a, b in zip(big["cik"], big["first"], big["last"])}
    chk = pd.read_csv(pc.CHECK_CSV)
    usable = set(chk.loc[chk["same_company"].astype("boolean").fillna(False), "cik"].astype(int))
    prices: dict[int, pd.DataFrame] = {}
    for rel in price_paths:
        cik = int(Path(rel).stem)
        if cik not in usable:
            continue
        px = pd.read_csv(ROOT / rel, index_col=0, parse_dates=True).sort_index()
        seal_check(px, rel)
        prices[cik] = px[["open", "close", "volume"]]
    if set(prices) != usable:
        raise StopRun(f"[pead] 같은 회사 CIK {len(usable)} 중 고정 가격 없음 {len(usable - set(prices))} — 멈춤")
    liq = {c: liquid_flags(px) for c, px in prices.items()}

    # 1) 신호 재현 (사전 등록 건수와 같아야 함)
    ev = classify_events(pd.read_csv(pc.EVENTS_CSV), days, ivs, big_ivs, usable, prices, qqq["close"], liq)
    st = undecided_stats(ev)
    exp = pre["expected"]
    got = {k: st[k] for k in ("judged", "undecided", "signals", "signals_liquid")}
    if got != {k: exp[k] for k in got}:
        raise StopRun(f"[pead] 신호 재현 불일치: {got} vs 사전 등록 {exp} — 멈춤")
    dup = ev[ev["status"] == "판정"].duplicated(["cik", "d0"]).sum()
    print(f"[pead] 신호 재현 일치: {got} · 판정 불가 {st['rate']:.2%} · 같은 날 같은 종목 중복 {dup}", flush=True)

    sig_all = ev[(ev["status"] == "판정") & ev["signal"]].copy()
    sig_all["ticker"] = sig_all["cik"]
    st["dup_signals"] = int(sig_all.duplicated(["ticker", "d0"]).sum())
    st["dup_signals_liquid"] = int(sig_all[sig_all["liquid"]].duplicated(["ticker", "d0"]).sum())
    sig_all = sig_all.drop_duplicates(["ticker", "d0"])
    sig = sig_all[sig_all["liquid"]]
    cols = ["ticker", "d0", "accn", "index", "liquid"]

    # 2) 주 판정 매매
    tr0, sk0 = sw1.make_trades(sig[cols], prices, days, hold)
    zone_guard(tr0, ivs, big_ivs)
    T = price_trades(tr0, qqq, fx, c_stock, c_bench, rate)
    S = summary(T)

    # 3) 무작위 1,000회: 같은 매수일, 그 매매의 반응일에 검증 ① 대상·유동성 통과·반응일 가격 있음, 신호 종목 자신 제외
    zone_members: dict[pd.Timestamp, list[int]] = {}
    for d0 in sorted(set(T["d0"])):
        zone_members[d0] = [c for c in prices if in_zone(ivs.get(c, []), big_ivs.get(c), d0)[0]
                            and bool(liq[c].get(d0, False))]
    pools = [[c for c in zone_members[d0] if c != t] for d0, t in zip(T["d0"], T["ticker"])]
    cands = candidate_table(T, pools, prices, fx, qqq, c_stock, c_bench)
    rmeans = sx.random_means(cands, int(rnd["iterations"]), int(rnd["seed"]), rate, 2017, 5)
    rpct = sw1.percentile_of(T["ex_after"].mean(), rmeans)
    pool_sizes = [len(c["ret"]) for c in cands]

    # 4) 판정
    sig_rate = st["signals"] / st["judged"]
    mm = missed_trades_mean(T["ex_after"], st["undecided"], sig_rate)
    crit = {"periods": [tuple(p) for p in cr["periods"]], "random_pct_min": cr["random_pct_min"],
            "min_pos_years": cr["min_pos_years"], "min_trades": cr["min_trades"], "drop_top": cr["drop_top"],
            "max_undecided": cr["max_undecided_rate"]}
    V = verdict(T["ex_after"], T["year"], rpct, st["rate"], mm["mean"], crit)
    print(f"[pead] 판정 {V['result']} · 매매 {S['n']} · 평균 초과(세금 뒤) {S['ex_after']:+.2f}% · 무작위 {rpct:.1f}백분위", flush=True)

    # 5) 참고 보고 (판정 제외)
    refs: dict[str, dict] = {}
    for h in (20, 40):
        a, _ = sw1.make_trades(sig[cols], prices, days, h)
        refs[f"{h}거래일 보유"] = summary(price_trades(a, qqq, fx, c_stock, c_bench, rate))
    refs["비교 지수 IJR (S&P 600)"] = summary(price_trades(tr0, ijr, fx, c_stock, c_bench, rate))
    refs["비교 지수 MDY (S&P 400)"] = summary(price_trades(tr0, mdy, fx, c_stock, c_bench, rate))
    a, _ = sw1.make_trades(sig_all[cols], prices, days, hold)
    zone_guard(a, ivs, big_ivs)
    refs[f"유동성 필터 없이 (신호 {len(sig_all)}건)"] = summary(price_trades(a, qqq, fx, c_stock, c_bench, rate))
    refs["S&P 400만"] = summary(T[T["index"] == "sp400"])
    refs["S&P 600만"] = summary(T[T["index"] == "sp600"])
    for s_ in (0.1, 0.6):
        c2 = sw1.Costs(co["commission_pct"], co["commission_pct"], co["fx_spread_pct"], s_)
        refs[f"슬리피지 편도 {s_}%"] = summary(price_trades(tr0, qqq, fx, c2, c_bench, rate))

    OUT_DIR.mkdir(exist_ok=True)
    T.to_csv(OUT_DIR / "pead_smallmid_trades.csv", index=False, encoding="utf-8-sig")
    write_report(pre, T, S, V, st, mm, sig_rate, rmeans, pool_sizes, refs, sk0, ev)
    print(f"[pead] 완료 → {RESULT_MD.relative_to(ROOT)}", flush=True)


def write_report(pre, T, S, V, st, mm, sig_rate, rmeans, pool_sizes, refs, sk0, ev) -> None:
    """결과 문서 (맨 위에 해시 기준)."""
    ok = lambda b: "충족" if b else "미충족"  # noqa: E731
    sw1r = pre["reference_sw1"]
    L = ["# 실적급등 다음날 매수·3달 보유 — 중소형 재현 결과 (pead-smallmid)", "",
         "> 해시 기준 (LF 정규화 sha256, 태그 `pead-smallmid-prereg` 커밋 내용 = 작업 폴더 = 스크립트 기록값 확인 후 실행)",
         *[f"> - `{k}` `{v}`" for k, v in PREREG_FILES.items()],
         f"> 캐시: \"10월 한도 시점\" 표 {len(pre['data_freeze']['files'])}개 · 가격 {len(pre['data_freeze']['prices'])}개 · 환율 시리즈 sha256 모두 사전 등록 값과 같음",
         "> 구현 해석: `docs/pead_smallmid_interpretations.md` (결과 계산 전 커밋)", "",
         "- 구역: 검증 ① (S&P 400 ∪ 600 − 나스닥100 ∪ S&P 100, 반응일 2017-01-01 ~ 2021-12-31), 시험 예산 1/5",
         "- 매수 반응일 다음 거래일 시가, 매도 60거래일 뒤 종가, 손절 없음 · 비용: 수수료 0.07%·환전 0.1%·슬리피지 0.3% (편도, 슬리피지는 종목만)",
         "- 세금: 매도 연도별 순이익 22% (공제 무시, 이익 비례 배분) · 무작위 1,000회, 시드 "
         f"{pre['benchmarks']['random']['seed']}", "",
         f"## 판정: **{V['result']}**", "",
         "| 번호 | 기준 | 값 | 결과 |", "| --- | --- | --- | --- |",
         f"| ① | 매매당 평균 초과 수익(QQQ 대비, 비용·세금 뒤) > 0 | {pct(V['mean'] * 100)} | {ok(V['c1'])} |",
         f"| ② | 무작위 1,000회의 99백분위 이상 | {V['random_pct']:.1f}백분위 | {ok(V['c2'])} |",
         f"| ③ | 2017~2019 · 2020~2021 모두 > 0 | {pct(V['period_means'][0] * 100)} / {pct(V['period_means'][1] * 100)} | {ok(V['c3'])} |",
         f"| ④ | 5년 중 3년 이상 > 0 | {V['pos_years']}/{V['n_years']}년 | {ok(V['c4'])} |",
         f"| ⑤ | 매매 300건 이상 | {V['n']}건 | {ok(V['c5'])} |",
         f"| ⑥ | 상위 5건 제외 평균 > 0 | {pct(V['top_out'] * 100)} | {ok(V['c6'])} |",
         f"| ⑦ | 놓친 매매 {mm['k']}건(판정 불가 {st['undecided']} × 신호율 {sig_rate:.2%})을 하위 10% 평균 {pct(mm['fill'] * 100)}로 채운 평균 > 0 | {pct(mm['mean'] * 100)} | {ok(V['c7'])} |",
         f"| ⑧ | 판정 불가 비율 ≤ 10% | {st['rate']:.2%} ({st['undecided']}/{st['den']}) | {ok(V['c8'])} |", "",
         "## 핵심 숫자", "", "| 항목 | 값 |", "| --- | --- |",
         f"| 매매 수 | {S['n']} |",
         f"| 평균 수익: 신호 / QQQ (달러, 비용 전) | {pct(S['ret_usd'])} / {pct(S['bench_usd'])} |",
         f"| 평균 초과 수익: 비용 전 | {pct(S['ex_pre'])} |",
         f"| 평균 초과 수익: 비용 뒤 (원화, 세금 전) | {pct(S['ex_cost'])} |",
         f"| 평균 초과 수익: 비용·세금 뒤 | {pct(S['ex_after'])} |",
         f"| 승률 (세금 뒤 QQQ보다 나은 매매) | {S['win']:.1f}% |",
         f"| 중앙값 초과(세금 뒤) | {pct(T['ex_after'].median() * 100)} |",
         f"| 무작위 평균 초과(세금 뒤) 중앙값 · 99백분위 | {pct(np.median(rmeans) * 100)} · {pct(np.percentile(rmeans, 99) * 100)} |",
         f"| 실제 값의 무작위 백분위 (p = 1 − 백분위/100) | {V['random_pct']:.1f} (p = {1 - V['random_pct'] / 100:.3f}) |",
         f"| 무작위 후보 수 (매매당) 중앙값 · 최소 | {int(np.median(pool_sizes))} · {min(pool_sizes)} |",
         "", "## 연도별 (반응일 연도)", "", "| 연도 | 매매 | 초과 비용 전 | 초과 비용 뒤 | 초과 세금 뒤 | 승률 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for y, g in T.groupby("year"):
        L.append(f"| {y} | {len(g)} | {pct(g['ex_pre'].mean() * 100)} | {pct(g['ex_cost'].mean() * 100)} | "
                 f"{pct(g['ex_after'].mean() * 100)} | {(g['ex_after'] > 0).mean() * 100:.0f}% |")
    L += ["", "## 구간별", "", "| 구간 | 매매 | 초과 비용 전 | 초과 비용 뒤 | 초과 세금 뒤 | 승률 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for nm, g in (("2017~2019", T[T["year"] <= 2019]), ("2020~2021", T[T["year"] >= 2020])):
        L.append(f"| {nm} | {len(g)} | {pct(g['ex_pre'].mean() * 100)} | {pct(g['ex_cost'].mean() * 100)} | "
                 f"{pct(g['ex_after'].mean() * 100)} | {(g['ex_after'] > 0).mean() * 100:.0f}% |")
    L += ["", "## 참고 보고 (판정 제외)", "",
          "| 보고 | 매매 | 초과 비용 전 | 초과 세금 뒤 | 승률 | 2017~19 / 2020~21 | 양수 연도 |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for nm, r in refs.items():
        if not r.get("n"):
            L.append(f"| {nm} | 0 | — | — | — | — | — |")
            continue
        L.append(f"| {nm} | {r['n']} | {pct(r['ex_pre'])} | {pct(r['ex_after'])} | {r['win']:.1f}% | "
                 f"{pct(r['p1'])} / {pct(r['p2'])} | {r['pos_years']}/{r['n_years']} |")
    L += ["", "- IJR·MDY 비교: 같은 매매, 비교 지수만 바꿈 (지수 쪽 슬리피지 없음). S&P 400만·600만: 주 판정 매매를 반응일 지수로 나눠 본 것(세금은 주 판정 장부 값)",
          "- 유동성 필터 없이: 신호 전체로 매매를 다시 만듦(보유 중 무시도 다시 적용)", "",
          "### 대형주 SW1과 나란히", "",
          "| | 대형 SW1 (2012~2021) | 대형 SW1 2017~2021 | **중소형 (2017~2021)** |", "| --- | --- | --- | --- |",
          f"| 매매 | {sw1r['n']} | {sw1r['n_2017_2021']} | **{S['n']}** |",
          f"| 평균 초과 비용 전 | {sw1r['ex_pre']} | {sw1r['ex_pre_2017_2021']} | **{pct(S['ex_pre'])}** |",
          f"| 평균 초과 비용·세금 뒤 | {sw1r['ex_after']} | {sw1r['ex_after_2017_2021']} | **{pct(S['ex_after'])}** |",
          f"| 승률 | {sw1r['win']} | {sw1r['win_2017_2021']} | **{S['win']:.1f}%** |",
          f"| 무작위 백분위 | {sw1r['random_pct']} | — | **{V['random_pct']:.1f}** |",
          "| 비용 | 수수료·환전 | 수수료·환전 | 수수료·환전 + 슬리피지 0.3% |", ""]
    sk = sk0["skip"].value_counts().to_dict()
    L += ["## 판정 불가·공백", "", "| 항목 | 값 |", "| --- | --- |",
          f"| 2017~2021 편입 중·대형 겹침 아닌 발표 | {st['den']} |",
          f"| 판정 | {st['judged']} (신호 {st['signals']}, 신호+유동성 {st['signals_liquid']}) |",
          f"| 판정 불가 (가격 공백·다른 회사 + 가격 부족) | {st['undecided']} ({st['rate']:.2%}) |",
          f"| 그 밖 제외 (편입 아님 · 대형 겹침) | {int((ev['status'] == '그날 편입 아님').sum())} · {int((ev['status'] == '대형 겹침(앞뒤 한 분기)').sum())} |",
          f"| 같은 CIK·같은 반응일 신호 중복 (1건만 씀, 해석 7) | 신호 {st['dup_signals']} · 신호+유동성 {st['dup_signals_liquid']} |",
          f"| 신호+유동성 중 매매 안 됨 | {sum(sk.values())}/{st['signals_liquid'] - st['dup_signals_liquid']} — " + (", ".join(f"{k} {v}" for k, v in sk.items()) or "없음") + " |",
          "| 매도 사유 | " + ", ".join(f"{k} {v}" for k, v in T["exit_reason"].value_counts().to_dict().items()) + " |", "",
          "## 결과를 본 뒤 바꾼 점", "",
          "없음 (계산·규칙 변경 없음. 첫 실행 뒤 보고 줄 1개만 추가: 같은 CIK·같은 반응일 신호 중복 건수 — 해석 7의 \"생기면 보고\")", ""]
    RESULT_MD.write_text("\n".join(L), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    run()
