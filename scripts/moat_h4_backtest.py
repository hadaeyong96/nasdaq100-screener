"""해자 과거 검증 (H4, docs/design/moat_plan.md 4장, 사용자 지시 2026-10-01).

계획서 4장 규칙을 그대로 구현한다. 시작할 때 core.moat.check_lock()을 확인하고
경고가 뜨면 즉시 멈춘다 — moat: 설정이 잠긴 기준과 다르면 등급 판정 자체가
무의미하다. 2022-01-01 이후 데이터는 core.seal.enforce_not_sealed로 막혀 있다
(engine.backtest.prepare_data가 이미 강제 — 이 스크립트는 SEAL_END=2021-12-31을
넘는 날짜를 어떤 호출에도 넘기지 않는다).

**결과를 보고 설정·규칙을 바꾸지 않는다**(사용자 지시) — 이 스크립트는 한 번 실행하고
report.json으로 결과만 남긴다. 결과가 마음에 안 들어도 여기 숫자·로직을 결과에 맞춰
고치지 않는다.

**모형 단순화(보고서에 명시할 것)**:
1. 세금은 "다음 해 5월 납부"(engine/backtest.py의 실제 현금 타이밍)가 아니라 매 교체일에
   그 자리에서 뗀다 — 기간별 세후 CAGR 계산 목적엔 충분하지만 정확한 현금 흐름 시점과는
   다르다.
2. 넓음·좁음이상·없음 포트폴리오의 "mdd_pct"는 연 1회 교체 시점만 평가한 값이다(사용자
   지시 2026-10-01로 "daily_mdd_pct" 필드를 추가해 일별 종가 기준 MDD도 함께 보고한다 —
   진입 비중을 교체일에 고정하고 기간 중엔 재배분 없이 자연스럽게 흘러가게 둔 값, 세전·
   배당 미반영. 연복리·백분위 같은 판정 값은 이 보강과 무관하게 그대로다). 무작위
   포트폴리오 1,000회는 전수 일별 MDD까지 계산하지 않는다(판정에 쓰는 백분위는 수익
   배수만 필요하고, 1,000회×일별 계산은 과한 비용).
3. 인수·상장폐지로 종목 데이터가 중간에 끝나면 마지막 종가로 처분한 뒤 다음 거래일
   QQQM 매수로 잔여 기간을 대체한다(슬리피지는 처분 시에만, QQQM 편입엔 매수수수료만).
4. Tiingo 대체 캐시 종목은 배당 데이터가 없어 배당수익률을 0으로 본다(F2에서 이미
   확인된 한계).
5. 업종 분류는 yfinance(무료, Morningstar 기반)로 현재 시점 분류를 쓴다 — 계획서에
   명시된 한계 그대로(과거 시점 분류와 다를 수 있음).

실행(.env에 SEC_USER_AGENT 필요, TIINGO_API_KEY는 있으면 대체 캐시를 추가로 씀):
    python -u -m scripts.moat_h4_backtest
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import engine.backtest as bt  # noqa: E402
from core import moat  # noqa: E402
from core import moat_backtest as mbt  # noqa: E402
from data import edgar  # noqa: E402
from data import universe_history as uh  # noqa: E402
from scripts.fund_dataqc_report import load_alt_source_indicator_map  # noqa: E402
from scripts.moat_report import (  # noqa: E402
    dedupe_rows_by_cik,
    fetch_sector_from_yfinance,
    get_sector,
    load_sector_cache,
    resolve_cik,
    save_sector_cache,
)
from store import trials as trials_store  # noqa: E402

OUT_DIR = ROOT / "outputs" / "moat"
REPORT_PATH = OUT_DIR / "h4_report.json"
LOG_PATH = OUT_DIR / "h4_report.log"

WARMUP_START = date(2014, 1, 1)
SEAL_END = date(2021, 12, 31)
YEARS = list(range(2015, 2022))
FORMAL_START_YEAR = 2019
LOCK_TRIAL_LABEL = "해자 지표 v1.1 확정 및 잠금"


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(msg, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def get_locked_hash() -> str:
    """store/trials.db에 기록된 잠금 시점 config_hash를 읽는다 (scripts/moat_lock.py 기록).

    출력: 잠긴 해시 문자열
    예외: 잠금 기록이 없으면 SystemExit로 멈춘다 — H4는 무엇과 비교해 확인해야 할지
         모르는 상태로 실행하면 안 된다.
    """
    conn = trials_store.connect()
    try:
        trials = trials_store.load_all_trials(conn)
    finally:
        conn.close()
    matches = [t for t in trials if t["label"] == LOCK_TRIAL_LABEL]
    if not matches:
        raise SystemExit(f"store/trials.db에 '{LOCK_TRIAL_LABEL}' 기록이 없습니다 — 잠금 해시를 확인할 수 없어 멈춥니다.")
    return matches[-1]["config_hash"]


def first_trading_day_of_month(trading_days: list[pd.Timestamp], year: int, month: int) -> pd.Timestamp:
    candidates = [d for d in trading_days if d.year == year and d.month == month]
    if not candidates:
        raise ValueError(f"{year}-{month:02d}에 거래일이 없습니다(가격 데이터 구간을 벗어남)")
    return min(candidates)


def next_trading_day_after(trading_days: list[pd.Timestamp], d: pd.Timestamp) -> pd.Timestamp:
    for td in trading_days:
        if td > d:
            return td
    raise ValueError(f"{d.date()} 다음 거래일이 없습니다(데이터 끝 — 봉인 경계 확인 필요)")


def _price_at(df: pd.DataFrame | None, ts: pd.Timestamp, col: str) -> float | None:
    if df is None or ts not in df.index:
        return None
    v = df.loc[ts, col]
    return float(v) if pd.notna(v) else None


def _dividends_between(div_series: pd.Series | None, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if div_series is None or not len(div_series):
        return 0.0
    window = div_series.loc[start:end]
    return float(window.sum()) if len(window) else 0.0


def stock_leg_return(
    ticker: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    is_final_mark: bool,
    price_map: dict,
    div_map: dict,
    cash_df: pd.DataFrame,
    cash_div: pd.Series,
    costs_cfg: dict,
    div_wh_rate: float,
) -> tuple[tuple[float, float] | None, str | None]:
    """한 종목의 한 교체 기간 수익을 구한다. 중간에 데이터가 끝나면(인수·상장폐지
    추정) 마지막 종가로 처분하고 잔여 기간은 QQQM으로 대체한다 (계획서 4장 "보유").

    출력: ((price_return_factor, dividend_aftertax_yield) 또는 None, 메모 또는 None).
         None이면 그 기간에 이 종목을 가격을 몰라 편입할 수 없다는 뜻 — 호출부가
         포트폴리오에서 제외하고 나머지 종목으로 재배분한다.
    """
    df = price_map.get(ticker)
    if df is None or not len(df.index):
        return None, "가격 데이터 없음"
    entry_open = _price_at(df, entry_date, "open")
    if entry_open is None or entry_open <= 0:
        return None, "진입일(체결일) 시가 없음"

    last_avail = df.index.max()
    div_series = div_map.get(ticker)

    if last_avail >= exit_date:
        exit_price = _price_at(df, exit_date, "close" if is_final_mark else "open")
        if exit_price is None:
            return None, "청산일 가격 없음"
        divs = _dividends_between(div_series, entry_date, exit_date)
        pf, dy = mbt.compute_stock_period_return(entry_open, exit_price, divs, costs_cfg, div_wh_rate, apply_sell_costs=not is_final_mark)
        return (pf, dy), None

    # 이 종목 데이터가 exit_date 전에 끝남 -> 인수·상장폐지로 실제 거래가 멈춘 것으로 본다.
    delist_date = last_avail
    exit_at_delist = _price_at(df, delist_date, "close")
    if exit_at_delist is None:
        return None, f"{delist_date.date()} 상장폐지 추정, 마지막 가격도 없음"
    divs1 = _dividends_between(div_series, entry_date, delist_date)
    pf1, dy1 = mbt.compute_stock_period_return(entry_open, exit_at_delist, divs1, costs_cfg, div_wh_rate, apply_sell_costs=True)
    note = f"{ticker}: {delist_date.date()} 이후 가격 데이터 없음(인수·상장폐지 추정) — 잔여 기간 QQQM 대체"

    qqqm_entry_candidates = cash_df.index[cash_df.index > delist_date]
    if len(qqqm_entry_candidates) == 0 or cash_df.index.max() < exit_date:
        return (pf1, dy1), note + " (QQQM 편입 실패 — 상장폐지 시점까지만 반영)"
    qqqm_entry_date = qqqm_entry_candidates[0]
    qqqm_entry_open = _price_at(cash_df, qqqm_entry_date, "open")
    qqqm_exit_price = _price_at(cash_df, exit_date, "close" if is_final_mark else "open")
    if qqqm_entry_open is None or qqqm_exit_price is None:
        return (pf1, dy1), note + " (QQQM 가격 조회 실패 — 상장폐지 시점까지만 반영)"
    divs2 = _dividends_between(cash_div, qqqm_entry_date, exit_date)
    pf2, dy2 = mbt.compute_stock_period_return(qqqm_entry_open, qqqm_exit_price, divs2, costs_cfg, div_wh_rate, apply_sell_costs=not is_final_mark)
    combined_pf = pf1 * pf2
    combined_dy = dy1 + dy2 * pf1  # 2구간 배당은 1구간 종료 후 자본 기준(근사)
    return (combined_pf, combined_dy), note


def build_group_periods(
    group_tickers_by_year: dict,
    stock_returns_by_year: dict,
    sectors_by_year: dict,
    execution_dates: dict,
    final_mark_date: pd.Timestamp,
    fx_by_date: dict,
    sector_cap_pct: float,
) -> tuple[list, dict]:
    """한 그룹(넓음/좁음이상/없음/무작위 한 회차)의 7개 교체 기간 PeriodResult를 만든다."""
    periods = []
    sector_cap_triggered = {}
    for i, y in enumerate(YEARS):
        entry_date = execution_dates[y]
        is_final = y == YEARS[-1]
        exit_date = execution_dates[YEARS[i + 1]] if not is_final else final_mark_date
        returns = stock_returns_by_year[y]
        tickers = [t for t in group_tickers_by_year[y] if t in returns]
        sectors = sectors_by_year[y]
        weights = mbt.allocate_equal_weight_with_sector_cap(tickers, sectors, sector_cap_pct)
        equal_w = 1 / len(tickers) if tickers else 0.0
        sector_cap_triggered[y] = any(abs(w - equal_w) > 1e-9 for w in weights.values())
        pf, dy = mbt.portfolio_period_return(returns, weights)
        fx_entry = fx_by_date.get(entry_date.date().isoformat())
        fx_exit = fx_by_date.get(exit_date.date().isoformat())
        periods.append(
            mbt.PeriodResult(
                entry_date=entry_date.date().isoformat(), exit_date=exit_date.date().isoformat(),
                fx_entry=fx_entry, fx_exit=fx_exit, price_return_factor=pf, dividend_aftertax_yield=dy,
            )
        )
    return periods, sector_cap_triggered


def build_qqqm_control_period(
    start_year: int, mark_date: pd.Timestamp, execution_dates: dict, fx_by_date: dict,
    costs_cfg: dict, div_wh_rate: float, cash_df: pd.DataFrame, cash_div: pd.Series,
) -> "mbt.PeriodResult":
    """QQQM(대조군 A) — 창(window) 시작에 한 번 사고 끝에서 평가(세금은 그때 한 번)."""
    entry_date = execution_dates[start_year]
    entry_open = _price_at(cash_df, entry_date, "open")
    exit_price = _price_at(cash_df, mark_date, "close")
    divs = _dividends_between(cash_div, entry_date, mark_date)
    pf, dy = mbt.compute_stock_period_return(entry_open, exit_price, divs, costs_cfg, div_wh_rate, apply_sell_costs=False)
    fx_entry = fx_by_date.get(entry_date.date().isoformat())
    fx_exit = fx_by_date.get(mark_date.date().isoformat())
    return mbt.PeriodResult(
        entry_date=entry_date.date().isoformat(), exit_date=mark_date.date().isoformat(),
        fx_entry=fx_entry, fx_exit=fx_exit, price_return_factor=pf, dividend_aftertax_yield=dy, note="QQQM control",
    )


def _ticker_valuation_series(
    ticker: str, entry_date: pd.Timestamp, exit_date: pd.Timestamp, price_map: dict, cash_df: pd.DataFrame, days_index: pd.DatetimeIndex,
) -> pd.Series | None:
    """[entry_date, exit_date] 구간의 종가 일별 시계열(일별 MDD용, 세전·배당 미반영
    — 상대적 낙폭 모양만 본다). 상장폐지·인수로 데이터가 중간에 끝나면 stock_leg_return과
    같은 규칙(처분가 고정 비율로 QQQM 종가를 이어 붙임)을 일별로 적용한다.
    """
    df = price_map.get(ticker)
    if df is None:
        return None
    window_idx = days_index[(days_index >= entry_date) & (days_index <= exit_date)]
    if len(window_idx) == 0:
        return None
    s = df["close"].reindex(window_idx).ffill()
    last_avail = df.index.max()
    if last_avail >= exit_date:
        return s
    before = s.loc[s.index <= last_avail].dropna()
    if before.empty:
        return None
    stock_price_at_delist = float(before.iloc[-1])
    qqqm_candidates = cash_df.index[cash_df.index > last_avail]
    if len(qqqm_candidates) == 0:
        return s.ffill()  # QQQM으로 못 넘어가면 마지막 값 그대로 유지(근사)
    qqqm_entry_date = qqqm_candidates[0]
    qqqm_entry_price = _price_at(cash_df, qqqm_entry_date, "open")
    if not qqqm_entry_price:
        return s.ffill()
    qqqm_close = cash_df["close"].reindex(window_idx).ffill()
    scale = stock_price_at_delist / qqqm_entry_price
    combined = s.copy()
    after_mask = window_idx > last_avail
    combined.loc[after_mask] = qqqm_close.loc[after_mask] * scale
    return combined.ffill()


def group_daily_equity_rows(
    tickers_by_year: dict,
    stock_returns_by_year: dict,
    sectors_by_year: dict,
    execution_dates: dict,
    final_mark_date: pd.Timestamp,
    fx_by_date: dict,
    sector_cap_pct: float,
    price_map: dict,
    cash_df: pd.DataFrame,
    days_index: pd.DatetimeIndex,
) -> list[dict]:
    """한 그룹(넓음/좁음이상/없음)의 2015~2021 전체를 일별로 이어 붙인 상대값 곡선
    (세전·배당 미반영, MDD 계산 전용 — 사용자 지시 2026-10-01: "교체일이 아니라
    일별 가격으로 다시 계산"). 연복리·백분위(판정에 쓰는 값)는 그대로 둔다 — 이
    함수는 보고서의 MDD 필드만 다시 채우는 데 쓴다.

    진입 비중은 교체일에 고정하고, 기간 중에는 각 종목의 실제 가격 변동에 따라
    자연히 비중이 흘러가게 둔다(재배분하지 않음 — 실제 보유 포지션과 같은 모양).
    """
    rows: list[dict] = []
    level = 1.0
    for i, y in enumerate(YEARS):
        entry_date = execution_dates[y]
        is_final = y == YEARS[-1]
        exit_date = execution_dates[YEARS[i + 1]] if not is_final else final_mark_date
        returns = stock_returns_by_year[y]
        tickers = [t for t in tickers_by_year[y] if t in returns]
        weights = mbt.allocate_equal_weight_with_sector_cap(tickers, sectors_by_year[y], sector_cap_pct)

        period_days = days_index[(days_index >= entry_date) & (days_index <= exit_date)]
        series_by_ticker: dict[str, pd.Series] = {}
        entry_prices: dict[str, float] = {}
        for t in weights:
            s = _ticker_valuation_series(t, entry_date, exit_date, price_map, cash_df, days_index)
            ep = _price_at(price_map.get(t), entry_date, "open")
            if s is None or not ep:
                continue
            series_by_ticker[t] = s
            entry_prices[t] = ep
        total_w = sum(weights[t] for t in series_by_ticker)
        if total_w <= 0 or not len(period_days):
            continue

        period_last_rel = 1.0
        for d in period_days:
            rel = 0.0
            for t, s in series_by_ticker.items():
                price = s.get(d)
                if price is None or pd.isna(price):
                    continue
                rel += (weights[t] / total_w) * (price / entry_prices[t])
            fx = fx_by_date.get(d.date().isoformat())
            if fx is None:
                continue
            rows.append({"date": d.date().isoformat(), "total_krw": level * rel * fx})
            period_last_rel = rel
        level = level * period_last_rel
    return rows


def qqqm_daily_shape_rows(cash_df: pd.DataFrame, entry_date: pd.Timestamp, exit_date: pd.Timestamp, fx_by_date: dict) -> list[dict]:
    """QQQM의 MDD를 일별 종가(세전, 배당 미반영)로 계산하기 위한 상대값 시계열.

    넓음 등 포트폴리오는 연 1회 교체 시점만 평가해 MDD가 실제보다 작게 나올 수 있다 —
    QQQM은 일별 데이터가 있어 더 정확한 MDD를 낼 수 있다(비교 시 이 비대칭을 감안).
    """
    window = cash_df.loc[(cash_df.index >= entry_date) & (cash_df.index <= exit_date)]
    if window.empty:
        return []
    base = float(window["close"].iloc[0])
    rows = []
    for ts, row in window.iterrows():
        fx = fx_by_date.get(ts.date().isoformat())
        if fx is None or pd.isna(row["close"]):
            continue
        rows.append({"date": ts.date().isoformat(), "total_krw": float(row["close"]) / base * fx})
    return rows


def main() -> None:
    cfg = bt.load_config()
    locked_hash = get_locked_hash()
    warning = moat.check_lock(cfg["moat"], locked_hash)
    if warning:
        log(warning)
        raise SystemExit("moat 설정이 잠긴 기준과 다릅니다 — H4를 멈춥니다.")
    log(f"check_lock 통과 — moat 설정이 잠긴 기준(해시 {locked_hash})과 일치")

    user_agent = edgar.get_user_agent()
    mbt_cfg = cfg["moat_backtest"]
    costs_cfg = cfg["backtest"]["costs"]
    tax_cfg = cfg["backtest"]["tax"]
    div_wh_rate = tax_cfg["dividend_withholding_pct"] / 100
    sector_cap_pct = mbt_cfg["sector_cap_pct"]

    log("가격·환율·시점별 유니버스 준비 중 (engine.backtest.prepare_data 재사용, 2021-12-31 봉인)...")
    data = bt.prepare_data(cfg, WARMUP_START, SEAL_END)
    log(f"yfinance 가격 확보 {len(data.indicator_map)}종목 / 실패 {len(data.failed_tickers)}종목")

    alt_prices = load_alt_source_indicator_map()
    added_from_alt = sorted(t for t in alt_prices if t not in data.indicator_map)
    price_map = dict(data.indicator_map)
    price_map.update({t: alt_prices[t] for t in added_from_alt})
    log(f"Tiingo 대체 캐시로 {len(added_from_alt)}종목 보완: {added_from_alt} (배당 데이터 없음 — 0으로 처리)")
    still_missing = sorted(t for t in data.failed_tickers if t not in alt_prices)
    if still_missing:
        log(f"가격을 끝내 못 구한 종목 {len(still_missing)}개(포트폴리오 구성 시 제외): {still_missing}")

    div_map = data.dividends
    trading_days = sorted(data.qqq_df.index)

    ticker_to_cik = edgar.fetch_ticker_to_cik(user_agent=user_agent)
    sector_cache = load_sector_cache()

    rebalance_dates = {y: first_trading_day_of_month(trading_days, y, mbt_cfg["rebalance_month"]) for y in YEARS}
    execution_dates = {y: next_trading_day_after(trading_days, rebalance_dates[y]) for y in YEARS}
    final_mark_date = trading_days[-1]
    log("교체일(매년 4월 첫 거래일) / 체결일(다음 거래일 시가):")
    for y in YEARS:
        log(f"  {y}: 교체일 {rebalance_dates[y].date()} -> 체결일 {execution_dates[y].date()}")
    log(f"봉인 평가일(2021년 4월 편입분 마감 평가): {final_mark_date.date()}")

    # ── 연도별 시점별 유니버스 + 해자 등급 ───────────────────────────────────
    universe_year_stats: dict[int, dict] = {}
    sectors_by_year: dict[int, dict] = {}
    for y in YEARS:
        rebalance_date_py = rebalance_dates[y].date()
        judgment_date = rebalance_date_py - timedelta(days=1)
        universe = sorted(uh.universe_on(data.checkpoints, rebalance_date_py))
        rows = []
        for ticker in universe:
            sector = get_sector(ticker, sector_cache, fetch_sector_from_yfinance)
            cik = resolve_cik(ticker, ticker_to_cik)
            if cik is None:
                rows.append({"ticker": ticker, "cik": None, "grade": "판단 불가", "sector": sector})
                continue
            try:
                facts = edgar.fetch_company_facts(cik, user_agent=user_agent)
            except edgar.EdgarFetchError:
                rows.append({"ticker": ticker, "cik": cik, "grade": "판단 불가", "sector": sector})
                continue
            profile = moat.analyze_company(facts, ticker, cfg, judgment_date)
            rows.append({"ticker": ticker, "cik": cik, "grade": profile.grade, "sector": sector})

        deduped = dedupe_rows_by_cik(rows)
        sectors_by_year[y] = {r["ticker"]: r["sector"] for r in deduped}
        wide = [r["ticker"] for r in deduped if r["grade"] == "넓음"]
        narrow_plus = [r["ticker"] for r in deduped if r["grade"] in ("넓음", "좁음")]
        none_grade = [r["ticker"] for r in deduped if r["grade"] == "없음"]
        inconclusive = sum(1 for r in deduped if r["grade"] == "판단 불가")
        universe_year_stats[y] = {
            "rebalance_date": str(rebalance_date_py), "judgment_date": str(judgment_date),
            "universe_size": len(deduped), "wide_count": len(wide), "narrow_plus_count": len(narrow_plus),
            "none_count": len(none_grade), "inconclusive_count": inconclusive,
            "wide_tickers": wide, "narrow_plus_tickers": narrow_plus, "none_tickers": none_grade,
            "all_tickers": [r["ticker"] for r in deduped],
        }
        log(f"{y}년: 유니버스(중복제거) {len(deduped)}개, 넓음 {len(wide)}개, 좁음이상 {len(narrow_plus)}개, 없음 {len(none_grade)}개, 판단불가 {inconclusive}개")

    save_sector_cache(sector_cache)

    # ── 종목×연도 기간 수익(가격·배당, 세전) 계산 ────────────────────────────
    stock_returns_by_year: dict[int, dict] = {}
    price_notes: list[str] = []
    for i, y in enumerate(YEARS):
        entry_date = execution_dates[y]
        is_final = y == YEARS[-1]
        exit_date = execution_dates[YEARS[i + 1]] if not is_final else final_mark_date
        returns = {}
        for ticker in universe_year_stats[y]["all_tickers"]:
            result, note = stock_leg_return(
                ticker, entry_date, exit_date, is_final, price_map, div_map, data.cash_etf_df, data.cash_etf_dividends, costs_cfg, div_wh_rate,
            )
            if result is None:
                price_notes.append(f"{y}년 {ticker}: 편입 불가 — {note}")
                continue
            returns[ticker] = result
            if note:
                price_notes.append(f"{y}년 {note}")
        stock_returns_by_year[y] = returns
        universe_year_stats[y]["priced_tickers"] = len(returns)
    for n in price_notes:
        log(f"  [가격 메모] {n}")

    # ── 넓음/좁음이상/없음 — 2015~2021 연속 체인 ─────────────────────────────
    starting_capital_krw = cfg["backtest"]["total_krw"]
    fx0 = data.fx_by_date.get(execution_dates[2015].date().isoformat())
    starting_capital_usd = bt._deposit_krw_to_usd(starting_capital_krw, fx0, costs_cfg["fx_spread_pct"], apply_costs=True)

    groups = {
        "넓음": {y: universe_year_stats[y]["wide_tickers"] for y in YEARS},
        "좁음이상": {y: universe_year_stats[y]["narrow_plus_tickers"] for y in YEARS},
        "없음": {y: universe_year_stats[y]["none_tickers"] for y in YEARS},
    }
    group_results: dict[str, dict] = {}
    sector_cap_triggered_wide: dict[int, bool] = {}
    days_index = pd.DatetimeIndex(trading_days)
    formal_boundary_iso = execution_dates[FORMAL_START_YEAR].date().isoformat()
    for name, tickers_by_year in groups.items():
        periods, triggered = build_group_periods(
            tickers_by_year, stock_returns_by_year, sectors_by_year, execution_dates, final_mark_date, data.fx_by_date, sector_cap_pct,
        )
        if name == "넓음":
            sector_cap_triggered_wide = triggered
        rows = mbt.chain_periods_with_tax(periods, starting_capital_usd, tax_cfg)
        reference_rows = rows[0:5]
        formal_rows = rows[4:8]

        # 일별 가격 기준 MDD 재계산 (사용자 지시 2026-10-01) — 연복리·백분위(판정 값)는 안 바꾼다.
        daily_rows = group_daily_equity_rows(
            tickers_by_year, stock_returns_by_year, sectors_by_year, execution_dates, final_mark_date,
            data.fx_by_date, sector_cap_pct, price_map, data.cash_etf_df, days_index,
        )
        daily_reference_rows = [r for r in daily_rows if r["date"] <= formal_boundary_iso]
        daily_formal_rows = [r for r in daily_rows if r["date"] >= formal_boundary_iso]
        reference_metrics = bt.compute_equity_metrics(reference_rows)
        formal_metrics = bt.compute_equity_metrics(formal_rows)
        reference_metrics["daily_mdd_pct"] = bt.compute_equity_metrics(daily_reference_rows).get("mdd_pct") if daily_reference_rows else None
        formal_metrics["daily_mdd_pct"] = bt.compute_equity_metrics(daily_formal_rows).get("mdd_pct") if daily_formal_rows else None

        group_results[name] = {
            "equity_rows": rows,
            "reference_metrics": reference_metrics,
            "formal_metrics": formal_metrics,
            "formal_total_return_factor": formal_rows[-1]["total_krw"] / formal_rows[0]["total_krw"] if formal_rows[0]["total_krw"] else None,
            "reference_total_return_factor": reference_rows[-1]["total_krw"] / reference_rows[0]["total_krw"] if reference_rows[0]["total_krw"] else None,
        }
        log(
            f"{name}: 참고구간 세후CAGR {reference_metrics.get('cagr_pct')}% (교체일MDD {reference_metrics.get('mdd_pct')}% / 일별MDD {reference_metrics.get('daily_mdd_pct')}%)"
            f" / 정식구간 세후CAGR {formal_metrics.get('cagr_pct')}% (교체일MDD {formal_metrics.get('mdd_pct')}% / 일별MDD {formal_metrics.get('daily_mdd_pct')}%)"
        )

    # ── QQQM 대조군 A (참고/정식 구간 각각 새로 매수) ───────────────────────
    qqqm_ref_period = build_qqqm_control_period(2015, execution_dates[FORMAL_START_YEAR], execution_dates, data.fx_by_date, costs_cfg, div_wh_rate, data.cash_etf_df, data.cash_etf_dividends)
    qqqm_formal_period = build_qqqm_control_period(FORMAL_START_YEAR, final_mark_date, execution_dates, data.fx_by_date, costs_cfg, div_wh_rate, data.cash_etf_df, data.cash_etf_dividends)

    fx_ref0 = data.fx_by_date.get(execution_dates[2015].date().isoformat())
    fx_formal0 = data.fx_by_date.get(execution_dates[FORMAL_START_YEAR].date().isoformat())
    cap_ref_usd = bt._deposit_krw_to_usd(starting_capital_krw, fx_ref0, costs_cfg["fx_spread_pct"], apply_costs=True)
    cap_formal_usd = bt._deposit_krw_to_usd(starting_capital_krw, fx_formal0, costs_cfg["fx_spread_pct"], apply_costs=True)

    qqqm_ref_rows = mbt.chain_periods_with_tax([qqqm_ref_period], cap_ref_usd, tax_cfg)
    qqqm_formal_rows = mbt.chain_periods_with_tax([qqqm_formal_period], cap_formal_usd, tax_cfg)

    qqqm_ref_daily = qqqm_daily_shape_rows(data.cash_etf_df, execution_dates[2015], execution_dates[FORMAL_START_YEAR], data.fx_by_date)
    qqqm_formal_daily = qqqm_daily_shape_rows(data.cash_etf_df, execution_dates[FORMAL_START_YEAR], final_mark_date, data.fx_by_date)
    qqqm_ref_metrics = bt.compute_equity_metrics(qqqm_ref_rows)
    qqqm_formal_metrics = bt.compute_equity_metrics(qqqm_formal_rows)
    qqqm_ref_metrics["mdd_pct"] = bt.compute_equity_metrics(qqqm_ref_daily).get("mdd_pct") if qqqm_ref_daily else qqqm_ref_metrics.get("mdd_pct")
    qqqm_formal_metrics["mdd_pct"] = bt.compute_equity_metrics(qqqm_formal_daily).get("mdd_pct") if qqqm_formal_daily else qqqm_formal_metrics.get("mdd_pct")
    qqqm_formal_total_return_factor = qqqm_formal_rows[-1]["total_krw"] / qqqm_formal_rows[0]["total_krw"]
    qqqm_ref_total_return_factor = qqqm_ref_rows[-1]["total_krw"] / qqqm_ref_rows[0]["total_krw"]
    log(f"QQQM/QQQ 대조군: 참고구간 세후CAGR {qqqm_ref_metrics.get('cagr_pct')}% (일별MDD {qqqm_ref_metrics.get('mdd_pct')}%) / 정식구간 세후CAGR {qqqm_formal_metrics.get('cagr_pct')}% (일별MDD {qqqm_formal_metrics.get('mdd_pct')}%)")

    # ── 무작위 포트폴리오 1,000회 ────────────────────────────────────────────
    n_trials = mbt_cfg["random_trials"]
    master_seed = mbt_cfg["random_seed"]
    log(f"무작위 포트폴리오 {n_trials}회 시뮬레이션 중 (고정 시드 {master_seed})...")
    formal_random_returns: list[float] = []
    reference_random_returns: list[float] = []
    for trial_i in range(n_trials):
        rng = random.Random(f"{master_seed}-{trial_i}")
        random_tickers_by_year = {}
        for y in YEARS:
            universe_pool = universe_year_stats[y]["all_tickers"]
            n = universe_year_stats[y]["wide_count"]
            random_tickers_by_year[y] = mbt.draw_random_portfolio(universe_pool, n, rng)
        periods, _ = build_group_periods(
            random_tickers_by_year, stock_returns_by_year, sectors_by_year, execution_dates, final_mark_date, data.fx_by_date, sector_cap_pct,
        )
        rows = mbt.chain_periods_with_tax(periods, starting_capital_usd, tax_cfg)
        reference_rows = rows[0:5]
        formal_rows = rows[4:8]
        if formal_rows[0]["total_krw"]:
            formal_random_returns.append(formal_rows[-1]["total_krw"] / formal_rows[0]["total_krw"])
        if reference_rows[0]["total_krw"]:
            reference_random_returns.append(reference_rows[-1]["total_krw"] / reference_rows[0]["total_krw"])
    log(f"무작위 포트폴리오 시뮬레이션 완료 ({len(formal_random_returns)}회 유효)")

    wide_formal_return = group_results["넓음"]["formal_total_return_factor"]
    percentile_vs_random = mbt.percentile_rank(wide_formal_return, formal_random_returns)
    log(f"넓음 정식구간 세후 총수익배수 {wide_formal_return:.4f} — 무작위 {len(formal_random_returns)}회 대비 백분위 {percentile_vs_random}")

    # ── 판정 (계획서 4장, 탈락용) ────────────────────────────────────────────
    fails_vs_qqqm = wide_formal_return < qqqm_formal_total_return_factor
    fails_vs_random = percentile_vs_random < 50
    verdict = "탈락" if (fails_vs_qqqm or fails_vs_random) else "탈락 아님"
    log(f"판정: {verdict} (QQQM 대비 {'탈락' if fails_vs_qqqm else '통과'}, 무작위 대비 {'탈락' if fails_vs_random else '통과'})")

    report = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "locked_config_hash": locked_hash,
        "rebalance_dates": {y: str(d.date()) for y, d in rebalance_dates.items()},
        "execution_dates": {y: str(d.date()) for y, d in execution_dates.items()},
        "final_mark_date": str(final_mark_date.date()),
        "universe_year_stats": {
            y: {k: v for k, v in s.items() if k != "all_tickers"} for y, s in universe_year_stats.items()
        },
        "sector_cap_triggered_wide_by_year": sector_cap_triggered_wide,
        "failed_price_tickers": still_missing,
        "tiingo_supplemented_tickers": added_from_alt,
        "price_notes": price_notes,
        "groups": {
            name: {
                "reference_metrics": g["reference_metrics"], "formal_metrics": g["formal_metrics"],
                "formal_total_return_factor": g["formal_total_return_factor"],
                "reference_total_return_factor": g["reference_total_return_factor"],
            }
            for name, g in group_results.items()
        },
        "qqqm_control": {
            "reference_metrics": qqqm_ref_metrics, "formal_metrics": qqqm_formal_metrics,
            "formal_total_return_factor": qqqm_formal_total_return_factor,
            "reference_total_return_factor": qqqm_ref_total_return_factor,
        },
        "random_portfolios": {
            "n_trials": n_trials, "n_valid": len(formal_random_returns), "seed": master_seed,
            "formal_percentile_of_wide": percentile_vs_random,
            "formal_returns_summary": {
                "min": min(formal_random_returns) if formal_random_returns else None,
                "max": max(formal_random_returns) if formal_random_returns else None,
                "median": sorted(formal_random_returns)[len(formal_random_returns) // 2] if formal_random_returns else None,
            },
        },
        "verdict": {"result": verdict, "fails_vs_qqqm": fails_vs_qqqm, "fails_vs_random_percentile": fails_vs_random},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"보고서 저장: {REPORT_PATH}")


if __name__ == "__main__":
    main()
