"""KJB-1 위성 전략 백테스트 엔진 (docs/kjb1_instructions.md 2번).

김종봉 투자법 종목 선택 핵심(core/kjb.py)을 신호로 쓰는 별도 엔진. engine/backtest.py
(B0.5 등 RSI 전략)는 전혀 건드리지 않는다 — 재사용하는 것은 그 파일의 순수 계산
함수(prepare_data, aggregate_positions, compute_position_stats, compute_equity_metrics,
compute_liquidated_cagr, _price_on_or_before)와 engine/portfolio.py의 PortfolioBroker
(대기 자금 = 합성 단기국채, "P6의 쿨다운 없는 방식" — 코어/QLD 필드는 쓰지 않는다)뿐이다.

체결 타이밍(계획서 3장):
- 매수: 신호일 종가×1.01을 다음 거래일 지정가로(core.execution.resolve_buy_fill).
- 손절·63거래일 청산: 둘 다 "confirmed 종가로 감지 -> 다음 거래일 시가 체결"
  (core.execution.exit_at_open) — 같은 날 장중 체결을 가정하지 않는다(RSI 엔진의
  손절과 다른 점, 계획서 3장 그대로).
- 손절이 63거래일 청산보다 우선한다(같은 날 둘 다 해당하면 손절로 기록).

포지션·비중: 위성 자금(현재 평가액 = 현금+대기자금+보유 종목 시가) 총액의 1/10씩,
최대 10종목, 섹터당 최대 3종목. 재진입은 청산 체결일로부터 20거래일 금지.

무작위 진입 비교(계획서 4장): simulate_kjb_satellite가 실제 신호로 admission_log
(날짜별 채택 건수)를 만들고, simulate_random_entry는 그 날짜·건수·청산 규칙을
그대로 두고 "어떤 종목을 살지"만 그날 활성 구성 종목 중 무작위로 고른다 — 같은
day-loop(_simulate)를 candidate_fn만 바꿔 공유한다(단일 변수 비교, 회귀 없음).
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import execution as ex  # noqa: E402
from core import kjb  # noqa: E402
from core import signals as sig  # noqa: E402
from core import synthetic_assets as sa  # noqa: E402
from core import tax  # noqa: E402
from data import market_calendar  # noqa: E402
from data import universe_history as uh  # noqa: E402
from data.fx import fetch_fred_series_range  # noqa: E402
from engine.backtest import (  # noqa: E402
    BacktestData,
    BacktestResult,
    _deposit_krw_to_usd,
    _price_on_or_before,
    _stock_unrealized_gain_krw,
)
from engine.portfolio import PortfolioBroker  # noqa: E402

_UNIT = "1"  # states[ticker]["units"]/["entries"]의 유일한 묶음 키 (compute_liquidated_cagr 호환용)


@dataclass
class TickerSignals:
    close: pd.Series
    open_: pd.Series
    low: pd.Series
    volume: pd.Series
    rel_return: pd.Series
    first_excess: pd.Series
    first_score: pd.Series
    dv_multiplier: pd.Series
    big_candle: pd.Series
    entry_signal: pd.Series


def prepare_reserve_daily_rate(trading_days: list, start: date, end: date) -> pd.Series:
    """단기 국채(DTB3) 일별 단리 수익률 (P6와 같은 계산, engine/portfolio.py 참고)."""
    dtb3_raw = fetch_fred_series_range("DTB3", start, end)
    annual_rate = sa.align_rate_to_trading_days(dtb3_raw, trading_days)
    return sa.annual_yield_pct_to_daily_rate(annual_rate.ffill())


def build_ticker_signals(
    data: BacktestData,
    kjb_cfg: dict,
    big_candle_mode: str = "pct",
    dv_multiplier_min: float | None = None,
) -> dict[str, TickerSignals]:
    """종목별로 core.kjb 신호 열을 전 기간 한 번에 계산해 둔다 (day-loop에서는 조회만).

    big_candle_mode: "pct"(주 설정, >=big_candle_pct_min%) | "volatility_margin"(주변값)
    dv_multiplier_min을 주면 main_settings 값 대신 그 값을 쓴다(주변값 실험용).
    """
    qqq_close = data.qqq_df["close"]
    dv_min = dv_multiplier_min if dv_multiplier_min is not None else kjb_cfg["dollar_volume_multiplier_min"]
    out: dict[str, TickerSignals] = {}
    for ticker, df in data.indicator_map.items():
        close, open_, low, volume = df["close"], df["open"], df["low"], df["volume"]
        qqq_aligned = qqq_close.reindex(close.index).ffill()
        rel_return = kjb.relative_return(close, qqq_aligned, kjb_cfg["relative_return_window_days"])
        daily_rel = kjb.daily_relative_return(close, qqq_aligned)
        first_excess = kjb.first_excess(rel_return, kjb_cfg["relative_return_window_days"])
        first_score = kjb.first_score(daily_rel, kjb_cfg["first_score_window_days"])
        dv_multiplier = kjb.dollar_volume_multiplier(close, volume)
        if big_candle_mode == "volatility_margin":
            big_candle = kjb.big_bull_candle_volatility_margin(open_, close)
        else:
            big_candle = kjb.big_bull_candle(open_, close, kjb_cfg["big_candle_pct_min"])
        entry_signal = kjb.kjb_entry_signal(rel_return, first_excess, dv_multiplier, big_candle, dv_min)
        out[ticker] = TickerSignals(
            close=close, open_=open_, low=low, volume=volume, rel_return=rel_return,
            first_excess=first_excess, first_score=first_score, dv_multiplier=dv_multiplier,
            big_candle=big_candle, entry_signal=entry_signal,
        )
    return out


def _open_positions_value_usd(open_positions: dict, signals: dict[str, TickerSignals], ts: pd.Timestamp) -> float:
    total = 0.0
    for ticker, pos in open_positions.items():
        price = _price_on_or_before(signals[ticker].close.to_frame("close"), ts)
        if price is not None:
            total += pos["qty"] * price
    return total


def _sector_counts(open_positions: dict, sector_by_ticker: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for ticker in open_positions:
        sector = sector_by_ticker.get(ticker)
        if sector:
            counts[sector] += 1
    return counts


def _settle_reserve_tax(broker: PortfolioBroker, year: int, fx_rate: float, cfg: dict, settlement_year: int, date_iso: str) -> float:
    """해에 실현한 손익의 양도세를 settlement_year 5월에 낸다(대기 자금을 팔아 마련).

    engine.portfolio.PortfolioBroker.settle_may_tax는 "코어"를 팔아 세금을 마련하지만,
    이 위성 엔진에는 코어가 없다(대기 자금이 그 역할) — 그래서 직접 구현한다.
    """
    tax_cfg = cfg["backtest"]["tax"]
    gain = broker.realized_gain_by_year.get(year, 0.0)
    amount_krw = (
        tax.capital_gains_tax(gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
        if broker.apply_tax
        else 0.0
    )
    if amount_krw <= 0:
        broker.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": 0.0})
        return 0.0
    spread_pct = cfg["backtest"]["costs"]["fx_spread_pct"] if broker.apply_costs else 0.0
    effective_fx_rate = fx_rate * (1 - spread_pct / 100)
    usd_needed = amount_krw / effective_fx_rate
    broker.move_reserve_to_cash(usd_needed, cfg, date_iso)
    broker.cash_usd -= usd_needed  # 세금은 재투자하지 않고 밖으로 나간다
    broker.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": amount_krw})
    return amount_krw


def compute_liquidated_value_krw(result: BacktestResult, data: BacktestData, cfg: dict, end: date) -> dict:
    """기간 끝에 보유 종목·대기 자금을 전량 정리하고 그해 양도세까지 낸 (a) 시나리오의
    세후 최종 가치(원화)를 계산한다. engine.backtest.compute_liquidated_cagr와 같은
    개념이지만, 이 위성 엔진의 브로커(PortfolioBroker, 대기 자금=reserve_usd)에 맞춰
    다시 구현했다 — engine.backtest.Broker는 qqqm_shares 필드가 있다고 가정해 그대로
    못 쓴다. result.broker는 건드리지 않는다(복사본에만 적용).

    출력: {"liquidated_value_krw", "tax_paid_krw"} 또는 계산할 수 없으면 빈 dict
    """
    ts = pd.Timestamp(end)
    fx_rate = data.fx_by_date.get(end.isoformat())
    if fx_rate is None:
        return {}
    costs = cfg["backtest"]["costs"]
    broker = result.broker

    stock_gain_krw, stock_proceeds_usd = _stock_unrealized_gain_krw(
        result.states, data.indicator_map, ts, fx_rate, costs, broker.apply_costs
    )
    realized = defaultdict(float, broker.realized_gain_by_year)
    realized[end.year] += stock_gain_krw
    total_gain = sum(realized.values())

    tax_cfg = cfg["backtest"]["tax"]
    tax_amount_krw = (
        tax.capital_gains_tax(total_gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
        if broker.apply_tax
        else 0.0
    )
    cash_after_usd = broker.cash_usd + stock_proceeds_usd + broker.reserve_usd
    if tax_amount_krw > 0:
        spread_pct = costs["fx_spread_pct"] if broker.apply_costs else 0.0
        effective_fx_rate = fx_rate * (1 - spread_pct / 100)
        cash_after_usd -= tax_amount_krw / effective_fx_rate

    liquidated_value_krw = cash_after_usd * fx_rate
    return {"liquidated_value_krw": round(liquidated_value_krw), "tax_paid_krw": round(tax_amount_krw)}


def _simulate(
    data: BacktestData,
    signals: dict[str, TickerSignals],
    sector_by_ticker: dict[str, str],
    cfg: dict,
    kjb_cfg: dict,
    start: date,
    end: date,
    candidate_fn,
    apply_costs: bool = True,
    apply_tax: bool = True,
    reserve_daily_rate: pd.Series | None = None,
) -> tuple[BacktestResult, list[dict]]:
    """공용 day-loop. candidate_fn(date_, active, eligible_tickers) -> [ticker, ...]
    (오늘 새로 진입시킬 순서대로 정렬된 후보 목록 — 실제 신호든 무작위든 이 함수가 정한다).

    reserve_daily_rate를 주지 않으면 이 함수 안에서 새로 받는다(FRED 네트워크 호출) —
    무작위 진입 500회처럼 반복 호출할 때는 호출하는 쪽에서 한 번만 받아 넘긴다.
    """
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    costs = bt_cfg["costs"]
    hold_days = kjb_cfg["hold_trading_days"]
    cooldown_days = kjb_cfg["cooldown_trading_days"]
    max_positions = kjb_cfg["max_concurrent_positions"]
    max_per_sector = kjb_cfg["max_positions_per_sector"]
    per_position_fraction = kjb_cfg["per_position_fraction_of_satellite"]

    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end)]
    if reserve_daily_rate is None:
        reserve_daily_rate = prepare_reserve_daily_rate(trading_days, start, end)

    broker = PortfolioBroker(apply_costs=apply_costs, apply_tax=apply_tax)
    open_positions: dict[str, dict] = {}  # ticker -> {id, entry_idx, entry_price, qty, stop_price, exit_idx}
    pending_buys: dict[str, dict] = {}  # ticker -> {limit_price, stop_price, qty}
    pending_sells: dict[str, dict] = {}  # ticker -> {reason}
    cooldown_until_idx: dict[str, int] = {}  # ticker -> 그 종목 df에서의 인덱스 위치
    next_position_id = [1]
    trades: list = []
    rejected: list = []
    equity_rows: list = []
    admission_log: list[dict] = []
    paid_years: set = set()

    fx0 = data.fx_by_date.get(trading_days[0].isoformat()) if trading_days else None
    if fx0:
        usd0 = _deposit_krw_to_usd(total_krw, fx0, costs["fx_spread_pct"], apply_costs)
        broker.cash_usd += usd0
        broker.move_cash_to_reserve(broker.cash_usd, cfg, trading_days[0].isoformat())

    for date_ in trading_days:
        ts = pd.Timestamp(date_)
        date_iso = date_.isoformat()
        fx_rate = data.fx_by_date.get(date_iso)
        members = uh.universe_on(data.checkpoints, date_) if data.checkpoints else frozenset(signals.keys())
        active = [
            t for t in signals
            if ts in signals[t].close.index and (data.universe_mode == "CURRENT_CONSTITUENTS" or t in members)
        ]

        # ── 1) 어제 신호의 매수 체결 ──
        for ticker in list(pending_buys.keys()):
            pend = pending_buys[ticker]
            if pend["signal_date"] >= ts:
                continue
            del pending_buys[ticker]
            if ts not in signals[ticker].close.index:
                continue
            row_open = signals[ticker].open_.get(ts)
            row_low = signals[ticker].low.get(ts)
            fill_price = ex.resolve_buy_fill(pend["limit_price"], row_open, row_low)
            if fill_price is None or pend["qty"] <= 0:
                rejected.append({"date": date_iso, "ticker": ticker, "reason": "미체결(저가가 지정가보다 높음)"})
                continue
            qty = pend["qty"]
            usd_amount = fill_price * qty
            broker.move_reserve_to_cash(usd_amount, cfg, date_iso)
            commission_buy_pct = costs["commission_buy_pct"] if apply_costs else 0.0
            broker.cash_usd -= usd_amount * (1 + commission_buy_pct / 100)
            entry_idx = signals[ticker].close.index.get_loc(ts)
            pid = next_position_id[0]
            next_position_id[0] += 1
            initial_risk_usd = qty * max(fill_price - pend["stop_price"], fill_price * 0.01)
            open_positions[ticker] = {
                "id": pid, "entry_idx": entry_idx, "entry_price": fill_price, "qty": qty,
                "stop_price": pend["stop_price"], "exit_idx": entry_idx + hold_days,
                "initial_risk_usd": initial_risk_usd,
            }
            trades.append({
                "date": date_iso, "ticker": ticker, "position_id": pid, "side": "진입", "stage": "KJB",
                "qty": qty, "price": fill_price, "fx_rate": fx_rate, "reason": "", "pnl_usd": None, "pnl_krw": None, "r": None,
                "initial_risk_krw": initial_risk_usd * (fx_rate or 0),
                "first_score": pend.get("first_score"), "sector": pend.get("sector"),
            })

        # ── 2) 어제 확정된 청산의 오늘 시가 체결 ──
        for ticker in list(pending_sells.keys()):
            pend = pending_sells[ticker]
            if pend["signal_date"] >= ts:
                continue
            del pending_sells[ticker]
            pos = open_positions.pop(ticker, None)
            if pos is None:
                continue
            if ts not in signals[ticker].close.index:
                # 데이터가 끊긴 경우 — 되돌릴 수 없으니 미체결로 남기지 않고 포지션만 제거한다.
                continue
            exit_price = ex.exit_at_open(signals[ticker].open_.get(ts))
            cooldown_until_idx[ticker] = signals[ticker].close.index.get_loc(ts) + cooldown_days
            if exit_price is None:
                continue
            qty = pos["qty"]
            commission_sell_pct = costs["commission_sell_pct"] if apply_costs else 0.0
            proceeds_usd = qty * exit_price * (1 - commission_sell_pct / 100)
            broker.cash_usd += proceeds_usd
            commission_buy_pct = costs["commission_buy_pct"] if apply_costs else 0.0
            cost_usd = pos["entry_price"] * qty * (1 + commission_buy_pct / 100)
            gain_usd = proceeds_usd - cost_usd
            gain_krw = gain_usd * fx_rate if fx_rate else 0.0
            if fx_rate:
                broker.realized_gain_by_year[date_.year] += gain_krw
            trades.append({
                "date": date_iso, "ticker": ticker, "position_id": pos["id"], "side": "청산", "stage": pend["reason"],
                "qty": qty, "price": exit_price, "fx_rate": fx_rate, "reason": pend["reason"],
                "entry_price": pos["entry_price"], "pnl_usd": round(gain_usd, 2),
                "pnl_krw": round(gain_krw), "r": None,
            })
            broker.move_cash_to_reserve(broker.cash_usd, cfg, date_iso)

        # ── 3) 오늘 종가 기준 청산 판정(손절 우선, 그다음 63거래일) -> 내일 시가 체결로 예약 ──
        for ticker, pos in list(open_positions.items()):
            if ticker in pending_sells or ts not in signals[ticker].close.index:
                continue
            close_today = signals[ticker].close.get(ts)
            if pd.isna(close_today):
                continue
            idx_today = signals[ticker].close.index.get_loc(ts)
            if close_today < pos["stop_price"]:
                pending_sells[ticker] = {"signal_date": ts, "reason": "손절"}
            elif idx_today >= pos["exit_idx"]:
                pending_sells[ticker] = {"signal_date": ts, "reason": "63일 청산"}

        # ── 4) 오늘 새 진입 후보 채택(섹터·한도 지키며) ──
        held_and_pending = set(open_positions) | set(pending_buys)
        eligible = []
        for ticker in active:
            if ticker in held_and_pending:
                continue
            idx_today = signals[ticker].close.index.get_loc(ts) if ts in signals[ticker].close.index else None
            if idx_today is not None and ticker in cooldown_until_idx and idx_today <= cooldown_until_idx[ticker]:
                continue
            eligible.append(ticker)

        chosen = candidate_fn(date_, active, eligible)
        if chosen:
            sector_counts = _sector_counts(open_positions, sector_by_ticker)
            for ticker in list(pending_buys):
                sector = sector_by_ticker.get(ticker)
                if sector:
                    sector_counts[sector] += 1
            admitted_today = []
            for ticker in chosen:
                if len(open_positions) + len(pending_buys) >= max_positions:
                    break
                sector = sector_by_ticker.get(ticker)
                if sector and sector_counts.get(sector, 0) >= max_per_sector:
                    continue
                signal_close = signals[ticker].close.get(ts)
                signal_low = signals[ticker].low.get(ts)
                if pd.isna(signal_close) or pd.isna(signal_low):
                    continue
                satellite_value = broker.cash_usd + broker.reserve_usd + _open_positions_value_usd(open_positions, signals, ts)
                target_usd = satellite_value * per_position_fraction
                limit_price = sig.entry_limit_price(signal_close, cfg)
                qty = int(target_usd // limit_price)
                if qty <= 0:
                    rejected.append({"date": date_iso, "ticker": ticker, "reason": "수량 0"})
                    continue
                pending_buys[ticker] = {
                    "signal_date": ts, "limit_price": limit_price, "stop_price": float(signal_low), "qty": qty,
                    "first_score": signals[ticker].first_score.get(ts), "sector": sector or "UNKNOWN",
                }
                admitted_today.append(ticker)
                if sector:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if admitted_today:
                admission_log.append({"date": date_iso, "tickers": admitted_today})

        # ── 5) 배당(보유 종목만, 위성은 QQQM 스윕이 없다) ──
        for ticker, pos in open_positions.items():
            div_series = data.dividends.get(ticker)
            if div_series is None or ts not in div_series.index:
                continue
            gross_usd = float(div_series.loc[ts]) * pos["qty"]
            if gross_usd:
                net_usd = tax.dividend_after_withholding(gross_usd, (bt_cfg["tax"]["dividend_withholding_pct"] / 100) if apply_tax else 0.0)
                broker.cash_usd += net_usd
                broker.dividend_withheld_usd += gross_usd - net_usd

        # ── 6) 대기 자금 성장 + 남는 현금 쓸어담기 ──
        broker.grow_reserve(reserve_daily_rate.get(ts))
        if broker.cash_usd > 0.01:
            broker.move_cash_to_reserve(broker.cash_usd, cfg, date_iso)
        elif broker.cash_usd < -0.01:
            broker.move_reserve_to_cash(-broker.cash_usd, cfg, date_iso)
            broker.cash_usd = 0.0

        # ── 7) 연간 양도세(5월, 전년도분) ──
        if fx_rate is not None and date_.month >= bt_cfg["tax"]["payment_month"]:
            prior_year = date_.year - 1
            if prior_year in broker.realized_gain_by_year and prior_year not in paid_years:
                _settle_reserve_tax(broker, prior_year, fx_rate, cfg, settlement_year=date_.year, date_iso=date_iso)
                paid_years.add(prior_year)

        # ── 8) 자산 기록 ──
        if fx_rate is not None:
            positions_value_usd = _open_positions_value_usd(open_positions, signals, ts)
            equity_rows.append({
                "date": date_iso,
                "qqqm_value_krw": 0,
                "positions_value_krw": round(positions_value_usd * fx_rate),
                "cash_krw": round((broker.cash_usd + broker.reserve_usd) * fx_rate),
                "total_krw": round((broker.cash_usd + broker.reserve_usd + positions_value_usd) * fx_rate),
            })

    still_open_ids = {v["id"] for v in open_positions.values()}
    states = {
        ticker: {"units": {_UNIT: pos["qty"]}, "entries": {_UNIT: pos["entry_price"]}}
        for ticker, pos in open_positions.items()
    }
    result = BacktestResult(
        equity_rows=equity_rows, trades=trades, rejected=rejected, broker=broker,
        still_open_position_ids=still_open_ids, final_date=trading_days[-1] if trading_days else None,
        states=states,
    )
    return result, admission_log


def make_kjb_candidate_fn(signals: dict[str, TickerSignals], sector_by_ticker: dict[str, str], top_n_deprioritized: int = 10):
    """실제 KJB 신호 기반 candidate_fn."""

    def _fn(date_, active: list[str], eligible: list[str]) -> list[str]:
        ts = pd.Timestamp(date_)
        signaled = [t for t in eligible if bool(signals[t].entry_signal.get(ts, False))]
        if not signaled:
            return []
        close_by_ticker = {t: signals[t].close for t in active}
        volume_by_ticker = {t: signals[t].volume for t in active}
        sector_returns = kjb.sector_period_return(close_by_ticker, sector_by_ticker, ts)
        qqq_ret = 0.0  # sector_relative_strength는 상대값이라 QQQ 자체는 순위에 영향 없음(모두 같은 값 차감)
        sector_rel_strength = kjb.sector_relative_strength(sector_returns, qqq_ret)
        dollar_volume_rank = kjb.dollar_volume_rank_today(close_by_ticker, volume_by_ticker, ts)
        candidates = [
            {"ticker": t, "sector": sector_by_ticker.get(t, "UNKNOWN"), "first_score": signals[t].first_score.get(ts, 999) or 999,
             "dollar_volume_multiplier": signals[t].dv_multiplier.get(ts, 0) or 0}
            for t in signaled
        ]
        ranked = kjb.rank_candidates(candidates, sector_rel_strength, dollar_volume_rank, top_n_deprioritized)
        return [c["ticker"] for c in ranked]

    return _fn


def make_random_candidate_fn(admission_log: list[dict], seed: int):
    """무작위 진입 candidate_fn(계획서 4장): 같은 날짜·같은 건수로 그날 활성 종목 중 무작위."""
    rng = random.Random(seed)
    counts_by_date = {row["date"]: len(row["tickers"]) for row in admission_log}

    def _fn(date_, active: list[str], eligible: list[str]) -> list[str]:
        n = counts_by_date.get(date_.isoformat(), 0)
        if n <= 0 or not eligible:
            return []
        pool = list(eligible)
        rng.shuffle(pool)
        return pool[:n]

    return _fn


def simulate_kjb_satellite(
    data: BacktestData, sector_by_ticker: dict[str, str], cfg: dict, kjb_cfg: dict, start: date, end: date,
    big_candle_mode: str = "pct", dv_multiplier_min: float | None = None,
    apply_costs: bool = True, apply_tax: bool = True,
    reserve_daily_rate: pd.Series | None = None,
) -> tuple[BacktestResult, list[dict]]:
    """KJB-1 위성 전략을 [start, end]에서 시뮬레이션한다."""
    signals = build_ticker_signals(data, kjb_cfg, big_candle_mode=big_candle_mode, dv_multiplier_min=dv_multiplier_min)
    candidate_fn = make_kjb_candidate_fn(signals, sector_by_ticker)
    return _simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, candidate_fn, apply_costs, apply_tax, reserve_daily_rate)


def build_price_only_signals(data: BacktestData) -> dict[str, TickerSignals]:
    """무작위 진입용 가벼운 signals — 가격 열만 있으면 되고(체결·평가·청산에만 쓰임),
    실제 김종봉 신호 열(entry_signal 등)은 무작위 모드에서 안 쓰이므로 계산하지 않는다
    (500회 반복에서 core.kjb의 롤링 계산을 매번 다시 하지 않기 위한 성능 최적화)."""
    out: dict[str, TickerSignals] = {}
    for ticker, df in data.indicator_map.items():
        idx = df.index
        placeholder = pd.Series(False, index=idx)
        placeholder_num = pd.Series(0.0, index=idx)
        out[ticker] = TickerSignals(
            close=df["close"], open_=df["open"], low=df["low"], volume=df["volume"],
            rel_return=placeholder_num, first_excess=placeholder, first_score=placeholder_num,
            dv_multiplier=placeholder_num, big_candle=placeholder, entry_signal=placeholder,
        )
    return out


def simulate_random_entry(
    data: BacktestData, sector_by_ticker: dict[str, str], cfg: dict, kjb_cfg: dict, start: date, end: date,
    admission_log: list[dict], seed: int, apply_costs: bool = True, apply_tax: bool = True,
    signals: dict[str, TickerSignals] | None = None,
    reserve_daily_rate: pd.Series | None = None,
) -> BacktestResult:
    """무작위 진입 1회(계획서 4장). 실제 신호의 admission_log(날짜·건수)를 그대로 쓴다.

    signals·reserve_daily_rate를 주지 않으면 이 함수 안에서 새로 만든다 — 무작위
    진입 500회처럼 반복 호출할 때는 호출하는 쪽에서 한 번만 만들어 넘긴다(성능).
    """
    signals = signals if signals is not None else build_price_only_signals(data)
    candidate_fn = make_random_candidate_fn(admission_log, seed)
    result, _ = _simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, candidate_fn, apply_costs, apply_tax, reserve_daily_rate)
    return result
