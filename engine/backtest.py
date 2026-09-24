"""과거 재생(백테스트) 엔진 (P5-1).

라이브(engine/daily.py)와 같은 core 함수(core.indicators, core.signals, core.filters,
core.sizing, core.state)를 쓴다. 신호 판정·상태 전이 로직은 새로 만들지 않는다 —
이 파일은 과거 시세를 거래일 하루씩 재생하며 core.state.process_day를 그대로 부르고,
거기에 체결 모형(core.execution)·비용·세금(core.tax)·QQQM 스윕을 더해 포트폴리오를
시뮬레이션하고 결과를 기록하는 오케스트레이션만 한다.

체결 타이밍(중요):
- 매수(A1·A2·A3·B): 신호일 종가로 지정가를 정하고(core.sizing로 그날 수량까지
  정해 pending에 저장), core.state의 "주문대기" 메커니즘 그대로 다음 거래일에
  resolve한다 — 그날 시가·저가로 core.execution.resolve_buy_fill 체결.
- 손절: core.state가 종가 기준으로 그날 감지하지만, 실제로는 증권사에 걸어 둔
  손절 예약이 장중에 트리거된 것으로 본다 — 그래서 같은 날(신호일 당일)의
  시가·저가로 core.execution.resolve_stop_fill 체결.
- E1·E2·E3·1차 만료: 신호 다음 거래일 시가에 체결(core.execution.exit_at_open).
  core.state는 신호 당일 즉시 수량을 확정하므로, 다음 날 시가를 미리 조회해
  그 자리에서 체결가만 정한다(별도의 대기 상태를 새로 만들지 않는다).

실행:
    python -m engine.backtest              config.yaml의 backtest.train_start~train_end(B0)
    python -m engine.backtest --benchmark-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import execution as ex  # noqa: E402
from core import filters  # noqa: E402
from core import sizing  # noqa: E402
from core import signals as sig  # noqa: E402
from core import state as st  # noqa: E402
from core import tax  # noqa: E402
from core.indicators import compute_indicators  # noqa: E402
from data import backtest_prices as bp  # noqa: E402
from data import fx  # noqa: E402
from data import market_calendar  # noqa: E402
from data import universe_history as uh  # noqa: E402
from data.universe import get_universe  # noqa: E402
from engine.daily import _BUY_KINDS, _NEW_POSITION_KINDS, _SELL_KINDS, _STAGE_LABEL, _label_filter_reason  # noqa: E402

OUTPUT_ROOT = ROOT / "outputs" / "backtest"

_SELL_REASON = {
    "STOP": "손절",
    "A1_EXPIRE": "A1 만료",
    "E3": "구조 붕괴(E3)",
    "E1": "모멘텀 약화(E1)",
    "E2": "추세 약화(E2)",
}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


# ── 자금 계좌 (KRW 현금 + QQQM 평균원가법) ──────────────────────────────────


@dataclass
class Broker:
    """KRW 현금 계좌 + QQQM(쉬는 돈) 보유분. 신호 종목 매매는 전부 이 계좌를 거쳐 간다.

    한국 해외주식 계좌 관행을 그대로 모형화한다: 미국 자산을 사고팔 때마다 원화
    현금과 환전한다(스프레드), 그래서 "QQQM을 팔아 종목을 산다"도 실제로는
    QQQM 매도(달러->원화) + 종목 매수(원화->달러) 두 번의 환전이다.
    """

    cash_krw: float
    qqqm_shares: float = 0.0
    qqqm_cost_usd: float = 0.0  # 평균원가법 총 매입원가(달러)
    realized_gain_by_year: dict = field(default_factory=lambda: defaultdict(float))
    dividend_tax_paid_krw: float = 0.0
    tax_log: list = field(default_factory=list)

    def buy_usd(self, usd_amount: float, commission_pct: float, fx_rate: float, spread_pct: float) -> float:
        """원화 현금으로 usd_amount달러어치 미국 자산을 산다. 실제로 나간 원화를 반환."""
        commission = usd_amount * commission_pct / 100
        krw_cost = (usd_amount + commission) * fx_rate / (1 - spread_pct / 100)
        self.cash_krw -= krw_cost
        return krw_cost

    def sell_usd(self, usd_amount: float, commission_pct: float, fx_rate: float, spread_pct: float) -> float:
        """미국 자산 usd_amount달러어치(수수료 전)를 팔아 원화로 바꿔 현금에 더한다. 받은 원화를 반환."""
        commission = usd_amount * commission_pct / 100
        krw_proceeds = (usd_amount - commission) * fx_rate * (1 - spread_pct / 100)
        self.cash_krw += krw_proceeds
        return krw_proceeds

    def buy_qqqm(self, usd_amount: float, price_usd: float, fx_rate: float, cfg: dict) -> float:
        if usd_amount <= 0 or price_usd <= 0:
            return 0.0
        costs = cfg["backtest"]["costs"]
        self.buy_usd(usd_amount, costs["commission_buy_pct"], fx_rate, costs["fx_spread_pct"])
        shares = usd_amount / price_usd
        self.qqqm_shares += shares
        self.qqqm_cost_usd += usd_amount
        return shares

    def sell_qqqm(self, usd_amount: float, price_usd: float, fx_rate: float, cfg: dict, year: int) -> float:
        if usd_amount <= 0 or self.qqqm_shares <= 0:
            return 0.0
        usd_amount = min(usd_amount, self.qqqm_shares * price_usd)
        costs = cfg["backtest"]["costs"]
        shares = usd_amount / price_usd
        avg_cost = self.qqqm_cost_usd / self.qqqm_shares
        cost_removed = avg_cost * shares
        gain_usd = usd_amount - cost_removed
        self.sell_usd(usd_amount, costs["commission_sell_pct"], fx_rate, costs["fx_spread_pct"])
        self.qqqm_shares -= shares
        self.qqqm_cost_usd -= cost_removed
        self.realized_gain_by_year[year] += gain_usd * fx_rate
        return shares

    def pay_capital_gains_tax(self, year: int, price_usd: float, fx_rate: float, cfg: dict, settlement_year: int) -> float:
        """year의 실현손익에 대한 양도세를 settlement_year(보통 year+1) 5월에 낸다 (QQQM을 팔아 마련).

        세금을 마련하려고 파는 QQQM 자체의 실현손익은 그 해(year)가 아니라 실제로
        파는 시점(settlement_year)에 잡아야 한다 — 여기서 그 둘을 구분한다.
        """
        tax_cfg = cfg["backtest"]["tax"]
        gain = self.realized_gain_by_year.get(year, 0.0)
        amount = tax.capital_gains_tax(gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
        if amount <= 0:
            self.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": 0.0})
            return 0.0
        usd_needed = amount / fx_rate
        self.sell_qqqm(usd_needed, price_usd, fx_rate, cfg, year=settlement_year)
        self.cash_krw -= amount  # 세금은 재투자하지 않고 밖으로 나간다
        self.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": amount})
        return amount

    def receive_dividend(self, gross_usd: float, fx_rate: float, cfg: dict) -> float:
        """배당(달러, 세전)을 15% 원천징수 후 원화로 환전해 현금에 더한다."""
        div_cfg = cfg["backtest"]["tax"]
        net_usd = tax.dividend_after_withholding(gross_usd, div_cfg["dividend_withholding_pct"] / 100)
        withheld_usd = gross_usd - net_usd
        krw = net_usd * fx_rate * (1 - cfg["backtest"]["costs"]["fx_spread_pct"] / 100)
        self.cash_krw += krw
        self.dividend_tax_paid_krw += withheld_usd * fx_rate
        return krw


# ── 데이터 준비 ─────────────────────────────────────────────────────────────


@dataclass
class BacktestData:
    indicator_map: dict
    dividends: dict
    checkpoints: list
    fx_by_date: dict
    universe_mode: str  # POINT_IN_TIME | CURRENT_CONSTITUENTS
    survivorship_bias: bool
    failed_tickers: dict
    data_gap: dict
    qqq_df: pd.DataFrame
    qqq_dividends: pd.Series
    cash_etf_df: pd.DataFrame  # QQQM(상장 후) / QQQ(상장 전 대체)
    cash_etf_dividends: pd.Series


def prepare_data(cfg: dict, warmup_start: date, end: date) -> BacktestData:
    """종목·QQQ·QQQM·환율·배당·시점별 구성 종목을 모두 받아 지표까지 계산해 둔다."""
    bt_cfg = cfg["backtest"]

    print("[backtest] 나스닥 100 현재 구성 종목을 가져오는 중...")
    current_tickers = set(get_universe()["ticker"])

    survivorship_bias = False
    universe_mode = "POINT_IN_TIME"
    checkpoints: list = []
    try:
        changes = uh.get_changes()
        checkpoints = uh.membership_checkpoints(current_tickers, changes, warmup_start)
        all_needed = set(current_tickers)
        for _, members in checkpoints:
            all_needed |= set(members)
        print(f"[backtest] 시점별 구성 종목 재구성 성공 — 기간 중 등장한 종목 {len(all_needed)}개")
    except Exception as exc:
        print(f"[backtest] 시점별 구성 종목 재구성 실패({exc}) — 현재 구성 종목만 쓴다 (SURVIVORSHIP_BIAS)")
        universe_mode = "CURRENT_CONSTITUENTS"
        survivorship_bias = True
        all_needed = set(current_tickers)
        checkpoints = [(warmup_start, frozenset(current_tickers))]

    print(f"[backtest] 종목 {len(all_needed)}개의 과거 일봉을 받는 중... (시간이 걸립니다)")
    price_result = bp.fetch_universe_history(sorted(all_needed), warmup_start, end)
    print(f"[backtest]   성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")

    indicator_map = {t: compute_indicators(df, cfg) for t, df in price_result.prices.items()}

    dividends: dict = {}
    for t in price_result.prices:
        try:
            dividends[t] = bp.fetch_dividends(t, warmup_start, end)
        except Exception:
            dividends[t] = pd.Series(dtype=float)

    print("[backtest] QQQ(벤치마크) 시세를 받는 중...")
    qqq_df_raw, _ = bp.fetch_history(bt_cfg["benchmark_ticker"], warmup_start, end)
    qqq_df = compute_indicators(qqq_df_raw, cfg)
    qqq_dividends = bp.fetch_dividends(bt_cfg["benchmark_ticker"], warmup_start, end)

    inception = _parse_date(bt_cfg["cash_etf_inception"])
    cash_ticker = bt_cfg["cash_etf_ticker"] if end >= inception else bt_cfg["benchmark_ticker"]
    print(f"[backtest] {cash_ticker}(쉬는 돈) 시세를 받는 중...")
    if cash_ticker == bt_cfg["benchmark_ticker"]:
        cash_etf_df, cash_etf_dividends = qqq_df_raw, qqq_dividends
    else:
        cash_etf_df, _ = bp.fetch_history(cash_ticker, warmup_start, end)
        cash_etf_dividends = bp.fetch_dividends(cash_ticker, warmup_start, end)
    # QQQM 상장 전 구간은 QQQ로 대체한다 (지시문 2번) — 두 ETF의 보수 차이를 반영해
    # QQQM 상장 전 동안은 (QQQ 보수 - QQQM 보수)만큼 매일 수익률에서 깎는다.
    #
    # 실제 QQQM 상장일은 config의 cash_etf_inception(2020-12-22, 공식 상장일)보다
    # 이르다 — yfinance가 그 전(2020-10-13~) 데이터도 내려준다. 접합 경계를 config
    # 값으로 고정하면 그 사이 구간(10/13~12/21)에서 합성 구간(pre)과 실제 QQQM
    # 데이터의 날짜가 겹치고, pd.concat().sort_index()의 기본 정렬(quicksort)은
    # 안정 정렬이 아니라 같은 날짜에서 어느 쪽이 남을지 날짜마다 들쭉날쭉해진다
    # (가격이 하루씩 오락가락하는 회귀를 낳았다). 실제 데이터가 존재하는 첫 날짜를
    # 그대로 접합 경계로 써서 겹치는 구간 자체를 없앤다.
    if cash_ticker != bt_cfg["benchmark_ticker"] and not cash_etf_df.empty and warmup_start < cash_etf_df.index[0].date():
        real_start = cash_etf_df.index[0]
        real_start_close = float(cash_etf_df.loc[real_start, "close"])
        pre = qqq_df_raw.loc[qqq_df_raw.index < real_start].copy()
        # QQQM은 QQQ의 보수만 다른 게 아니라 아예 다른 주당 가격대(별도 주식군)라, 그냥
        # QQQ 가격에 보수 차이만 곱하면 접합 경계에서 가격 수준이 어긋난다(예: QQQ
        # ~290 -> QQQM ~120로 하루 만에 반토막 나는 것처럼 보임). QQQ의 "수익률"만
        # 대용으로 쓰고, 접합일 QQQM 실제 종가에 맞춰 수준을 다시 고정한다.
        expense_drag_daily = (bt_cfg.get("qqq_expense_ratio_pct", 0.20) - bt_cfg.get("qqqm_expense_ratio_pct", 0.15)) / 100 / 252
        drag_factor = (1 - expense_drag_daily) ** pd.Series(range(len(pre)), index=pre.index)[::-1]
        qqq_close_at_boundary = float(qqq_df_raw.loc[qqq_df_raw.index < real_start, "close"].iloc[-1])
        level_scale = real_start_close / (qqq_close_at_boundary * drag_factor.iloc[-1])
        for col in ("open", "high", "low", "close"):
            pre[col] = pre[col] * drag_factor.values * level_scale
        cash_etf_df = pd.concat([pre, cash_etf_df])
        assert cash_etf_df.index.is_monotonic_increasing and not cash_etf_df.index.duplicated().any()

    print("[backtest] 원/달러 환율을 받는 중...")
    fx_history = fx.fetch_usd_krw_range(warmup_start, end)
    trading_days = market_calendar.trading_days_between(warmup_start, end)
    fx_by_date: dict[str, float] = {}
    last = None
    for d in trading_days:
        key = d.date().isoformat()
        if key in fx_history:
            last = fx_history[key]
        fx_by_date[key] = last

    return BacktestData(
        indicator_map=indicator_map,
        dividends=dividends,
        checkpoints=checkpoints,
        fx_by_date=fx_by_date,
        universe_mode=universe_mode,
        survivorship_bias=survivorship_bias,
        failed_tickers=price_result.failed,
        data_gap=price_result.data_gap,
        qqq_df=qqq_df,
        qqq_dividends=qqq_dividends,
        cash_etf_df=compute_indicators(cash_etf_df, cfg) if not cash_etf_df.empty else cash_etf_df,
        cash_etf_dividends=cash_etf_dividends,
    )


# ── 포트폴리오 시뮬레이션 ────────────────────────────────────────────────────


def _held_snapshot(states: dict, indicator_map: dict, date_) -> list[dict]:
    out = []
    for ticker, state_ in states.items():
        qty_held = sum(q for q in state_["units"].values() if q > 0)
        if qty_held <= 0:
            continue
        df = indicator_map.get(ticker)
        close = None
        if df is not None and date_ in df.index:
            c = df.loc[date_, "close"]
            close = float(c) if not pd.isna(c) else None
        label = state_["state"]
        if label == "주문대기" and state_.get("pending"):
            label = state_["pending"].get("prev_state", label)
        out.append({"ticker": ticker, "qty": qty_held, "close": close, "state_label": label})
    return out


@dataclass
class BacktestResult:
    equity_rows: list
    trades: list
    rejected: list
    broker: Broker
    final_date: date | None


def simulate_portfolio(data: BacktestData, cfg: dict, start: date, end: date) -> BacktestResult:
    """core.state.process_day를 하루씩 재생해 포트폴리오를 시뮬레이션한다."""
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    cash_buffer_krw = total_krw * cfg["plan"]["cash_buffer_pct"] / 100
    max_slots = cfg["plan"]["max_slots"]
    costs = bt_cfg["costs"]

    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end)]
    indicator_map = data.indicator_map
    states = {t: st.init_state(t) for t in indicator_map}
    broker = Broker(cash_krw=total_krw)

    equity_rows: list = []
    trades: list = []
    rejected: list = []
    paid_years: set = set()

    # 첫 거래일에 버퍼를 뺀 전액을 QQQM에 넣는다.
    first_ts = pd.Timestamp(trading_days[0])
    cash_etf_price0 = _price_on_or_before(data.cash_etf_df, first_ts)
    fx0 = data.fx_by_date.get(trading_days[0].isoformat())
    if cash_etf_price0 and fx0:
        usd0 = (total_krw - cash_buffer_krw) / fx0
        broker.buy_qqqm(usd0, cash_etf_price0, fx0, cfg)

    for date_ in trading_days:
        ts = pd.Timestamp(date_)
        fx_rate = data.fx_by_date.get(date_.isoformat())
        members = uh.universe_on(data.checkpoints, date_) if data.checkpoints else frozenset(indicator_map.keys())

        active = [
            t for t in indicator_map
            if ts in indicator_map[t].index and (data.universe_mode == "CURRENT_CONSTITUENTS" or t in members)
        ]

        # ── 매수 체결 해소(전날 신호 -> 오늘 시가/저가) — process_day 전에 반영 ──
        for ticker in active:
            state_ = states[ticker]
            if state_["state"] != "주문대기":
                continue
            pending = state_.get("pending")
            if not pending or pd.Timestamp(pending["date"]) >= ts:
                continue
            row = indicator_map[ticker].loc[ts]
            limit_price = sig.entry_limit_price(pending["price"], cfg)
            fill_price = ex.resolve_buy_fill(limit_price, row.get("open"), row.get("low"))
            qty = pending.get("sized_qty", 0)
            if fill_price is not None and qty > 0:
                usd_amount = fill_price * qty
                broker.sell_qqqm(usd_amount, _price_on_or_before(data.cash_etf_df, ts) or fill_price, fx_rate, cfg, date_.year)
                broker.buy_usd(usd_amount, costs["commission_buy_pct"], fx_rate, costs["fx_spread_pct"])
                states[ticker] = st.apply_fill(state_, {"unit": pending["unit"], "side": "buy", "price": fill_price, "qty": qty}, cfg)
                trades.append(
                    {"date": date_.isoformat(), "ticker": ticker, "side": "진입", "stage": pending["kind"],
                     "qty": qty, "price": fill_price, "fx_rate": fx_rate, "reason": "", "pnl_usd": None, "r": None}
                )
            elif qty > 0:
                rejected.append({"date": date_.isoformat(), "ticker": ticker, "stage": pending["kind"], "reason": "미체결(저가가 지정가보다 높음)"})

        # ── Pass 1: 오늘 새 진입(A1·B) 후보를 점수 순으로 추린다 ──
        candidates = []
        skip_today = set()
        for ticker in active:
            if date_.isoformat() in data.data_gap.get(ticker, []):
                skip_today.add(ticker)
                rejected.append({"date": date_.isoformat(), "ticker": ticker, "stage": "", "reason": "data_gap"})
                continue
            cand = st.preview_new_entry(indicator_map[ticker], ts, states[ticker], cfg, None)
            if cand:
                candidates.append({**cand, "ticker": ticker})
        candidates.sort(key=lambda c: c["score"], reverse=True)
        held_count = sum(1 for s in states.values() if any(q > 0 for q in s["units"].values()))
        slots = max(max_slots - held_count, 0)
        admitted = {c["ticker"] for c in candidates[:slots]}

        # ── Pass 2: process_day 실행, 매도 체결·매수 신호 사이징 ──
        today_buy_events: dict[str, list] = {}
        for ticker in active:
            if ticker in skip_today:
                continue
            df = indicator_map[ticker]
            events, states[ticker] = st.process_day(df, ts, states[ticker], cfg, earnings_date=None, new_entry_allowed=(ticker in admitted))
            for event in events:
                event["ticker"] = ticker
                if event["kind"] == "BLOCKED":
                    rejected.append(
                        {"date": date_.isoformat(), "ticker": ticker, "stage": event["stage"],
                         "reason": ", ".join(_label_filter_reason(r) for r in event["reasons"])}
                    )
                elif event["kind"] in _SELL_KINDS:
                    _settle_sell(event, ticker, date_, ts, indicator_map[ticker], broker, cfg, fx_rate, trades)
                elif event["kind"] in _BUY_KINDS:
                    today_buy_events.setdefault(ticker, []).append(event)

        if today_buy_events:
            _size_and_stash(today_buy_events, states, indicator_map, ts, cfg, fx_rate, rejected, date_)

        # ── 배당 ──
        for ticker, state_ in states.items():
            qty_held = sum(q for q in state_["units"].values() if q > 0)
            if qty_held <= 0 or fx_rate is None:
                continue
            div_series = data.dividends.get(ticker)
            if div_series is None or ts not in div_series.index:
                continue
            gross_usd = float(div_series.loc[ts]) * qty_held
            broker.receive_dividend(gross_usd, fx_rate, cfg)
        if fx_rate is not None and broker.qqqm_shares > 0 and ts in data.cash_etf_dividends.index:
            gross_usd = float(data.cash_etf_dividends.loc[ts]) * broker.qqqm_shares
            broker.receive_dividend(gross_usd, fx_rate, cfg)

        # ── 연간 양도세 (5월, 전년도분) ──
        if fx_rate is not None and date_.month >= bt_cfg["tax"]["payment_month"]:
            prior_year = date_.year - 1
            if prior_year in broker.realized_gain_by_year and prior_year not in paid_years:
                price_now = _price_on_or_before(data.cash_etf_df, ts)
                if price_now:
                    broker.pay_capital_gains_tax(prior_year, price_now, fx_rate, cfg, settlement_year=date_.year)
                    paid_years.add(prior_year)

        # ── 하루 끝: 남는 현금(버퍼 제외)을 QQQM으로 쓸어 담는다 ──
        if fx_rate is not None:
            price_now = _price_on_or_before(data.cash_etf_df, ts)
            if price_now:
                surplus_krw = broker.cash_krw - cash_buffer_krw
                if surplus_krw > 1000:
                    broker.buy_qqqm(surplus_krw / fx_rate, price_now, fx_rate, cfg)
                elif surplus_krw < -1000 and broker.qqqm_shares > 0:
                    broker.sell_qqqm(-surplus_krw / fx_rate, price_now, fx_rate, cfg, date_.year)

        # ── 자산 기록 ──
        if fx_rate is not None:
            price_now = _price_on_or_before(data.cash_etf_df, ts)
            qqqm_value_krw = broker.qqqm_shares * (price_now or 0) * fx_rate
            positions_value_krw = 0.0
            for ticker, state_ in states.items():
                qty_held = sum(q for q in state_["units"].values() if q > 0)
                if qty_held <= 0:
                    continue
                df = indicator_map[ticker]
                if ts in df.index:
                    close = df.loc[ts, "close"]
                    if not pd.isna(close):
                        positions_value_krw += qty_held * float(close) * fx_rate
            equity_rows.append(
                {
                    "date": date_.isoformat(),
                    "qqqm_value_krw": round(qqqm_value_krw),
                    "positions_value_krw": round(positions_value_krw),
                    "cash_krw": round(broker.cash_krw),
                    "total_krw": round(qqqm_value_krw + positions_value_krw + broker.cash_krw),
                }
            )

    return BacktestResult(equity_rows=equity_rows, trades=trades, rejected=rejected, broker=broker, final_date=trading_days[-1] if trading_days else None)


def _price_on_or_before(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None
    sub = df.loc[:ts]
    if sub.empty:
        return None
    close = sub.iloc[-1]["close"]
    return None if pd.isna(close) else float(close)


def _settle_sell(event, ticker, date_, ts, df, broker: Broker, cfg, fx_rate, trades: list) -> None:
    """매도 이벤트의 체결가를 정해(손절=당일, 나머지=다음날 시가) 현금에 반영하고 trades에 남긴다."""
    kind = event["kind"]
    qty = event["qty"]
    entry_price = event.get("entry_price")
    if kind == "STOP":
        row = df.loc[ts]
        exit_price = ex.resolve_stop_fill(event.get("stop_price"), row.get("open"), row.get("low"))
    else:
        idx = df.index.get_loc(ts)
        if idx + 1 >= len(df):
            exit_price = None
        else:
            next_row = df.iloc[idx + 1]
            exit_price = ex.exit_at_open(next_row.get("open"))
    if exit_price is None or entry_price is None or fx_rate is None:
        trades.append(
            {"date": date_.isoformat(), "ticker": ticker, "side": "청산", "stage": kind, "qty": qty,
             "price": None, "fx_rate": fx_rate, "reason": _SELL_REASON.get(kind, kind), "pnl_usd": None, "r": None}
        )
        return

    usd_amount = exit_price * qty
    krw_proceeds = broker.sell_usd(usd_amount, cfg["backtest"]["costs"]["commission_sell_pct"], fx_rate, cfg["backtest"]["costs"]["fx_spread_pct"])
    cost_krw = entry_price * qty * fx_rate
    gain_krw = krw_proceeds - cost_krw
    broker.realized_gain_by_year[date_.year] += gain_krw
    pnl_usd = (exit_price - entry_price) * qty
    # R 배수의 분모(주당 위험): STOP·E3(core.state._liquidate_all)만 이벤트에 stop_price를
    # 남긴다. E1·E2·A1 만료는 분할 매도라 이벤트에 손절가가 없어, 최소 위험 기준
    # (entry_price의 min_risk_per_share_pct, 보통 1%)으로 근사한다.
    stop_ref = event.get("stop_price")
    risk_per_share = sizing.per_share_risk(entry_price, stop_ref, cfg) if stop_ref is not None else entry_price * cfg["risk"]["min_risk_per_share_pct"] / 100
    r_multiple = pnl_usd / (risk_per_share * qty) if risk_per_share and qty else None
    trades.append(
        {
            "date": date_.isoformat(), "ticker": ticker, "side": "청산", "stage": kind, "qty": qty,
            "price": exit_price, "fx_rate": fx_rate, "reason": _SELL_REASON.get(kind, kind),
            "entry_price": entry_price, "pnl_usd": round(pnl_usd, 2), "pnl_krw": round(gain_krw), "r": round(r_multiple, 2) if r_multiple is not None else None,
        }
    )


def _size_and_stash(today_buy_events, states, indicator_map, ts, cfg, fx_rate, rejected, date_) -> None:
    """오늘 매수 신호들에 core.sizing.size_buy_signals로 수량을 매겨 pending에 저장한다."""
    held = _held_snapshot(states, indicator_map, ts)
    signals = []
    lookup = {}
    for ticker, events in today_buy_events.items():
        for event in events:
            entry_price = sig.entry_limit_price(event["price"], cfg)
            stop_price = states[ticker].get("stop")
            key = f"{ticker}-{event['kind']}"
            signals.append(
                {"key": key, "stage": event["kind"], "entry_price": entry_price, "stop_price": stop_price,
                 "score": event.get("score", 0), "is_new_position": event["kind"] in _NEW_POSITION_KINDS}
            )
            lookup[key] = ticker
    sized = sizing.size_buy_signals(signals, held, cfg, fx_rate)
    for key, ticker in lookup.items():
        row = sized["rows"].get(key, {"qty": 0})
        pending = states[ticker].get("pending")
        if pending is not None:
            pending["sized_qty"] = row["qty"]
        if row["qty"] <= 0:
            stage = pending["kind"] if pending else ""
            reason = "손절가 계산 불가" if states[ticker].get("stop") is None else "남은 한도 부족"
            rejected.append({"date": date_.isoformat(), "ticker": ticker, "stage": stage, "reason": reason})


# ── 벤치마크(QQQ 매수 후 보유) ────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    equity_rows: list
    final_value_usd: float
    final_shares: float
    cost_basis_usd: float
    realized_gain_by_year: dict
    cagr_pretax: float
    cagr_liquidated: float  # (a) 기간 끝에 전량 매도, 세금까지 낸 경우
    cagr_unrealized: float  # (b) 팔지 않은 경우(미실현)
    mdd_pct: float


def simulate_benchmark(data: BacktestData, cfg: dict, start: date, end: date) -> BenchmarkResult:
    """QQQ를 첫날 전액 매수해 배당 재투자하며 보유한다. 세후는 (a) 청산 (b) 미청산 둘 다 계산."""
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    costs = bt_cfg["costs"]
    tax_cfg = bt_cfg["tax"]
    df = data.qqq_df
    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end) if pd.Timestamp(d) in df.index]
    if not trading_days:
        raise ValueError("QQQ 시세에 백테스트 구간 데이터가 없습니다")

    broker = Broker(cash_krw=total_krw)
    first_ts = pd.Timestamp(trading_days[0])
    fx0 = data.fx_by_date.get(trading_days[0].isoformat())
    price0 = float(df.loc[first_ts, "close"])
    usd0 = total_krw / fx0
    broker.buy_qqqm(usd0, price0, fx0, cfg)  # buy_qqqm은 일반적인 "미국 자산 사기"로 그대로 재사용한다

    equity_rows = []
    peak = None
    mdd = 0.0
    for date_ in trading_days:
        ts = pd.Timestamp(date_)
        fx_rate = data.fx_by_date.get(date_.isoformat())
        if fx_rate is None:
            continue
        price = float(df.loc[ts, "close"])
        if ts in data.qqq_dividends.index and broker.qqqm_shares > 0:
            gross_usd = float(data.qqq_dividends.loc[ts]) * broker.qqqm_shares
            broker.receive_dividend(gross_usd, fx_rate, cfg)
            surplus_krw = broker.cash_krw
            if surplus_krw > 0:
                broker.buy_qqqm(surplus_krw / fx_rate, price, fx_rate, cfg)
        value_krw = broker.qqqm_shares * price * fx_rate + broker.cash_krw
        equity_rows.append({"date": date_.isoformat(), "total_krw": round(value_krw)})
        peak = value_krw if peak is None else max(peak, value_krw)
        if peak:
            mdd = min(mdd, value_krw / peak - 1)

    last_ts = pd.Timestamp(trading_days[-1])
    last_price = float(df.loc[last_ts, "close"])
    last_fx = data.fx_by_date.get(trading_days[-1].isoformat()) or fx0
    unrealized_value_krw = broker.qqqm_shares * last_price * last_fx + broker.cash_krw

    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr_pretax = (unrealized_value_krw / total_krw) ** (1 / years) - 1 if years > 0 else 0.0
    cagr_unrealized = cagr_pretax  # 세전=세후(미실현, 아직 세금 낼 일이 없음)

    # (a) 청산: 전량 매도 후 그동안 실현 안 된 손익까지 세금 반영
    liq_broker = Broker(cash_krw=0.0, qqqm_shares=broker.qqqm_shares, qqqm_cost_usd=broker.qqqm_cost_usd,
                         realized_gain_by_year=defaultdict(float, broker.realized_gain_by_year))
    liq_broker.sell_qqqm(broker.qqqm_shares * last_price, last_price, last_fx, cfg, trading_days[-1].year)
    total_gain = sum(liq_broker.realized_gain_by_year.values())
    tax_amount = tax.capital_gains_tax(total_gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
    liquidated_value_krw = unrealized_value_krw - broker.cash_krw + liq_broker.cash_krw - tax_amount
    cagr_liquidated = (liquidated_value_krw / total_krw) ** (1 / years) - 1 if years > 0 and liquidated_value_krw > 0 else -1.0

    return BenchmarkResult(
        equity_rows=equity_rows, final_value_usd=broker.qqqm_shares * last_price, final_shares=broker.qqqm_shares,
        cost_basis_usd=broker.qqqm_cost_usd, realized_gain_by_year=broker.realized_gain_by_year,
        cagr_pretax=cagr_pretax, cagr_liquidated=cagr_liquidated, cagr_unrealized=cagr_unrealized, mdd_pct=mdd * 100,
    )


# ── 지표 계산 ────────────────────────────────────────────────────────────────


def compute_metrics(equity_rows: list, trades: list, start: date, end: date) -> dict:
    if not equity_rows:
        return {}
    values = [r["total_krw"] for r in equity_rows]
    dates = [pd.Timestamp(r["date"]) for r in equity_rows]
    start_v, end_v = values[0], values[-1]
    years = (dates[-1] - dates[0]).days / 365.25

    cagr_pretax = (end_v / start_v) ** (1 / years) - 1 if years > 0 and start_v > 0 and end_v > 0 else None

    peak = values[0]
    mdd = 0.0
    mdd_start = mdd_end = None
    peak_date = dates[0]
    cur_dd_start = None
    for v, d in zip(values, dates):
        if v > peak:
            peak = v
            cur_dd_start = None
        else:
            if cur_dd_start is None:
                cur_dd_start = peak_date
        dd = v / peak - 1 if peak else 0
        if dd < mdd:
            mdd = dd
            mdd_start = cur_dd_start or peak_date
            mdd_end = d
        if v >= peak:
            peak_date = d

    recovery_days = None
    if mdd_end is not None:
        trough_value = min(v for v, d in zip(values, dates) if d == mdd_end)
        after = [(v, d) for v, d in zip(values, dates) if d > mdd_end]
        for v, d in after:
            if v >= peak:
                recovery_days = (d - mdd_end).days
                break

    daily_returns = [(values[i] / values[i - 1] - 1) for i in range(1, len(values)) if values[i - 1]]
    sharpe = None
    if len(daily_returns) > 1 and statistics.pstdev(daily_returns) > 0:
        sharpe = statistics.mean(daily_returns) / statistics.pstdev(daily_returns) * (252 ** 0.5)

    closed = [t for t in trades if t.get("side") == "청산" and t.get("pnl_usd") is not None]
    wins = [t for t in closed if t["pnl_usd"] > 0]
    losses = [t for t in closed if t["pnl_usd"] <= 0]
    win_rate = len(wins) / len(closed) if closed else None
    avg_win = statistics.mean(t["pnl_usd"] for t in wins) if wins else 0
    avg_loss = statistics.mean(t["pnl_usd"] for t in losses) if losses else 0
    payoff_ratio = (avg_win / abs(avg_loss)) if avg_loss else None
    expectancy_r = statistics.mean(t["r"] for t in closed if t.get("r") is not None) if any(t.get("r") is not None for t in closed) else None
    gross_profit = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss else None

    worst_trade = min(closed, key=lambda t: t["pnl_usd"], default=None)

    return {
        # equity_rows의 total_krw는 이미 수수료·환전 스프레드·양도세·배당 원천징수를
        # 반영한 실제 현금 잔고 기반이라(Broker가 매 거래마다 깎는다), 이 CAGR은
        # "세전"이 아니라 세후(비용+세금 다 뺀) 값이다 — 이름에 주의.
        "cagr_posttax_pct": round(cagr_pretax * 100, 2) if cagr_pretax is not None else None,
        "mdd_pct": round(mdd * 100, 2),
        "mdd_start": mdd_start.date().isoformat() if mdd_start is not None else None,
        "mdd_end": mdd_end.date().isoformat() if mdd_end is not None else None,
        "mdd_recovery_days": recovery_days,
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "win_rate_pct": round(win_rate * 100, 1) if win_rate is not None else None,
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
        "expectancy_r": round(expectancy_r, 2) if expectancy_r is not None else None,
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "trade_count": len(closed),
        "worst_trade_usd": round(worst_trade["pnl_usd"], 2) if worst_trade else None,
        "worst_trade_ticker": worst_trade["ticker"] if worst_trade else None,
    }


# ── 출력 ────────────────────────────────────────────────────────────────────


def make_run_id(cfg: dict, start: date, end: date) -> str:
    payload = json.dumps({"cfg": cfg.get("backtest"), "plan": cfg.get("plan"), "start": str(start), "end": str(end)}, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{date.today().isoformat()}_{digest}"


def write_outputs(run_id: str, cfg: dict, result: BacktestResult, benchmark: BenchmarkResult, data: BacktestData, metrics: dict, start: date, end: date) -> Path:
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result.equity_rows).to_csv(out_dir / "equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result.trades).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result.rejected).to_csv(out_dir / "rejected.csv", index=False, encoding="utf-8-sig")
    (out_dir / "config_snapshot.yaml").write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    rejected_reasons = defaultdict(int)
    for r in result.rejected:
        rejected_reasons[r["reason"]] += 1
    top_reasons = sorted(rejected_reasons.items(), key=lambda kv: kv[1], reverse=True)[:10]
    unfilled = sum(1 for r in result.rejected if "미체결" in r["reason"])
    entries = sum(1 for t in result.trades if t.get("side") == "진입")

    monthly_signals = defaultdict(int)
    for t in result.trades:
        if t.get("side") == "진입":
            monthly_signals[t["date"][:7]] += 1
    avg_monthly_signals = round(sum(monthly_signals.values()) / max(len(monthly_signals), 1), 1)

    lines = [
        f"# 백테스트 결과 — {run_id}",
        "",
        f"기간: {start} ~ {end} (학습 구간, 검증 구간 2022- 봉인)",
        f"유니버스: {data.universe_mode} · SURVIVORSHIP_BIAS = {'TRUE' if data.survivorship_bias else 'FALSE'}",
        f"data_gap 종목: {len(data.data_gap)}개, 시세 실패 종목: {len(data.failed_tickers)}개",
        "",
        "## 벤치마크 (QQQ 매수 후 보유, 배당 재투자 — 비용·배당 원천징수는 이미 반영)",
        f"- 세후(b, 미청산 — 자본이득세는 아직 안 냄) CAGR: {benchmark.cagr_unrealized * 100:.2f}%",
        f"- 세후(a, 기간 끝 청산해 자본이득세까지 낸 경우) CAGR: {benchmark.cagr_liquidated * 100:.2f}%",
        f"- MDD: {benchmark.mdd_pct:.2f}%",
        "",
        "## B0 (전략 v3 + QQQM + 슬롯 8 + 전략한도 60% + 4,000만 원)",
        f"- 세후(비용·양도세·배당원천징수 모두 반영) CAGR: {metrics.get('cagr_posttax_pct')}%",
        f"- QQQ 대비(세후 b 기준): {(metrics.get('cagr_posttax_pct') or 0) - benchmark.cagr_unrealized * 100:.2f}%p",
        f"- MDD: {metrics.get('mdd_pct')}% ({metrics.get('mdd_start')} ~ {metrics.get('mdd_end')}, 회복 {metrics.get('mdd_recovery_days')}일)",
        f"- 샤프: {metrics.get('sharpe')}",
        f"- 승률: {metrics.get('win_rate_pct')}%",
        f"- 손익비: {metrics.get('payoff_ratio')}",
        f"- 기대값(R): {metrics.get('expectancy_r')}",
        f"- Profit Factor: {metrics.get('profit_factor')}",
        f"- 청산 거래 수: {metrics.get('trade_count')} (진입 {entries}건)",
        f"- 월평균 신규 진입: {avg_monthly_signals}건",
        f"- 최악의 거래: {metrics.get('worst_trade_ticker')} {metrics.get('worst_trade_usd')}달러",
        "",
        "## 데이터·체결 상태",
        f"- 미체결(지정가 도달 못 함) 건수: {unfilled}",
        "- 탈락 신호 상위 사유: " + ", ".join(f"{k}({v})" for k, v in top_reasons),
        "",
        "## 상태",
        "PASS/FAIL/BLOCKED/UNKNOWN 판정은 완료 보고에 기록 (p5_plan.md 1장 합격 기준 — 이번 단계는 B0 확정까지만, 실험 A~F는 P5-2 이후).",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def run(cfg: dict, start: date, end: date, warmup_start: date | None = None) -> Path:
    warmup_start = warmup_start or _parse_date(cfg["backtest"]["warmup_start"])
    data = prepare_data(cfg, warmup_start, end)
    print("[backtest] B0 포트폴리오 시뮬레이션 중...")
    result = simulate_portfolio(data, cfg, start, end)
    print("[backtest] 벤치마크(QQQ 매수 후 보유) 시뮬레이션 중...")
    benchmark = simulate_benchmark(data, cfg, start, end)
    metrics = compute_metrics(result.equity_rows, result.trades, start, end)
    run_id = make_run_id(cfg, start, end)
    out_dir = write_outputs(run_id, cfg, result, benchmark, data, metrics, start, end)
    print(f"[backtest] 결과: {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="나스닥 100 MACD 스크리너 — 백테스트(P5-1)")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = cfg["backtest"]
    start = _parse_date(args.start) if args.start else _parse_date(bt_cfg["train_start"])
    end = _parse_date(args.end) if args.end else _parse_date(bt_cfg["train_end"])
    run(cfg, start, end)


if __name__ == "__main__":
    main()
