"""포트폴리오 엔진 (P6-1). 코어(QQQM)·대기 자금(합성 단기국채)·위성(QLD)·현금을
비중·규칙으로 운용하는 일별 시뮬레이터.

engine/backtest.py(신호 전략, P5)는 건드리지 않는다. 여기서 재사용하는 것은
그 파일의 순수 계산 함수(compute_equity_metrics 등)와 접합 함수(splice_pre_inception_series)뿐이다.

체결 규칙(중요, docs/p6_1_instructions.md 2번):
- 리밸런싱 신호는 confirmed 종가(그날 장 마감 후 확정된 봉)로 판단하고, 다음 거래일
  시가에 체결한다(국채/대기 자금은 시가 개념이 없어 원금만 옮기고, 다음 날부터의
  일별 수익부터 반영한다).
- 코어(QQQM)·위성(QLD)은 core.tax의 양도세(연간 실현손익, 250만 원 공제 후 22%,
  다음 해 5월 납부, 취득가는 이동평균법)를 적용한다. 대기 자금(합성 단기국채)은
  가격 변동이 없는 이자성 자산으로 보고 양도세 대상에서 뺀다 — [가정, 판단 필요].
- 초기 자금은 시작일에 원화->달러 1회 환전(costs.fx_spread_pct) 후 그날 종가로
  최초 비중을 매수한다.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import synthetic_assets as sa  # noqa: E402
from core import tax  # noqa: E402
from data import backtest_prices as bp  # noqa: E402
from data import fx as fxmod  # noqa: E402
from engine.backtest import (  # noqa: E402
    OUTPUT_ROOT,
    compute_block_bootstrap_ci,
    compute_equity_metrics,
    compute_sma_regime,
    compute_yearly_returns_from_equity,
    load_config,
    make_run_id,
    splice_pre_inception_series,
)

_ = OUTPUT_ROOT, compute_block_bootstrap_ci, make_run_id  # re-export for scripts/p6_1_experiments.py


# ── 자산 원장(이동평균법 취득가) ──────────────────────────────────────────────


@dataclass
class PortfolioAsset:
    """평균원가법으로 취득가를 관리하는 보유분 하나(코어=QQQM 또는 위성=QLD).

    shares: 보유 수량(달러 자산 기준)
    cost_usd: 총 매입원가(달러, 수수료 포함)
    """

    shares: float = 0.0
    cost_usd: float = 0.0

    def value_usd(self, price_usd: float) -> float:
        return self.shares * price_usd

    def buy(self, usd_amount: float, price_usd: float, commission_pct: float, apply_costs: bool) -> float:
        """usd_amount달러(수수료 전)를 이 자산에 투입한다. 실제로 나간 총 달러(수수료 포함)를 반환."""
        if usd_amount <= 0 or price_usd <= 0:
            return 0.0
        commission = usd_amount * commission_pct / 100 if apply_costs else 0.0
        shares = usd_amount / price_usd
        self.shares += shares
        self.cost_usd += usd_amount
        return usd_amount + commission

    def sell(self, usd_amount: float, price_usd: float, commission_pct: float, apply_costs: bool) -> tuple[float, float]:
        """usd_amount달러(수수료 전 평가액)만큼 판다. (받은 순 달러, 실현손익 달러)를 반환.

        평균원가법: 매도 비율만큼 총 매입원가에서 비례 차감한다.
        """
        if usd_amount <= 0 or self.shares <= 0 or price_usd <= 0:
            return 0.0, 0.0
        usd_amount = min(usd_amount, self.shares * price_usd)
        shares_sold = usd_amount / price_usd
        avg_cost = self.cost_usd / self.shares
        cost_removed = avg_cost * shares_sold
        commission = usd_amount * commission_pct / 100 if apply_costs else 0.0
        proceeds = usd_amount - commission
        gain = proceeds - cost_removed
        self.shares -= shares_sold
        self.cost_usd -= cost_removed
        return proceeds, gain


@dataclass
class PortfolioBroker:
    """달러 계좌: 코어(QQQM)·위성(QLD)·대기 자금(합성 단기국채)·현금.

    대기 자금은 가격 변동이 없는 이자성 자산으로 보아 양도세 대상에서 뺀다(이 파일
    상단 docstring의 가정 참고). 코어·위성만 realized_gain_by_year(원화)에 반영한다.
    """

    cash_usd: float = 0.0
    core: PortfolioAsset = field(default_factory=PortfolioAsset)
    qld: PortfolioAsset = field(default_factory=PortfolioAsset)
    reserve_usd: float = 0.0
    realized_gain_by_year: dict = field(default_factory=lambda: defaultdict(float))
    dividend_withheld_usd: float = 0.0
    tax_log: list = field(default_factory=list)
    trade_log: list = field(default_factory=list)
    apply_costs: bool = True
    apply_tax: bool = True

    def _record_gain(self, gain_usd: float, fx_rate: float | None, year: int) -> None:
        if fx_rate:
            self.realized_gain_by_year[year] += gain_usd * fx_rate

    def buy_core(self, usd_amount: float, price_usd: float, cfg: dict, date_iso: str, count_as_trade: bool = True) -> float:
        cost = self.core.buy(usd_amount, price_usd, cfg["backtest"]["costs"]["commission_buy_pct"], self.apply_costs)
        self.cash_usd -= cost
        if cost > 0 and count_as_trade:
            self.trade_log.append({"date": date_iso, "asset": "core", "side": "buy", "usd": cost})
        return cost

    def sell_core(self, usd_amount: float, price_usd: float, cfg: dict, fx_rate: float | None, year: int, date_iso: str) -> float:
        proceeds, gain = self.core.sell(usd_amount, price_usd, cfg["backtest"]["costs"]["commission_sell_pct"], self.apply_costs)
        self.cash_usd += proceeds
        self._record_gain(gain, fx_rate, year)
        if proceeds > 0:
            self.trade_log.append({"date": date_iso, "asset": "core", "side": "sell", "usd": proceeds})
        return proceeds

    def buy_qld(self, usd_amount: float, price_usd: float, cfg: dict, date_iso: str, count_as_trade: bool = True) -> float:
        cost = self.qld.buy(usd_amount, price_usd, cfg["backtest"]["costs"]["commission_buy_pct"], self.apply_costs)
        self.cash_usd -= cost
        if cost > 0 and count_as_trade:
            self.trade_log.append({"date": date_iso, "asset": "qld", "side": "buy", "usd": cost})
        return cost

    def sell_qld(self, usd_amount: float, price_usd: float, cfg: dict, fx_rate: float | None, year: int, date_iso: str) -> float:
        proceeds, gain = self.qld.sell(usd_amount, price_usd, cfg["backtest"]["costs"]["commission_sell_pct"], self.apply_costs)
        self.cash_usd += proceeds
        self._record_gain(gain, fx_rate, year)
        if proceeds > 0:
            self.trade_log.append({"date": date_iso, "asset": "qld", "side": "sell", "usd": proceeds})
        return proceeds

    def move_cash_to_reserve(self, usd_amount: float, cfg: dict, date_iso: str) -> float:
        if usd_amount <= 0:
            return 0.0
        commission = usd_amount * cfg["backtest"]["costs"]["commission_buy_pct"] / 100 if self.apply_costs else 0.0
        credited = usd_amount - commission
        self.cash_usd -= usd_amount
        self.reserve_usd += credited
        self.trade_log.append({"date": date_iso, "asset": "reserve", "side": "buy", "usd": usd_amount})
        return credited

    def move_reserve_to_cash(self, usd_amount: float, cfg: dict, date_iso: str) -> float:
        if usd_amount <= 0 or self.reserve_usd <= 0:
            return 0.0
        usd_amount = min(usd_amount, self.reserve_usd)
        commission = usd_amount * cfg["backtest"]["costs"]["commission_sell_pct"] / 100 if self.apply_costs else 0.0
        proceeds = usd_amount - commission
        self.reserve_usd -= usd_amount
        self.cash_usd += proceeds
        self.trade_log.append({"date": date_iso, "asset": "reserve", "side": "sell", "usd": usd_amount})
        return proceeds

    def grow_reserve(self, daily_rate: float | None) -> None:
        if daily_rate is not None and self.reserve_usd > 0:
            self.reserve_usd *= 1 + daily_rate

    def receive_dividend(self, gross_usd: float, cfg: dict) -> float:
        rate = cfg["backtest"]["tax"]["dividend_withholding_pct"] / 100 if self.apply_tax else 0.0
        net_usd = tax.dividend_after_withholding(gross_usd, rate)
        self.cash_usd += net_usd
        self.dividend_withheld_usd += gross_usd - net_usd
        return net_usd

    def settle_may_tax(self, year: int, core_price_usd: float, fx_rate: float, cfg: dict, settlement_year: int, date_iso: str) -> float:
        """year에 실현한 손익의 양도세를 settlement_year 5월에 낸다(코어를 팔아 마련)."""
        tax_cfg = cfg["backtest"]["tax"]
        gain = self.realized_gain_by_year.get(year, 0.0)
        amount_krw = (
            tax.capital_gains_tax(gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
            if self.apply_tax
            else 0.0
        )
        if amount_krw <= 0:
            self.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": 0.0})
            return 0.0
        spread_pct = cfg["backtest"]["costs"]["fx_spread_pct"] if self.apply_costs else 0.0
        effective_fx_rate = fx_rate * (1 - spread_pct / 100)
        usd_needed = amount_krw / effective_fx_rate
        self.sell_core(usd_needed, core_price_usd, cfg, fx_rate, settlement_year, date_iso)
        self.cash_usd -= usd_needed  # 세금은 재투자하지 않고 밖으로 나간다
        self.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": amount_krw})
        return amount_krw

    def total_value_usd(self, core_price: float, qld_price: float) -> float:
        return self.cash_usd + self.core.value_usd(core_price) + self.qld.value_usd(qld_price) + self.reserve_usd


# ── 데이터 준비 ─────────────────────────────────────────────────────────────


@dataclass
class PortfolioData:
    qqq_df: pd.DataFrame
    qqq_dividends: pd.Series
    core_df: pd.DataFrame  # QQQM(2020-12+ 실제) + 그 이전 QQQ 접합
    core_dividends: pd.Series
    qld_df: pd.DataFrame  # QLD(2006-06+ 실제) + 그 이전 합성 접합
    qld_dividends: pd.Series
    reserve_daily_rate: pd.Series  # DTB3 -> 일율, 거래일에 맞춰 정렬
    fx_by_date: dict
    fx_fallback_stats: dict
    dtb3_stats: dict  # 보고용(휴일 채움 횟수 등)
    qld_synthesis_check: dict  # 겹치는 기간(2006~) 실제 QLD 대비 합성값 오차
    sma_source_df: pd.DataFrame  # ^NDX(QQQ 상장 전) + QQQ 접합 — 이동평균 워밍업용(P6-1.1 1번)


def splice_pre_inception_dividends(
    proxy_dividends: pd.Series, real_dividends: pd.Series,
    proxy_price_df: pd.DataFrame, spliced_price_df: pd.DataFrame, boundary: pd.Timestamp,
) -> pd.Series:
    """real_df(실제 상장 후 배당)보다 이른 구간은 proxy(QQQ) 배당을 spliced_price_df
    가격 수준에 맞춰(배당수익률 = 배당/가격 유지) 환산해 채운다 (순수 함수, P6-1.1 2번).

    QQQM은 2020-12 이전 배당 이력이 아예 없어(상장 전이라 당연히 없음), 그 구간을
    비워 두면 접합된 core_df(가격은 QQQ 수준을 이어받음)가 배당을 하나도 못 받는
    문제가 생긴다(P0가 QQQ보다 계속 뒤처지는 원인). boundary 이전 각 날짜의
    "배당/QQQ종가" 비율을 그대로 가져와 spliced_price_df의 그날 가격에 곱해
    같은 배당수익률을 유지한다.

    입력: proxy_dividends(QQQ 배당, 주당 달러), real_dividends(QQQM 실제 배당),
         proxy_price_df(QQQ 종가, boundary 이전 배당일을 모두 포함),
         spliced_price_df(접합된 core_df — boundary 이전은 proxy_price_df와 같은 인덱스),
         boundary(real_dividends가 시작되는 실제 상장일)
    출력: pd.Series(날짜 오름차순, 주당 달러) — boundary 이전은 환산값, 이후는 실제값
    """
    # 빈 Series(bp.fetch_dividends가 못 받으면 주는 기본 RangeIndex)는 boundary와
    # 비교할 수 없어 데이터가 있을 때만 날짜로 걸러낸다.
    pre = proxy_dividends.loc[proxy_dividends.index < boundary] if len(proxy_dividends) else proxy_dividends
    scaled: dict = {}
    for ts, amt in pre.items():
        if ts in proxy_price_df.index and ts in spliced_price_df.index:
            proxy_close = float(proxy_price_df.loc[ts, "close"])
            spliced_close = float(spliced_price_df.loc[ts, "close"])
            if proxy_close:
                scaled[ts] = float(amt) * (spliced_close / proxy_close)
    pre_series = pd.Series(scaled, dtype=float)
    post = real_dividends.loc[real_dividends.index >= boundary] if len(real_dividends) else real_dividends
    parts = [s for s in (pre_series, post) if len(s)]
    return pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)


def prepare_data(cfg: dict, start: date, end: date) -> PortfolioData:
    """P6-1 전용 데이터(QQQ·QQQM 접합·QLD 접합·DTB3·환율)를 받아 지표까지 계산해 둔다."""
    bt_cfg = cfg["backtest"]

    print("[portfolio] QQQ 시세·배당 받는 중 ...", flush=True)
    qqq_df, _ = bp.fetch_history(bt_cfg["benchmark_ticker"], start, end)
    qqq_dividends = bp.fetch_dividends(bt_cfg["benchmark_ticker"], start, end)

    print("[portfolio] QQQM 시세·배당 받는 중 (상장 전은 QQQ 접합) ...", flush=True)
    qqqm_real, _ = bp.fetch_history("QQQM", start, end)
    qqqm_dividends = bp.fetch_dividends("QQQM", start, end)
    if not qqqm_real.empty and start < qqqm_real.index[0].date():
        core_df = splice_pre_inception_series(qqq_df, qqqm_real, bt_cfg["qqq_expense_ratio_pct"], bt_cfg["qqqm_expense_ratio_pct"])
        # QQQM은 상장 전 배당 이력이 아예 없다(당연히) — 그대로 두면 접합 구간(QQQ 수준
        # 가격을 물려받은 core_df)이 배당을 하나도 못 받아 QQQ보다 계속 뒤처진다
        # (P6-1.1 2번 진단). QQQ 배당수익률을 유지해 접합 구간을 채운다.
        core_dividends = splice_pre_inception_dividends(qqq_dividends, qqqm_dividends, qqq_df, core_df, qqqm_real.index[0])
    else:
        core_df = qqqm_real if not qqqm_real.empty else qqq_df
        core_dividends = qqqm_dividends if not qqqm_real.empty else qqq_dividends

    print("[portfolio] 200일선 워밍업용 ^NDX 과거 데이터 받는 중 (QQQ 상장 전 구간) ...", flush=True)
    # QQQ는 1999-03-10부터만 있어, 그 시점의 200일 이동평균은 원래 계산이 안 된다
    # (P6-1.1 1번 진단 — P4가 워밍업 동안 기본값 FULL로 굳어 P3와 구분이 안 됐다).
    # QQQ 상장 전 나스닥 100 지수(^NDX)로 이어 붙여 첫날부터 유효한 이동평균을 만든다.
    # 주변값 최대 210일 이동평균에 필요한 여유를 넉넉히 두고 받는다.
    ndx_lookback_start = start - timedelta(days=450)
    try:
        ndx_df, _ = bp.fetch_history("^NDX", ndx_lookback_start, start)
    except Exception:
        ndx_df = pd.DataFrame()
    if not ndx_df.empty and ndx_df.index[0].date() < qqq_df.index[0].date():
        sma_source_df = splice_pre_inception_series(ndx_df, qqq_df, 0.0, 0.0)
    else:
        sma_source_df = qqq_df

    print("[portfolio] DTB3(단기 국채 연율) 받는 중 ...", flush=True)
    dtb3_raw = fxmod.fetch_fred_series_range("DTB3", start, end)
    trading_days = list(qqq_df.index)
    annual_rate = sa.align_rate_to_trading_days(dtb3_raw, trading_days)
    filled_holidays = int(annual_rate.isna().sum())  # 시작일 이전 값이 없어 못 채운 날짜 수(보통 0)
    reserve_daily_rate = sa.annual_yield_pct_to_daily_rate(annual_rate.ffill())

    print("[portfolio] QLD 시세·배당 받는 중 (2006-06 이전은 합성) ...", flush=True)
    qld_real, _ = bp.fetch_history("QLD", start, end)
    qld_dividends = bp.fetch_dividends("QLD", start, end)
    qld_synthesis_check: dict = {}
    if not qld_real.empty and start < qld_real.index[0].date():
        synthetic_returns = sa.synthesize_qld_daily_returns(qqq_df["close"], reserve_daily_rate.reindex(qqq_df.index).fillna(0.0))
        qld_df = sa.splice_synthetic_returns_before_real(synthetic_returns, qld_real)

        overlap_start = qld_real.index[0]
        overlap_end = min(qld_real.index[-1], pd.Timestamp(end) - pd.Timedelta(days=1))
        overlap_returns_real = qld_real["close"].pct_change().loc[overlap_start:overlap_end]
        overlap_returns_synth = synthetic_returns.reindex(overlap_returns_real.index)
        diff = (overlap_returns_real - overlap_returns_synth).dropna()
        if len(diff):
            real_annual = (1 + overlap_returns_real.dropna()).prod() ** (252 / len(overlap_returns_real.dropna())) - 1
            synth_annual = (1 + overlap_returns_synth.dropna()).prod() ** (252 / len(overlap_returns_synth.dropna())) - 1
            qld_synthesis_check = {
                "overlap_days": int(len(diff)),
                "annual_return_diff_pp": round((synth_annual - real_annual) * 100, 2),
                "tracking_error_daily_std_pct": round(float(diff.std()) * 100, 4),
                "synthetic_better_than_real": bool(synth_annual > real_annual),
            }
    else:
        qld_df = qld_real if not qld_real.empty else qqq_df

    print("[portfolio] 원/달러 환율(FRED 보완) 받는 중 ...", flush=True)
    fx_primary = fxmod.fetch_usd_krw_range(start, end)
    fx_fred = fxmod.fetch_fred_series_range("DEXKOUS", start, end)
    fx_merged, fx_stats = fxmod.merge_fx_with_fallback(fx_primary, fx_fred)
    fx_by_date: dict = {}
    last = None
    for d in trading_days:
        key = d.date().isoformat()
        if key in fx_merged:
            last = fx_merged[key]
        fx_by_date[key] = last

    return PortfolioData(
        qqq_df=qqq_df, qqq_dividends=qqq_dividends,
        core_df=core_df, core_dividends=core_dividends,
        qld_df=qld_df, qld_dividends=qld_dividends,
        reserve_daily_rate=reserve_daily_rate,
        fx_by_date=fx_by_date, fx_fallback_stats=fx_stats,
        dtb3_stats={"holiday_fill_unresolved_days": filled_holidays},
        qld_synthesis_check=qld_synthesis_check,
        sma_source_df=sma_source_df,
    )


# ── 시뮬레이션 ──────────────────────────────────────────────────────────────


@dataclass
class PortfolioResult:
    equity_rows: list
    broker: PortfolioBroker
    trade_count: int


def _price_on(df: pd.DataFrame, ts: pd.Timestamp, col: str = "close") -> float | None:
    if ts not in df.index:
        return None
    v = df.loc[ts, col]
    return None if pd.isna(v) else float(v)


def _may_settlement_days(trading_days: list) -> dict[int, pd.Timestamp]:
    """연도별로 그해 5월의 첫 거래일을 찾는다(다음 해 5월 납부일)."""
    out: dict[int, pd.Timestamp] = {}
    for d in trading_days:
        if d.month == 5 and d.year not in out:
            out[d.year] = d
    return out


def simulate_portfolio(
    data: PortfolioData, cfg: dict, start: date, end: date, candidate: str, *,
    drawdown_triggers_pct: tuple[float, ...] = (-20.0, -30.0, -40.0),
    sma_days: int = 200, qld_max_pct_of_total: float = 20.0,
    apply_costs: bool = True, apply_tax: bool = True,
) -> PortfolioResult:
    """P0·P3·P4·P5 후보를 [start, end]에서 시뮬레이션한다(순수 함수 — 같은 입력엔 같은 출력).

    candidate: "P0" | "P3" | "P4" | "P5". drawdown_triggers_pct·sma_days·qld_max_pct_of_total은
    주변값 실험용 파라미터(사전 등록의 neighborhood_values).
    """
    total_krw = cfg["backtest"]["total_krw"]
    costs = cfg["backtest"]["costs"]
    trading_days = [d for d in data.qqq_df.index if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
    if not trading_days:
        return PortfolioResult(equity_rows=[], broker=PortfolioBroker(apply_costs=apply_costs, apply_tax=apply_tax), trade_count=0)

    sma_ok = compute_sma_regime(data.sma_source_df, sma_days)  # {날짜 iso: bool}, ^NDX 워밍업 포함(P6-1.1 1번)
    all_trading_days = list(data.qqq_df.index)
    may_days = _may_settlement_days(all_trading_days)

    broker = PortfolioBroker(apply_costs=apply_costs, apply_tax=apply_tax)
    day0 = trading_days[0]
    fx0 = data.fx_by_date.get(day0.date().isoformat())
    core_price0 = _price_on(data.core_df, day0)
    if fx0 is None or core_price0 is None:
        return PortfolioResult(equity_rows=[], broker=broker, trade_count=0)

    spread = costs["fx_spread_pct"] / 100 if apply_costs else 0.0
    initial_usd = total_krw / fx0 * (1 - spread)
    broker.cash_usd = initial_usd
    day0_iso = day0.date().isoformat()
    # P4는 "코어 60%+대기 40%"가 기본 배분이 아니라 원래 100% 투자이고, 이동평균 아래로
    # 내려갔을 때만 40%를 대기로 뺀다 — 그런데 첫날 배분을 항상 60/40으로 고정해 두면
    # (아래 P3/P5와 같은 취급), 첫날 신호가 이미 "위 국면"이어도 크로스 이벤트가 한 번
    # 일어나기 전까지는 계속 60/40에 머문다(P6-1.2 진단 — 1999년에 QQQ가 200일선 위에
    # 있었는데도 P4가 P3처럼 60/40으로 시작해 +36.6%로 나온 원인). 첫날의 실제 국면을
    # 보고 100%(위)/60:40(아래)를 바로 정한다.
    day0_above_sma = sma_ok.get(day0_iso) if candidate == "P4" else None

    if candidate == "P0":
        broker.buy_core(initial_usd, core_price0, cfg, day0_iso)
    elif candidate == "P4":
        if day0_above_sma is False:
            broker.buy_core(initial_usd * 0.6, core_price0, cfg, day0_iso)
            broker.move_cash_to_reserve(initial_usd * 0.4, cfg, day0_iso)
        else:  # 위 국면이거나(True) 국면 미상(None) — 100% 투자로 시작
            broker.buy_core(initial_usd, core_price0, cfg, day0_iso)
    elif candidate in ("P3", "P5"):
        broker.buy_core(initial_usd * 0.6, core_price0, cfg, day0_iso)
        broker.move_cash_to_reserve(initial_usd * 0.4, cfg, day0_iso)
    else:
        raise ValueError(f"알 수 없는 후보: {candidate}")

    # ── 상태(P3/P5 낙폭 단계, P4/P5 이동평균 국면) ──
    peak = float(data.qqq_df.loc[day0, "close"])
    # peak은 에피소드 도중(낙폭 진행 중)에는 갱신되지 않고 그대로 "직전 최고점" 역할을 한다
    # (새 고점을 찍는 순간 = 회복 조건도 동시에 참이 되므로 별도 변수가 필요 없다).
    triggered: set[float] = set()
    core_state = "DEFENSIVE" if day0_above_sma is False else "FULL"  # P4: FULL | DEFENSIVE — 첫날 실제 배분과 일치시킴
    qld_active = False
    pending: list = []  # [(action_name, usd_amount_or_None)] — 다음 거래일 시가에 실행

    trade_count_before = len(broker.trade_log)
    equity_rows: list = []

    for i, d in enumerate(trading_days):
        date_iso = d.date().isoformat()
        fx_rate = data.fx_by_date.get(date_iso)

        broker.grow_reserve(data.reserve_daily_rate.get(d) if d in data.reserve_daily_rate.index else None)

        core_open = _price_on(data.core_df, d, "open")
        qld_open = _price_on(data.qld_df, d, "open")
        core_close = _price_on(data.core_df, d, "close")
        qld_close = _price_on(data.qld_df, d, "close")

        # 대기 중이던 액션을 오늘 시가로 체결
        for action, amount in pending:
            if action == "reserve_to_core" and core_open:
                usd = broker.move_reserve_to_cash(amount, cfg, date_iso)
                broker.buy_core(usd, core_open, cfg, date_iso)
            elif action == "core_to_reserve" and core_close:
                proceeds = broker.sell_core(amount, core_open or core_close, cfg, fx_rate, d.year, date_iso)
                broker.move_cash_to_reserve(proceeds, cfg, date_iso)
            elif action == "reserve_to_qld" and qld_open:
                usd = broker.move_reserve_to_cash(amount, cfg, date_iso)
                broker.buy_qld(usd, qld_open, cfg, date_iso)
            elif action == "core_to_qld" and qld_open and core_open:
                proceeds = broker.sell_core(amount, core_open, cfg, fx_rate, d.year, date_iso)
                broker.buy_qld(proceeds, qld_open, cfg, date_iso)
            elif action == "qld_to_core" and core_open:
                proceeds = broker.sell_qld(amount, qld_open or 0.0, cfg, fx_rate, d.year, date_iso)
                broker.buy_core(proceeds, core_open, cfg, date_iso)
            elif action == "rebalance_60_40" and core_open:
                total = broker.total_value_usd(core_open, qld_open or 0.0)
                target_core = total * 0.6
                cur_core = broker.core.value_usd(core_open)
                if cur_core > target_core:
                    proceeds = broker.sell_core(cur_core - target_core, core_open, cfg, fx_rate, d.year, date_iso)
                    broker.move_cash_to_reserve(proceeds, cfg, date_iso)
                elif cur_core < target_core:
                    need = target_core - cur_core
                    usd = broker.move_reserve_to_cash(need, cfg, date_iso)
                    broker.buy_core(usd, core_open, cfg, date_iso)
        pending = []

        # 배당(대상 기간 안의 배당락일) — 받은 즉시 같은 날 종가로 재투자한다
        # (engine.backtest.simulate_benchmark과 동일한 규칙. P6-1 재현 확인에서 이걸
        # 빠뜨렸던 것이 QQQ 벤치마크 재현 잔차의 주원인이었다 — 완료 보고 참고).
        if broker.core.shares > 0 and d in data.core_dividends.index and core_close:
            net_usd = broker.receive_dividend(float(data.core_dividends.loc[d]) * broker.core.shares, cfg)
            if net_usd > 0:
                broker.buy_core(net_usd, core_close, cfg, date_iso, count_as_trade=False)
        if broker.qld.shares > 0 and d in data.qld_dividends.index and qld_close:
            net_usd = broker.receive_dividend(float(data.qld_dividends.loc[d]) * broker.qld.shares, cfg)
            if net_usd > 0:
                broker.buy_qld(net_usd, qld_close, cfg, date_iso, count_as_trade=False)

        # 5월 양도세 정산
        if may_days.get(d.year) == d and core_close:
            broker.settle_may_tax(d.year - 1, core_close, fx_rate or 0.0, cfg, d.year, date_iso)

        qqq_close_today = float(data.qqq_df.loc[d, "close"])
        if qqq_close_today > peak:
            peak = qqq_close_today
        drawdown_pct = (qqq_close_today / peak - 1) * 100 if peak else 0.0

        if candidate in ("P3", "P5"):
            for th in drawdown_triggers_pct:
                if drawdown_pct <= th and th not in triggered:
                    triggered.add(th)
                    reserve_now = broker.reserve_usd
                    pending.append(("reserve_to_core", reserve_now / 3))
            if triggered and qqq_close_today >= peak:
                pending.append(("rebalance_60_40", None))
                triggered = set()

        if candidate == "P4":
            above = sma_ok.get(date_iso)
            if above is True and core_state == "DEFENSIVE":
                pending.append(("reserve_to_core", broker.reserve_usd))
                core_state = "FULL"
            elif above is False and core_state == "FULL" and core_close:
                pending.append(("core_to_reserve", broker.core.value_usd(core_close) * 0.4))
                core_state = "DEFENSIVE"

        if candidate == "P5":
            above = sma_ok.get(date_iso)
            deployed_this_episode = len(triggered) > 0
            if above is True and deployed_this_episode and not qld_active and core_close:
                total_now = broker.total_value_usd(core_close, qld_close or 0.0)
                target_qld = total_now * qld_max_pct_of_total / 100
                from_reserve = min(target_qld, broker.reserve_usd)
                pending.append(("reserve_to_qld", from_reserve))
                remaining = target_qld - from_reserve
                if remaining > 0:
                    pending.append(("core_to_qld", remaining))
                qld_active = True
            elif qld_active and (above is False or qqq_close_today >= peak):
                pending.append(("qld_to_core", broker.qld.shares * (qld_close or 0.0)))
                qld_active = False

        total_value_usd = broker.total_value_usd(core_close or 0.0, qld_close or 0.0)
        equity_rows.append({
            "date": date_iso,
            "total_krw": round(total_value_usd * (fx_rate or 0.0)) if fx_rate else None,
            "core_usd": broker.core.value_usd(core_close or 0.0),
            "qld_usd": broker.qld.value_usd(qld_close or 0.0),
            "reserve_usd": broker.reserve_usd,
            "cash_usd": broker.cash_usd,
        })

    trade_count = len(broker.trade_log) - trade_count_before
    return PortfolioResult(equity_rows=equity_rows, broker=broker, trade_count=trade_count)


# ── 지표 ────────────────────────────────────────────────────────────────────


def compute_longest_recovery_days(equity_rows: list) -> dict:
    """모든 고점-회복 구간 중 가장 긴 회복 기간(거래일 수)을 구한다 (순수 함수, P6-1 2번).

    구간 끝까지도 직전 고점을 못 넘은 낙폭이 지금까지의 최장값보다 길면 그것을
    "진행 중(open_ended)"으로 표시해 대신 보고한다.
    """
    rows = [r for r in equity_rows if r["total_krw"] is not None]
    if not rows:
        return {}
    peak_idx = 0
    peak_val = rows[0]["total_krw"]
    longest = 0
    longest_start = longest_end = None
    open_ended = False
    for i in range(1, len(rows)):
        v = rows[i]["total_krw"]
        if v >= peak_val:
            gap = i - peak_idx
            if gap > longest:
                longest, longest_start, longest_end, open_ended = gap, rows[peak_idx]["date"], rows[i]["date"], False
            peak_val = v
            peak_idx = i
    ongoing_gap = len(rows) - 1 - peak_idx
    if ongoing_gap > longest:
        longest, longest_start, longest_end, open_ended = ongoing_gap, rows[peak_idx]["date"], rows[-1]["date"], True
    return {
        "longest_recovery_trading_days": longest, "longest_recovery_start": longest_start,
        "longest_recovery_end": longest_end, "open_ended": open_ended,
    }


def compute_worst_year_month(equity_rows: list) -> dict:
    """연도별·월별 수익률 중 최악의 해·달을 구한다 (순수 함수)."""
    rows = [r for r in equity_rows if r["total_krw"] is not None]
    if not rows:
        return {"worst_year": None, "worst_month": None}
    yearly = compute_yearly_returns_from_equity(rows)
    worst_year = min(yearly.items(), key=lambda kv: (kv[1] is None, kv[1])) if yearly else None

    by_month = defaultdict(list)
    for r in rows:
        by_month[pd.Timestamp(r["date"]).strftime("%Y-%m")].append(r["total_krw"])
    months = sorted(by_month)
    monthly_pct = {}
    prev_last = by_month[months[0]][0] if months else None
    for m in months:
        vals = by_month[m]
        monthly_pct[m] = round((vals[-1] / prev_last - 1) * 100, 2) if prev_last else None
        prev_last = vals[-1]
    worst_month = min(monthly_pct.items(), key=lambda kv: (kv[1] is None, kv[1])) if monthly_pct else None
    return {"worst_year": worst_year, "worst_month": worst_month}


def compute_posttax_a(result: PortfolioResult, data: PortfolioData, cfg: dict, end: date) -> dict:
    """기간 끝에 코어·위성을 전량 매도하고 그해 양도세까지 낸 세후(a) 최종 가치(원화)를 구한다.

    result.broker는 건드리지 않는다(카피본에 적용).
    """
    ts = pd.Timestamp(end)
    fx_rate = data.fx_by_date.get(end.isoformat())
    core_price = _price_on(data.core_df, ts)
    qld_price = _price_on(data.qld_df, ts)
    if fx_rate is None or core_price is None:
        return {}

    broker = result.broker
    liq = PortfolioBroker(
        cash_usd=broker.cash_usd, core=PortfolioAsset(broker.core.shares, broker.core.cost_usd),
        qld=PortfolioAsset(broker.qld.shares, broker.qld.cost_usd), reserve_usd=broker.reserve_usd,
        realized_gain_by_year=defaultdict(float, broker.realized_gain_by_year),
        apply_costs=broker.apply_costs, apply_tax=broker.apply_tax,
    )
    date_iso = end.isoformat()
    liq.sell_core(liq.core.value_usd(core_price), core_price, cfg, fx_rate, end.year, date_iso)
    if qld_price is not None:
        liq.sell_qld(liq.qld.value_usd(qld_price), qld_price, cfg, fx_rate, end.year, date_iso)
    liq.cash_usd += liq.reserve_usd
    liq.reserve_usd = 0.0

    tax_cfg = cfg["backtest"]["tax"]
    gain = liq.realized_gain_by_year.get(end.year, 0.0)
    tax_amount_krw = (
        tax.capital_gains_tax(gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
        if liq.apply_tax else 0.0
    )
    spread_pct = cfg["backtest"]["costs"]["fx_spread_pct"] if liq.apply_costs else 0.0
    if tax_amount_krw > 0:
        effective_fx_rate = fx_rate * (1 - spread_pct / 100)
        liq.cash_usd -= tax_amount_krw / effective_fx_rate

    liquidated_value_krw = liq.cash_usd * fx_rate
    total_krw = cfg["backtest"]["total_krw"]
    years = (pd.Timestamp(end) - pd.Timestamp(result.equity_rows[0]["date"])).days / 365.25 if result.equity_rows else 0
    cagr = (liquidated_value_krw / total_krw) ** (1 / years) - 1 if years > 0 and liquidated_value_krw > 0 else -1.0
    return {"cagr_liquidated_pct": round(cagr * 100, 2), "tax_paid_krw": round(tax_amount_krw), "liquidated_value_krw": round(liquidated_value_krw)}
