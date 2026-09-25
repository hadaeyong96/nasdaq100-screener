"""과거 재생(백테스트) 엔진 (P5-1, P5-1.1 검산 반영).

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

계좌 통화(P5-1.1 3번, 중요): Broker는 **달러 계좌**다. QQQM<->종목 재배분은
달러 안에서만 일어나 환전이 없다. 환전은 시작할 때 원화를 달러로 바꾸는 한 번뿐이고
(costs.fx_spread_pct 반영), 그 뒤로는 수수료(commission_buy/sell_pct)만 매매마다
붙는다. 원화는 ① 세금 계산(원화 기준 실현손익) ② 보고용 평가(그날 환율로 환산)
에만 쓴다.

포지션·R(P5-1.1 2번): 1차·2차·3차(또는 재진입 1회)로 나뉜 부분 매수·매도를 전부
하나의 "포지션"으로 묶는다. R = 그 포지션의 **초기 위험**(1차 또는 재진입 진입
시점의 수량 × 주당위험, 원화) 분의 그 포지션 전체 손익(원화) — core/sizing.py의
계산과 같은 뜻이다. trades.csv의 개별 진입·청산 행에는 그 행이 속한 position_id를
남기고, 승률·손익비·기대값 같은 "거래 품질" 지표는 position_id로 묶은 뒤에만 계산한다
(부분 청산을 거래 하나로 세지 않는다).

실행:
    python -m engine.backtest              config.yaml의 backtest.train_start~train_end(B0)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
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
from engine.daily import _BUY_KINDS, _NEW_POSITION_KINDS, _SELL_KINDS, _label_filter_reason  # noqa: E402

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


# ── 자금 계좌 (달러 계좌 + QQQM 평균원가법, P5-1.1 3번) ─────────────────────


@dataclass
class Broker:
    """달러 현금 계좌 + QQQM(쉬는 돈) 보유분. 신호 종목 매매는 전부 이 계좌를 거쳐 간다.

    QQQM<->종목 재배분은 둘 다 달러 자산이라 환전이 없다(수수료만). 원화는 세금
    계산과 보고용 평가에만 등장한다. apply_costs=False면 수수료를, apply_tax=False면
    양도세·배당 원천징수를 0으로 둔다(원인 분해용, P5-1.1 4번).
    """

    cash_usd: float
    qqqm_shares: float = 0.0
    qqqm_cost_usd: float = 0.0  # 평균원가법 총 매입원가(달러)
    realized_gain_by_year: dict = field(default_factory=lambda: defaultdict(float))  # 원화, 신호종목+QQQM 전체
    qqqm_realized_gain_by_year: dict = field(default_factory=lambda: defaultdict(float))  # 원화, QQQM 매매분만
    dividend_withheld_usd: float = 0.0
    dividend_log: list = field(default_factory=list)  # [{"date","ticker","gross_usd","net_usd"}] — P5-3 1-1번 귀속 분해용
    tax_log: list = field(default_factory=list)
    apply_costs: bool = True
    apply_tax: bool = True

    def _commission(self, usd_amount: float, pct: float) -> float:
        return usd_amount * pct / 100 if self.apply_costs else 0.0

    def buy_usd_asset(self, usd_amount: float, commission_pct: float) -> float:
        """달러로 usd_amount달러어치 미국 자산을 산다(환전 없음, 수수료만). 실제로 나간 달러를 반환."""
        cost = usd_amount + self._commission(usd_amount, commission_pct)
        self.cash_usd -= cost
        return cost

    def sell_usd_asset(self, usd_amount: float, commission_pct: float) -> float:
        """미국 자산 usd_amount달러어치(수수료 전)를 판다. 받은 순 달러(수수료 뺀 값)를 반환."""
        proceeds = usd_amount - self._commission(usd_amount, commission_pct)
        self.cash_usd += proceeds
        return proceeds

    def buy_qqqm(self, usd_amount: float, price_usd: float, cfg: dict) -> float:
        if usd_amount <= 0 or price_usd <= 0:
            return 0.0
        costs = cfg["backtest"]["costs"]
        self.buy_usd_asset(usd_amount, costs["commission_buy_pct"])
        shares = usd_amount / price_usd
        self.qqqm_shares += shares
        self.qqqm_cost_usd += usd_amount
        return shares

    def sell_qqqm(self, usd_amount: float, price_usd: float, cfg: dict, fx_rate: float, year: int) -> float:
        """QQQM을 usd_amount달러어치 판다. fx_rate는 실현손익을 원화로 환산해 세금 계산에
        쓸 뿐 환전 비용은 없다(P5-1.1 3번). 받은 순 달러를 반환."""
        if usd_amount <= 0 or self.qqqm_shares <= 0:
            return 0.0
        usd_amount = min(usd_amount, self.qqqm_shares * price_usd)
        costs = cfg["backtest"]["costs"]
        shares = usd_amount / price_usd
        avg_cost = self.qqqm_cost_usd / self.qqqm_shares
        cost_removed = avg_cost * shares
        proceeds = self.sell_usd_asset(usd_amount, costs["commission_sell_pct"])
        gain_usd = proceeds - cost_removed  # 매도수수료까지 반영한 실현손익(달러)
        self.qqqm_shares -= shares
        self.qqqm_cost_usd -= cost_removed
        if fx_rate:
            gain_krw = gain_usd * fx_rate
            self.realized_gain_by_year[year] += gain_krw
            self.qqqm_realized_gain_by_year[year] += gain_krw
        return proceeds

    def pay_capital_gains_tax(self, year: int, price_usd: float, fx_rate: float, cfg: dict, settlement_year: int) -> float:
        """year의 실현손익에 대한 양도세를 settlement_year(보통 year+1) 5월에 낸다 (QQQM을 팔아 마련).

        세금은 원화로 낸다고 보고, 달러를 원화로 바꾸는 그 순간에 환전 스프레드를
        적용한다(P5-2 0-2번) — 세금을 마련하려고 파는 QQQM 자체의 실현손익은 그 해
        (year)가 아니라 실제로 파는 시점(settlement_year)에 잡는다.
        """
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
        effective_fx_rate = fx_rate * (1 - spread_pct / 100)  # 원화로 낼 때 스프레드만큼 덜 받는다
        usd_needed = amount_krw / effective_fx_rate
        self.sell_qqqm(usd_needed, price_usd, cfg, fx_rate, year=settlement_year)
        self.cash_usd -= usd_needed  # 세금은 재투자하지 않고 밖으로 나간다
        self.tax_log.append({"year": year, "gain_krw": gain, "tax_krw": amount_krw})
        return amount_krw

    def receive_dividend(self, gross_usd: float, cfg: dict, date_iso: str | None = None, ticker: str | None = None) -> float:
        """배당(달러, 세전)을 15% 원천징수 후 현금에 더한다(환전 없음). 순 달러를 반환.

        date_iso·ticker(선택)를 주면 dividend_log에 남긴다 (P5-3 1-1번 2016년 귀속 분해용)."""
        div_cfg = cfg["backtest"]["tax"]
        rate = div_cfg["dividend_withholding_pct"] / 100 if self.apply_tax else 0.0
        net_usd = tax.dividend_after_withholding(gross_usd, rate)
        self.cash_usd += net_usd
        self.dividend_withheld_usd += gross_usd - net_usd
        if date_iso is not None:
            self.dividend_log.append({"date": date_iso, "ticker": ticker, "gross_usd": gross_usd, "net_usd": net_usd})
        return net_usd


def _deposit_krw_to_usd(krw_amount: float, fx_rate: float, spread_pct: float, apply_costs: bool) -> float:
    """맨 처음 원화를 달러로 바꾼다(P5-1.1 3번: 환전은 여기 한 번뿐)."""
    spread = spread_pct / 100 if apply_costs else 0.0
    return krw_amount / fx_rate * (1 - spread)


# ── 데이터 준비 ─────────────────────────────────────────────────────────────


@dataclass
class BacktestData:
    indicator_map: dict
    dividends: dict
    checkpoints: list
    fx_by_date: dict
    universe_mode: str  # POINT_IN_TIME | CURRENT_CONSTITUENTS
    survivorship_bias: str  # "FALSE" | "PARTIAL" | "TRUE"
    failed_tickers: dict
    data_gap: dict
    qqq_df: pd.DataFrame
    qqq_dividends: pd.Series
    cash_etf_df: pd.DataFrame  # QQQM(상장 후) / QQQ(상장 전 대체, 수준을 맞춰 접합)
    cash_etf_dividends: pd.Series
    all_needed_tickers: set = field(default_factory=set)


def splice_pre_inception_series(proxy_df: pd.DataFrame, real_df: pd.DataFrame, proxy_expense_ratio_pct: float, real_expense_ratio_pct: float) -> pd.DataFrame:
    """real_df(실제 상장 후 데이터)보다 이른 구간을 proxy_df(대용 자산)의 "수익률"로
    채워 real_df 앞에 붙인다 (P5-1.1 2·6번, 순수 함수 — 회귀 방지 테스트 대상).

    접합 경계(real_df의 첫 날짜)에서 proxy 구간의 수준을 real_df의 실제 종가에
    맞춰 다시 고정한다 — 두 자산이 아예 다른 주당 가격대(별도 주식군)일 수 있어,
    보수 차이만 곱하면 접합 경계에서 가격이 어긋나기 때문이다(예: QQQ ~290 ->
    QQQM ~120로 하루 만에 반토막 나는 것처럼 보이는 문제, 실제로 있었던 회귀).

    입력: proxy_df(대용 자산의 open/high/low/close, real_df보다 이전 구간 포함),
         real_df(실제 자산의 open/high/low/close, index[0]이 접합 경계),
         proxy_expense_ratio_pct(대용 자산 보수, %), real_expense_ratio_pct(실제 자산 보수, %)
    출력: proxy 구간(접합 경계 이전, 수준을 맞춘) + real_df를 이어 붙인 DataFrame,
         날짜 오름차순, 중복 없음
    """
    real_start = real_df.index[0]
    real_start_close = float(real_df.loc[real_start, "close"])
    pre = proxy_df.loc[proxy_df.index < real_start].copy()
    if pre.empty:
        return real_df.copy()
    expense_drag_daily = (proxy_expense_ratio_pct - real_expense_ratio_pct) / 100 / 252
    drag_factor = (1 - expense_drag_daily) ** pd.Series(range(len(pre)), index=pre.index)[::-1]
    proxy_close_at_boundary = float(pre["close"].iloc[-1])
    level_scale = real_start_close / (proxy_close_at_boundary * drag_factor.iloc[-1])
    for col in ("open", "high", "low", "close"):
        pre[col] = pre[col] * drag_factor.values * level_scale
    out = pd.concat([pre, real_df])
    assert out.index.is_monotonic_increasing and not out.index.duplicated().any()
    return out


def prepare_data(cfg: dict, warmup_start: date, end: date) -> BacktestData:
    """종목·QQQ·QQQM·환율·배당·시점별 구성 종목을 모두 받아 지표까지 계산해 둔다."""
    bt_cfg = cfg["backtest"]

    print("[backtest] 나스닥 100 현재 구성 종목을 가져오는 중...")
    current_tickers = set(get_universe()["ticker"])

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
        all_needed = set(current_tickers)
        checkpoints = [(warmup_start, frozenset(current_tickers))]

    print(f"[backtest] 종목 {len(all_needed)}개의 과거 일봉을 받는 중... (시간이 걸립니다)")
    price_result = bp.fetch_universe_history(sorted(all_needed), warmup_start, end)
    print(f"[backtest]   성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")

    # 생존자 편향 표시 (P5-1.1 5번): 시점별 구성 종목을 못 구했으면 TRUE, 구했지만
    # 그중 일부(특히 인수·상폐 종목)의 시세를 못 받았으면 FALSE가 아니라 PARTIAL.
    if universe_mode == "CURRENT_CONSTITUENTS":
        survivorship_bias = "TRUE"
    elif price_result.failed:
        survivorship_bias = "PARTIAL"
    else:
        survivorship_bias = "FALSE"

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
        if not cash_etf_df.empty and warmup_start < cash_etf_df.index[0].date():
            cash_etf_df = splice_pre_inception_series(
                qqq_df_raw, cash_etf_df, bt_cfg["qqq_expense_ratio_pct"], bt_cfg["qqqm_expense_ratio_pct"]
            )

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
        all_needed_tickers=all_needed,
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
    still_open_position_ids: set
    final_date: date | None
    states: dict = field(default_factory=dict)  # {ticker: 마지막 상태} — 청산 시나리오 계산용(P5-2 0-1번)


def simulate_portfolio(
    data: BacktestData, cfg: dict, start: date, end: date, apply_costs: bool = True, apply_tax: bool = True,
    max_slots: int | None = None, regime_ok: dict | None = None,
    entry_type_allowed: set | None = None, rs_filter: dict | None = None,
    a1_whipsaw_block: dict | None = None, volume_filter: dict | None = None,
    atr_trail_mult: float | None = None, partial_tp_r_mult: float | None = None,
    blocked_new_entries: set | None = None,
) -> BacktestResult:
    """core.state.process_day를 하루씩 재생해 포트폴리오를 시뮬레이션한다.

    apply_costs=False/apply_tax=False는 원인 분해용(P5-1.1 4번)이다. core.sizing이
    쓰는 수량 결정은 cfg의 고정값(총자금·전략한도)과 시세만 보고 Broker의 실제
    현금·비용과 무관하게 정해지므로, 세 시나리오(무비용무세금/비용만/비용+세금)의
    신호·체결·수량은 항상 같고 현금 흐름만 달라진다.

    max_slots(P5-2 A1)는 cfg["plan"]["max_slots"] 대신 쓸 동시 보유 한도(생략하면
    cfg 값). regime_ok(P5-2 B1·B2)는 {날짜.isoformat(): bool} — False인 날은 신규
    진입(A1·B) 후보를 전혀 받지 않는다(보유 종목의 추가매수·청산 규칙은 그대로).

    P5-3 실험용 오버라이드(모두 실험 하나만 켜는 게 원칙 — 동시에 여러 개를 켜면
    "단일 변수"가 아니게 된다):
    - entry_type_allowed(T1): {"A1","B"}의 부분집합 — 이 kind의 신규 진입만 후보로 남긴다.
    - rs_filter(C1): compute_relative_strength_top_half 결과 — 상위 50% 밖 종목은 후보에서 뺀다.
    - a1_whipsaw_block(C2): compute_rsi_whipsaw_block 결과 — True인 (종목,날짜)의 A1만 뺀다.
    - volume_filter(C3): compute_volume_entry_filter 결과 — False인 (종목,날짜)는 후보에서 뺀다.
    - atr_trail_mult(E1): 진입 후 최고 종가 − 이 배수×ATR(14, indicator_map의 "atr" 열 필요)
      아래로 종가가 내려가면 손절(기존 손절가는 그대로 두고 더 타이트한 쪽을 쓴다).
    - partial_tp_r_mult(E2): 평균단가 대비 이 배수×R(진입 시점 주당위험) 도달 시 보유의
      절반을 그날 종가로 즉시 매도(단순화 — 다른 청산은 다음날 시가 체결이지만 이건
      "도달하면 바로 판다"는 규칙을 그대로 반영해 같은 날 체결로 둔다).
    - blocked_new_entries(P5-3 1-2번 반사실 재시뮬레이션): {(종목, 체결일.isoformat())} —
      해당 신규 진입이 실제로 없었던 것처럼 미체결 경로로 되돌린다.
    """
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    cash_buffer_krw = total_krw * cfg["plan"]["cash_buffer_pct"] / 100
    max_slots = max_slots if max_slots is not None else cfg["plan"]["max_slots"]
    costs = bt_cfg["costs"]

    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end)]
    indicator_map = data.indicator_map
    states = {t: st.init_state(t) for t in indicator_map}
    broker = Broker(cash_usd=0.0, apply_costs=apply_costs, apply_tax=apply_tax)

    equity_rows: list = []
    trades: list = []
    rejected: list = []
    paid_years: set = set()
    open_positions: dict[str, dict] = {}  # ticker -> {"id", "initial_risk_krw", "risk_per_share_usd"}
    next_position_id = [1]
    trailing_high: dict[str, float] = {}  # P5-3 E1(ATR 추적 손절)용 — 진입 후 최고 종가
    partial_tp_taken: set = set()  # P5-3 E2(분할 익절)용 — 이미 절반을 판 position_id

    # 첫 거래일에 버퍼를 뺀 전액을 달러로 바꿔(딱 한 번 환전) QQQM에 넣는다.
    first_ts = pd.Timestamp(trading_days[0])
    cash_etf_price0 = _price_on_or_before(data.cash_etf_df, first_ts)
    fx0 = data.fx_by_date.get(trading_days[0].isoformat())
    if cash_etf_price0 and fx0:
        usd0 = _deposit_krw_to_usd(total_krw - cash_buffer_krw, fx0, costs["fx_spread_pct"], apply_costs)
        broker.cash_usd += usd0
        broker.buy_qqqm(usd0, cash_etf_price0, cfg)

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
            is_new_fill = pending["kind"] in _NEW_POSITION_KINDS
            if is_new_fill and blocked_new_entries is not None and (ticker, date_.isoformat()) in blocked_new_entries:
                fill_price = None  # P5-3 1-2번 반사실: 이 신규 진입은 없었던 것으로(기존 미체결 경로 재사용)
            if fill_price is not None and qty > 0:
                usd_amount = fill_price * qty
                cash_etf_price = _price_on_or_before(data.cash_etf_df, ts) or fill_price
                broker.sell_qqqm(usd_amount, cash_etf_price, cfg, fx_rate, date_.year)
                broker.buy_usd_asset(usd_amount, costs["commission_buy_pct"])

                is_new = pending["kind"] in _NEW_POSITION_KINDS
                if is_new and ticker not in open_positions:
                    stop_price = state_.get("stop")
                    risk_per_share = sizing.per_share_risk(fill_price, stop_price, cfg) if stop_price is not None else fill_price * cfg["risk"]["min_risk_per_share_pct"] / 100
                    initial_risk_krw = qty * risk_per_share * (fx_rate or 0)
                    open_positions[ticker] = {"id": next_position_id[0], "initial_risk_krw": initial_risk_krw, "risk_per_share_usd": risk_per_share}
                    next_position_id[0] += 1
                position_open = open_positions.get(ticker, {})
                position_id = position_open.get("id")

                states[ticker] = st.apply_fill(state_, {"unit": pending["unit"], "side": "buy", "price": fill_price, "qty": qty}, cfg)
                trades.append(
                    {"date": date_.isoformat(), "ticker": ticker, "position_id": position_id, "side": "진입", "stage": pending["kind"],
                     "qty": qty, "price": fill_price, "fx_rate": fx_rate, "reason": "", "pnl_usd": None, "pnl_krw": None, "r": None,
                     # 포지션이 방금 새로 열렸을 때만(is_new) 채운다 — 2·3차 추가매수 행에는 안 남겨
                     # aggregate_positions가 "최초 진입" 값을 덮어쓰지 않게 한다(R의 정의, P5-1.1 2번).
                     "initial_risk_krw": position_open.get("initial_risk_krw") if is_new else None}
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

        # P5-3 단일 변수 실험 필터 — 지정된 것 하나만 후보를 줄인다(admitted 이전 단계).
        if entry_type_allowed is not None:
            candidates = [c for c in candidates if c["kind"] in entry_type_allowed]
        if rs_filter is not None:
            rs_allowed_today = rs_filter.get(date_.isoformat(), set())
            candidates = [c for c in candidates if c["ticker"] in rs_allowed_today]
        if a1_whipsaw_block is not None:
            candidates = [c for c in candidates if not (c["kind"] == "A1" and a1_whipsaw_block.get((c["ticker"], date_.isoformat()), False))]
        if volume_filter is not None:
            candidates = [c for c in candidates if volume_filter.get((c["ticker"], date_.isoformat()), False)]

        candidates.sort(key=lambda c: c["score"], reverse=True)
        held_count = sum(1 for s in states.values() if any(q > 0 for q in s["units"].values()))
        slots = max(max_slots - held_count, 0)
        regime_blocks_new_entries = regime_ok is not None and not regime_ok.get(date_.isoformat(), False)
        admitted = set() if regime_blocks_new_entries else {c["ticker"] for c in candidates[:slots]}

        # ── P5-3 E1: ATR 추적 손절 — 오늘 process_day가 보기 전에 stop을 더 타이트하게
        # 덮어쓴다(기존 손절가·추적 손절 중 더 높은 쪽이 먼저 걸린다). 어제까지의 최고
        # 종가로 오늘을 판정하고(당일 종가로 자기 자신을 미리 판정하지 않는다), 그다음
        # 오늘 종가로 최고치를 갱신해 내일 판정에 쓴다.
        if atr_trail_mult is not None:
            for ticker in active:
                held_qty = sum(q for q in states[ticker]["units"].values() if q > 0)
                if held_qty <= 0:
                    trailing_high.pop(ticker, None)
                    continue
                df = indicator_map[ticker]
                if ts not in df.index or "atr" not in df.columns:
                    continue
                close_today = df.loc[ts, "close"]
                atr_val = df.loc[ts, "atr"]
                prev_high = trailing_high.get(ticker)
                if prev_high is not None and not pd.isna(atr_val):
                    trail_stop = prev_high - atr_trail_mult * atr_val
                    cur_stop = states[ticker].get("stop")
                    if cur_stop is None or trail_stop > cur_stop:
                        states[ticker]["stop"] = trail_stop
                if not pd.isna(close_today):
                    trailing_high[ticker] = max(prev_high, float(close_today)) if prev_high is not None else float(close_today)

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
                    position_id = open_positions.get(ticker, {}).get("id")
                    _settle_sell(event, ticker, position_id, date_, ts, indicator_map[ticker], broker, cfg, fx_rate, trades)
                elif event["kind"] in _BUY_KINDS:
                    today_buy_events.setdefault(ticker, []).append(event)

            # 오늘 이 종목의 포지션이 완전히 청산됐으면(대기로 복귀) 포지션을 닫는다.
            if ticker in open_positions and states[ticker]["state"] == "대기":
                del open_positions[ticker]

        # ── P5-3 E2: 분할 익절 — 평균단가 대비 +partial_tp_r_mult×R 도달 시 보유의 절반을
        # 그날 종가로 즉시 매도한다(단순화 — 다른 청산은 다음날 시가 체결이지만, "도달하면
        # 즉시 판다"는 규칙 자체를 그대로 반영해 같은 날 체결로 둔다). 한 포지션당 한 번만.
        if partial_tp_r_mult is not None:
            for ticker in active:
                position_open = open_positions.get(ticker)
                if not position_open or position_open["id"] in partial_tp_taken:
                    continue
                risk_per_share = position_open.get("risk_per_share_usd")
                if not risk_per_share:
                    continue
                state_ = states[ticker]
                held_units = {u: q for u, q in state_["units"].items() if q and q > 0}
                total_qty = sum(held_units.values())
                if total_qty <= 0:
                    continue
                avg_cost = sum(q * state_["entries"][u] for u, q in held_units.items()) / total_qty
                df = indicator_map[ticker]
                if ts not in df.index:
                    continue
                close_today = df.loc[ts, "close"]
                if pd.isna(close_today):
                    continue
                target = avg_cost + partial_tp_r_mult * risk_per_share
                if float(close_today) < target:
                    continue
                half_qty = total_qty // 2
                if half_qty <= 0:
                    partial_tp_taken.add(position_open["id"])
                    continue
                sell_amounts: dict[str, int] = {}
                for u, q in held_units.items():
                    sell_amounts[u] = min(q, round(q * half_qty / total_qty))
                diff = half_qty - sum(sell_amounts.values())
                if diff != 0:
                    biggest_unit = max(held_units, key=held_units.get)
                    sell_amounts[biggest_unit] = max(0, min(held_units[biggest_unit], sell_amounts[biggest_unit] + diff))

                commission_sell_pct = costs["commission_sell_pct"] if broker.apply_costs else 0.0
                commission_buy_pct = costs["commission_buy_pct"] if broker.apply_costs else 0.0
                proceeds_usd = broker.sell_usd_asset(float(close_today) * half_qty, commission_sell_pct)
                cost_usd = avg_cost * half_qty * (1 + commission_buy_pct / 100)
                gain_usd = proceeds_usd - cost_usd
                gain_krw = gain_usd * fx_rate if fx_rate else 0.0
                if fx_rate:
                    broker.realized_gain_by_year[date_.year] += gain_krw
                trades.append(
                    {"date": date_.isoformat(), "ticker": ticker, "position_id": position_open["id"], "side": "청산",
                     "stage": "PARTIAL_TP", "qty": half_qty, "price": float(close_today), "fx_rate": fx_rate,
                     "reason": f"분할 익절(+{partial_tp_r_mult}R)", "entry_price": avg_cost,
                     "pnl_usd": round(gain_usd, 2), "pnl_krw": round(gain_krw), "r": None}
                )
                for u, take in sell_amounts.items():
                    if take > 0:
                        state_["units"][u] = max(state_["units"].get(u, 0) - take, 0)
                partial_tp_taken.add(position_open["id"])

        if today_buy_events:
            _size_and_stash(today_buy_events, states, indicator_map, ts, cfg, fx_rate, rejected, date_)

        # ── 배당 (환전 없음, P5-1.1 3번) ──
        for ticker, state_ in states.items():
            qty_held = sum(q for q in state_["units"].values() if q > 0)
            if qty_held <= 0:
                continue
            div_series = data.dividends.get(ticker)
            if div_series is None or ts not in div_series.index:
                continue
            gross_usd = float(div_series.loc[ts]) * qty_held
            broker.receive_dividend(gross_usd, cfg, date_iso=date_.isoformat(), ticker=ticker)
        if broker.qqqm_shares > 0 and ts in data.cash_etf_dividends.index:
            gross_usd = float(data.cash_etf_dividends.loc[ts]) * broker.qqqm_shares
            broker.receive_dividend(gross_usd, cfg, date_iso=date_.isoformat(), ticker="__CASH_ETF__")

        # ── 연간 양도세 (5월, 전년도분) ──
        if fx_rate is not None and date_.month >= bt_cfg["tax"]["payment_month"]:
            prior_year = date_.year - 1
            if prior_year in broker.realized_gain_by_year and prior_year not in paid_years:
                price_now = _price_on_or_before(data.cash_etf_df, ts)
                if price_now:
                    broker.pay_capital_gains_tax(prior_year, price_now, fx_rate, cfg, settlement_year=date_.year)
                    paid_years.add(prior_year)

        # ── 하루 끝: 남는 현금(버퍼만큼 달러로 환산한 값 제외)을 QQQM으로 쓸어 담는다 ──
        if fx_rate is not None:
            price_now = _price_on_or_before(data.cash_etf_df, ts)
            if price_now:
                cash_buffer_usd = cash_buffer_krw / fx_rate
                surplus_usd = broker.cash_usd - cash_buffer_usd
                if surplus_usd > 1:
                    broker.buy_qqqm(surplus_usd, price_now, cfg)
                elif surplus_usd < -1 and broker.qqqm_shares > 0:
                    broker.sell_qqqm(-surplus_usd, price_now, cfg, fx_rate, date_.year)

        # ── 자산 기록 ──
        if fx_rate is not None:
            price_now = _price_on_or_before(data.cash_etf_df, ts)
            qqqm_value_krw = broker.qqqm_shares * (price_now or 0) * fx_rate
            cash_krw = broker.cash_usd * fx_rate
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
                    "cash_krw": round(cash_krw),
                    "total_krw": round(qqqm_value_krw + positions_value_krw + cash_krw),
                }
            )

    still_open_ids = {v["id"] for v in open_positions.values()}
    return BacktestResult(
        equity_rows=equity_rows, trades=trades, rejected=rejected, broker=broker,
        still_open_position_ids=still_open_ids, final_date=trading_days[-1] if trading_days else None,
        states=states,
    )


def _price_on_or_before(df: pd.DataFrame, ts: pd.Timestamp) -> float | None:
    if df is None or df.empty:
        return None
    sub = df.loc[:ts]
    if sub.empty:
        return None
    close = sub.iloc[-1]["close"]
    return None if pd.isna(close) else float(close)


def _settle_sell(event, ticker, position_id, date_, ts, df, broker: Broker, cfg, fx_rate, trades: list) -> None:
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
            {"date": date_.isoformat(), "ticker": ticker, "position_id": position_id, "side": "청산", "stage": kind, "qty": qty,
             "price": None, "fx_rate": fx_rate, "reason": _SELL_REASON.get(kind, kind), "pnl_usd": None, "pnl_krw": None, "r": None}
        )
        return

    costs = cfg["backtest"]["costs"]
    usd_amount = exit_price * qty
    proceeds_usd = broker.sell_usd_asset(usd_amount, costs["commission_sell_pct"])
    cost_usd = entry_price * qty * (1 + (costs["commission_buy_pct"] / 100 if broker.apply_costs else 0))
    gain_usd = proceeds_usd - cost_usd
    gain_krw = gain_usd * fx_rate
    broker.realized_gain_by_year[date_.year] += gain_krw
    pnl_usd = (exit_price - entry_price) * qty
    trades.append(
        {
            "date": date_.isoformat(), "ticker": ticker, "position_id": position_id, "side": "청산", "stage": kind, "qty": qty,
            "price": exit_price, "fx_rate": fx_rate, "reason": _SELL_REASON.get(kind, kind),
            "entry_price": entry_price, "pnl_usd": round(pnl_usd, 2), "pnl_krw": round(gain_krw), "r": None,
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


# ── 포지션 단위 거래 품질 지표 (P5-1.1 2번) ──────────────────────────────────


def aggregate_positions(trades: list, still_open_position_ids: set) -> list[dict]:
    """trades를 position_id로 묶어 포지션 단위 손익·R을 계산한다 (순수 함수, 테스트 가능).

    입력: trades(진입·청산 행 목록, 각 행에 position_id·pnl_krw 등), still_open_position_ids
         (백테스트가 끝날 때까지 청산 안 된 포지션 id — 통계에서 뺀다)
    출력: [{"position_id","ticker","opened_date","closed_date","initial_risk_krw",
           "pnl_krw","r"}, ...] 청산 완료된 포지션만, position_id 오름차순
    """
    by_id: dict[int, dict] = {}
    for t in trades:
        pid = t.get("position_id")
        if pid is None:
            continue
        rec = by_id.setdefault(pid, {"position_id": pid, "ticker": t["ticker"], "pnl_krw": 0.0, "opened_date": None, "closed_date": None, "initial_risk_krw": t.get("initial_risk_krw")})
        if t["side"] == "진입":
            if rec["opened_date"] is None:
                rec["opened_date"] = t["date"]
            if t.get("initial_risk_krw") is not None:
                rec["initial_risk_krw"] = t["initial_risk_krw"]
        elif t["side"] == "청산":
            rec["closed_date"] = t["date"]
            if t.get("pnl_krw") is not None:
                rec["pnl_krw"] += t["pnl_krw"]

    out = []
    for pid, rec in sorted(by_id.items()):
        if pid in still_open_position_ids:
            continue
        if not rec["initial_risk_krw"]:
            continue
        rec["r"] = rec["pnl_krw"] / rec["initial_risk_krw"]
        out.append(rec)
    return out


def compute_position_stats(positions: list[dict]) -> dict:
    """포지션 목록(aggregate_positions 결과)에서 승률·손익비·기대값(R)을 계산한다 (순수 함수).

    기대값 = 승률 × 평균 이익(R) − 패율 × |평균 손실(R)| — 이 식과 실제 값이
    일치하는지는 tests에서 검증한다.
    """
    if not positions:
        return {"count": 0}
    wins = [p for p in positions if p["r"] > 0]
    losses = [p for p in positions if p["r"] <= 0]
    win_rate = len(wins) / len(positions)
    loss_rate = 1 - win_rate
    avg_win_r = statistics.mean(p["r"] for p in wins) if wins else 0.0
    avg_loss_r = statistics.mean(p["r"] for p in losses) if losses else 0.0
    max_loss_r = min((p["r"] for p in positions), default=0.0)
    payoff_ratio = (avg_win_r / abs(avg_loss_r)) if avg_loss_r else None
    expectancy_r = win_rate * avg_win_r - loss_rate * abs(avg_loss_r)
    gross_profit = sum(p["pnl_krw"] for p in wins)
    gross_loss = abs(sum(p["pnl_krw"] for p in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss else None
    worst = min(positions, key=lambda p: p["pnl_krw"], default=None)
    return {
        "count": len(positions),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_win_r": round(avg_win_r, 2),
        "avg_loss_r": round(avg_loss_r, 2),
        "max_loss_r": round(max_loss_r, 2),
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio is not None else None,
        "expectancy_r": round(expectancy_r, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor is not None else None,
        "worst_position_krw": round(worst["pnl_krw"]) if worst else None,
        "worst_position_ticker": worst["ticker"] if worst else None,
    }


# ── 청산 시나리오 (a) 공용 — 벤치마크·B0·실험 모두 같은 함수로 계산한다 ──────────


def _liquidate_qqqm_and_pay_tax(broker: Broker, price_usd: float | None, fx_rate: float, cfg: dict, year: int) -> tuple[float, float]:
    """QQQM을 전량 팔고, 그 시점까지 쌓인 실현손익 전체에 대한 양도세를 낸 뒤 실제로
    손에 남는 달러 현금(청산 전 cash_usd 포함)을 계산한다. 세금은 원화로 낸다고 보고
    그 환전에만 스프레드를 적용한다(P5-2 0-2번, 최종 원화 환산 자체는 평가용이라
    스프레드가 없다 — P5-1.1 3번).

    입력: broker(원본은 건드리지 않는다), price_usd(청산 시점 QQQM 가격),
         fx_rate, cfg, year(청산 연도 — 그때까지의 실현손익 전부에 과세)
    출력: (청산 후 남은 현금(달러), 낸 세금(원화))
    """
    liq_broker = Broker(
        cash_usd=0.0, qqqm_shares=broker.qqqm_shares, qqqm_cost_usd=broker.qqqm_cost_usd,
        realized_gain_by_year=defaultdict(float, broker.realized_gain_by_year),
        apply_costs=broker.apply_costs, apply_tax=broker.apply_tax,
    )
    if liq_broker.qqqm_shares > 0 and price_usd:
        liq_broker.sell_qqqm(liq_broker.qqqm_shares * price_usd, price_usd, cfg, fx_rate, year)

    total_gain = sum(liq_broker.realized_gain_by_year.values())
    tax_cfg = cfg["backtest"]["tax"]
    tax_amount_krw = (
        tax.capital_gains_tax(total_gain, tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100)
        if broker.apply_tax
        else 0.0
    )
    spread_pct = cfg["backtest"]["costs"]["fx_spread_pct"] if broker.apply_costs else 0.0
    effective_fx = fx_rate * (1 - spread_pct / 100) if fx_rate else None
    tax_amount_usd = (tax_amount_krw / effective_fx) if effective_fx else 0.0

    cash_after_usd = broker.cash_usd + liq_broker.cash_usd - tax_amount_usd
    return cash_after_usd, tax_amount_krw


def _stock_unrealized_gain_krw(states: dict, indicator_map: dict, ts: pd.Timestamp, fx_rate: float, costs: dict, apply_costs: bool) -> tuple[float, float]:
    """지금 이 시점에 보유 종목을 전량 팔면 생길 실현손익(원화)과 받을 순 달러(수수료 반영)를
    계산한다 (순수 함수 — states/indicator_map을 건드리지 않는다). compute_liquidated_cagr와
    compute_stock_leg_gain_krw(P5-3 2장)가 공유한다."""
    gain_krw = 0.0
    proceeds_usd = 0.0
    for ticker, state_ in states.items():
        for unit, qty in state_["units"].items():
            if not qty or qty <= 0:
                continue
            df = indicator_map.get(ticker)
            if df is None or ts not in df.index:
                continue
            close = df.loc[ts, "close"]
            if pd.isna(close):
                continue
            entry_price = state_["entries"].get(unit)
            if entry_price is None:
                continue
            usd_amount = float(close) * qty
            commission_sell_pct = costs["commission_sell_pct"] if apply_costs else 0.0
            proceeds = usd_amount * (1 - commission_sell_pct / 100)
            commission_buy_pct = costs["commission_buy_pct"] if apply_costs else 0.0
            cost_usd = entry_price * qty * (1 + commission_buy_pct / 100)
            gain_krw += (proceeds - cost_usd) * fx_rate
            proceeds_usd += proceeds
    return gain_krw, proceeds_usd


def compute_stock_leg_gain_krw(trades: list, states: dict, data: BacktestData, cfg: dict, as_of: date, apply_costs: bool) -> float:
    """신호 종목 몫의 손익(원화) = as_of까지 실현된 손익 합 + as_of 시점 미실현 손익 (P5-3 2장,
    성과 기여 분해용). trades는 전체 기간 것을 줘도 된다(as_of 이후는 걸러낸다) — 단 states는
    반드시 as_of에서 끝난 시뮬레이션 결과의 것이어야 한다(그래야 "그 시점 보유분"이 맞는다).
    """
    as_of_iso = as_of.isoformat()
    realized = sum(t.get("pnl_krw") or 0 for t in trades if t["side"] == "청산" and t["date"] <= as_of_iso)
    ts = pd.Timestamp(as_of)
    fx_rate = data.fx_by_date.get(as_of_iso)
    if fx_rate is None:
        return realized
    unrealized, _ = _stock_unrealized_gain_krw(states, data.indicator_map, ts, fx_rate, cfg["backtest"]["costs"], apply_costs)
    return realized + unrealized


def compute_liquidated_cagr(result: BacktestResult, data: BacktestData, cfg: dict, start: date, end: date) -> dict:
    """B0(또는 실험)이 기간 끝에 보유 종목·QQQM을 전량 매도하고 그해 양도세까지 낸
    (a) 시나리오의 세후 CAGR을 계산한다 (P5-2 0-1번). result.broker/states는 건드리지 않는다.

    출력: {"cagr_liquidated_pct", "tax_paid_krw", "liquidated_value_krw"} 또는 계산할
         수 없으면(환율·가격 없음) 빈 dict
    """
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    ts = pd.Timestamp(end)
    fx_rate = data.fx_by_date.get(end.isoformat())
    price_now = _price_on_or_before(data.cash_etf_df, ts)
    if fx_rate is None or price_now is None:
        return {}

    costs = bt_cfg["costs"]
    broker = result.broker
    stock_realized_gain_krw, stock_proceeds_usd = _stock_unrealized_gain_krw(
        result.states, data.indicator_map, ts, fx_rate, costs, broker.apply_costs
    )

    liq_broker = Broker(
        cash_usd=broker.cash_usd + stock_proceeds_usd, qqqm_shares=broker.qqqm_shares, qqqm_cost_usd=broker.qqqm_cost_usd,
        realized_gain_by_year=defaultdict(float, broker.realized_gain_by_year), apply_costs=broker.apply_costs, apply_tax=broker.apply_tax,
    )
    liq_broker.realized_gain_by_year[end.year] += stock_realized_gain_krw
    cash_after_usd, tax_amount_krw = _liquidate_qqqm_and_pay_tax(liq_broker, price_now, fx_rate, cfg, end.year)
    liquidated_value_krw = cash_after_usd * fx_rate

    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    cagr = (liquidated_value_krw / total_krw) ** (1 / years) - 1 if years > 0 and liquidated_value_krw > 0 else -1.0
    return {"cagr_liquidated_pct": round(cagr * 100, 2), "tax_paid_krw": round(tax_amount_krw), "liquidated_value_krw": round(liquidated_value_krw)}


# ── 벤치마크(QQQ 매수 후 보유) ────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    equity_rows: list
    final_value_usd: float
    final_shares: float
    cost_basis_usd: float
    realized_gain_by_year: dict
    cagr_liquidated: float  # (a) 기간 끝에 전량 매도, 세금까지 낸 경우
    cagr_unrealized: float  # (b) 팔지 않은 경우(미실현, 자본이득세 아직 없음)
    mdd_pct: float


def simulate_benchmark(data: BacktestData, cfg: dict, start: date, end: date, apply_costs: bool = True, apply_tax: bool = True) -> BenchmarkResult:
    """QQQ를 첫날 전액 매수해 배당 재투자하며 보유한다. 세후는 (a) 청산 (b) 미청산 둘 다 계산.

    apply_costs/apply_tax(P5-3 5번)는 B0/실험과 같은 정의의 "세전"(수수료 반영, 세금 제외
    = apply_costs=True, apply_tax=False)을 QQQ 줄에도 똑같이 계산하려고 추가했다."""
    bt_cfg = cfg["backtest"]
    total_krw = bt_cfg["total_krw"]
    costs = bt_cfg["costs"]
    df = data.qqq_df
    trading_days = [d.date() for d in market_calendar.trading_days_between(start, end) if pd.Timestamp(d) in df.index]
    if not trading_days:
        raise ValueError("QQQ 시세에 백테스트 구간 데이터가 없습니다")

    broker = Broker(cash_usd=0.0, apply_costs=apply_costs, apply_tax=apply_tax)
    first_ts = pd.Timestamp(trading_days[0])
    fx0 = data.fx_by_date.get(trading_days[0].isoformat())
    price0 = float(df.loc[first_ts, "close"])
    usd0 = _deposit_krw_to_usd(total_krw, fx0, costs["fx_spread_pct"], apply_costs)
    broker.cash_usd += usd0
    broker.buy_qqqm(usd0, price0, cfg)

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
            broker.receive_dividend(gross_usd, cfg)
            if broker.cash_usd > 0:
                broker.buy_qqqm(broker.cash_usd, price, cfg)
        value_krw = broker.qqqm_shares * price * fx_rate + broker.cash_usd * fx_rate
        equity_rows.append({"date": date_.isoformat(), "total_krw": round(value_krw)})
        peak = value_krw if peak is None else max(peak, value_krw)
        if peak:
            mdd = min(mdd, value_krw / peak - 1)

    last_ts = pd.Timestamp(trading_days[-1])
    last_price = float(df.loc[last_ts, "close"])
    last_fx = data.fx_by_date.get(trading_days[-1].isoformat()) or fx0
    unrealized_value_krw = broker.qqqm_shares * last_price * last_fx + broker.cash_usd * last_fx

    years = (trading_days[-1] - trading_days[0]).days / 365.25
    cagr_unrealized = (unrealized_value_krw / total_krw) ** (1 / years) - 1 if years > 0 and unrealized_value_krw > 0 else -1.0

    # (a) 청산: 전량 매도 후 자본이득세까지 낸 경우. 파는 그 자체의 수수료·양도세를
    # 뺀 "실제 손에 쥐는 돈"만 남겨야 한다 — 원래 보유분(unrealized_value_krw)을
    # 다시 더하면 안 된다(이전 버전의 버그, P5-1.1 1번: (a)가 (b)보다 커지는 원인).
    cash_after_usd, tax_amount_krw = _liquidate_qqqm_and_pay_tax(broker, last_price, last_fx, cfg, trading_days[-1].year)
    liquidated_value_krw = cash_after_usd * last_fx
    cagr_liquidated = (liquidated_value_krw / total_krw) ** (1 / years) - 1 if years > 0 and liquidated_value_krw > 0 else -1.0

    return BenchmarkResult(
        equity_rows=equity_rows, final_value_usd=broker.qqqm_shares * last_price, final_shares=broker.qqqm_shares,
        cost_basis_usd=broker.qqqm_cost_usd, realized_gain_by_year=broker.realized_gain_by_year,
        cagr_liquidated=cagr_liquidated, cagr_unrealized=cagr_unrealized, mdd_pct=mdd * 100,
    )


# ── 연도별 수익률 표 (P5-1.1 1·6번) ──────────────────────────────────────────


def compute_yearly_returns_from_equity(equity_rows: list) -> dict[int, float]:
    """equity_rows(각 행에 date, total_krw)에서 달력 연도별 수익률(%)을 계산한다 (순수 함수).

    그해 첫 거래일 값 대비 마지막 거래일 값의 변화율. 연도 첫해는 시작일 값을 쓴다.
    """
    if not equity_rows:
        return {}
    by_year: dict[int, list] = defaultdict(list)
    for r in equity_rows:
        by_year[pd.Timestamp(r["date"]).year].append(r["total_krw"])
    years = sorted(by_year)
    out = {}
    prev_last = by_year[years[0]][0]
    for y in years:
        vals = by_year[y]
        start_v = prev_last
        end_v = vals[-1]
        out[y] = round((end_v / start_v - 1) * 100, 2) if start_v else None
        prev_last = end_v
    return out


# ── 시장 국면 필터 (P5-2 B1·B2) ──────────────────────────────────────────────


def compute_sma_regime(qqq_df: pd.DataFrame, window: int) -> dict[str, bool]:
    """QQQ 종가가 window일 단순이동평균 위인 날짜만 True (신규 매수 허용 국면, 순수 함수).

    입력: qqq_df(close 열 포함), window(이동평균 일수)
    출력: {날짜.isoformat(): bool} — 이동평균을 계산할 데이터가 모자란 날짜는 없음
    """
    sma = qqq_df["close"].rolling(window).mean()
    ok = qqq_df["close"] > sma
    return {d.date().isoformat(): bool(v) for d, v in ok.items() if not pd.isna(sma.loc[d])}


def compute_cloud_regime(qqq_df: pd.DataFrame) -> dict[str, bool]:
    """QQQ 종가가 구름 상단 위 + 앞구름 양운인 날짜만 True (신규 매수 허용 국면, 순수 함수).

    입력: qqq_df(core.indicators.compute_indicators 결과 — cloud_top·future_yang 포함)
    출력: {날짜.isoformat(): bool} — 구름을 계산할 데이터가 모자란 날짜는 없음
    """
    out = {}
    for d, row in qqq_df.iterrows():
        close = row.get("close")
        cloud_top = row.get("cloud_top")
        future_yang = row.get("future_yang")
        if pd.isna(close) or pd.isna(cloud_top) or pd.isna(future_yang):
            continue
        ok = bool(close > cloud_top and future_yang)
        out[d.date().isoformat()] = ok
    return out


def compute_regime_split_stats(equity_rows: list, positions: list[dict], regime_ok: dict) -> dict:
    """국면(regime_ok — 예: compute_sma_regime 결과)별로 B0(또는 실험)의 성과를 나눈다 (순수 함수, P5-2 0장).

    입력: equity_rows(날짜별 total_krw), positions(aggregate_positions 결과),
         regime_ok({날짜.isoformat(): bool}, True=국면 위)
    출력: {"above": {...}, "below": {...}} 각각 {"days", "cum_return_pct", "win_rate_pct",
         "expectancy_r", "position_count"} — compute_position_stats를 그대로 재사용한다.
    각 날의 수익률은 그날 국면(전날 대비 오늘 등락)에 귀속시키고, 포지션은 진입일(opened_date)
    기준 국면으로 나눈다. regime_ok에 없는 날짜/포지션은 두 그룹 어디에도 넣지 않는다.
    """
    rows = sorted(equity_rows, key=lambda r: r["date"])
    above_factor = 1.0
    below_factor = 1.0
    above_days = below_days = 0
    for i in range(1, len(rows)):
        prev_v, cur_v = rows[i - 1]["total_krw"], rows[i]["total_krw"]
        if not prev_v:
            continue
        day_ret = cur_v / prev_v
        ok = regime_ok.get(rows[i]["date"])
        if ok is True:
            above_factor *= day_ret
            above_days += 1
        elif ok is False:
            below_factor *= day_ret
            below_days += 1

    above_positions = [p for p in positions if regime_ok.get(p["opened_date"]) is True]
    below_positions = [p for p in positions if regime_ok.get(p["opened_date"]) is False]
    above_stats = compute_position_stats(above_positions)
    below_stats = compute_position_stats(below_positions)

    return {
        "above": {
            "days": above_days, "cum_return_pct": round((above_factor - 1) * 100, 2),
            "win_rate_pct": above_stats.get("win_rate_pct"), "expectancy_r": above_stats.get("expectancy_r"),
            "position_count": above_stats.get("count", 0),
        },
        "below": {
            "days": below_days, "cum_return_pct": round((below_factor - 1) * 100, 2),
            "win_rate_pct": below_stats.get("win_rate_pct"), "expectancy_r": below_stats.get("expectancy_r"),
            "position_count": below_stats.get("count", 0),
        },
    }


def compute_topN_excluded_cagr(positions: list[dict], final_total_krw: float, total_krw: float, start: date, end: date, n: int = 10) -> dict:
    """pnl_krw 상위 n개 포지션을 뺀 최종 평가액으로 CAGR을 근사한다 (순수 함수, P5-2 0-4번).

    근사다: 상위 포지션의 이익을 단순히 최종 평가액에서 빼고 다시 CAGR을 계산할 뿐,
    그 자금이 실제로 재투자됐을 경로(복리 효과)는 무시한다.
    출력: {"cagr_pct", "excluded_sum_pnl_krw", "excluded_tickers"}
    """
    top = sorted(positions, key=lambda p: p["pnl_krw"], reverse=True)[:n]
    excluded_sum = sum(p["pnl_krw"] for p in top)
    adjusted_final = final_total_krw - excluded_sum
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    cagr = (adjusted_final / total_krw) ** (1 / years) - 1 if years > 0 and adjusted_final > 0 else -1.0
    return {
        "cagr_pct": round(cagr * 100, 2),
        "excluded_sum_pnl_krw": round(excluded_sum),
        "excluded_tickers": [p["ticker"] for p in top],
    }


def compute_relative_strength_top_half(indicator_map: dict, trading_days: list, months: int) -> dict[str, set[str]]:
    """각 날짜마다 활성 종목들의 최근 months개월(약 21×months거래일) 수익률을 계산해
    상위 50%(중앙값 이상)에 드는 종목 집합을 돌려준다 (순수 함수, P5-3 C1).

    입력: indicator_map({ticker: df}), trading_days(date 목록), months(6 또는 12)
    출력: {날짜.isoformat(): {상위 50% 종목 집합}} — 데이터가 모자란 종목은 그날 순위 계산에서 빠진다
    """
    window = months * 21
    out: dict[str, set[str]] = {}
    for d in trading_days:
        ts = pd.Timestamp(d)
        rets = {}
        for ticker, df in indicator_map.items():
            if ts not in df.index:
                continue
            idx = df.index.get_loc(ts)
            if idx < window:
                continue
            close_now = df.iloc[idx]["close"]
            close_then = df.iloc[idx - window]["close"]
            if pd.isna(close_now) or pd.isna(close_then) or close_then == 0:
                continue
            rets[ticker] = close_now / close_then - 1
        if not rets:
            out[d.isoformat()] = set()
            continue
        threshold = statistics.median(rets.values())
        out[d.isoformat()] = {t for t, r in rets.items() if r >= threshold}
    return out


def compute_rsi_whipsaw_block(indicator_map: dict, lookback: int = 15, max_crosses: int = 2) -> dict[tuple[str, str], bool]:
    """종목·날짜별로 최근 lookback거래일 내 RSI가 30을 상향 돌파한 횟수가 max_crosses
    이상이면 True(그날 A1 제외 대상) (순수 함수, P5-3 C2).

    출력: {(ticker, 날짜.isoformat()): bool}
    """
    out: dict[tuple[str, str], bool] = {}
    for ticker, df in indicator_map.items():
        if "rsi" not in df.columns:
            continue
        rsi = df["rsi"]
        cross_up = (rsi > 30) & (rsi.shift(1) <= 30)
        cross_count = cross_up.rolling(lookback, min_periods=1).sum()
        blocked = cross_count >= max_crosses
        for d, v in blocked.items():
            out[(ticker, d.date().isoformat())] = bool(v)
    return out


def compute_volume_entry_filter(indicator_map: dict, mode: str) -> dict[tuple[str, str], bool]:
    """종목·날짜별 거래량 조건 통과 여부 (순수 함수, P5-3 C3).

    mode="volume": 그날 거래량 > 20일 평균 × 1.5 (compute_indicators의 vol_ratio 열 재사용).
    mode="obv": OBV가 20거래일 전보다 높음 (core.indicators.compute_obv로 새로 계산).
    출력: {(ticker, 날짜.isoformat()): bool}
    """
    out: dict[tuple[str, str], bool] = {}
    for ticker, df in indicator_map.items():
        if mode == "volume":
            if "vol_ratio" not in df.columns:
                continue
            ok = df["vol_ratio"] > 1.5
        elif mode == "obv":
            from core.indicators import compute_obv

            obv = compute_obv(df)
            ok = obv > obv.shift(20)
        else:
            raise ValueError(f"알 수 없는 거래량 조건 mode: {mode}")
        for d, v in ok.items():
            out[(ticker, d.date().isoformat())] = bool(v) if not pd.isna(v) else False
    return out


def compute_stock_sleeve_twr_usd(equity_rows: list, trades: list, fx_by_date: dict, annualize_basis: int = 252) -> dict:
    """신호 종목 몫(positions_value_krw)만의 일별 수익률을 모디파이드 다이츠 방식(그날의
    순매수·순매도를 뺀 진짜 가격 변화분만)으로 계산해 이어붙이고 연율화한다 (순수 함수,
    P5-3 2장). 달러 기준으로 계산해 환율 변동을 분리한다(포트폴리오 전체가 달러 계좌라
    환율 효과는 종목·QQQM 어느 쪽에 있든 똑같이 걸리므로, "종목 선택"만 보려면 빼는 게 맞다).
    종목에 아무것도 안 들어 있던 날(전날 보유가치 0)은 건너뛴다.

    출력: {"twr_annualized_pct", "invested_days", "daily_returns": {날짜: 수익률}}
    """
    rows = sorted(equity_rows, key=lambda r: r["date"])
    buys_by_date: dict = defaultdict(float)
    sells_by_date: dict = defaultdict(float)
    for t in trades:
        amount = (t.get("price") or 0) * (t.get("qty") or 0)
        if t["side"] == "진입":
            buys_by_date[t["date"]] += amount
        elif t["side"] == "청산":
            sells_by_date[t["date"]] += amount

    daily_returns: dict = {}
    factor = 1.0
    for i in range(1, len(rows)):
        d = rows[i]["date"]
        fx_prev = fx_by_date.get(rows[i - 1]["date"])
        fx_cur = fx_by_date.get(d)
        if not fx_prev or not fx_cur:
            continue
        prev_v_usd = rows[i - 1]["positions_value_krw"] / fx_prev
        cur_v_usd = rows[i]["positions_value_krw"] / fx_cur
        if prev_v_usd <= 0:
            continue
        numerator = cur_v_usd - prev_v_usd - buys_by_date.get(d, 0.0) + sells_by_date.get(d, 0.0)
        day_ret = numerator / prev_v_usd
        daily_returns[d] = day_ret
        factor *= 1 + day_ret

    n = len(daily_returns)
    twr = (factor ** (annualize_basis / n) - 1) if n > 0 else None
    return {"twr_annualized_pct": round(twr * 100, 2) if twr is not None else None, "invested_days": n, "daily_returns": daily_returns}


def compute_qqq_twr_over_days_usd(qqq_df: pd.DataFrame, qqq_dividends: pd.Series, invested_days: set, annualize_basis: int = 252) -> dict:
    """신호 종목이 실제로 물려 있던 날짜 집합(compute_stock_sleeve_twr_usd의 daily_returns 키)에
    대해서만, QQQ 자체의 그날그날 수익률(배당 재투자, 달러)을 이어붙여 연율화한다 (순수 함수,
    P5-3 2장 — "같은 날들에 QQQ를 들고 있었다면"과 비교하기 위한 기준선).
    """
    factor = 1.0
    n = 0
    prev_val = None
    for ts in qqq_df.index:
        close = qqq_df.loc[ts, "close"]
        if pd.isna(close):
            continue
        div = float(qqq_dividends.loc[ts]) if ts in qqq_dividends.index else 0.0
        val = float(close) + div
        if prev_val is not None and prev_val > 0:
            day_ret = val / prev_val - 1
            if ts.date().isoformat() in invested_days:
                factor *= 1 + day_ret
                n += 1
        prev_val = val
    twr = (factor ** (annualize_basis / n) - 1) if n > 0 else None
    return {"twr_annualized_pct": round(twr * 100, 2) if twr is not None else None, "days": n}


def compute_position_weight_stats(equity_rows: list) -> dict:
    """일별 신호 종목 비중(총자금 대비 %)의 평균·최대·최소 (순수 함수, P5-3 2장)."""
    weights = [r["positions_value_krw"] / r["total_krw"] for r in equity_rows if r.get("total_krw")]
    if not weights:
        return {}
    return {
        "avg_weight_pct": round(statistics.mean(weights) * 100, 2),
        "max_weight_pct": round(max(weights) * 100, 2),
        "min_weight_pct": round(min(weights) * 100, 2),
    }


def compute_entry_type_breakdown(positions: list[dict], trades: list[dict]) -> dict:
    """진입 유형(A형=A1로 시작 / B형=B로 시작)별 포지션 수·승률·기대값 R·총손익 (순수 함수,
    P5-3 2장)."""
    stage_by_position = {}
    for t in trades:
        if t["side"] == "진입" and t.get("initial_risk_krw") is not None:
            stage_by_position[t["position_id"]] = t["stage"]
    out = {}
    for label, stage in (("A형", "A1"), ("B형", "B")):
        subset = [p for p in positions if stage_by_position.get(p["position_id"]) == stage]
        stats = compute_position_stats(subset)
        out[label] = {
            "count": stats.get("count", 0), "win_rate_pct": stats.get("win_rate_pct"),
            "expectancy_r": stats.get("expectancy_r"), "total_pnl_krw": round(sum(p["pnl_krw"] for p in subset)),
        }
    return out


def compute_exit_type_breakdown(trades: list[dict]) -> dict:
    """청산 유형(손절/E1/E2/E3/1차 만료 등, trades의 stage 값)별 건수·평균 R (순수 함수,
    P5-3 2장). R은 그 청산 건의 pnl_krw ÷ 그 포지션의 initial_risk_krw(부분 청산이면
    그 부분의 R 기여)."""
    risk_by_position = {}
    for t in trades:
        if t["side"] == "진입" and t.get("initial_risk_krw"):
            risk_by_position[t["position_id"]] = t["initial_risk_krw"]
    agg: dict = defaultdict(lambda: {"count": 0, "r_sum": 0.0, "r_n": 0})
    for t in trades:
        if t["side"] != "청산":
            continue
        kind = t.get("stage")
        entry = agg[kind]
        entry["count"] += 1
        risk = risk_by_position.get(t.get("position_id"))
        if risk and t.get("pnl_krw") is not None:
            entry["r_sum"] += t["pnl_krw"] / risk
            entry["r_n"] += 1
    out = {}
    for kind, v in agg.items():
        avg_r = v["r_sum"] / v["r_n"] if v["r_n"] else None
        out[kind] = {"count": v["count"], "avg_r": round(avg_r, 2) if avg_r is not None else None}
    return out


def compute_pretax_benchmark_yearly_returns(qqq_df: pd.DataFrame, qqq_dividends: pd.Series, fx_by_date: dict, start: date, end: date) -> dict:
    """세전(비용·세금 없음) QQQ 연도별 수익률을 달러 가격만/달러 총수익(배당 재투자)/원화
    총수익 세 가지로 계산한다 (P5-1.1 1번 검산용, 순수 함수).
    """
    trading_days = [d for d in market_calendar.trading_days_between(start, end) if pd.Timestamp(d) in qqq_df.index]
    if not trading_days:
        return {}

    price_series = []
    tr_series = []  # 배당 재투자 총수익 지수(달러)
    krw_series = []
    tr_value = 1.0
    prev_close = None
    for d in trading_days:
        ts = pd.Timestamp(d)
        close = float(qqq_df.loc[ts, "close"])
        if prev_close is not None:
            div = float(qqq_dividends.loc[ts]) if ts in qqq_dividends.index else 0.0
            tr_value *= (close + div) / prev_close
        prev_close = close
        fx_rate = fx_by_date.get(d.date().isoformat())
        price_series.append((d, close))
        tr_series.append((d, tr_value))
        krw_series.append((d, tr_value * (fx_rate or float("nan"))))

    def _yearly(series):
        by_year = defaultdict(list)
        for d, v in series:
            by_year[d.year].append(v)
        years = sorted(by_year)
        out = {}
        prev_last = by_year[years[0]][0]
        for y in years:
            vals = by_year[y]
            out[y] = round((vals[-1] / prev_last - 1) * 100, 2) if prev_last else None
            prev_last = vals[-1]
        return out

    return {
        "usd_price_only": _yearly(price_series),
        "usd_total_return": _yearly(tr_series),
        "krw_total_return": _yearly(krw_series),
    }


# ── 생존 편향 영향 추정 (P5-1.1 5번) ─────────────────────────────────────────


def ticker_membership_trading_days(ticker: str, checkpoints: list, start: date, end: date) -> int:
    """checkpoints(membership_checkpoints 결과)에서 ticker가 [start, end] 동안 구성
    종목이었던 거래일 수를 센다 (순수 함수)."""
    total = 0
    for i, (cp_date, members) in enumerate(checkpoints):
        if ticker not in members:
            continue
        seg_start = max(cp_date, start)
        seg_end_exclusive = checkpoints[i + 1][0] if i + 1 < len(checkpoints) else end + timedelta(days=1)
        seg_end = min(seg_end_exclusive - timedelta(days=1), end)
        if seg_start > seg_end:
            continue
        total += len(market_calendar.trading_days_between(seg_start, seg_end))
    return total


def estimate_survivorship_impact(data: BacktestData, entries_count: int, start: date, end: date) -> dict:
    """실패 종목이 구성 종목이었던 기간 비율과, 그 기간에 놓쳤을 신호 수를 대략 추정한다."""
    if not data.checkpoints or data.universe_mode != "POINT_IN_TIME":
        return {}
    all_tickers = set(data.all_needed_tickers)
    failed = set(data.failed_tickers)
    successful = all_tickers - failed

    failed_days = sum(ticker_membership_trading_days(t, data.checkpoints, start, end) for t in failed)
    success_days = sum(ticker_membership_trading_days(t, data.checkpoints, start, end) for t in successful)
    total_days = failed_days + success_days
    if total_days == 0:
        return {}

    signal_rate_per_ticker_day = entries_count / success_days if success_days else 0.0
    estimated_missed_signals = round(signal_rate_per_ticker_day * failed_days)

    return {
        "failed_ticker_count": len(failed),
        "total_ticker_count": len(all_tickers),
        "failed_ticker_days": failed_days,
        "total_ticker_days": total_days,
        "failed_days_pct": round(failed_days / total_days * 100, 1),
        "estimated_missed_signals": estimated_missed_signals,
    }


def write_missing_tickers_csv(out_dir: Path, data: BacktestData, start: date, end: date) -> None:
    rows = []
    for ticker, reason in data.failed_tickers.items():
        days = ticker_membership_trading_days(ticker, data.checkpoints, start, end) if data.checkpoints else None
        rows.append({"ticker": ticker, "reason": reason, "membership_trading_days": days})
    pd.DataFrame(rows).sort_values("membership_trading_days", ascending=False, na_position="last").to_csv(
        out_dir / "missing_tickers.csv", index=False, encoding="utf-8-sig"
    )


# ── 지표 계산 ────────────────────────────────────────────────────────────────


def compute_equity_metrics(equity_rows: list) -> dict:
    """CAGR·MDD·샤프 등 equity 곡선 지표만 계산한다 (거래 품질 지표는 별도, P5-1.1 2번)."""
    if not equity_rows:
        return {}
    values = [r["total_krw"] for r in equity_rows]
    dates = [pd.Timestamp(r["date"]) for r in equity_rows]
    start_v, end_v = values[0], values[-1]
    years = (dates[-1] - dates[0]).days / 365.25

    cagr = (end_v / start_v) ** (1 / years) - 1 if years > 0 and start_v > 0 and end_v > 0 else None

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
        after = [(v, d) for v, d in zip(values, dates) if d > mdd_end]
        for v, d in after:
            if v >= peak:
                recovery_days = (d - mdd_end).days
                break

    daily_returns = [(values[i] / values[i - 1] - 1) for i in range(1, len(values)) if values[i - 1]]
    sharpe = None
    if len(daily_returns) > 1 and statistics.pstdev(daily_returns) > 0:
        sharpe = statistics.mean(daily_returns) / statistics.pstdev(daily_returns) * (252 ** 0.5)

    return {
        "cagr_pct": round(cagr * 100, 2) if cagr is not None else None,
        "mdd_pct": round(mdd * 100, 2),
        "mdd_start": mdd_start.date().isoformat() if mdd_start is not None else None,
        "mdd_end": mdd_end.date().isoformat() if mdd_end is not None else None,
        "mdd_recovery_days": recovery_days,
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
    }


# ── 출력 ────────────────────────────────────────────────────────────────────


def make_run_id(cfg: dict, start: date, end: date) -> str:
    payload = json.dumps({"cfg": cfg.get("backtest"), "plan": cfg.get("plan"), "start": str(start), "end": str(end)}, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{date.today().isoformat()}_{digest}"


def _fmt_yearly(table: dict) -> str:
    return " · ".join(f"{y}: {v:+.1f}%" if v is not None else f"{y}: -" for y, v in sorted(table.items()))


def write_outputs(
    run_id: str, cfg: dict, result: BacktestResult, benchmark: BenchmarkResult, data: BacktestData,
    equity_metrics: dict, position_stats: dict, decomposition: dict, survivorship: dict,
    pretax_yearly: dict, b0_yearly: dict, qqq_yearly: dict, start: date, end: date,
    b0_liquidated: dict | None = None,
) -> Path:
    b0_liquidated = b0_liquidated or {}
    out_dir = OUTPUT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(result.equity_rows).to_csv(out_dir / "equity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result.trades).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result.rejected).to_csv(out_dir / "rejected.csv", index=False, encoding="utf-8-sig")
    positions = aggregate_positions(result.trades, result.still_open_position_ids)
    pd.DataFrame(positions).to_csv(out_dir / "positions.csv", index=False, encoding="utf-8-sig")
    write_missing_tickers_csv(out_dir, data, start, end)
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

    qqq_tax_line = ""
    if decomposition:
        qqq_tax_line = f"- QQQM 반복 매매로 생긴 양도세(추정): {decomposition.get('qqqm_attributable_tax_krw', 0):,.0f}원"

    lines = [
        f"# 백테스트 결과 — {run_id}",
        "",
        f"기간: {start} ~ {end} (학습 구간, 검증 구간 2022- 봉인)",
        f"유니버스: {data.universe_mode} · SURVIVORSHIP_BIAS = {data.survivorship_bias}",
        f"data_gap 종목: {len(data.data_gap)}개, 시세 실패 종목: {len(data.failed_tickers)}개 (목록: missing_tickers.csv)",
        "실적 필터 미적용: 과거 시점별 실적 발표일 데이터가 없어 실적 임박 필터는 백테스트에서 항상 통과 처리됨",
        "",
        "## 벤치마크 — 세전 연도별 수익률 (참고 검산: 달러 대략 2015 +9% 2016 +7% 2017 +33% 2018 0% 2019 +39% 2020 +49% 2021 +27%, CAGR 약 22%)",
        f"- 달러 가격만: {_fmt_yearly(pretax_yearly.get('usd_price_only', {}))}",
        f"- 달러 총수익(배당 재투자): {_fmt_yearly(pretax_yearly.get('usd_total_return', {}))}",
        f"- 원화 총수익: {_fmt_yearly(pretax_yearly.get('krw_total_return', {}))}",
        "",
        "## 벤치마크 (QQQ 매수 후 보유, 배당 재투자, 비용 반영 — 세후)",
        f"- 세후(b, 미청산 — 자본이득세는 아직 안 냄) CAGR: {benchmark.cagr_unrealized * 100:.2f}%",
        f"- 세후(a, 기간 끝 전량 청산해 자본이득세까지 낸 경우) CAGR: {benchmark.cagr_liquidated * 100:.2f}%",
        f"- MDD: {benchmark.mdd_pct:.2f}%",
        f"- 연도별(세후, 총액 기준): {_fmt_yearly(qqq_yearly)}",
        "",
        "## B0 (전략 v3 + QQQM + 슬롯 8 + 전략한도 60% + 4,000만 원, 세후)",
        f"- 세후(b, 미청산) CAGR: {equity_metrics.get('cagr_pct')}%",
        f"- 세후(a, 기간 끝 전량 청산해 자본이득세까지 낸 경우) CAGR: {b0_liquidated.get('cagr_liquidated_pct')}% (세금 {b0_liquidated.get('tax_paid_krw'):,}원)" if b0_liquidated else "- 세후(a): 계산 불가",
        f"- QQQ 대비(세후 b 기준): {(equity_metrics.get('cagr_pct') or 0) - benchmark.cagr_unrealized * 100:.2f}%p",
        f"- QQQ 대비(세후 a 기준): {(b0_liquidated.get('cagr_liquidated_pct') or 0) - benchmark.cagr_liquidated * 100:.2f}%p" if b0_liquidated else "",
        f"- MDD: {equity_metrics.get('mdd_pct')}% ({equity_metrics.get('mdd_start')} ~ {equity_metrics.get('mdd_end')}, 회복 {equity_metrics.get('mdd_recovery_days')}일)",
        f"- 샤프: {equity_metrics.get('sharpe')}",
        f"- 연도별: {_fmt_yearly(b0_yearly)}",
        "",
        "## B0 거래 품질 (포지션 단위, P5-1.1 2번 — 부분 청산을 거래로 세지 않음)",
        f"- 청산 완료 포지션 수: {position_stats.get('count')} (진입 신호 {entries}건, 아직 열려 있는 포지션 {len(result.still_open_position_ids)}개는 제외)",
        f"- 승률: {position_stats.get('win_rate_pct')}%",
        f"- 평균 이익: {position_stats.get('avg_win_r')}R · 평균 손실: {position_stats.get('avg_loss_r')}R · 최대 손실: {position_stats.get('max_loss_r')}R",
        f"- 손익비: {position_stats.get('payoff_ratio')}",
        f"- 기대값(R) = 승률×평균이익 − 패율×|평균손실|: {position_stats.get('expectancy_r')}",
        f"- Profit Factor: {position_stats.get('profit_factor')}",
        f"- 최악의 포지션: {position_stats.get('worst_position_ticker')} {position_stats.get('worst_position_krw')}원",
        f"- 월평균 신규 진입: {avg_monthly_signals}건",
        "",
        "## 수익 저하 원인 분해 (P5-1.1 4번, 세 시나리오 모두 같은 신호·수량 — 현금 흐름만 다름)",
        f"- ① 비용·세금 없음: {decomposition.get('no_cost_no_tax_cagr_pct')}%",
        f"- ② 비용만(수수료+최초 환전 스프레드): {decomposition.get('cost_only_cagr_pct')}%",
        f"- ③ 비용+세금(= 위 B0 세후 수치와 같아야 함): {decomposition.get('cost_and_tax_cagr_pct')}%",
        qqq_tax_line,
        "",
        "## 생존 편향 영향 추정",
        f"- 실패 종목 {survivorship.get('failed_ticker_count')}개 / 전체 {survivorship.get('total_ticker_count')}개",
        f"- 실패 종목의 구성 종목 재직 일수 비율: {survivorship.get('failed_days_pct')}% (실패 {survivorship.get('failed_ticker_days')}일 / 전체 {survivorship.get('total_ticker_days')}일)",
        f"- 놓쳤을 신호 수(대략, 성공 종목의 신호 발생률로 추정): {survivorship.get('estimated_missed_signals')}건",
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

    print("[backtest] B0 포트폴리오 시뮬레이션 중 (③ 비용+세금 — 기준)...")
    result = simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=True)
    equity_metrics = compute_equity_metrics(result.equity_rows)
    positions = aggregate_positions(result.trades, result.still_open_position_ids)
    position_stats = compute_position_stats(positions)

    print("[backtest] 원인 분해 시뮬레이션 중 (① 무비용무세금, ② 비용만)...")
    result_nocost = simulate_portfolio(data, cfg, start, end, apply_costs=False, apply_tax=False)
    metrics_nocost = compute_equity_metrics(result_nocost.equity_rows)
    result_costonly = simulate_portfolio(data, cfg, start, end, apply_costs=True, apply_tax=False)
    metrics_costonly = compute_equity_metrics(result_costonly.equity_rows)

    qqqm_gain_total = sum(result.broker.qqqm_realized_gain_by_year.values())
    non_qqqm_gain_total = sum(
        result.broker.realized_gain_by_year.get(y, 0) - result.broker.qqqm_realized_gain_by_year.get(y, 0)
        for y in result.broker.realized_gain_by_year
    )
    tax_cfg = cfg["backtest"]["tax"]
    tax_on_total = sum(t["tax_krw"] for t in result.broker.tax_log)
    tax_on_non_qqqm_only = sum(
        tax.capital_gains_tax(
            result.broker.realized_gain_by_year.get(y, 0) - result.broker.qqqm_realized_gain_by_year.get(y, 0),
            tax_cfg["capital_gains_deduction_krw"], tax_cfg["capital_gains_rate_pct"] / 100,
        )
        for y in result.broker.realized_gain_by_year
    )
    decomposition = {
        "no_cost_no_tax_cagr_pct": compute_equity_metrics(result_nocost.equity_rows).get("cagr_pct"),
        "cost_only_cagr_pct": metrics_costonly.get("cagr_pct"),
        "cost_and_tax_cagr_pct": equity_metrics.get("cagr_pct"),
        "qqqm_attributable_tax_krw": round(tax_on_total - tax_on_non_qqqm_only),
    }

    print("[backtest] 벤치마크(QQQ 매수 후 보유) 시뮬레이션 중...")
    benchmark = simulate_benchmark(data, cfg, start, end)
    benchmark_lookup_test_ab = benchmark.cagr_liquidated <= benchmark.cagr_unrealized + 1e-9

    pretax_yearly = compute_pretax_benchmark_yearly_returns(data.qqq_df, data.qqq_dividends, data.fx_by_date, start, end)
    qqq_yearly = compute_yearly_returns_from_equity(benchmark.equity_rows)
    b0_yearly = compute_yearly_returns_from_equity(result.equity_rows)

    entries_count = sum(1 for t in result.trades if t.get("side") == "진입")
    survivorship = estimate_survivorship_impact(data, entries_count, start, end)

    b0_liquidated = compute_liquidated_cagr(result, data, cfg, start, end)

    run_id = make_run_id(cfg, start, end)
    out_dir = write_outputs(
        run_id, cfg, result, benchmark, data, equity_metrics, position_stats, decomposition, survivorship,
        pretax_yearly, b0_yearly, qqq_yearly, start, end, b0_liquidated=b0_liquidated,
    )
    print(f"[backtest] (a)<=(b) 검산: {'OK' if benchmark_lookup_test_ab else 'FAIL'}")
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
