"""engine/kjb_satellite.py 테스트 (docs/kjb1_instructions.md 3번). 네트워크 없음
(DTB3 fetch를 몽키패치) — day-loop 규칙(섹터당 최대 3종목, 최대 10종목, 재진입
20거래일 금지)을 확인한다. 실제 김종봉 신호 판정은 core/kjb.py에서 이미 테스트했으니
여기서는 candidate_fn을 직접 주입해 엔진 메커니즘만 분리해서 검증한다.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
import yaml

from engine import kjb_satellite as ks
from engine.backtest import BacktestData


@pytest.fixture
def kjb_cfg():
    return {
        "dollar_volume_multiplier_min": 2.0,
        "big_candle_pct_min": 5.0,
        "relative_return_window_days": 63,
        "first_score_window_days": 60,
        "hold_trading_days": 63,
        "cooldown_trading_days": 20,
        "max_concurrent_positions": 10,
        "max_positions_per_sector": 3,
        "per_position_fraction_of_satellite": 0.1,
    }


@pytest.fixture
def cfg():
    with open("config.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    base["backtest"]["total_krw"] = 30_000_000
    return base


def _flat_df(n: int, price: float = 50.0) -> pd.DataFrame:
    index = pd.bdate_range("2019-06-01", periods=n, name="date")
    return pd.DataFrame(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 1_000_000.0},
        index=index,
    )


def _make_data(tickers: list[str], n: int) -> BacktestData:
    indicator_map = {t: _flat_df(n) for t in tickers}
    index = indicator_map[tickers[0]].index
    fx_by_date = {d.date().isoformat(): 1300.0 for d in index}
    return BacktestData(
        indicator_map=indicator_map, dividends={t: pd.Series(dtype=float) for t in tickers},
        checkpoints=[], fx_by_date=fx_by_date, universe_mode="CURRENT_CONSTITUENTS",
        survivorship_bias="FALSE", failed_tickers={}, data_gap={t: [] for t in tickers},
        qqq_df=_flat_df(n), qqq_dividends=pd.Series(dtype=float),
        cash_etf_df=_flat_df(n), cash_etf_dividends=pd.Series(dtype=float),
    )


def _signals_from_data(data: BacktestData) -> dict[str, ks.TickerSignals]:
    out = {}
    for ticker, df in data.indicator_map.items():
        idx = df.index
        false_series = pd.Series(False, index=idx)
        num_series = pd.Series(0.0, index=idx)
        out[ticker] = ks.TickerSignals(
            close=df["close"], open_=df["open"], low=df["low"], volume=df["volume"],
            rel_return=num_series, first_excess=false_series, first_score=num_series,
            dv_multiplier=num_series, big_candle=false_series, entry_signal=false_series,
        )
    return out


@pytest.fixture(autouse=True)
def _no_network_dtb3(monkeypatch):
    monkeypatch.setattr(ks, "fetch_fred_series_range", lambda series_id, start, end: {"2000-01-01": 5.0})


def _always_all_fn(date_, active, eligible):
    return list(eligible)


def test_max_10_concurrent_positions_enforced(kjb_cfg, cfg):
    n = 40
    tickers = [f"T{i}" for i in range(15)]
    data = _make_data(tickers, n)
    sector_by_ticker = {t: "Tech" if i % 2 == 0 else "Health" for i, t in enumerate(tickers)}
    # 섹터 한도(3)에 안 걸리게 섹터를 여러 개로 늘린다 -> 최대 보유(10)만 걸리게.
    sector_by_ticker = {t: f"Sector{i % 6}" for i, t in enumerate(tickers)}
    signals = _signals_from_data(data)

    start, end = date(2019, 6, 3), data.indicator_map[tickers[0]].index[-1].date()
    result, admission_log = ks._simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, _always_all_fn)

    entries = [t for t in result.trades if t["side"] == "진입"]
    # 한 번에 살 수 있는 후보가 15개라도, 동시 보유는 10을 넘지 않아야 한다(초반 하루 이틀 동안 매수 신호가
    # 몰려도 최대 보유 한도로 걸러진다).
    max_ever_held = 0
    held = set()
    for t in sorted(entries, key=lambda r: r["date"]):
        held.add(t["ticker"])
        max_ever_held = max(max_ever_held, len(held))
    assert max_ever_held <= kjb_cfg["max_concurrent_positions"]


def test_max_3_per_sector_enforced(kjb_cfg, cfg):
    n = 40
    tickers = [f"T{i}" for i in range(6)]
    data = _make_data(tickers, n)
    sector_by_ticker = {t: "Tech" for t in tickers}  # 전부 같은 섹터
    signals = _signals_from_data(data)

    start, end = date(2019, 6, 3), data.indicator_map[tickers[0]].index[-1].date()
    result, admission_log = ks._simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, _always_all_fn)

    entries = [t["ticker"] for t in result.trades if t["side"] == "진입"]
    assert len(set(entries)) <= 3  # 같은 섹터라 3종목을 넘길 수 없다(63일 안에 청산도 없음)


def test_reentry_blocked_within_20_trading_days_after_exit(kjb_cfg, cfg):
    """청산 체결일로부터 20거래일 안에는 같은 종목을 다시 사지 않는다."""
    n = 120
    tickers = ["A"]
    data = _make_data(tickers, n)
    sector_by_ticker = {"A": "Tech"}
    signals = _signals_from_data(data)
    kjb_cfg = {**kjb_cfg, "hold_trading_days": 5}  # 빨리 청산시켜 재진입 창을 시험한다

    call_dates = []

    def _fn(date_, active, eligible):
        call_dates.append((date_, list(eligible)))
        return list(eligible)  # "A"가 살 수 있을 때마다 항상 시도

    start, end = date(2019, 6, 3), data.indicator_map["A"].index[-1].date()
    result, admission_log = ks._simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, _fn)

    entry_dates = sorted(t["date"] for t in result.trades if t["side"] == "진입")
    assert len(entry_dates) >= 2  # 재진입이 결국 한 번은 일어나야 시험이 의미 있다

    exit_dates = sorted(t["date"] for t in result.trades if t["side"] == "청산")
    idx = data.indicator_map["A"].index
    first_exit_pos = idx.get_indexer([pd.Timestamp(exit_dates[0])])[0]
    second_entry_pos = idx.get_indexer([pd.Timestamp(entry_dates[1])])[0]
    assert second_entry_pos - first_exit_pos >= kjb_cfg["cooldown_trading_days"]


def test_make_kjb_candidate_fn_detects_real_numpy_bool_entry_signal(kjb_cfg, cfg):
    """entry_signal 열은 pandas bool dtype(numpy.bool_)이다. `x is True`로 비교하면
    numpy.bool_(True)는 파이썬 bool 싱글턴이 아니라서 항상 거짓으로 판정돼 실제
    신호가 하나도 채택되지 않는 회귀가 있었다 — 실제 신호 경로(make_kjb_candidate_fn)를
    직접 태워서 다시 나지 않게 막는다."""
    n = 40
    tickers = ["A", "B"]
    data = _make_data(tickers, n)
    sector_by_ticker = {"A": "Tech", "B": "Health"}
    signals = _signals_from_data(data)

    signal_date = data.indicator_map["A"].index[5]
    entry_signal = pd.Series(False, index=data.indicator_map["A"].index)
    entry_signal.loc[signal_date] = True  # dtype=bool 열에 대입 -> numpy.bool_
    signals["A"].entry_signal = entry_signal
    signals["A"].first_score.loc[signal_date] = 5.0
    signals["A"].dv_multiplier.loc[signal_date] = 3.0

    candidate_fn = ks.make_kjb_candidate_fn(signals, sector_by_ticker)
    chosen = candidate_fn(signal_date.date(), ["A", "B"], ["A", "B"])
    assert chosen == ["A"]


def test_compute_liquidated_value_krw_matches_equity_when_all_cash(kjb_cfg, cfg):
    """보유 종목이 하나도 없을 때(현금·대기 자금뿐)는 청산해도 세금이 붙을 실현손익이
    없으므로 청산가치가 마지막 equity_rows의 total_krw와 같아야 한다."""
    n = 10
    tickers = ["A"]
    data = _make_data(tickers, n)
    sector_by_ticker = {"A": "Tech"}
    signals = _signals_from_data(data)  # entry_signal 전부 False -> 매매 전혀 없음

    start, end = date(2019, 6, 3), data.indicator_map["A"].index[-1].date()
    no_candidates_fn = lambda date_, active, eligible: []  # noqa: E731
    result, _ = ks._simulate(data, signals, sector_by_ticker, cfg, kjb_cfg, start, end, no_candidates_fn)

    liquidated = ks.compute_liquidated_value_krw(result, data, cfg, end)
    assert liquidated["tax_paid_krw"] == 0
    assert liquidated["liquidated_value_krw"] == pytest.approx(result.equity_rows[-1]["total_krw"], rel=0.01)


def test_random_entry_uses_same_dates_and_counts_as_real_admission_log(kjb_cfg, cfg):
    admission_log = [{"date": "2019-06-05", "tickers": ["A", "B"]}, {"date": "2019-06-10", "tickers": ["C"]}]
    tickers = ["A", "B", "C", "D", "E"]
    data = _make_data(tickers, 100)
    sector_by_ticker = {t: "Tech" for t in tickers}
    monkeypatched_signals = _signals_from_data(data)

    import engine.kjb_satellite as ks_mod

    original_build = ks_mod.build_ticker_signals
    ks_mod.build_ticker_signals = lambda *a, **k: monkeypatched_signals
    try:
        result = ks.simulate_random_entry(
            data, sector_by_ticker, cfg, kjb_cfg, date(2019, 6, 3), data.indicator_map["A"].index[-1].date(),
            admission_log, seed=42,
        )
    finally:
        ks_mod.build_ticker_signals = original_build

    entries_by_date: dict[str, int] = {}
    for t in result.trades:
        if t["side"] == "진입":
            # 체결일은 신호 다음날이라, 신호일 기준 건수만 확인하면 되므로 날짜별 총 매수 건수만 비교
            entries_by_date[t["date"]] = entries_by_date.get(t["date"], 0) + 1
    assert sum(entries_by_date.values()) == sum(len(row["tickers"]) for row in admission_log)
