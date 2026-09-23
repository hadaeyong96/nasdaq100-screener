"""P3.3: 사라진 거래일 탐지·복구, A1 안전장치 테스트. 네트워크 없이 돈다.

9/22 CSX·ODFL·PEP 사례(캐시에서 행이 통째로 사라져 A1이 9/23으로 밀리거나
사라짐)를 합성 데이터로 재현하고, 탐지 -> 복구 -> 신호 날짜 복원을 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import signals as sig
from core.indicators import compute_indicators
from data.prices import (
    CLOSE_SOURCE_HOURLY,
    CLOSE_SOURCE_YAHOO,
    _expand_with_next_trading_day,
    _restore_missing_rows_from_cache,
    find_mid_series_gaps,
    recover_gap_days,
)
from engine.daily import simulate_since
from core import state as st
from tests.conftest import make_synthetic_ohlcv


def _first_a1_date(df_ind: pd.DataFrame, start: int = 60, end_margin: int = 5) -> pd.Timestamp:
    """합성 지표 DataFrame에서 실제로 A1이 성립하는 첫 날짜를 찾는다 (수작업 RSI 계산 없이)."""
    idx = df_ind.index
    for i in range(start, len(df_ind) - end_margin):
        prev_rsi = df_ind["rsi"].iloc[i - 1]
        rsi = df_ind["rsi"].iloc[i]
        if sig.check_a1(prev_rsi, rsi):
            return idx[i]
    raise AssertionError("synthetic 데이터에서 A1 신호를 하나도 찾지 못함 - seed를 바꿔야 함")


# ── 1. 누락 거래일 탐지 (거래일 달력 기준) ───────────────────────────────


def test_find_mid_series_gaps_detects_fully_missing_row():
    """close가 NaN인 게 아니라 그 날짜 행 자체가 없어도 구멍으로 잡는다."""
    df = pd.DataFrame(
        {"close": [100.0, 101.0, 102.0]},
        index=pd.DatetimeIndex(["2026-09-18", "2026-09-21", "2026-09-23"], name="date"),
    )
    # 9/22(화)는 정상 거래일인데 인덱스에 없다.
    gaps = find_mid_series_gaps(df)
    assert gaps == [pd.Timestamp("2026-09-22")]


def test_find_mid_series_gaps_ignores_weekend_and_pre_listing():
    """주말은 원래 거래일이 아니라 구멍이 아니고, df 시작일 이전은 탐지 범위 밖이다."""
    df = pd.DataFrame(
        {"close": [100.0, 101.0]},
        index=pd.DatetimeIndex(["2026-09-18", "2026-09-21"], name="date"),  # 금요일 -> 월요일
    )
    assert find_mid_series_gaps(df) == []


# ── 2. 누락 거래일 복구 ────────────────────────────────────────────────


def test_recover_gap_days_uses_daily_provider_first():
    df = pd.DataFrame(
        {
            "open": [10.0, 12.0],
            "high": [10.5, 12.5],
            "low": [9.5, 11.5],
            "close": [10.2, 12.1],
            "volume": [100, 200],
            "close_source": [CLOSE_SOURCE_YAHOO, CLOSE_SOURCE_YAHOO],
        },
        index=pd.DatetimeIndex(["2026-09-18", "2026-09-23"], name="date"),
    )
    gap = pd.Timestamp("2026-09-21")

    def daily_provider(d):
        assert d == gap.date()
        return {"open": 11.0, "high": 11.6, "low": 10.6, "close": 11.2, "volume": 150}

    def hourly_provider(d):
        raise AssertionError("일봉이 성공하면 60분봉은 부르면 안 된다")

    out, recovered, unresolved, warnings = recover_gap_days(df, [gap], daily_provider, hourly_provider)

    assert recovered == [gap]
    assert unresolved == []
    assert list(out.index) == sorted([pd.Timestamp("2026-09-18"), gap, pd.Timestamp("2026-09-23")])
    assert out.loc[gap, "close"] == pytest.approx(11.2)
    assert out.loc[gap, "close_source"] == CLOSE_SOURCE_YAHOO


def test_recover_gap_days_falls_back_to_hourly_when_daily_fails():
    df = pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2], "volume": [100], "close_source": ["yahoo"]},
        index=pd.DatetimeIndex(["2026-09-18"], name="date"),
    )
    gap = pd.Timestamp("2026-09-21")

    out, recovered, unresolved, warnings = recover_gap_days(
        df,
        [gap],
        daily_provider=lambda d: None,
        hourly_provider=lambda d: {"open": 10.1, "high": 10.9, "low": 9.9, "close": 10.6, "volume": 77},
    )

    assert recovered == [gap]
    assert out.loc[gap, "close_source"] == CLOSE_SOURCE_HOURLY
    assert out.loc[gap, "close"] == pytest.approx(10.6)
    assert out.loc[gap, "volume"] == pytest.approx(77)


def test_recover_gap_days_unresolved_when_both_sources_fail():
    df = pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2], "volume": [100], "close_source": ["yahoo"]},
        index=pd.DatetimeIndex(["2026-09-18"], name="date"),
    )
    gap = pd.Timestamp("2026-09-21")

    out, recovered, unresolved, warnings = recover_gap_days(
        df, [gap], daily_provider=lambda d: None, hourly_provider=lambda d: None
    )

    assert recovered == []
    assert unresolved == [gap]
    assert gap not in out.index
    assert any("복구 실패" in w for w in warnings)


def test_recover_gap_days_rejects_value_outside_low_high_range():
    df = pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2], "volume": [100], "close_source": ["yahoo"]},
        index=pd.DatetimeIndex(["2026-09-18"], name="date"),
    )
    gap = pd.Timestamp("2026-09-21")

    out, recovered, unresolved, warnings = recover_gap_days(
        df,
        [gap],
        daily_provider=lambda d: {"open": 10.0, "high": 10.2, "low": 9.8, "close": 999.0, "volume": 1},
        hourly_provider=lambda d: None,
    )

    assert recovered == []
    assert unresolved == [gap]
    assert any("범위를 벗어나" in w for w in warnings)


# ── 신호 날짜 복원 재현 (CSX 9/22 시나리오) ────────────────────────────


def test_gap_then_recovery_restores_original_a1_date(cfg):
    """구멍이 생기면 그 날짜의 A1을 판정할 수 없고, 복구하면 원래 날짜로 돌아온다."""
    raw = make_synthetic_ohlcv(n=500, seed=3)
    raw = raw.copy()
    raw["close_source"] = CLOSE_SOURCE_YAHOO

    full_ind = compute_indicators(raw, cfg)
    target_date = _first_a1_date(full_ind)
    original_row = raw.loc[target_date]

    gapped = raw.drop(index=target_date)
    gaps = find_mid_series_gaps(gapped)
    assert gaps == [target_date]

    def daily_provider(d):
        if d == target_date.date():
            return {
                "open": float(original_row["open"]),
                "high": float(original_row["high"]),
                "low": float(original_row["low"]),
                "close": float(original_row["close"]),
                "volume": float(original_row["volume"]),
            }
        return None

    recovered, recovered_dates, unresolved, _ = recover_gap_days(
        gapped, gaps, daily_provider, hourly_provider=lambda d: None
    )
    assert recovered_dates == [target_date]
    assert unresolved == []

    recovered_ind = compute_indicators(recovered, cfg)
    pos = recovered_ind.index.get_loc(target_date)
    assert sig.check_a1(recovered_ind["rsi"].iloc[pos - 1], recovered_ind["rsi"].iloc[pos])
    # 복구된 지표 값이 원래 값과 같다 (구멍이 없었던 것처럼).
    assert recovered_ind.loc[target_date, "rsi"] == pytest.approx(full_ind.loc[target_date, "rsi"])


# ── 새로 받은 데이터가 비어도 과거 확정 봉이 지워지지 않음 ─────────────────


def test_restore_missing_rows_from_cache_protects_confirmed_bar():
    """캐시에는 확정 종가가 있는데 새로 받은 데이터에 그 날짜가 아예 없으면 되살린다."""
    cached = pd.DataFrame(
        {
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.5, 10.5],
            "close": [10.2, 11.2],
            "volume": [100, 200],
            "close_source": [CLOSE_SOURCE_YAHOO, CLOSE_SOURCE_YAHOO],
        },
        index=pd.DatetimeIndex(["2026-09-21", "2026-09-22"], name="date"),
    )
    # 새로 받은 데이터엔 9/22가 통째로 빠져 있다 (야후 응답 누락 흉내).
    fresh = pd.DataFrame(
        {
            "open": [10.0, 12.0],
            "high": [10.5, 12.5],
            "low": [9.5, 11.5],
            "close": [10.2, 12.1],
            "volume": [100, 300],
            "close_source": [CLOSE_SOURCE_YAHOO, CLOSE_SOURCE_YAHOO],
        },
        index=pd.DatetimeIndex(["2026-09-21", "2026-09-23"], name="date"),
    )

    out, restored = _restore_missing_rows_from_cache(fresh, cached)

    assert restored == 1
    assert pd.Timestamp("2026-09-22") in out.index
    assert out.loc[pd.Timestamp("2026-09-22"), "close"] == pytest.approx(11.2)
    assert list(out.index) == sorted(out.index)


def test_restore_missing_rows_noop_when_cache_lacks_the_date():
    fresh = pd.DataFrame(
        {"open": [10.0], "high": [10.5], "low": [9.5], "close": [10.2], "volume": [100], "close_source": ["yahoo"]},
        index=pd.DatetimeIndex(["2026-09-21"], name="date"),
    )
    out, restored = _restore_missing_rows_from_cache(fresh, None)
    assert restored == 0
    assert out.equals(fresh)


# ── 3. A1 안전장치: 직전 행이 직전 거래일이 아니면 신호 없음 + data_gap ────────


def test_expand_with_next_trading_day_flags_the_day_after_a_gap():
    index = pd.DatetimeIndex(["2026-09-18", "2026-09-21", "2026-09-23"], name="date")
    gap = pd.Timestamp("2026-09-22")  # 9/22는 없다(끝내 복구 못 함) -> 다음 존재하는 날짜는 9/23
    expanded = _expand_with_next_trading_day(index, [gap])
    assert expanded == [gap, pd.Timestamp("2026-09-23")]


def test_expand_with_next_trading_day_empty_when_no_gaps():
    index = pd.DatetimeIndex(["2026-09-18", "2026-09-21"], name="date")
    assert _expand_with_next_trading_day(index, []) == []


def test_unresolved_gap_day_after_produces_no_signal_and_data_gap_warning(cfg):
    """구멍 다음 날짜를 data_gap으로 넘기면 core.state가 그날 이 종목을 완전히 건너뛴다
    (A1이든 다른 신호든 나지 않는다) — check_a1이 어긋난 '전날'로 판정하지 않게 막는다."""
    raw = make_synthetic_ohlcv(n=200, seed=7)
    raw = raw.copy()
    raw["close_source"] = CLOSE_SOURCE_YAHOO
    df = compute_indicators(raw, cfg)
    target_date = _first_a1_date(df, start=60, end_margin=2)

    # target_date가 통째로 사라진 상황을 흉내 낸다: 그 날짜와 다음 거래일을
    # data_gap으로 넘긴다 (fetch_one이 이제 이렇게 만든다).
    idx = df.index
    pos = idx.get_loc(target_date)
    next_date = idx[pos + 1]
    ticker = "TEST"
    gap_dates_by_ticker = {ticker: {target_date, next_date}}

    indicator_map = {ticker: df}
    per_ticker_dates = {ticker: [target_date, next_date]}
    states = {ticker: st.init_state(ticker)}

    sim = simulate_since(
        indicator_map,
        per_ticker_dates,
        states,
        cfg,
        earnings_map={},
        gap_dates_by_ticker=gap_dates_by_ticker,
        fills_df=pd.DataFrame(columns=["date", "ticker", "unit", "side", "price", "qty"]),
        virtual_fill=False,
        max_concurrent=8,
    )

    assert sim["all_events"] == []  # 둘 다 data_gap이라 아무 신호도 나지 않는다
    assert ticker in sim["data_gap_tickers"]
    assert any("data_gap" in w for w in sim["warnings"])
