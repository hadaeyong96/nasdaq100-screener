"""4장·5장 신호 판정 함수 테스트. NaN 입력은 "없음"으로 처리되는지,
지표가 실제로 계산된 데이터에서 미래 데이터를 쓰지 않는지 확인한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import signals as sig
from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv


def test_check_a1_nan_is_no_signal():
    assert sig.check_a1(np.nan, 32.0) is False
    assert sig.check_a1(20.0, np.nan) is False


def test_check_a1_crosses_up():
    assert sig.check_a1(29.9, 30.1) is True
    assert sig.check_a1(31.0, 32.0) is False  # 이미 30 이상이었으면 돌파가 아니다


def test_check_a3_breakout_nan_cloud_is_no_signal(cfg):
    """일목 구름이 아직 없어도(NaN) A3는 크래시 없이 False를 낸다 (SPCX 사례)."""
    row = pd.Series(
        {
            "close": 100.0,
            "cloud_top": np.nan,
            "future_yang": True,
            "chikou_ok": True,
            "macd": 1.0,
            "signal": 0.5,
            "rsi": 60.0,
        }
    )
    assert sig.check_a3_breakout(row, cfg) is False


def test_a1_and_a2_possible_without_cloud_data(cfg):
    """SPCX처럼 구름이 없어도 A1(RSI)·A2(MACD)는 판정할 수 있어야 한다."""
    assert sig.check_a1(29.0, 31.0) is True
    row = pd.Series({"gc": True, "rsi": 45.0})
    assert sig.check_a2(row) is True


def test_check_stop_none_is_no_signal():
    assert sig.check_stop(90.0, None) is False


def test_check_e2_nan_is_no_signal():
    assert sig.check_e2(np.nan, 45.0) is False


def test_entry_limit_price_rounds_to_cent(cfg):
    assert sig.entry_limit_price(100.0, cfg) == 101.0


# ── 손절 근접 경고 (P3.5 1번) ────────────────────────────────────────────


def test_stop_near_at_exactly_the_boundary_pct_is_shown():
    """종가가 손절가보다 정확히 3.0% 위면 근접 표시(경계 포함)."""
    assert sig.stop_near(close=103.0, stop_price=100.0, pct=3) is True


def test_stop_near_just_over_the_boundary_pct_is_not_shown():
    """3.1%면 표시하지 않는다."""
    assert sig.stop_near(close=103.1, stop_price=100.0, pct=3) is False


def test_stop_near_below_stop_price_is_false():
    """이미 손절가 아래(손절 신호일)면 근접이 아니라 손절이다 — False."""
    assert sig.stop_near(close=99.0, stop_price=100.0, pct=3) is False


def test_stop_near_no_stop_price_is_false():
    assert sig.stop_near(close=100.0, stop_price=None, pct=3) is False


# ── 미래 데이터 방지: t까지 자른 데이터의 지표 == 전체 데이터의 t행 ─────────
# (core.signals가 참조하는 gc/dc/cloud_top 등은 core.indicators가 만든다.
#  여기서는 그 지표를 바탕으로 core.signals 조건이 같은 결과를 내는지 본다.)


def test_signals_use_same_result_on_truncated_indicator_data(cfg):
    df = make_synthetic_ohlcv(n=400, seed=11)
    full = compute_indicators(df, cfg)

    t = 300
    truncated = compute_indicators(df.iloc[: t + 1], cfg)

    full_row, trunc_row = full.iloc[t], truncated.iloc[t]
    assert sig.check_a2(full_row) == sig.check_a2(trunc_row)
    assert sig.check_e1(full_row) == sig.check_e1(trunc_row)
    assert sig.check_e3(full_row) == sig.check_e3(trunc_row)
    assert sig.check_a3(full_row, cfg) == sig.check_a3(trunc_row, cfg)
