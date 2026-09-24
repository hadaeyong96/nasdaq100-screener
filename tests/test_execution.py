"""core/execution.py 체결 모형 경계값 테스트 (P5-1 6번). 네트워크 없음."""

from __future__ import annotations

import math

from core import execution as ex


def test_buy_fills_at_open_when_open_below_limit():
    assert ex.resolve_buy_fill(limit_price=101.0, open_=99.0, low=98.0) == 99.0


def test_buy_fills_at_open_when_open_equals_limit():
    """경계값: 시가 == 지정가면 시가 체결."""
    assert ex.resolve_buy_fill(limit_price=100.0, open_=100.0, low=99.0) == 100.0


def test_buy_fills_at_limit_when_open_above_but_low_reaches_limit():
    assert ex.resolve_buy_fill(limit_price=100.0, open_=102.0, low=99.0) == 100.0


def test_buy_fills_at_limit_when_low_equals_limit_exactly():
    """경계값: 저가 == 지정가면 지정가 체결(미체결 아님)."""
    assert ex.resolve_buy_fill(limit_price=100.0, open_=102.0, low=100.0) == 100.0


def test_buy_unfilled_when_low_above_limit():
    assert ex.resolve_buy_fill(limit_price=100.0, open_=102.0, low=100.01) is None


def test_buy_unfilled_on_nan_inputs():
    assert ex.resolve_buy_fill(limit_price=100.0, open_=math.nan, low=99.0) is None
    assert ex.resolve_buy_fill(limit_price=100.0, open_=99.0, low=math.nan) is None


def test_stop_fills_at_stop_price_when_low_reaches_it_but_open_above():
    assert ex.resolve_stop_fill(stop_price=90.0, open_=92.0, low=89.0) == 90.0


def test_stop_fills_at_open_on_gap_down_below_stop():
    """손절가 아래로 갭 하락하면 시가에 체결(더 불리한 실제가)."""
    assert ex.resolve_stop_fill(stop_price=90.0, open_=85.0, low=83.0) == 85.0


def test_stop_boundary_low_equals_stop_price():
    assert ex.resolve_stop_fill(stop_price=90.0, open_=92.0, low=90.0) == 90.0


def test_stop_unfilled_when_low_above_stop_price():
    assert ex.resolve_stop_fill(stop_price=90.0, open_=95.0, low=90.01) is None


def test_exit_at_open_returns_open_price():
    assert ex.exit_at_open(101.5) == 101.5


def test_exit_at_open_none_when_open_missing():
    assert ex.exit_at_open(math.nan) is None
