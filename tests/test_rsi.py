"""Wilder RSI를 반복문으로 직접 계산한 참조 구현과 비교 (오차 1e-6 이내)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv


def _reference_wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Wilder 평활(alpha=1/period)을 반복문으로 직접 계산한 참조 구현.

    입력: 종가 Series
    출력: RSI Series (index 0은 diff 기준값이 없어 NaN)
    """
    n = len(close)
    rsi = [np.nan] * n
    avg_gain = avg_loss = None
    alpha = 1 / period
    for i in range(1, n):
        change = close.iloc[i] - close.iloc[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        if avg_gain is None:  # 첫 유효값으로 시드
            avg_gain, avg_loss = gain, loss
        else:
            avg_gain = avg_gain * (1 - alpha) + gain * alpha
            avg_loss = avg_loss * (1 - alpha) + loss * alpha
        rsi[i] = 100.0 if avg_loss == 0 else 100 - 100 / (1 + avg_gain / avg_loss)
    return pd.Series(rsi, index=close.index)


def test_rsi_matches_reference_loop(cfg):
    df = make_synthetic_ohlcv(n=200, seed=2)
    out = compute_indicators(df, cfg)

    period = cfg["indicators"]["rsi"]["period"]
    expected = _reference_wilder_rsi(df["close"], period=period)

    diff = (out["rsi"] - expected).abs()
    assert diff.iloc[1:].max() < 1e-6


def test_rsi_is_100_when_loss_is_zero(cfg):
    """상승만 이어지는 구간에서는 loss가 0이 되어 RSI가 100이어야 한다."""
    n = 60
    index = pd.bdate_range("2021-01-01", periods=n, name="date")
    close = pd.Series(100 + np.arange(n, dtype=float), index=index)
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000_000,
        },
        index=index,
    )
    out = compute_indicators(df, cfg)
    period = cfg["indicators"]["rsi"]["period"]
    assert out["rsi"].iloc[period + 5 :].eq(100.0).all()
