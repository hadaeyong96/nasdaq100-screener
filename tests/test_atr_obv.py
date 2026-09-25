"""core.indicators.compute_atr/compute_obv 테스트 (P5-3 E1·C3 실험용 새 지표)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import compute_atr, compute_obv
from tests.conftest import make_synthetic_ohlcv


def _reference_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    n = len(df)
    atr = [np.nan] * n
    prev_atr = None
    alpha = 1 / period
    for i in range(n):
        prev_close = close.iloc[i - 1] if i > 0 else np.nan
        tr = max(high.iloc[i] - low.iloc[i], abs(high.iloc[i] - prev_close) if i > 0 else -np.inf, abs(low.iloc[i] - prev_close) if i > 0 else -np.inf)
        if i == 0:
            tr = high.iloc[i] - low.iloc[i]
        prev_atr = tr if prev_atr is None else prev_atr * (1 - alpha) + tr * alpha
        atr[i] = prev_atr
    return pd.Series(atr, index=df.index)


def test_atr_matches_reference_loop():
    df = make_synthetic_ohlcv(n=100, seed=3)
    out = compute_atr(df, period=14)
    expected = _reference_atr(df, period=14)
    diff = (out - expected).abs()
    assert diff.max() < 1e-6


def test_atr_is_never_negative():
    df = make_synthetic_ohlcv(n=100, seed=4)
    out = compute_atr(df, period=14)
    assert (out >= 0).all()


def test_obv_rises_on_up_day_and_falls_on_down_day():
    idx = pd.bdate_range("2021-01-01", periods=5, name="date")
    close = pd.Series([100, 105, 103, 103, 108], index=idx, dtype=float)
    volume = pd.Series([1000, 2000, 1500, 1200, 3000], index=idx)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=idx)
    obv = compute_obv(df)
    # day0: 기준 0. day1(상승): +2000. day2(하락): -1500. day3(보합): 그대로. day4(상승): +3000.
    assert obv.iloc[0] == 0
    assert obv.iloc[1] == 2000
    assert obv.iloc[2] == 500
    assert obv.iloc[3] == 500
    assert obv.iloc[4] == 3500
