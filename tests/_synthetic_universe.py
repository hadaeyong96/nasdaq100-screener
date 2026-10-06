"""네트워크 없이 쓰는 합성 종목 묶음 (동등성 테스트·스크리닝 견본용).

같은 seed면 항상 같은 OHLCV를 만든다. 지표는 core.indicators.compute_indicators로
실제와 똑같이 계산한다 — 신호·상태 전이·수량이 실제 코드 경로를 그대로 지나가게 하기 위해서다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.indicators import compute_indicators


def synthetic_ohlcv(seed: int, periods: int = 320, start: str = "2025-06-02") -> pd.DataFrame:
    """seed -> 하락·반등이 섞인 합성 일봉 DataFrame(open, high, low, close, volume)."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    period = rng.uniform(35, 80)
    drift = 0.004 * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    rets = drift + rng.normal(0, 0.018, periods)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, 0.008, periods))
    open_[1:] = np.where(rng.random(periods - 1) < 0.03, close[:-1] * 1.05, open_[1:])  # 가끔 갭
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, periods)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, periods)))
    volume = rng.integers(800_000, 2_400_000, periods).astype(float)
    index = pd.bdate_range(start, periods=periods, name="date")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)


def synthetic_indicator_map(cfg: dict, n: int = 24, periods: int = 320) -> dict:
    """{ticker: 지표 DataFrame} — 티커는 T00, T01, ..."""
    return {f"T{i:02d}": compute_indicators(synthetic_ohlcv(1000 + i, periods), cfg) for i in range(n)}
