"""테스트 공용 픽스처. 네트워크 없이 합성 데이터로 돈다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg() -> dict:
    """config.yaml을 그대로 읽어 반환한다."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_synthetic_ohlcv(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """네트워크 없이 쓰는 합성 OHLCV 데이터를 만든다.

    입력: 생성할 행 수, 난수 시드
    출력: DataFrame(open, high, low, close, volume), 영업일 오름차순 인덱스
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.05, scale=1.2, size=n)
    close = 100 + np.cumsum(steps)
    close = np.clip(close, 5, None)
    high = close + rng.uniform(0.1, 1.5, size=n)
    low = close - rng.uniform(0.1, 1.5, size=n)
    low = np.minimum(low, close - 0.01)
    open_ = low + rng.uniform(0, 1, size=n) * (high - low)
    volume = rng.integers(1_000_000, 5_000_000, size=n)
    index = pd.bdate_range("2020-01-02", periods=n, name="date")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
