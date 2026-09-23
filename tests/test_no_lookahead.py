"""미래 데이터 금지 테스트.

여러 절단 시점 t에서 "t까지 자른 데이터의 마지막 행"과
"전체 데이터의 t행"이 모든 지표 열에서 같아야 한다.
(shift(-n) 같은 음수 이동으로 미래 값을 끌어오면 이 테스트가 깨진다.)
"""

from __future__ import annotations

import numpy as np
import pytest

from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv

INDICATOR_COLUMNS = [
    "macd", "signal", "hist", "gc", "dc", "macd_norm",
    "rsi",
    "tenkan", "kijun", "span_a", "span_b", "cloud_top", "cloud_bot",
    "future_yang", "chikou_ok", "chikou_broken",
    "vol_ratio", "swing_low", "bb_width_pct",
    "bars", "insufficient_history",
]


def _values_equal(a, b) -> bool:
    if isinstance(a, (bool, np.bool_)):
        return bool(a) == bool(b)
    return bool(np.isclose(a, b, atol=1e-9, equal_nan=True))


@pytest.mark.parametrize("cutoff", [300, 350, 400, 450, 499])
def test_truncated_last_row_matches_full_row(cfg, cutoff):
    df = make_synthetic_ohlcv(n=500, seed=1)
    full = compute_indicators(df, cfg)

    truncated_input = df.iloc[: cutoff + 1]  # t까지만 자른 원본 데이터
    truncated = compute_indicators(truncated_input, cfg)

    for col in INDICATOR_COLUMNS:
        full_val = full[col].iloc[cutoff]
        truncated_val = truncated[col].iloc[-1]
        assert _values_equal(full_val, truncated_val), (
            f"{col} 열이 절단 시점(cutoff={cutoff})에서 미래 데이터를 쓴 것으로 보임: "
            f"전체={full_val!r}, 절단={truncated_val!r}"
        )
