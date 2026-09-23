"""cloud_top/cloud_bot이 D일 전의 max/min(span_a, span_b)와 같은지 확인."""

from __future__ import annotations

import numpy as np

from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv


def test_cloud_top_bot_are_shifted_by_D(cfg):
    df = make_synthetic_ohlcv(n=400, seed=4)
    out = compute_indicators(df, cfg)

    D = cfg["indicators"]["ichimoku_shift"]
    expected_top = np.maximum(out["span_a"], out["span_b"]).shift(D)
    expected_bot = np.minimum(out["span_a"], out["span_b"]).shift(D)

    diff_top = (out["cloud_top"] - expected_top).abs()
    diff_bot = (out["cloud_bot"] - expected_bot).abs()
    assert diff_top.dropna().max() < 1e-9
    assert diff_bot.dropna().max() < 1e-9

    # NaN이 나타나는 구간도 D칸만큼 밀려야 한다 (초반 워밍업 구간).
    assert (out["cloud_top"].isna() == expected_top.isna()).all()
    assert (out["cloud_bot"].isna() == expected_bot.isna()).all()

    # D일 전 span_a > span_b인 날에는 cloud_top이 그때의 span_a와 같아야 한다.
    shifted_span_a = out["span_a"].shift(D)
    shifted_span_b = out["span_b"].shift(D)
    mask = (shifted_span_a > shifted_span_b).fillna(False)
    assert np.allclose(
        out.loc[mask, "cloud_top"], shifted_span_a[mask], atol=1e-9
    )
