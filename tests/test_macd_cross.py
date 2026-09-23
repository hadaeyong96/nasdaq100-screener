"""교차가 알려진 합성 시계열에서 gc/dc가 정확한 날짜에만 True인지 확인."""

from __future__ import annotations

from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv


def test_gc_dc_fire_only_at_known_cross_dates(cfg):
    df = make_synthetic_ohlcv(n=400, seed=3)
    out = compute_indicators(df, cfg)

    # strategy_v3.md 9장 수식으로 macd/signal을 독립적으로 다시 계산해
    # 교차가 실제로 일어나는 날짜(정답)를 구한다.
    macd_cfg = cfg["indicators"]["macd"]
    ema_fast = df["close"].ewm(span=macd_cfg["fast"], adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_cfg["slow"], adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_cfg["signal"], adjust=False).mean()

    expected_gc = (macd.shift(1) <= signal.shift(1)) & (macd > signal)
    expected_dc = (macd.shift(1) >= signal.shift(1)) & (macd < signal)

    assert (out["gc"].fillna(False) == expected_gc.fillna(False)).all()
    assert (out["dc"].fillna(False) == expected_dc.fillna(False)).all()

    # gc와 dc가 같은 날 동시에 True인 경우는 없어야 한다.
    assert not (out["gc"] & out["dc"]).any()

    # 형식적으로만 통과하지 않도록, 실제로 교차가 여러 번 있는지도 확인한다.
    assert expected_gc.sum() >= 2
    assert expected_dc.sum() >= 2
