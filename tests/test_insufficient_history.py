"""상장 기간이 짧은 종목(예: 최근 IPO) 처리 테스트.

- 짧은 합성 데이터에서도 compute_indicators가 예외 없이 동작해야 한다.
- bars 열이 실제 보유 거래일 수와 같아야 한다.
- 일목 구름·볼린저 백분위처럼 lookback이 긴 지표가 아직 NaN인 동안은
  insufficient_history가 True여야 하고, 데이터가 충분히 쌓이면 False가 돼야 한다.
"""

from __future__ import annotations

from core.indicators import compute_indicators
from tests.conftest import make_synthetic_ohlcv


def test_short_history_runs_without_exception_and_flags_insufficient(cfg):
    # 일목 구름(52+26=78일)도, 볼린저 백분위(20+120=140일)도 채우지 못하는 짧은 데이터.
    df = make_synthetic_ohlcv(n=20, seed=5)

    out = compute_indicators(df, cfg)  # 예외 없이 끝나야 한다

    assert len(out) == 20
    assert list(out["bars"]) == list(range(1, 21))
    assert out["insufficient_history"].all()  # 전 구간이 데이터 부족


def test_insufficient_history_becomes_false_once_enough_bars(cfg):
    # 일목 구름·볼린저 백분위를 모두 채울 만큼 충분히 긴 데이터.
    df = make_synthetic_ohlcv(n=400, seed=6)
    out = compute_indicators(df, cfg)

    assert out["bars"].iloc[-1] == 400
    # 초반(워밍업 구간)은 데이터 부족, 후반은 충분해야 한다.
    assert out["insufficient_history"].iloc[:50].all()
    assert not out["insufficient_history"].iloc[-1]


def test_rsi_available_before_ichimoku_cloud(cfg):
    """RSI처럼 lookback이 짧은 지표는 일목 구름보다 먼저 유효값을 낸다."""
    rsi_period = cfg["indicators"]["rsi"]["period"]
    df = make_synthetic_ohlcv(n=rsi_period + 5, seed=7)
    out = compute_indicators(df, cfg)

    assert out["rsi"].notna().any()  # RSI는 이미 계산 가능
    assert out["insufficient_history"].all()  # 구름·볼린저는 아직 부족
