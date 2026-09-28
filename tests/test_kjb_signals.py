"""core/kjb.py 순수 함수 테스트 (docs/kjb1_instructions.md 3번). 네트워크 없이 돈다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import kjb


def _series(values, start="2020-01-01"):
    index = pd.bdate_range(start=start, periods=len(values), name="date")
    return pd.Series(values, index=index, dtype=float)


# ── relative_return / first_excess 경계값 ──────────────────────────────────


def test_relative_return_zero_boundary():
    # 63일 전 100 -> 오늘 105(+5%), QQQ는 100 -> 105(+5%) : 상대수익 정확히 0
    close = _series([100.0] * 63 + [105.0])
    qqq = _series([100.0] * 63 + [105.0])
    rel = kjb.relative_return(close, qqq, window_days=63)
    assert rel.iloc[-1] == pytest.approx(0.0)


def test_first_excess_true_when_crosses_above_zero_with_no_prior_history():
    # 직전 63일 내내 상대수익 <= 0 이다가, 오늘 처음으로 > 0.
    n = 70
    rel_values = [-0.01] * (n - 1) + [0.02]
    rel = pd.Series(rel_values, index=pd.bdate_range("2020-01-01", periods=n))
    flag = kjb.first_excess(rel, window_days=63)
    assert bool(flag.iloc[-1]) is True


def test_first_excess_false_when_excess_happened_within_prior_63_days():
    """직전 63거래일 동안 이미 한 번이라도 초과한 적이 있으면 "처음"이 아니다."""
    n = 200
    rng = np.random.default_rng(1)
    rel_values = rng.normal(-0.01, 0.02, n)
    rel_values[100] = 0.05  # 이미 한 번 초과했던 이력
    rel_values[150] = -0.01
    rel_values[199] = 0.03  # 오늘도 양전환이지만 "처음"은 아님(100번째 날 때문)
    rel = pd.Series(rel_values, index=pd.bdate_range("2020-01-01", periods=n))
    flag = kjb.first_excess(rel, window_days=63)
    assert bool(flag.iloc[199]) is False


def test_first_excess_false_without_yesterday_le_zero():
    rel = _series([0.01, 0.02, 0.03])  # 어제도 이미 양수였다 -> "처음"이 아님
    flag = kjb.first_excess(rel, window_days=2)
    assert bool(flag.iloc[-1]) is False


# ── dollar_volume_multiplier 경계값 (1.99 / 2.00) ──────────────────────────


def test_dollar_volume_multiplier_boundary_just_below_and_at_threshold():
    # 직전 20일 평균 거래대금 = 1000, 오늘 거래대금 = 1990 -> 배율 1.99
    close = _series([10.0] * 21)
    volume = _series([100.0] * 20 + [199.0])
    mult = kjb.dollar_volume_multiplier(close, volume, window_days=20)
    assert mult.iloc[-1] == pytest.approx(1.99)

    volume2 = _series([100.0] * 20 + [200.0])
    mult2 = kjb.dollar_volume_multiplier(close, volume2, window_days=20)
    assert mult2.iloc[-1] == pytest.approx(2.00)


def test_kjb_entry_signal_requires_multiplier_at_least_min():
    idx = pd.bdate_range("2020-01-01", periods=1)
    first_excess_flag = pd.Series([True], index=idx)
    big_candle_flag = pd.Series([True], index=idx)

    below = pd.Series([1.99], index=idx)
    at = pd.Series([2.00], index=idx)

    assert bool(kjb.kjb_entry_signal(None, first_excess_flag, below, big_candle_flag, 2.0).iloc[0]) is False
    assert bool(kjb.kjb_entry_signal(None, first_excess_flag, at, big_candle_flag, 2.0).iloc[0]) is True


# ── big_bull_candle 경계값 (4.99% / 5.00%, 음봉 제외) ───────────────────────


def test_big_bull_candle_boundary_just_below_and_at_threshold():
    close = _series([100.0, 104.99])
    open_ = _series([100.0, 100.0])
    flag = kjb.big_bull_candle(open_, close, pct_min=5.0)
    assert bool(flag.iloc[-1]) is False

    close2 = _series([100.0, 105.00])
    flag2 = kjb.big_bull_candle(open_, close2, pct_min=5.0)
    assert bool(flag2.iloc[-1]) is True


def test_big_bull_candle_excludes_bearish_candle_even_with_enough_pct_change():
    """전일 종가 대비로는 +5% 이상 올랐어도 당일 시가보다 종가가 낮으면(음봉) 제외."""
    close = _series([100.0, 106.0])
    open_ = _series([100.0, 110.0])  # 시가 110, 종가 106 -> 음봉
    flag = kjb.big_bull_candle(open_, close, pct_min=5.0)
    assert bool(flag.iloc[-1]) is False


def test_big_bull_candle_volatility_margin_uses_prior_std_not_today():
    close = _series([100.0] * 21 + [110.0])
    open_ = _series([100.0] * 21 + [100.0])
    flag = kjb.big_bull_candle_volatility_margin(open_, close, window_days=20, mult=2.5)
    # 직전 20일은 변동 없음(표준편차 0) -> 문턱값 0 -> 오늘 아무리 작은 상승도 통과
    assert bool(flag.iloc[-1]) is True


# ── 미래 데이터 금지 ────────────────────────────────────────────────────────


def test_no_lookahead_relative_return_and_first_excess_stable_when_truncated():
    n = 200
    rng = np.random.default_rng(3)
    stock = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
    qqq = 100 + np.cumsum(rng.normal(0.05, 0.8, n))
    close = pd.Series(stock, index=pd.bdate_range("2020-01-01", periods=n))
    qqq_s = pd.Series(qqq, index=pd.bdate_range("2020-01-01", periods=n))

    cut = 150
    rel_full = kjb.relative_return(close, qqq_s, window_days=63)
    rel_trunc = kjb.relative_return(close.iloc[:cut], qqq_s.iloc[:cut], window_days=63)
    assert rel_full.iloc[cut - 1] == pytest.approx(rel_trunc.iloc[-1])

    flag_full = kjb.first_excess(rel_full, window_days=63)
    flag_trunc = kjb.first_excess(rel_trunc, window_days=63)
    assert bool(flag_full.iloc[cut - 1]) == bool(flag_trunc.iloc[-1])

    # 절단 이후 미래 자료를 완전히 다른 값으로 바꿔도(예: 폭락) 과거 신호는 그대로.
    close_altered = close.copy()
    close_altered.iloc[cut:] = close_altered.iloc[cut:] * 0.1
    rel_altered = kjb.relative_return(close_altered, qqq_s, window_days=63)
    assert rel_altered.iloc[cut - 1] == pytest.approx(rel_full.iloc[cut - 1])


def test_no_lookahead_dollar_volume_multiplier_and_big_candle():
    n = 40
    close = _series([10.0 + i * 0.1 for i in range(n)])
    volume = _series([100.0 + i for i in range(n)])
    open_ = _series([10.0 + i * 0.1 for i in range(n)])

    cut = 30
    mult_full = kjb.dollar_volume_multiplier(close, volume, window_days=20)
    mult_trunc = kjb.dollar_volume_multiplier(close.iloc[:cut], volume.iloc[:cut], window_days=20)
    assert mult_full.iloc[cut - 1] == pytest.approx(mult_trunc.iloc[-1])

    candle_full = kjb.big_bull_candle(open_, close, pct_min=5.0)
    candle_trunc = kjb.big_bull_candle(open_.iloc[:cut], close.iloc[:cut], pct_min=5.0)
    assert bool(candle_full.iloc[cut - 1]) == bool(candle_trunc.iloc[-1])


# ── first_score ──────────────────────────────────────────────────────────


def test_first_score_counts_prior_positive_days_excluding_today():
    daily = _series([0.01, -0.01, 0.02, 0.01, -0.02, 0.5])  # 오늘(마지막)은 세지 않는다
    score = kjb.first_score(daily, window_days=5)
    assert score.iloc[-1] == 3  # 앞 5일 중 양수 3개(0.01, 0.02, 0.01)


# ── rank_candidates 우선순위 ────────────────────────────────────────────────


def test_rank_candidates_orders_by_sector_strength_then_first_score_then_multiplier():
    candidates = [
        {"ticker": "A", "sector": "Tech", "first_score": 5, "dollar_volume_multiplier": 2.0},
        {"ticker": "B", "sector": "Tech", "first_score": 2, "dollar_volume_multiplier": 3.0},
        {"ticker": "C", "sector": "Health", "first_score": 1, "dollar_volume_multiplier": 10.0},
    ]
    sector_rel_strength = {"Tech": 0.10, "Health": 0.02}
    dollar_volume_rank = {"A": 50, "B": 51, "C": 52}

    ranked = kjb.rank_candidates(candidates, sector_rel_strength, dollar_volume_rank, top_n_deprioritized=10)
    assert [c["ticker"] for c in ranked] == ["B", "A", "C"]


def test_rank_candidates_deprioritizes_top_dollar_volume_tickers():
    candidates = [
        {"ticker": "MEGA", "sector": "Tech", "first_score": 1, "dollar_volume_multiplier": 5.0},
        {"ticker": "SMALL", "sector": "Health", "first_score": 20, "dollar_volume_multiplier": 2.0},
    ]
    sector_rel_strength = {"Tech": 0.20, "Health": 0.01}  # MEGA가 섹터 강도로는 훨씬 앞섬
    dollar_volume_rank = {"MEGA": 3, "SMALL": 80}  # 그러나 MEGA는 거래대금 상위 10위 안 -> 맨 뒤로

    ranked = kjb.rank_candidates(candidates, sector_rel_strength, dollar_volume_rank, top_n_deprioritized=10)
    assert [c["ticker"] for c in ranked] == ["SMALL", "MEGA"]

pytest.approx = pytest.approx
