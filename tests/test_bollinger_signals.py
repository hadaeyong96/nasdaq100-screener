"""볼린저·RSI·MACD 신호(core/bollinger_signals.py) 단위 테스트 + 기존 신호 회귀 테스트.

손으로 만든 가격 데이터로 각 조건(하단 터치 후 RSI 30 돌파, 중심선 위·상승, 중심선 아래·하락, 밴드폭
수축 후 확장)을 확인한다. 기존 core.signals 함수와 core.state.process_day(기본 규칙)의 결과는 바뀌지 않아야 한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import bollinger_signals as bb
from core import state as st
from core.indicators import compute_indicators

BB = {"period": 20, "std_mult": 2, "ddof": 0, "slope_days": 5, "touch_days": 5, "width_rank_period": 120,
      "width_rank_max": 0.2, "squeeze_bonus": 20}
CFG = {
    "assumptions": {"s_grade_macd_norm_min_pct": -0.5, "b_grade_macd_norm_max_pct": -2.0, "whipsaw_max_crosses_20d": 4,
                    "gap_filter_pct": 4.0, "reentry_cooldown_days": 5, "a1_to_a2_expiry_days": 10},
    "a3": {"mode": "breakout", "pullback_tolerance_pct": 0.5},
}


def _df(close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    return pd.DataFrame({"close": close}, index=pd.bdate_range("2024-01-02", periods=len(close)))


def test_band_values_match_hand_formula():
    close = 100 + np.sin(np.arange(60)) * 5
    out = bb.add_bollinger_columns(_df(close), BB)
    w = close[-20:]
    mid, sd = w.mean(), w.std(ddof=0)
    assert out["bb_mid"].iat[-1] == pytest.approx(mid)
    assert out["bb_upper"].iat[-1] == pytest.approx(mid + 2 * sd)
    assert out["bb_lower"].iat[-1] == pytest.approx(mid - 2 * sd)
    assert out["bb_width"].iat[-1] == pytest.approx(4 * sd / mid)
    assert out["bb_mid_up"].iat[-1] == bool(mid > close[-25:-5].mean())


def test_lower_touch_window_is_five_days_including_today():
    close = np.full(40, 100.0)
    close[:20] = 100 + np.arange(20) % 2  # 변동이 있어야 밴드가 생긴다
    close[30] = 80.0  # 30일에 하단 아래로
    out = bb.add_bollinger_columns(_df(close), BB)
    assert bool(out["bb_touch_lower"].iat[30])
    assert bool(out["bb_touch_lower"].iat[34])  # 30~34 (5거래일)
    assert not bool(out["bb_touch_lower"].iat[35])  # 6거래일 뒤에는 아님


def _row(**kw) -> pd.Series:
    base = {"close": 100.0, "rsi": 31.0, "bb_mid": 95.0, "bb_mid_up": True, "bb_mid_down": False, "bb_touch_lower": True,
            "macd": 1.0, "signal": 0.5, "gc": True, "macd_norm": 0.0}
    base.update(kw)
    return pd.Series(base)


def test_a1_needs_rsi_cross_and_recent_lower_touch():
    assert bb.check_a1_bb(28.0, _row(rsi=31.0), CFG)
    assert not bb.check_a1_bb(28.0, _row(rsi=31.0, bb_touch_lower=False), CFG)  # 하단 터치 없음
    assert not bb.check_a1_bb(31.0, _row(rsi=33.0), CFG)  # RSI 30 돌파 아님


def test_a3_above_rising_mid_with_macd_and_rsi():
    assert bb.check_a3_bb(_row(rsi=55.0), CFG)
    assert not bb.check_a3_bb(_row(rsi=55.0, bb_mid_up=False), CFG)  # 중심선 상승 아님
    assert not bb.check_a3_bb(_row(rsi=55.0, close=90.0), CFG)  # 종가 < 중심선
    assert not bb.check_a3_bb(_row(rsi=45.0), CFG)  # RSI < 50
    assert not bb.check_a3_bb(_row(rsi=55.0, macd=0.1), CFG)  # MACD < 시그널


def test_b_trend_replaced_by_mid_rest_same():
    assert bb.check_b_bb(_row(rsi=60.0), CFG)
    assert not bb.check_b_bb(_row(rsi=60.0, bb_mid_up=False), CFG)
    assert not bb.check_b_bb(_row(rsi=72.0), CFG)  # RSI 70 이상
    assert not bb.check_b_bb(_row(rsi=60.0, gc=False), CFG)
    assert not bb.check_b_bb(_row(rsi=60.0, macd_norm=-1.0), CFG)  # S등급 하한 미만


def test_e3_below_falling_mid():
    assert bb.check_e3_bb(_row(close=90.0, bb_mid_down=True, bb_mid_up=False))
    assert not bb.check_e3_bb(_row(close=90.0, bb_mid_down=False))  # 중심선 아직 상승
    assert not bb.check_e3_bb(_row(close=100.0, bb_mid_down=True))  # 종가가 중심선 위


def test_hand_made_path_triggers_each_condition():
    # 완만한 상승 → 급락(하단 이탈) → 반등(RSI 30 돌파) → 상승 추세(중심선 위·상승) → 하락(중심선 아래·하락)
    up = list(np.linspace(100, 120, 140))
    crash = list(np.linspace(118, 92, 10))
    rebound = list(np.linspace(96, 118, 15))  # 가파른 반등 — 마지막 하단 이탈 뒤 5거래일 안에 RSI 30 돌파
    trend = list(np.linspace(118, 133, 20))
    fall = list(np.linspace(132, 100, 20))
    close = np.array(up + crash + rebound + trend + fall)
    rng = np.random.default_rng(0)
    close = close * (1 + rng.normal(0, 0.002, len(close)))
    df = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6},
                      index=pd.bdate_range("2024-01-02", periods=len(close)))
    cfg = {**CFG, "indicators": {"macd": {"fast": 12, "slow": 26, "signal": 9}, "rsi": {"period": 14},
                                 "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52}, "ichimoku_shift": 26,
                                 "volume_ma": {"period": 20}, "swing_low": {"period": 10},
                                 "bollinger": {"period": 20, "std_mult": 2, "percentile_period": 120}}}
    ind = bb.add_bollinger_columns(compute_indicators(df, cfg), BB)
    a1 = [i for i in range(1, len(ind)) if bb.check_a1_bb(ind["rsi"].iat[i - 1], ind.iloc[i], cfg)]
    a3 = [i for i in range(len(ind)) if bb.check_a3_bb(ind.iloc[i], cfg)]
    e3 = [i for i in range(len(ind)) if bb.check_e3_bb(ind.iloc[i])]
    assert a1 and 150 <= a1[0] < 156  # 급락(140~149) 뒤 반등 초입
    assert any(160 <= i < 185 for i in a3)  # 반등·상승 추세 구간 (중심선 위·상승)
    assert any(i >= 190 for i in e3)  # 하락 구간 (중심선 아래·하락)
    assert not any(140 <= i < 150 for i in a3)  # 급락 중에는 3차 없음


def test_squeeze_release_bonus_and_no_cloud_bans():
    close = np.r_[100 + np.sin(np.arange(150)) * 3, np.full(30, 100.0) + np.arange(30) * 0.01, 100 + np.arange(10) * 2.0]
    df = bb.add_bollinger_columns(_df(close), BB)
    rel = df.index[df["bb_squeeze_release"].eq(True)]
    late = [x for x in rel if x >= df.index[180]]
    assert late  # 거의 평평하다가(밴드 수축) 오르기 시작하는 순간에 가점 조건이 켜진다
    d = late[0]
    df = df.assign(gc=False, dc=False, rsi=55.0, vol_ratio=1.0, macd_norm=0.0, open=df["close"],
                   cloud_top=200.0, cloud_bot=150.0, future_yang=False)  # 일목 기준이면 '구름 아래·음운'
    _, score = bb.entry_score_bb(df, d, df.loc[d], "A3", CFG)
    assert score == 20  # 등급 없음 + 거래량 1배 + 가점 20
    assert bb.ban_reasons_bb(df, d, df.loc[d], "A3", CFG, None) == []  # 구름 금지는 빠진다
    df.loc[d, "rsi"] = 75.0
    assert any("RSI" in r for r in bb.ban_reasons_bb(df, d, df.loc[d], "B", CFG, None))  # RSI 70 금지는 그대로


@pytest.mark.parametrize("cut", [40, 120, 160])
def test_no_lookahead_bollinger_columns(cut):
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 1, 200))
    full = bb.add_bollinger_columns(_df(close), BB)
    part = bb.add_bollinger_columns(_df(close[:cut]), BB)
    cols = ["bb_mid", "bb_lower", "bb_width", "bb_touch_lower", "bb_mid_up", "bb_mid_down", "bb_squeeze_release"]
    pd.testing.assert_frame_equal(part[cols], full[cols].iloc[:cut], check_dtype=False)


# ── 회귀: 기본 규칙(일목)은 예전과 같은 결과 ─────────────────────────────────


def test_default_rules_are_existing_functions():
    from core import signals as sig

    r = st.ICHIMOKU_RULES
    assert r.a2 is sig.check_a2 and r.a3 is sig.check_a3 and r.b is sig.check_b and r.e3 is sig.check_e3


def test_process_day_default_equals_explicit_ichimoku_rules():
    from engine.daily import load_config

    cfg = load_config()
    rng = np.random.default_rng(7)
    close = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, 400))
    df = pd.DataFrame({"open": close * (1 + rng.normal(0, 0.005, 400)), "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": rng.integers(1e6, 2e6, 400).astype(float)},
                      index=pd.bdate_range("2023-01-02", periods=400))
    ind = compute_indicators(df, cfg)
    s_default, s_explicit = st.init_state("X"), st.init_state("X")
    ev_default, ev_explicit = [], []
    for d in ind.index[150:]:
        e1, s_default = st.process_day(ind, d, s_default, cfg)
        e2, s_explicit = st.process_day(ind, d, s_explicit, cfg, rules=st.ICHIMOKU_RULES)
        if s_default["state"] == "주문대기":  # 가상 체결(paper와 같은 방식)로 단계 진행까지 확인
            fill = {"unit": s_default["pending"]["unit"], "side": "buy", "price": float(ind.at[d, "close"]), "qty": 10}
            s_default, s_explicit = st.apply_fill(s_default, fill, cfg), st.apply_fill(s_explicit, fill, cfg)
        ev_default += e1
        ev_explicit += e2
    assert ev_default == ev_explicit and s_default == s_explicit
    assert {e["kind"] for e in ev_default} & {"A1", "A2", "B", "STOP", "E1", "E2", "E3", "A1_EXPIRE"}
