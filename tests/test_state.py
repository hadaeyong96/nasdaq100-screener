"""8장 상태 전이 표의 화살표를 합성 시나리오로 검증한다. 네트워크 없이 돈다.

core/state.py는 지표가 이미 계산된 DataFrame을 받으므로, 여기서는 필요한 지표
열만 채운 합성 DataFrame을 직접 만들어 각 상태 전이를 독립적으로 검사한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import state as st

_DEFAULTS = {
    "open": np.nan,
    "high": np.nan,
    "low": np.nan,
    "close": np.nan,
    "volume": 1_000_000,
    "rsi": np.nan,
    "macd": np.nan,
    "signal": np.nan,
    "hist": np.nan,
    "gc": False,
    "dc": False,
    "macd_norm": np.nan,
    "tenkan": np.nan,
    "kijun": np.nan,
    "span_a": np.nan,
    "span_b": np.nan,
    "cloud_top": np.nan,
    "cloud_bot": np.nan,
    "future_yang": True,
    "chikou_ok": True,
    "chikou_broken": False,
    "vol_ratio": 1.0,
    "swing_low": np.nan,
    "bb_width_pct": 0.5,
}


def make_df(rows: list[dict], start="2026-01-02") -> pd.DataFrame:
    """지표 열이 채워진 합성 DataFrame을 만든다. 지정하지 않은 열은 기본값."""
    index = pd.bdate_range(start, periods=len(rows), name="date")
    full_rows = []
    for r in rows:
        merged = {**_DEFAULTS, **r}
        full_rows.append(merged)
    return pd.DataFrame(full_rows, index=index)


@pytest.fixture
def cfg() -> dict:
    return {
        "assumptions": {
            "a1_to_a2_expiry_days": 10,
            "s_grade_macd_norm_min_pct": -0.5,
            "b_grade_macd_norm_max_pct": -2.0,
            "gap_filter_pct": 4.0,
            "whipsaw_max_crosses_20d": 4,
            "reentry_cooldown_days": 5,
        },
        "a3": {"mode": "breakout", "pullback_tolerance_pct": 0.5},
        "entry": {"limit_markup": 1.01},
    }


# ── A1 → A2 유효기간 (10거래일) ─────────────────────────────────────────


def test_a2_still_valid_on_10th_trading_day(cfg):
    """A1 당일 포함 10거래일째(오프셋 9)는 아직 유효해 A2가 그대로 발생한다."""
    rows = [{"close": 100, "rsi": 20}]  # day-1: A1이 상향 돌파할 기준(전일 rsi<30)
    rows += [{"close": 100, "rsi": 32}]  # day0: A1 (오프셋0)
    rows += [{"close": 100, "rsi": 40} for _ in range(8)]  # day1..8 (오프셋1~8)
    rows += [{"close": 100, "rsi": 45, "gc": True}]  # day9 (오프셋9=10일째)
    df = make_df(rows)

    state = st.init_state("TEST")
    events = []
    for date in df.index[:-1]:
        evs, state = st.process_day(df, date, state, cfg)
        events.extend(evs)
    evs, state = st.process_day(df, df.index[-1], state, cfg)

    kinds = [e["kind"] for e in evs]
    assert kinds == ["A2"]
    assert state["state"] == "확인"


def test_a1_expires_on_11th_trading_day():
    cfg_local = {
        "assumptions": {
            "a1_to_a2_expiry_days": 10,
            "reentry_cooldown_days": 5,
        },
        "a3": {"mode": "breakout", "pullback_tolerance_pct": 0.5},
        "entry": {"limit_markup": 1.01},
    }
    rows = [{"close": 100, "rsi": 20}]  # day-1: A1이 상향 돌파할 기준
    rows += [{"close": 100, "rsi": 32, "swing_low": 94}]  # day0: A1 (오프셋0)
    rows += [{"close": 100, "rsi": 40} for _ in range(9)]  # day1..9 (오프셋1~9)
    rows += [{"close": 100, "rsi": 45, "gc": True}]  # day10 (오프셋10=11일째, 만료일)
    df = make_df(rows)

    state = st.init_state("TEST")
    events = []
    for date in df.index[:-1]:
        evs, state = st.process_day(df, date, state, cfg_local)
        events.extend(evs)
        for e in evs:
            if e["kind"] == "A1":  # A1이 실제 체결됐다고 가정(수량 55주)
                state = st.apply_fill(state, {"unit": "1", "side": "buy", "price": 101.0, "qty": 55}, cfg_local)
    evs, state = st.process_day(df, df.index[-1], state, cfg_local)

    kinds = [e["kind"] for e in evs]
    assert kinds == ["A1_EXPIRE"]  # A2 조건이 True여도 이미 대기로 돌아가 발생하지 않는다
    assert evs[0]["qty"] == 55
    assert state["state"] == "대기"
    assert state["cooldown_until"] is not None


# ── 손절 최우선 ──────────────────────────────────────────────────────────


def test_stop_takes_priority_over_e1_same_day(cfg):
    """손절과 E1이 같은 날이면 손절만 발생하고 나머지는 무시한다."""
    state = st.init_state("TEST")
    state["state"] = "확인"
    state["units"] = {"1": 37, "2": 49}
    state["entries"] = {"1": 100.0, "2": 103.0}
    state["stop"] = 94.0

    df = make_df([{"close": 90.0, "dc": True, "rsi": 45}])  # 종가<손절가, 동시에 데드크로스
    events, new_state = st.process_day(df, df.index[0], state, cfg)

    # 손절은 보유 중인 묶음(1·2)을 모두 전량 매도한다 — E1(데드크로스)은 무시된다.
    assert all(e["kind"] == "STOP" for e in events)
    assert {e["unit"] for e in events} == {"1", "2"}
    assert sum(e["qty"] for e in events) == 37 + 49
    assert new_state["state"] == "대기"
    assert new_state["units"] == {}


# ── E2가 E1보다 먼저 와도 묶음만 매도 ────────────────────────────────────


def test_e2_alone_sells_only_bundle_2(cfg):
    state = st.init_state("TEST")
    state["state"] = "확인"
    state["units"] = {"1": 37, "2": 49}
    state["entries"] = {"1": 100.0, "2": 103.0}
    state["stop"] = 94.0

    # E2만 발생(전일 rsi>=50, 오늘 <50), dc는 False라 E1은 없다.
    df = make_df([{"close": 105.0, "rsi": 49.0}], start="2026-01-05")
    prev = make_df([{"close": 104.0, "rsi": 51.0}], start="2026-01-02")
    full = pd.concat([prev, df])

    events, new_state = st.process_day(full, full.index[-1], state, cfg)

    assert [e["kind"] for e in events] == ["E2"]
    assert events[0]["unit"] == "2"
    assert new_state["units"]["1"] == 37  # 묶음 1은 그대로
    assert new_state["units"]["2"] == 0
    assert new_state["state"] == "청산중"  # 잔량(묶음1)이 남아 있어 대기로 복귀하지 않는다


# ── 청산중에는 추가 매수 없음 ────────────────────────────────────────────


def test_no_new_buy_while_liquidating_even_if_a3_conditions_met(cfg):
    state = st.init_state("TEST")
    state["state"] = "청산중"
    state["units"] = {"1": 0, "2": 49}
    state["entries"] = {"1": 100.0, "2": 103.0}
    state["stop"] = 94.0

    # A3의 4가지 조건이 모두 참이어도(원래는 "확인" 상태에서만 A3를 검사) 청산중에는 매수가 없다.
    df = make_df(
        [
            {
                "close": 120.0,
                "cloud_top": 110.0,
                "future_yang": True,
                "chikou_ok": True,
                "macd": 1.0,
                "signal": 0.5,
                "rsi": 60.0,
            }
        ]
    )
    events, new_state = st.process_day(df, df.index[0], state, cfg)

    assert all(e["kind"] != "A3" for e in events)
    assert new_state["state"] != "확정"


# ── 재진입 대기 중 A1 무시 ───────────────────────────────────────────────


def test_a1_ignored_during_cooldown(cfg):
    state = st.init_state("TEST")
    state["cooldown_until"] = pd.Timestamp("2026-01-10")

    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")
    events, new_state = st.process_day(df, df.index[-1], state, cfg)

    assert events == []
    assert new_state["state"] == "대기"


def test_a1_allowed_after_cooldown_passes(cfg):
    state = st.init_state("TEST")
    state["cooldown_until"] = pd.Timestamp("2026-01-05")

    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-06")
    events, new_state = st.process_day(df, df.index[-1], state, cfg)

    assert [e["kind"] for e in events] == ["A1"]
    assert new_state["state"] == "정찰"


# ── 중복 알림 방지: 같은 날 두 번 실행해도 이벤트는 한 번만 ─────────────────


def test_duplicate_run_same_day_emits_event_once(cfg):
    state = st.init_state("TEST")
    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")

    events1, state_after1 = st.process_day(df, df.index[-1], state, cfg)
    events2, state_after2 = st.process_day(df, df.index[-1], state_after1, cfg)

    assert [e["kind"] for e in events1] == ["A1"]
    assert events2 == []  # 두 번째 실행은 이미 sent_alerts에 있어 아무 것도 내지 않는다


# ── 신호 미래 데이터 방지: t까지 자른 데이터 == 전체 데이터의 t행 ──────────


def test_no_lookahead_process_day_matches_on_truncated_data(cfg):
    rows = [
        {"close": 100, "rsi": 20},
        {"close": 101, "rsi": 32},  # A1
        {"close": 102, "rsi": 40},
        {"close": 103, "rsi": 45, "gc": True},  # A2
        {"close": 130, "rsi": 60, "cloud_top": 110, "future_yang": True, "chikou_ok": True, "macd": 1, "signal": 0.5},  # A3
    ]
    df = make_df(rows)

    def replay(frame):
        state = st.init_state("TEST")
        all_events = []
        for date in frame.index:
            evs, state = st.process_day(frame, date, state, cfg)
            all_events.append((date, [e["kind"] for e in evs]))
        return all_events, state

    full_events, full_state = replay(df)
    truncated_events, truncated_state = replay(df.iloc[:4])  # t=3(A2일)까지만

    assert full_events[:4] == truncated_events
    assert truncated_state["state"] == "확인"
