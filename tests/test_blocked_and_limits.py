"""P2.1 보완 1번: 매매 금지 구간·동시 보유 한도에 걸린 신호는 상태를 바꾸지
않는다. 네트워크 없이, tests/test_state.py의 합성 DataFrame 방식을 그대로 쓴다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core import state as st
from tests.test_state import cfg, make_df  # noqa: F401 (cfg는 pytest fixture로 재사용)


# ── 금지된 A2 → 정찰 유지, 유효기간 안에서 다시 판정 ─────────────────────


def test_blocked_a2_keeps_scouting_then_succeeds_on_later_cross(cfg):
    # check_a2 자체가 이미 "RSI 30~70"을 조건으로 요구해 RSI 70 이상 차단은 이
    # 경로로는 재현되지 않는다(도달 불가능). 대신 독립적인 차단 조건인 "실적
    # 발표 3거래일 이내"로 같은 시나리오(막힘 -> 유효기간 안 재판정 -> 성공)를 본다.
    rows = [
        {"close": 100, "rsi": 20},  # idx0: A1이 상향 돌파할 기준
        {"close": 100, "rsi": 32, "swing_low": 94},  # idx1: A1 (실적일까지 4거래일 남아 차단 안 됨)
        {"close": 100, "rsi": 40},  # idx2
        {"close": 100, "rsi": 40},  # idx3
        {"close": 100, "rsi": 40},  # idx4
        {"close": 100, "rsi": 45, "gc": True},  # idx5: 골든크로스, 실적 발표 당일 -> 차단
        {"close": 100, "rsi": 50},  # idx6: 그냥 지나가는 날
        {"close": 100, "rsi": 45, "gc": True},  # idx7: 다시 골든크로스, 이번엔 실적이 지나가 통과
    ]
    df = make_df(rows)
    earnings_date = df.index[5]  # idx5 당일이 실적 발표일 -> idx5는 3거래일 이내, idx7은 이미 지남

    state = st.init_state("TEST")
    all_events = []
    for date in df.index:
        events, state = st.process_day(df, date, state, cfg, earnings_date=earnings_date)
        all_events.append((date, events))

    kinds_by_day = [[e["kind"] for e in evs] for _, evs in all_events]
    assert kinds_by_day[1] == ["A1"]
    assert kinds_by_day[5] == ["BLOCKED"]
    assert all_events[5][1][0]["stage"] == "A2"
    assert all_events[5][1][0]["blocked_type"] == "ban"
    assert any("실적" in r for r in all_events[5][1][0]["reasons"])
    assert kinds_by_day[6] == []  # 그냥 지나가는 날
    assert kinds_by_day[7] == ["A2"]  # 유효기간(10거래일) 안에서 다시 성공
    assert state["state"] == "확인"


def test_blocked_a2_does_not_advance_state_or_units(cfg):
    rows = [
        {"close": 100, "rsi": 20},
        {"close": 100, "rsi": 32, "swing_low": 94},  # A1 (실적일까지 4거래일 남아 차단 안 됨)
        {"close": 100, "rsi": 40},
        {"close": 100, "rsi": 40},
        {"close": 100, "rsi": 40},
        {"close": 100, "rsi": 45, "gc": True},  # 차단된 A2 (실적 발표 당일)
    ]
    df = make_df(rows)
    earnings_date = df.index[5]
    state = st.init_state("TEST")
    for date in df.index:
        events, state = st.process_day(df, date, state, cfg, earnings_date=earnings_date)

    assert state["state"] == "정찰"  # 확인으로 넘어가지 않았다
    assert "2" not in state["units"]  # 묶음 2는 생기지 않았다(가상 체결 없음)


# ── 확인 상태에서 막힌 A3는 확인 유지, 다음 날 다시 판정 ────────────────────


def test_blocked_a3_keeps_confirming_state(cfg):
    state = st.init_state("TEST")
    state["state"] = "확인"
    state["grade"] = "A"

    # 구름 안(close가 cloud_top·cloud_bot 사이)이라 A3 조건 자체가 거짓이 되는 걸 피하려고,
    # A3의 4가지 조건은 모두 충족시키되 시가 갭 4% 이상으로 매매 금지에 걸리게 한다.
    df = make_df(
        [
            {
                "open": 125.0,  # 전일 종가(가상 100) 대비 5% 갭 — prev_close는 make_df 기본 close=NaN이라
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
    prev = make_df([{"close": 119.0, "rsi": 55.0}], start="2026-01-01")
    full = pd.concat([prev, df])

    events, new_state = st.process_day(full, full.index[-1], state, cfg)

    assert [e["kind"] for e in events] == ["BLOCKED"]
    assert events[0]["stage"] == "A3"
    assert new_state["state"] == "확인"  # 확정으로 넘어가지 않았다


# ── B가 막히면 대기 유지 ──────────────────────────────────────────────────


def test_blocked_b_keeps_waiting_state(cfg):
    # check_b도 이미 "RSI 50~70"을 조건에 포함해 RSI 70 차단은 도달 불가능하다.
    # 독립적인 차단 조건인 실적 발표 임박으로 같은 동작(대기 유지)을 본다.
    state = st.init_state("TEST")
    df = make_df(
        [
            {
                "close": 120.0,
                "cloud_top": 110.0,
                "future_yang": True,
                "chikou_ok": True,
                "gc": True,
                "macd_norm": -0.2,
                "rsi": 60.0,
            }
        ]
    )
    events, new_state = st.process_day(df, df.index[0], state, cfg, earnings_date=df.index[0])

    assert [e["kind"] for e in events] == ["BLOCKED"]
    assert events[0]["stage"] == "B"
    assert new_state["state"] == "대기"
    assert new_state["units"] == {}


# ── 동시 보유 한도(9번째 신규 A1) → blocked, 상태 전이 없음 ─────────────────


def test_new_entry_allowed_false_blocks_a1_without_transition(cfg):
    state = st.init_state("TEST")
    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")

    events, new_state = st.process_day(df, df.index[-1], state, cfg, new_entry_allowed=False)

    assert [e["kind"] for e in events] == ["BLOCKED"]
    assert events[0]["stage"] == "A1"
    assert events[0]["blocked_type"] == "limit"
    assert new_state["state"] == "대기"  # 정찰로 넘어가지 않았다
    assert new_state["a1_date"] is None
    assert new_state["units"] == {}  # 가상 체결도 없다


def test_preview_new_entry_returns_candidate_when_allowed(cfg):
    state = st.init_state("TEST")
    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")

    candidate = st.preview_new_entry(df, df.index[-1], state, cfg)
    assert candidate is not None
    assert candidate["kind"] == "A1"


def test_preview_new_entry_returns_none_when_banned():
    """실적 발표 3거래일 이내면 한도 계산에도 후보로 넣지 않는다."""
    cfg_local = {
        "assumptions": {
            "gap_filter_pct": 4.0,
            "whipsaw_max_crosses_20d": 4,
            "s_grade_macd_norm_min_pct": -0.5,
            "b_grade_macd_norm_max_pct": -2.0,
        }
    }
    state = st.init_state("TEST")
    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")

    earnings_date = df.index[-1] + pd.tseries.offsets.BDay(1)
    candidate = st.preview_new_entry(df, df.index[-1], state, cfg_local, earnings_date=earnings_date)
    assert candidate is None


def test_new_entry_allowed_true_after_slot_frees_up_still_works(cfg):
    """new_entry_allowed=True(기본값)면 예전처럼 정상적으로 A1이 발생한다."""
    state = st.init_state("TEST")
    df = make_df([{"close": 50.0, "rsi": 20.0}, {"close": 51.0, "rsi": 32.0}], start="2026-01-05")

    events, new_state = st.process_day(df, df.index[-1], state, cfg, new_entry_allowed=True)

    assert [e["kind"] for e in events] == ["A1"]
    assert new_state["state"] == "정찰"
