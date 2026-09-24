"""P3.4 1번: 9칸 매수 표의 비고 요약(_buy_stage_summary). 네트워크 없음."""

from __future__ import annotations

import pandas as pd

from engine.daily import _buy_stage_summary


def test_a1_note_shows_rsi_change():
    note = _buy_stage_summary("A1", {"rsi_prev": 25.1, "rsi_now": 31.2, "vol_ratio": 1.3}, None, None)
    assert note == "RSI 25.1 → 31.2"


def test_a1_note_flags_low_volume():
    note = _buy_stage_summary("A1", {"rsi_prev": 25.1, "rsi_now": 31.2, "vol_ratio": 0.7}, None, None)
    assert "거래량 부족(0.7배)" in note


def test_a2_note_shows_grade_and_macd_norm():
    note = _buy_stage_summary("A2", {"grade": "S", "macd_norm": -0.42}, None, None)
    assert "등급 S" in note
    assert "MACD 정규화 -0.42%" in note


def test_a3_note_shows_large_gap_only():
    note_small = _buy_stage_summary("A3", {"gap_pct": 0.5}, None, None)
    assert note_small == ""  # 1% 미만 갭은 굳이 언급하지 않는다
    note_large = _buy_stage_summary("A3", {"gap_pct": 2.3}, None, None)
    assert "시가 갭 +2.3%" in note_large


def test_b_note_always_mentions_reentry_risk():
    note = _buy_stage_summary("B", {"grade": "S"}, None, None)
    assert "재진입 1회 전량 · 위험 1%" in note
    assert "등급 S" in note


def test_earnings_within_two_weeks_is_flagged():
    as_of = pd.Timestamp("2026-09-23")
    earnings = pd.Timestamp("2026-10-01").date()  # 8일 뒤
    note = _buy_stage_summary("A1", {"rsi_prev": 25.0, "rsi_now": 31.0, "vol_ratio": 1.0}, earnings, as_of)
    assert "실적 임박" in note


def test_earnings_far_away_is_not_flagged():
    as_of = pd.Timestamp("2026-09-23")
    earnings = pd.Timestamp("2026-12-01").date()
    note = _buy_stage_summary("A1", {"rsi_prev": 25.0, "rsi_now": 31.0, "vol_ratio": 1.0}, earnings, as_of)
    assert "실적" not in note
