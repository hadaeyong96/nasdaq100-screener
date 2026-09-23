"""7장 필터·등급·매매 금지 구간 테스트. 네트워크 없이 돈다."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core import filters


def _row(**kwargs) -> pd.Series:
    defaults = {
        "close": 100.0,
        "open": 100.0,
        "rsi": 50.0,
        "cloud_top": 90.0,
        "cloud_bot": 80.0,
        "future_yang": True,
        "macd_norm": -1.0,
    }
    defaults.update(kwargs)
    return pd.Series(defaults, name=pd.Timestamp("2026-03-10"))


# ── RSI 70 이상 골든크로스는 A2·B형 진입 금지 ────────────────────────────


def test_rsi_70_blocks_b_type_entry(cfg):
    row = _row(rsi=72.0)
    reasons = filters.ban_reasons(stage="B", row=row, gc_count_20d=1, prev_close=100.0, cfg=cfg)
    assert any("RSI 70" in r for r in reasons)


def test_rsi_70_blocks_a2_entry(cfg):
    row = _row(rsi=71.0)
    reasons = filters.ban_reasons(stage="A2", row=row, gc_count_20d=1, prev_close=100.0, cfg=cfg)
    assert any("RSI 70" in r for r in reasons)


def test_rsi_below_70_does_not_block(cfg):
    row = _row(rsi=65.0)
    reasons = filters.ban_reasons(stage="B", row=row, gc_count_20d=1, prev_close=100.0, cfg=cfg)
    assert reasons == []


# ── 최근 20거래일 MACD 교차 4회 이상(휩소) → A2·B형 차단 ───────────────────


def test_whipsaw_4_crosses_blocks_a2(cfg):
    row = _row()
    reasons = filters.ban_reasons(stage="A2", row=row, gc_count_20d=4, prev_close=100.0, cfg=cfg)
    assert any("교차" in r for r in reasons)


def test_whipsaw_below_threshold_does_not_block(cfg):
    row = _row()
    reasons = filters.ban_reasons(stage="A2", row=row, gc_count_20d=3, prev_close=100.0, cfg=cfg)
    assert reasons == []


def test_macd_cross_count_counts_gc_and_dc():
    idx = pd.bdate_range("2026-01-02", periods=25)
    gc = [False] * 25
    dc = [False] * 25
    for i in (5, 10, 15, 20):
        gc[i] = True
    df = pd.DataFrame({"gc": gc, "dc": dc}, index=idx)
    # 최근 20거래일(인덱스 4~23) 안에는 5,10,15,20 네 번의 교차가 있다.
    assert filters.macd_cross_count(df, idx[23], window=20) == 4


# ── 실적 발표 3거래일 이내 → 신규 매수 차단 ────────────────────────────────


def test_earnings_within_3_trading_days_blocks_entry(cfg):
    row = _row()
    earnings_date = pd.Timestamp("2026-03-12")  # 화요일, date(월)로부터 1거래일 후
    reasons = filters.ban_reasons(
        stage="A2", row=row, gc_count_20d=1, prev_close=100.0, cfg=cfg, earnings_date=earnings_date
    )
    assert any("실적" in r for r in reasons)


def test_earnings_far_away_does_not_block(cfg):
    row = _row()
    earnings_date = pd.Timestamp("2026-04-01")
    reasons = filters.ban_reasons(
        stage="A2", row=row, gc_count_20d=1, prev_close=100.0, cfg=cfg, earnings_date=earnings_date
    )
    assert reasons == []


def test_earnings_unknown_does_not_block():
    """실적일을 모르면(None) 막지 않는다 — 호출부가 earnings_unknown으로 표시만 한다."""
    assert filters.is_earnings_within(pd.Timestamp("2026-03-10"), None) is False


# ── 등급(S/A/B) ──────────────────────────────────────────────────────────


def test_grade_s_when_macd_norm_near_zero(cfg):
    assert filters.grade(-0.3, gc_count_20d=1, cfg=cfg) == "S"


def test_grade_a_when_first_cross_in_range(cfg):
    assert filters.grade(-1.0, gc_count_20d=1, cfg=cfg) == "A"


def test_grade_b_when_not_first_cross_even_if_in_a_range(cfg):
    assert filters.grade(-1.0, gc_count_20d=2, cfg=cfg) == "B"


def test_grade_b_when_macd_norm_too_negative(cfg):
    assert filters.grade(-3.0, gc_count_20d=1, cfg=cfg) == "B"


def test_grade_none_when_nan(cfg):
    assert filters.grade(float("nan"), gc_count_20d=1, cfg=cfg) is None


# ── 구름 안/음운 금지 (A3, B) ───────────────────────────────────────────


def test_close_inside_cloud_blocks_a3():
    row = _row(close=85.0, cloud_top=90.0, cloud_bot=80.0)
    cfg = {"assumptions": {"whipsaw_max_crosses_20d": 4, "gap_filter_pct": 4.0}}
    reasons = filters.ban_reasons(stage="A3", row=row, gc_count_20d=0, prev_close=85.0, cfg=cfg)
    assert any("구름 안" in r for r in reasons)


def test_future_yin_blocks_a3():
    row = _row(future_yang=False)
    cfg = {"assumptions": {"whipsaw_max_crosses_20d": 4, "gap_filter_pct": 4.0}}
    reasons = filters.ban_reasons(stage="A3", row=row, gc_count_20d=0, prev_close=100.0, cfg=cfg)
    assert any("음운" in r for r in reasons)


def test_gap_filter_blocks_a3():
    row = _row(open=105.0)  # 전일 종가 대비 5% 갭
    cfg = {"assumptions": {"whipsaw_max_crosses_20d": 4, "gap_filter_pct": 4.0}}
    reasons = filters.ban_reasons(stage="A3", row=row, gc_count_20d=0, prev_close=100.0, cfg=cfg)
    assert any("갭" in r for r in reasons)


def test_nan_condition_does_not_block():
    """구름 값이 NaN이면(데이터 부족) 금지하지 않는다."""
    row = _row(cloud_top=np.nan, cloud_bot=np.nan)
    cfg = {"assumptions": {"whipsaw_max_crosses_20d": 4, "gap_filter_pct": 4.0}}
    reasons = filters.ban_reasons(stage="A3", row=row, gc_count_20d=0, prev_close=100.0, cfg=cfg)
    assert reasons == []
