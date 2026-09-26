"""core.macro_status 테스트 (P3.8). 상태 판정 경계값 + 스파크라인 좌표."""

from __future__ import annotations

import pytest

from core import macro_status as ms

FG_TH = {"extreme_fear": 25, "fear": 45, "neutral_high": 55, "greed": 75}
VIX_TH = {"caution": 20, "alert": 30}
T10Y2Y_TH = {"normal": 0.5, "alert": 0}
HY_TH = {"caution": 4, "alert": 6}
FX_TH = {"pct_high": 80, "pct_low": 20}


@pytest.mark.parametrize(
    "value,expected_label",
    [(24, "극단적 공포"), (25, "공포"), (44, "공포"), (45, "중립"), (55, "중립"), (56, "탐욕"), (75, "탐욕"), (76, "극단적 탐욕")],
)
def test_classify_fear_greed_boundaries(value, expected_label):
    assert ms.classify_fear_greed(value, FG_TH)["label"] == expected_label


@pytest.mark.parametrize("value,expected_label", [(19.99, "안정"), (20, "주의"), (29.99, "주의"), (30, "경계")])
def test_classify_vix_fallback_boundaries(value, expected_label):
    assert ms.classify_vix_fallback(value, VIX_TH)["label"] == expected_label


def test_classify_rise_over_window_boundary_exact_half_point():
    assert ms.classify_rise_over_window(4.5, 4.0, 0.5)["label"] == "주의"  # 정확히 +0.5%p
    assert ms.classify_rise_over_window(4.49, 4.0, 0.5)["label"] == "안정"


def test_classify_rise_over_window_no_comparison_value():
    out = ms.classify_rise_over_window(4.5, None, 0.5)
    assert out["status"] == ms.STATUS_INFO


@pytest.mark.parametrize("value,expected_label", [(0.5, "정상"), (0.49, "주의"), (0, "주의"), (-0.01, "경계(역전)")])
def test_classify_t10y2y_boundaries(value, expected_label):
    assert ms.classify_t10y2y(value, T10Y2Y_TH)["label"] == expected_label


@pytest.mark.parametrize("value,expected_label", [(3.99, "안정"), (4, "주의"), (5.99, "주의"), (6, "경계")])
def test_classify_hy_spread_boundaries(value, expected_label):
    assert ms.classify_hy_spread(value, HY_TH)["label"] == expected_label


def test_classify_fed_funds_trend_up_down_flat():
    assert ms.classify_fed_funds_trend(4.75, 4.50)["label"] == "인상 흐름"
    assert ms.classify_fed_funds_trend(4.25, 4.50)["label"] == "인하 흐름"
    assert ms.classify_fed_funds_trend(4.50, 4.50)["label"] == "동결"


def test_classify_fed_funds_trend_no_comparison_value():
    assert ms.classify_fed_funds_trend(4.50, None)["status"] == ms.STATUS_INFO


def test_percentile_rank_basic():
    series = [10, 20, 30, 40, 50]
    assert ms.percentile_rank(series, 30) == 60.0  # 10,20,30 <= 30 -> 3/5
    assert ms.percentile_rank(series, 5) == 0.0
    assert ms.percentile_rank(series, 50) == 100.0


def test_percentile_rank_empty_series_is_neutral():
    assert ms.percentile_rank([], 100) == 50.0


def test_classify_fx_percentile_boundaries():
    series = list(range(1, 101))  # 1..100, value=v -> percentile=v%
    assert ms.classify_fx_percentile(80, series, FX_TH)["label"] == "달러 비쌈"
    assert ms.classify_fx_percentile(79, series, FX_TH)["label"] == "보통"
    assert ms.classify_fx_percentile(20, series, FX_TH)["label"] == "달러 쌈"
    assert ms.classify_fx_percentile(21, series, FX_TH)["label"] == "보통"


# ── 지연("stale") 표시 ────────────────────────────────────────────────────


def test_is_stale_within_limit_is_false():
    trading_days = lambda a, b: ["d0", "d1", "d2", "d3"]  # 3거래일 차이(gap=len-1=3)
    assert ms.is_stale("2026-09-20", "2026-09-25", stale_days=5, trading_days_between=trading_days) is False


def test_is_stale_beyond_limit_is_true():
    trading_days = lambda a, b: ["d"] * 8  # gap=7
    assert ms.is_stale("2026-09-10", "2026-09-25", stale_days=5, trading_days_between=trading_days) is True


def test_is_stale_future_as_of_is_false():
    assert ms.is_stale("2026-09-26", "2026-09-25", stale_days=5, trading_days_between=lambda a, b: []) is False


# ── 스파크라인 좌표 ──────────────────────────────────────────────────────────


def test_sparkline_points_no_division_by_zero_when_flat():
    points = ms.sparkline_points([50.0, 50.0, 50.0])
    ys = [float(p.split(",")[1]) for p in points.split(" ")]
    assert all(y == pytest.approx(17.0) for y in ys)  # height/2 = 34/2 = 17


def test_sparkline_points_empty_is_empty_string():
    assert ms.sparkline_points([]) == ""


def test_sparkline_points_endpoints_use_full_range():
    points = ms.sparkline_points([0.0, 100.0]).split(" ")
    y0 = float(points[0].split(",")[1])
    y1 = float(points[1].split(",")[1])
    assert y0 > y1  # 낮은 값이 아래(y가 큼), 높은 값이 위(y가 작음)


def test_value_to_y_flat_series_returns_midpoint():
    assert ms.value_to_y(5.0, 5.0, 5.0, height=34.0) == 17.0
