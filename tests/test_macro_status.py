"""core.macro_status 테스트. 🟢안정·🟡주의·🔴위험 3단계 판정 경계값(config.yaml 기준값 그대로)
+ 3칸 눈금(★) + 스파크라인 좌표."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core import macro_status as ms

_ROOT = Path(__file__).resolve().parents[1]
with open(_ROOT / "config.yaml", encoding="utf-8") as _f:
    TH = yaml.safe_load(_f)["macro"]["thresholds"]

OK, WARN, BAD = "안정", "주의", "위험"


def _label(code, value, **kw):
    return ms.classify_macro(code, value, TH, **kw)["label"]


def test_config_thresholds_match_agreed_values():
    """기준값은 코드가 아니라 config.yaml에 있다 — 합의한 값인지 한 번에 확인."""
    assert TH["FEAR_GREED"] == {"extreme_fear": 25, "fear": 45, "neutral_high": 55, "greed": 75}
    assert TH["DGS10"] == {"caution": 4.0, "danger": 4.5}
    assert TH["T10Y2Y"] == {"stable": 0.5, "danger": 0}
    assert TH["BAMLH0A0HYM2"] == {"caution": 4, "danger": 6}
    assert TH["DFEDTARU"] == {"lookback_months": 6}
    assert TH["DEXKOUS"] == {"caution": 1300, "danger": 1400}
    assert TH["VIXCLS"] == {"caution": 20, "danger": 30}


@pytest.mark.parametrize(
    "value,label,zone",
    [
        (0, BAD, "극단적 공포"), (24, BAD, "극단적 공포"), (25, WARN, "공포"), (44, WARN, "공포"),
        (45, OK, "중립"), (55, OK, "중립"), (56, WARN, "탐욕"), (75, WARN, "탐욕"),
        (76, BAD, "극단적 탐욕"), (100, BAD, "극단적 탐욕"),
    ],
)
def test_fear_greed_boundaries_and_zone_names(value, label, zone):
    out = ms.classify_macro("FEAR_GREED", value, TH)
    assert (out["label"], out["zone"]) == (label, zone)


@pytest.mark.parametrize("value,label", [(3.99, OK), (4.0, WARN), (4.49, WARN), (4.5, BAD), (5.17, BAD)])
def test_dgs10_boundaries(value, label):
    assert _label("DGS10", value) == label


@pytest.mark.parametrize("value,label", [(0.5, OK), (1.2, OK), (0.49, WARN), (0.0, WARN), (-0.01, BAD)])
def test_t10y2y_boundaries(value, label):
    assert _label("T10Y2Y", value) == label


@pytest.mark.parametrize("value,label", [(3.99, OK), (4, WARN), (5.99, WARN), (6, BAD)])
def test_hy_spread_boundaries(value, label):
    assert _label("BAMLH0A0HYM2", value) == label


@pytest.mark.parametrize("value,label", [(1299.9, OK), (1300, WARN), (1399.9, WARN), (1400, BAD)])
def test_usdkrw_boundaries(value, label):
    assert _label("DEXKOUS", value) == label


@pytest.mark.parametrize("value,label", [(19.99, OK), (20, WARN), (29.99, WARN), (30, BAD)])
def test_vix_boundaries(value, label):
    assert _label("VIXCLS", value) == label


def test_fed_funds_cut_hold_hike_over_six_months():
    assert _label("DFEDTARU", 4.25, value_before=4.50) == OK  # 인하
    assert _label("DFEDTARU", 4.50, value_before=4.50) == WARN  # 동결
    assert _label("DFEDTARU", 4.75, value_before=4.50) == BAD  # 인상
    out = ms.classify_macro("DFEDTARU", 4.50, TH, value_before=None)
    assert out["status"] == ms.STATUS_INFO and not any(r["current"] for r in out["scale"])


def test_value_months_ago_picks_last_value_on_or_before_target():
    series = {"2026-03-15": 4.75, "2026-03-31": 4.5, "2026-04-15": 4.25, "2026-09-29": 4.0}
    assert ms.value_months_ago(series, "2026-09-29", 6) == 4.75  # 목표일 03-29 이하 마지막 값(03-15)
    assert ms.value_months_ago(series, "2026-10-01", 6) == 4.5  # 목표일 04-01 -> 03-31 값
    assert ms.value_months_ago({"2026-02-27": 3.0}, "2026-08-31", 6) == 3.0  # 2월 말일로 맞춤
    assert ms.value_months_ago({"2026-09-01": 1.0}, "2026-09-29", 6) is None


@pytest.mark.parametrize("code", ["FEAR_GREED", "VIXCLS", "DGS10", "T10Y2Y", "BAMLH0A0HYM2", "DEXKOUS"])
def test_scale_has_three_levels_in_order_with_single_star(code):
    probe = {"FEAR_GREED": 60, "T10Y2Y": 0.2}.get(code, TH[code].get("caution", 0) + 0.01)
    out = ms.classify_macro(code, probe, TH)
    assert [r["label"] for r in out["scale"]] == [OK, WARN, BAD]
    assert [r["symbol"] for r in out["scale"]] == ["🟢", "🟡", "🔴"]
    assert sum(r["current"] for r in out["scale"]) == 1
    assert next(r for r in out["scale"] if r["current"])["label"] == out["label"] == WARN
    assert out["text"] == "🟡 주의"


def test_scale_range_texts_match_spec():
    assert [r["range"] for r in ms.scale_ranges("DGS10", TH)] == ["4.0% 미만", "4.0~4.5%", "4.5% 이상"]
    assert [r["range"] for r in ms.scale_ranges("T10Y2Y", TH)] == ["+0.5%p 이상", "0~+0.5%p", "0 미만 (역전)"]
    assert [r["range"] for r in ms.scale_ranges("DEXKOUS", TH)] == ["1,300원 미만", "1,300~1,400원", "1,400원 이상"]
    assert [r["range"] for r in ms.scale_ranges("FEAR_GREED", TH)][0] == "45~55 (중립)"


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
