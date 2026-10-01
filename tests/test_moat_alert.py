"""core/moat_alert.py 테스트 — 네트워크 없이, 합성 XBRL 데이터로 돈다 (해자 약화 경보 B3)."""

from __future__ import annotations

from datetime import date

import pytest

from core import moat, moat_alert


def _cfg():
    return {
        "moat": {
            "lookback_years": 5,
            "min_annual_years_required": 5,
            "roic_tax_rate_pct": 21,
            "revenue_tag_mismatch_tolerance_pct": 5,
            "revenue_tag_mismatch_review_threshold_pct": 50,
            "m1_roic": {"good_threshold_pct": 15, "good_min_years": 4, "caution_max_good_years": 1, "min_invested_capital_ratio_of_revenue": 0.02},
            "m2_gross_margin": {"std_good_max_pp": 3, "std_caution_min_pp": 6},
            "m3_recession_resilience": {"drop_good_max_pp": 5, "drop_caution_min_pp": 10},
            "m4_cash_conversion": {"good_min_ratio": 0.9, "caution_max_ratio": 0.5, "reference_good_min_ratio": 0.8},
            "m5_dilution": {"good_max_pct": 3, "caution_min_pct": 10},
        }
    }


def _q(end, val, filed, start, form="10-Q", fp="Q1"):
    return {"end": end, "val": val, "filed": filed, "start": start, "form": form, "fp": fp}


def _annual(end, val, filed, start, fy):
    return {"end": end, "val": val, "filed": filed, "start": start, "form": "10-K", "fp": "FY", "fy": fy}


def _facts(**tags_by_concept_tag):
    facts: dict = {"facts": {}}
    for key, entries in tags_by_concept_tag.items():
        taxonomy, tag = key.split(":", 1)
        facts["facts"].setdefault(taxonomy, {})[tag] = {"units": {"USD": entries}}
    return facts


# ── 분기 추출·Q4 계산 ────────────────────────────────────────────────────────


def test_quarterly_concept_series_picks_3month_only():
    facts = _facts(**{
        "us-gaap:Revenues": [
            _q("2026-03-31", 100, "2026-05-01", "2026-01-01"),  # 90일
            _q("2026-06-30", 200, "2026-08-01", "2026-01-01", fp="Q2"),  # 6개월 누적(181일) - 제외
        ]
    })
    series = moat_alert.quarterly_concept_series(facts, "revenue", date(2026, 9, 30))
    assert [r["end"] for r in series] == ["2026-03-31"]
    assert series[0]["val"] == 100


def test_quarterly_series_with_q4_derives_from_annual_minus_9mo():
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _q("2026-03-31", 100, "2026-05-01", "2026-01-01"),
            _q("2026-06-30", 110, "2026-08-01", "2026-04-01", fp="Q2"),
            _q("2026-09-30", 120, "2026-11-01", "2026-07-01", fp="Q3"),
            # 9개월 누적(2026-01-01~2026-09-30, 273일)
            {"end": "2026-09-30", "val": 330, "filed": "2026-11-01", "start": "2026-01-01", "form": "10-Q", "fp": "Q3"},
            # 연간(FY, 2026-01-01~2026-12-31)
            _annual("2026-12-31", 470, "2027-02-01", "2026-01-01", 2026),
        ]
    })
    series = moat_alert.quarterly_series_with_q4(facts, "revenue", date(2027, 3, 1))
    by_end = {r["end"]: r["val"] for r in series}
    assert by_end["2026-03-31"] == 100
    assert by_end["2026-06-30"] == 110
    # Q4 = 연간(470) - 9개월 누적(330) = 140
    assert by_end["2026-12-31"] == 140


def test_quarterly_series_with_q4_uses_reconciled_revenue_when_tags_mismatch():
    # 실측 버그(MELI, 2026-10-01): 연간 매출 후보 태그 두 개가 30% 가까이 달라서, 보정 없이
    # 낮은 쪽 태그를 그대로 쓰면 Q4(연간-9개월누적)가 터무니없이 작게 나왔다.
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _q("2026-03-31", 100, "2026-05-01", "2026-01-01"),
            _q("2026-06-30", 110, "2026-08-01", "2026-04-01", fp="Q2"),
            _q("2026-09-30", 120, "2026-11-01", "2026-07-01", fp="Q3"),
            {"end": "2026-09-30", "val": 330, "filed": "2026-11-01", "start": "2026-01-01", "form": "10-Q", "fp": "Q3"},
            _annual("2026-12-31", 350, "2027-02-01", "2026-01-01", 2026),  # 낮은 태그 — 9개월 누적과 거의 같음(보정 전이면 Q4=20)
        ],
        "us-gaap:Revenues": [
            _annual("2026-12-31", 470, "2027-02-01", "2026-01-01", 2026),  # 높은 태그(실제 전사 매출에 가까움)
        ],
    })
    series = moat_alert.quarterly_series_with_q4(facts, "revenue", date(2027, 3, 1), cfg=_cfg())
    by_end = {r["end"]: r["val"] for r in series}
    assert by_end["2026-12-31"] == 140  # 470(보정된 연간, 큰 값) - 330(9개월 누적) = 140


def test_quarterly_series_with_q4_skips_when_no_cumulative_match():
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _annual("2026-12-31", 470, "2027-02-01", "2026-01-01", 2026),
        ]
    })
    series = moat_alert.quarterly_series_with_q4(facts, "revenue", date(2027, 3, 1))
    assert series == []  # 9개월 누적을 못 찾아 Q4 계산 불가 -> 그냥 없음


# ── 이익률 경보 ──────────────────────────────────────────────────────────────


def test_gross_margin_alert_triggers_on_both_conditions():
    series = [
        {"end": "2025-09-30", "margin_pct": 42.0},  # 전년 동분기
        {"end": "2025-12-31", "margin_pct": 40.0},
        {"end": "2026-06-30", "margin_pct": 38.0},
        {"end": "2026-09-30", "margin_pct": 36.0},  # 전년동분기(42.0) 대비 -6.0%p, 2분기 연속 하락
    ]
    result = moat_alert.gross_margin_alert(series)
    assert result["status"] == "경보"
    assert result["consecutive_decline"] is True
    assert result["yoy_drop_pp"] == pytest.approx(6.0)


def test_gross_margin_alert_not_triggered_when_only_consecutive_decline():
    series = [
        {"end": "2025-09-30", "margin_pct": 37.0},  # 전년동분기: 36보다 낮아 하락폭 음수(오히려 상승)
        {"end": "2025-12-31", "margin_pct": 40.0},
        {"end": "2026-06-30", "margin_pct": 38.0},
        {"end": "2026-09-30", "margin_pct": 36.0},  # 2분기 연속 하락은 맞지만 YoY 조건 미충족
    ]
    result = moat_alert.gross_margin_alert(series)
    assert result["status"] == "경보 아님"
    assert result["consecutive_decline"] is True


def test_gross_margin_alert_not_triggered_when_only_yoy_drop():
    series = [
        {"end": "2025-09-30", "margin_pct": 42.0},
        {"end": "2025-12-31", "margin_pct": 30.0},
        {"end": "2026-06-30", "margin_pct": 35.0},  # 직전보다 상승 -> 연속 하락 아님
        {"end": "2026-09-30", "margin_pct": 36.0},  # 전년동분기 대비는 -6.0%p로 충족하지만 연속하락 아님
    ]
    result = moat_alert.gross_margin_alert(series)
    assert result["status"] == "경보 아님"
    assert result["consecutive_decline"] is False


def test_gross_margin_alert_insufficient_quarters_is_inconclusive():
    series = [{"end": "2026-06-30", "margin_pct": 38.0}, {"end": "2026-09-30", "margin_pct": 36.0}]
    assert moat_alert.gross_margin_alert(series)["status"] == "판단 불가"


def test_gross_margin_alert_no_yoy_match_is_inconclusive():
    series = [
        {"end": "2026-03-31", "margin_pct": 40.0},
        {"end": "2026-06-30", "margin_pct": 38.0},
        {"end": "2026-09-30", "margin_pct": 36.0},
    ]
    assert moat_alert.gross_margin_alert(series)["status"] == "판단 불가"


# ── 수익성 경보 ──────────────────────────────────────────────────────────────


def test_profitability_alert_triggers_below_threshold():
    m1 = moat.IndicatorResult(status="보통", yearly_values={"2024-12-31": 20.0, "2025-12-31": 12.0})
    result = moat_alert.profitability_alert(m1, _cfg())
    assert result["status"] == "경보"


def test_profitability_alert_not_triggered_above_threshold():
    m1 = moat.IndicatorResult(status="좋음", yearly_values={"2024-12-31": 20.0, "2025-12-31": 18.0})
    result = moat_alert.profitability_alert(m1, _cfg())
    assert result["status"] == "경보 아님"


def test_profitability_alert_no_data_is_inconclusive():
    m1 = moat.IndicatorResult(status="판단 불가")
    assert moat_alert.profitability_alert(m1, _cfg())["status"] == "판단 불가"


# ── 등급 경보 ────────────────────────────────────────────────────────────────


def test_grade_alert_triggers_when_downgraded_from_wide():
    assert moat_alert.grade_alert("넓음", "좁음")["status"] == "경보"
    assert moat_alert.grade_alert("넓음", "없음")["status"] == "경보"


def test_grade_alert_not_triggered_when_staying_wide_or_never_wide():
    assert moat_alert.grade_alert("넓음", "넓음")["status"] == "경보 아님"
    assert moat_alert.grade_alert("좁음", "없음")["status"] == "경보 아님"


# ── 미래 데이터 방지 ──────────────────────────────────────────────────────────


def test_quarterly_concept_series_future_data_excluded():
    facts = _facts(**{
        "us-gaap:Revenues": [
            _q("2026-06-30", 200, "2026-08-01", "2026-04-01", fp="Q2"),
            _q("2026-09-30", 999, "2026-11-01", "2026-07-01", fp="Q3"),  # as_of 이후 filed
        ]
    })
    full = moat_alert.quarterly_concept_series(facts, "revenue", date(2027, 1, 1))
    cut = moat_alert.quarterly_concept_series(facts, "revenue", date(2026, 9, 1))
    assert [r["end"] for r in cut] == ["2026-06-30"]
    # t시점(2026-09-01)까지 자른 결과와, 전체 데이터에서 같은 as_of로 다시 구한 값이 같다
    common_end = "2026-06-30"
    full_val = next(r["val"] for r in full if r["end"] == common_end)
    cut_val = next(r["val"] for r in cut if r["end"] == common_end)
    assert full_val == cut_val
