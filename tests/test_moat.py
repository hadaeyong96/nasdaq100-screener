"""core/moat.py 테스트 — 네트워크 없이, 합성 XBRL 데이터로 돈다.

미래 데이터 방지(annual_value_series의 filed<=as_of 필터), 등급 판정, 태그 대체,
누락 처리를 확인한다.
"""

from __future__ import annotations

from datetime import date

import pytest

from core import moat


def _cfg():
    return {
        "moat": {
            "lookback_years": 5,
            "min_annual_years_required": 5,
            "roic_tax_rate_pct": 21,
            "m1_roic": {"good_threshold_pct": 15, "good_min_years": 4, "caution_max_good_years": 1},
            "m2_gross_margin": {"std_good_max_pp": 3, "level_good_min_pct": 40, "std_caution_min_pp": 6, "level_caution_max_pct": 25},
            "m3_recession_resilience": {"drop_good_max_pp": 5, "drop_caution_min_pp": 10},
            "m4_cash_conversion": {"good_min_ratio": 0.8, "caution_max_ratio": 0.5},
            "m5_dilution": {"good_max_pct": 3, "caution_min_pct": 10},
            "outliers": {"roic_extreme_pct": 100},
        }
    }


def _entry(end, val, filed, fy, form="10-K", fp="FY", start=None):
    e = {"end": end, "val": val, "filed": filed, "fy": fy, "fp": fp, "form": form}
    if start:
        e["start"] = start
    return e


def _facts(**tags_by_concept_tag):
    """{"us-gaap:OperatingIncomeLoss": [entries...]} 형태를 companyfacts 구조로 감싼다."""
    facts: dict = {"facts": {}}
    for key, entries in tags_by_concept_tag.items():
        taxonomy, tag = key.split(":", 1)
        facts["facts"].setdefault(taxonomy, {})[tag] = {"units": {"USD": entries}}
    return facts


# ── annual_value_series: 미래 데이터 방지 ───────────────────────────────────


def test_annual_value_series_excludes_entries_filed_after_as_of():
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2019-12-31", 100, "2020-02-01", 2019),
            _entry("2020-12-31", 200, "2021-02-01", 2020),
        ]
    })
    series, tag = moat.annual_value_series(facts, "operating_income", date(2020, 6, 1))
    assert [r["end"] for r in series] == ["2019-12-31"]  # 2020년 값은 2021-02-01에 제출돼 미래
    assert tag == ("us-gaap", "OperatingIncomeLoss")


def test_annual_value_series_picks_latest_filed_before_as_of_for_restated_period():
    """같은 회계연도가 여러 번 제출되면(비교연도 재수록·정정) as_of 이전 중 가장 최근 것을 쓴다."""
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2019-12-31", 100, "2020-02-01", 2019),  # 최초 제출
            _entry("2019-12-31", 105, "2021-02-01", 2020),  # 다음 해 10-K가 비교연도로 재수록(정정)
            _entry("2019-12-31", 110, "2025-02-01", 2024),  # 훨씬 나중 — as_of보다 미래
        ]
    })
    series, _ = moat.annual_value_series(facts, "operating_income", date(2022, 1, 1))
    assert len(series) == 1
    assert series[0]["val"] == 105  # 2021-02-01 값(2022-01-01 이전, 가장 최근 제출)


def test_annual_value_series_ignores_non_annual_and_wrong_form():
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2019-12-31", 100, "2020-02-01", 2019, form="10-K", fp="FY"),
            _entry("2019-06-30", 40, "2019-08-01", 2019, form="10-Q", fp="Q2"),  # 분기
            _entry("2019-12-31", 999, "2020-02-01", 2019, form="8-K", fp="FY"),  # 연간이지만 10-K 아님
        ]
    })
    series, _ = moat.annual_value_series(facts, "operating_income", date(2021, 1, 1))
    assert series == [{"fy": 2019, "end": "2019-12-31", "val": 100, "filed": "2020-02-01", "form": "10-K"}]


def test_annual_value_series_missing_concept_returns_empty():
    series, tag = moat.annual_value_series(_facts(), "operating_income", date(2021, 1, 1))
    assert series == []
    assert tag is None


def test_annual_value_series_accepts_40f_canadian_annual_form():
    """SHOP(Shopify)은 40-F(캐나다 MJDS 연차보고서)로 공시한다 — 2026-10-01 실제 확인."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _entry("2023-12-31", 7000, "2024-02-13", 2023, form="40-F", fp="FY", start="2023-01-01"),
        ]
    })
    series, _ = moat.annual_value_series(facts, "revenue", date(2025, 1, 1))
    assert series == [{"fy": 2023, "end": "2023-12-31", "val": 7000, "filed": "2024-02-13", "form": "40-F"}]


def test_annual_value_series_filters_out_quarter_length_entries_mistagged_as_fy():
    """KLAC(회계연도 6월 말)에서 분기말 날짜가 form=10-K·fp=FY로 같이 섞여 나오는 문제
    (2026-10-01 실측) — (end-start)가 1년이 아니면 걸러낸다."""
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2020-06-30", 1000, "2020-08-01", 2020, form="10-K", fp="FY", start="2019-07-01"),  # 진짜 연간(365일)
            _entry("2020-09-30", 300, "2020-08-01", 2020, form="10-K", fp="FY", start="2020-07-01"),  # 분기(92일)인데 FY로 잘못 태깅
        ]
    })
    series, _ = moat.annual_value_series(facts, "operating_income", date(2021, 1, 1))
    assert [r["end"] for r in series] == ["2020-06-30"]


def test_annual_value_series_instant_concept_without_start_is_unaffected_by_period_check():
    """자기자본·현금 같은 시점(instant) 개념은 "start"가 없어 기간 검사를 안 받는다."""
    facts = _facts(**{"us-gaap:StockholdersEquity": [_entry("2020-06-30", 5000, "2020-08-01", 2020, form="10-K", fp="FY")]})
    series, _ = moat.annual_value_series(facts, "stockholders_equity", date(2021, 1, 1))
    assert len(series) == 1


def test_annual_value_series_skips_tag_with_only_quarterly_entries_for_next_candidate():
    """BKNG(Booking Holdings)는 우선순위가 높은 태그를 10-Q 각주에만 쓰고, 연간 합계는
    다음 후보 태그("Revenues")로 공시한다(2026-10-01 실측) — 사실이 있다는 것만으로
    태그를 확정하면(옛 extract_first_available 방식) 이 연간 값을 영영 못 찾는다."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _entry("2023-03-31", 100, "2023-05-01", 2023, form="10-Q", fp="Q1", start="2023-01-01"),
        ],
        "us-gaap:Revenues": [
            _entry("2023-12-31", 900, "2024-02-01", 2023, form="10-K", fp="FY", start="2023-01-01"),
        ],
    })
    series, tag = moat.annual_value_series(facts, "revenue", date(2025, 1, 1))
    assert tag == ("us-gaap", "Revenues")
    assert series[0]["val"] == 900


def test_annual_value_series_falls_back_to_ifrs_tag():
    facts = _facts(**{"ifrs-full:Revenue": [_entry("2020-12-31", 500, "2021-03-01", 2020, form="20-F")]})
    series, tag = moat.annual_value_series(facts, "revenue", date(2021, 6, 1))
    assert tag == ("ifrs-full", "Revenue")
    assert series[0]["val"] == 500


# ── 지표 계산 (5년치 합성 데이터) ────────────────────────────────────────────


def _make_years(base_end_year=2017, n=5):
    return [f"{base_end_year + i}-12-31" for i in range(n)]


def test_compute_m1_roic_good_when_most_years_above_threshold():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:StockholdersEquity": [_entry(e, 400, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": [_entry(e, 50, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    # ROIC = 100*(1-0.21)/(400-50) = 79/350 = 22.57% >= 15%, 5/5년
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"
    assert len(result.yearly_values) == 5


def test_compute_m1_roic_caution_when_almost_no_good_years():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [_entry(e, 10, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:StockholdersEquity": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": [_entry(e, 0, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    # ROIC = 10*0.79/1000 = 0.79% — 0/5년 15% 이상
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "주의"


def test_compute_m1_roic_insufficient_data_when_core_items_missing():
    result = moat.compute_m1_roic(_facts(), _cfg(), date(2099, 1, 1))
    assert result.status == "판단 불가"


def test_compute_m1_roic_skips_negative_invested_capital_years():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:StockholdersEquity": [_entry(e, -500, "2099-01-01", 2000 + i) for i, e in enumerate(ends[:3])]
        + [_entry(e, 400, "2099-01-01", 2000 + i) for i, e in enumerate(ends[3:], start=3)],
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": [_entry(e, 50, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    # 앞 3년은 자기자본이 음수라 invested_capital<=0 -> 건너뜀, 뒤 2년만 남음(< 3 최소치) -> 판단 불가
    assert result.status == "판단 불가"


def test_compute_m2_gross_margin_good_when_stable_and_high():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:GrossProfit": [_entry(e, 450, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m2_gross_margin(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"


def test_compute_m2_gross_margin_caution_when_volatile():
    ends = _make_years()
    margins = [10, 90, 10, 90, 10]
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:GrossProfit": [_entry(e, margins[i] * 10, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m2_gross_margin(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "주의"


def test_compute_m2_gross_margin_falls_back_to_revenue_minus_cost_of_revenue():
    """COST·SBUX·KLAC류(GrossProfit 태그 없음, 2026-10-01 확인) — 매출원가로 직접 계산."""
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:CostOfGoodsAndServicesSold": [_entry(e, 600, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m2_gross_margin(facts, _cfg(), date(2099, 12, 31))
    # (1000-600)/1000 = 40% 수준, std=0 -> 좋음
    assert result.status == "좋음"
    assert result.tags_used["cost_of_revenue"] == ("us-gaap", "CostOfGoodsAndServicesSold")


def test_compute_m3_recession_resilience_good_when_stable():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:OperatingIncomeLoss": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m3_recession_resilience(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"


def test_compute_m3_recession_resilience_caution_when_one_bad_year():
    ends = _make_years()
    op = [300, 300, 0, 300, 300]
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:OperatingIncomeLoss": [_entry(e, op[i], "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m3_recession_resilience(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "주의"


def test_compute_m4_cash_conversion_good():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": [_entry(e, 10, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    # (100-10)/100 = 0.9 >= 0.8
    result = moat.compute_m4_cash_conversion(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"


def test_compute_m4_cash_conversion_missing_capex_defaults_to_zero():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 90, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m4_cash_conversion(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"  # (90-0)/100 = 0.9


def test_compute_m5_dilution_good_when_shrinking():
    facts = _facts(**{
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [
            _entry("2020-12-31", 1000, "2099-01-01", 2020),
            _entry("2021-12-31", 990, "2099-01-01", 2021),
        ]
    })
    result = moat.compute_m5_dilution(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"


def test_compute_m5_dilution_caution_when_heavy_dilution():
    facts = _facts(**{
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [
            _entry("2020-12-31", 1000, "2099-01-01", 2020),
            _entry("2021-12-31", 1150, "2099-01-01", 2021),
        ]
    })
    result = moat.compute_m5_dilution(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "주의"


def test_compute_m5_dilution_insufficient_when_only_one_year():
    facts = _facts(**{"us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [_entry("2020-12-31", 1000, "2099-01-01", 2020)]})
    result = moat.compute_m5_dilution(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "판단 불가"


# ── 등급 판정 ────────────────────────────────────────────────────────────


def _ind(status):
    return moat.IndicatorResult(status=status)


def test_compute_moat_grade_wide_when_core_three_good_and_no_caution():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("좋음"), _ind("보통"), _ind("좋음"), _ind("보통"))
    assert grade == "넓음"


def test_compute_moat_grade_not_wide_if_m3_caution_even_with_core_good():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("좋음"), _ind("주의"), _ind("좋음"), _ind("보통"))
    assert grade == "좁음"  # M1,M2,M4 다 좋음(3개) -> 좁음 조건도 만족하지만 넓음 조건(M3 주의)엔 못 미침


def test_compute_moat_grade_narrow_when_two_of_three_good():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("좋음"), _ind("보통"), _ind("보통"), _ind("보통"))
    assert grade == "좁음"


def test_compute_moat_grade_none_when_less_than_two_good():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("보통"), _ind("보통"), _ind("주의"), _ind("보통"))
    assert grade == "없음"


# ── 이상치 탐지 ────────────────────────────────────────────────────────────


def test_detect_outliers_flags_extreme_roic_and_negative_equity():
    m1 = moat.IndicatorResult(status="좋음", yearly_values={"2020-12-31": 150.0, "2021-12-31": 20.0})
    equity_series = [{"end": "2020-12-31", "val": -500}, {"end": "2021-12-31", "val": 400}]
    out = moat.detect_outliers(m1, _cfg(), equity_series)
    assert any("ROIC" in o and "2020-12-31" in o for o in out)
    assert any("자기자본 음수" in o and "2020-12-31" in o for o in out)
    assert not any("2021-12-31" in o for o in out)


def test_detect_outliers_empty_when_nothing_unusual():
    m1 = moat.IndicatorResult(status="좋음", yearly_values={"2020-12-31": 20.0})
    assert moat.detect_outliers(m1, _cfg(), [{"end": "2020-12-31", "val": 400}]) == []


# ── analyze_company 통합 ────────────────────────────────────────────────────


def test_analyze_company_marks_insufficient_when_core_years_below_minimum():
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry("2020-12-31", 1000, "2099-01-01", 2020)],
        "us-gaap:OperatingIncomeLoss": [_entry("2020-12-31", 100, "2099-01-01", 2020)],
        "us-gaap:StockholdersEquity": [_entry("2020-12-31", 400, "2099-01-01", 2020)],
    })
    profile = moat.analyze_company(facts, "XYZ", _cfg(), date(2099, 12, 31))
    assert profile.grade == "판단 불가"
    assert profile.insufficient_data_reason is not None


def test_analyze_company_full_five_years_produces_real_grade():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:OperatingIncomeLoss": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:GrossProfit": [_entry(e, 450, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:StockholdersEquity": [_entry(e, 400, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": [_entry(e, 50, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 250, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [_entry(e, 1000 + i, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    profile = moat.analyze_company(facts, "GOOD", _cfg(), date(2099, 12, 31))
    assert profile.grade != "판단 불가"
    assert profile.insufficient_data_reason is None
    assert len(profile.data_years) == 5
