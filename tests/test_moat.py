"""core/moat.py 테스트 — 네트워크 없이, 합성 XBRL 데이터로 돈다.

미래 데이터 방지(annual_value_series의 filed<=as_of 필터), 여러 태그 병합, 영업이익 대체
계산, 새 M1 정의(순운전자본+순유형자산), M2 수준 기준 제거, 등급 판정, 누락 처리를
확인한다.
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
            "revenue_tag_mismatch_tolerance_pct": 5,
            "revenue_tag_mismatch_review_threshold_pct": 50,
            "m1_roic": {"good_threshold_pct": 15, "good_min_years": 4, "caution_max_good_years": 1, "min_invested_capital_ratio_of_revenue": 0.02},
            "m2_gross_margin": {"std_good_max_pp": 3, "std_caution_min_pp": 6},
            "m3_recession_resilience": {"drop_good_max_pp": 5, "drop_caution_min_pp": 10},
            "m4_cash_conversion": {"good_min_ratio": 0.9, "caution_max_ratio": 0.5, "reference_good_min_ratio": 0.8},
            "m5_dilution": {"good_max_pct": 3, "caution_min_pct": 10},
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


def _make_years(base_end_year=2017, n=5):
    return [f"{base_end_year + i}-12-31" for i in range(n)]


# ── annual_value_series: 미래 데이터 방지 ───────────────────────────────────


def test_annual_value_series_excludes_entries_filed_after_as_of():
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2019-12-31", 100, "2020-02-01", 2019),
            _entry("2020-12-31", 200, "2021-02-01", 2020),
        ]
    })
    series = moat.annual_value_series(facts, "operating_income", date(2020, 6, 1))
    assert [r["end"] for r in series] == ["2019-12-31"]  # 2020년 값은 2021-02-01에 제출돼 미래
    assert series[0]["tag"] == ("us-gaap", "OperatingIncomeLoss")


def test_annual_value_series_picks_latest_filed_before_as_of_for_restated_period():
    """같은 회계연도가 여러 번 제출되면(비교연도 재수록·정정) as_of 이전 중 가장 최근 것을 쓴다."""
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2019-12-31", 100, "2020-02-01", 2019),  # 최초 제출
            _entry("2019-12-31", 105, "2021-02-01", 2020),  # 다음 해 10-K가 비교연도로 재수록(정정)
            _entry("2019-12-31", 110, "2025-02-01", 2024),  # 훨씬 나중 — as_of보다 미래
        ]
    })
    series = moat.annual_value_series(facts, "operating_income", date(2022, 1, 1))
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
    series = moat.annual_value_series(facts, "operating_income", date(2021, 1, 1))
    assert len(series) == 1
    assert series[0]["val"] == 100
    assert series[0]["end"] == "2019-12-31"


def test_annual_value_series_missing_concept_returns_empty():
    assert moat.annual_value_series(_facts(), "operating_income", date(2021, 1, 1)) == []


def test_annual_value_series_accepts_40f_canadian_annual_form():
    """SHOP(Shopify)은 40-F(캐나다 MJDS 연차보고서)로 공시한다 — 2026-10-01 실제 확인."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _entry("2023-12-31", 7000, "2024-02-13", 2023, form="40-F", fp="FY", start="2023-01-01"),
        ]
    })
    series = moat.annual_value_series(facts, "revenue", date(2025, 1, 1))
    assert len(series) == 1
    assert series[0]["val"] == 7000
    assert series[0]["tag"] == ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax")


def test_annual_value_series_filters_out_quarter_length_entries_mistagged_as_fy():
    """KLAC(회계연도 6월 말)에서 분기말 날짜가 form=10-K·fp=FY로 같이 섞여 나오는 문제
    (2026-10-01 실측) — (end-start)가 1년이 아니면 걸러낸다."""
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [
            _entry("2020-06-30", 1000, "2020-08-01", 2020, form="10-K", fp="FY", start="2019-07-01"),  # 진짜 연간(365일)
            _entry("2020-09-30", 300, "2020-08-01", 2020, form="10-K", fp="FY", start="2020-07-01"),  # 분기(92일)인데 FY로 잘못 태깅
        ]
    })
    series = moat.annual_value_series(facts, "operating_income", date(2021, 1, 1))
    assert [r["end"] for r in series] == ["2020-06-30"]


def test_annual_value_series_instant_concept_without_start_is_unaffected_by_period_check():
    """자기자본·현금 같은 시점(instant) 개념은 "start"가 없어 기간 검사를 안 받는다."""
    facts = _facts(**{"us-gaap:StockholdersEquity": [_entry("2020-06-30", 5000, "2020-08-01", 2020, form="10-K", fp="FY")]})
    series = moat.annual_value_series(facts, "stockholders_equity", date(2021, 1, 1))
    assert len(series) == 1


def test_annual_value_series_merges_tags_across_years():
    """AVGO 실측(2026-10-01): "StockholdersEquity"는 2019년까지, 2023년부터는
    "...IncludingPortionAttributableToNoncontrollingInterest" — 두 태그를 병합해야
    전체 기간이 나온다. 같은 연도에 둘 다 있으면 우선순위가 높은 쪽이 이긴다."""
    facts = _facts(**{
        "us-gaap:StockholdersEquity": [
            _entry("2019-12-31", 100, "2020-02-01", 2019),
            _entry("2020-12-31", 999, "2021-02-01", 2020),  # 우선순위 높은 태그 값 — 이게 이겨야 함
        ],
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": [
            _entry("2020-12-31", 111, "2021-02-01", 2020),  # 같은 연도, 우선순위 낮은 태그
            _entry("2023-12-31", 300, "2024-02-01", 2023),  # 2023년은 이 태그로만 있음
        ],
    })
    series = moat.annual_value_series(facts, "stockholders_equity", date(2025, 1, 1))
    by_end = {r["end"]: r for r in series}
    assert by_end["2019-12-31"]["val"] == 100
    assert by_end["2020-12-31"]["val"] == 999  # 우선순위 높은 태그가 이김
    assert by_end["2020-12-31"]["tag"] == ("us-gaap", "StockholdersEquity")
    assert by_end["2023-12-31"]["val"] == 300  # 우선순위 높은 태그엔 없어서 대체 태그로 채워짐
    assert by_end["2023-12-31"]["tag"] == ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")


def test_annual_value_series_skips_tag_with_only_quarterly_entries_for_next_candidate():
    """BKNG(Booking Holdings)는 우선순위가 높은 태그를 10-Q 각주에만 쓰고, 연간 합계는
    다음 후보 태그("Revenues")로 공시한다(2026-10-01 실측)."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _entry("2023-03-31", 100, "2023-05-01", 2023, form="10-Q", fp="Q1", start="2023-01-01"),
        ],
        "us-gaap:Revenues": [
            _entry("2023-12-31", 900, "2024-02-01", 2023, form="10-K", fp="FY", start="2023-01-01"),
        ],
    })
    series = moat.annual_value_series(facts, "revenue", date(2025, 1, 1))
    assert len(series) == 1
    assert series[0]["tag"] == ("us-gaap", "Revenues")
    assert series[0]["val"] == 900


def test_annual_value_series_falls_back_to_ifrs_tag():
    facts = _facts(**{"ifrs-full:Revenue": [_entry("2020-12-31", 500, "2021-03-01", 2020, form="20-F")]})
    series = moat.annual_value_series(facts, "revenue", date(2021, 6, 1))
    assert series[0]["tag"] == ("ifrs-full", "Revenue")
    assert series[0]["val"] == 500


# ── revenue_series: 매출 태그 불일치 검사 ───────────────────────────────────


def test_revenue_series_no_mismatch_when_only_one_tag_has_value():
    facts = _facts(**{"us-gaap:Revenues": [_entry("2020-12-31", 1000, "2099-01-01", 2020)]})
    series, notes = moat.revenue_series(facts, _cfg(), date(2099, 12, 31))
    assert series[0]["val"] == 1000
    assert notes == []


def test_revenue_series_flags_mismatch_over_tolerance_and_uses_larger_value():
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry("2020-12-31", 100, "2099-01-01", 2020)],
        "us-gaap:Revenues": [_entry("2020-12-31", 900, "2099-01-01", 2020)],
    })
    series, notes = moat.revenue_series(facts, _cfg(), date(2099, 12, 31))
    assert series[0]["val"] == 900  # 큰 값 사용
    assert series[0].get("tag_mismatch") is True
    assert len(notes) == 1
    assert "2020-12-31" in notes[0] and "태그 불일치" in notes[0]


def test_revenue_series_no_flag_when_difference_within_tolerance():
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry("2020-12-31", 980, "2099-01-01", 2020)],
        "us-gaap:Revenues": [_entry("2020-12-31", 1000, "2099-01-01", 2020)],
    })
    series, notes = moat.revenue_series(facts, _cfg(), date(2099, 12, 31))
    assert notes == []
    assert series[0]["val"] == 980  # 우선순위 태그 값 그대로(병합 규칙), 불일치 아니므로 안 바꿈


def test_revenue_series_large_mismatch_uses_neighbor_fit_and_marks_review():
    """차이가 review_threshold_pct(50%) 넘으면 "큰 값 사용" 대신 앞뒤 연도와 가장 가까운
    값을 쓰고 "검토 필요"로 표시한다(사용자 지시 2026-10-01). 앞뒤 연도는 100·102 —
    기대값(평균) 101에 더 가까운 건 90(작은 값)이지 900(큰 값)이 아니다."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [
            _entry("2019-12-31", 100, "2099-01-01", 2019),
            _entry("2020-12-31", 90, "2099-01-01", 2020),  # 우선순위 태그(병합 기준값) — 이웃과 잘 이어짐
            _entry("2021-12-31", 102, "2099-01-01", 2021),
        ],
        "us-gaap:Revenues": [_entry("2020-12-31", 900, "2099-01-01", 2020)],  # 다른 태그값 — 이웃과 안 이어짐
    })
    series, notes = moat.revenue_series(facts, _cfg(), date(2099, 12, 31))
    by_end = {r["end"]: r for r in series}
    assert by_end["2020-12-31"]["val"] == 90  # 900이 아니라 이웃(100,102)에 가까운 90을 씀
    assert by_end["2020-12-31"].get("tag_mismatch_review") is True
    assert len(notes) == 1
    assert "검토 필요" in notes[0] and "2020-12-31" in notes[0]


def test_revenue_series_large_mismatch_falls_back_to_larger_value_when_no_neighbors():
    """앞뒤 연도가 아예 없으면(연도가 하나뿐) 이웃 비교를 못 하니 그냥 큰 값을 쓴다."""
    facts = _facts(**{
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": [_entry("2020-12-31", 90, "2099-01-01", 2020)],
        "us-gaap:Revenues": [_entry("2020-12-31", 900, "2099-01-01", 2020)],
    })
    series, notes = moat.revenue_series(facts, _cfg(), date(2099, 12, 31))
    assert series[0]["val"] == 900
    assert series[0].get("tag_mismatch_review") is True
    assert len(notes) == 1


# ── 기준 잠금(compute_config_hash·check_lock) ───────────────────────────────


def test_compute_config_hash_same_content_same_hash_regardless_of_key_order():
    a = {"lookback_years": 5, "roic_tax_rate_pct": 21}
    b = {"roic_tax_rate_pct": 21, "lookback_years": 5}
    assert moat.compute_config_hash(a) == moat.compute_config_hash(b)


def test_compute_config_hash_different_content_different_hash():
    a = {"lookback_years": 5}
    b = {"lookback_years": 6}
    assert moat.compute_config_hash(a) != moat.compute_config_hash(b)


def test_check_lock_returns_none_when_matching():
    cfg = _cfg()["moat"]
    locked_hash = moat.compute_config_hash(cfg)
    assert moat.check_lock(cfg, locked_hash) is None


def test_check_lock_warns_when_config_changed():
    cfg = _cfg()["moat"]
    locked_hash = moat.compute_config_hash(cfg)
    changed_cfg = {**cfg, "roic_tax_rate_pct": 25}
    warning = moat.check_lock(changed_cfg, locked_hash)
    assert warning is not None
    assert locked_hash in warning


# ── operating_income_series: 영업이익 대체 계산 ─────────────────────────────


def test_operating_income_series_uses_real_tag_when_present():
    facts = _facts(**{"us-gaap:OperatingIncomeLoss": [_entry("2020-12-31", 500, "2099-01-01", 2020)]})
    series = moat.operating_income_series(facts, date(2099, 12, 31))
    assert series[0]["val"] == 500
    assert series[0]["tag"] == ("us-gaap", "OperatingIncomeLoss")


def test_operating_income_series_falls_back_to_pretax_plus_interest_when_missing():
    """KLAC·PCAR·ADP류(영업이익 줄 자체가 없음, 2026-10-01 확인) — 세전이익+이자비용으로 대체."""
    facts = _facts(**{
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry("2020-12-31", 400, "2099-01-01", 2020)
        ],
        "us-gaap:InterestExpense": [_entry("2020-12-31", 50, "2099-01-01", 2020)],
    })
    series = moat.operating_income_series(facts, date(2099, 12, 31))
    assert series[0]["val"] == 450
    assert series[0]["tag"] == ("computed", "pretax_income+interest_expense")


def test_operating_income_series_prefers_real_tag_per_year_over_computed():
    facts = _facts(**{
        "us-gaap:OperatingIncomeLoss": [_entry("2020-12-31", 500, "2099-01-01", 2020)],
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry("2020-12-31", 400, "2099-01-01", 2020),
            _entry("2021-12-31", 600, "2099-01-01", 2021),
        ],
        "us-gaap:InterestExpense": [
            _entry("2020-12-31", 50, "2099-01-01", 2020),
            _entry("2021-12-31", 70, "2099-01-01", 2021),
        ],
    })
    series = moat.operating_income_series(facts, date(2099, 12, 31))
    by_end = {r["end"]: r for r in series}
    assert by_end["2020-12-31"]["val"] == 500  # 실제 영업이익 태그가 있으면 그걸 씀(대체 계산 무시)
    assert by_end["2020-12-31"]["tag"] == ("us-gaap", "OperatingIncomeLoss")
    assert by_end["2021-12-31"]["val"] == 670  # 실제 영업이익이 없는 연도만 대체 계산


def test_operating_income_series_pretax_only_when_no_interest_expense_tag():
    """PCAR류(이자비용 태그가 아예 없음, 2026-10-01 확인) — 세전이익만으로 대신하고
    "이자비용 없음"을 표시한다(사용자 지시)."""
    facts = _facts(**{
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry("2020-12-31", 400, "2099-01-01", 2020)
        ]
        # 이자비용 공시가 없음
    })
    series = moat.operating_income_series(facts, date(2099, 12, 31))
    assert len(series) == 1
    assert series[0]["val"] == 400
    assert series[0]["tag"] == ("computed", "pretax_income_only_no_interest_expense")


# ── M1 ROIC (새 정의: 순운전자본 + 순유형자산) ────────────────────────────────


def _m1_facts(op_income_by_year, current_assets, current_liabilities, net_ppe, revenue=None, equity=None, cash=None, years=None):
    years = years or _make_years()
    d = {
        "us-gaap:OperatingIncomeLoss": [_entry(e, op_income_by_year, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:AssetsCurrent": [_entry(e, current_assets, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:LiabilitiesCurrent": [_entry(e, current_liabilities, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:PropertyPlantAndEquipmentNet": [_entry(e, net_ppe, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
    }
    if revenue is not None:
        d["us-gaap:Revenues"] = [_entry(e, revenue, "2099-01-01", 2000 + i) for i, e in enumerate(years)]
    if equity is not None:
        d["us-gaap:StockholdersEquity"] = [_entry(e, equity, "2099-01-01", 2000 + i) for i, e in enumerate(years)]
    if cash is not None:
        d["us-gaap:CashAndCashEquivalentsAtCarryingValue"] = [_entry(e, cash, "2099-01-01", 2000 + i) for i, e in enumerate(years)]
    return _facts(**d)


def test_compute_m1_roic_good_using_working_capital_plus_ppe():
    # invested_capital = (300-100) + 200 = 400; op_income*0.79/400 = 100*0.79/400 = 19.75% >= 15%
    facts = _m1_facts(op_income_by_year=100, current_assets=300, current_liabilities=100, net_ppe=200, revenue=1000)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"
    assert len(result.yearly_values) == 5


def test_compute_m1_roic_reference_values_use_old_formula():
    # 새 식: WC=(300-50-0)-(100-0)=150, IC=150+200=350 -> 22.57%
    # 참고(기존) 식: equity(600)+debt(0)-cash(50)=550 -> 100*0.79/550*100=14.36% (새 값과 달라야 함)
    facts = _m1_facts(op_income_by_year=100, current_assets=300, current_liabilities=100, net_ppe=200, revenue=1000, equity=600, cash=50)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.reference_yearly_values
    assert list(result.reference_yearly_values.values())[0] != list(result.yearly_values.values())[0]


def test_compute_m1_roic_tiny_denominator_with_loss_is_unmeasurable():
    """분모가 매출의 2% 미만인데 그해 세후영업이익이 0 이하면 "측정 불가"."""
    # WC=(101-0-0)-(100-0)=1, IC=1+0=1, ratio=1/1000=0.001 < 0.02 -> too_small, 영업이익<=0 -> 측정 불가
    facts = _m1_facts(op_income_by_year=-10, current_assets=101, current_liabilities=100, net_ppe=0, revenue=1000)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "판단 불가"
    assert len(result.unmeasurable_years) == 5
    assert result.capital_light_years == []


def test_compute_m1_roic_negative_working_capital_floored_to_zero_not_unmeasurable():
    """ADSK류(구독모델, 순운전자본이 항상 마이너스) — 0으로 바닥을 깔아 순유형자산만 남는다.
    분모가 매출의 2% 미만이어도 세후영업이익>0이면 "기준 충족"(100% 이상)으로 센다."""
    # WC=(50-0-0)-(200-0)=-150 -> max(-150,0)=0, IC=0+0(PPE)=0, ratio=0 < 0.02 -> too_small, 영업이익>0 -> 기준 충족
    facts = _m1_facts(op_income_by_year=100, current_assets=50, current_liabilities=200, net_ppe=0, revenue=1000)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"
    assert len(result.capital_light_years) == 5
    assert all(v == 100.0 for v in result.yearly_values.values())
    assert all(d == "100% 이상" for d in result.yearly_display.values())
    assert result.unmeasurable_years == []


def test_compute_m1_roic_negative_working_capital_with_loss_is_unmeasurable():
    facts = _m1_facts(op_income_by_year=-5, current_assets=50, current_liabilities=200, net_ppe=0, revenue=1000)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "판단 불가"
    assert len(result.unmeasurable_years) == 5
    assert result.capital_light_years == []


def test_compute_m1_roic_normal_high_value_displayed_as_100_percent_or_more():
    """분모가 작지 않아도(too_small 아님) 실제 ROIC가 100%를 넘으면 표시만 "100% 이상"."""
    # WC=(1000-0-0)-(0-0)=1000, IC=1000+0=1000, revenue=10000 -> ratio=0.1 (too_small 아님)
    # val = 100000*0.79/1000*100 = 7900% (>100)
    facts = _m1_facts(op_income_by_year=100000, current_assets=1000, current_liabilities=0, net_ppe=0, revenue=10000)
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert all(v >= 100 for v in result.yearly_values.values())
    assert all(d == "100% 이상" for d in result.yearly_display.values())
    assert result.capital_light_years == []  # too_small 경로가 아니라 일반 계산 경로로 나온 값


def test_compute_m1_roic_insufficient_data_when_core_items_missing():
    result = moat.compute_m1_roic(_facts(), _cfg(), date(2099, 1, 1))
    assert result.status == "판단 불가"


def test_compute_m1_roic_uses_operating_income_fallback_when_no_direct_tag():
    """KLAC류: 영업이익 태그가 없어도 세전이익+이자비용으로 계산한다."""
    years = _make_years()
    facts = _facts(**{
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry(e, 130, "2099-01-01", 2000 + i) for i, e in enumerate(years)
        ],
        "us-gaap:InterestExpense": [_entry(e, 20, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:AssetsCurrent": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:LiabilitiesCurrent": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
        "us-gaap:PropertyPlantAndEquipmentNet": [_entry(e, 200, "2099-01-01", 2000 + i) for i, e in enumerate(years)],
    })
    result = moat.compute_m1_roic(facts, _cfg(), date(2099, 12, 31))
    assert result.status != "판단 불가"
    assert len(result.yearly_values) == 5


# ── M2 매출총이익률: 수준 기준 없음, 표준편차만 ──────────────────────────────


def test_compute_m2_gross_margin_good_when_stable_regardless_of_level():
    """COST류(수준이 낮아도 안정적이면 "좋음") — 2026-10-01 사용자 지시로 수준 기준 제거."""
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:GrossProfit": [_entry(e, 125, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],  # 12.5% 수준, 표준편차 0
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
    assert result.status == "좋음"
    assert result.tags_used["cost_of_revenue"][ends[0]] == ("us-gaap", "CostOfGoodsAndServicesSold")


# ── M3 (영업이익 대체 계산 사용) ───────────────────────────────────────────


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


def test_compute_m3_uses_operating_income_fallback():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry(e, 280, "2099-01-01", 2000 + i) for i, e in enumerate(ends)
        ],
        "us-gaap:InterestExpense": [_entry(e, 20, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m3_recession_resilience(facts, _cfg(), date(2099, 12, 31))
    assert result.status != "판단 불가"


# ── M4·M5 (변경 없음, 회귀 확인만) ───────────────────────────────────────────


def test_compute_m4_cash_conversion_good():
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": [_entry(e, 10, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
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


def test_compute_m4_main_is_ocf_over_ni_reference_is_old_capex_subtracted_formula():
    """2026-10-01 확정(잠금): 주 지표 = 영업현금흐름÷순이익(설비투자 안 뺌, 0.9 기준).
    기존 식((영업현금흐름-설비투자)÷순이익, 0.8 기준)은 참고 열로만 남는다."""
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 95, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": [_entry(e, 20, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    result = moat.compute_m4_cash_conversion(facts, _cfg(), date(2099, 12, 31))
    assert result.status == "좋음"  # 주 지표: 95/100 = 0.95 >= 0.9
    assert result.reference_status == "보통"  # 참고(기존 식): (95-20)/100 = 0.75, 0.5<0.75<0.8
    assert result.reference_yearly_values


def test_compute_m4_reference_status_none_when_insufficient_data():
    result = moat.compute_m4_cash_conversion(_facts(), _cfg(), date(2099, 12, 31))
    assert result.reference_status is None


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
    assert grade == "좁음"


def test_compute_moat_grade_narrow_when_two_of_three_good():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("좋음"), _ind("보통"), _ind("보통"), _ind("보통"))
    assert grade == "좁음"


def test_compute_moat_grade_none_when_less_than_two_good():
    grade = moat.compute_moat_grade(_ind("좋음"), _ind("보통"), _ind("보통"), _ind("주의"), _ind("보통"))
    assert grade == "없음"


# ── 이상치 탐지 ────────────────────────────────────────────────────────────


def test_detect_outliers_no_longer_flags_high_roic():
    """2026-10-01 사용자 지시: ROIC 100% 초과는 더 이상 이상치로 안 본다(M1이 "100% 이상"으로
    표시만 한다)."""
    m1 = moat.IndicatorResult(status="좋음", yearly_values={"2020-12-31": 150.0})
    out = moat.detect_outliers(m1, _cfg(), None)
    assert not any("ROIC" in o for o in out)


def test_detect_outliers_flags_negative_equity():
    m1 = moat.IndicatorResult(status="좋음")
    equity_series = [{"end": "2020-12-31", "val": -500}, {"end": "2021-12-31", "val": 400}]
    out = moat.detect_outliers(m1, _cfg(), equity_series)
    assert any("자기자본 음수" in o and "2020-12-31" in o for o in out)
    assert not any("2021-12-31" in o for o in out)


def test_detect_outliers_restricted_to_lookback_years():
    """2026-10-01 사용자 지시: 이상치 점검은 최근 lookback_years(기본 5) 안으로 좁힌다 —
    오래된 상장 전·초기 성장기 음수 자기자본까지 매번 뜨지 않게."""
    m1 = moat.IndicatorResult(status="좋음")
    equity_series = [{"end": f"{2000+i}-12-31", "val": -100} for i in range(10)]  # 10년치, 전부 음수
    out = moat.detect_outliers(m1, _cfg(), equity_series)  # lookback_years=5
    assert len(out) == 5
    assert all(str(2005 + i) in out[i] for i in range(5))


def test_detect_outliers_empty_when_nothing_unusual():
    m1 = moat.IndicatorResult(status="좋음")
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
        "us-gaap:AssetsCurrent": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:LiabilitiesCurrent": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:PropertyPlantAndEquipmentNet": [_entry(e, 200, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetCashProvidedByUsedInOperatingActivities": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:NetIncomeLoss": [_entry(e, 250, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": [_entry(e, 1000 + i, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    profile = moat.analyze_company(facts, "GOOD", _cfg(), date(2099, 12, 31))
    assert profile.grade != "판단 불가"
    assert profile.insufficient_data_reason is None
    assert len(profile.data_years) == 5


def test_analyze_company_operating_income_fallback_avoids_inconclusive():
    """KLAC류: 영업이익 태그가 없어도 대체 계산으로 판단 불가를 면할 수 있다."""
    ends = _make_years()
    facts = _facts(**{
        "us-gaap:Revenues": [_entry(e, 1000, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest": [
            _entry(e, 280, "2099-01-01", 2000 + i) for i, e in enumerate(ends)
        ],
        "us-gaap:InterestExpense": [_entry(e, 20, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:StockholdersEquity": [_entry(e, 400, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:CashAndCashEquivalentsAtCarryingValue": [_entry(e, 50, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:AssetsCurrent": [_entry(e, 300, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:LiabilitiesCurrent": [_entry(e, 100, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
        "us-gaap:PropertyPlantAndEquipmentNet": [_entry(e, 200, "2099-01-01", 2000 + i) for i, e in enumerate(ends)],
    })
    profile = moat.analyze_company(facts, "KLAC_LIKE", _cfg(), date(2099, 12, 31))
    assert profile.grade != "판단 불가"
