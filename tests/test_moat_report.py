"""scripts/moat_report.py의 순수 함수 테스트 — 네트워크 없이 돈다."""

from __future__ import annotations

from scripts import moat_report as rpt
from core import moat


def test_resolve_cik_exact_match():
    assert rpt.resolve_cik("AAPL", {"AAPL": 320193}) == 320193


def test_resolve_cik_case_insensitive():
    assert rpt.resolve_cik("aapl", {"AAPL": 320193}) == 320193


def test_resolve_cik_dot_dash_variants():
    assert rpt.resolve_cik("BRK-B", {"BRK.B": 1067983}) == 1067983
    assert rpt.resolve_cik("BRK.B", {"BRK-B": 1067983}) == 1067983


def test_resolve_cik_none_when_not_found():
    assert rpt.resolve_cik("ZZZZ", {"AAPL": 320193}) is None


def test_summarize_grades_counts_each_grade():
    rows = [{"grade": "넓음"}, {"grade": "넓음"}, {"grade": "좁음"}, {"grade": "판단 불가"}]
    counts = rpt.summarize_grades(rows)
    assert counts["넓음"] == 2
    assert counts["좁음"] == 1
    assert counts["없음"] == 0
    assert counts["판단 불가"] == 1


def _blank_profile(grade="넓음", reason=None, outliers=None):
    ind = moat.IndicatorResult(status="좋음", detail="d", yearly_values={"2020-12-31": 1.0}, tags_used={"revenue": ("us-gaap", "Revenues")})
    return moat.MoatProfile(
        ticker="AAA", grade=grade, m1=ind, m2=ind, m3=ind, m4=ind, m5=ind,
        outliers=outliers or [], insufficient_data_reason=reason, data_years=["2019-12-31", "2020-12-31"],
    )


def test_profile_to_row_includes_grade_and_indicator_status():
    row = rpt.profile_to_row("AAA", "AAA Inc", _blank_profile())
    assert row["ticker"] == "AAA"
    assert row["grade"] == "넓음"
    assert row["m1_status"] == "좋음"
    assert row["data_years"] == 2
    assert row["latest_data_year"] == "2020-12-31"
    assert "revenue=('us-gaap', 'Revenues')" in row["m1_tags"]


def test_profile_to_row_empty_data_years_gives_none_latest_year():
    ind = moat.IndicatorResult(status="판단 불가")
    profile = moat.MoatProfile(ticker="ZZZ", grade="판단 불가", m1=ind, m2=ind, m3=ind, m4=ind, m5=ind, data_years=[])
    row = rpt.profile_to_row("ZZZ", "ZZZ Inc", profile)
    assert row["latest_data_year"] is None
    assert row["data_years"] == 0


def test_render_html_includes_summary_counts_and_wide_list():
    rows = [rpt.profile_to_row("AAA", "AAA Inc", _blank_profile("넓음"))]
    rows.append(rpt.profile_to_row("BBB", "BBB Inc", _blank_profile("판단 불가", reason="데이터 부족")))
    html = rpt.render_html(rows, "2026-10-01T00:00:00")
    assert "AAA" in html
    assert "데이터 부족" in html
    assert "넓음: 1" in html
    assert "판단 불가: 1" in html


def test_render_html_escapes_html_special_characters():
    row = rpt.profile_to_row("AAA", "A&B <Co>", _blank_profile())
    html = rpt.render_html([row], "now")
    assert "<Co>" not in html
    assert "&lt;Co&gt;" in html


def test_render_html_lists_outliers():
    rows = [rpt.profile_to_row("AAA", "AAA Inc", _blank_profile(outliers=["ROIC 2020-12-31 150.0% (>100%)"]))]
    html = rpt.render_html(rows, "now")
    assert "150.0%" in html
