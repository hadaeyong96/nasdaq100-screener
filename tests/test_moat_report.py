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


def test_profile_to_row_stores_cik():
    row = rpt.profile_to_row("GOOGL", "Alphabet Inc.", _blank_profile(), cik=1652044)
    assert row["cik"] == 1652044


def test_dedupe_rows_by_cik_keeps_first_occurrence_per_company():
    """GOOGL·GOOG는 둘 다 Alphabet(같은 CIK) — 등급 집계는 한 번만 센다."""
    rows = [
        {"ticker": "GOOGL", "cik": 1652044, "grade": "넓음"},
        {"ticker": "GOOG", "cik": 1652044, "grade": "넓음"},
        {"ticker": "AAPL", "cik": 320193, "grade": "넓음"},
    ]
    out = rpt.dedupe_rows_by_cik(rows)
    assert [r["ticker"] for r in out] == ["GOOGL", "AAPL"]


def test_dedupe_rows_by_cik_keeps_all_rows_with_unresolved_cik():
    rows = [{"ticker": "AAA", "cik": None, "grade": "넓음"}, {"ticker": "BBB", "cik": None, "grade": "없음"}]
    assert len(rpt.dedupe_rows_by_cik(rows)) == 2


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


def test_render_html_dedupes_same_company_multiple_share_classes_in_summary_counts():
    """GOOGL·GOOG(Alphabet, 같은 CIK)는 등급 집계에서 한 번만 세지만, 전체 표 티커 수엔 둘 다 남는다."""
    rows = [
        rpt.profile_to_row("GOOGL", "Alphabet Inc.", _blank_profile("넓음"), cik=1652044),
        rpt.profile_to_row("GOOG", "Alphabet Inc.", _blank_profile("넓음"), cik=1652044),
        rpt.profile_to_row("AAPL", "Apple Inc.", _blank_profile("넓음"), cik=320193),
    ]
    html = rpt.render_html(rows, "now")
    assert "넓음: 2" in html  # Alphabet 1 + Apple 1, GOOG 중복 제외
    assert "티커 3개" in html


def test_render_html_lists_outliers():
    rows = [rpt.profile_to_row("AAA", "AAA Inc", _blank_profile(outliers=["ROIC 2020-12-31 150.0% (>100%)"]))]
    html = rpt.render_html(rows, "now")
    assert "150.0%" in html


# ── 업종(섹터) ──────────────────────────────────────────────────────────────


def test_get_sector_uses_cache_without_calling_fetch_fn():
    cache = {"AAPL": "Technology"}

    def _boom(t):
        raise AssertionError("캐시에 있으면 fetch_fn을 부르면 안 된다")

    assert rpt.get_sector("AAPL", cache, fetch_fn=_boom) == "Technology"


def test_get_sector_fetches_and_caches_on_miss():
    cache: dict = {}
    out = rpt.get_sector("AAPL", cache, fetch_fn=lambda t: "Technology")
    assert out == "Technology"
    assert cache["AAPL"] == "Technology"


def test_get_sector_retries_fetch_when_cached_value_is_unknown():
    cache = {"AAPL": rpt.UNKNOWN_SECTOR}
    out = rpt.get_sector("AAPL", cache, fetch_fn=lambda t: "Technology")
    assert out == "Technology"
    assert cache["AAPL"] == "Technology"


def test_sector_breakdown_counts_and_sorts_descending():
    rows = [{"sector": "Technology"}, {"sector": "Technology"}, {"sector": "Healthcare"}]
    assert rpt.sector_breakdown(rows) == {"Technology": 2, "Healthcare": 1}


def test_sector_breakdown_defaults_missing_to_unknown():
    assert rpt.sector_breakdown([{}]) == {rpt.UNKNOWN_SECTOR: 1}


def test_profile_to_row_includes_sector():
    row = rpt.profile_to_row("AAPL", "Apple Inc.", _blank_profile(), sector="Technology")
    assert row["sector"] == "Technology"


def test_profile_to_row_defaults_sector_to_unknown():
    row = rpt.profile_to_row("AAA", "AAA Inc", _blank_profile())
    assert row["sector"] == rpt.UNKNOWN_SECTOR


# ── M4 참고 비율 등급 비교 ───────────────────────────────────────────────────


def test_m4_reference_grade_none_when_no_reference_status():
    ind = moat.IndicatorResult(status="좋음")
    profile = moat.MoatProfile(ticker="AAA", grade="좁음", m1=ind, m2=ind, m3=ind, m4=ind, m5=ind)
    assert rpt.m4_reference_grade(profile) is None


def test_m4_reference_grade_computes_alternate_grade():
    good = moat.IndicatorResult(status="좋음")
    m4 = moat.IndicatorResult(status="보통", reference_status="좋음")  # 참고 비율로는 "좋음"
    profile = moat.MoatProfile(ticker="AAA", grade="좁음", m1=good, m2=good, m3=good, m4=m4, m5=good)
    # 실제: M1·M2·좋음, M4 보통 -> core 2/3 좋음 -> 좁음. 참고: M4도 좋음 -> core 3/3 -> 넓음
    assert profile.grade == "좁음"
    assert rpt.m4_reference_grade(profile) == "넓음"


def test_profile_to_row_flags_m4_grade_would_change():
    good = moat.IndicatorResult(status="좋음")
    m4 = moat.IndicatorResult(status="보통", reference_status="좋음")
    profile = moat.MoatProfile(ticker="AAA", grade="좁음", m1=good, m2=good, m3=good, m4=m4, m5=good)
    row = rpt.profile_to_row("AAA", "AAA Inc", profile)
    assert row["m4_grade_would_change"] is True
    assert row["m4_alternate_grade"] == "넓음"


def test_profile_to_row_no_change_flag_when_alternate_grade_same():
    row = rpt.profile_to_row("AAA", "AAA Inc", _blank_profile("넓음"))
    assert row["m4_grade_would_change"] is False
