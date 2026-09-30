"""AI 펀드 F2: scripts/fund_dataqc_report.py의 순수 함수 테스트 (사용자 지시 2026-09-30).

빠진 종목 목록(유료 데이터 문의용)을 만드는 로직이 맞는지 확인한다 — 네트워크 없이,
고정된 위키텍스트 조각으로 돈다.
"""

from __future__ import annotations

from datetime import date

from scripts import fund_dataqc_report as rpt


def test_ticker_membership_intervals_single_span_still_active():
    checkpoints = [
        (date(2010, 1, 1), frozenset({"AAA"})),
        (date(2015, 6, 1), frozenset({"AAA", "BBB"})),
    ]
    assert rpt.ticker_membership_intervals(checkpoints, "AAA") == [(date(2010, 1, 1), None)]


def test_ticker_membership_intervals_added_then_removed():
    checkpoints = [
        (date(2000, 1, 1), frozenset({"XXX"})),
        (date(2010, 1, 1), frozenset({"BBB"})),
        (date(2012, 1, 1), frozenset({"BBB", "AAA"})),
        (date(2015, 6, 1), frozenset({"BBB"})),
    ]
    assert rpt.ticker_membership_intervals(checkpoints, "AAA") == [(date(2012, 1, 1), date(2015, 6, 1))]


def test_ticker_membership_intervals_two_separate_spans():
    checkpoints = [
        (date(2000, 1, 1), frozenset({"AAA"})),
        (date(2005, 1, 1), frozenset()),
        (date(2010, 1, 1), frozenset({"AAA"})),
        (date(2012, 1, 1), frozenset()),
    ]
    assert rpt.ticker_membership_intervals(checkpoints, "AAA") == [
        (date(2000, 1, 1), date(2005, 1, 1)),
        (date(2010, 1, 1), date(2012, 1, 1)),
    ]


def test_overlap_returns_intersection():
    assert rpt.overlap(date(2014, 1, 1), date(2018, 1, 1), date(2015, 1, 1), date(2021, 12, 31)) == (date(2015, 1, 1), date(2018, 1, 1))


def test_overlap_open_ended_interval_clips_to_study_end():
    assert rpt.overlap(date(2020, 1, 1), None, date(2015, 1, 1), date(2021, 12, 31)) == (date(2020, 1, 1), date(2021, 12, 31))


def test_overlap_returns_none_when_disjoint():
    assert rpt.overlap(date(2022, 1, 1), None, date(2015, 1, 1), date(2021, 12, 31)) is None


_SAMPLE_WIKITEXT = """
{| class="wikitable sortable" id="changes"
!Date
!colspan="2"|Added
!colspan="2"|Removed
!Reason
|-
!
!Ticker
!Security
!Ticker
!Security
|-
|October 7, 2015
|
|
|ALTR
|[[Altera]]
|Altera merged with Intel.
|-
|December 24, 2018
|ALAB
|[[Astera Labs]]
|
|
|Index reconstitution.
|}
"""


def test_parse_changes_with_detail_extracts_reason_and_names():
    changes = rpt.parse_changes_with_detail(_SAMPLE_WIKITEXT)
    assert len(changes) == 2
    removal = next(c for c in changes if c.removed == "ALTR")
    assert removal.removed_name == "Altera"
    assert removal.reason == "Altera merged with Intel."
    assert removal.date == date(2015, 10, 7)


def test_build_missing_ticker_rows_fills_name_reason_and_needed_range():
    changes = rpt.parse_changes_with_detail(_SAMPLE_WIKITEXT)
    full_checkpoints = [
        (date(2000, 1, 1), frozenset({"ALTR"})),
        (date(2015, 10, 7), frozenset()),
    ]
    rows = rpt.build_missing_ticker_rows(
        failed_tickers={"ALTR": "ALTR: 가격 데이터 없음"},
        full_checkpoints=full_checkpoints,
        changes=changes,
        study_start=date(2014, 1, 1),
        study_end=date(2021, 12, 31),
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "ALTR"
    assert row["company_name"] == "Altera"
    assert "2015-10-07" in row["removal_reason"] and "merged with Intel" in row["removal_reason"]
    assert row["needed_price_range"] == "2014-01-01~2015-10-07"


def test_build_missing_ticker_rows_marks_out_of_study_range_tickers():
    """연구 구간 이후에만 소속된 종목은 needed_price_range가 "겹치는 구간 없음"이어야 한다."""
    full_checkpoints = [(date(2023, 1, 1), frozenset({"NEW"}))]
    rows = rpt.build_missing_ticker_rows(
        failed_tickers={"NEW": "NEW: 가격 데이터 없음"},
        full_checkpoints=full_checkpoints,
        changes=[],
        study_start=date(2014, 1, 1),
        study_end=date(2021, 12, 31),
    )
    assert rows[0]["needed_price_range"] == "겹치는 구간 없음(연구 구간 밖)"
