"""data/universe_history.py 테스트. 네트워크 없이 고정 위키텍스트·합성 데이터로 돈다 (P5-1 6번)."""

from __future__ import annotations

from datetime import date

from data import universe_history as uh

_SAMPLE_WIKITEXT = """{{Short description|x}}
{| class="wikitable sortable" id="changes"
! rowspan="2" data-sort-type="date" |Date
! colspan="2" |Added
! colspan="2" |Removed
! rowspan="2" |Reason
|-
!Ticker
!Security
!Ticker
!Security
|-
|December 21, 2015
|TCOM
|[[Ctrip]]
|CHRW
|[[C. H. Robinson Worldwide]]
|Annual index reconstitution.<ref name=":13">{{Cite press release |title=x |date=December 11, 2015}}</ref>
|-
|July 2, 2015
|KHC
|[[Kraft Heinz]]
|KRFT
|[[Kraft Foods]]
|Kraft Foods merged with [[Heinz]]
|-
|March 23, 2015
|WBA
|[[Walgreens Boots Alliance]]
|EQIX
|[[Equinix]]
|Equinix converted into a REIT
|-
|July 31, 2015
|
|
|SIAL
|[[Sigma-Aldrich]]
|Sigma-Aldrich was acquired
|-
|August 3, 2015
|SWKS
|[[Skyworks Solutions]]
|
|
|Skyworks replaced Sigma-Aldrich
|}
"""


def test_parse_changes_extracts_expected_rows():
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    assert len(changes) == 5
    first = changes[0]
    assert first.date == date(2015, 12, 21)
    assert first.added == "TCOM"
    assert first.removed == "CHRW"


def test_parse_changes_handles_ticker_with_dot_and_blank_cells():
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    by_date = {c.date: c for c in changes}
    only_removed = by_date[date(2015, 7, 31)]
    assert only_removed.added is None
    assert only_removed.removed == "SIAL"
    only_added = by_date[date(2015, 8, 3)]
    assert only_added.added == "SWKS"
    assert only_added.removed is None


def test_parse_changes_ignores_header_rows_and_empty_table():
    assert uh.parse_changes("no table here") == []


def test_reconstruct_membership_undoes_changes_after_as_of():
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    current = {"TCOM", "KHC", "WBA", "SWKS"}  # 오늘 구성 종목(예시)

    # 2016-01-01 기준: 모든 변경이 이미 적용된 뒤 -> 오늘과 같다
    assert uh.reconstruct_membership(current, changes, date(2016, 1, 1)) == current

    # 2015-12-20 기준 (12/21 변경 전날): TCOM 대신 CHRW가 있어야 한다
    members = uh.reconstruct_membership(current, changes, date(2015, 12, 20))
    assert "CHRW" in members
    assert "TCOM" not in members

    # 2015-01-01 기준: 모든 2015년 변경 이전 -> KHC/WBA/SWKS는 없고 KRFT/EQIX/SIAL이 있어야 한다
    members_early = uh.reconstruct_membership(current, changes, date(2015, 1, 1))
    assert members_early == {"CHRW", "KRFT", "EQIX", "SIAL"}


def test_reconstruct_membership_boundary_date_is_inclusive():
    """변경 날짜 당일은 그 변경이 이미 반영된 것으로 본다."""
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    current = {"TCOM"}
    members = uh.reconstruct_membership(current, changes, date(2015, 12, 21))
    assert "TCOM" in members and "CHRW" not in members


def test_membership_checkpoints_and_universe_on_roundtrip():
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    current = {"TCOM", "KHC", "WBA", "SWKS"}
    checkpoints = uh.membership_checkpoints(current, changes, date(2015, 1, 1))

    # 시작일: 아직 아무 변경도 적용 안 된 가장 이른 구성
    assert uh.universe_on(checkpoints, date(2015, 1, 1)) == frozenset({"CHRW", "KRFT", "EQIX", "SIAL"})
    # 3/23 이후 ~ 7/2 이전: WBA는 들어오고 EQIX는 빠짐
    members = uh.universe_on(checkpoints, date(2015, 4, 1))
    assert "WBA" in members and "EQIX" not in members
    # 맨 끝(연말 이후)은 현재 구성과 같다
    assert uh.universe_on(checkpoints, date(2016, 6, 1)) == frozenset(current)


def test_universe_on_before_start_returns_empty():
    changes = uh.parse_changes(_SAMPLE_WIKITEXT)
    checkpoints = uh.membership_checkpoints({"TCOM"}, changes, date(2015, 6, 1))
    assert uh.universe_on(checkpoints, date(2014, 1, 1)) == frozenset()


def test_clean_ticker_cell_converts_dot_to_dash_and_strips_refs():
    assert uh._clean_ticker_cell("BRK.B<ref>x</ref>") == "BRK-B"
    assert uh._clean_ticker_cell("") is None
    assert uh._clean_ticker_cell("  ") is None
    assert uh._clean_ticker_cell("[[Some Company]]") is None  # 공백 섞인 이름은 티커로 안 본다
