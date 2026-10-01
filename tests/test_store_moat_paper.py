"""store/moat_paper.py 테스트. 네트워크 없음, 임시 DB 파일 사용."""

from __future__ import annotations

import pytest

from store import moat_paper as store_mp


@pytest.fixture
def conn(tmp_path):
    c = store_mp.connect(tmp_path / "moat_paper.db")
    yield c
    c.close()


def test_record_and_load_monthly_records(conn):
    store_mp.record_monthly(conn, "2026-10-02T09:00:00", "2026-10", "P1", 1.0, "평가")
    store_mp.record_monthly(conn, "2026-10-02T09:00:00", "2026-10", "QQQM", 1.02, "평가")
    rows = store_mp.load_monthly_records(conn)
    assert len(rows) == 2
    assert {r["portfolio"] for r in rows} == {"P1", "QQQM"}


def test_record_monthly_pending_status_allows_null_value(conn):
    store_mp.record_monthly(conn, "2026-10-01T09:00:00", "2026-10", "P2", None, "진입 대기", notes="가격 미확정")
    rows = store_mp.load_monthly_records(conn, portfolio="P2")
    assert rows[0]["value_factor"] is None
    assert rows[0]["status"] == "진입 대기"


def test_latest_grade_before_returns_most_recent_prior_period(conn):
    store_mp.record_grade(conn, "2026-10-02T09:00:00", "2026-10", "AAPL", "넓음")
    store_mp.record_grade(conn, "2026-11-02T09:00:00", "2026-11", "AAPL", "좁음")
    assert store_mp.latest_grade_before(conn, "AAPL", "2026-12") == "좁음"
    assert store_mp.latest_grade_before(conn, "AAPL", "2026-11") == "넓음"
    assert store_mp.latest_grade_before(conn, "AAPL", "2026-10") is None


def test_latest_grade_before_unknown_ticker_is_none(conn):
    assert store_mp.latest_grade_before(conn, "ZZZZ", "2026-12") is None
