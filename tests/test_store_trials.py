"""store/trials.py 테스트 (AI 펀드 F2, docs/design/fund_sim.md 2.2·5.9). 네트워크 없음."""

from __future__ import annotations

import pytest

from store import trials


@pytest.fixture
def conn(tmp_path):
    c = trials.connect(tmp_path / "trials.db")
    yield c
    c.close()


def test_record_and_load_trial(conn):
    trial_id = trials.record_trial(
        conn, run_at="2026-09-30T12:00:00", label="B0.5",
        metrics={"cagr_pretax_pct": 26.39, "cagr_posttax_a_pct": 19.78},
        universe_mode="POINT_IN_TIME", date_range_start="2016-01-01", date_range_end="2021-12-31",
    )
    row = trials.load_trial(conn, trial_id)
    assert row["label"] == "B0.5"
    assert row["metrics"]["cagr_posttax_a_pct"] == 19.78
    assert row["universe_mode"] == "POINT_IN_TIME"
    assert row["variant_id"] is None


def test_record_trial_with_variant_id(conn):
    trial_id = trials.record_trial(conn, run_at="2026-09-30T12:00:00", label="v1", metrics={}, variant_id="var-1")
    row = trials.load_trial(conn, trial_id)
    assert row["variant_id"] == "var-1"


def test_load_trial_missing_returns_none(conn):
    assert trials.load_trial(conn, 999) is None


def test_load_all_trials_ordered(conn):
    trials.record_trial(conn, run_at="2026-09-30T12:00:00", label="first", metrics={})
    trials.record_trial(conn, run_at="2026-09-30T12:01:00", label="second", metrics={})
    rows = trials.load_all_trials(conn)
    assert [r["label"] for r in rows] == ["first", "second"]


def test_count_trials(conn):
    assert trials.count_trials(conn) == 0
    trials.record_trial(conn, run_at="2026-09-30T12:00:00", label="x", metrics={})
    assert trials.count_trials(conn) == 1
