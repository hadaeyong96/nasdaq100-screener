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
    assert row["variant_id"] is None  # variant 체계 밖 실험도 기록할 수 있다


def test_record_trial_with_variant_id(conn):
    trial_id = trials.record_trial(
        conn, run_at="2026-09-30T12:00:00", label="team1-v0 워크포워드 2018",
        metrics={"cagr_pretax_pct": 10.0}, variant_id="team1-v0",
    )
    assert trials.load_trial(conn, trial_id)["variant_id"] == "team1-v0"


def test_load_trial_missing_returns_none(conn):
    assert trials.load_trial(conn, 999) is None


def test_count_trials_is_n_raw(conn):
    assert trials.count_trials(conn) == 0
    trials.record_trial(conn, run_at="t1", label="A", metrics={})
    trials.record_trial(conn, run_at="t2", label="B", metrics={})
    assert trials.count_trials(conn) == 2


def test_load_all_trials_returns_in_insertion_order(conn):
    trials.record_trial(conn, run_at="t1", label="첫번째", metrics={})
    trials.record_trial(conn, run_at="t2", label="두번째", metrics={})
    rows = trials.load_all_trials(conn)
    assert [r["label"] for r in rows] == ["첫번째", "두번째"]


def test_trials_are_append_only_no_update_or_delete_function_exposed():
    """설계 5.9: N_raw는 이 로그에서 나오므로 트라이얼을 조용히 고치거나 지울 수단이
    없어야 한다 — update_trial·delete_trial 같은 함수를 일부러 안 만들었는지 확인."""
    assert not hasattr(trials, "update_trial")
    assert not hasattr(trials, "delete_trial")
