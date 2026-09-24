"""store/db.py 저장·조회 왕복 테스트. 임시 파일 DB로 돈다 (네트워크 없음)."""

from __future__ import annotations

import pandas as pd

from core import state as st
from store import db


def test_save_and_load_position_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    state = st.init_state("NVDA", "엔비디아")
    state["state"] = "확인"
    state["units"] = {"1": 37, "2": 49}
    state["entries"] = {"1": 100.0, "2": 103.0}
    state["stop"] = 94.0
    state["a1_date"] = pd.Timestamp("2026-09-10")
    state["sent_alerts"] = ["A1:2026-09-10"]

    db.save_position(conn, state)
    loaded = db.load_position(conn, "NVDA")

    assert loaded["state"] == "확인"
    assert loaded["units"] == {"1": 37, "2": 49}
    assert loaded["entries"] == {"1": 100.0, "2": 103.0}
    assert loaded["stop"] == 94.0
    assert loaded["sent_alerts"] == ["A1:2026-09-10"]


def test_load_all_positions_and_upsert(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.save_position(conn, st.init_state("AAPL"))
    db.save_position(conn, st.init_state("MSFT"))

    all_positions = db.load_all_positions(conn)
    assert set(all_positions) == {"AAPL", "MSFT"}

    updated = all_positions["AAPL"]
    updated["state"] = "정찰"
    db.save_position(conn, updated)
    assert db.load_position(conn, "AAPL")["state"] == "정찰"
    assert len(db.load_all_positions(conn)) == 2  # upsert가 새 행을 만들지 않는다


def test_record_price_snapshots(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_price_snapshots(
        conn,
        run_at="2026-09-22T07:30:00",
        rows=[
            {"ticker": "NVDA", "date": "2026-09-22", "close": 228.87, "close_source": "meta", "meta_time": "2026-09-22T16:00:00-04:00"},
            {"ticker": "AAPL", "date": "2026-09-22", "close": 339.75, "close_source": "yahoo", "meta_time": None},
        ],
    )
    rows = conn.execute("SELECT ticker, close, close_source, meta_time FROM price_snapshots ORDER BY ticker").fetchall()
    assert rows == [
        ("AAPL", 339.75, "yahoo", None),
        ("NVDA", 228.87, "meta", "2026-09-22T16:00:00-04:00"),
    ]


def test_record_events_and_run(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_events(conn, [{"date": "2026-09-10", "ticker": "NVDA", "kind": "A1", "unit": "1", "price": 100.0}])
    db.record_run(conn, run_at="2026-09-10T07:30:00", as_of_date="2026-09-10", ticker_count=101, warning_count=2)

    events = conn.execute("SELECT date, ticker, kind FROM events").fetchall()
    runs = conn.execute("SELECT as_of_date, ticker_count, warning_count FROM runs").fetchall()
    assert events == [("2026-09-10", "NVDA", "A1")]
    assert runs == [("2026-09-10", 101, 2)]


def test_get_events_for_date_round_trips_detail_and_excludes_other_dates_and_funnel(tmp_path):
    """상태를 다시 계산하지 않고 보고서만 다시 만들 때 쓰는 조회 (P3.5 보완)."""
    conn = db.connect(tmp_path / "state.db")
    db.record_events(
        conn,
        [
            {"date": "2026-09-23", "ticker": "CMCSA", "kind": "A1", "unit": "1", "price": 22.78, "score": 35},
            {"date": "2026-09-23", "ticker": "", "kind": "FUNNEL", "1차 RSI 30 돌파": 3},
            {"date": "2026-09-22", "ticker": "AAPL", "kind": "A2", "unit": "2"},  # 다른 날짜
        ],
    )

    events = db.get_events_for_date(conn, "2026-09-23")

    assert len(events) == 1  # FUNNEL과 다른 날짜는 빠진다
    event = events[0]
    assert event["ticker"] == "CMCSA"
    assert event["kind"] == "A1"
    assert event["unit"] == "1"
    assert event["price"] == 22.78
    assert event["score"] == 35
    assert event["date"] == pd.Timestamp("2026-09-23")


def test_get_events_for_date_empty_when_none_recorded(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    assert db.get_events_for_date(conn, "2026-09-23") == []


def test_get_events_for_date_collapses_exact_duplicate_rows(tmp_path):
    """같은 날짜를 실수로 두 번 실제 처리해 완전히 같은 이벤트가 두 번 기록돼도
    (예: --replay 재실행이 이미 처리됨 가드를 우회) 보고서엔 한 번만 나와야 한다."""
    conn = db.connect(tmp_path / "state.db")
    event = {"date": "2026-09-23", "ticker": "CMCSA", "kind": "A1", "unit": "1", "price": 22.55, "score": 35}
    db.record_events(conn, [event, event])  # 같은 이벤트가 두 번 기록됨

    events = db.get_events_for_date(conn, "2026-09-23")

    assert len(events) == 1
    assert events[0]["ticker"] == "CMCSA"


def test_get_events_for_date_matches_timestamp_string_form(tmp_path):
    """record_events는 event["date"]가 pd.Timestamp면 str()로 시각까지 저장한다
    ("2026-09-23 00:00:00"). 순수 날짜 문자열로 조회해도 찾아져야 한다."""
    conn = db.connect(tmp_path / "state.db")
    db.record_events(conn, [{"date": pd.Timestamp("2026-09-23"), "ticker": "CMCSA", "kind": "A1", "unit": "1"}])

    stored_date = conn.execute("SELECT date FROM events").fetchone()[0]
    assert stored_date == "2026-09-23 00:00:00"  # 실제 저장 형태(시각 포함) 확인

    events = db.get_events_for_date(conn, "2026-09-23")
    assert len(events) == 1
    assert events[0]["ticker"] == "CMCSA"
