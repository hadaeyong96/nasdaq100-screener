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
