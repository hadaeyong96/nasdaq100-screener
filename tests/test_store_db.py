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


# ── 라이브 어드바이저 1단계: last_judgment, fill_ledger, daily_mark ──────────


def test_position_last_judgment_round_trips_and_defaults_to_none(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    state = st.init_state("NVDA", "엔비디아")  # core.state.init_state는 last_judgment를 모른다
    db.save_position(conn, state)
    assert db.load_position(conn, "NVDA")["last_judgment"] is None

    state["last_judgment"] = "추가매수"
    db.save_position(conn, state)
    assert db.load_position(conn, "NVDA")["last_judgment"] == "추가매수"


def test_last_judgment_column_migrates_onto_pre_existing_db(tmp_path):
    """1단계 이전에 만들어진 DB(last_judgment 컬럼 없음)에 connect()하면
    마이그레이션으로 컬럼이 생겨야 한다 (pending 컬럼과 같은 패턴)."""
    path = tmp_path / "state.db"
    import sqlite3

    raw = sqlite3.connect(path)
    raw.execute(
        """
        CREATE TABLE positions (
            ticker TEXT PRIMARY KEY, name_kr TEXT, state TEXT, units TEXT, entries TEXT,
            stop REAL, a1_date TEXT, cooldown_until TEXT, sent_alerts TEXT, updated_at TEXT,
            b_entry_date TEXT, b_total_qty INTEGER, pending TEXT
        )
        """
    )
    raw.commit()
    raw.close()

    conn = db.connect(path)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(positions)")}
    assert "last_judgment" in columns


def test_record_and_get_fill_ledger_round_trip(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_fill_ledger(
        conn,
        [
            {
                "date": "2026-09-25",
                "mode": "live",
                "ticker": "NVDA",
                "side": "buy",
                "qty": 4,
                "price_usd": 180.25,
                "fx_rate": 1350.0,
                "amount_krw": 973_350,
                "qqqm_close": 220.10,
            },
        ],
    )
    rows = db.get_fill_ledger(conn)
    assert len(rows) == 1
    row = rows[0]
    assert row["ticker"] == "NVDA"
    assert row["mode"] == "live"
    assert row["side"] == "buy"
    assert row["qty"] == 4
    assert row["amount_krw"] == 973_350
    assert row["qqqm_close"] == 220.10


def test_get_fill_ledger_filters_by_mode_and_preserves_insertion_order(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_fill_ledger(
        conn,
        [
            {"date": "2026-09-24", "mode": "paper", "ticker": "AAPL", "side": "buy", "qty": 1, "price_usd": 1, "fx_rate": 1, "amount_krw": 1, "qqqm_close": 1},
            {"date": "2026-09-25", "mode": "live", "ticker": "NVDA", "side": "buy", "qty": 4, "price_usd": 180.25, "fx_rate": 1350.0, "amount_krw": 973_350, "qqqm_close": 220.10},
            {"date": "2026-09-26", "mode": "live", "ticker": "AVGO", "side": "sell", "qty": 2, "price_usd": 300, "fx_rate": 1350, "amount_krw": 810_000, "qqqm_close": 222.0},
        ],
    )
    live_rows = db.get_fill_ledger(conn, mode="live")
    assert [r["ticker"] for r in live_rows] == ["NVDA", "AVGO"]

    all_rows = db.get_fill_ledger(conn)
    assert [r["ticker"] for r in all_rows] == ["AAPL", "NVDA", "AVGO"]


def test_get_fill_ledger_empty_when_none_recorded(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    assert db.get_fill_ledger(conn) == []


def test_record_daily_mark_upserts_same_day_ticker_mode(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_daily_mark(
        conn, [{"date": "2026-09-25", "mode": "live", "ticker": "NVDA", "qty": 4, "price_usd": 180.25, "fx_rate": 1350.0}]
    )
    db.record_daily_mark(
        conn, [{"date": "2026-09-25", "mode": "live", "ticker": "NVDA", "qty": 4, "price_usd": 183.00, "fx_rate": 1352.0}]
    )
    rows = db.get_daily_mark(conn)
    assert len(rows) == 1  # 같은 날짜+모드+종목은 덮어쓴다(행이 늘지 않는다)
    assert rows[0]["price_usd"] == 183.00
    assert rows[0]["fx_rate"] == 1352.0


def test_get_daily_mark_filters_by_mode_and_orders_by_date_then_ticker(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    db.record_daily_mark(
        conn,
        [
            {"date": "2026-09-25", "mode": "paper", "ticker": "ZZZ", "qty": 1, "price_usd": 1, "fx_rate": 1},
            {"date": "2026-09-24", "mode": "live", "ticker": "AAPL", "qty": 1, "price_usd": 1, "fx_rate": 1},
            {"date": "2026-09-25", "mode": "live", "ticker": "NVDA", "qty": 4, "price_usd": 180.25, "fx_rate": 1350.0},
        ],
    )
    live_rows = db.get_daily_mark(conn, mode="live")
    assert [(r["date"], r["ticker"]) for r in live_rows] == [("2026-09-24", "AAPL"), ("2026-09-25", "NVDA")]


def test_get_daily_mark_empty_when_none_recorded(tmp_path):
    conn = db.connect(tmp_path / "state.db")
    assert db.get_daily_mark(conn) == []
