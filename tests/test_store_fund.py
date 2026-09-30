"""store/fund.py 테스트 (AI 펀드 F2, docs/design/fund_sim.md 3.4). 네트워크 없음."""

from __future__ import annotations

import pytest

from store import fund


@pytest.fixture
def conn(tmp_path):
    c = fund.connect(tmp_path / "fund.db")
    yield c
    c.close()


# ── params_hash ──────────────────────────────────────────────────────────


def test_params_hash_same_params_different_key_order_match():
    assert fund.params_hash({"a": 1, "b": 2}) == fund.params_hash({"b": 2, "a": 1})


def test_params_hash_different_params_differ():
    assert fund.params_hash({"a": 1}) != fund.params_hash({"a": 2})


# ── strategies ───────────────────────────────────────────────────────────


def test_upsert_and_load_strategy(conn):
    fund.upsert_strategy(conn, "team1", "MACD 추세", "core.strategies.macd_trend", "시계열", "2026-09-30T00:00:00")
    row = fund.load_strategy(conn, "team1")
    assert row["name"] == "MACD 추세"
    assert row["signal_type"] == "시계열"


def test_upsert_strategy_updates_name_keeps_id(conn):
    fund.upsert_strategy(conn, "team1", "이름1", None, None, "2026-09-30T00:00:00")
    fund.upsert_strategy(conn, "team1", "이름2", None, None, "2026-09-30T00:00:00")
    assert fund.load_strategy(conn, "team1")["name"] == "이름2"
    assert len(fund.load_all_strategies(conn)) == 1


def test_load_strategy_missing_returns_none(conn):
    assert fund.load_strategy(conn, "nope") is None


# ── variants ─────────────────────────────────────────────────────────────


def test_insert_and_load_variant(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    h = fund.insert_variant(conn, "team1-v0", "team1", {"rsi_threshold": 30}, "manual", "2026-09-30T00:00:00")
    row = fund.load_variant(conn, "team1-v0")
    assert row["params"] == {"rsi_threshold": 30}
    assert row["params_hash"] == h
    assert row["parent_variant_id"] is None
    assert row["created_by"] == "manual"


def test_insert_variant_twice_with_same_id_raises():
    """변형은 불변 — 같은 variant_id로 다시 만들면 오류(다른 파라미터로 몰래 덮어쓰기 방지)."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    fund._ensure_schema(conn)
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    with pytest.raises(ValueError, match="이미 있습니다"):
        fund.insert_variant(conn, "team1-v0", "team1", {"a": 2}, "manual", "2026-09-30T00:00:00")


def test_load_variants_for_strategy_filters_by_strategy(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.upsert_strategy(conn, "team2", "RSI", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team2-v0", "team2", {"b": 1}, "manual", "2026-09-30T00:00:00")
    rows = fund.load_variants_for_strategy(conn, "team1")
    assert [r["variant_id"] for r in rows] == ["team1-v0"]


def test_variant_lineage_walks_parent_chain(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0-pbt1", "team1", {"a": 1.1}, "PBT", "2026-10-01T00:00:00", parent_variant_id="team1-v0")
    fund.insert_variant(conn, "team1-v0-pbt2", "team1", {"a": 1.21}, "PBT", "2026-10-02T00:00:00", parent_variant_id="team1-v0-pbt1")
    lineage = fund.variant_lineage(conn, "team1-v0-pbt2")
    assert [v["variant_id"] for v in lineage] == ["team1-v0-pbt2", "team1-v0-pbt1", "team1-v0"]


def test_variant_lineage_single_generation_when_no_parent(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    lineage = fund.variant_lineage(conn, "team1-v0")
    assert [v["variant_id"] for v in lineage] == ["team1-v0"]


def test_variant_lineage_missing_variant_returns_empty(conn):
    assert fund.variant_lineage(conn, "nope") == []


# ── employees ────────────────────────────────────────────────────────────


def test_insert_and_load_employee(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    fund.insert_employee(conn, "emp-team1-1", "team1-v0", "2026-09-30", allocation_history=[{"date": "2026-09-30", "weight": 0.125}])
    row = fund.load_employee(conn, "emp-team1-1")
    assert row["variant_id"] == "team1-v0"
    assert row["active_to"] is None
    assert row["allocation_history"] == [{"date": "2026-09-30", "weight": 0.125}]


def test_load_active_employees_filters_by_date_range(conn):
    fund.upsert_strategy(conn, "team1", "MACD", None, None, "2026-09-30T00:00:00")
    fund.insert_variant(conn, "team1-v0", "team1", {"a": 1}, "manual", "2026-09-30T00:00:00")
    fund.insert_employee(conn, "emp-active", "team1-v0", "2026-01-01")
    fund.insert_employee(conn, "emp-retired", "team1-v0", "2025-01-01", active_to="2025-12-31")
    fund.insert_employee(conn, "emp-future", "team1-v0", "2027-01-01")

    active = fund.load_active_employees(conn, "2026-06-01")
    assert [e["employee_id"] for e in active] == ["emp-active"]
