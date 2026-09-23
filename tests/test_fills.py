"""data/fills.py 로더와 core.state.apply_fill 체결 반영 테스트. 네트워크 없음."""

from __future__ import annotations

import pandas as pd

from core import state as st
from data.fills import fills_for, load_fills


def test_load_fills_empty_file_returns_empty_frame(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text("date,ticker,unit,side,price,qty\n", encoding="utf-8")
    df = load_fills(path)
    assert df.empty


def test_load_fills_parses_rows(tmp_path):
    path = tmp_path / "fills.csv"
    path.write_text(
        "date,ticker,unit,side,price,qty\n2026-09-10,NVDA,1,buy,180.25,55\n",
        encoding="utf-8",
    )
    df = load_fills(path)
    assert len(df) == 1
    assert df.iloc[0]["unit"] == "1"
    assert df.iloc[0]["side"] == "buy"
    row = fills_for(df, "NVDA", pd.Timestamp("2026-09-10"))
    assert len(row) == 1
    assert row[0]["qty"] == 55


def test_apply_fill_records_real_price_and_qty(cfg):
    state = st.init_state("NVDA")
    state["state"] = "정찰"
    state["units"] = {"1": 0}
    state["entries"] = {"1": None}

    fill = {"unit": "1", "side": "buy", "price": 180.25, "qty": 55}
    new_state = st.apply_fill(state, fill, cfg)

    assert new_state["units"]["1"] == 55
    assert new_state["entries"]["1"] == 180.25


def test_apply_fill_unfilled_a1_returns_to_waiting_without_cooldown(cfg):
    """정찰 단계에서 A1이 체결 안 됨(qty=0)이면 쿨다운 없이 대기로 돌린다."""
    state = st.init_state("NVDA")
    state["state"] = "정찰"
    state["a1_date"] = pd.Timestamp("2026-09-10")
    state["stop"] = 170.0
    state["units"] = {"1": 0}
    state["entries"] = {"1": None}
    state["cooldown_until"] = pd.Timestamp("2026-09-20")  # 혹시 남아 있던 값도 지워져야 한다

    fill = {"unit": "1", "side": "buy", "price": 180.25, "qty": 0}
    new_state = st.apply_fill(state, fill, cfg)

    assert new_state["state"] == "대기"
    assert new_state["a1_date"] is None
    assert new_state["cooldown_until"] is None
    assert "1" not in new_state["units"]


def test_apply_fill_sell_reduces_held_qty(cfg):
    state = st.init_state("NVDA")
    state["units"] = {"1": 55}
    fill = {"unit": "1", "side": "sell", "price": 228.87, "qty": 55}
    new_state = st.apply_fill(state, fill, cfg)
    assert new_state["units"]["1"] == 0
