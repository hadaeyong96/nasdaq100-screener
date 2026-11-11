"""engine/pead_watch.py — 가짜 입력(네트워크 없음)으로 하루 실행·기록 규칙·편지 칸."""

from __future__ import annotations

import shutil
from datetime import date

import numpy as np
import pandas as pd
import pytest

from engine import pead_watch as ew
from store import pead_watch as st
from store import virtual_books as vbstore

DAYS = pd.bdate_range("2025-06-02", "2027-12-31")
D0 = pd.Timestamp("2026-10-21")  # A 반응일(수) — 장전 접수


def _px(seed, jump=None):
    rng = np.random.default_rng(seed)
    n = len(DAYS)
    c = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
    v = rng.uniform(1e6, 1.2e6, n)
    if jump is not None:
        i = DAYS.get_loc(jump)
        c[i:] *= 1.10
        v[i] *= 3
    return pd.DataFrame({"open": c * 0.998, "close": c, "volume": v}, index=DAYS)


def _src(raise_on=None):
    px = {"A": _px(1, D0), "B": _px(2), "QQQ": _px(3)}

    def prices(t):
        if raise_on == "prices":
            raise RuntimeError("boom")
        return px[t]

    filings = {
        1: [{"form": "8-K", "items": "2.02,9.01", "filing_date": "2026-07-22", "accn": "a-1"},
            {"form": "8-K", "items": "2.02,9.01", "filing_date": "2026-10-21", "accn": "a-2"}],
        2: [{"form": "8-K", "items": "2.02", "filing_date": "2026-10-20", "accn": "b-1"}],
    }
    acc = {"a-2": pd.Timestamp("2026-10-21 07:30:00"), "b-1": pd.Timestamp("2026-10-20 16:10:00"),
           "a-1": pd.Timestamp("2026-07-22 16:05:00")}
    return ew.Sources(
        current_ndx=lambda: pd.DataFrame({"ticker": ["A"], "name": ["A Corp"], "name_kr": ["에이"]}),
        ndx_at=lambda me, cur: ({"A"}, "가짜"),
        sp100_at=lambda me: ({"A", "B"}, "가짜"),
        cik_map=lambda: {"A": 1, "B": 2},
        recent_8k=lambda cik: filings[cik],
        accepted=lambda cik, accn: acc[accn],
        prices=prices,
        fx=lambda a, b: {d.date().isoformat(): 1400.0 for d in pd.bdate_range(a, b)},
        earnings=lambda tick: {"A": date(2026, 10, 28), "B": None},
        calendar=lambda a, b: DAYS[(DAYS >= pd.Timestamp(a)) & (DAYS <= pd.Timestamp(b))],
    )


CFG = {"pead_watch": {"enabled": True, "start_date": None}}


@pytest.fixture
def db(tmp_path):
    return tmp_path / "vb.db"


def _actions(monkeypatch, event="schedule"):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)


def test_preview_judges_signal_and_writes_nothing(db, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    res = ew.run(CFG, D0, db_path=db, src=_src(), start="2026-10-19", today_kst="2026-10-22")
    assert res["status"] == "preview"
    sig = [j for j in res["judged"] if j["signal"]]
    assert [(j["ticker"], j["d0"], j["label"]) for j in sig] == [("A", "2026-10-21", "둘 다")]
    assert {j["ticker"]: j["status"] for j in res["judged"]}["B"] == "조건 미달"  # B 반응일 10/21(장후 접수 10/20)
    L = res["letter"]
    assert [r["ticker"] for r in L["actions"]["buy"]] == ["A"] and L["next_day"] == "2026-10-22"
    assert L["new"][0]["buy_day"] == "2026-10-22" and L["new"][0]["name"] == "에이"
    assert L["earnings"]["failed"] == 1 and [r["ticker"] for r in L["earnings"]["rows"]] == ["A"]
    conn = vbstore.connect(db)
    st.ensure_schema(conn)
    assert st.load_trades(conn) == [] and st.get_meta(conn, st.META_START) is None
    conn.close()


def test_schedule_run_writes_and_next_run_fills_buy(db, monkeypatch):
    _actions(monkeypatch)
    r1 = ew.run(CFG, D0, db_path=db, src=_src(), today_kst="2026-10-22")
    assert r1["status"] == "ok" and r1["start_date"] == "2026-10-21"
    conn = vbstore.connect(db)
    assert st.get_meta(conn, st.META_START) == "2026-10-21" and st.get_meta(conn, st.META_LAST) == "2026-10-21"
    tr = {(t["ledger"], t["ticker"]): t for t in st.load_trades(conn)}
    assert tr[("main", "A")]["status"] == "pending" and tr[("main", "A")]["plan_day"] == DAYS[DAYS.get_loc("2026-10-22") + 90].date().isoformat()
    assert {t[0] for t in tr} == {"main", "compare", "reference"}
    conn.close()
    r2 = ew.run(CFG, pd.Timestamp("2026-10-27"), db_path=db, src=_src(), today_kst="2026-10-28")
    assert r2["todo"] == ["2026-10-22", "2026-10-23", "2026-10-26", "2026-10-27"]
    w = r2["letter"]["watching"][0]
    assert w["status"] == "open" and w["held"] == 3 and w["ret"] is not None
    conn = vbstore.connect(db)
    a = next(t for t in st.load_trades(conn) if t["ledger"] == "main")
    assert a["status"] == "open" and a["fx_buy"] == 1400.0 and a["buy_px"] == pytest.approx(_px(1, D0).at[pd.Timestamp("2026-10-22"), "open"])
    conn.close()
    r3 = ew.run(CFG, pd.Timestamp("2026-10-27"), db_path=db, src=_src(), today_kst="2026-10-28")
    assert r3["status"] == "already" and r3["letter"]["watching"][0]["held"] == 3


def test_manual_dispatch_does_not_write(db, monkeypatch):
    _actions(monkeypatch, "workflow_dispatch")
    res = ew.run(CFG, D0, db_path=db, src=_src())
    assert res["status"] == "preview"
    conn = vbstore.connect(db)
    assert st.get_meta(conn, st.META_START) is None
    conn.close()


def test_run_safely_wraps_errors_and_respects_enabled(db):
    assert ew.run_safely({"pead_watch": {"enabled": False}}, D0) is None
    res = ew.run_safely(CFG, D0, db_path=db, src=_src(raise_on="prices"), start="2026-10-19")
    assert res["status"] == "error" and res["letter"]["new"] == []


def test_prereg_hash_mismatch_stops(tmp_path):
    for rel in ew.PREREG_FILES:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ew.ROOT / rel, tmp_path / rel)
    ew.verify_prereg(tmp_path)
    (tmp_path / ew.PREREG_YAML).write_text("changed: true\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="사전 등록"):
        ew.verify_prereg(tmp_path)


def test_chart_data_lines_start_at_buy_day(db, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    res = ew.run(CFG, pd.Timestamp("2026-11-10"), db_path=db, src=_src(), start="2026-10-19")
    ch = res["letter"]["chart"]
    ln = ch["lines"][0]
    assert ln["ticker"] == "A" and ln["dates"][0] == "2026-10-22" and ln["values"][0] == pytest.approx(
        (_px(1, D0).at[pd.Timestamp("2026-10-22"), "close"] / _px(1, D0).at[pd.Timestamp("2026-10-22"), "open"] - 1) * 100)
    assert ch["qqq"]["dates"][0] == "2026-10-22" and ch["qqq"]["dates"][-1] == "2026-11-10"
