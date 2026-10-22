"""추천 따라하기 장부 2개 — 엔진(engine/picks_books.py·engine/virtual_books.py) 테스트. 네트워크 없음.

여러 날에 걸친 실행(하루씩 이어 가기), 같은 날 재실행, dry-run, 오류 격리, DB 유실 시 새로 시작하지 않음,
명단 이탈 종목, 시트 세 탭에서 DB 복구, 시트·요약 열을 확인한다. 신호는 test_picks_book.STUB로 통제한다.
"""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core import picks_book as pb
from data import virtual_books_sheet as vbsheet
from engine import picks_books
from engine import virtual_books as vbe
from store import virtual_books as vbstore
from tests.test_picks_book import STUB
from tests.legacy import legacy_cfg

NOW_ET = datetime(2030, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))
IDX = pd.bdate_range("2024-01-02", periods=280)
B1, B2 = "추천 종목 따라하기", "볼린저·RSI·MACD 따라하기"
S = 260  # 첫 실행 기준일 위치


class Market:
    """기준일(upto)을 바꿔 가며 ETF·종목 데이터를 잘라 주는 가짜 시장."""

    def __init__(self):
        self.upto = S
        self.universe = ["AAA", "BBB", "CCC"]
        n = len(IDX)
        self.frames = {t: pd.DataFrame({
            "close": 50.0 + i * 10 + np.zeros(n), "rsi": 40.0, "swing_low": 1.0, "dc": False, "gc": False,
            "macd_norm": 0.0, "x_score": 10 + i, "x_a1": False, "x_b": False, "x_a2": False, "x_a3": False, "x_e3": False,
        }, index=IDX) for i, t in enumerate(["AAA", "BBB", "CCC", "ZZZ"])}
        self.calls = 0

    def on(self, ticker, col, pos, value=True):
        self.frames[ticker].iloc[pos, self.frames[ticker].columns.get_loc(col)] = value

    def history_fn(self, ticker, start):
        idx = IDX[: self.upto + 1].tz_localize("America/New_York")
        close = {"QQQ": 400.0, "QLD": 80.0, "TQQQ": 60.0, "SGOV": 100.0}[ticker] * (1 + 0.0005 * np.arange(len(idx)))
        close = close if ticker != "SGOV" else np.full(len(idx), 100.0)
        return pd.DataFrame({"Close": close, "Dividends": 0.0}, index=idx)

    def market_fn(self, cfg, extra):
        self.calls += 1
        return {"universe": list(self.universe), "earnings": {}, "warnings": [],
                "frames": {t: df.iloc[: self.upto + 1] for t, df in self.frames.items()}}


def actions_fn(tickers, start):
    return {t: pd.DataFrame({"dividend": [], "split": []}) for t in tickers}


class FakeWs:
    def __init__(self, title):
        self.title, self.values, self.row_count, self.col_count = title, [], 1000, 30

    def clear(self):
        self.values = []

    def resize(self, rows, cols):
        self.row_count, self.col_count = rows, cols

    def update(self, values, range_name, value_input_option):
        self.values = [[("" if v is None else str(v)) for v in r] for r in values]

    def get_all_values(self):
        return self.values


class FakeClient:
    def __init__(self):
        self.tabs = {}

    def open_by_key(self, key):
        return self

    def worksheet(self, name):
        import gspread

        if name not in self.tabs:
            raise gspread.WorksheetNotFound(name)
        return self.tabs[name]

    def add_worksheet(self, title, rows, cols):
        self.tabs[title] = FakeWs(title)
        return self.tabs[title]


@pytest.fixture
def cfg(monkeypatch):
    from engine.daily import load_config

    monkeypatch.setitem(picks_books.RULES, "stub", STUB)
    monkeypatch.setenv("VB_TEST_SHEET_ID", "sheet")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    c = legacy_cfg(load_config())
    vb = c["virtual_books"]
    vb.update({"start_date": None, "start_fx": None, "sheet_id_env": "VB_TEST_SHEET_ID"})
    vb["picks"].update({"start_date": None, "start_fx": None,
                        "books": [{"name": B1, "rules": "stub"}, {"name": B2, "rules": "stub"}]})
    vb["momentum"]["enabled"] = False  # 8번 장부는 tests/test_momentum_book.py에서 따로 본다
    vb["accel"]["enabled"] = False  # 9번 장부는 tests/test_accel_book.py에서 따로 본다
    return c


def _run(cfg, m: Market, tmp_path, client=None, **kw):
    return vbe.run(cfg, db_path=tmp_path / "vb.db", sheets_client=client or FakeClient(), now_et=NOW_ET,
                   history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None),
                   year_end_fn=lambda a, b: set(), next_day_fn=lambda d: None,
                   market_fn=m.market_fn, actions_fn=actions_fn, **kw)


def _days(cfg, m, tmp_path, upto, client=None, briefing=None):
    out = None
    while m.upto < upto:
        m.upto += 1
        out = _run(cfg, m, tmp_path, client=client, briefing=(briefing or {}).get(m.upto))
    return out


def _db(tmp_path, table, book):
    conn = vbstore.connect(tmp_path / "vb.db")
    try:
        return vbstore.picks_rows(conn, table, book)
    finally:
        conn.close()


def test_first_run_starts_in_sgov_then_fills_next_day_and_writes_sheet(cfg, tmp_path):
    m = Market()
    m.on("AAA", "x_a1", S)
    client = FakeClient()
    r0 = _run(cfg, m, tmp_path, client=client, briefing={"buy": ["AAA"], "sell": []})
    assert not r0["errors"], r0["errors"]
    assert {x["name"] for x in r0["picks"]["extra"]} == {B1, B2}
    assert [s["ticker"] for s in r0["picks"]["today"][B1] if s["kind"] == "A1"] == ["AAA"]
    assert _db(tmp_path, "picks_trades", B1) == []  # 시작일에는 SGOV만(매매기록 없음)
    r1 = _days(cfg, m, tmp_path, S + 1, client=client)
    trades = _db(tmp_path, "picks_trades", B1)
    assert [(t["side"], t["ticker"], t["signal_date"], t["fill_date"], t["in_briefing"]) for t in trades] == \
        [("매수", "AAA", IDX[S].date().isoformat(), IDX[S + 1].date().isoformat(), 1)]
    assert len(_db(tmp_path, "picks_daily", B1)) == 2
    # 시트: 일별 열·매매기록 열·요약 추가 열
    daily_head = client.tabs[vbsheet.TAB_DAILY].values[0]
    L1, L2 = "기술지표 전략", "볼린저 전략"  # 시트에는 보여 주는 이름(2026-10-17, config display_names)
    assert f"{L1} 총자산($)" in daily_head and f"{L2} QQQ 대비(%p)" in daily_head
    assert not any(B1 in h for h in daily_head)
    th = client.tabs[vbsheet.TAB_TRADES].values
    assert th[0][-1] == "브리핑 추천에도 있었음"
    assert [r for r in th if r[0] == L1][0][-1] == "예"
    summ = client.tabs[vbsheet.TAB_SUMMARY].values
    head = summ[0]
    assert head[-6:] == ["완료 거래 수", "승률(%)", "평균 보유일", "SGOV 보유액($)", "교체 횟수", "현재 보유 10종목"]
    row_b1 = next(r for r in summ if r and r[0] == L1)
    row_qqq = next(r for r in summ if r and r[0] == "QQQ 전략")
    assert row_b1[head.index("완료 거래 수")] == "0" and float(row_b1[head.index("SGOV 보유액($)")]) > 0
    assert row_qqq[head.index("완료 거래 수")] == ""
    assert any(r and r[0] == "config 고정값(picks)" for r in summ)


def test_same_day_rerun_changes_nothing_and_dry_run_writes_nothing(cfg, tmp_path):
    m = Market()
    m.on("AAA", "x_a1", S)
    _run(cfg, m, tmp_path)
    _days(cfg, m, tmp_path, S + 2)
    before = (_db(tmp_path, "picks_daily", B1), _db(tmp_path, "picks_trades", B1))
    conn = vbstore.connect(tmp_path / "vb.db")
    state_before = vbstore.load_picks_book(conn, B1)
    conn.close()
    _run(cfg, m, tmp_path)  # 같은 기준일 다시
    m.upto += 1
    r = _run(cfg, m, tmp_path, dry_run=True)  # 다음 날 dry-run: 결과는 보이지만 DB는 그대로
    assert any(x["name"] == B1 and x["daily"].index[-1] == IDX[S + 3] for x in r["picks"]["extra"])
    assert (_db(tmp_path, "picks_daily", B1), _db(tmp_path, "picks_trades", B1)) == before
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.load_picks_book(conn, B1) == state_before
    conn.close()


def test_errors_are_isolated_from_other_books_and_briefing(cfg, tmp_path):
    m = Market()

    def broken(cfg_, extra):
        raise ConnectionError("prices down")

    r = vbe.run(cfg, db_path=tmp_path / "vb.db", sheets_client=FakeClient(), now_et=NOW_ET, history_fn=m.history_fn,
                fx_fn=lambda d: (1400.0, d.isoformat(), None), year_end_fn=lambda a, b: set(),
                market_fn=broken, actions_fn=actions_fn)
    assert r["regime_line"].startswith("오늘 국면:")  # 기존 장부·국면 줄은 그대로
    assert len(r["errors"]) == 2 and all("prices down" in e for e in r["errors"])
    conn = vbstore.connect(tmp_path / "vb.db")
    assert vbstore.count_rows(conn, "daily") == 5  # 기존 5개 장부는 저장됨
    conn.close()

    cfg["virtual_books"]["picks"]["books"][1]["rules"] = "없는규칙"
    ops = []
    r2 = vbe.run_safely(cfg, ops_sender=ops.append, db_path=tmp_path / "vb2.db", sheets_client=FakeClient(),
                        now_et=NOW_ET, history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None),
                        year_end_fn=lambda a, b: set(), market_fn=m.market_fn, actions_fn=actions_fn)
    assert r2 is not None and [x["name"] for x in r2["picks"]["extra"]] == [B1]  # 한 장부만 실패
    assert len(ops) == 1 and B2 in ops[0]


def test_lost_db_with_fixed_start_does_not_restart(cfg, tmp_path):
    m = Market()
    cfg["virtual_books"]["picks"]["start_date"] = IDX[S - 5].date().isoformat()
    r = _run(cfg, m, tmp_path)
    assert all("--recover-picks" in e for e in r["errors"]) and r["picks"]["extra"] == []
    assert _db(tmp_path, "picks_daily", B1) == []


def test_held_ticker_dropped_from_universe_is_still_tracked(cfg, tmp_path):
    m = Market()
    m.universe = ["AAA", "ZZZ"]
    m.on("ZZZ", "x_a1", S)
    _run(cfg, m, tmp_path)
    _days(cfg, m, tmp_path, S + 2)
    m.universe = ["AAA"]  # 명단에서 빠짐
    m.on("ZZZ", "x_a2", S + 3)
    m.on("ZZZ", "dc", S + 5)
    _days(cfg, m, tmp_path, S + 6)
    trades = _db(tmp_path, "picks_trades", B1)
    assert [(t["side"], t["ticker"], t["kind"]) for t in trades] == [("매수", "ZZZ", "A1"), ("매도", "ZZZ", "E1")]


def test_recover_from_sheet_rebuilds_positions_and_value(cfg, tmp_path):
    m = Market()
    for t, pos in (("AAA", S), ("BBB", S + 1), ("CCC", S + 2)):
        m.on(t, "x_a1", pos)
    m.on("BBB", "swing_low", S + 1, 70.0)  # BBB(종가 60) 진입일 손절가 70 → 다음 날 손절
    client = FakeClient()
    _run(cfg, m, tmp_path, client=client)
    _days(cfg, m, tmp_path, S + 6, client=client)
    conn = vbstore.connect(tmp_path / "vb.db")
    orig = vbstore.load_picks_book(conn, B1)
    conn.close()
    assert set(orig["positions"]) == {"AAA", "CCC"}

    new_db = tmp_path / "new"
    new_db.mkdir()
    out = vbe.recover_picks(cfg, db_path=new_db / "vb.db", sheets_client=client, market_fn=m.market_fn,
                            history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None), now_et=NOW_ET)
    assert set(out) == {B1, B2} and all(v.startswith("복구") for v in out.values())
    conn = vbstore.connect(new_db / "vb.db")
    rec = vbstore.load_picks_book(conn, B1)
    assert {t: p["qty"] for t, p in rec["positions"].items()} == {t: p["qty"] for t, p in orig["positions"].items()}
    assert rec["last_date"] == orig["last_date"] and rec["start_date"] == orig["start_date"]
    value = rec["cash"] + rec["sgov_qty"] * 100.0 + sum(p["qty"] * p["last_close"] for p in rec["positions"].values())
    orig_value = orig["cash"] + orig["sgov_qty"] * 100.0 + sum(p["qty"] * p["last_close"] for p in orig["positions"].values())
    assert value == pytest.approx(orig_value, abs=0.01)  # 시트 총자산이 소수 둘째 자리라 그 정도 차이
    assert rec["signal_states"]["AAA"]["state"] == "정찰" and rec["signal_states"]["BBB"]["state"] == "대기"
    assert len(vbstore.picks_rows(conn, "picks_trades", B1)) == len(_db(tmp_path, "picks_trades", B1))
    conn.close()
    # 이미 상태가 있으면 덮어쓰지 않는다
    again = vbe.recover_picks(cfg, db_path=new_db / "vb.db", sheets_client=client, market_fn=m.market_fn,
                              history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None), now_et=NOW_ET)
    assert all("건너뜀" in v for v in again.values())


def test_bollinger_rules_path_runs_on_real_indicators(cfg, tmp_path, monkeypatch):
    from core.indicators import compute_indicators

    cfg["virtual_books"]["picks"]["books"] = [{"name": B2, "rules": "bollinger"}]
    m = Market()
    rng = np.random.default_rng(3)
    for t in ("AAA", "BBB"):
        close = 100 * np.cumprod(1 + rng.normal(0, 0.02, len(IDX)))
        raw = pd.DataFrame({"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e6}, index=IDX)
        m.frames[t] = compute_indicators(raw, cfg)
    m.universe = ["AAA", "BBB"]
    m.frames = {t: m.frames[t] for t in ("AAA", "BBB")}
    _run(cfg, m, tmp_path)
    r = _days(cfg, m, tmp_path, S + 15)
    assert not r["errors"], r["errors"]
    assert len(_db(tmp_path, "picks_daily", B2)) == 16


# ── 빈 종가·따라잡기 (2026-10-15 수정) ────────────────────────────────────────


def _ledger(tmp_path, book=B1):
    """비교용: 일별 행(날짜·총자산·현금)과 매매 행(체결일·구분·종목·수량)."""
    daily = [(r["date"], round(float(r["value"]), 6)) for r in _db(tmp_path, "picks_daily", book)]
    trades = [(t["fill_date"], t["side"], t["ticker"], t["qty"]) for t in _db(tmp_path, "picks_trades", book)]
    return daily, trades


def _signal_market():
    m = Market()
    m.on("AAA", "x_a1", S)
    m.on("BBB", "x_b", S + 2)
    m.on("AAA", "dc", S + 4)
    return m


def test_skipped_runs_catch_up_in_order_same_as_daily_runs(cfg, tmp_path):
    """실행이 며칠 빠져도(예약 실패·지연) 다음 실행이 빠진 날을 순서대로 처리 — 매일 돈 것과 같다."""
    daily_path, gap_path = tmp_path / "daily", tmp_path / "gap"
    m1 = _signal_market()
    _run(cfg, m1, daily_path)
    _days(cfg, m1, daily_path, S + 6)
    m2 = _signal_market()
    _run(cfg, m2, gap_path)
    m2.upto = S + 6  # S+1 ~ S+5 실행 없음
    r = _run(cfg, m2, gap_path)
    assert not r["errors"], r["errors"]
    assert [d for d, _ in _ledger(gap_path)[0]] == [IDX[i].date().isoformat() for i in range(S, S + 7)]
    assert _ledger(gap_path) == _ledger(daily_path)
    assert _ledger(gap_path, B2) == _ledger(daily_path, B2)
    assert any(side == "매수" for _, side, _, _ in _ledger(gap_path)[1])


def test_failed_run_is_caught_up_next_run(cfg, tmp_path):
    m = _signal_market()
    _run(cfg, m, tmp_path)
    _days(cfg, m, tmp_path, S + 2)
    good = m.market_fn
    m.upto = S + 3
    m.market_fn = lambda c, e: (_ for _ in ()).throw(ConnectionError("prices down"))
    r_fail = _run(cfg, m, tmp_path)
    assert r_fail["errors"] and _db(tmp_path, "picks_daily", B1)[-1]["date"] == IDX[S + 2].date().isoformat()
    m.market_fn = good
    m.upto = S + 4
    _run(cfg, m, tmp_path)
    ref = _signal_market()
    _run(cfg, ref, tmp_path / "ref")
    _days(cfg, ref, tmp_path / "ref", S + 4)
    assert _ledger(tmp_path) == _ledger(tmp_path / "ref")


def test_last_day_with_empty_closes_is_deferred_then_processed(cfg, tmp_path):
    """맨 끝 날 종목 종가가 비면 그날은 미루고 경고 — 다음 실행에서 채워진 값으로 처리, 매일 돈 것과 같다."""
    m = _signal_market()
    _run(cfg, m, tmp_path)
    _days(cfg, m, tmp_path, S + 2)
    m.upto = S + 3
    clean = {t: df.copy() for t, df in m.frames.items()}
    for t in ("AAA", "BBB"):
        m.on(t, "close", S + 3, np.nan)
    r = _run(cfg, m, tmp_path)
    assert any("다음 실행에서 처리" in w and "AAA" in w for w in r["warnings"])
    assert _db(tmp_path, "picks_daily", B1)[-1]["date"] == IDX[S + 2].date().isoformat()
    m.frames = clean  # 다음 실행 때는 종가가 들어와 있다
    m.upto = S + 4
    _run(cfg, m, tmp_path)
    ref = _signal_market()
    _run(cfg, ref, tmp_path / "ref")
    _days(cfg, ref, tmp_path / "ref", S + 4)
    assert _ledger(tmp_path) == _ledger(tmp_path / "ref")


def test_ticker_already_missing_before_last_day_does_not_defer():
    """전 거래일부터 이미 없던 종목(거래 정지 등)은 미루기 사유가 아니다 — 많아야 하루만 미룬다."""
    idx = IDX[:5]
    frames = {"AAA": pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx),
              "HALT": pd.DataFrame({"close": [1.0, 1.0, 1.0, np.nan, np.nan]}, index=idx)}
    days, w = picks_books.defer_incomplete_last_day(list(idx[3:]), frames, ["AAA", "HALT"], idx)
    assert days == list(idx[3:]) and w is None
    frames["AAA"].iloc[-1, 0] = np.nan
    days, w = picks_books.defer_incomplete_last_day(list(idx[3:]), frames, ["AAA", "HALT"], idx)
    assert days == [idx[3]] and "AAA" in w and "HALT" not in w


def test_start_day_deferred_keeps_fixed_start_without_restart_error(cfg, tmp_path):
    """시작일 당일이 미뤄져도 고정 시작일 장부가 '상태 없음' 오류로 막히지 않고 다음 실행에서 시작일부터 처리."""
    cfg["virtual_books"]["picks"]["start_date"] = IDX[S].date().isoformat()
    m = _signal_market()
    clean = {t: df.copy() for t, df in m.frames.items()}
    m.on("AAA", "close", S, np.nan)
    r0 = _run(cfg, m, tmp_path)
    assert not r0["errors"], r0["errors"]
    assert _db(tmp_path, "picks_daily", B1) == []
    m.frames = clean
    m.upto = S + 1
    r1 = _run(cfg, m, tmp_path)
    assert not r1["errors"], r1["errors"]
    assert [r["date"] for r in _db(tmp_path, "picks_daily", B1)] == [IDX[S].date().isoformat(), IDX[S + 1].date().isoformat()]
