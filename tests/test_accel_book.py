"""9번 장부 「한 달 가속 따라하기」 (작업 6-2) — 네트워크 없이 합성 데이터만.

20거래일 주기(휴장 포함), 60거래일 미만·5달러 미만 제외, 점수 계산, 같은 비중 재조정, 빈 종가(맨 끝 날 미루기),
시작(처음 돈 날 = 첫 신호), 따라잡기(빠진 날 순서대로·저장된 명단), 전략요약 열.
"""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core import accel
from engine import picks_books
from engine import virtual_books as vbe
from store import virtual_books as vbstore

NAME = "한 달 가속 따라하기"


# ── 20거래일 주기 (NYSE 달력) ────────────────────────────────────────────


def test_cycle_counts_nyse_trading_days_skipping_holidays():
    got = picks_books.accel_cycle_days("2026-11-20", "2027-03-31", 20)
    holidays = {pd.Timestamp(x) for x in ("2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15")}
    bdays = [d for d in pd.bdate_range("2026-11-20", "2027-03-31") if d not in holidays]
    assert got == bdays[::20]
    assert got[0] == pd.Timestamp("2026-11-20") and got[1] == pd.Timestamp("2026-12-21")  # 추수감사절이 빠져 하루 밀림
    assert all(d not in holidays for d in got)


# ── 점수·제외 (core.accel, 브리핑과 같은 함수) ─────────────────────────────


IDX = pd.bdate_range("2024-01-02", periods=320)


def _path(n, p0, p1, p2):
    """끝에서 40거래일 전 p0, 20거래일 전 p1, 마지막 p2 (그 사이 선형, 그 전은 p0)."""
    return np.r_[np.full(n - 41, p0), np.linspace(p0, p1, 21), np.linspace(p1, p2, 21)[1:]]


def test_score_and_exclusions_match_rule():
    d = IDX[100]
    q = pd.Series(_path(101, 100, 110, 121), index=IDX[:101])
    closes = {"A": pd.Series(_path(101, 10, 11, 14.52), index=IDX[:101]),
              "SHORT": pd.Series(_path(101, 10, 11, 14.52), index=IDX[:101]).iloc[-59:],
              "PENNY": pd.Series(_path(101, 3, 3.3, 4.5), index=IDX[:101])}
    r = picks_books.accel_signal(closes, q, d, list(closes), accel.AccelParams())
    assert [x["ticker"] for x in r["targets"]] == ["A"]
    assert r["targets"][0]["score_pp"] == pytest.approx(20.0)  # 최근 +20% − 직전 0% (RS 기준)
    assert set(r["excluded"]) == {"SHORT", "PENNY"}


# ── 엔진 ────────────────────────────────────────────────────────────────


NOW_ET = datetime(2030, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))


def _cycle(start, d):
    return int(((IDX >= pd.Timestamp(start)) & (IDX <= pd.Timestamp(d))).sum())


class Market:
    """종목마다 다른 가속을 주는 가짜 시장 — 신호일마다 순위가 바뀌도록 위상을 다르게."""

    def __init__(self, upto):
        self.upto = upto
        self.universe = [f"S{i:02d}" for i in range(14)]
        n = len(IDX)
        x = np.arange(n)
        self.frames = {t: pd.DataFrame({"close": 50 + 10 * np.sin(x / (15 + i) + i)}, index=IDX)
                       for i, t in enumerate(self.universe)}

    def history_fn(self, ticker, start):
        idx = IDX[: self.upto + 1].tz_localize("America/New_York")
        base = {"QQQ": 400.0, "QLD": 80.0, "TQQQ": 60.0, "SGOV": 100.0}[ticker]
        close = base * (1 + 0.0005 * np.arange(len(idx))) if ticker != "SGOV" else np.full(len(idx), 100.0)
        return pd.DataFrame({"Close": close, "Dividends": 0.0}, index=idx)

    def market_fn(self, cfg, extra):
        return {"universe": list(self.universe), "earnings": {}, "warnings": [],
                "frames": {t: df.iloc[: self.upto + 1] for t, df in self.frames.items()}}


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
        ws = type("Ws", (), {})()
        ws.title, ws.values, ws.row_count, ws.col_count = title, [], 1000, 30
        ws.clear = lambda: setattr(ws, "values", [])
        ws.resize = lambda rows, cols: None
        ws.update = lambda values, range_name, value_input_option: setattr(ws, "values", [[("" if v is None else str(v)) for v in r] for r in values])
        self.tabs[title] = ws
        return ws


@pytest.fixture
def cfg(monkeypatch):
    from engine.daily import load_config

    monkeypatch.setenv("VB_TEST_SHEET_ID", "sheet")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    c = copy.deepcopy(load_config())
    vb = c["virtual_books"]
    vb.update({"start_date": IDX[201].date().isoformat(), "start_fx": None, "sheet_id_env": "VB_TEST_SHEET_ID"})
    vb["picks"]["enabled"] = False
    vb["momentum"]["enabled"] = False
    vb["accel"].update({"enabled": True, "start_fx": None, "start_date": None})
    return c


def _run(cfg, m, path, client=None, **kw):
    return vbe.run(cfg, db_path=path / "vb.db", sheets_client=client or FakeClient(), now_et=NOW_ET,
                   history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None),
                   year_end_fn=lambda a, b: set(), next_day_fn=lambda d: None, market_fn=m.market_fn,
                   actions_fn=lambda t, s: {x: pd.DataFrame({"dividend": [], "split": []}) for x in t},
                   cycle_fn=_cycle, **kw)


def _rows(path, table):
    conn = vbstore.connect(path / "vb.db")
    try:
        return vbstore.picks_rows(conn, table, NAME)
    finally:
        conn.close()


def _daily_runs(cfg, m, path, upto, client=None):
    while m.upto < upto:
        m.upto += 1
        _run(cfg, m, path, client=client)


ARMED = 210


def test_first_run_is_first_signal_then_every_20_trading_days(cfg, tmp_path):
    m = Market(ARMED)
    r = _run(cfg, m, tmp_path)
    assert not r["errors"], r["errors"]
    assert [d["date"] for d in _rows(tmp_path, "picks_daily")] == [IDX[ARMED].date().isoformat()]  # 시작일 = 첫 신호(현금)
    client = FakeClient()
    _daily_runs(cfg, m, tmp_path, ARMED + 45, client=client)
    fills = sorted({t["fill_date"] for t in _rows(tmp_path, "picks_trades") if t["side"] != "건너뜀"})
    assert fills == [IDX[ARMED + 1].date().isoformat(), IDX[ARMED + 21].date().isoformat(), IDX[ARMED + 41].date().isoformat()]
    first = [t for t in _rows(tmp_path, "picks_trades") if t["fill_date"] == IDX[ARMED + 1].date().isoformat()]
    assert len(first) == 10 and all(t["signal_date"] == IDX[ARMED].date().isoformat() for t in first)
    summ = client.tabs["전략요약"].values
    head, row = summ[0], next(x for x in summ if x and x[0] == "QQQ 대비 1개월 강세 전략")  # 시트는 보여 주는 이름
    assert row[head.index("교체 횟수")] == "3" and row[head.index("현재 보유 10종목")].count(",") == 9
    assert any(x and x[0] == "config 고정값(accel)" and f'start_date: "{IDX[ARMED].date().isoformat()}"' in x[1] for x in summ)
    assert any(x and x[0] == "시작 조건(QQQ 대비 1개월 강세)" for x in summ)


def test_rebalance_keeps_equal_weight_and_trades_only_difference(cfg, tmp_path):
    m = Market(ARMED)
    _run(cfg, m, tmp_path)
    _daily_runs(cfg, m, tmp_path, ARMED + 21)
    second = [t for t in _rows(tmp_path, "picks_trades") if t["fill_date"] == IDX[ARMED + 21].date().isoformat()]
    kinds = {t["kind"] for t in second}
    assert kinds <= {"MOM_IN", "MOM_OUT", "MOM_ADJ", "MOM_LIST_OUT"}
    sides = [t["side"] for t in second if t["side"] != "건너뜀"]
    assert "매수" not in sides or sides.index("매수") > max((i for i, x in enumerate(sides) if x == "매도"), default=-1)
    conn = vbstore.connect(tmp_path / "vb.db")
    book = vbstore.load_picks_book(conn, NAME)
    conn.close()
    px = {t: float(m.frames[t]["close"].iloc[ARMED + 21]) for t in book["positions"]}
    values = [p["qty"] * px[t] for t, p in book["positions"].items()]
    assert len(values) == 10 and max(values) / min(values) < 1.15  # 같은 비중(정수 주식 오차)


def test_catch_up_skipped_days_equals_daily_runs(cfg, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    m1 = Market(ARMED)
    _run(cfg, m1, a)
    _daily_runs(cfg, m1, a, ARMED + 50)
    m2 = Market(ARMED)
    _run(cfg, m2, b)
    m2.upto = ARMED + 50  # 그사이 실행 없음 — 신호일 둘을 건너뛰고 한 번에 따라잡는다
    r = _run(cfg, m2, b)
    assert not r["errors"], r["errors"]
    assert _rows(a, "picks_daily") == _rows(b, "picks_daily")
    assert _rows(a, "picks_trades") == _rows(b, "picks_trades")


def test_last_day_with_empty_closes_is_deferred(cfg, tmp_path):
    m = Market(ARMED)
    _run(cfg, m, tmp_path)
    m.upto = ARMED + 1
    clean = {t: df.copy() for t, df in m.frames.items()}
    m.frames["S03"].iloc[ARMED + 1, 0] = np.nan
    r = _run(cfg, m, tmp_path)
    assert any(NAME in w and "다음 실행에서 처리" in w for w in r["warnings"])
    assert _rows(tmp_path, "picks_trades") == []
    m.frames = clean
    m.upto = ARMED + 2
    _run(cfg, m, tmp_path)
    assert {t["fill_date"] for t in _rows(tmp_path, "picks_trades")} == {IDX[ARMED + 1].date().isoformat()}


def test_catch_up_uses_saved_universe(cfg, tmp_path):
    m = Market(ARMED)
    _run(cfg, m, tmp_path)
    _daily_runs(cfg, m, tmp_path, ARMED + 19)
    conn = vbstore.connect(tmp_path / "vb.db")
    try:  # 두 번째 신호일(ARMED+20) 명단에는 S00~S04만 있었다고 남겨 둔다(그날 실행은 없었음)
        vbstore.save_universe(conn, IDX[ARMED + 20].date().isoformat(), [f"S{i:02d}" for i in range(5)])
    finally:
        conn.close()
    m.upto = ARMED + 21
    _run(cfg, m, tmp_path)
    bought = {t["ticker"] for t in _rows(tmp_path, "picks_trades")
              if t["signal_date"] == IDX[ARMED + 20].date().isoformat() and t["side"] == "매수"}
    assert bought and bought <= {f"S{i:02d}" for i in range(5)}


def test_fixed_start_without_db_state_refuses_to_restart(cfg, tmp_path):
    cfg["virtual_books"]["accel"]["start_date"] = IDX[205].date().isoformat()
    r = _run(cfg, Market(ARMED), tmp_path)
    assert any(NAME in e and "새로 시작하지 않습니다" in e for e in r["errors"])
