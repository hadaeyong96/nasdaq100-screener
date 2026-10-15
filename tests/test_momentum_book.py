"""8번 장부 「6개월 강세 따라하기」 (작업 5-2) — 네트워크 없이 합성 데이터만 쓴다.

월말 판정(휴장 포함), 126거래일 미만·5달러 미만 제외, 순위, 같은 비중 재조정(남은 종목은 차이만),
명단 이탈 종목은 다음 교체일에 매도, 빈 종가(체결일 종가 없음·맨 끝 날 미루기), 시작(첫 월말 신호 대기),
따라잡기, 전략요약 열.
"""

from __future__ import annotations

import copy
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core import momentum_book as mb
from data import virtual_books_data as vbdata
from engine import picks_books
from engine import virtual_books as vbe
from store import virtual_books as vbstore

NAME = "6개월 강세 따라하기"
P = mb.MomentumParams(name=NAME, commission_pct=0.0)


# ── 월말 판정 (NYSE 달력, 휴장 포함) ─────────────────────────────────────────


def test_month_end_uses_last_nyse_trading_day_with_holidays_and_weekends():
    me = vbdata.month_end_trading_days(2026, 2029)
    assert pd.Timestamp("2026-10-30") in me and pd.Timestamp("2026-10-31") not in me  # 10/31 토요일
    assert pd.Timestamp("2027-05-28") in me and pd.Timestamp("2027-05-31") not in me  # 5/31 메모리얼 데이
    assert pd.Timestamp("2029-03-29") in me and pd.Timestamp("2029-03-30") not in me  # 3/30 성금요일
    assert pd.Timestamp("2026-12-31") in me
    assert len([d for d in me if d.year == 2027]) == 12


# ── 순위 ─────────────────────────────────────────────────────────────────


IDX = pd.bdate_range("2024-01-02", periods=300)


def _series(start_px: float, end_px: float, n: int = 300, idx=IDX) -> pd.Series:
    return pd.Series(np.linspace(start_px, end_px, n), index=idx[:n])


def test_rank_score_is_126d_return_minus_qqq_and_excludes_short_history_and_under_5():
    d = IDX[200]
    qqq = _series(100, 120)
    closes = {
        "AAA": _series(10, 30),  # 강함
        "BBB": _series(10, 11),  # 약함
        "NEW": _series(10, 40).iloc[-150:],  # 126거래일 전 가격 없음(신규 상장)
        "PENNY": _series(2, 4.9),  # 신호일 종가 5달러 미만
        "GONE": _series(10, 30).iloc[:150],  # 신호일 종가 없음
    }
    r = mb.rank(closes, qqq, d, list(closes), P)
    assert r["ok"] and r["window"] == (IDX[200 - 126], d)
    q_ret = qqq[d] / qqq[IDX[74]] - 1
    a = next(x for x in r["ranked"] if x["ticker"] == "AAA")
    assert a["score_pp"] == pytest.approx((closes["AAA"][d] / closes["AAA"][IDX[74]] - 1 - q_ret) * 100)
    assert [x["ticker"] for x in r["ranked"]] == ["AAA", "BBB"]
    assert set(r["excluded"]) == {"NEW", "PENNY", "GONE"}
    assert "5달러 미만" in r["excluded"]["PENNY"] and "126거래일" in r["excluded"]["NEW"]


def test_rank_takes_top_10_only_from_universe():
    closes = {f"T{i:02d}": _series(10, 10 + i) for i in range(15)}
    r = mb.rank(closes, _series(100, 100), IDX[200], [t for t in closes if t != "T14"], P)
    assert [x["ticker"] for x in r["targets"]] == [f"T{i:02d}" for i in range(13, 3, -1)]


def test_rank_without_qqq_history_is_not_ok():
    r = mb.rank({"AAA": _series(10, 20)}, _series(100, 120).iloc[:100], IDX[99], ["AAA"], P)
    assert not r["ok"] and r["targets"] == []


# ── 하루 처리: 시작·재조정·명단 이탈·빈 종가 ──────────────────────────────


def _sig(d, tickers, scores=None):
    return {"ok": True, "targets": [{"ticker": t, "score_pp": (scores or {}).get(t, 10.0 - i), "ret": 0.0, "close": 0.0}
                                    for i, t in enumerate(tickers)]}


def _day(book, d, closes, signal=None, universe=None, year_end=False, params=P):
    return mb.process_day(book, d, closes=closes, dividends={}, splits={}, signal=signal,
                          universe=universe if universe is not None else list(closes), params=params, year_end=year_end)


def _px(n=10, px=100.0, **over):
    out = {f"S{i}": px for i in range(n)}
    out.update(over)
    return out


def test_waiting_book_has_no_rows_until_started():
    b = mb.new_book(NAME, IDX[0])
    b, row, trades = _day(b, IDX[0], _px())
    assert row is None and trades == [] and b["status"] == mb.WAITING and b["last_date"] == IDX[0]


def test_signal_day_holds_cash_then_buys_equal_weights_next_close():
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    names = [f"S{i}" for i in range(10)]
    b, row0, t0 = _day(b, IDX[0], _px(), _sig(IDX[0], names))
    assert t0 == [] and row0["value"] == pytest.approx(100_000) and b["pending"]["signal_date"] == IDX[0]
    b, row1, t1 = _day(b, IDX[1], _px())
    assert [(t["side"], t["ticker"], t["qty"]) for t in t1] == [("매수", n, 100) for n in names]
    assert all(t["fill_date"] == IDX[1] and t["signal_date"] == IDX[0] and "신규 편입" in t["reason"] for t in t1)
    assert b["cash"] == pytest.approx(0.0) and b["rebalances"] == 1 and b["pending"] is None


def test_next_month_trades_only_weight_difference_for_kept_and_sells_dropped():
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    first = [f"S{i}" for i in range(10)]
    b, _, _ = _day(b, IDX[0], _px(), _sig(IDX[0], first))
    b, _, _ = _day(b, IDX[1], _px())
    # 한 달 뒤: S0 두 배(비중 초과), S9 탈락, 새 종목 N0 편입
    px = _px(S0=200.0, N0=50.0)
    second = [f"S{i}" for i in range(9)] + ["N0"]
    b, _, _ = _day(b, IDX[20], px, _sig(IDX[20], second))
    b, row, trades = _day(b, IDX[21], px)
    by = {(t["ticker"], t["side"]): t for t in trades}
    equity = 100 * 200 + 9 * 100 * 100  # 체결 전 총자산 = 110,000
    slot = equity / 10
    assert by[("S9", "매도")]["qty"] == 100 and "상위 10 탈락" in by[("S9", "매도")]["reason"]
    assert by[("S0", "매도")]["qty"] == 100 - int(slot // 200) and by[("S0", "매도")]["kind"] == mb.KIND_ADJ
    assert by[("N0", "매수")]["qty"] == int(slot // 50) and "신규 편입" in by[("N0", "매수")]["reason"]
    for t in ("S1", "S2", "S8"):  # 남은 종목은 차이만 (100주 → 110주)
        assert by[(t, "매수")]["qty"] == int(slot // 100) - 100 and by[(t, "매수")]["kind"] == mb.KIND_ADJ
    assert all(t["side"] != "매도" or t["ticker"] in ("S9", "S0") for t in trades)
    sides = [t["side"] for t in trades]
    assert sides.index("매수") > max(i for i, s in enumerate(sides) if s == "매도")  # 매도 먼저
    assert b["positions"]["N0"]["qty"] * 50 == pytest.approx(slot, rel=0.01)
    assert row["value"] == pytest.approx(equity, rel=1e-9)


def test_off_list_holding_is_kept_until_next_rebalance_then_sold():
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    names = [f"S{i}" for i in range(10)]
    b, _, _ = _day(b, IDX[0], _px(), _sig(IDX[0], names))
    b, _, _ = _day(b, IDX[1], _px())
    universe_wo_s3 = [n for n in names if n != "S3"] + ["N0"]
    for i in range(2, 20):  # 명단에서 빠져도 교체일 전까지는 판 기록이 없다
        b, _, t = _day(b, IDX[i], _px(N0=100.0), universe=universe_wo_s3)
        assert t == [] and "S3" in b["positions"]
    b, _, _ = _day(b, IDX[20], _px(N0=100.0), _sig(IDX[20], [n for n in universe_wo_s3]), universe=universe_wo_s3)
    assert b["pending"]["off_list"] == ["S3"]
    b, _, trades = _day(b, IDX[21], _px(N0=100.0), universe=universe_wo_s3)
    s3 = [t for t in trades if t["ticker"] == "S3"]
    assert len(s3) == 1 and s3[0]["side"] == "매도" and s3[0]["kind"] == mb.KIND_LIST_OUT and "명단 이탈" in s3[0]["reason"]
    assert "S3" not in b["positions"] and "N0" in b["positions"]


def test_fill_day_missing_close_skips_buy_and_carries_sell():
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    b, _, _ = _day(b, IDX[0], _px(), _sig(IDX[0], [f"S{i}" for i in range(10)]))
    b, _, _ = _day(b, IDX[1], _px())
    new = [f"S{i}" for i in range(9)] + ["N0"]
    b, _, _ = _day(b, IDX[20], _px(N0=100.0), _sig(IDX[20], new))
    holed = {k: v for k, v in _px(N0=100.0).items() if k not in ("S9", "N0")}  # 판 종목·살 종목 종가 없음
    b, _, trades = _day(b, IDX[21], holed)
    assert any(t["ticker"] == "N0" and t["side"] == "건너뜀" and "체결일 종가 없음" in t["reason"] for t in trades)
    assert "S9" in b["positions"] and b["carry_sells"][0]["ticker"] == "S9"
    b, _, t2 = _day(b, IDX[22], _px(N0=100.0))
    assert [(t["ticker"], t["side"]) for t in t2] == [("S9", "매도")] and "종가 없어 넘김" in t2[0]["reason"]


def test_costs_dividends_and_year_end_tax_like_other_books():
    params = mb.MomentumParams(name=NAME, commission_pct=0.1)
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    names = [f"S{i}" for i in range(10)]
    b, _, _ = _day(b, IDX[0], _px(), _sig(IDX[0], names), params=params)
    b, _, t1 = _day(b, IDX[1], _px(), params=params)
    assert all(t["commission"] == pytest.approx(t["qty"] * 100 * 0.001) for t in t1)
    b2, _, _ = mb.process_day(b, IDX[2], closes=_px(), dividends={"S0": 1.0}, splits={}, signal=None,
                              universe=names, params=params, year_end=False)
    assert b2["cash"] - b["cash"] == pytest.approx(b["positions"]["S0"]["qty"] * 1.0 * 0.85)
    b2["realized_ytd"] = 1_000.0
    cash_before = b2["cash"]
    b3, row, _ = _day(b2, IDX[3], _px(), year_end=True, params=params)
    assert b3["tax_cum"] == pytest.approx(220.0) and b3["realized_ytd"] >= 0 and row["tax_cum"] == pytest.approx(220.0)
    assert b3["cash"] >= -1e-9 and b3["cash"] != cash_before


def test_split_multiplies_quantity():
    b = mb.start_book(mb.new_book(NAME, IDX[0]), IDX[0], 100_000.0)
    b, _, _ = _day(b, IDX[0], _px(), _sig(IDX[0], [f"S{i}" for i in range(10)]))
    b, _, _ = _day(b, IDX[1], _px())
    b2, row, _ = mb.process_day(b, IDX[2], closes=_px(S0=25.0), dividends={}, splits={"S0": 4.0}, signal=None,
                                universe=list(_px()), params=P, year_end=False)
    assert b2["positions"]["S0"]["qty"] == 400 and row["value"] == pytest.approx(100_000.0)


# ── 엔진: 첫 월말 대기·시작·따라잡기·빈 종가 미루기·시트 열 ───────────────────


NOW_ET = datetime(2030, 1, 10, 18, 0, tzinfo=ZoneInfo("America/New_York"))
MONTH_ENDS = set(pd.Series(IDX, index=IDX).groupby([IDX.year, IDX.month]).max())


class Market:
    def __init__(self, upto):
        self.upto = upto
        self.universe = [f"S{i:02d}" for i in range(12)]
        n = len(IDX)
        self.frames = {t: pd.DataFrame({"close": np.linspace(20, 20 + 3 * i, n)}, index=IDX)
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
    vb["momentum"].update({"enabled": True, "start_fx": None})
    return c


def _run(cfg, m, path, client=None, **kw):
    return vbe.run(cfg, db_path=path / "vb.db", sheets_client=client or FakeClient(), now_et=NOW_ET,
                   history_fn=m.history_fn, fx_fn=lambda d: (1400.0, d.isoformat(), None),
                   year_end_fn=lambda a, b: set(), next_day_fn=lambda d: None, market_fn=m.market_fn,
                   actions_fn=lambda t, s: {x: pd.DataFrame({"dividend": [], "split": []}) for x in t},
                   month_end_fn=lambda a, b: MONTH_ENDS, **kw)


def _rows(path, table):
    conn = vbstore.connect(path / "vb.db")
    try:
        return vbstore.picks_rows(conn, table, NAME)
    finally:
        conn.close()


def _first_me_after(pos):
    return min(i for i in range(pos, len(IDX)) if IDX[i] in MONTH_ENDS)


def _daily_runs(cfg, m, path, upto, client=None):
    r = None
    while m.upto < upto:
        m.upto += 1
        r = _run(cfg, m, path, client=client)
    return r


def test_engine_waits_for_first_month_end_then_starts_and_fills_next_day(cfg, tmp_path):
    armed = 205
    me = _first_me_after(armed + 1)
    m = Market(armed)
    r = _run(cfg, m, tmp_path)
    assert not r["errors"], r["errors"]
    assert any("첫 월말 신호" in w for w in r["warnings"]) and _rows(tmp_path, "picks_daily") == []
    _daily_runs(cfg, m, tmp_path, me)
    daily = _rows(tmp_path, "picks_daily")
    assert [d["date"] for d in daily] == [IDX[me].date().isoformat()] and _rows(tmp_path, "picks_trades") == []
    client = FakeClient()
    _daily_runs(cfg, m, tmp_path, me + 1, client=client)
    trades = _rows(tmp_path, "picks_trades")
    assert len(trades) == 10 and {t["fill_date"] for t in trades} == {IDX[me + 1].date().isoformat()}
    assert {t["ticker"] for t in trades} == {f"S{i:02d}" for i in range(2, 12)}  # 가장 강한 10개
    summ = client.tabs["전략요약"].values
    head = summ[0]
    row = next(x for x in summ if x and x[0] == NAME)
    assert head[-2:] == ["교체 횟수", "현재 보유 10종목"]
    assert row[head.index("교체 횟수")] == "1" and row[head.index("현재 보유 10종목")].count(",") == 9
    assert row[head.index("완료 거래 수")] == "" and row[head.index("시작일")] == IDX[me].date().isoformat()
    other = next(x for x in summ if x and x[0] == "QQQ 그냥 보유")
    assert other[head.index("교체 횟수")] == "" and any(x and x[0] == "config 고정값(momentum)" for x in summ)


def test_engine_catches_up_skipped_days_same_as_daily(cfg, tmp_path):
    me = _first_me_after(206)
    end = me + 30
    m1 = Market(205)
    _run(cfg, m1, tmp_path / "a")
    _daily_runs(cfg, m1, tmp_path / "a", end)
    m2 = Market(205)
    _run(cfg, m2, tmp_path / "b")
    m2.upto = end  # 그사이 실행 없음 (예약 실패·지연)
    r = _run(cfg, m2, tmp_path / "b")
    assert not r["errors"], r["errors"]
    strip = lambda rows: [{k: v for k, v in x.items()} for x in rows]  # noqa: E731
    assert strip(_rows(tmp_path / "a", "picks_daily")) == strip(_rows(tmp_path / "b", "picks_daily"))
    assert strip(_rows(tmp_path / "a", "picks_trades")) == strip(_rows(tmp_path / "b", "picks_trades"))
    assert len({t["fill_date"] for t in _rows(tmp_path / "b", "picks_trades")}) >= 2  # 두 번 이상 교체


def test_engine_defers_last_day_with_empty_closes(cfg, tmp_path):
    me = _first_me_after(206)
    m = Market(205)
    _run(cfg, m, tmp_path)
    _daily_runs(cfg, m, tmp_path, me)
    m.upto = me + 1
    clean = {t: df.copy() for t, df in m.frames.items()}
    for t in ("S05", "S11"):
        m.frames[t].iloc[me + 1, 0] = np.nan
    r = _run(cfg, m, tmp_path)
    assert any(NAME in w and "다음 실행에서 처리" in w for w in r["warnings"])
    assert _rows(tmp_path, "picks_trades") == []  # 체결일이 미뤄졌다
    m.frames = clean
    m.upto = me + 2
    _run(cfg, m, tmp_path)
    assert {t["fill_date"] for t in _rows(tmp_path, "picks_trades")} == {IDX[me + 1].date().isoformat()}
