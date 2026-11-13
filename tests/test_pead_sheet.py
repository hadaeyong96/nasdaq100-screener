"""전략공장 시트 실적급등 전용(2026-11-13) — 줄 만들기·탭 쓰기·예약 실행 연결·웅덩이/QQQM 끔. 가짜 시트, 네트워크 없음."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from core import pead_sheet as ps
from core import pead_timing as pt
from data import pead_sheet
from engine import pead_watch as ew
from store import pead_watch as st
from store import virtual_books as vbstore

# ── 가짜 시트 ───────────────────────────────────────────────────────────────


class FakeWs:
    def __init__(self, title):
        self.title = title
        self.grid: list[list] = []
        self.row_count = 400
        self.frozen = None
        self.formats = []

    def get_all_values(self):
        return [list(r) for r in self.grid]

    def batch_update(self, updates, value_input_option=None):
        for u in updates:
            m = re.fullmatch(r"A(\d+)", u["range"])
            r = int(m.group(1))
            while len(self.grid) < r:
                self.grid.append([])
            self.grid[r - 1] = [str(x) for x in u["values"][0]]

    def resize(self, rows):
        self.row_count = rows

    def freeze(self, rows):
        self.frozen = rows

    def format(self, rng, fmt):
        self.formats.append((rng, fmt))


class FakeSpreadsheet:
    def __init__(self, titles=()):
        self.tabs = {t: FakeWs(t) for t in titles}
        self.added, self.deleted = [], []

    def worksheets(self):
        return list(self.tabs.values())

    def worksheet(self, title):
        if title not in self.tabs:
            raise LookupError(title)  # gspread.WorksheetNotFound 대신
        return self.tabs[title]

    def add_worksheet(self, title, rows, cols):
        self.added.append(title)
        self.tabs[title] = FakeWs(title)
        return self.tabs[title]

    def del_worksheet(self, ws):
        self.deleted.append(ws.title)
        self.tabs.pop(ws.title)


class FakeClient:
    def __init__(self, sp):
        self.sp = sp
        self.opened = []

    def open_by_key(self, key):
        self.opened.append(key)
        return self.sp


# ── 「일별 성적」 ────────────────────────────────────────────────────────────


def _pf(date, value, cash, q100, vs, fx):
    return {"date": date, "value_usd": value, "cash_usd": cash, "stock_usd": value - cash, "qqq100_usd": q100,
            "vs_qqq_pct": vs, "n_pos": 0, "fx": fx, "value_krw": value * fx, "qqq100_krw": q100 * fx}


def test_daily_rows_holdings_trades_and_realized():
    pf = [_pf("2026-11-11", 1000.0, 1000.0, 1000.0, 0.0, 1300.0), _pf("2026-11-12", 1010.0, 900.0, 1001.0, 0.899, 1310.0),
          _pf("2026-11-13", 1020.0, 1020.0, 1002.0, 1.796, 1320.0)]
    ev = [{"date": "2026-11-12", "ticker": "AAA", "d0": "2026-11-11", "action": "buy", "shares": 2, "px": 50.0, "amount_usd": 100.07, "note": ""},
          {"date": "2026-11-12", "ticker": "BBB", "d0": "2026-11-11", "action": "skip", "shares": 0, "px": 9.0, "amount_usd": 5.0, "note": "최소 금액 미만"},
          {"date": "2026-11-13", "ticker": "AAA", "d0": "2026-11-11", "action": "sell", "shares": 2, "px": 60.0, "amount_usd": 119.9, "note": "손익 +20달러"}]
    rows = ps.daily_rows(pf, ev)
    assert [r[0] for r in rows] == ["2026-11-11", "2026-11-12", "2026-11-13"]
    assert rows[0] == ["2026-11-11", 1300000, 1300000, 0.0, "100.0%", 0, "없음", "", 0]
    d2 = rows[1]
    assert d2[1] == round(1010 * 1310) and d2[3] == 0.9 and d2[4] == "89.1%" and d2[5] == 1 and d2[6] == "AAA"
    assert d2[7] == "매수 AAA 2주 $50.00 · 건너뜀 BBB(최소 금액 미만)"
    d3 = rows[2]
    assert d3[5] == 0 and d3[6] == "없음" and d3[7].startswith("매도 AAA 2주 $60.00")
    assert d3[8] == round(119.9 * 1320 - 100.07 * 1310)  # 받은 돈 × 매도일 환율 − 들어간 돈 × 매수일 환율
    assert len(d3) == len(ps.DAILY_HEADER)


def test_daily_rows_realized_resets_each_year():
    pf = [_pf("2026-12-30", 1000.0, 1000.0, 1000.0, 0.0, 1300.0), _pf("2026-12-31", 1000.0, 1000.0, 1000.0, 0.0, 1300.0),
          _pf("2027-01-04", 1000.0, 1000.0, 1000.0, 0.0, 1300.0)]
    ev = [{"date": "2026-12-30", "ticker": "A", "d0": "x", "action": "buy", "shares": 1, "px": 10.0, "amount_usd": 10.0, "note": ""},
          {"date": "2026-12-31", "ticker": "A", "d0": "x", "action": "sell", "shares": 1, "px": 11.0, "amount_usd": 11.0, "note": ""}]
    rows = ps.daily_rows(pf, ev)
    assert rows[1][8] == 1300 and rows[2][8] == 0


# ── 「매매이력」 ────────────────────────────────────────────────────────────

DAYS = pd.bdate_range("2026-01-01", "2027-12-31")


def _al(base, slope):
    c = base * (1 + slope * np.arange(len(DAYS)))
    return pt.align(pd.DataFrame({"open": c, "close": c}, index=DAYS), DAYS)


def _trade(tk, d0, status, buy_day, **kw):
    ib = DAYS.get_loc(pd.Timestamp(buy_day))
    return {"ledger": "main", "ticker": tk, "d0": d0, "label": "나스닥100", "status": status, "buy_day": buy_day,
            "plan_day": DAYS[ib + 90].date().isoformat(), "buy_px": None, "q_buy": None, "sell_day": None, "sell_px": None,
            "q_sell": None, "fx_buy": None, "fx_sell": None, "skip": None, **kw}


def test_trade_rows_statuses_marks_and_tags():
    al = {"A": _al(100, 0.002), "QQQ": _al(500, 0.001)}
    as_of = pd.Timestamp("2026-08-03")
    ib = DAYS.get_loc(pd.Timestamp("2026-04-01"))
    trades = [
        _trade("A", "2026-03-31", "open", "2026-04-01", buy_px=float(al["A"].open[ib]), q_buy=float(al["QQQ"].open[ib])),
        _trade("C", "2026-01-05", "closed", "2026-01-06", buy_px=10.0, q_buy=400.0, sell_day="2026-05-15", sell_px=12.0, q_sell=420.0),
        _trade("P", "2026-07-31", "pending", "2026-08-04"),
    ]
    sigs = [{"ticker": "A", "d0": "2026-03-31", "ret": 0.12, "excess": 0.10, "vol_ratio": 3.04, "tag_sma200_above": 1,
             "tag_sma200_dist": 8.2, "tag_hi52": -3.1, "tag_close_pos": 0.83, "tag_n202_d0": 12, "tag_q_before_close": 0,
             "tag_lm_pos": 1.5, "tag_lm_neg": 0.7, "tag_errors": None},
            {"ticker": "C", "d0": "2026-01-05", "ret": 0.2, "excess": 0.15, "vol_ratio": 2.5}]
    rows = ps.trade_rows(trades, sigs, al, al["QQQ"], DAYS, as_of, {"A": "에이"})
    assert [r[1] for r in rows] == ["C", "A", "P"] and all(len(r) == len(ps.TRADE_HEADER) for r in rows)
    c, a, p = rows
    assert c[17] == "끝" and c[11] == "2026-05-15" and c[12] == 12.0 and c[13] == "+20.00%" and c[14] == "+5.00%" and c[15] == 15.0
    assert c[18] == "기록 없음"
    assert a[2] == "에이" and a[4] == "+12.00%" and a[5] == 10.0 and a[6] == 3.0 and a[17] == "보유 중" and a[11] == "" and a[12] == ""
    ia = DAYS.get_loc(as_of)
    want = (al["A"].close[ia] / al["A"].open[ib] - 1) - (al["QQQ"].close[ia] / al["QQQ"].open[ib] - 1)
    assert a[15] == round(want * 100, 2)  # 보유 중 = 기준일 종가
    i60 = ib + 60
    want60 = (al["A"].close[i60] / al["A"].open[ib] - 1) - (al["QQQ"].close[i60] / al["QQQ"].open[ib] - 1)
    assert a[9] == DAYS[i60].date().isoformat() and a[16] == round(want60 * 100, 2)
    assert a[18] == "200일선 위(+8.2%) · 52주 고점 대비 -3.1% · 종가 위치 0.83 · 같은 날 발표 12곳 · 10-Q 동시 제출 아니오 · 어조 +0.80(긍정−부정 %)"
    assert p[17] == "매수 대기" and p[8] == "" and p[13] == "" and p[16] == ""


def test_trade_rows_no_60day_value_before_day_60():
    al = {"A": _al(100, 0.002), "QQQ": _al(500, 0.001)}
    ib = DAYS.get_loc(pd.Timestamp("2026-04-01"))
    t = _trade("A", "2026-03-31", "open", "2026-04-01", buy_px=100.0, q_buy=500.0)
    row = ps.trade_rows([t], [], al, al["QQQ"], DAYS, DAYS[ib + 59], {})[0]
    assert row[16] == "" and row[15] != ""


def test_tag_summary_blank_reason():
    assert ps.tag_summary({"tag_errors": "보도자료: 단어 0개"}) == "기록 없음(보도자료: 단어 0개)"
    assert ps.tag_summary({"tag_hi52": -1.0, "tag_errors": "8-K 목록 실패"}) == "52주 고점 대비 -1.0% · 빈칸: 8-K 목록 실패"


# ── 탭 쓰기 ──────────────────────────────────────────────────────────────────


def test_write_all_creates_two_tabs_and_upserts_by_key():
    sp = FakeSpreadsheet(["다른 탭"])
    cl = FakeClient(sp)
    trade = ["2026-11-11", "CRWV", "코어위브", "나스닥100"] + [""] * 3 + ["2026-11-12"] + [""] * 9 + ["보유 중", "기록 없음"]
    n = pead_sheet.write_all(cl, "sid", [["2026-11-11", 1, 2, 0.0, "100.0%", 0, "없음", "", 0]], [trade])
    assert n == {"일별 성적": 1, "매매이력": 1} and sp.added == ["일별 성적", "매매이력"]
    d, t = sp.tabs["일별 성적"], sp.tabs["매매이력"]
    assert d.grid[0] == [ps.TITLE] and d.grid[1] == ps.DAILY_HEADER and t.grid[1] == ps.TRADE_HEADER
    assert d.frozen == 2 and t.formats and sp.tabs["다른 탭"].grid == []
    # 다음 실행: 같은 티커+매수일 줄은 갱신, 새 날짜는 끝에
    closed = list(trade)
    closed[17] = "끝"
    pead_sheet.write_all(cl, "sid", [["2026-11-11", 1, 2, 0.0, "100.0%", 0, "없음", "", 0],
                                     ["2026-11-12", 3, 4, 0.5, "90.0%", 1, "CRWV", "", 0]], [closed])
    assert [r[0] for r in d.grid[2:]] == ["2026-11-11", "2026-11-12"]
    assert len(t.grid) == 3 and t.grid[2][17] == "끝"
    assert sp.added == ["일별 성적", "매매이력"]  # 두 번째엔 만들지 않음


def test_open_tab_refuses_other_tabs_including_retired():
    sp = FakeSpreadsheet()
    for tab in pead_sheet.RETIRED_TABS:
        with pytest.raises(pead_sheet.WrongTabError):
            pead_sheet.open_tab(sp, tab)
    assert sp.added == []


# ── 예약 실행 연결 ────────────────────────────────────────────────────────────


def _engine_src():
    from tests.test_pead_watch_engine import _src

    return _src()


def _actions(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GOOGLE_SHEETS_ID", "fake-sheet")


def test_schedule_run_writes_sheet_from_db(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0

    _actions(monkeypatch)
    sp = FakeSpreadsheet()
    monkeypatch.setattr(ew, "_sheet_client", lambda: FakeClient(sp))
    db = tmp_path / "vb.db"
    r1 = ew.run(CFG, D0, db_path=db, src=_engine_src(), today_kst="2026-10-22")
    assert r1["sheet"]["status"] == "ok"
    assert sp.tabs["매매이력"].grid[2][1] == "A" and sp.tabs["매매이력"].grid[2][17] == "매수 대기"
    r2 = ew.run(CFG, pd.Timestamp("2026-10-27"), db_path=db, src=_engine_src(), today_kst="2026-10-28")
    assert r2["sheet"]["status"] == "ok"
    t = sp.tabs["매매이력"].grid
    assert len(t) == 3 and t[2][17] == "보유 중" and t[2][13] != ""  # 같은 줄 갱신
    conn = vbstore.connect(db)
    n_pf = len(st.load_portfolio(conn))
    conn.close()
    assert len(sp.tabs["일별 성적"].grid) - 2 == n_pf


def test_sheet_failure_keeps_ledger_and_next_run_fills(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0

    _actions(monkeypatch)
    db = tmp_path / "vb.db"
    r1 = ew.run(CFG, D0, db_path=db, src=_engine_src(), today_kst="2026-10-22")  # conftest가 _sheet_client를 막음 → 실패
    assert r1["status"] == "ok" and r1["sheet"]["status"] == "fail"
    assert any("전략공장 시트 기록 실패" in w for w in r1["warnings"])
    conn = vbstore.connect(db)
    assert st.get_meta(conn, st.META_LAST) == "2026-10-21" and st.load_trades(conn)  # 장부는 저장됨
    conn.close()
    sp = FakeSpreadsheet()
    monkeypatch.setattr(ew, "_sheet_client", lambda: FakeClient(sp))
    r2 = ew.run(CFG, pd.Timestamp("2026-10-23"), db_path=db, src=_engine_src(), today_kst="2026-10-24")
    days = [r[0] for r in sp.tabs["일별 성적"].grid[2:]]
    assert r2["sheet"]["status"] == "ok" and days[0] == "2026-10-21"  # 빠진 첫날부터 채움


def test_no_sheet_write_without_schedule(tmp_path, monkeypatch):
    from tests.test_pead_watch_engine import CFG, D0

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setattr(ew, "_sheet_client", lambda: (_ for _ in ()).throw(AssertionError("시트 접속")))
    res = ew.run(CFG, D0, db_path=tmp_path / "vb.db", src=_engine_src())
    assert res["status"] == "preview" and "sheet" not in res


# ── 웅덩이·QQQM·12월 공제 끔 (config.yaml 그대로) ──────────────────────────────


def test_config_turns_off_puddle_qqqm_and_dec_tax(cfg):
    assert cfg["puddle_v32"]["enabled"] is False and cfg["puddle_rebound_v2"]["enabled"] is False
    assert cfg["holdings"]["enabled"] is False and cfg["dec_tax"]["enabled"] is False
    assert cfg["pead_watch"]["enabled"] is True


def test_daily_skips_puddle_books_and_sheet_writes(cfg, monkeypatch):
    from engine import daily
    from engine import puddle_rebound as epr
    from engine import puddle_v31 as e31

    def boom(*a, **k):
        raise AssertionError("웅덩이 장부가 돌면 안 됨")

    monkeypatch.setattr(e31, "run", boom)
    monkeypatch.setattr(epr, "run", boom)
    summary = {"as_of": pd.Timestamp("2026-11-12"), "letter_check": {}}
    daily._attach_puddle_v31(summary, cfg)
    daily._attach_puddle_rebound(summary, cfg)
    assert "puddle_v31" not in summary and "puddle_rebound" not in summary
    assert daily._puddle_on(cfg) is False


def test_daily_does_not_read_qqqm_or_build_dec_tax(cfg):
    from engine import daily

    def boom(*a, **k):
        raise AssertionError("읽으면 안 됨")

    assert daily._read_holdings(cfg, read_fn=boom) is None
    summary = {"as_of": pd.Timestamp("2026-12-01"), "fx_rate": 1400.0}
    daily._attach_dec_tax(summary, cfg, fills_fn=boom, qqqm_fn=boom, days_fn=boom)
    assert "dec_tax" not in summary


# ── 1회 정리 스크립트 · sheet_setup ───────────────────────────────────────────


def test_tabs_to_delete_and_backup_check(tmp_path):
    from scripts import sheet_pead_only as spo

    titles = ["QQQM 모으기", "웅덩이 v3.2 (가상·시총)", "웅덩이 반등 확인 v2 (가상·시총)", "일별 성적", "매매이력", "메모"]
    doomed = spo.tabs_to_delete(titles)
    assert doomed == titles[:3]
    (tmp_path / "01_QQQM 모으기.csv").write_text("x", encoding="utf-8")
    (tmp_path / "02_웅덩이 v3.2 (가상·시총).csv").write_text("x", encoding="utf-8")
    assert spo.backed_up(doomed, tmp_path) == ["웅덩이 반등 확인 v2 (가상·시총)"]
    assert spo.backed_up(doomed, tmp_path / "없음") == doomed


def test_sheet_setup_only_manages_pead_tabs():
    from scripts import sheet_setup as ss

    sp = FakeSpreadsheet(["QQQM 모으기", "매매이력"])
    sp.tabs["매매이력"].grid = [[ps.TITLE], ps.TRADE_HEADER]
    plan = ss.pead_setup(sp, apply=False)
    assert plan == {"create": ["일별 성적"], "fix": [], "others": ["QQQM 모으기"]} and sp.added == []
    ss.pead_setup(sp, apply=True)
    assert sp.added == ["일별 성적"] and sp.tabs["일별 성적"].grid[1] == ps.DAILY_HEADER
    assert sp.tabs["QQQM 모으기"].grid == [] and sp.deleted == []
    assert ss.pead_setup(sp, apply=True) == {"create": [], "fix": [], "others": ["QQQM 모으기"]}
