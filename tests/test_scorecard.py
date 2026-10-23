"""factory/scorecard.py·scorecard_run.py 테스트 (네트워크 없음, 가짜 장부 데이터)."""

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factory import scorecard as sc
from factory import scorecard_run as run

STRATS = [sc.Strategy(1, "QQQ 그냥 보유"), sc.Strategy(2, "QLD 그냥 보유"), sc.Strategy(3, "가짜 전략")]


def _book(days, values):
    v = pd.Series(values, index=days, dtype=float)
    return pd.DataFrame({"value": v, "value_after_tax_liq": v})


def _qqq(n=400, end="2026-10-30"):
    days = pd.bdate_range(end=end, periods=n)
    return pd.Series(np.linspace(300, 500, n), index=days)


def test_metrics_on_fake_ledger():
    days = pd.bdate_range("2026-08-03", "2026-10-30")
    m = days.to_period("M")
    q = np.where(m == pd.Period("2026-08"), 100.0, np.where(m == pd.Period("2026-09"), 110.0, 121.0))
    q[0] = 100.0
    s = q.copy()
    s[m == pd.Period("2026-09")] = 105.0   # 9월 QQQ +10% vs 전략 +5%
    s[m == pd.Period("2026-10")] = 126.0   # 10월 QQQ +10% vs 전략 +20%
    s[5] = 90.0                            # 8월 중 낙폭 −10%
    daily = {"QQQ 그냥 보유": _book(days, q), "가짜 전략": _book(days, s)}
    trades = {"가짜 전략": pd.DataFrame({"fill_date": ["2026-09-01", "2026-10-05", "2026-10-06"], "commission": [1.0, 2.0, 3.0]})}
    card = sc.build(STRATS, daily, trades, _qqq(), pd.Period("2026-10"))
    r = card["rows"][2]
    assert r["month_vs_qqq_pp"] == pytest.approx((126 / 105 - 1.1) * 100)
    assert r["cum_vs_qqq_pp"] == pytest.approx((1.26 - 1.21) * 100)
    assert (r["months_won"], r["months_total"]) == (1, 3)  # 8월 동률(이긴 달 아님), 9월 짐, 10월 이김
    assert r["mdd_pct"] == pytest.approx(-10.0)
    assert (r["trades_month"], r["cost_month"], r["trades_total"], r["cost_total"]) == (2, 5.0, 3, 6.0)
    assert card["rows"][1]["status"] == "기록 없음"
    assert card["rows"][0]["cum_vs_qqq_pp"] == pytest.approx(0.0)
    assert sum(card["regime_ratio"].values()) == pytest.approx(1.0)
    for v in r["regimes"].values():
        assert v["inconclusive"] == (v["days"] < 60)
    lines = sc.briefing_lines(card)
    assert lines[2].startswith("3. 가짜 전략 · 누적 QQQ 대비 +5.0%p · 이긴 달 1/3")
    assert all(ord(ch) < 0x1F000 for ln in lines for ch in ln)  # 이모지 없음


def _row(**kw):
    base = {"no": 5, "status": "ok", "record_months": 6, "cum_vs_qqq_pp": -5.0, "ann_vs_qqq_pp": 0.0,
            "months_won": 0, "months_total": 6, "mdd_pct": -20.0, "qqq_mdd_pct": -20.0}
    return {**base, **kw}


def test_early_stop_boundaries():
    assert sc.proposal(_row(), quarter_end=True) == "중단 제안"
    assert sc.proposal(_row(cum_vs_qqq_pp=-4.99), quarter_end=True) is None
    assert sc.proposal(_row(record_months=5), quarter_end=True) is None
    assert sc.proposal(_row(), quarter_end=False) is None          # 분기 마지막 달 성적표에서만
    assert sc.proposal(_row(no=1), quarter_end=True) is None       # 기준선 제외
    assert sc.proposal(_row(no=2), quarter_end=True) is None


def test_twelve_month_review_boundaries():
    ok = _row(record_months=12, cum_vs_qqq_pp=3.0, ann_vs_qqq_pp=2.0, months_won=6, months_total=10,
              mdd_pct=-30.0, qqq_mdd_pct=-20.0)
    assert sc.proposal(ok, quarter_end=False) == "실제 돈 일부 검토"
    assert sc.proposal({**ok, "ann_vs_qqq_pp": 1.99}, quarter_end=False) is None
    assert sc.proposal({**ok, "months_won": 5}, quarter_end=False) is None
    assert sc.proposal({**ok, "mdd_pct": -30.01}, quarter_end=False) is None
    assert sc.proposal({**ok, "record_months": 11}, quarter_end=False) is None
    assert sc.proposal({**ok, "no": 2}, quarter_end=False) == "실제 돈 일부 검토"  # 2번도 검토 제안은 받을 수 있다
    assert sc.proposal({**ok, "no": 1}, quarter_end=False) is None


def test_record_months():
    assert sc._months_between(pd.Timestamp("2026-10-14"), pd.Timestamp("2027-04-13")) == 5
    assert sc._months_between(pd.Timestamp("2026-10-14"), pd.Timestamp("2027-04-14")) == 6


def test_first_trading_day_weekend_and_holiday():
    from data.market_calendar import trading_days_between

    td = trading_days_between(date(2026, 10, 1), date(2027, 2, 10))
    assert run.scorecard_month(date(2026, 11, 2), td) == pd.Period("2026-10")   # 11/1 일요일 → 11/2 월
    assert run.scorecard_month(date(2026, 11, 1), td) is None
    assert run.scorecard_month(date(2026, 11, 3), td) is None
    assert run.scorecard_month(date(2027, 1, 1), td) is None                     # 새해 휴장
    first_jan = td[(td.year == 2027) & (td.month == 1)].min()
    assert run.scorecard_month(first_jan.date(), td) == pd.Period("2026-12")
    assert run.scorecard_month(date(2027, 2, 1), td) == pd.Period("2027-01")


def _make_db(path, start="2026-10-14"):
    from store import virtual_books as vbstore

    conn = vbstore.connect(path)
    vbstore._ensure_picks_schema(conn)
    days = pd.bdate_range(start, "2026-10-30")
    for name, mult in (("QQQ 그냥 보유", 1.0), ("TQQQ 그냥 보유", 1.01)):
        for k, d in enumerate(days):
            v = 74609.6 * (1 + 0.001 * k) * (mult ** k)
            conn.execute("INSERT INTO daily (book, date, value, value_after_tax_liq) VALUES (?, ?, ?, ?)",
                         (name, d.date().isoformat(), v, v * 0.999))
    for k, d in enumerate(days[2:]):
        conn.execute("INSERT INTO picks_daily (book, date, value, value_after_tax_liq) VALUES (?, ?, ?, ?)",
                     ("한 달 가속 따라하기", d.date().isoformat(), 74663.0 + k, 74663.0 + k))
    conn.execute("INSERT INTO trades (book, fill_date, side, ticker, commission) VALUES ('TQQQ 그냥 보유', '2026-10-14', '매수', 'TQQQ', 52.14)")
    conn.commit()
    conn.close()


def test_make_on_first_trading_day_reads_db_readonly(tmp_path):
    db = tmp_path / "vb.db"
    _make_db(db)
    before = db.read_bytes()
    cfg = {"virtual_books": {"books": [{"name": "QQQ 그냥 보유"}, {"name": "QLD 그냥 보유"}, {"name": "TQQQ 그냥 보유"}],
                             "accel": {"name": "한 달 가속 따라하기"}}}
    card = run.make(cfg, date(2026, 11, 2), _qqq(), db_path=db)
    assert card["month"] == "2026-10"
    names = [r["name"] for r in card["rows"]]
    assert names == ["QQQ 그냥 보유", "QLD 그냥 보유", "TQQQ 그냥 보유", "한 달 가속 따라하기"]
    tq = card["rows"][2]
    assert tq["short_record"] and "기록 짧음" in tq["notes"] and tq["trades_total"] == 1
    assert card["rows"][3]["start"] == "2026-10-16"
    assert "기록 짧음" in card["lines"][2]
    assert db.read_bytes() == before
    assert run.make(cfg, date(2026, 11, 3), _qqq(), db_path=db) is None


class _FakeSheet:
    def __init__(self, available=True):
        self.available, self.rows = available, []

    def months(self):
        return {r[0] for r in self.rows}

    def append(self, rows):
        self.rows += rows
        return len(rows)


def test_attach_failure_keeps_briefing(monkeypatch):
    summary = {"report_path": "orig.html"}
    calls, notes = [], []

    def boom():
        raise RuntimeError("QQQ 못 받음")

    run.attach_scorecard(summary, {}, date(2026, 11, 2), boom, lambda s: calls.append(s) or "new.html",
                         sheet=_FakeSheet(), notify=lambda e, s: notes.append((e, s)))
    assert summary == {"report_path": "orig.html"} and calls == []
    assert notes == [("error", {"stage": "월간 성적표 계산"})]  # 어느 단계인지만, 오류 상세 없음

    def bad_render(s):
        raise ValueError("그리기 실패")

    card = {"month": "2026-10", "lines": ["x"], "markdown": "m", "rows": [], "regime_ratio": {}, "proposals": []}
    monkeypatch.setattr(run, "make", lambda *a, **k: card)
    run.attach_scorecard(summary, {}, date(2026, 11, 2), lambda: None, bad_render, sheet=_FakeSheet(),
                         notify=lambda e, s: None)
    assert summary == {"report_path": "orig.html"}


def _cfg():
    return {"virtual_books": {"books": [{"name": "QQQ 그냥 보유"}, {"name": "QLD 그냥 보유"}, {"name": "TQQQ 그냥 보유"}],
                              "accel": {"name": "한 달 가속 따라하기"}}}


def test_catch_up_after_failed_first_day_inserts_once(tmp_path, monkeypatch):
    db = tmp_path / "vb.db"
    _make_db(db)
    monkeypatch.setattr(run, "OUT_DIR", tmp_path / "out")
    orig_make = run.make
    monkeypatch.setattr(run, "make", lambda cfg, d, q, **k: orig_make(cfg, d, q, db_path=db, **k))
    report = tmp_path / "report_live.html"
    sheet, notes = _FakeSheet(), []

    def render(s):
        report.write_text("성적표 있음" if s.get("scorecard") else "성적표 없음", encoding="utf-8")
        return report

    def go(day, qqq):
        report.write_text("성적표 없음", encoding="utf-8")
        s = {"report_path": report}
        run.attach_scorecard(s, _cfg(), day, qqq, render, sheet=sheet, notify=lambda e, x: notes.append(e))
        return s

    def qqq_fail():
        raise RuntimeError("첫 거래일 실패")

    s1 = go(date(2026, 11, 2), qqq_fail)                       # 첫 거래일 실패 → 시트·접기 없음
    assert sheet.rows == [] and "scorecard" not in s1
    s2 = go(date(2026, 11, 3), _qqq)                           # 다음 실행에서 넣는다
    assert {r[0] for r in sheet.rows} == {"2026-10"} and len(sheet.rows) == 4 and s2["scorecard"]["month"] == "2026-10"
    assert Path(s2["group_report_path"]).read_text(encoding="utf-8") == "성적표 없음"  # 단체방 판에는 성적표 없음
    assert Path(s2["report_path"]).read_text(encoding="utf-8") == "성적표 있음"
    s3 = go(date(2026, 11, 4), _qqq)                           # 같은 달 두 번 없음
    assert len(sheet.rows) == 4 and "scorecard" not in s3 and "group_report_path" not in s3
    assert notes.count("scorecard_done") == 1 and notes.count("error") == 1


def test_without_sheet_only_first_trading_day():
    from data.market_calendar import trading_days_between

    td = trading_days_between(date(2026, 10, 1), date(2026, 12, 10))
    assert run.scorecard_month(date(2026, 11, 3), td, done=None) is None
    assert run.scorecard_month(date(2026, 11, 3), td, done=set()) == pd.Period("2026-10")
    assert run.scorecard_month(date(2026, 11, 1), td, done=set()) is None      # 그달 첫 거래일 전(일요일)
    assert run.scorecard_month(date(2026, 11, 3), td, done={"2026-10"}) is None


def test_group_send_uses_group_edition_without_personal_fallback():
    """engine/daily.py가 단체방에 group_report_path(단체방판 — 성적표 없음)만 보내고, 없을 때 개인판으로 대신하지 않는지(정적 확인)."""
    src = (Path(__file__).resolve().parents[1] / "engine" / "daily.py").read_text(encoding="utf-8")
    i = src.index("telegram.send_group_briefing(")
    call = src[i:i + 300]
    assert 'summary.get("group_report_path")' in call and 'or summary.get("report_path")' not in call


def test_scorecard_sheet_tab_is_append_only():
    from data import virtual_books_sheet as vbsheet

    assert set(vbsheet.ALLOWED_TABS) - set(vbsheet.LEDGER_TABS) == {"월간성적표"}
    with pytest.raises(vbsheet.VirtualBooksSheetForbiddenError):
        vbsheet.write_tables(object(), "x", {vbsheet.TAB_SCORECARD: [["a"]]})   # 통째로 다시 쓰기 금지
    with pytest.raises(vbsheet.VirtualBooksSheetForbiddenError):
        vbsheet.write_tables(object(), "x", {"계획": [["a"]]})

    class WS:
        title = "월간성적표"

        def __init__(self):
            self.rows = []

        def get_all_values(self):
            return self.rows

        def append_rows(self, rows, value_input_option=None):
            self.rows += rows

        def clear(self):
            raise AssertionError("지우면 안 된다")

    ws = WS()

    class SS:
        def worksheet(self, name):
            assert name == "월간성적표"
            return ws

    class Client:
        def open_by_key(self, k):
            return SS()

    vbsheet.append_scorecard(Client(), "x", [["2026-10", 1]])
    vbsheet.append_scorecard(Client(), "x", [["2026-11", 1]])
    assert ws.rows == [vbsheet.SCORECARD_HEADER, ["2026-10", 1], ["2026-11", 1]]


def test_daily_hook_isolated_from_briefing_send():
    """engine/daily.py 연결점이 try 안에 있고, 그 뒤에 브리핑 발송 코드가 그대로 이어지는지(정적 확인)."""
    src = (Path(__file__).resolve().parents[1] / "engine" / "daily.py").read_text(encoding="utf-8")
    i = src.index("from factory import scorecard_run")
    assert src.rfind("try:", 0, i) > src.rfind("\n", 0, i) - 200
    assert src.count("scorecard_run.attach_scorecard") == 1
    assert src.index("telegram.send_briefing(text, summary") > i


def test_report_renders_scorecard_fold(tmp_path):
    """보고서 템플릿이 성적표 접기를 그 날만 그리는지 (tests/test_report_titles.py의 summary 재사용)."""
    from engine.daily import load_config
    from notify import report_html
    from tests.test_report_titles import _summary

    cfg = load_config()
    summary = _summary(cfg)
    summary["scorecard"] = {"month": "2026-10", "lines": ["1. QQQ 그냥 보유 · 누적 QQQ 대비 +0.0%p · 이긴 달 0/1 · 기록 짧음"]}
    html = report_html.render_report(summary, cfg, tmp_path).read_text(encoding="utf-8")
    assert "▶ 이번 달 가상 성적표" in html and "기록 짧음" in html
    group = report_html.render_report(summary, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    assert "월간 성적표" not in group and "기록 짧음" not in group  # 단체방판에는 없음
    summary.pop("scorecard")
    html2 = report_html.render_report(summary, cfg, tmp_path).read_text(encoding="utf-8")
    assert "▶ 이번 달 가상 성적표" not in html2
