"""아침 편지 짧은 판 (2026-10-17) — notify/letter.py, 보고서 두 판, 월간 성적표 시작 전 달, 보여 주는 이름. 네트워크 없음."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

from engine.daily import load_config
from factory import scorecard as sc
from factory import scorecard_run as scr
from notify import letter, report_html
from notify import virtual_books as vbtext
from tests.legacy import legacy_cfg

BASE = pd.Timestamp("2026-10-16")


@pytest.fixture
def cfg():
    c = legacy_cfg(load_config())  # 가상 장부 칸이 있던 편지(새 출발 전) — 새 구조는 tests/test_fresh_start.py
    c["virtual_books"]["penny"]["enabled"] = False  # 1~9번 기준 문구 시험 — 10번은 tests/test_penny_book.py
    return c


def _sum(book, cum, holding, start="2026-10-14", value=100_000.0):
    return {"book": book, "cum_return_pct": cum, "holding": holding, "start": pd.Timestamp(start), "value": value,
            "mdd_pct": -1.0, "switches": 0}


def _extra(name, cum, positions=None, trades=None, start="2026-10-16"):
    return {"name": name, "start": pd.Timestamp(start), "summary": {"cum_return_pct": cum, "value": 75_000.0, "mdd_pct": 0.0},
            "positions": positions or {}, "trades": trades or []}


def _bench():
    return pd.DataFrame({"value": [100.0, 100.5, 101.0]}, index=pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16"]))


def _books(cfg, *, trend_holding="QLD", accel_positions=None, accel_trades=None, waiting=True, book_trades=None):
    sums = [_sum("QQQ 그냥 보유", 1.0, "QQQ"), _sum("QLD 그냥 보유", 2.0, "QLD"), _sum("TQQQ 그냥 보유", 3.0, "TQQQ"),
            _sum("QLD 200일선 전환", 2.0 if trend_holding == "QLD" else 0.0, trend_holding if trend_holding != "QLD" else "QLD"),
            _sum("TQQQ 200일선 전환", 3.0 if trend_holding == "QLD" else 0.0, "TQQQ" if trend_holding == "QLD" else trend_holding)]
    extra = [_extra("추천 종목 따라하기", 0.0), _extra("볼린저·RSI·MACD 따라하기", 0.0),
             _extra("한 달 가속 따라하기", 0.4, accel_positions, accel_trades)]
    wait = {"가치평가 따라하기": pd.Timestamp("2026-10-19")} if waiting else None  # 8번: 2026-10-18 교체
    return letter.letter_books(cfg, BASE, sums, book_trades or {}, extra, _bench(), waiting=wait)


# ── 할 일 ──────────────────────────────────────────────────────────────────


def test_todo_none_switch_and_scorecard_day(cfg):
    assert letter.todo_line({}, cfg) == "오늘 할 일: 없음 — QQQM 적립만 그대로"
    s = {"vbooks": {"switch_today": True, "switch_next_day": pd.Timestamp("2026-10-19")},
         "scorecard": {"month": "2026-10", "lines": []}}
    assert letter.todo_line(s, cfg) == "오늘 할 일: 추세 전략 전환 신호(가상) — 10/19 체결 · 월간 성적표 보기(10월)"
    # 단체방판: 개인 항목(월간 성적표)은 빼고 전환만
    assert letter.todo_line(s, cfg, letter.GROUP) == "오늘 할 일: 추세 전략 전환 신호(가상) — 10/19 체결"
    assert letter.todo_line({"scorecard": {"month": "2026-10"}}, cfg, letter.GROUP) == "오늘 할 일: 없음 — QQQM 적립만 그대로"
    assert letter.december_items({}, cfg) == []  # 12월 공제 자리(다음 작업에서 연결)


# ── 시장 한눈에 ──────────────────────────────────────────────────────────────


def _macro(slug, name, value, label, zone=None):
    return {"slug": slug, "name": name, "value": value, "badge": {"label": label, "zone": zone}}


def test_glance_danger_line_lists_only_red_with_short_names():
    q = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    rows = [_macro("fear-greed", "공포·탐욕 지수", 18, "위험", "극단적 공포"), _macro("dgs10", "미국 10년물 국채금리", 5.10, "위험"),
            _macro("t10y2y", "장단기 금리차", 0.35, "주의"), _macro("fed", "미국 기준금리 (상단)", 4.0, "위험"),
            _macro("fx", "원/달러 환율", 1334.2, "주의")]
    lines = letter.glance_lines(q, rows)
    assert len(lines) == 3
    assert lines[0].startswith("QQQ 하루 +") and "1주 +" in lines[0] and "200일선 대비 +" in lines[0] and lines[0].endswith("상승장")
    assert lines[1] == "공포·탐욕 18(극단적 공포) · 미 10년 금리 5.10% · 원/달러 1,334원"
    assert lines[2] == "위험: 공포 극단 · 10년 금리 5.1% · 기준금리 인상"  # 주의(노랑)는 넣지 않음
    yellow_only = [dict(r, badge={"label": "주의", "zone": None}) for r in rows]
    assert len(letter.glance_lines(q, yellow_only)) == 2  # 위험이 없으면 줄 생략
    assert not any(x.startswith(("위험", "주의")) for x in letter.glance_lines(q, yellow_only))
    assert letter.glance_lines(None, yellow_only)[0].startswith("공포·탐욕")  # QQQ를 못 받으면 1줄만 빠짐


def test_strategy_lines_group_same_status_returns_in_parens_and_six_to_nine(cfg):
    books = _books(cfg)
    assert [b["label"] for b in books] == ["QQQ 전략", "QLD 전략", "TQQQ 전략", "QLD 추세 전략", "TQQQ 추세 전략",
                                           "기술지표 전략", "볼린저 전략", "가치평가 전략", "QQQ 대비 1개월 강세 전략"]
    lines = letter.strategy_lines(books, cfg)
    assert lines == [
        "QQQ·QLD·TQQQ 전략: 그냥 보유 (+1.0% · +2.0% · +3.0%)",
        "QLD·TQQQ 추세 전략: 상승 추세라 QLD·TQQQ 보유 (+2.0% · +3.0%)",
        "기술지표·볼린저·QQQ 대비 1개월 강세 전략: 살 종목이 없어 현금 보관 (0.0% · 0.0% · +0.4%)",
        "가치평가 전략: 10/19 시작",
        "오늘 거래 없음",  # 칸 맨 끝에 한 번만
    ]
    assert not any("%p" in x for x in lines)  # QQQ 대비 %p는 표로 보기 접기에만
    assert letter.strategy_header(books) == "[실험 중인 전략] 가상 1억 · 10/14 시작 · 매수 추천 아님"


def test_status_texts_down_and_positions_and_minus_sign(cfg):
    down = _books(cfg, trend_holding="SGOV")
    for b in down:
        if b["family"] == "trend":
            b["cum_return_pct"] = -5.26
    assert letter.strategy_lines(down, cfg)[1] == "QLD·TQQQ 추세 전략: 하락 추세라 단기 국채로 피신 (−5.3% · −5.3%)"
    held = _books(cfg, accel_positions={"MU": {}, "AMD": {}})
    lines = letter.strategy_lines(held, cfg)
    assert "기술지표·볼린저 전략: 살 종목이 없어 현금 보관 (0.0% · 0.0%)" in lines and "QQQ 대비 1개월 강세 전략: 2종목 보유 (+0.4%)" in lines


def test_virtual_trades_once_at_end_personal_names_tickers_group_count(cfg):
    trades = [{"side": "매수", "ticker": "MU", "fill_date": BASE}, {"side": "매수", "ticker": "AMD", "fill_date": BASE},
              {"side": "건너뜀", "ticker": "X", "fill_date": BASE}, {"side": "매도", "ticker": "OLD", "fill_date": BASE - pd.Timedelta(days=1)}]
    books = _books(cfg, accel_positions={"MU": {}, "AMD": {}}, accel_trades=trades,
                   book_trades={"QLD 200일선 전환": [{"side": "매도", "ticker": "QLD", "fill_date": BASE}]})
    p = letter.strategy_lines(books, cfg)
    assert p[-1] == "오늘 가상 매수: QQQ 대비 1개월 강세 전략 MU, AMD / 오늘 가상 매도: QLD 추세 전략 QLD"
    assert sum("오늘 가상" in x or "오늘 거래" in x for x in p) == 1
    g = letter.strategy_lines(books, cfg, letter.GROUP)
    assert g[-1] == "오늘 가상 거래 3건" and "MU" not in "".join(g) and "2종목 보유" in "".join(g)


def test_zero_only_cells_hidden(cfg):
    books = _books(cfg)
    for b in books:
        if b["family"] == "hold":
            b["vs_qqq_pp"] = 0.0 if b["vs_qqq_pp"] is not None else None
    assert "QQQ 대비 +0.0%p" not in letter.strategy_lines(books, cfg)[0]
    head, rows = letter.table_rows(books, cfg)
    assert "전환 횟수" not in head  # 모두 0인 열은 뺀다
    assert head[0] == "전략" and len(rows) == 9


# ── 보고서 두 판 ─────────────────────────────────────────────────────────────


def _summary(cfg):
    from tests.test_report_titles import _summary as base

    s = base(cfg)
    s["vbooks"] = {"start": "2026-10-14", "benchmark": "QQQ 그냥 보유", "rows": [], "books": _books(cfg),
                   "switch_today": False}
    s["weekly_rs"] = None
    s["scorecard"] = {"month": "2026-10", "lines": ["1. QQQ 전략 · 기록 짧음"]}
    s["qqq_close"] = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    return s


def _folds(html):
    return re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", html)


def test_personal_order_and_screening_switch(cfg, tmp_path):
    s = _summary(cfg)
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    body = html[html.index("<header>"):]
    i = [body.index(x) for x in ("<header>", "오늘 할 일:", "[1 내 돈 · QQQM]", "[2 시장 한눈에]", "[3 가상 장부]",
                                 '<details class="fold"')]
    assert i == sorted(i)
    assert _folds(body) == ["▶ 시장 지표 그래프 (지금의 시장 온도 전체)", "▶ 월간 성적표", "▶ 지표 설명 (읽는 법·용어)", "▶ 보고서 정보"]
    assert 'id="fold-candidates"' not in html and 'id="fold-detail"' not in html and "오늘 매수 추천" not in html
    assert "엔비디아" not in body  # 스크리닝 추천 종목 숨김
    on = {**cfg, "report": {**cfg["report"], "show_screening": True}}
    html_on = report_html.render_report(s, on, tmp_path).read_text(encoding="utf-8")
    assert 'id="fold-candidates"' in html_on and 'id="fold-detail"' in html_on and "엔비디아" in html_on


def test_group_edition_excludes_personal_sections_and_tickers(cfg, tmp_path):
    s = _summary(cfg)
    s["vbooks"]["books"] = _books(cfg, accel_positions={"MU": {}, "AMD": {}},
                                  accel_trades=[{"side": "매수", "ticker": "MU", "fill_date": BASE}])
    s["flow_watch"] = {"as_of_str": "2026-10-16", "scan_count": 1, "excluded": [], "sectors": [],
                       "top": [{"rank": 1, "name": "마이크론", "ticker": "MU", "status": "유입", "share_ratio": 2.0,
                                "volume_ratio": 1.0, "weekly_return": 0.1, "up_volume_ratio": 0.6, "dollar_volume_5d": 1e9,
                                "high_52w": -0.1, "our_signal": ""}]}
    on = {**cfg, "report": {**cfg["report"], "show_screening": True}}  # 켜도 단체방판엔 스크리닝 없음
    path = report_html.render_report(s, on, tmp_path, edition="group")
    assert path.name.endswith("_group.html")
    html = path.read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    assert _folds(body) == ["▶ 시장 지표 그래프 (지금의 시장 온도 전체)", "▶ 지표 설명 (읽는 법·용어)", "▶ 보고서 정보"]
    for x in ("오늘 할 일:", "[시장 한눈에]", "[가상 장부]", "오늘 가상 거래 1건"):
        assert x in body
    for x in ("점검:", "[1 내 돈", "[4 연구", "[2 시장", "[3 가상"):
        assert x not in body
    for x in ("월간 성적표", "돈이 몰리는", "강한 종목", "올해 세금 없는 매도 한도", "오늘의 스크리닝", 'id="fold-detail"'):
        assert x not in body
    # 티커·종목명 목록 없음(전략 이름의 QQQ·QLD·TQQQ 제외)
    for t in ("MU", "AMD", "NVDA", "엔비디아", "마이크론", "SGOV"):
        assert not re.search(rf"(?<![A-Za-z]){t}(?![A-Za-z])", body), t
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "오늘 가상 매수: QQQ 대비 1개월 강세 전략 MU" in personal and "▶ 돈이 몰리는 종목 (관찰용)" in personal


# ── 월간 성적표: 시작 전 달은 만들지 않음 ────────────────────────────────────


def test_scorecard_skips_months_before_ledger_start():
    days = pd.bdate_range("2026-09-01", "2026-12-31")
    start = pd.Timestamp("2026-10-14")
    # 10/16 실행 + 시트에 9월 없음 → 예전에는 2026-09를 넣었다. 이제 없음
    assert scr.scorecard_month(pd.Timestamp("2026-10-16").date(), days, done=set(), start=start) is None
    assert scr.scorecard_month(pd.Timestamp("2026-10-01").date(), days, done=None, start=start) is None
    assert str(scr.scorecard_month(pd.Timestamp("2026-11-02").date(), days, done=set(), start=start)) == "2026-10"
    # 다시 넣기(첫 거래일 실패 → 다음 실행)도 같은 조건
    assert str(scr.scorecard_month(pd.Timestamp("2026-11-05").date(), days, done={"2026-09"}, start=start)) == "2026-10"
    assert scr.scorecard_month(pd.Timestamp("2026-11-05").date(), days, done={"2026-10"}, start=start) is None
    assert scr.ledger_start(load_config()) == start


# ── 보여 주는 이름 ───────────────────────────────────────────────────────────


def test_display_names_in_scorecard_and_sheet_tables_roundtrip(cfg):
    strats = sc.strategies_from_cfg(cfg)
    assert [s.label for s in strats] == ["QQQ 전략", "QLD 전략", "TQQQ 전략", "QLD 추세 전략", "TQQQ 추세 전략",
                                         "기술지표 전략", "볼린저 전략", "가치평가 전략", "QQQ 대비 1개월 강세 전략"]
    assert strats[7].name == "가치평가 따라하기"  # 8번 자리 교체(2026-10-18), 꺼 둔 6개월 강세는 빠짐
    assert strats[5].name == "추천 종목 따라하기"  # DB 조회용 id는 그대로
    m = vbtext.display_map(cfg)
    tables = {"전략비교_일일": [["날짜", "QLD 200일선 전환 총자산($)", "추천 종목 따라하기 누적(%)"]],
              "매매기록": [["장부"], ["볼린저·RSI·MACD 따라하기"]], "전략요약": [["장부"], ["config 고정값(picks)"]]}
    shown = vbtext.rename_tables(tables, m)
    assert shown["전략비교_일일"][0] == ["날짜", "QLD 추세 전략 총자산($)", "기술지표 전략 누적(%)"]
    assert shown["매매기록"][1] == ["볼린저 전략"] and shown["전략요약"][1] == ["config 고정값(picks)"]
    assert vbtext.unrename_tables(shown, m) == tables  # 시트 복구는 장부 id로 되돌려 읽는다


# ── 12월 공제 안내 연결 ──────────────────────────────────────────────────────


def _dec_days(year=2026):
    d = pd.bdate_range(f"{year}-11-20", f"{year}-12-31")
    return d[~d.isin(pd.to_datetime([f"{year}-11-26", f"{year}-12-25"]))]


def _qqqm_fills():
    return pd.DataFrame([("2025-01-02", "QQQM", "buy", 100, 100.0, 1000.0)],
                        columns=["date", "ticker", "side", "qty", "price", "fx_rate"])


def test_dec_tax_attached_only_on_show_days_and_isolated(cfg):
    from engine import daily

    close = pd.Series([160.0], index=[pd.Timestamp("2026-12-01")])
    s = {"as_of": pd.Timestamp("2026-12-01"), "fx_rate": 1000.0}
    daily._attach_dec_tax(s, cfg, fills_fn=_qqqm_fills, qqqm_fn=lambda: close, days_fn=lambda y: _dec_days(y))
    assert s["dec_tax"]["todo"] == "올해 세금 없는 매도 한도 확인" and s["dec_tax"]["status"] == "ok"
    assert any("팔 수량 약 41주" in x for x in s["dec_tax"]["lines"]) and any("12/29까지 체결" in x for x in s["dec_tax"]["lines"])
    assert letter.todo_line(s, cfg) == "오늘 할 일: 올해 세금 없는 매도 한도 확인"
    assert letter.todo_line(s, cfg, letter.GROUP) == "오늘 할 일: 없음 — QQQM 적립만 그대로"  # 단체방판 제외
    other = {"as_of": pd.Timestamp("2026-12-02"), "fx_rate": 1000.0}
    daily._attach_dec_tax(other, cfg, fills_fn=_qqqm_fills, qqqm_fn=lambda: close, days_fn=lambda y: _dec_days(y))
    assert "dec_tax" not in other  # 표시일이 아님
    mid = {"as_of": pd.Timestamp("2026-12-15"), "fx_rate": 1000.0}
    daily._attach_dec_tax(mid, cfg, fills_fn=_qqqm_fills, qqqm_fn=lambda: close, days_fn=lambda y: _dec_days(y))
    assert "dec_tax" in mid

    def broken():
        raise ConnectionError("sheet down")

    bad = {"as_of": pd.Timestamp("2026-12-01"), "fx_rate": 1000.0, "dec_tax": {"todo": "old"}}
    daily._attach_dec_tax(bad, cfg, fills_fn=broken, qqqm_fn=lambda: close, days_fn=lambda y: _dec_days(y))
    assert "dec_tax" not in bad  # 예외 격리 — 안내만 빠짐


def test_dec_tax_fold_personal_only(cfg, tmp_path):
    s = _summary(cfg)
    s["dec_tax"] = {"todo": "올해 세금 없는 매도 한도 확인", "lines": ["남은 한도 2,500,000원", "마감: 12/29까지 체결"]}
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "▶ 올해 세금 없는 매도 한도" in personal and "오늘 할 일: 월간 성적표 보기(10월) · 올해 세금 없는 매도 한도 확인" in personal
    folds = _folds(personal[personal.index("<header>"):])
    assert folds.index("▶ 월간 성적표") < folds.index("▶ 올해 세금 없는 매도 한도") < folds.index("▶ 지표 설명 (읽는 법·용어)")
    group = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = group[group.index("<header>"):group.index("<script>")]
    assert "세금 없는 매도 한도" not in body and "남은 한도 2,500,000원" not in body


# ── 발송 관문: 아침 편지도 Actions 또는 --send에서만 ──────────────────────────


def test_briefing_sends_are_dry_run_locally(monkeypatch, tmp_path):
    from notify import telegram

    calls = []
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-2")
    for name in ("_send_document_checked", "_send_text_checked", "_send_text", "_send_document"):
        monkeypatch.setattr(telegram, name, lambda *a, **k: calls.append(a) or (True, None, None))
    assert not telegram.real_send_allowed()
    report = tmp_path / "r.html"
    report.write_text("x", encoding="utf-8")
    s = {"as_of": pd.Timestamp("2026-10-16"), "mode": "live", "report_path": report, "mode_label": "실전"}
    telegram.send_briefing("본문", s, {})
    res = telegram.send_group_briefing("본문", report, s, {})
    assert res["skipped_reason"] == "local_dry_run" and not res["attempted"]
    assert telegram.notify_ops_error("오류") is False and telegram.send_alert_text("알림") is False
    assert calls == []
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert telegram.real_send_allowed()
    monkeypatch.delenv("GITHUB_ACTIONS")
    monkeypatch.setenv("SCREENER_REAL_SEND", "1")  # engine.daily --send
    assert telegram.real_send_allowed()


def test_daily_send_flag_sets_env_and_ps1_does_not_pass_it():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    src = (root / "engine" / "daily.py").read_text(encoding="utf-8")
    assert '"--send"' in src and "os.environ[telegram.SEND_ENV] = \"1\"" in src
    ps1 = (root / "scripts" / "run_daily.ps1").read_text(encoding="utf-8")
    assert "--send" not in ps1.split("NOTE (2026-10-17)")[1].split("#>")[1]  # 로컬 예약 실행은 dry-run


# ── 시트 「월간성적표」 머리글 ────────────────────────────────────────────────


class _WS:
    title = "월간성적표"

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    def get_all_values(self):
        return self.rows

    def col_values(self, i):
        return [r[i - 1] if len(r) >= i else "" for r in self.rows]

    def append_rows(self, rows, value_input_option=None):
        self.rows += [list(r) for r in rows]

    def insert_row(self, row, index, value_input_option=None):
        self.rows.insert(index - 1, list(row))

    def clear(self):
        raise AssertionError("지우면 안 된다")


def _client(ws):
    class SS:
        def worksheet(self, name):
            return ws

    class C:
        def open_by_key(self, k):
            return SS()

    return C()


def test_scorecard_tab_gets_header_on_top_without_touching_data():
    from data import virtual_books_sheet as vbsheet

    data = [["2026-09", "1", "QQQ 그냥 보유"], ["2026-09", "2", "QLD 그냥 보유"]]
    ws = _WS(data)
    assert vbsheet.scorecard_months(_client(ws), "x") == {"2026-09"}  # 머리글 없는 1행도 달로 셈
    vbsheet.append_scorecard(_client(ws), "x", [["2026-10", "1", "QQQ 전략"]])
    assert ws.rows[0] == vbsheet.SCORECARD_HEADER and ws.rows[1:3] == data and ws.rows[3][0] == "2026-10"
    vbsheet.append_scorecard(_client(ws), "x", [["2026-11", "1", "QQQ 전략"]])
    assert sum(r == vbsheet.SCORECARD_HEADER for r in ws.rows) == 1  # 이미 있으면 다시 넣지 않음
    empty = _WS([])
    vbsheet.append_scorecard(_client(empty), "x", [["2026-10", "1", "QQQ 전략"]])
    assert empty.rows[0] == vbsheet.SCORECARD_HEADER and empty.rows[1][0] == "2026-10"
    assert vbsheet.scorecard_months(_client(empty), "x") == {"2026-10"}
