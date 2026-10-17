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

BASE = pd.Timestamp("2026-10-16")


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


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
    wait = {"6개월 강세 따라하기": pd.Timestamp("2026-10-30")} if waiting else None
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


def test_glance_three_lines_and_caution_line_omitted_when_all_stable():
    q = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    rows = [_macro("fear-greed", "공포·탐욕 지수", 62, "주의", "탐욕"), _macro("dgs10", "미국 10년물 국채금리", 4.12, "안정"),
            _macro("fx", "원/달러 환율", 1334.2, "위험")]
    lines = letter.glance_lines(q, rows)
    assert len(lines) == 3
    assert lines[0].startswith("QQQ 하루 +") and "1주 +" in lines[0] and "200일선 대비 +" in lines[0] and lines[0].endswith("상승장")
    assert lines[1] == "공포·탐욕 62(탐욕) · 미 10년 금리 4.12% · 원/달러 1,334원"
    assert lines[2] == "주의: 공포·탐욕 지수, 원/달러 환율"
    calm = [dict(r, badge={"label": "안정", "zone": None}) for r in rows]
    assert len(letter.glance_lines(q, calm)) == 2 and not any(x.startswith("주의") for x in letter.glance_lines(q, calm))
    assert letter.glance_lines(None, calm)[0].startswith("공포·탐욕")  # QQQ를 못 받으면 1줄만 빠짐


# ── 실험 중인 전략 ────────────────────────────────────────────────────────────


def test_strategy_lines_names_four_statuses_and_six_to_nine(cfg):
    books = _books(cfg)
    assert [b["label"] for b in books] == ["QQQ 전략", "QLD 전략", "TQQQ 전략", "QLD 추세 전략", "TQQQ 추세 전략",
                                           "기술지표 전략", "볼린저 전략", "6개월 강세 전략", "한 달 가속 전략"]
    lines = letter.strategy_lines(books, cfg)
    assert len(lines) == 4
    assert lines[0] == "QQQ 전략 +1.0% · QLD 전략 +2.0%(QQQ 대비 +1.0%p) · TQQQ 전략 +3.0%(QQQ 대비 +2.0%p) — 오늘 거래 없음"
    assert "QLD 추세 전략 +2.0%(QQQ 대비 +1.0%p) 지금 상승 추세라 QLD 보유 중" in lines[1]
    # 결과가 같은 6·7번은 한 줄로 합침, 상태 "살 종목이 없어 현금 보관", QQQ 대비는 늦게 시작한 날부터
    assert lines[2].startswith("기술지표 전략·볼린저 전략 +0.0%") and "살 종목이 없어 현금 보관" in lines[2]
    assert "6개월 강세 전략 10/30 시작" in lines[3] and "한 달 가속 전략 +0.4%" in lines[3]
    assert letter.strategy_header(books) == "[실험 중인 전략] 가상 1억 · 10/14 시작 · 매수 추천 아님"


def test_trend_down_status_and_merged_up_status(cfg):
    down = _books(cfg, trend_holding="SGOV")
    line = letter.strategy_lines(down, cfg)[1]
    assert line.startswith("QLD 추세 전략·TQQQ 추세 전략 +0.0%") and "하락 추세라 단기 국채로 피신 중" in line
    b = _books(cfg)
    for x in b:
        if x["family"] == "trend":
            x["cum_return_pct"], x["vs_qqq_pp"] = 2.0, 1.0
    assert "지금 상승 추세라 QLD(TQQQ) 보유 중" in letter.strategy_lines(b, cfg)[1]


def test_virtual_trades_one_line_personal_tickers_group_count(cfg):
    trades = [{"side": "매수", "ticker": "MU", "fill_date": BASE}, {"side": "매수", "ticker": "AMD", "fill_date": BASE},
              {"side": "건너뜀", "ticker": "X", "fill_date": BASE}, {"side": "매도", "ticker": "OLD", "fill_date": BASE - pd.Timedelta(days=1)}]
    books = _books(cfg, accel_positions={"MU": {}, "AMD": {}}, accel_trades=trades,
                   book_trades={"QLD 200일선 전환": [{"side": "매도", "ticker": "QLD", "fill_date": BASE}]})
    p = letter.strategy_lines(books, cfg)
    assert p[1].endswith("— 오늘 가상 매도: QLD")
    assert p[3].endswith("— 오늘 가상 매수: MU, AMD") and "2종목 보유" in p[3]
    g = letter.strategy_lines(books, cfg, letter.GROUP)
    assert g[3].endswith("— 오늘 가상 거래 2건") and "MU" not in "".join(g) and g[0].endswith("— 오늘 거래 없음")


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
    i = [body.index(x) for x in ("<header>", "오늘 할 일:", "[시장 한눈에]", "[실험 중인 전략]", '<details class="fold"')]
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
    for x in ("오늘 할 일:", "[시장 한눈에]", "[실험 중인 전략]", "오늘 가상 거래 1건"):
        assert x in body
    for x in ("월간 성적표", "돈이 몰리는", "강한 종목", "올해 세금 없는 매도 한도", "오늘의 스크리닝", 'id="fold-detail"'):
        assert x not in body
    # 티커·종목명 목록 없음(전략 이름의 QQQ·QLD·TQQQ 제외)
    for t in ("MU", "AMD", "NVDA", "엔비디아", "마이크론", "SGOV"):
        assert not re.search(rf"(?<![A-Za-z]){t}(?![A-Za-z])", body), t
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "오늘 가상 매수: MU" in personal and "▶ 돈이 몰리는 종목 (관찰용)" in personal


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
                                         "기술지표 전략", "볼린저 전략", "6개월 강세 전략", "한 달 가속 전략"]
    assert strats[5].name == "추천 종목 따라하기"  # DB 조회용 id는 그대로
    m = vbtext.display_map(cfg)
    tables = {"전략비교_일일": [["날짜", "QLD 200일선 전환 총자산($)", "추천 종목 따라하기 누적(%)"]],
              "매매기록": [["장부"], ["볼린저·RSI·MACD 따라하기"]], "전략요약": [["장부"], ["config 고정값(picks)"]]}
    shown = vbtext.rename_tables(tables, m)
    assert shown["전략비교_일일"][0] == ["날짜", "QLD 추세 전략 총자산($)", "기술지표 전략 누적(%)"]
    assert shown["매매기록"][1] == ["볼린저 전략"] and shown["전략요약"][1] == ["config 고정값(picks)"]
    assert vbtext.unrename_tables(shown, m) == tables  # 시트 복구는 장부 id로 되돌려 읽는다
