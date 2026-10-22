"""새 출발 (2026-10-22): 가상 장부 끔·사용자 시트 「QQQM 모으기」 읽기 전용·편지 새 구조. 네트워크 없음."""

from __future__ import annotations

import copy
import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from data import holdings as hd
from engine import daily
from engine.daily import KST, load_config
from notify import briefing, letter, report_html


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


SHEET = [  # 사용자 시트 「전략공장」 「QQQM 모으기」 (2026-10-22 배치: 2행 머리글, 3행 값 — B3·C3·D3·K3)
    ["QQQM 모으기 (실제 돈) — 노란 칸만 직접 고치세요"],
    ["종목", "보유 주식 수", "평균단가($)", "평균 환율(원/$)", "현재가($)", "오늘 환율(원/$)", "손익률", "투자 원금(원)", "평가금액(원)",
     "손익(원)", "올해 실현 이익(원)", "250만 원 공제 남은 금액(원)"],
    ["QQQM", "10", "250.5", "1,350", "298.64", "1321.94924", "", "", "", "", "500,000", "2000000"],
]

# ── 설정 기본값 ──────────────────────────────────────────────────────────────


def test_defaults_everything_off(cfg):
    assert cfg["virtual_books"]["enabled"] is False and cfg["virtual_books"]["penny"]["enabled"] is False
    assert cfg["live"]["auto_plan"]["enabled"] is False and cfg["live"]["sheet_plan_fills"] is False
    assert cfg["report"]["show_stock_folds"] is False and cfg["report"]["monthly_scorecard"] is False
    assert cfg["letter"]["show_books"] is False
    from engine import virtual_books as vbooks

    assert vbooks.run_safely(cfg) is None  # 장부·장부 시트 쓰기·동전주 알림 모두 안 돈다
    assert "in_progress" not in cfg["research_news"]  # 예전 연구 중 목록은 뺐다
    assert [x["name"] for x in cfg["research_news"]["finished"]] == ["상승장 RSI 눌림 전략"]  # 끝난 기록만 남김


# ── 「QQQM 모으기」 읽기 ────────────────────────────────────────────────────────


def test_parse_holdings_by_header_and_label(cfg):
    h = hd.parse_holdings(SHEET, cfg)
    assert (h["qty"], h["avg_price"], h["avg_fx"], h["realized_krw"], h["warnings"]) == (10.0, 250.5, 1350.0, 500_000.0, [])


def test_parse_holdings_columns_moved_and_blanks(cfg):
    moved = [SHEET[0], ["종목", "올해 실현 이익(원)", "평균단가($)", "메모", "보유 주식 수", "평균 환율(원/$)"],
             ["QQQM", "0", "250.5", "x", "7", ""]]
    h = hd.parse_holdings(moved, cfg)
    assert (h["qty"], h["avg_price"], h["avg_fx"], h["realized_krw"]) == (7.0, 250.5, None, 0.0)  # 평균 환율 빈칸 → None
    empty = copy.deepcopy(SHEET)
    empty[2] = ["QQQM", "", "", ""]
    h = hd.parse_holdings(empty, cfg)
    assert h["qty"] is None and h["avg_price"] is None and h["realized_krw"] is None and h["warnings"] == []
    broken = hd.parse_holdings([["x"], ["종목"]], cfg)
    assert len(broken["warnings"]) == 4  # 머리글 4개 없음
    assert hd.settings(cfg)["tab"] == "QQQM 모으기"


def test_read_holdings_read_only(cfg, monkeypatch):
    calls = []

    class WS:
        def get_all_values(self):
            calls.append("read")
            return SHEET

        def __getattr__(self, name):  # 쓰기 메서드를 부르면 실패
            raise AssertionError(f"쓰기 금지: {name}")

    class C:
        def open_by_key(self, key):
            return type("S", (), {"worksheet": lambda self, name: WS() if name == "QQQM 모으기" else 1 / 0})()

    from data import sheets

    monkeypatch.setattr(sheets, "_sheet_id", lambda: "x")
    assert hd.read_holdings(cfg, client=C())["qty"] == 10.0 and calls == ["read"]


# ── 내 돈·12월 공제 ──────────────────────────────────────────────────────────


def test_holdings_money_and_empty_message(cfg):
    h = hd.parse_holdings(SHEET, cfg)
    m = letter.holdings_money(h, 300.0, 1400.0, 2026)
    assert m["value_krw"] == pytest.approx(10 * 300 * 1400)
    assert m["pnl_pct"] == pytest.approx((10 * 300 * 1400) / (10 * 250.5 * 1350) * 100 - 100)
    assert m["remaining_krw"] == 2_000_000
    lines = letter.money_lines(m)
    assert lines[0].startswith("QQQM 10주 · 평가 4,200,000원") and "12월 공제 남은 금액 2,000,000원" in lines[1]
    assert letter.holdings_money({**h, "qty": None}, 300.0, 1400.0, 2026) is None
    assert letter.money_lines(None) == ["QQQM 모으기 탭에 보유 주식 수·평균단가를 넣으면 표시됩니다"]
    no_fx = letter.holdings_money({**h, "avg_fx": None}, 300.0, 1400.0, 2026)
    assert no_fx["cost_krw"] == pytest.approx(10 * 250.5 * 1400)  # 평균 환율 빈칸 → 오늘 환율


def test_dec_tax_uses_holdings(cfg):
    s = {"as_of": pd.Timestamp("2026-12-01"), "fx_rate": 1400.0}
    h = hd.parse_holdings(SHEET, cfg)
    close = pd.Series([300.0], index=[pd.Timestamp("2026-12-01")])
    daily._attach_dec_tax(s, cfg, holdings=h, qqqm_fn=lambda: close)
    assert s["dec_tax"]["todo"] == letter.DEC_TAX_TODO
    assert "남은 한도 2,000,000원" in " ".join(s["dec_tax"]["lines"])


# ── 연구 소식 ────────────────────────────────────────────────────────────────


def test_research_lines_monday_and_day_after_verdict(cfg):
    cfg["research_news"]["current"].update(stage="사전 등록", verdict=None, result=None, date=None)  # 판정 전 모양부터
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-26")) == [
        "「변동성 맞춤 레버리지 전략」 [사전 등록] → 탐색 중 → 판정"]  # 월요일
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-27")) == []  # 화요일
    cfg["research_news"]["current"].update(stage="탐색 중")
    assert letter.research_lines(cfg, pd.Timestamp("2026-11-02")) == ["「변동성 맞춤 레버리지 전략」 사전 등록 → [탐색 중] → 판정"]
    cfg["research_news"]["current"].update(verdict="탈락", result="미국 세 구간 중 1곳만 1배 보유보다 높음",
                                           date="2026-10-22")
    after = letter.research_lines(cfg, pd.Timestamp("2026-10-23"))  # 판정 다음 날(금)
    assert after == ["「변동성 맞춤 레버리지 전략」 사전 등록 → 탐색 중 → [판정]",
                     "결과: 미국 세 구간 중 1곳만 1배 보유보다 높음 → 탈락"]
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-24")) == []
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-26")) == after  # 다음 월요일


# ── 편지 새 구조 ─────────────────────────────────────────────────────────────


def _summary(cfg, monday=True):
    from tests.test_report_titles import _summary as base

    s = base(cfg)
    s["qqq_close"] = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    s["weekly_rs"] = {"top": []}
    s["flow_watch"] = {"top": []}
    today = pd.Timestamp("2026-10-26" if monday else "2026-10-27")
    s["letter_today_kst"] = today
    s["letter_check"] = letter.check_info(None, s["as_of"], s["as_of"], 0)
    s["research_news"] = letter.research_lines(cfg, today)
    s["my_money"] = None
    return s


def _folds(html):
    return re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", html)


def test_personal_letter_new_order_no_books_no_stock_folds(cfg, tmp_path):
    s = _summary(cfg)
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    keys = ("<header>", "점검: 기준일", "오늘 할 일: 없음 — QQQM 적립만 그대로", "[1 내 돈 · QQQM]", "[2 시장 한눈에]",
            "[3 연구 소식]", "「변동성 맞춤 레버리지 전략」", '<details class="fold"')
    i = [body.index(x) for x in keys]
    assert i == sorted(i)
    check = re.search(r"점검: [^<]+", body).group(0)
    assert "장부" not in check and check.endswith("데이터 지연 없음")
    assert "QQQM 모으기 탭에 보유 주식 수·평균단가를 넣으면 표시됩니다" in body
    for x in ("가상 장부", "[3 가상", "QQQ 대비 누적", "전략 표로 보기", "강한 종목", "돈이 몰리는", "월간 성적표", "상승장 RSI"):
        assert x not in body, x
    assert _folds(body) == ["▶ 시장 지표 그래프 (지금의 시장 온도 전체)", "▶ 지표 설명 (읽는 법·용어)", "▶ 보고서 정보"]
    tue = report_html.render_report(_summary(cfg, monday=False), cfg, tmp_path).read_text(encoding="utf-8")
    assert "연구 소식" not in tue


def test_group_letter_excludes_check_and_money(cfg, tmp_path):
    s = _summary(cfg)
    s["my_money"] = {"qty": 8.0, "close_usd": 250.0, "value_krw": 2_800_000.0, "cost_krw": 1.0, "pnl_krw": 1.0, "pnl_pct": 1.0,
                     "realized_krw": 0.0, "remaining_krw": 2_500_000.0}
    html = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    for x in ("점검:", "내 돈", "2,800,000", "가상 장부", "QQQ 대비 누적", "강한 종목", "돈이 몰리는"):
        assert x not in body, x
    i = [body.index(x) for x in ("오늘 할 일:", "[시장 한눈에]", "[연구 소식]", "「변동성 맞춤 레버리지 전략」")]
    assert i == sorted(i)
    cap = briefing.build_report_caption(s, cfg).split("\n")
    assert cap[1].startswith("점검: 기준일") and "장부" not in cap[1] and cap[2].startswith("오늘 할 일:")
    assert "점검" not in briefing.build_report_caption(s, cfg, edition="group")


def test_attach_letter_v3_lag_money_research(cfg):
    s = {"as_of": pd.Timestamp("2026-10-19"), "fx_rate": 1400.0}
    now = datetime(2026, 10, 26, 8, 7, tzinfo=KST)  # 월요일 08:07 → 마지막 마감 10/23(금)
    close = pd.Series([290.0, 300.0, 999.0], index=pd.to_datetime(["2026-10-16", "2026-10-19", "2026-10-20"]))
    daily._attach_letter_v3(s, cfg, hd.parse_holdings(SHEET, cfg), qqqm_fn=lambda: close, now_fn=lambda: now)
    assert s["letter_check"]["lag"] == 4 and letter.check_line(s["letter_check"]).startswith("점검: 기준일 10/19(월) · 데이터 지연 4거래일")
    assert s["my_money"]["close_usd"] == 300.0  # 기준일까지 종가
    assert s["research_news"][0].startswith("「변동성 맞춤 레버리지 전략」")
    daily._attach_letter_v3(s, cfg, None, now_fn=lambda: now)
    assert s["my_money"] is None


def test_read_holdings_failure_is_isolated(cfg):
    from data import sheets

    def boom():
        raise sheets.SheetsConfigError("없음")

    assert daily._read_holdings(cfg, boom) is None
    assert daily._read_holdings(cfg, lambda: (_ for _ in ()).throw(RuntimeError("네트워크"))) is None


# ── 시트: 계획·체결 안 읽음, sheet_setup 비활성 ─────────────────────────────────


def test_run_live_does_not_read_plan_or_fills_sheet(cfg, monkeypatch, tmp_path):
    from data import sheets
    from tests.test_live_advisor import _patch_run_io

    engine_daily = _patch_run_io(monkeypatch, tmp_path)
    monkeypatch.setattr(sheets, "read_sheets", lambda *a, **k: (_ for _ in ()).throw(AssertionError("계획·체결 시트 읽음")))
    monkeypatch.setattr(engine_daily, "load_fills", lambda: (_ for _ in ()).throw(AssertionError("로컬 체결 읽음")))
    summary = engine_daily.run(cfg, "live", do_replay=False, dry_run=True)
    assert summary["plan_by_ticker"] == {}


def test_sheet_setup_disabled_by_default(monkeypatch, capsys):
    from data import sheets
    from scripts import sheet_setup

    monkeypatch.setattr(sheets, "get_client", lambda: (_ for _ in ()).throw(AssertionError("시트 접속")))
    assert sheet_setup.main(["--apply"]) == 0
    assert "비활성" in capsys.readouterr().out


def test_research_current_verdict_in_config(cfg):
    lines = letter.research_lines(cfg, pd.Timestamp("2026-10-23"))  # 판정(10/22) 다음 날
    assert lines[0] == "「변동성 맞춤 레버리지 전략」 사전 등록 → 탐색 중 → [판정]" and lines[1].endswith("→ 탈락")
    assert letter.research_lines(cfg, pd.Timestamp("2026-10-24")) == []


# ── 아침 편지 개선 2줄 (2026-10-22): A2 급락장 경고·A1 저축일 ────────────────────────────


def _qqq(kind):
    idx = pd.bdate_range("2024-01-01", periods=300)
    up = np.linspace(100, 160, 300)
    if kind == "up":
        v = up
    elif kind == "down":  # 200일선 아래로 천천히(폭락 조건은 안 됨: 252일 고점 대비 −20% 안, 20일 −10% 안)
        v = np.concatenate([np.linspace(100, 130, 240), np.linspace(130, 114, 60)])
    else:  # 폭락: 20일 −10% 넘게
        v = np.concatenate([up[:280], np.linspace(up[279], up[279] * 0.85, 20)])
    return pd.Series(v, index=idx)


def test_crash_warning_regimes():
    from factory import regimes

    assert regimes.classify(_qqq("down")).iloc[-1] == regimes.DOWN
    assert regimes.classify(_qqq("crash")).iloc[-1] == regimes.CRASH
    assert letter.crash_warning(_qqq("up")) is None
    assert letter.crash_warning(_qqq("down")) == letter.CRASH_WARNING
    assert letter.crash_warning(_qqq("crash")) == letter.CRASH_WARNING
    assert letter.crash_warning(None) is None and letter.crash_warning(_qqq("up").iloc[:50]) is None  # 판단불가


def test_crash_warning_both_editions_below_todo(cfg, tmp_path):
    for kind, shown in (("up", False), ("down", True), ("crash", True)):
        s = _summary(cfg)
        s["qqq_close"] = _qqq(kind)
        for ed in ("personal", "group"):
            body = report_html.render_report(s, cfg, tmp_path, edition=ed).read_text(encoding="utf-8")
            body = body[body.index("<header>"):body.index("<script>")]
            assert (letter.CRASH_WARNING in body) is shown, (kind, ed)
            if shown:
                assert body.index("오늘 할 일:") < body.index(letter.CRASH_WARNING) < body.index("시장 한눈에]")


def test_savings_day_weekend_holiday_and_blank(cfg):
    from data.market_calendar import trading_days_between

    tdays = trading_days_between(pd.Timestamp("2026-09-01").date(), pd.Timestamp("2027-02-28").date())
    cfg["savings_day"] = 25
    assert letter.is_savings_day(cfg, "2026-11-25", tdays)  # 수요일
    assert not letter.is_savings_day(cfg, "2026-11-24", tdays)
    assert not letter.is_savings_day(cfg, "2026-10-25", tdays)  # 일요일 → 다음 거래일 10/26(월)
    assert letter.is_savings_day(cfg, "2026-10-26", tdays)
    assert letter.is_savings_day(cfg, "2026-12-28", tdays) and not letter.is_savings_day(cfg, "2026-12-25", tdays)  # 성탄 휴장·주말
    cfg["savings_day"] = 31
    assert letter.is_savings_day(cfg, "2026-11-30", tdays)  # 31일이 없는 달 → 그달 마지막 날
    assert letter.is_savings_day(cfg, "2026-12-31", tdays) and not letter.is_savings_day(cfg, "2027-01-04", tdays)  # 12/31(목) 거래일
    assert letter.is_savings_day(cfg, "2027-02-01", tdays) and not letter.is_savings_day(cfg, "2027-01-29", tdays)  # 1/31(일) → 2/1(월)
    cfg["savings_day"] = None
    assert not letter.is_savings_day(cfg, "2026-11-25", tdays)
    cfg["savings_day"] = ""
    assert not letter.is_savings_day(cfg, "2026-11-25", tdays)


def test_savings_todo_personal_only(cfg):
    s = _summary(cfg)
    s["savings_day"] = True
    assert letter.todo_line(s, cfg, letter.PERSONAL) == "오늘 할 일: " + letter.SAVINGS_TODO
    assert letter.todo_line(s, cfg, letter.GROUP) == "오늘 할 일: " + letter.TODO_NONE
    s["savings_day"] = False
    assert letter.todo_line(s, cfg, letter.PERSONAL) == "오늘 할 일: " + letter.TODO_NONE


def test_attach_v3_sets_savings_flag(cfg):
    cfg["savings_day"] = 26
    s = {"as_of": pd.Timestamp("2026-10-23"), "fx_rate": 1400.0}
    daily._attach_letter_v3(s, cfg, None, now_fn=lambda: datetime(2026, 10, 26, 8, 7, tzinfo=KST))
    assert s["savings_day"] is True
    daily._attach_letter_v3(s, cfg, None, now_fn=lambda: datetime(2026, 10, 27, 8, 7, tzinfo=KST))
    assert s["savings_day"] is False
