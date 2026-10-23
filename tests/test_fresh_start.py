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
from notify import letter_plain as lp


@pytest.fixture
def cfg():
    return copy.deepcopy(load_config())


SHEET = [  # 사용자 시트 「전략공장」 「QQQM 모으기」 (2026-10-23 배치: 2행 머리글, 3행 QQQM — B3·C3·D3·K3, 4행 ISA — B4·C4)
    ["QQQM · ISA 모으기 (실제 돈) — 노란 칸만 직접 고치세요"],
    ["종목", "보유 주식 수", "평균단가($/원)", "평균 환율(원/$)", "현재가($/원)", "오늘 환율(원/$)", "손익률", "투자 원금(원)",
     "평가금액(원)", "손익(원)", "올해 실현 이익(원)", "250만 원 공제 남은 금액(원)"],
    ["QQQM", "10", "250.5", "1,350", "298.64", "1321.94924", "", "", "", "", "500,000", "2000000"],
    ["TIGER 미국나스닥100 (ISA·원화)", "", "", "—", "170375", "—", "", "", "", "", "—", "—"],
    ["합계", "", "", "", "", "", "", "0", "0", "0"],
]

# ── 설정 기본값 ──────────────────────────────────────────────────────────────


def test_defaults_everything_off(cfg):
    assert cfg["virtual_books"]["enabled"] is False and cfg["virtual_books"]["penny"]["enabled"] is False
    assert cfg["live"]["auto_plan"]["enabled"] is False and cfg["live"]["sheet_plan_fills"] is False
    assert cfg["report"]["show_stock_folds"] is False and cfg["report"]["monthly_scorecard"] is False
    assert cfg["letter"]["show_books"] is False
    from engine import virtual_books as vbooks

    assert vbooks.run_safely(cfg) is None  # 장부·장부 시트 쓰기·동전주 알림 모두 안 돈다
    assert cfg["research_news"]["in_progress"] == ["상승장 대형주 웅덩이 v3.1(가상 기록, 12개월)"]  # 2026-10-23 웅덩이 v3.1 가상 기록
    assert [x["name"] for x in cfg["research_news"]["finished"]] == ["상승장 RSI 눌림 전략"]  # 끝난 기록만 남김


# ── 「QQQM 모으기」 읽기 ────────────────────────────────────────────────────────


def test_parse_holdings_by_header_and_label(cfg):
    h = hd.parse_holdings(SHEET, cfg)
    assert (h["qty"], h["avg_price"], h["avg_fx"], h["realized_krw"], h["warnings"]) == (10.0, 250.5, 1350.0, 500_000.0, [])
    assert h["isa"] == {"qty": None, "avg_price": None}  # ISA 노란 칸 빈칸


def test_parse_holdings_columns_moved_and_blanks(cfg):
    moved = [SHEET[0], ["종목", "올해 실현 이익(원)", "평균단가($)", "메모", "보유 주식 수", "평균 환율(원/$)"],  # 예전 머리글 이름도 읽음
             ["QQQM", "0", "250.5", "x", "7", ""], ["TIGER 미국나스닥100 (ISA·원화)", "—", "15,000원", "", "20", "—"]]
    h = hd.parse_holdings(moved, cfg)
    assert (h["qty"], h["avg_price"], h["avg_fx"], h["realized_krw"]) == (7.0, 250.5, None, 0.0)  # 평균 환율 빈칸 → None
    assert h["isa"] == {"qty": 20.0, "avg_price": 15000.0} and h["warnings"] == []
    empty = copy.deepcopy(SHEET)
    empty[2] = ["QQQM", "", "", ""]
    h = hd.parse_holdings(empty, cfg)
    assert h["qty"] is None and h["avg_price"] is None and h["realized_krw"] is None and h["warnings"] == []
    broken = hd.parse_holdings([["x"], ["종목"]], cfg)
    assert len(broken["warnings"]) == 5  # 머리글 4개·ISA 행 없음
    assert broken["isa"] == {"qty": None, "avg_price": None}
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
    cfg["research_news"].pop("in_progress", None)  # 진행 중 줄은 tests/test_puddle_v31.py에서 본다
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
    keys = ("<header>", "[오늘 할 일]", "없어요. 적립은 그대로 두면 돼요.", "[내 돈]", "[시장]",
            "[가상 실험 — 실제 돈 아님]", "「변동성 맞춤 레버리지 전략」", "기준: ", '<details class="fold"')
    i = [body.index(x) for x in keys]
    assert i == sorted(i)
    check = re.search(r"기준: [^<]+", body).group(0)  # 점검 줄은 맨 아래 작은 줄(2026-10-23 쉬운 문장판)
    assert "장부" not in check and check.endswith("자료 정상") and "자료 문제" not in body
    assert "시트 노란 칸에 보유 수와 평균단가를 넣으면 여기에 보여 드려요." in body
    for x in ("가상 장부", "[3 가상", "QQQ 대비 누적", "전략 표로 보기", "강한 종목", "돈이 몰리는", "월간 성적표", "상승장 RSI"):
        assert x not in body, x
    assert _folds(body) == ["▶ 그래프로 보기", "▶ 용어 풀이", "▶ 이 편지에 대해"]
    tue = report_html.render_report(_summary(cfg, monday=False), cfg, tmp_path).read_text(encoding="utf-8")
    assert "연구 소식" not in tue and "[가상 실험" not in tue


def test_group_letter_excludes_check_and_money(cfg, tmp_path):
    s = _summary(cfg)
    s["my_money"] = {"qty": 8.0, "close_usd": 250.0, "value_krw": 2_800_000.0, "cost_krw": 1.0, "pnl_krw": 1.0, "pnl_pct": 1.0,
                     "realized_krw": 0.0, "remaining_krw": 2_500_000.0}
    html = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    for x in ("점검:", "내 돈", "2,800,000", "280만", "가상 장부", "QQQ 대비 누적", "강한 종목", "돈이 몰리는"):
        assert x not in body, x
    i = [body.index(x) for x in ("[오늘 할 일]", "[시장]", "[가상 실험 — 실제 돈 아님]", "「변동성 맞춤 레버리지 전략」")]
    assert i == sorted(i)
    cap = briefing.build_report_caption(s, cfg).split("\n")
    assert cap[2] == "[오늘 할 일]" and "[내 돈]" in cap and "QQQM 280만 원, 1.0% 올랐어요." in cap
    assert cap[-1].startswith("기준: ") and "장부" not in cap[-1]
    group_cap = briefing.build_report_caption(s, cfg, edition="group")
    assert "점검" not in group_cap and "[내 돈]" not in group_cap and "280만" not in group_cap


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
            assert (lp.CRASH[0] in body) is shown, (kind, ed)
            if shown:
                assert body.index("[오늘 할 일]") < body.index(lp.CRASH[0]) < body.index("[시장]")
                assert all(x in body for x in lp.CRASH)


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


# ── 「내 돈」 ISA 줄·합계 (2026-10-23) ─────────────────────────────────────────────


def _with_isa(qty="20", avg="150,000"):
    sheet = copy.deepcopy(SHEET)
    sheet[3][1], sheet[3][2] = qty, avg
    return sheet


def _wallet(cfg, sheet, *, isa_fn=None, qqqm=300.0):
    s = {"as_of": pd.Timestamp("2026-10-22"), "fx_rate": 1400.0}
    close = pd.Series([qqqm], index=[pd.Timestamp("2026-10-22")])
    daily._attach_letter_v3(s, cfg, hd.parse_holdings(sheet, cfg), qqqm_fn=lambda: close,
                            now_fn=lambda: datetime(2026, 10, 23, 8, 7, tzinfo=KST),
                            isa_fn=isa_fn or (lambda: (pd.Timestamp("2026-10-22"), 170_000.0)))
    return s, letter.wallet_lines(s, cfg)


def test_isa_config_limits(cfg):
    assert letter.isa_tax_free_limit(cfg) == ("서민형", 4_000_000.0)
    assert letter.isa_tax_free_limit({**cfg, "isa_type": "일반형"}) == ("일반형", 2_000_000.0)


def test_wallet_isa_only(cfg):
    sheet = _with_isa()
    sheet[2][1] = sheet[2][2] = ""  # QQQM 빈칸
    s, lines = _wallet(cfg, sheet)
    # ISA 20주 × 170,000 = 3,400,000 / 원금 20 × 150,000 = 3,000,000 / 손익 +400,000 → 비과세 남은 3,600,000
    assert lines == ["QQQM: 노란 칸을 채우면 표시됩니다",
                     "ISA 20주 · 평가 3,400,000원 · 손익 +400,000원(+13.3%) · 비과세 남은 금액 3,600,000원(서민형 400만 원)",
                     "합계 평가 3,400,000원 · 손익 +400,000원(+13.3%)"]


def test_wallet_qqqm_only(cfg):
    s, lines = _wallet(cfg, SHEET, isa_fn=lambda: 1 / 0)  # ISA 빈칸이면 가격도 안 부른다
    assert s["wallet"]["isa_state"] == "empty"
    assert lines[0].startswith("QQQM 10주 · 평가 4,200,000원 · 손익 ") and lines[0].endswith("250만 원 공제 남은 금액 2,000,000원")
    assert lines[1] == "ISA: 노란 칸을 채우면 표시됩니다"
    cost = 10 * 250.5 * 1350
    assert lines[2] == f"합계 평가 4,200,000원 · 손익 {4_200_000 - cost:+,.0f}원({(4_200_000 / cost - 1) * 100:+.1f}%)"


def test_wallet_both(cfg):
    s, lines = _wallet(cfg, _with_isa())
    cost = 10 * 250.5 * 1350 + 3_000_000
    assert len(lines) == 3 and lines[0].startswith("QQQM 10주") and lines[1].startswith("ISA 20주")
    assert lines[2] == f"합계 평가 7,600,000원 · 손익 {7_600_000 - cost:+,.0f}원({(7_600_000 / cost - 1) * 100:+.1f}%)"
    # 손익이 한도보다 크면 비과세 남은 금액은 0
    big = letter.isa_money({"qty": 100.0, "avg_price": 100_000.0}, 150_000.0, 4_000_000.0)
    assert big["tax_free_left_krw"] == 0.0


def test_wallet_both_blank(cfg, tmp_path):
    sheet = copy.deepcopy(SHEET)
    sheet[2][1] = sheet[2][2] = ""
    s, lines = _wallet(cfg, sheet, isa_fn=lambda: 1 / 0)
    assert lines == ["QQQM: 노란 칸을 채우면 표시됩니다", "ISA: 노란 칸을 채우면 표시됩니다", "합계: 노란 칸을 채우면 표시됩니다"]
    assert letter.wallet_lines({"my_money": None}, cfg) == lines  # 시트를 못 읽었을 때(wallet 없음)도 같은 안내


def test_wallet_isa_price_failure(cfg):
    def boom():
        raise ConnectionError("x")

    s, lines = _wallet(cfg, _with_isa(), isa_fn=boom)
    assert s["wallet"]["isa_state"] == "no_price" and lines[1] == "ISA: 가격 못 받음"
    assert lines[0].startswith("QQQM 10주") and lines[2].startswith("합계 평가 4,200,000원")
    assert lines[2].endswith("(ISA 가격 못 받아 빼고)")
    s, lines = _wallet(cfg, _with_isa(), qqqm=float("nan"))  # QQQM 종가도 없으면 둘 다 가격 못 받음
    assert lines[0] == "QQQM: 가격 못 받음" and lines[2].endswith("(QQQM 가격 못 받아 빼고)")


def test_wallet_personal_only(cfg, tmp_path):
    s = _summary(cfg)
    s2, _ = _wallet(cfg, _with_isa())
    s["my_money"], s["wallet"] = s2["my_money"], s2["wallet"]
    personal = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "[내 돈]" in personal and "ISA 340만 원, 13.3% 올랐어요." in personal and "ISA 비과세 한도는 360만 원" in personal
    group = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    group = group[group.index("<header>"):group.index("<script>")]
    for x in ("내 돈", "ISA", "340만", "3,400,000", "비과세"):
        assert x not in group, x


def test_fetch_isa_close_drops_unfinished_bar():
    idx = pd.DatetimeIndex(["2026-10-21", "2026-10-22", "2026-10-23"]).tz_localize("Asia/Seoul")
    raw = pd.DataFrame({"Close": [169_000.0, 170_375.0, 171_000.0]}, index=idx)
    morning = datetime(2026, 10, 23, 8, 7, tzinfo=KST)
    assert hd.fetch_isa_close(None, morning, history_fn=lambda: raw) == (pd.Timestamp("2026-10-22"), 170_375.0)
    evening = datetime(2026, 10, 23, 16, 0, tzinfo=KST)
    assert hd.fetch_isa_close(None, evening, history_fn=lambda: raw)[1] == 171_000.0
    with pytest.raises(ValueError):
        hd.fetch_isa_close(None, morning, history_fn=lambda: pd.DataFrame({"Close": []}))
