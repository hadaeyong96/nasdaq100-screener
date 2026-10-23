"""아침 편지 v2.1 (2026-10-21) — 점검 줄·내 돈·가상 장부 3줄 묶음·연구 소식(판정 다음 날·월요일)·단체방판 비노출. 네트워크 없음."""

from __future__ import annotations

import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from core import tax_harvest as th
from engine import daily
from engine.daily import KST, load_config
from notify import briefing, letter, report_html
from tests.test_letter import BASE, _books, _summary
from tests.legacy import legacy_cfg


@pytest.fixture
def cfg():
    c = legacy_cfg(load_config())  # 가상 장부 칸이 있던 편지(새 출발 전) — 새 구조는 tests/test_fresh_start.py
    c["virtual_books"]["penny"]["enabled"] = False  # 1~9번 기준 문구 시험 — 10번은 tests/test_penny_book.py
    return c


def _info(books, *, lag=0, skipped=None, base=BASE, expected=BASE):
    return letter.check_info(books, base, expected, lag, skipped)


# ── 점검 줄 4경우 ────────────────────────────────────────────────────────────


def test_check_line_normal_with_waiting(cfg):
    line = letter.check_line(_info(_books(cfg)))
    assert line == "점검: 기준일 10/16(금) · 장부 8개 기록, 1개 시작 대기(가치평가 전략) · 데이터 지연 없음"


def test_check_line_all_recorded_no_waiting(cfg):
    books = [b for b in _books(cfg) if b["status"] != "waiting"]
    line = letter.check_line(_info(books))
    assert line == "점검: 기준일 10/16(금) · 장부 8개 기록 · 데이터 지연 없음"  # 시작 대기·건너뜀이 없으면 그 말도 없음


def test_check_line_lag(cfg):
    line = letter.check_line(_info(_books(cfg), lag=2, expected=pd.Timestamp("2026-10-20")))
    assert line.endswith("데이터 지연 2거래일(기대 10/20(화))")
    assert "데이터 지연 없음" not in line


def test_check_line_skipped_with_reason(cfg):
    books = _books(cfg, waiting=False)  # 8번이 시작 대기도 아니고 계산도 안 됨 → missing
    msgs = ["가치평가 따라하기: SEC_USER_AGENT가 없어 오늘은 건너뜁니다(GitHub Secret 추가 필요, 넣으면 시작일부터 따라잡음)"]
    line = letter.check_line(_info(books, skipped=letter.skip_reasons(cfg, msgs)))
    assert line.endswith("건너뜀(가치평가 전략·SEC 키 없음)")
    assert letter.skip_reasons(cfg, ["가치평가 따라하기: 시작일 2026-10-19 전이라 계산하지 않았습니다"]) == {}  # 시작 전은 건너뜀 아님
    assert letter.skip_reasons(cfg, ["한 달 가속 따라하기 오류: KeyError: 'x'"]) == {"QQQ 대비 1개월 강세 전략": "오류"}
    # 계산은 된 장부의 경고(점수 못 낸 종목 등)는 건너뜀이 아니다
    ok_books = [b for b in _books(cfg) if b["status"] != "waiting"]
    warn = letter.skip_reasons(cfg, ["한 달 가속 따라하기: 2026-09-30 점수 못 낸 종목 1개 — X"])
    assert "건너뜀" not in letter.check_line(_info(ok_books, skipped=warn))


# ── 내 돈 ────────────────────────────────────────────────────────────────────


def test_money_empty_message_when_no_fills():
    assert letter.money_lines(None) == ["QQQM 모으기 탭에 보유 주식 수·평균단가를 넣으면 표시됩니다"]
    assert letter.money_summary(th.moving_average_position(pd.DataFrame()), 250.0, 1400.0, 2026) is None


def test_money_lines_from_fills():
    fills = pd.DataFrame([
        {"date": "2026-01-05", "ticker": "QQQM", "side": "buy", "qty": 10, "price": 200.0, "fx_rate": 1400.0},
        {"date": "2026-06-01", "ticker": "QQQM", "side": "sell", "qty": 2, "price": 230.0, "fx_rate": 1400.0},
    ])
    pos = th.moving_average_position(fills)
    m = letter.money_summary(pos, 250.0, 1400.0, 2026)
    # 8주 × 250 × 1400 = 2,800,000 / 취득가 8 × 200 × 1400 = 2,240,000 / 실현 2 × 30 × 1400 = 84,000
    assert m["value_krw"] == pytest.approx(2_800_000) and m["pnl_krw"] == pytest.approx(560_000)
    assert m["realized_krw"] == pytest.approx(84_000) and m["remaining_krw"] == pytest.approx(2_416_000)
    assert letter.money_lines(m) == ["QQQM 8주 · 평가 2,800,000원 · 손익 +560,000원(+25.0%)",
                                     "올해 실현 이익 84,000원 · 12월 공제 남은 금액 2,416,000원"]


# ── 가상 장부 3줄 묶음 ────────────────────────────────────────────────────────


def test_book_lines_three_groups_vs_qqq_pp(cfg):
    books = _books(cfg)
    lines = letter.book_lines(books, cfg)
    assert lines[0] == "기준선: QQQ 전략 +1.0% (아래 %p의 기준)"
    assert lines[1] == "레버리지(1~5): QLD 전략 +1.0%p · TQQQ 전략 +2.0%p · QLD 추세 전략 +1.0%p · TQQQ 추세 전략 +2.0%p"
    assert lines[2].startswith("종목 고르기(6~10): 기술지표 전략 ") and "가치평가 전략 시작 대기" in lines[2]
    assert lines[3] == "오늘 거래 없음"


def test_book_lines_trades_count_personal_detail_folded_group_no_names(cfg):
    trades = [{"side": "매수", "ticker": "MU", "fill_date": BASE}]
    books = _books(cfg, accel_positions={"MU": {}}, accel_trades=trades)
    p = letter.book_lines(books, cfg)
    g = letter.book_lines(books, cfg, letter.GROUP)
    assert p[-1] == "오늘 가상 거래 1건 (상세는 아래 접기)" and "MU" not in "".join(p)
    assert letter.trade_detail_line(books) == "오늘 가상 매수: QQQ 대비 1개월 강세 전략 MU"
    assert g[-1] == "오늘 가상 거래 1건" and g[0] == "기준선: QQQ 전략 (아래 %p의 기준)"  # 단체방판: 기준선 수익률도 빼고 %p만


# ── 연구 소식 (판정 다음 첫 편지 + 월요일) ─────────────────────────────────────────


def test_research_news_shown_recorded_only_when_live(cfg, tmp_path):
    from store import virtual_books as vbstore

    db = tmp_path / "vb.db"
    vbstore.connect(db).close()
    now = datetime(2026, 10, 22, 8, 10, tzinfo=KST)
    s = {"as_of": BASE}
    daily._attach_letter_v2(s, {"base_date": BASE, "db_path": db, "dry_run": True}, cfg, fills_fn=lambda: pd.DataFrame(),
                            now_fn=lambda: now)
    assert s["research_news"][0].startswith("시험 끝")
    daily._attach_letter_v2(s, {"base_date": BASE, "db_path": db, "dry_run": True}, cfg, fills_fn=lambda: pd.DataFrame(),
                            now_fn=lambda: now)
    assert s["research_news"][0].startswith("시험 끝")  # dry-run은 기록하지 않아 다시 보인다
    daily._attach_letter_v2(s, {"base_date": BASE, "db_path": db, "dry_run": False}, cfg, fills_fn=lambda: pd.DataFrame(),
                            now_fn=lambda: now)
    assert s["research_news"][0].startswith("시험 끝")
    daily._attach_letter_v2(s, {"base_date": BASE, "db_path": db, "dry_run": False}, cfg, fills_fn=lambda: pd.DataFrame(),
                            now_fn=lambda: now)
    assert s["research_news"] == []  # 실제 실행에서 한 번 보였으니 다음 편지(같은 평일)엔 없음


# ── 보고서 순서·단체방판 ──────────────────────────────────────────────────────


def _full_summary(cfg, monday=True):
    s = _summary(cfg)
    s["letter_check"] = _info(s["vbooks"]["books"])
    s["my_money"] = None
    s["letter_today_kst"] = pd.Timestamp("2026-10-26" if monday else "2026-10-27")
    s["research_news"] = letter.research_news(cfg, s["letter_today_kst"], ["상승장 RSI 눌림 전략"])[0]
    return s


def test_personal_order_full(cfg, tmp_path):
    s = _full_summary(cfg)
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    body = html[html.index("<header>"):]
    keys = ("<header>", "점검: 기준일", "오늘 할 일:", "[1 내 돈 · QQQM · ISA]", "[2 시장 한눈에]", "[3 가상 장부]", "[4 연구 소식]",
            '<details class="fold"')
    i = [body.index(x) for x in keys]
    assert i == sorted(i)
    assert "QQQM: 노란 칸을 채우면 표시됩니다" in body and "ISA: 노란 칸을 채우면 표시됩니다" in body
    tue = report_html.render_report(_full_summary(cfg, monday=False), cfg, tmp_path).read_text(encoding="utf-8")
    assert "연구 소식" not in tue


def test_group_excludes_check_money_keeps_research(cfg, tmp_path):
    s = _full_summary(cfg)
    s["my_money"] = {"qty": 8.0, "close_usd": 250.0, "value_krw": 2_800_000.0, "cost_krw": 1.0, "pnl_krw": 1.0, "pnl_pct": 1.0,
                     "realized_krw": 0.0, "remaining_krw": 2_500_000.0}
    html = report_html.render_report(s, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    for x in ("점검:", "내 돈", "QQQM 8주", "[4 연구", "2,800,000"):
        assert x not in body, x
    assert "[가상 장부]" in body and "%p" in body
    assert "[연구 소식]" in body and "시험 끝: 「상승장 RSI 눌림 전략」" in body  # 연구 소식은 단체방판에도


def test_caption_check_line_personal_only(cfg):
    s = _full_summary(cfg)
    s["as_of"] = BASE
    p = briefing.build_report_caption(s, cfg).split("\n")
    assert p[1].startswith("점검: 기준일 10/16(금)") and p[2].startswith("오늘 할 일:")
    g = briefing.build_report_caption(s, cfg, edition="group")
    assert "점검" not in g


def test_no_momentum_waiting_text(cfg, tmp_path):
    cfg["virtual_books"]["momentum"]["enabled"] = False
    w = daily._waiting_books(cfg, BASE, set())
    assert list(w) == ["가치평가 따라하기"]
    s = _full_summary(cfg)
    html = report_html.render_report(s, cfg, tmp_path).read_text(encoding="utf-8")
    assert "6개월 강세" not in html and "가치평가 전략" in html


# ── 엔진 연결 (주입, 네트워크 없음) ────────────────────────────────────────────


def test_attach_letter_v2_lag_and_money(cfg):
    s = {"as_of": BASE, "vbooks": {"books": _books(cfg)}, "fx_rate": 1400.0}
    vb = {"base_date": BASE, "warnings": [], "errors": []}
    now = datetime(2026, 10, 21, 7, 0, tzinfo=KST)  # 수요일 07:00 KST → 마지막 마감 거래일 10/20(화)
    daily._attach_letter_v2(s, vb, cfg, fills_fn=lambda: pd.DataFrame(), now_fn=lambda: now)
    info = s["letter_check"]
    assert info["lag"] == 2 and info["expected"] == pd.Timestamp("2026-10-20")  # 10/19·10/20 빠짐
    assert s["my_money"] is None and s["letter_today_kst"] == pd.Timestamp("2026-10-21")
    fills = pd.DataFrame([{"date": "2026-10-01", "ticker": "QQQM", "side": "buy", "qty": 4, "price": 240.0, "fx_rate": 1400.0}])
    close = pd.Series([250.0, 251.0, 999.0], index=pd.to_datetime(["2026-10-15", "2026-10-16", "2026-10-19"]))
    daily._attach_letter_v2(s, {**vb, "base_date": pd.Timestamp("2026-10-20")}, cfg, fills_fn=lambda: fills,
                            qqqm_fn=lambda: close, now_fn=lambda: now)
    assert s["letter_check"]["lag"] == 0
    assert s["my_money"]["close_usd"] == 251.0  # 브리핑 기준일(10/16)까지 종가만


def test_attach_letter_v2_isolates_errors(cfg):
    s = {"as_of": BASE}

    def boom():
        raise RuntimeError("시트 없음")

    daily._attach_letter_v2(s, None, cfg, fills_fn=boom, now_fn=lambda: datetime(2026, 10, 19, 7, 0, tzinfo=KST))
    assert s["my_money"] is None and "letter_check" in s and s["letter_check"]["recorded"] == []


# ── 이름 변경 (2026-10-21): 9번 표시 이름만 바뀌고 기존 열 값은 이어진다 ──────────────────────


def test_rename_9_header_keeps_history(cfg):
    from core.virtual_books import BookSpec, LedgerResult
    from notify import virtual_books as vbtext

    assert vbtext.display_name(cfg, "한 달 가속 따라하기") == "QQQ 대비 1개월 강세 전략"
    idx = pd.to_datetime(["2026-10-14", "2026-10-15", "2026-10-16", "2026-10-19"])
    q = pd.DataFrame({"value": [100.0, 101, 102, 103], "cum_return_pct": [0.0, 1, 2, 3], "drawdown_pct": [0.0] * 4,
                      "holding": ["QQQ"] * 4}, index=idx)
    regime = pd.DataFrame({"close": [1.0] * 4, "sma": [1.0] * 4, "above": [1.0] * 4}, index=idx)
    led = LedgerResult(start=idx[0], base_date=idx[-1], books={"QQQ 그냥 보유": type("B", (), {"daily": q, "trades": []})()},
                       summaries=[], regime=regime, switches=[])
    acc = pd.DataFrame({"value": [70.0, 71, 72], "cum_return_pct": [0.0, 1.4, 2.9], "drawdown_pct": [0.0] * 3},
                       index=idx[1:])
    old_cfg = copy.deepcopy(cfg)
    old_cfg["virtual_books"]["display_names"]["한 달 가속 따라하기"] = "한 달 가속 전략"
    raw = {"전략비교_일일": vbtext.daily_table(led, [BookSpec("QQQ 그냥 보유", "hold", "QQQ")], "QQQ 그냥 보유",
                                         [{"name": "한 달 가속 따라하기", "start": idx[1], "daily": acc}])}
    old = vbtext.rename_tables(raw, vbtext.display_map(old_cfg))["전략비교_일일"]
    new = vbtext.rename_tables(raw, vbtext.display_map(cfg))["전략비교_일일"]
    io, inew = old[0].index("한 달 가속 전략 누적(%)"), new[0].index("QQQ 대비 1개월 강세 전략 누적(%)")
    assert io == inew and [r[io] for r in old[1:]] == [r[inew] for r in new[1:]] == ["", 0.0, 1.4, 2.9]
    # 복구: 이름을 바꾸기 전 시트(옛 머리글)도, 바꾼 뒤 시트도 같은 장부 id로 되돌린다
    back_old = vbtext.unrename_tables({"전략비교_일일": old}, vbtext.display_map(cfg), vbtext.display_aliases(cfg))
    back_new = vbtext.unrename_tables({"전략비교_일일": new}, vbtext.display_map(cfg), vbtext.display_aliases(cfg))
    assert back_old == back_new == raw
