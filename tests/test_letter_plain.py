"""아침 편지 쉬운 문장판 (notify/letter_plain.py) — 표시 전용, 네트워크 없음."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from notify import letter_plain as lp


def test_number_words():
    assert lp.manwon(12_345_678) == "1,235만 원" and lp.manwon(4_000) == "0만 원"
    assert lp.updown(1.14) == "1.1% 올랐어요" and lp.updown(-0.44) == "0.4% 내렸어요" and lp.updown(0.01) == "그대로예요"
    assert lp.vs_qqq(0.44) == "QQQ보다 0.4%p 앞서요" and lp.vs_qqq(-1.26) == "QQQ보다 1.3%p 뒤처져요" and lp.vs_qqq(0.0) == "QQQ와 같아요"


def _summary(**kw):
    q = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    s = {"as_of": pd.Timestamp("2026-10-23"), "mode": "live", "qqq_close": q, "letter_today_kst": pd.Timestamp("2026-10-24"),
         "letter_check": {"lag": 0, "notes": []},
         "macro_rows": [{"slug": "fear-greed", "value": 28, "badge": {"zone": "공포", "label": "주의"}},
                        {"slug": "dgs10", "value": 5.12, "badge": {"label": "위험"}}, {"slug": "fx", "value": 1325.3, "badge": {}}]}
    s.update(kw)
    return s


CFG = {"report": {"title": "오빠의 아침 편지"}, "research_news": {}}


def test_market_lines_plain():
    lines = lp.market_lines(_summary())
    assert lines[0].startswith("나스닥100(QQQ)은 어제 ") and "일주일 동안" in lines[0]
    assert lines[1].startswith("장기 평균선(200일)보다 ") and lines[1].endswith("큰 흐름은 아직 상승장이에요.")
    assert lines[2] == "투자 심리는 '공포' 쪽이에요(28)."
    assert lines[3] == "미국 10년 금리 5.1%, 환율 1,325원이에요."
    assert lines[4] == "조심할 신호도 있어요: 10년 금리 5.1%."
    assert all("200일선" not in x and "%p" not in x for x in lines)


def test_todo_savings_dec_tax_and_crash():
    s = _summary(savings_day=True, dec_tax={"status": "ok", "remaining_krw": 1_234_567, "deadline": pd.Timestamp("2026-12-29")})
    p = lp.todo_lines(s, CFG, lp.PERSONAL)
    assert p[:3] == lp.SAVINGS and p[3] == "올해 세금 없이 팔 수 있는 이익이 123만 원 남았어요." and p[4] == "12/29(화)에 정리하세요."
    assert lp.todo_lines(s, CFG, lp.GROUP) == [lp.TODO_NONE]  # 저축일·12월 공제는 개인판만
    crash = _summary(qqq_close=pd.Series(np.linspace(130, 80, 260), index=pd.bdate_range("2025-10-01", periods=260)))
    assert lp.todo_lines(crash, CFG, lp.GROUP) == [lp.TODO_NONE] + lp.CRASH


def test_money_lines_and_empty():
    assert lp.money_lines({}, CFG) == [lp.MONEY_EMPTY]
    s = {"my_money": {"value_krw": 12_345_678, "pnl_pct": 5.2, "remaining_krw": 2_500_000},
         "wallet": {"qqqm_state": "ok", "isa": {"value_krw": 8_560_000, "pnl_pct": 3.1, "tax_free_left_krw": 4_000_000},
                    "isa_state": "ok"}}
    assert lp.money_lines(s, CFG) == ["QQQM 1,235만 원, 5.2% 올랐어요.", "ISA 856만 원, 3.1% 올랐어요.", "합계 2,091만 원이에요.",
                                      "세금 없이 팔 수 있는 이익은 250만 원, ISA 비과세 한도는 400만 원 남았어요."]
    s["wallet"]["isa_state"], s["wallet"]["isa"] = "no_price", None
    assert lp.money_lines(s, CFG)[1] == "ISA는 오늘 가격을 못 받았어요."


def test_problem_line_on_top_and_footer_at_bottom():
    ok = lp.build(_summary(), CFG, lp.PERSONAL)
    assert ok["problem"] is None and ok["footer"] == "기준: 10/23 미국 종가 · 자료 정상"
    lag = _summary(letter_check={"lag": 2, "expected": pd.Timestamp("2026-10-27"), "notes": []},
                   puddle_v31={"personal": [], "group": [], "check_personal": "실적 확인 실패(AMD)", "check_group": "실적 확인 실패(1종목)"})
    p, g = lp.build(lag, CFG, lp.PERSONAL), lp.build(lag, CFG, lp.GROUP)
    assert p["problem"] == "자료 문제: 미국 자료가 2거래일 늦어요(기대 10/27(화)) · 실적 발표 확인을 못 한 종목이 있어요(AMD)."
    assert "AMD" not in g["problem"] and "(1종목)" in g["problem"] and p["footer"].endswith("자료 문제 있음")
    text = lp.to_text(p).split("\n")
    assert text[1].startswith("자료 문제:") and text[-1].startswith("기준:")


def test_group_has_no_money_section_and_caption_limit():
    s = _summary(my_money={"value_krw": 12_345_678, "pnl_pct": 5.2, "remaining_krw": 2_500_000})
    g = lp.to_text(lp.build(s, CFG, lp.GROUP))
    assert "[내 돈]" not in g and "1,235만" not in g
    big = copy.deepcopy(s)
    big["puddle_v31"] = {"personal": ["가" * 100] * 12, "group": []}
    t = lp.to_text(lp.build(big, CFG, lp.PERSONAL))
    assert len(t) <= lp.CAPTION_LIMIT and lp.CAPTION_MORE in t and t.endswith("자료 정상")
