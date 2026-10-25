"""아침 편지 v3 (notify/letter_plain.py + config/letter_phrases.yaml) — 표시 전용, 네트워크 없음."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from notify import letter_plain as lp

PH = lp.load_phrases()


def _summary(**kw):
    q = pd.Series(np.linspace(100, 130, 260), index=pd.bdate_range("2025-10-01", periods=260))
    q.iloc[-1] = q.iloc[-2] * 1.003
    s = {"as_of": pd.Timestamp("2026-10-23"), "mode": "live", "qqq_close": q, "letter_today_kst": pd.Timestamp("2026-10-24"),
         "letter_check": {"lag": 0, "notes": []},
         "macro_rows": [{"slug": "fear-greed", "value": 28, "badge": {"zone": "공포", "label": "주의"}},
                        {"slug": "fed", "value": 4.5, "badge": {"label": "위험"}},
                        {"slug": "dgs10", "value": 4.12, "badge": {"label": "주의"}}, {"slug": "fx", "value": 1325.3, "badge": {}}],
         "news": {"status": "ok", "top": [{"title": "연준 금리 동결", "source": "연합뉴스", "url": "https://example.com/a"}],
                  "topics": {"국제정세": [{"title": "관세 협상 재개", "source": "KBS", "url": "https://example.com/b"}]}}}
    s.update(kw)
    return s


CFG = {"report": {"title": "오빠의 아침 편지"}, "research_news": {}, "savings_day": 17}


def test_number_words_and_josa():
    assert lp.manwon(12_345_678) == "1,235만 원" and lp.man(4_000) == "0"
    assert lp.signed(1.14) == "+1.1" and lp.signed(-0.44) == "-0.4" and lp.signed(0.01) == "0.0"
    assert lp.vs_qqq_parts(0.44) == ("ahead", "0.4") and lp.vs_qqq_parts(-1.26) == ("behind", "1.3") and lp.vs_qqq_parts(0)[0] == "same"
    assert lp.fill("{종목}을 처음 샀습니다.", 종목="AMD") == "AMD를 처음 샀습니다."
    assert lp.fill("{종목}을 처음 샀습니다.", 종목="AAPL") == "AAPL을 처음 샀습니다."
    assert lp.fill("{종목}이 밀려", 종목="AMZN") == "AMZN이 밀려"


def test_phrase_rotation_differs_from_previous_day():
    ph = {"x": ["가입니다.", "나입니다."]}
    d = pd.Timestamp("2026-10-24")
    assert lp.phrase(ph, "x", d) != lp.phrase(ph, "x", d - pd.Timedelta(days=1))
    big = _summary()
    big["qqq_close"] = big["qqq_close"].copy()
    big["qqq_close"].iloc[-1] = big["qqq_close"].iloc[-2] * 1.03
    a = lp.headline(big, CFG, PH, lp.PERSONAL)
    b = lp.headline({**big, "letter_today_kst": pd.Timestamp("2026-10-23")}, CFG, PH, lp.PERSONAL)
    assert a[1] == b[1] == "big_up" and a[0] != b[0]


def test_section_order_and_format():
    d = lp.build(_summary(my_money={"value_krw": 1e7, "pnl_pct": 5.0, "remaining_krw": 2.5e6},
                          puddle_v31={"personal": ["웅덩이 v3.1: QQQ와 같습니다."], "group": ["웅덩이 v3.1: QQQ와 같습니다."]}),
                 CFG, lp.PERSONAL)
    heads = [s["header"] for s in d["sections"]]
    assert heads == ["■ 오늘의 핵심", "■ 국제정세", "■ 미국 증시 마감", "■ 오늘 할 일", "■ 내 돈", "■ 가상 실험 (실제 돈 아님)"]
    t = lp.to_text(d)
    lines = t.split("\n")
    assert lines[0] == "[오빠의 아침 편지] 10/24(토)"
    assert all(not x.startswith((" ", "\t")) for x in lines)  # 들여쓰기 없음
    body = [x for x in lines[1:] if x and not x.startswith("■")]
    numbers = ("▶ 나스닥100(QQQ) ", "▶ 미국 10년 금리 ")  # 숫자 줄(문장 아님)
    assert all(x.endswith("다.") or x.endswith(")") or x.startswith(numbers) for x in body), body  # 경어체(뉴스 줄은 "(언론사)")
    assert all(x.startswith("▶ ") for x in body[:-2])
    assert lines[-2] == "미국 10/23 종가 기준입니다." and lines[-1] == "※ 뉴스는 제목 모음입니다. 매매 신호가 아닙니다."
    assert "▶ 연준 금리 동결 (연합뉴스)" in lines
    assert not re.search(r"[\U0001F300-\U0001FAFF]", t)


def test_market_section():
    m = lp.market_lines(_summary(index_close={"spx": pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-10-22", "2026-10-23"]))}),
                        CFG, PH, lp.PERSONAL)
    assert m[0] == "시장이 +0.3% 움직이며 조용합니다."
    assert m[1] == "나스닥100(QQQ) +0.3% · S&P500 +1.0%"  # SOXX 못 받으면 그 부분만 빠진다
    assert m[2] == "미국 10년 금리 4.1% · 환율 1,325원"
    assert m[3].startswith("장기 평균선보다 ") and m[3].endswith("큰 흐름은 상승입니다.")
    assert m[4] == "투자 심리 28. 조금 겁먹은 분위기입니다." and m[5] == "기준금리 인상은 조심해서 봅니다."


def test_todo_savings_moved_dec_tax_and_crash():
    s = _summary(savings_day=True, letter_today_kst=pd.Timestamp("2026-10-19"),
                 dec_tax={"status": "ok", "remaining_krw": 1_234_567, "deadline": pd.Timestamp("2026-12-29")})
    p = lp.todo_lines(s, CFG, PH, lp.PERSONAL)
    assert p[0] == "ISA에서 TIGER 미국나스닥100을 삽니다. 한도를 넘는 돈은 QQQM을 삽니다."
    assert p[1] == "원래 저축일은 10/17(토)였습니다. ISA는 그날 사도 됩니다."
    assert p[2] == "세금 없이 팔 수 있는 이익이 123만 원 남았습니다. 12/29(화)까지 QQQM을 팔고 바로 다시 삽니다."
    assert lp.todo_lines(s, CFG, PH, lp.GROUP) == ["투자 포지션 변경 없습니다."]
    crash = _summary(qqq_close=pd.Series(np.linspace(130, 80, 260), index=pd.bdate_range("2025-10-01", periods=260)))
    assert lp.todo_lines(crash, CFG, PH, lp.GROUP) == ["시장이 크게 빠졌습니다. 팔지 않습니다. 적립만 이어갑니다."]
    assert lp.headline(crash, CFG, PH, lp.GROUP)[1] == "crash"


def test_money_lines_and_empty():
    assert lp.money_lines({}, CFG, PH) == ["시트 노란 칸을 채우면 표시됩니다."]
    s = {"my_money": {"value_krw": 12_345_678, "pnl_pct": 5.2, "remaining_krw": 2_500_000},
         "wallet": {"qqqm_state": "ok", "isa": {"value_krw": 8_560_000, "pnl_pct": -3.1, "tax_free_left_krw": 4_000_000},
                    "isa_state": "ok"}}
    assert lp.money_lines(s, CFG, PH) == ["QQQM 1,235만 원, 5.2% 올랐습니다.", "ISA 856만 원, 3.1% 내렸습니다.", "합계 2,091만 원입니다.",
                                          "세금 없이 팔 수 있는 이익 250만 원, ISA 비과세 한도 400만 원 남았습니다."]


def test_problem_line_under_title():
    ok = lp.build(_summary(), CFG, lp.PERSONAL)
    assert ok["problem"] is None
    lag = _summary(letter_check={"lag": 2, "expected": pd.Timestamp("2026-10-27"), "notes": []},
                   puddle_v31={"personal": [], "group": [], "check_personal": "실적 확인 실패(AMD)", "check_group": "실적 확인 실패(1종목)"})
    p, g = lp.build(lag, CFG, lp.PERSONAL), lp.build(lag, CFG, lp.GROUP)
    assert p["problem"].startswith("▶ 자료 문제: 미국 자료가 2일 늦었습니다. 오늘 시장과 다를 수 있습니다.")
    assert "AMD" in p["problem"] and "AMD" not in g["problem"]
    assert lp.to_text(p).split("\n")[1] == p["problem"]


def test_news_failure_line():
    d = lp.build(_summary(news={"status": "fail", "top": [], "topics": {}}), CFG, lp.GROUP)
    assert d["sections"][0] == {"header": "■ 오늘의 핵심", "lines": ["뉴스를 불러오지 못했습니다."]}
    assert "▶ 뉴스를 불러오지 못했습니다." in lp.to_text(d)


def test_group_has_no_money_or_tickers():
    s = _summary(my_money={"value_krw": 12_345_678, "pnl_pct": 5.2, "remaining_krw": 2_500_000})
    g = lp.to_text(lp.build(s, CFG, lp.GROUP))
    assert "■ 내 돈" not in g and "1,235만" not in g and "만 원" not in g


def test_length_limit_drops_topic_news_first():
    s = _summary()
    many = [{"title": "가" * 58, "source": "연합뉴스", "url": ""}] * 3
    s["news"] = {"status": "ok", "top": [{"title": "나" * 58, "source": "KBS", "url": ""}] * 10,
                 "topics": {"국제정세": many, "전쟁·지정학": many, "미국 경제·금리": many}}
    d = lp.build(s, CFG, lp.PERSONAL)
    full = lp.to_text(d, limit=100_000)
    assert "■ 미국 경제·금리" in full
    t = lp.to_text(d, limit=len(full) - 10)
    assert len(t) <= len(full) - 10 and "■ 미국 경제·금리" not in t and "■ 국제정세" in t
    assert "▶ 나머지는 첨부 파일에 있습니다." in t and "■ 오늘 할 일" in t and t.endswith("매매 신호가 아닙니다.")
    assert len(lp.to_text(d)) <= lp.MESSAGE_LIMIT
