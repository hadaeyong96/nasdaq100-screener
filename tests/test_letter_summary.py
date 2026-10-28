"""텔레그램 문자 = 요약 쪽지, 첨부 HTML = 전부 (2026-10-28, notify/letter_plain summary_text·report_sections)."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

BS = chr(10)  # 줄바꿈

from notify import briefing, letter_plain, report_html


@pytest.fixture
def rich(cfg, tmp_path, monkeypatch):
    from tests.test_fresh_start import _summary

    s = _summary(copy.deepcopy(cfg))
    long = "미국 연준 의장이 금리 인하 속도를 늦출 수 있다고 말해 시장이 크게 흔들렸다는 소식과 함께 전한 이야기"
    s["news"] = {"status": "ok",
                 "top": [{"title": f"{long} {i}", "source": f"언론{i}", "url": f"https://example.com/{i}", "score": 9 - i} for i in range(5)],
                 "topics": {"국제정세": [{"title": "유럽 정상회의 결과", "source": "언론9", "url": "https://example.com/9", "score": 1}]}}
    s["my_money"] = {"qty": 8.0, "close_usd": 250.0, "value_krw": 2_800_000.0, "cost_krw": 2_772_277.0, "pnl_krw": 27_723.0,
                     "pnl_pct": 1.0, "realized_krw": 0.0, "remaining_krw": 2_500_000.0}
    s["wallet"] = {"qqqm_state": "ok", "isa_state": "empty"}
    s["puddle_v31"] = {"personal": ["웅덩이 v3.2: QQQ보다 0.4%p 앞섭니다. NVDA 10% 담고 있습니다. NVDA를 처음 샀습니다."],
                       "group": ["웅덩이 v3.2: QQQ보다 0.4%p 앞섭니다. 1종목 10% 담고 있습니다. 오늘 가상으로 1종목을 샀습니다."],
                       "vs_qqq_pp": 0.42, "buys_today": 1, "sells_today": 0}
    s["puddle_rebound"] = {"personal": ["웅덩이 반등 확인 v2: QQQ와 같습니다. QQQ만 들고 있습니다."],
                           "group": ["웅덩이 반등 확인 v2: QQQ와 같습니다. QQQ만 들고 있습니다."],
                           "cand_personal": ["반등 확인 — 오늘 가상 매수: AAPL (RSI 41, MACD 골든크로스)", "반등 확인 대기: MSFT (웅덩이 3일째, RSI 35)"],
                           "cand_group": ["반등 확인 1종목 · 대기 1종목 · 진입 0종목"], "cand_count": 2}
    s["fred_missing"] = ["DGS10", "T10Y2Y"]
    return s


def test_text_within_12_lines_700_chars(rich, cfg):
    for ed in ("personal", "group"):
        t = briefing.letter_message(rich, cfg, edition=ed)
        assert len(t) <= 700 and len(t.split("\n")) <= 12, (ed, len(t))
        assert t.split("\n")[0] == "[오빠의 아침 편지] 10/26(월)"
        assert t.split("\n")[1].startswith("▶ 자료 문제: ")
        assert t.split("\n")[-1] == "자세한 내용은 첨부 보고서에 있습니다(뉴스 6건 · 웅덩이 후보 2종목)."
        assert all(not ln.startswith(" ") for ln in t.split("\n"))


def test_text_has_exactly_three_news(rich, cfg):
    t = briefing.letter_message(rich, cfg)
    news = [ln for ln in t.split("\n") if ln.startswith("▶ 핵심 뉴스: ")]
    assert len(news) == 3
    assert [re.search(r"\((언론\d)\)$", ln).group(1) for ln in news] == ["언론0", "언론1", "언론2"]  # 점수 상위 3개


def test_text_line_order_and_lab(rich, cfg):
    t = briefing.letter_message(rich, cfg).split("\n")
    heads = [ln.split(":")[0] for ln in t[2:-1]]
    assert heads == ["▶ 시장", "▶ 할 일", "▶ 내 돈", "▶ 핵심 뉴스", "▶ 핵심 뉴스", "▶ 핵심 뉴스", "▶ 가상 실험"]
    assert "▶ 내 돈: 합계 280만 원, 1.0% 올랐습니다." in t
    assert "▶ 가상 실험: 웅덩이 v3.2 QQQ보다 0.4%p 앞섭니다 · 오늘 가상 매수 1건." in t


def test_todo_line_by_situation(rich, cfg):
    def todo(s, ed="personal"):
        return next(ln for ln in briefing.letter_message(s, cfg, edition=ed).split("\n") if ln.startswith("▶ 할 일: "))

    assert todo(rich) == "▶ 할 일: 투자 포지션 변경 없습니다."
    s = copy.deepcopy(rich)
    s["savings_day"] = True
    assert todo(s) == "▶ 할 일: ISA에서 TIGER 미국나스닥100을 삽니다."
    assert todo(s, "group") == "▶ 할 일: 투자 포지션 변경 없습니다."  # 저축일은 개인판만
    s = copy.deepcopy(rich)
    s["dec_tax"] = {"status": "ok", "remaining_krw": 2_500_000, "deadline": pd.Timestamp("2026-12-29")}
    assert todo(s) == "▶ 할 일: 세금 없이 팔 수 있는 이익이 250만 원 남았습니다."
    s["savings_day"] = True
    assert todo(s).startswith("▶ 할 일: ISA에서")  # 저축일이 12월 공제보다 먼저
    s = copy.deepcopy(rich)
    q = np.concatenate([np.linspace(100, 130, 240), np.linspace(130, 90, 20)])
    s["qqq_close"] = pd.Series(q, index=pd.bdate_range("2025-10-01", periods=260))
    s["savings_day"] = True
    assert todo(s) == "▶ 할 일: 시장이 크게 빠졌습니다." and todo(s, "group") == "▶ 할 일: 시장이 크게 빠졌습니다."  # 급락장이 맨 먼저


def test_group_has_no_amounts_or_tickers(rich, cfg, tmp_path):
    rich["candidate_charts"] = []
    t = briefing.letter_message(rich, cfg, edition="group")
    html = report_html.render_report(rich, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    for x in ("만 원", "280", "NVDA", "AAPL", "MSFT", "내 돈"):
        assert x not in t, x
        assert x not in body, x
    assert "반등 확인 1종목 · 대기 1종목" in body
    p = briefing.letter_message(rich, cfg)
    assert "280만 원" in p and "NVDA" not in p  # 개인판 문자도 종목명은 첨부 보고서에만


def test_html_order_toc_and_subs(rich, cfg, tmp_path):
    html = report_html.render_report(rich, cfg, tmp_path).read_text(encoding="utf-8")
    body = html[html.index("<header>"):html.index("<script>")]
    ids = re.findall(r'<div class="rsec" id="(sec-[a-z]+)"', body)
    assert ids == ["sec-summary", "sec-news", "sec-market", "sec-thermometer", "sec-todo", "sec-money", "sec-candidates", "sec-lab"]
    nav = body[body.index('<nav class="toc'):body.index("</nav>")]
    assert re.findall(r'href="#(sec-[a-z]+)"', nav) == ids
    folds = re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", body)
    assert folds == ["▶ 그래프로 보기", "▶ 용어 풀이", "▶ 이 편지에 대해"]
    assert body.index('id="sec-lab"') < body.index("▶ 그래프로 보기")
    assert '<span class="rsub">— 6건</span>' in body and '<span class="rsub">— 2종목</span>' in body
    assert 'href="https://example.com/9"' in body  # 뉴스 전체(주제 칸까지) 링크
    assert "<td class=\"l\">나스닥100(QQQ)</td>" in body  # 지수 표
    assert "오늘 온도계 기록이 아직 없습니다." in body
    group = report_html.render_report(rich, cfg, tmp_path, edition="group").read_text(encoding="utf-8")
    assert "sec-money" not in group


def test_thermometer_in_text_and_html(rich, cfg, tmp_path):
    from thermometer.core import get_temperature
    from tests.test_thermometer import make_prices

    r = get_temperature("2026-09-30", prices=make_prices(end="2026-09-30"))  # 편지 기준일과 같은 날
    rich["thermometer"] = r
    t = briefing.letter_message(rich, cfg).split(BS)
    assert t[2] == f"▶ 시장: QQQ +0.1% · 시장 온도 {r['temp']}도({r['zone']})"
    html = report_html.render_report(rich, cfg, tmp_path).read_text(encoding="utf-8")
    sec = html[html.index('id="sec-thermometer"'):html.index('id="sec-todo"')]
    assert f"— {r['temp']}도 · {r['zone']}" in sec and "150일선" in sec and f"<b>{r['temp']}도</b>" in sec
    rich["thermometer"] = None
    assert briefing.letter_message(rich, cfg).split(BS)[2] == "▶ 시장: QQQ +0.1%"


def test_more_line_omits_zero_candidates(rich, cfg):
    rich["puddle_rebound"]["cand_count"] = 0
    rich["puddle_rebound"]["cand_personal"] = ["오늘은 웅덩이 후보가 없습니다."]
    rich["puddle_rebound"]["cand_group"] = ["오늘은 웅덩이 후보가 없습니다."]
    assert briefing.letter_message(rich, cfg).split(BS)[-1] == "자세한 내용은 첨부 보고서에 있습니다(뉴스 6건)."


def test_money_line_omitted_when_empty(rich, cfg):
    rich["my_money"], rich["wallet"] = None, {}
    assert "내 돈" not in briefing.letter_message(rich, cfg)


def test_summary_phrases_in_yaml():
    ph = letter_plain.load_phrases()
    assert ph["summary"]["max_lines"] == 12 and ph["summary"]["max_chars"] == 700
    for k in ("market", "temp", "todo", "money", "news", "lab", "more"):
        assert k in ph["summary"]
