"""텔레그램 = 보고서 파일 하나 + 설명 한 줄, 시장 온도는 보고서에 통합, 중복 정리 (2026-10-29, notify/letter_plain
caption_line·report_sections). 네트워크 없음."""

from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd
import pytest

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
    base = s["macro_rows"][0]
    s["macro_rows"] = [base, {**base, "slug": "dgs10", "name": "미국 10년물 국채금리", "value": 4.12, "unit": "%"},
                       {**base, "slug": "fx", "name": "원/달러 환율", "value": 1325.3, "unit": "원"}]
    s["index_close"] = {"spx": pd.Series([100.0, 101.0], index=pd.to_datetime(["2026-09-29", "2026-09-30"]))}
    return s


@pytest.fixture
def temp(rich):
    from tests.test_thermometer import make_prices
    from thermometer.core import get_temperature

    r = get_temperature("2026-09-30", prices=make_prices(end="2026-09-30"))  # 편지 기준일과 같은 날
    rich["thermometer"] = r
    return r


def _body(summary, cfg, tmp_path, edition="personal"):
    html = report_html.render_report(summary, cfg, tmp_path, edition=edition).read_text(encoding="utf-8")
    return html[html.index("<header>"):html.index("<script>")]


def _sec(body, key):
    s = body[body.index(f'id="sec-{key}"'):]
    nxt = s.find('<div class="rsec"', 1)
    return s[:nxt] if nxt > 0 else s[:s.index("</section>")]


# ── 1. 설명 한 줄 ────────────────────────────────────────────────────────────


def test_caption_line_format_and_flags(rich, temp, cfg):
    t = temp["temp"]
    assert briefing.build_report_caption(rich, cfg) == f"[오빠의 아침 편지] 10/26(월) · 시장 온도 {t}도"
    assert briefing.build_report_caption(rich, cfg, edition="group") == f"[오빠의 아침 편지] 10/26(월) · 시장 온도 {t}도"
    s = copy.deepcopy(rich)
    s["savings_day"] = True  # 저축일 할 일은 개인판만
    assert briefing.build_report_caption(s, cfg) == f"[오빠의 아침 편지] 10/26(월) · 시장 온도 {t}도 · 오늘 할 일 있음"
    assert briefing.build_report_caption(s, cfg, edition="group") == f"[오빠의 아침 편지] 10/26(월) · 시장 온도 {t}도"
    s["fred_missing"] = ["DGS10"]
    assert briefing.build_report_caption(s, cfg).endswith(" · 오늘 할 일 있음 · 자료 문제 있음")
    assert briefing.build_report_caption(s, cfg, edition="group").endswith(f"{t}도 · 자료 문제 있음")
    s["thermometer"] = None  # 온도가 없으면 그 부분만 생략
    assert briefing.build_report_caption(s, cfg) == "[오빠의 아침 편지] 10/26(월) · 오늘 할 일 있음 · 자료 문제 있음"


def test_has_todo_condition(rich, cfg):
    assert not letter_plain.has_todo(rich, cfg) and not letter_plain.has_todo(rich, cfg, "group")
    crash = copy.deepcopy(rich)
    q = np.concatenate([np.linspace(100, 130, 240), np.linspace(130, 90, 20)])
    crash["qqq_close"] = pd.Series(q, index=pd.bdate_range("2025-10-01", periods=260))
    assert letter_plain.has_todo(crash, cfg) and letter_plain.has_todo(crash, cfg, "group")  # 급락장은 두 판 모두
    dec = copy.deepcopy(rich)
    dec["dec_tax"] = {"status": "ok", "remaining_krw": 2_500_000, "deadline": pd.Timestamp("2026-12-29")}
    assert letter_plain.has_todo(dec, cfg) and not letter_plain.has_todo(dec, cfg, "group")


# ── 2. 보고서 칸 순서 ────────────────────────────────────────────────────────


def test_report_order_toc_and_folds(rich, temp, cfg, tmp_path):
    body = _body(rich, cfg, tmp_path)
    ids = re.findall(r'<div class="rsec" id="(sec-[a-z]+)"', body)
    assert ids == ["sec-summary", "sec-thermometer", "sec-news", "sec-todo", "sec-money", "sec-candidates", "sec-lab"]
    nav = body[body.index('<nav class="toc'):body.index("</nav>")]
    assert re.findall(r'href="#(sec-[a-z]+)"', nav) == ids
    folds = re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", body)
    assert folds == ["▶ 용어 풀이", "▶ 이 편지에 대해"]
    assert '<span class="rsub">— 6건</span>' in body and '<span class="rsub">— 2종목</span>' in body
    news = _sec(body, "news")
    heads = re.findall(r"<h3>([^<]+)</h3>", news)
    assert heads == ["■ 오늘의 핵심", "■ 국제정세"] and 'href="https://example.com/9"' in news
    group = _body(rich, cfg, tmp_path, "group")
    assert re.findall(r'<div class="rsec" id="(sec-[a-z]+)"', group) == [x for x in ids if x != "sec-money"]


def test_summary_at_most_five_lines(rich, temp, cfg, tmp_path):
    secs = {s["key"]: s for s in letter_plain.report_sections(rich, cfg)}
    lines = secs["summary"]["lines"]
    assert lines == [f"시장 온도 {temp['temp']}도({temp['zone']}) · QQQ +0.1%", "할 일: 투자 포지션 변경 없습니다.",
                     "내 돈: 합계 280만 원, 1.0% 올랐습니다.", "가상 실험: 웅덩이 v3.2 QQQ보다 0.4%p 앞섭니다 · 오늘 가상 매수 1건."]
    s = copy.deepcopy(rich)
    s["fred_missing"] = ["DGS10", "T10Y2Y"]
    lines = {x["key"]: x for x in letter_plain.report_sections(s, cfg)}["summary"]["lines"]
    assert len(lines) <= 5 and lines[0].startswith("자료 문제: ")
    g = {x["key"]: x for x in letter_plain.report_sections(rich, cfg, "group")}["summary"]["lines"]
    assert not any(x.startswith("내 돈") for x in g)
    rich["thermometer"] = None
    assert {x["key"]: x for x in letter_plain.report_sections(rich, cfg)}["summary"]["lines"][0] == "QQQ +0.1%"


def test_thermometer_section_with_index_rates(rich, temp, cfg, tmp_path):
    sec = _sec(_body(rich, cfg, tmp_path), "thermometer")
    assert "■ 시장 온도" in sec and f"<b>오늘 온도 {temp['temp']}도 · {temp['zone']}" in sec
    for name in ("150일선", "200일선", "시장 폭", "공포 지수 VIX", "공포·탐욕"):
        assert f'<td class="l">{name}</td>' in sec, name
    assert f"▶ {temp['note']}" in sec
    sub = sec[sec.index("<h3>지수·금리·환율</h3>"):]
    assert '<td class="l">나스닥100(QQQ)</td>' in sub and '<td class="l">S&P500</td>' in sub
    assert "▶ 미국 10년 금리 4.1% · 환율 1,325원" in sub
    rich["thermometer"] = None
    none = _sec(_body(rich, cfg, tmp_path), "thermometer")
    assert "오늘 시장 온도를 구하지 못했습니다." in none and "<h3>지수·금리·환율</h3>" in none


# ── 3. 중복 문구 없음 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("edition", ["personal", "group"])
def test_no_duplicate_market_phrases(rich, temp, cfg, tmp_path, edition):
    body = _body(rich, cfg, tmp_path, edition)
    text = re.sub(r"<[^>]+>", " ", body)
    for gone in ("장기 평균선", "투자 심리 ", "시장 온도계 ", "움직이며 조용합니다", "크게 올랐습니다", "힘차게 올랐습니다",
                 "방향을 고르고", "큰 흐름은", "■ 미국 증시 마감", "그래프로 보기"):
        assert gone not in text, gone
    assert text.count("미국 10년 금리") == 1 and text.count("4.1%") == 1  # 같은 숫자는 한 칸만
    assert text.count("1,325") == 1


def test_crash_warning_only_in_todo(rich, temp, cfg, tmp_path):
    q = np.concatenate([np.linspace(100, 130, 240), np.linspace(130, 90, 20)])
    rich["qqq_close"] = pd.Series(q, index=pd.bdate_range("2025-10-01", periods=260))
    for ed in ("personal", "group"):
        body = _body(rich, cfg, tmp_path, ed)
        text = re.sub(r"<[^>]+>", " ", body)
        assert text.count("크게 빠졌습니다") == 1 and "크게 빠졌습니다" in _sec(body, "todo"), ed
        assert "폭락" not in text and "무너졌습니다" not in text
        assert "할 일: 오늘 할 일 칸을 꼭 봐 주십시오." in _sec(body, "summary")


# ── 4. 단체방판 ──────────────────────────────────────────────────────────────


def test_group_has_no_amounts_or_tickers(rich, temp, cfg, tmp_path):
    rich["candidate_charts"] = []
    body = _body(rich, cfg, tmp_path, "group")
    cap = briefing.build_report_caption(rich, cfg, edition="group")
    for x in ("만 원", "280", "NVDA", "AAPL", "MSFT", "내 돈"):
        assert x not in body, x
        assert x not in cap, x
    assert "반등 확인 1종목 · 대기 1종목" in body


# ── 5. 문구 ──────────────────────────────────────────────────────────────────


def test_phrases_yaml_caption_and_retired_keys():
    ph = letter_plain.load_phrases()
    assert ph["caption"] == {"temp": "시장 온도 {T}도", "todo": "오늘 할 일 있음", "problem": "자료 문제 있음"}
    for k in ("temp", "market", "todo", "todo_crash", "money", "lab"):
        assert k in ph["summary"]
    assert "headline" not in ph and "regime" not in ph["market"] and "mood" not in ph["market"]  # 주석 처리(삭제 아님)
    raw = letter_plain.PHRASES_PATH.read_text(encoding="utf-8")
    assert "# headline:" in raw and "#   mood:" in raw and "#   regime:" in raw
