"""아침 보고서 다이어트 (2026-10-31, notify/letter_plain.report_view + report.html.j2) — 보여 주는 것만, 네트워크 없음.

본문: (자료 문제) → ■ 오늘 할 일 → ■ 내 돈(개인판) → ■ 시장 → ■ 오늘의 뉴스(상위 3) → 조건부 줄 → 기준 줄 → 접기 「자세히 보기」 하나.
"""

from __future__ import annotations

import copy
import html as htmllib
import re

import numpy as np
import pandas as pd
import pytest

from notify import briefing, letter_plain, report_html
from tests.test_letter_summary import rich, temp  # noqa: F401 — 픽스처 재사용
from thermometer.core import load_config
from thermometer.fear_board import board

FEAR_LINE = "공포 구간입니다. 여윳돈이 있으면 추가 매수해도 좋은 구간입니다. 가진 주식은 팔지 않습니다."
CALM = {"fear_greed": 50, "rsi": 50, "close": 100, "bb_lower": 90, "sma_long": 95, "sma_short": 99, "sma_short_lag": 98, "vix": 18}


@pytest.fixture
def wk(rich, temp):  # noqa: F811 — 평일(화요일) 편지
    s = copy.deepcopy(rich)
    s["letter_today_kst"] = pd.Timestamp("2026-10-27")
    s["thermometer"] = temp
    return s


def _html(s, cfg, tmp_path, edition="personal"):
    html = report_html.render_report(s, cfg, tmp_path, edition=edition).read_text(encoding="utf-8")
    return html[html.index("<header>"):html.index("<script>")]


def _main(body):
    """본문(접기 앞 편지 칸) — <section class="letter plain"> … </section>."""
    i = body.index('<section class="letter plain"')
    return body[i:body.index("</section>", i)]


def _lines(main):
    """본문 글 줄(▶ 줄 + 기준 줄, 칸 제목 제외)."""
    return [htmllib.unescape(re.sub(r"<[^>]+>", "", m)).strip() for m in re.findall(r'<div class="ln[^"]*">(.*?)</div>', main, re.S)]


def _fold(body):
    i = body.index('id="fold-details"')
    return body[i:body.index("</details>\n", body.index('id="guide"', i))]


def _fb(n, day="2026-09-30"):
    vals = dict(CALM)
    for upd in [{"vix": 40}, {"rsi": 20}, {"fear_greed": 10}, {"bb_lower": 101}, {"sma_long": 100, "sma_short": 97}][:n]:
        vals.update(upd)
    return {"date": day, "data_date": day, **board(vals, load_config())}


# ── 1. 평일 기본 칸 ─────────────────────────────────────────────────────────


def test_weekday_personal_body_exact_and_short(wk, temp, cfg, tmp_path):
    body = _html(wk, cfg, tmp_path)
    main = _main(body)
    assert re.findall(r'<div class="rsec" id="(sec-[a-z]+)"', main) == ["sec-todo", "sec-money", "sec-market", "sec-news"]
    assert re.findall(r"<h2>([^<]+)</h2>", main) == ["■ 오늘 할 일", "■ 내 돈", "■ 시장", "■ 오늘의 뉴스"]
    t = temp
    d = t["delta"]
    change = "어제와 같음" if d == 0 else f"어제보다 {abs(d)}도 {'↑' if d > 0 else '↓'}"
    lines = _lines(main)
    assert lines[:5] == ["▶ 투자 포지션 변경 없습니다.",
                         "▶ 합계 280만 원, 1.0% 올랐습니다.", "▶ 세금 없이 팔 수 있는 이익 250만 원 남았습니다.",
                         f"▶ 시장 온도 {t['temp']}도 · {t['zone']} ({change})",
                         "▶ QQQ +0.1% · 미국 10년 금리 4.1% · 환율 1,325원"]
    assert lines[-1] == "미국 9/30 종가 기준입니다."
    assert len(lines) <= 12 and len(lines) == 9  # 할 일 1 + 내 돈 2 + 시장 2 + 뉴스 3 + 기준 1 (칸 제목·머리줄 제외)
    for gone in ("오늘 요약", "바로가기", "nav class=\"toc", "이 편지에 대해", "■ 가상 실험", "■ 웅덩이 후보", "국제정세"):
        assert gone not in body, gone
    assert re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", body) == ["▶ 자세히 보기"]
    assert body.index("종가 기준입니다.") < body.index('id="fold-details"')


def test_money_section_omitted_when_empty(wk, cfg, tmp_path):
    wk["my_money"], wk["wallet"] = None, {"qqqm_state": "empty", "isa_state": "empty"}
    main = _main(_html(wk, cfg, tmp_path))
    assert "■ 내 돈" not in main and 'id="sec-money"' not in main
    assert letter_plain.money_main_lines(wk, letter_plain.load_phrases()) == []


def test_market_line_without_temperature_or_numbers(wk, cfg):
    ph = letter_plain.load_phrases()
    wk["thermometer"] = None
    wk["macro_rows"] = [r for r in wk["macro_rows"] if r["slug"] != "dgs10"]
    assert letter_plain.market_main_lines(wk, cfg, ph) == ["오늘 시장 온도를 구하지 못했습니다.", "QQQ +0.1% · 환율 1,325원"]


# ── 2. 조건부 줄 ────────────────────────────────────────────────────────────


def test_savings_day_personal_only(wk, cfg, tmp_path):
    wk["savings_day"] = True
    p, g = _lines(_main(_html(wk, cfg, tmp_path))), _lines(_main(_html(wk, cfg, tmp_path, "group")))
    assert p[0] == "▶ ISA에서 TIGER 미국나스닥100을 삽니다. 한도를 넘는 돈은 QQQM을 삽니다."
    assert g[0] == "▶ 투자 포지션 변경 없습니다."
    assert briefing.build_report_caption(wk, cfg).endswith(" · 오늘 할 일 있음")
    assert "오늘 할 일 있음" not in briefing.build_report_caption(wk, cfg, edition="group")


def test_crash_line_in_todo_both_editions(wk, cfg, tmp_path):
    q = np.concatenate([np.linspace(100, 130, 240), np.linspace(130, 90, 20)])
    wk["qqq_close"] = pd.Series(q, index=pd.bdate_range("2025-10-01", periods=260))
    for ed in ("personal", "group"):
        main = _main(_html(wk, cfg, tmp_path, ed))
        todo = main[main.index('id="sec-todo"'):main.index('id="sec-market"' if ed == "group" else 'id="sec-money"')]
        assert "▶ 시장이 크게 빠졌습니다. 팔지 않습니다. 적립만 이어갑니다." in todo, ed
        assert main.count("크게 빠졌습니다") == 1


@pytest.mark.parametrize("n,shown", [(0, False), (1, False), (2, False), (3, True), (5, True)])
def test_fear_line_only_in_fear_zone(wk, cfg, tmp_path, n, shown):
    wk["fear_board"] = _fb(n)
    for ed in ("personal", "group"):
        main = _main(_html(wk, cfg, tmp_path, ed))
        market = main[main.index('id="sec-market"'):main.index('id="sec-news"')]
        assert (f"▶ 공포 매수판 {n}/5 · {FEAR_LINE}" in market) is shown, (n, ed)
        assert "눌림 구간입니다" not in main and "적립만 그대로입니다" not in main
        assert briefing.build_report_caption(wk, cfg, edition=ed).endswith(f" · 공포 매수판 {n}/5") is shown


def test_virtual_trade_lines_only_on_trade_days(wk, cfg, tmp_path):
    main = _main(_html(wk, cfg, tmp_path))
    assert "가상 실험:" not in main and 'id="sec-extras"' not in main
    wk["puddle_v31"] = {**wk["puddle_v31"], "trades_personal": ["NVDA를 처음 샀습니다."], "trades_group": ["오늘 가상으로 1종목을 샀습니다."]}
    wk["puddle_rebound"] = {**wk["puddle_rebound"], "trades_personal": ["AAPL을 처음 샀습니다.", "META 웅덩이가 메워져 팔았습니다."],
                            "trades_group": ["오늘 가상으로 1종목을 샀습니다.", "오늘 가상으로 1종목을 팔았습니다."]}
    p = _lines(_main(_html(wk, cfg, tmp_path)))
    assert "▶ 가상 실험: 웅덩이 v3.2 NVDA를 처음 샀습니다(실제 돈 아님)." in p
    assert "▶ 가상 실험: 웅덩이 반등 확인 v2 AAPL을 처음 샀습니다. META 웅덩이가 메워져 팔았습니다(실제 돈 아님)." in p
    g = _lines(_main(_html(wk, cfg, tmp_path, "group")))
    assert "▶ 가상 실험: 웅덩이 v3.2 오늘 가상으로 1종목을 샀습니다(실제 돈 아님)." in g
    assert not any(t in x for x in g for t in ("NVDA", "AAPL", "META"))


def test_monday_line_chart_and_research(wk, cfg, tmp_path):
    from tests.test_curve_chart import START, fake_history
    from notify import curve_chart as cc

    v, r = fake_history(30)
    out = cc.build(v, r, tmp_path, start=START, as_of=v[-1]["date"])
    wk["curve_chart"] = {"personal": str(out["personal"]["path"]), "group": str(out["group"]["path"])}
    wk["puddle_rebound"] = {**wk["puddle_rebound"], "vs_qqq_pp": -0.26}
    tue = _html(wk, cfg, tmp_path)
    assert "QQQ 대비" not in _main(tue) and 'aria-label="가상 실험 성적 그래프"' in _fold(tue)  # 평일 그래프는 접기 안
    assert "시험 결과는" not in _main(tue)
    wk["letter_today_kst"] = pd.Timestamp("2026-10-26")  # 월요일
    mon = _html(wk, cfg, tmp_path)
    main = _main(mon)
    lines = _lines(main)
    assert "▶ 가상 실험: 웅덩이 v3.2 QQQ 대비 +0.4%p · 반등 확인 v2 -0.3%p" in lines
    assert "▶ 「변동성 맞춤 레버리지 전략」 시험 결과는 탈락입니다." in lines  # 연구 소식 한 줄
    assert 'aria-label="가상 실험 성적 그래프"' in main and mon.count('aria-label="가상 실험 성적 그래프"') == 1
    assert len(lines) <= 12


def test_problem_line_on_top(wk, cfg, tmp_path):
    wk["fred_missing"] = ["DGS10"]
    main = _main(_html(wk, cfg, tmp_path))
    assert _lines(main)[0] == "▶ 자료 문제: 시장 지표 1개를 받지 못했습니다."
    assert main.index("자료 문제") < main.index('id="sec-todo"')


# ── 3. 뉴스 3개 ─────────────────────────────────────────────────────────────


def test_news_top_three_only(wk, cfg, tmp_path):
    main = _main(_html(wk, cfg, tmp_path))
    news = main[main.index('id="sec-news"'):]
    links = re.findall(r'href="(https://example\.com/\d+)"', news)
    assert links == ["https://example.com/0", "https://example.com/1", "https://example.com/2"]  # 점수 상위 3(top 순서)
    assert "(언론0)" in news and "<h3>" not in news
    wk["news"] = {"status": "fail", "top": [], "topics": {}, "failed": ["x"]}
    assert "▶ 뉴스를 불러오지 못했습니다." in _main(_html(wk, cfg, tmp_path))


# ── 4. 접기 「자세히 보기」 ─────────────────────────────────────────────────


def test_details_fold_keeps_all_detail(wk, temp, cfg, tmp_path):
    wk["fear_board"] = _fb(2)
    body = _html(wk, cfg, tmp_path)
    fold = _fold(body)
    for name in ("150일선", "200일선", "시장 폭", "공포 지수 VIX", "공포·탐욕"):
        assert f'<td class="l">{name}</td>' in fold, name  # 시장 온도 항목 5개 표
    assert f"<b>오늘 온도 {temp['temp']}도 · {temp['zone']}" in fold
    assert "<h3>공포 매수판 · 2/5 · 눌림</h3>" in fold and fold.count("✓") == 2 and fold.count("✗") == 3
    assert "▶ 눌림 구간입니다. 저축일에 그대로 삽니다." in fold  # 눌림은 본문엔 없고 접기 안 표에만
    assert '<td class="l">S&P500</td>' in fold and "미국 10년 금리" not in fold  # 금리·환율은 본문에만(같은 숫자 한 번)
    assert "웅덩이 후보 (가상 실험 신호, 실제 매수 권유 아님)" in fold and "▶ 반등 확인 — 오늘 가상 매수: AAPL" in fold
    assert "※ 웅덩이 후보는 12개월 시험 중인 가상 전략의 신호입니다." in fold
    assert "가상 실험 (실제 돈 아님)" in fold and "▶ 웅덩이 v3.2: QQQ보다 0.4%p 앞섭니다." in fold
    assert 'href="#guide"' in fold and 'id="guide"' in fold and "용어 풀이" in fold
    assert "※ 뉴스는 제목 모음입니다. 매매 신호가 아닙니다." in fold
    assert not re.search(r"<details class=\"fold\"[^>]*\bopen\b", body)


def test_details_fold_candidate_charts_personal_only(wk, cfg, tmp_path):
    png = tmp_path / "c.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")  # 내용은 보지 않음(data URI로만 넣는다)
    wk["candidate_charts"] = [{"path": str(png), "title": "AAPL 반등 확인", "tk": "AAPL"}]
    assert "cand-chart" in _fold(_html(wk, cfg, tmp_path))
    assert "cand-chart" not in _html(wk, cfg, tmp_path, "group")


# ── 5. 단체방판 ─────────────────────────────────────────────────────────────


def test_group_has_no_amounts_or_tickers_anywhere(wk, cfg, tmp_path):
    wk["fear_board"] = _fb(4)
    wk["savings_day"] = True
    wk["dec_tax"] = {"status": "ok", "remaining_krw": 2_500_000, "deadline": pd.Timestamp("2026-12-29"), "lines": ["250만 원"]}
    wk["puddle_v31"] = {**wk["puddle_v31"], "trades_personal": ["NVDA를 처음 샀습니다."], "trades_group": ["오늘 가상으로 1종목을 샀습니다."]}
    body = _html(wk, cfg, tmp_path, "group")
    cap = briefing.build_report_caption(wk, cfg, edition="group")
    for x in ("만 원", "280", "250", "NVDA", "AAPL", "MSFT", "내 돈", "비과세", "세금 없이"):
        assert x not in body, x
        assert x not in cap, x
