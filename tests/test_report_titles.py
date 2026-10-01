"""보고서·브리핑 이름(config.yaml report), 공개용 보고서 표시 정리 테스트. 네트워크 없음.

- 텔레그램 첫 줄·HTML 제목·부제·첨부 파일 이름·신호 요약 제목이 config.yaml report 값을 쓰는지
- 공개용: 오늘의 추천 티커 finviz 링크, 시장 온도 칸 아래 상태 표시(.stb) 제거, 폰 폭 카드 배치
- 시장 온도 판정 기준 목록·머리글의 🟢🟡🔴 이모지 -> CSS 원(.dot), 개인용·공개용 모두
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from core import macro_status
from engine import daily
from notify import briefing, report_html


def _macro_row(th, **overrides):
    row = {
        "slug": "fear-greed", "name": "공포·탐욕 지수", "code": "CNN Fear & Greed", "value": 38, "unit": "/100",
        "as_of": "2026-09-29", "change_1w": -2.0, "series": [45.0, 40.0, 38.0], "is_stale": False,
        "short_range": False, "note": "지금 구간: 공포",
        "badge": macro_status.classify_macro("FEAR_GREED", 38, th), "ref_values": [],
    }
    row.update(overrides)
    return row


def _buy_row(ticker="NVDA", condition="RSI 27.4 → 31.9 · 거래량 부족(0.62배) · 실적 임박 10/9"):
    return {
        "ticker": ticker, "kr": "엔비디아", "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
        "key": f"{ticker}-A1", "limit": 1234.5, "stop": 1180.05, "stop_pct": -4.4, "decision": "매수",
        "note": condition, "condition_summary": condition, "score": 20, "grade": "",
        "qty": 3, "amount_krw": 800_000, "max_loss_krw": 40_000,
    }


def _summary(cfg, macro_rows=None, b1=None):
    th = cfg["macro"]["thresholds"]
    b1 = [_buy_row()] if b1 is None else b1
    return {
        "mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-30"),
        "buy_groups": {"b1": b1, "b2": [], "b3": [], "b9": []}, "buy_count": len(b1),
        "filtered_rows": [], "sell_rows": [], "warn_rows": [], "data_status_rows": [],
        "unfilled_rows": [], "hold_rows": [], "watch_rows": [], "held_tickers_count": 0,
        "max_concurrent": 8, "macro_rows": [_macro_row(th)] if macro_rows is None else macro_rows,
        "stage_counts": {}, "data_gap_tickers": [], "earnings_unknown_count": 0,
    }


@pytest.fixture
def titles(cfg):
    return cfg["report"]


def _public_html(tmp_path, cfg, **kw):
    return report_html.render_public_report(_summary(cfg, **kw), cfg, tmp_path).read_text(encoding="utf-8")


def _private_html(tmp_path, cfg, **kw):
    return report_html.render_report(_summary(cfg, **kw), cfg, tmp_path).read_text(encoding="utf-8")


# ── 1. 이름 (config.yaml report) ──────────────────────────────────────────


def test_config_has_report_titles(titles):
    for key in ("title", "subtitle", "short_title", "public_suffix"):
        assert titles[key]


def test_report_titles_falls_back_to_defaults_without_report_section(cfg):
    assert briefing.report_titles({}) == briefing.report_titles(cfg)
    assert briefing.report_titles({"report": {"short_title": "X"}})["short_title"] == "X"


def test_private_briefing_first_line_uses_short_title(cfg, titles):
    text = briefing.build_briefing_text(_summary(cfg), cfg)
    assert text.splitlines()[0] == f"📊 {titles['short_title']} · 9/30(수) 마감 · 실전"
    assert "나스닥100" not in text


def test_public_briefing_first_line_uses_short_title_and_public_suffix(cfg, titles):
    text = briefing.build_public_briefing_text(_summary(cfg), cfg)
    assert text.splitlines()[0] == f"📊 {titles['short_title']} · 9/30(수) 마감 · {titles['public_suffix']}"
    assert "공개용" not in text and "나스닥100" not in text


def test_briefing_first_line_follows_changed_config(cfg):
    changed = {**cfg, "report": {"short_title": "테스트브리핑", "public_suffix": "모임"}}
    assert briefing.build_public_briefing_text(_summary(cfg), changed).startswith("📊 테스트브리핑 · 9/30(수) 마감 · 모임\n")


def test_private_report_title_h1_and_subtitle(tmp_path, cfg, titles):
    html = _private_html(tmp_path, cfg)
    assert f"<title>{titles['title']} · 2026-09-30 (실전)</title>" in html
    assert re.search(rf"<h1>{titles['title']}<span class=\"mode-badge\">실전</span></h1>", html)
    assert f'<div class="subtitle">{titles["subtitle"]}</div>' in html
    assert "나스닥 100 일일 브리핑" not in html
    assert "MACD 중심 1:2:6 분할 전략" in html


def test_public_report_title_h1_subtitle_and_suffix(tmp_path, cfg, titles):
    html = _public_html(tmp_path, cfg)
    assert f"<title>{titles['title']} ({titles['public_suffix']}) · 2026-09-30</title>" in html
    assert f'<h1>{titles["title"]} <span class="suffix">({titles["public_suffix"]})</span></h1>' in html
    assert f'<div class="subtitle">{titles["subtitle"]}</div>' in html
    assert "(공개용)" not in html and "나스닥 100 일일 브리핑" not in html
    assert "MACD 중심 1:2:6 분할 전략" in html  # 리드 문구는 그대로


def test_signal_summary_md_title_uses_short_title(tmp_path, monkeypatch, cfg, titles):
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    daily._write_outputs(_summary(cfg), cfg)
    md = (tmp_path / "signals_live_2026-09-30.md").read_text(encoding="utf-8")
    assert md.splitlines()[0] == f"# {titles['short_title']} 신호 — 기준일 2026-09-30 (실전)"


# ── 2. 공개용 티커 finviz 링크 ──────────────────────────────────────────────


def test_public_recommend_ticker_links_to_finviz(tmp_path, cfg):
    html = _public_html(tmp_path, cfg)
    assert '<a class="tk" href="https://finviz.com/quote.ashx?t=NVDA" target="_blank" rel="noopener"' in html
    assert "a.tk{color:inherit;text-decoration:none;border-bottom:1px dotted var(--mute)}" in html


def test_public_report_prices_have_thousands_separator(tmp_path, cfg):
    html = _public_html(tmp_path, cfg)
    assert "$1,234.50" in html and "$1,180.05 (-4.4%)" in html


# ── 3. 오늘의 추천 폰 폭 카드 배치 ─────────────────────────────────────────


def test_public_mobile_cards_use_grid_not_table_cells(tmp_path, cfg):
    """폰 폭에서 tr만 block이고 td가 table-cell로 남으면 칸이 옆으로 끼여 깨진다 (회귀 방지)."""
    html = _public_html(tmp_path, cfg)
    mobile = html.split("@media (max-width:640px){\n  .wrap", 1)[1].split("\n}\n", 1)[0]
    tr_rule = re.search(r"\n  tr\{(.*?)\}", mobile, re.S).group(1)
    assert "display:grid" in tr_rule and "display:block" not in tr_rule
    assert "td.note{grid-column:1/-1;min-width:0" in mobile
    assert "td.note:empty{display:none}" in mobile


# ── 4. 공개용 시장 온도 칸 아래 상태 표시 제거 ──────────────────────────────


def test_public_macro_tile_has_no_status_badge_but_keeps_notes(tmp_path, cfg):
    th = cfg["macro"]["thresholds"]
    rows = [_macro_row(th, is_stale=True, short_range=True)]
    html = _public_html(tmp_path, cfg, macro_rows=rows)
    assert 'class="stb' not in html
    assert '<span class="mr">지연(2026-09-29) · 기간 짧음 · 지금 구간: 공포</span>' in html
    badge = rows[0]["badge"]["text"]
    assert re.search(rf"<h3>.*— 오늘 38 {badge}</h3>", html)  # 설명 섹션 제목의 상태 표시는 그대로


def test_public_macro_tile_without_notes_omits_note_line(tmp_path, cfg):
    html = _public_html(tmp_path, cfg, macro_rows=[_macro_row(cfg["macro"]["thresholds"], note=None)])
    assert 'class="mr"' not in html


def test_private_macro_tile_keeps_status_badge(tmp_path, cfg):
    assert 'class="stb st-' in _private_html(tmp_path, cfg)


# ── 5. 🟢🟡🔴 -> CSS 원 ────────────────────────────────────────────────────


@pytest.mark.parametrize("render", [_public_html, _private_html], ids=["public", "private"])
def test_macro_scale_and_header_use_css_dots(tmp_path, cfg, render):
    html = render(tmp_path, cfg)
    head = re.search(r'<div class="mhead">.*?</div>', html, re.S).group(0)
    scale = re.search(r'<ol class="mscale".*?</ol>', html, re.S).group(0)
    for part in (head, scale):
        assert not any(sym in part for sym in ("🟢", "🟡", "🔴"))
        for status in ("ok", "warn", "bad"):
            assert f'<i class="dot dot-{status}" aria-hidden="true"></i>' in part
    assert ".dot{display:inline-block;width:7px;height:7px;border-radius:50%" in html
    assert ".dot-ok{background:var(--buy)}.dot-warn{background:var(--warn)}.dot-bad{background:var(--sell)}" in html


def test_telegram_macro_lines_keep_emoji(cfg):
    text = briefing.build_public_briefing_text(_summary(cfg), cfg)
    assert "🟡주의" in text
