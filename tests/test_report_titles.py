"""보고서·브리핑 이름(config.yaml report) 테스트. 네트워크 없음.

- "오빠의 아침 편지 · 10/14(수)" 한 줄 머리말·HTML 제목·첨부 설명 첫 줄·첨부 파일 이름·신호 요약 제목 (2026-10-15)
- 지운 머리말 정보는 맨 아래 「보고서 정보」 접기 안으로
- 시장 온도 판정 기준 목록·머리글의 🟢🟡🔴 이모지 -> CSS 원(.dot)
(단체방용 공개 보고서는 2026-10-15 없앴다 — 단체방도 같은 보고서를 받는다)
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


def _private_html(tmp_path, cfg, **kw):
    shown = {**cfg, "report": {**cfg["report"], "show_screening": True}}  # 예전 화면(스크리닝 켬)을 검증
    return report_html.render_report(_summary(cfg, **kw), shown, tmp_path).read_text(encoding="utf-8")


# ── 1. 이름 (config.yaml report) ──────────────────────────────────────────


def test_config_has_report_titles(titles):
    for key in ("title", "subtitle", "short_title"):
        assert titles[key]
    assert titles["title"] == "오빠의 아침 편지" and titles["short_title"] == "오빠의 아침 편지"
    assert "public_suffix" not in titles


def test_report_titles_falls_back_to_defaults_without_report_section(cfg):
    assert briefing.report_titles({}) == briefing.report_titles(cfg)
    assert briefing.report_titles({"report": {"short_title": "X"}})["short_title"] == "X"


def test_private_briefing_first_line_uses_short_title(cfg, titles):
    text = briefing.build_briefing_text(_summary(cfg), cfg)
    assert text.splitlines()[0] == f"{titles['title']} · 9/30(수)"  # 이모지·"마감"·모드 없음 (2026-10-15)
    assert "나스닥100" not in text


def test_briefing_first_line_follows_changed_config(cfg):
    changed = {**cfg, "report": {"title": "테스트 편지"}}
    assert briefing.build_briefing_text(_summary(cfg), changed).startswith("테스트 편지 · 9/30(수)\n")
    assert briefing.title_line({**_summary(cfg), "mode": "paper"}, cfg) == "오빠의 아침 편지 · 9/30(수) · 모의"


def test_private_report_title_h1_and_subtitle(tmp_path, cfg, titles):
    html = _private_html(tmp_path, cfg)
    assert "<title>오빠의 아침 편지 · 9/30(수)</title>" in html
    header = re.search(r"<header>(.*?)</header>", html, re.S).group(1)
    assert re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", header)).strip() == "오빠의 아침 편지 · 9/30(수)"  # 날짜 한 줄만
    assert "마감" not in header and "📊" not in html.split("<body", 1)[1].split("</header>", 1)[0]
    # 2026-10-31 보고서 다이어트: 「이 편지에 대해」 접기 삭제 — 기준일은 본문 기준 줄, 용어 풀이는 「자세히 보기」 안
    assert 'id="fold-info"' not in html and titles["subtitle"] not in html
    assert "미국 9/30 종가 기준입니다." in html
    assert html.index('id="fold-details"') < html.index('id="guide"') < html.index("<footer>")


def test_signal_summary_md_title_uses_short_title(tmp_path, monkeypatch, cfg, titles):
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    daily._write_outputs(_summary(cfg), cfg)
    md = (tmp_path / "signals_live_2026-09-30.md").read_text(encoding="utf-8")
    assert md.splitlines()[0] == f"# {titles['short_title']} 신호 — 기준일 2026-09-30 (실전)"


# ── 2. 공개용 티커 finviz 링크 ──────────────────────────────────────────────


# ── 3. 오늘의 추천 폰 폭 카드 배치 ─────────────────────────────────────────


# ── 4. 공개용 시장 온도 칸 아래 상태 표시 제거 ──────────────────────────────


def _legacy_html(tmp_path, cfg):
    """시장 지표 타일은 예전 판(letter.show_books true)에만 — 2026-10-29 보고서판은 시장 온도 칸이 대신한다."""
    shown = {**cfg, "letter": {**(cfg.get("letter") or {}), "show_books": True}}
    return report_html.render_report(_summary(cfg), shown, tmp_path).read_text(encoding="utf-8")


def test_private_macro_tile_keeps_status_badge(tmp_path, cfg):
    assert 'class="stb st-' in _legacy_html(tmp_path, cfg)


# ── 5. 🟢🟡🔴 -> CSS 원 ────────────────────────────────────────────────────


def test_macro_scale_and_header_use_css_dots(tmp_path, cfg):
    html = _legacy_html(tmp_path, cfg)
    head = re.search(r'<div class="mhead">.*?</div>', html, re.S).group(0)
    scale = re.search(r'<ol class="mscale".*?</ol>', html, re.S).group(0)
    for part in (head, scale):
        assert not any(sym in part for sym in ("🟢", "🟡", "🔴"))
        for status in ("ok", "warn", "bad"):
            assert f'<i class="dot dot-{status}" aria-hidden="true"></i>' in part
    assert ".dot{display:inline-block;width:7px;height:7px;border-radius:50%" in html
    assert ".dot-ok{background:var(--buy)}.dot-warn{background:var(--warn)}.dot-bad{background:var(--sell)}" in html


# ── 시장 온도 그래프: 공개용 = 개인용 ─────────────────────────────────────


# ── 개인용 안내 문구 (구글 시트) ──────────────────────────────────────────


def test_private_buy_panel_has_plan_sheet_note_above_table(tmp_path, cfg):
    html = _private_html(tmp_path, cfg)
    panel = html.split('<section class="panel on" id="buy">', 1)[1]
    note = "수량·투입금액은 구글 시트 &#39;계획&#39; 탭의 계획금액으로 계산합니다. 계획이 없는 종목은 비워 둡니다."
    note_raw = note.replace("&#39;", "'")
    assert note in panel or note_raw in panel
    idx = panel.find(note) if note in panel else panel.find(note_raw)
    assert idx < panel.find('<div class="sheet">')  # 표 바로 위


def test_private_guide_points_fills_to_google_sheet(tmp_path, cfg):
    html = _private_html(tmp_path, cfg)
    assert "<b>체결 기록</b>구글 시트 '체결' 탭에 체결 내역을 적어요." in html
    assert "fills.xlsx에 체결 내역을 적어요" not in html


def test_private_fills_notice_points_to_google_sheet_fills_tab(tmp_path, cfg):
    html = _private_html(tmp_path, cfg)
    assert "오늘 주문할 종목은 체결 후 구글 시트 '체결' 탭에 한 줄씩 기록하세요." in html
    assert "형식: 날짜, 종목, 매수매도, 수량, 체결가, 차수, 메모" in html  # data/sheets.py 체결 탭 헤더 순서
    assert "fills.xlsx" not in html
