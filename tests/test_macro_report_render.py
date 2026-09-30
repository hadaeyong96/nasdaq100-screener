"""시장 온도 칸이 보고서 HTML·텔레그램 글에 제대로 반영되는지 (P3.8). 네트워크 없음."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import yaml

from core import macro_status
from notify import briefing, report_html

with open(Path(__file__).resolve().parents[1] / "config.yaml", encoding="utf-8") as _f:
    _TH = yaml.safe_load(_f)["macro"]["thresholds"]  # 기준값은 실제 config.yaml 그대로

_CFG = {
    "account": {"total_krw": 100_000_000},
    "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
    "risk": {
        "a1_budget_pct": 0.2222222222222222, "a2_budget_pct": 0.4444444444444444,
        "a3_budget_pct": 1.3333333333333333, "b_budget_pct": 1.0,
        "gap_buffer_pct": 1.5, "min_risk_per_share_pct": 1.0, "max_position_pct": 25, "max_concurrent_positions": 8,
    },
    "indicators": {
        "ichimoku_shift": 26, "rsi": {"period": 14}, "macd": {"fast": 12, "slow": 26, "signal": 9},
        "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
    },
    "assumptions": {
        "a1_to_a2_expiry_days": 10, "reentry_cooldown_days": 5, "gap_filter_pct": 4.0,
        "whipsaw_max_crosses_20d": 4, "swing_low_period": 10, "s_grade_macd_norm_min_pct": -0.5,
        "b_grade_macd_norm_max_pct": -2.0,
    },
    "entry": {"limit_markup": 1.01},
    "macro": {"enabled": True, "thresholds": _TH},
}


def _empty_summary(macro_rows=None):
    return {
        "mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23"),
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []}, "buy_count": 0,
        "filtered_rows": [], "sell_rows": [], "warn_rows": [], "data_status_rows": [],
        "unfilled_rows": [], "hold_rows": [], "watch_rows": [], "held_tickers_count": 0,
        "max_concurrent": 8, "macro_rows": macro_rows or [],
    }


def _row(slug, name, code, fcode, value, unit, series, note=None, value_before=None):
    return {
        "slug": slug, "name": name, "code": code, "value": value, "unit": unit, "as_of": "2026-09-29",
        "change_1w": -0.08, "series": series, "is_stale": False, "short_range": False, "note": note,
        "badge": macro_status.classify_macro(fcode, value, _TH, value_before=value_before),
        "ref_values": [],
    }


# engine.daily._build_macro_rows가 만드는 6칸(공포·탐욕 + FRED 5개)과 VIX 대체 칸
_SAMPLE_MACRO_ROWS = [
    _row("fear-greed", "공포·탐욕 지수", "CNN Fear & Greed", "FEAR_GREED", 38, "/100", [40.0, 45.0, 50.0, 38.0], "지금 구간: 공포"),
    _row("dgs10", "미국 10년물 국채금리", "DGS10", "DGS10", 5.17, "%", [4.0, 4.5, 5.0, 5.17], "3개월 변화 +0.30%p (보조 설명)"),
    _row("t10y2y", "장단기 금리차 (10년−2년)", "T10Y2Y", "T10Y2Y", 0.52, "%p", [0.1, 0.3, 0.52]),
    _row("hy", "하이일드 스프레드", "BAMLH0A0HYM2", "BAMLH0A0HYM2", 3.05, "%", [3.2, 3.1, 3.05]),
    _row("fed", "미국 기준금리 (상단)", "DFEDTARU", "DFEDTARU", 4.25, "%", [4.5, 4.25], value_before=4.5),
    _row("fx", "원/달러 환율", "DEXKOUS · KRW=X", "DEXKOUS", 1392.0, "원", [1350.0, 1380.0, 1392.0]),
]
_VIX_ROW = _row("vix", "변동성 지수 VIX (공포·탐욕 대체)", "VIXCLS", "VIXCLS", 31.2, "", [15.0, 25.0, 31.2])


def test_build_context_includes_macro_rows_with_sparkline_scale_and_explain():
    ctx = report_html.build_context(_empty_summary(_SAMPLE_MACRO_ROWS), _CFG)
    rows = ctx["macro_rows"]
    assert len(rows) == 6
    assert rows[0]["spark_points"] != ""
    assert rows[0]["value_str"] == "38"
    assert rows[1]["value_str"] == "5.17"
    assert rows[2]["value_str"] == "+0.52"
    assert rows[5]["value_str"] == "1,392"
    assert rows[1]["change_str"] == "-0.08"
    dgs10 = rows[1]
    assert [r["current"] for r in dgs10["scale"]] == [False, False, True]
    assert dgs10["explain"]["now"].startswith("오늘 값은 5.17%로, 기준표의 🔴 위험 칸(4.5% 이상)에 있어요.")
    assert "3개월 변화 +0.30%p" in dgs10["explain"]["now"]


def _render(tmp_path, rows):
    return report_html.render_report(_empty_summary(rows), _CFG, tmp_path).read_text(encoding="utf-8")


@pytest.mark.parametrize("rows", [_SAMPLE_MACRO_ROWS, [_VIX_ROW] + _SAMPLE_MACRO_ROWS[1:]], ids=["cnn", "vix-fallback"])
def test_every_indicator_has_anchor_links_star_scale_and_explanation(tmp_path, rows):
    html = _render(tmp_path, rows)
    for row in rows:
        slug = row["slug"]
        tile = re.search(rf'<div class="mt" id="macro-{slug}".*?</div>\s*</div>\s*</div>', html, re.S)
        assert tile, f"{slug} 칸이 없음"
        tile_html = tile.group(0)
        assert f'href="#explain-{slug}"' in tile_html  # 제목·그래프 -> 설명
        scale = re.search(r'<ol class="mscale".*?</ol>', tile_html, re.S).group(0)
        assert scale.count("<li") == 3
        assert scale.count("★") == 1  # 오늘 칸에만 ★
        assert "🟢 안정" in scale and "🟡 주의" in scale and "🔴 위험" in scale
        star_li = re.search(r'<li class="cur">(.*?)</li>', scale, re.S).group(1)
        assert row["badge"]["text"] in star_li
        article = re.search(rf'<article id="explain-{slug}">.*?</article>', html, re.S)
        assert article, f"{slug} 설명 없음"
        body = article.group(0)
        for head in ("① 이게 뭔가요", "② 주식시장에 왜 영향을 주나요", "③ 실제 예시", "④ 지금 수치는 어떻게 읽나요"):
            assert head in body
        assert f'href="#macro-{slug}">↑ 그래프로 돌아가기' in body
    assert "이 기준은 참고용이며 매수·매도 신호가 아닙니다." in html
    assert 'href="http' not in html.split('id="macro-explain"')[1].split("</section>")[0]  # 설명은 외부 링크 없음


def test_fear_greed_shows_zone_name(tmp_path):
    html = _render(tmp_path, _SAMPLE_MACRO_ROWS)
    assert "🟡 주의 · 공포" in html
    assert "0~24 (극단적 공포) · 76~100 (극단적 탐욕)" in html


def test_render_report_without_macro_rows_hides_section(tmp_path):
    html = _render(tmp_path, [])
    assert 'aria-label="시장 온도"' not in html
    assert 'id="macro-explain"' not in html


def test_render_report_guide_table_uses_config_thresholds(tmp_path):
    html = _render(tmp_path, [])
    assert "<td>4.0% 미만</td><td>4.0~4.5%</td><td>4.5% 이상</td>" in html
    assert "<td>1,300원 미만</td>" in html
    assert "<td>최근 6개월 인하</td>" in html


# ── 텔레그램 "시장 온도" 줄 ──────────────────────────────────────────────────


def test_telegram_macro_items_use_same_three_level_judgment():
    items = [briefing._macro_item_text(r) for r in _SAMPLE_MACRO_ROWS + [_VIX_ROW]]
    assert items == [
        "공포·탐욕 38(공포) 🟡주의",
        "10년물 5.17% 🔴위험",
        "금리차 +0.52%p 🟢안정",
        "HY 스프레드 3.05% 🟢안정",
        "기준금리 4.25%(인하) 🟢안정",
        "환율 1,392원 🟡주의",
        "VIX 31.2 🔴위험",
    ]


def test_build_macro_line_none_when_no_rows():
    assert briefing.build_macro_line([]) is None


def test_briefing_text_includes_market_temp_group_near_top():
    summary = _empty_summary(_SAMPLE_MACRO_ROWS)
    text = briefing.build_briefing_text(summary, _CFG)
    lines = text.splitlines()
    market_idx = lines.index("📈 시장 온도")
    assert market_idx <= 2  # 헤더 줄 바로 다음(빈 줄 하나 포함)
    assert lines[market_idx + 1] == "- 공포·탐욕 38(공포) 🟡주의"
    assert "- 10년물 5.17% 🔴위험" in lines
