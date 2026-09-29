"""시장 온도 칸이 보고서 HTML·텔레그램 글에 제대로 반영되는지 (P3.8). 네트워크 없음."""

from __future__ import annotations

import pandas as pd
import pytest

from notify import briefing, report_html

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
    "macro": {
        "enabled": True,
        "thresholds": {
            "FEAR_GREED": {"extreme_fear": 25, "fear": 45, "neutral_high": 55, "greed": 75},
            "VIXCLS": {"caution": 20, "alert": 30}, "DGS10": {"rise_3m_caution": 0.5},
            "T10Y2Y": {"normal": 0.5, "alert": 0}, "BAMLH0A0HYM2": {"caution": 4, "alert": 6},
            "DEXKOUS": {"pct_high": 80, "pct_low": 20},
        },
    },
}


def _empty_summary(macro_rows=None):
    return {
        "mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23"),
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []}, "buy_count": 0,
        "filtered_rows": [], "sell_rows": [], "warn_rows": [], "data_status_rows": [],
        "unfilled_rows": [], "hold_rows": [], "watch_rows": [], "held_tickers_count": 0,
        "max_concurrent": 8, "macro_rows": macro_rows or [],
    }


_SAMPLE_MACRO_ROWS = [
    {
        "name": "공포·탐욕 지수", "code": "CNN Fear & Greed", "value": 58, "unit": "/100",
        "as_of": "2026-09-22", "change_1w": 6, "series": [40.0, 45.0, 50.0, 55.0, 58.0],
        "badge": {"status": "warn", "symbol": "▲", "label": "탐욕", "text": "▲ 탐욕"},
        "is_stale": False, "short_range": False, "note": "25 이하 극단적 공포 · 75 이상 극단적 탐욕",
        "ref_values": [25, 75],
    },
    {
        "name": "미국 10년물 국채금리", "code": "DGS10", "value": 4.12, "unit": "%",
        "as_of": "2026-09-22", "change_1w": -0.08, "series": [4.0, 4.05, 4.10, 4.12],
        "badge": {"status": "ok", "symbol": "●", "label": "안정", "text": "● 안정"},
        "is_stale": False, "short_range": False, "note": None, "ref_values": [],
    },
]


def test_build_context_includes_macro_rows_with_sparkline():
    ctx = report_html.build_context(_empty_summary(_SAMPLE_MACRO_ROWS), _CFG)
    rows = ctx["macro_rows"]
    assert len(rows) == 2
    assert rows[0]["spark_points"] != ""
    assert rows[0]["value_str"] == "58"
    assert rows[0]["ref_ys"] == [round(v, 1) for v in rows[0]["ref_ys"]]  # 좌표 계산됨(예외 없음)
    assert rows[1]["change_str"] == "-0.08"


def test_render_report_with_macro_rows_contains_macro_section(tmp_path):
    path = report_html.render_report(_empty_summary(_SAMPLE_MACRO_ROWS), _CFG, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'class="macro"' in html
    assert "공포·탐욕 지수" in html
    assert "시장 온도 지표 6가지" in html  # 읽는 법 탭


def test_render_report_without_macro_rows_hides_section(tmp_path):
    path = report_html.render_report(_empty_summary([]), _CFG, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'aria-label="시장 온도"' not in html


def test_render_report_guide_table_uses_config_thresholds(tmp_path):
    path = report_html.render_report(_empty_summary([]), _CFG, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "0~24 극단적 공포" in html
    assert "25~44 공포" in html


# ── 텔레그램 "시장 온도" 줄 ──────────────────────────────────────────────────


def test_build_macro_line_formats_all_six_and_flags_warn():
    rows = [
        {"name": "공포·탐욕 지수", "value": 58, "badge": {"status": "warn", "label": "탐욕"}},
        {"name": "미국 10년물 국채금리", "value": 4.12, "unit": "%", "badge": {"status": "ok", "label": "안정"}},
        {"name": "장단기 금리차 (10년−2년)", "value": 0.48, "badge": {"status": "ok", "label": "정상"}},
        {"name": "하이일드 스프레드", "value": 3.05, "unit": "%", "badge": {"status": "ok", "label": "안정"}},
        {"name": "미국 기준금리 (상단)", "value": 4.50, "badge": {"status": "info", "label": "인하 흐름"}},
        {"name": "원/달러 환율", "value": 1380, "badge": {"status": "warn", "label": "달러 비쌈"}},
    ]
    line = briefing.build_macro_line(rows)
    assert line.startswith("시장 온도: ")
    assert "⚠공포·탐욕 58 탐욕" in line  # warn 상태는 ⚠ 접두
    assert "10년물 4.12% ·" in line  # ok 상태는 접두 없음
    assert "금리차 +0.48" in line
    assert "기준금리 4.5% 인하" in line
    assert "⚠환율 1,380 달러 비쌈" in line


def test_build_macro_line_none_when_no_rows():
    assert briefing.build_macro_line([]) is None


def test_briefing_text_includes_market_temp_group_near_top():
    summary = _empty_summary(_SAMPLE_MACRO_ROWS)
    text = briefing.build_briefing_text(summary, _CFG)
    lines = text.splitlines()
    market_idx = lines.index("📈 시장 온도")
    assert market_idx <= 2  # 헤더 줄 바로 다음(빈 줄 하나 포함)
    assert lines[market_idx + 1].startswith("- ")
