"""notify/report_html.py 렌더링 테스트. 네트워크 없음 (템플릿 렌더링만)."""

from __future__ import annotations

import pandas as pd
import pytest

from notify import report_html


def _empty_summary(mode="live"):
    return {
        "mode": mode,
        "mode_label": "실전" if mode == "live" else "모의",
        "as_of": pd.Timestamp("2026-09-23"),
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []},
        "buy_count": 0,
        "filtered_rows": [],
        "sell_rows": [],
        "warn_rows": [],
        "data_status_rows": [],
        "unfilled_rows": [],
        "hold_rows": [],
        "watch_rows": [],
        "held_tickers_count": 0,
        "max_concurrent": 8,
    }


@pytest.fixture
def cfg():
    return {
        "account": {"equity_usd": 100000},
        "risk": {
            "a1_budget_pct": 0.2222222222222222,
            "a2_budget_pct": 0.4444444444444444,
            "a3_budget_pct": 1.3333333333333333,
            "b_budget_pct": 1.0,
            "gap_buffer_pct": 1.5,
            "min_risk_per_share_pct": 1.0,
            "max_position_pct": 25,
            "max_concurrent_positions": 8,
        },
        "indicators": {"ichimoku_shift": 26},
    }


def test_render_report_succeeds_when_all_sections_empty(tmp_path, cfg):
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    assert path.exists()
    html = path.read_text(encoding="utf-8")
    assert html.count("오늘 해당 종목 없음") >= 5  # 신규 매수 4탭 + 매도 등
    assert "실전" in html


def test_render_report_paper_mode_label(tmp_path, cfg):
    path = report_html.render_report(_empty_summary(mode="paper"), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "모의" in html


def test_buy_rows_carry_data_stage_limit_stop_attributes(tmp_path, cfg):
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": "PEP",
            "kr": "펩시코",
            "stage": "A1",
            "bucket": "b1",
            "limit": 132.50,
            "stop": 127.98,
            "stop_pct": -3.4,
            "qty": 34,
            "stop_basis": "10일 최저가",
            "earnings": "2026-10-07",
            "decision": "매수",
            "note": "",
            "score": 20,
            "grade": "",
            "rsi_prev": 25.1,
            "rsi_now": 31.2,
            "vol_ratio": 1.3,
        }
    ]
    summary["buy_count"] = 1
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'data-stage="b1"' in html
    assert 'data-limit="132.5"' in html
    assert 'data-stop="127.98"' in html
    assert "PEP" in html and "펩시코" in html


def test_buy_row_without_stop_leaves_data_stop_empty(tmp_path, cfg):
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": "XYZ",
            "kr": "엑스와이지",
            "stage": "A1",
            "bucket": "b1",
            "limit": 50.0,
            "stop": None,
            "stop_pct": None,
            "qty": 0,
            "stop_basis": "10일 최저가",
            "earnings": "확인불가",
            "decision": "보류",
            "note": "손절가 계산 불가로 수량 미산정 — 매수 보류",
            "score": 0,
            "grade": "",
            "rsi_prev": 25.0,
            "rsi_now": 31.0,
            "vol_ratio": None,
        }
    ]
    summary["buy_count"] = 1
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'data-stop=""' in html
    assert "미확정" in html


def test_filtered_reason_label_mapping():
    from engine.daily import _label_filter_reason

    assert _label_filter_reason("골든크로스 당일 RSI 70 이상") == "과열 (RSI 70 이상)"
    assert _label_filter_reason("최근 20거래일 MACD 교차 5회 이상 (휩소)") == "잦은 교차 (횡보)"
    assert _label_filter_reason("실적 발표 3거래일 이내") == "실적 발표 3거래일 이내"
