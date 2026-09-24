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
        "account": {"total_krw": 100_000_000},
        "plan": {"strategy_limit_pct": 60, "cash_buffer_pct": 5, "max_slots": 8},
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
    assert 'data-stage="A1"' in html
    assert 'data-key="PEP-A1"' in html
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


def test_stale_summary_shows_red_banner_and_empty_buy_tab(tmp_path, cfg):
    """P3.2 2번: 데이터 지연 모드는 배너가 뜨고 매수 탭은 비어 있어야 한다."""
    summary = _empty_summary()
    summary["stale"] = True
    summary["expected_date"] = "2026-09-23"
    summary["actual_date"] = "2026-09-22"
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "데이터 지연" in html
    assert "2026-09-23" in html and "2026-09-22" in html
    assert summary["buy_count"] == 0


def test_non_stale_summary_has_no_banner(tmp_path, cfg):
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'class="stale-banner"' not in html


def test_buy_table_header_has_max_loss_column(tmp_path, cfg):
    """P3.6 6-3번: 차수별 매수 표는 종목·티커·결정·지정가·수량·투입금액·손절가·손절폭·최대손실·점수·비고 11칸이다."""
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    for header in ("종목", "티커", "결정", "지정가", "수량", "투입금액", "손절가", "손절폭", "최대손실", "점수", "비고"):
        assert f"<th" in html and header in html
    # 이전 버전의 조건 열(탭별로 달랐던 헤더)은 더 이상 없어야 한다.
    assert "RSI 30 돌파" not in html
    assert "앞구름 양운" not in html
    assert "계좌%" not in html
    assert "위험금액" not in html


def test_buy_tabs_include_all_tab_selected_by_default(tmp_path, cfg):
    """P3.6 2번: "전체" 탭이 하위 버튼 맨 앞에 있고 기본 선택돼야 한다."""
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert '<div class="sub on" id="all">' in html
    assert 'data-sub="all" aria-selected="true"' in html
    all_pos = html.index('data-sub="all"')
    b1_pos = html.index('data-sub="b1"')
    assert all_pos < b1_pos


def test_empty_subtab_button_is_dimmed_and_nonempty_is_highlighted(tmp_path, cfg):
    """P3.6 2번: 종목이 있는 하위 버튼은 초록 테두리(has), 0건인 버튼은 흐리게(empty) 표시된다."""
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": "PEP", "kr": "펩시코", "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
            "key": "PEP-A1", "is_new_position": True, "limit": 132.5, "stop": 127.98, "qty": 34,
            "amount_krw": 4_500_000, "max_loss_krw": 220_000, "target_qty": 34, "risk_cap_qty": 40,
            "stop_pct": -3.4, "decision": "매수", "note": "", "score": 20, "grade": "",
        }
    ]
    summary["buy_count"] = 1
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert '<button data-sub="b1" aria-selected="false" class="has">' in html
    assert '<button data-sub="b2" aria-selected="false" class="empty">' in html


def test_all_tab_row_count_equals_sum_of_stage_tabs(tmp_path, cfg):
    """P3.6 2번: "전체" 표 행 수는 차수별 표 행 수의 합과 같아야 한다(중복 집계 없음)."""

    def _row(ticker, stage, bucket, score):
        return {
            "ticker": ticker,
            "kr": ticker,
            "stage": stage,
            "bucket": bucket,
            "stage_label": "1차 정찰",
            "key": f"{ticker}-{stage}",
            "is_new_position": stage in ("A1", "B"),
            "limit": 100.0,
            "stop": 94.0,
            "qty": 10,
            "amount_krw": 1_000_000,
            "max_loss_krw": 100_000,
            "target_qty": 10,
            "risk_cap_qty": 20,
            "stop_pct": -6.0,
            "decision": "매수",
            "note": "",
            "score": score,
            "grade": "",
        }

    summary = _empty_summary()
    summary["buy_groups"] = {
        "b1": [_row("AAA", "A1", "b1", 20), _row("BBB", "A1", "b1", 5)],
        "b2": [],
        "b3": [_row("CCC", "A3", "b3", 15)],
        "b9": [_row("DDD", "B", "b9", 75)],
    }
    context = report_html.build_context(summary, cfg)
    all_tab = next(t for t in context["buy_tabs"] if t["key"] == "all")
    stage_total = sum(t["count"] for t in context["buy_tabs"] if t["key"] != "all")
    assert all_tab["count"] == 4 == stage_total
    scores = [r["score"] for r in all_tab["rows"]]
    assert scores == sorted(scores, reverse=True)  # 전체 표는 점수 내림차순


def test_finviz_link_href_format(tmp_path, cfg):
    """P3.6 1번: 티커는 Finviz 종목 페이지로 새 창 링크가 걸린다."""
    summary = _empty_summary()
    summary["watch_rows"] = [
        {"티커": "BRK-B", "종목명": "버크셔", "현재단계": "1차", "기다리는신호": "2차", "남은거래일": 5}
    ]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'href="https://finviz.com/quote.ashx?t=BRK-B&p=d"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener"' in html


def test_bottom_hold_table_is_removed_but_fills_notice_remains(tmp_path, cfg):
    """P3.4 1번: 하단 "내 보유 종목" 표는 삭제하고, 체결 기록 안내 줄만 남긴다."""
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "내 보유 종목" not in html
    assert "체결 후 아래 파일에 한 줄씩 기록하세요" in html
    assert html.count('id="hold"') == 1  # "보유 현황" 탭 하나만 남는다


def test_hold_row_renders_stop_alert_badge(tmp_path, cfg):
    """P3.5: 보유 현황 비고 칸은 배지(b-warn/b-info/b-sell)로 손절 알림을 보여준다."""
    summary = _empty_summary()
    summary["hold_rows"] = [
        {
            "티커": "AMZN",
            "종목명": "아마존",
            "단계": "정찰",
            "수량": 12,
            "평균단가": 221.40,
            "종가": 224.10,
            "평가금액": 2689.20,
            "손익률": 1.2,
            "손절가": 212.80,
            "손절까지": -5.0,
            "오늘신호": "",
            "배지클래스": "b-warn",
            "배지": "손절 근접 · 예약 $212.80 확인",
        }
    ]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert '<span class="badge b-warn">손절 근접 · 예약 $212.80 확인</span>' in html


def test_sell_row_renders_order_guidance_column(tmp_path, cfg):
    """P3.5 5번: 매도·손절 탭에 주문 안내 칸이 있다."""
    summary = _empty_summary()
    summary["sell_rows"] = [
        {
            "티커": "ROP",
            "종목명": "로퍼",
            "kind": "STOP",
            "신호": "손절",
            "매도범위": "전량",
            "수량": 14,
            "평균단가": 441.20,
            "종가": 412.30,
            "손익률": -6.6,
            "주문안내": "손절 예약 확인 · 증권사 손절 예약($405.00)이 오늘 체결됐으면 체결 기록만 입력. 체결 안 됐으면 다음 거래일 장 시작 시 14주 전량 시장가 매도 예약",
            "비고": "",
        }
    ]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "주문 안내" in html
    assert "손절 예약 확인" in html


def test_filtered_reason_label_mapping():
    from engine.daily import _label_filter_reason

    assert _label_filter_reason("골든크로스 당일 RSI 70 이상") == "과열 (RSI 70 이상)"
    assert _label_filter_reason("최근 20거래일 MACD 교차 5회 이상 (휩소)") == "잦은 교차 (횡보)"
    assert _label_filter_reason("실적 발표 3거래일 이내") == "실적 발표 3거래일 이내"
