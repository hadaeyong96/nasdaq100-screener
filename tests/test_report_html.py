"""notify/report_html.py 렌더링 테스트. 네트워크 없음 (템플릿 렌더링만)."""

from __future__ import annotations

import re

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
        "indicators": {
            "ichimoku_shift": 26,
            "rsi": {"period": 14},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
        },
        "assumptions": {
            "a1_to_a2_expiry_days": 10,
            "reentry_cooldown_days": 5,
            "gap_filter_pct": 4.0,
            "whipsaw_max_crosses_20d": 4,
            "swing_low_period": 10,
            "s_grade_macd_norm_min_pct": -0.5,
            "b_grade_macd_norm_max_pct": -2.0,
        },
        "entry": {"limit_markup": 1.01},
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


def test_render_report_filenames_are_namespaced_by_mode(tmp_path, cfg):
    """같은 날짜라도 live·paper 보고서 파일이 서로 덮어쓰면 안 된다."""
    live_path = report_html.render_report(_empty_summary(mode="live"), cfg, tmp_path)
    paper_path = report_html.render_report(_empty_summary(mode="paper"), cfg, tmp_path)

    assert live_path != paper_path
    assert live_path.name == "report_live_2026-09-23.html"
    assert paper_path.name == "report_paper_2026-09-23.html"
    assert live_path.exists() and paper_path.exists()


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
    # 이전 버전의 조건 열(탭별로 달랐던 헤더)은 더 이상 없어야 한다 — 머리글(thead) 안에서만 확인
    # (P3.7 "읽는 법" 탭 본문에는 "RSI 30 돌파" 같은 설명 문구가 정상적으로 등장한다).
    theads = re.findall(r"<thead>.*?</thead>", html, re.S)
    for phrase in ("RSI 30 돌파", "앞구름 양운", "계좌%", "위험금액"):
        assert not any(phrase in t for t in theads), phrase


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
    assert "체결 후 구글 시트 '체결' 탭에 한 줄씩 기록하세요" in html
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


# ── 라이브 어드바이저 1단계 (docs/design/live_advisor.md 3·6·7번) ──────────


def _live_judgment_row(**overrides):
    row = {
        "ticker": "NVDA", "name_kr": "엔비디아", "judgment": "추가매수", "reason": "1차 · RSI 30 탈출 신호",
        "changed": False, "stop_price": 170.0, "has_plan": True, "plan_budget_krw": 9_000_000.0,
        "qty": 5, "tranche_krw": 1_000_000.0, "tranche_usd": 740.7, "one_share_warning": None,
    }
    row.update(overrides)
    return row


def test_live_mode_hides_total_krw_in_header(tmp_path, cfg):
    path = report_html.render_report(_empty_summary(mode="live"), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    header = html.split("<nav")[0]
    assert "총자금" not in header


def test_paper_mode_still_shows_total_krw_in_header(tmp_path, cfg):
    path = report_html.render_report(_empty_summary(mode="paper"), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    header = html.split("<nav")[0]
    assert "총자금" in header


def test_live_mode_renders_judgment_table_row(cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [_live_judgment_row()]
    context = report_html.build_context(summary, cfg)
    assert context["is_live"] is True
    row = context["live_judgment_rows"][0]
    assert row["badge_class"] == "b-buy"
    assert row["plan_budget_krw_str"] == "900만"
    assert row["tranche_krw_str"] == "100만"


def test_live_judgment_table_appears_in_rendered_html(tmp_path, cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [_live_judgment_row()]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "NVDA" in html
    assert '<span class="badge b-buy">추가매수</span>' in html


def test_paper_mode_never_renders_live_judgment_table(tmp_path, cfg):
    """live_judgment_rows가 어쩌다 섞여 들어와도 paper 보고서에는 안 나와야 한다."""
    summary = _empty_summary(mode="paper")
    summary["live_judgment_rows"] = [_live_judgment_row()]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "계획·체결 종목" not in html


def test_changed_judgment_banner_appears_only_when_changed_rows_exist(tmp_path, cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [_live_judgment_row(changed=True)]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "판정이 바뀐 종목" in html
    assert "NVDA(추가매수)" in html


def test_changed_judgment_banner_absent_when_nothing_changed(tmp_path, cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [_live_judgment_row(changed=False)]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "판정이 바뀐 종목" not in html


def test_one_share_warning_banner_appears_when_present(tmp_path, cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [
        _live_judgment_row(
            qty=0, tranche_krw=None,
            one_share_warning={"min_budget_krw": 1_800_000.0, "ref_price": 180.0},
        )
    ]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "1차 매수 0주" in html
    assert "180만" in html or "1,800,000" in html or "180" in html


def test_one_share_warning_banner_absent_when_no_warnings(tmp_path, cfg):
    summary = _empty_summary(mode="live")
    summary["live_judgment_rows"] = [_live_judgment_row(one_share_warning=None)]
    path = report_html.render_report(summary, cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "1차 매수 0주" not in html


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


# ── P3.7: "설명" 칸·왜? 버튼 ───────────────────────────────────────────────


def _sample_explain(title="테스트 · 설명"):
    return {"title": title, "badge": "슬롯의 1/9", "checks": [{"level": "y", "text": "체크1"}], "body": "본문 설명", "next": "다음 단계"}


def _lxml():
    lxml_html = pytest.importorskip("lxml.html")
    return lxml_html


def test_why_row_colspan_matches_buy_table_header_count(tmp_path, cfg):
    lxml_html = _lxml()
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": "PEP", "kr": "펩시코", "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
            "key": "PEP-A1", "is_new_position": True, "limit": 132.5, "stop": 127.98, "qty": 34,
            "amount_krw": 4_500_000, "max_loss_krw": 220_000, "target_qty": 34, "risk_cap_qty": 40,
            "stop_pct": -3.4, "decision": "매수", "note": "", "score": 20, "grade": "",
            "explain": _sample_explain("펩시코 · 1차 정찰 매수"),
        }
    ]
    summary["buy_count"] = 1
    path = report_html.render_report(summary, cfg, tmp_path)
    doc = lxml_html.fromstring(path.read_text(encoding="utf-8"))
    table = doc.get_element_by_id("b1").find(".//table")
    header_count = len(table.findall(".//thead/tr/th"))
    why_tds = table.findall(".//tbody/tr[@class='why']/td")
    assert len(why_tds) == 1
    assert int(why_tds[0].get("colspan")) == header_count
    # 종목 없는 표(예: b2)는 빈 줄 colspan도 같은 칸 수여야 한다.
    empty_td = doc.get_element_by_id("b2").find(".//table//tbody/tr/td[@class='empty']")
    assert int(empty_td.get("colspan")) == header_count


def test_why_row_immediately_follows_its_signal_row(tmp_path, cfg):
    lxml_html = _lxml()
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": "AAA", "kr": "에이", "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
            "key": "AAA-A1", "is_new_position": True, "limit": 100.0, "stop": 94.0, "qty": 10,
            "amount_krw": 1_000_000, "max_loss_krw": 100_000, "target_qty": 10, "risk_cap_qty": 20,
            "stop_pct": -6.0, "decision": "매수", "note": "", "score": 20, "grade": "",
            "explain": _sample_explain("에이 · 설명"),
        },
        {
            "ticker": "BBB", "kr": "비", "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
            "key": "BBB-A1", "is_new_position": True, "limit": 50.0, "stop": None, "qty": 0,
            "amount_krw": 0, "max_loss_krw": 0, "target_qty": 0, "risk_cap_qty": 0,
            "stop_pct": None, "decision": "보류", "note": "손절가 계산 불가로 수량 미산정 — 매수 보류", "score": 0, "grade": "",
            "explain": None,
        },
    ]
    summary["buy_count"] = 2
    path = report_html.render_report(summary, cfg, tmp_path)
    doc = lxml_html.fromstring(path.read_text(encoding="utf-8"))
    rows = doc.get_element_by_id("b1").findall(".//tbody/tr")
    classes = [r.get("class") for r in rows]
    # 설명 있는 AAA 뒤에는 tr.why가 바로 오고, 설명 없는 BBB 뒤에는 오지 않는다.
    assert classes == ["sig", "why", "sig"]
    aaa_btn = rows[0].find(".//button")
    bbb_btn = rows[2].find(".//button")
    assert aaa_btn.get("disabled") is None
    assert bbb_btn.get("disabled") is not None


def test_sell_watch_filtered_warn_tables_get_explain_column_and_why_row(tmp_path, cfg):
    lxml_html = _lxml()
    summary = _empty_summary()
    summary["sell_rows"] = [
        {
            "티커": "ROP", "종목명": "로퍼", "kind": "E3", "신호": "구조 붕괴(E3)", "매도범위": "3차분(67%) 또는 잔량",
            "수량": 14, "평균단가": 441.20, "종가": 412.30, "예상손익_krw": -558900, "손익률": -6.6,
            "주문안내": "안내", "비고": "", "explain": _sample_explain("로퍼 · E3"),
        }
    ]
    summary["watch_rows"] = [
        {"티커": "PEP", "종목명": "펩시코", "현재단계": "1차", "기다리는신호": "2차", "남은거래일": 5, "explain": _sample_explain("펩시코 · 관찰")}
    ]
    summary["filtered_rows"] = [
        {"티커": "MU", "종목명": "마이크론", "단계": "재진입", "유형": "매매금지", "사유": "실적 발표 3거래일 이내", "explain": _sample_explain("마이크론 · 제외")}
    ]
    summary["warn_rows"] = [
        {"티커": "QCOM", "종목명": "퀄컴", "내용": "목표 도달 (매도 아님)", "badge_class": "b-info", "explain": _sample_explain("퀄컴 · 경고")}
    ]
    path = report_html.render_report(summary, cfg, tmp_path)
    doc = lxml_html.fromstring(path.read_text(encoding="utf-8"))
    for section_id in ("sell", "watch", "filtered", "warn"):
        table = doc.get_element_by_id(section_id).find(".//table")
        header_count = len(table.findall(".//thead/tr/th"))
        headers = [th.text_content().strip() for th in table.findall(".//thead/tr/th")]
        assert headers[-1] == "설명"
        why_tds = table.findall(".//tbody/tr[@class='why']/td")
        assert len(why_tds) == 1, section_id
        assert int(why_tds[0].get("colspan")) == header_count, section_id


def test_guide_tab_uses_config_values_not_hardcoded(tmp_path, cfg):
    """P3.7: "읽는 법" 탭의 숫자 기준은 config.yaml 값을 그대로 반영해야 한다."""
    cfg = {**cfg, "assumptions": {**cfg["assumptions"], "a1_to_a2_expiry_days": 7, "reentry_cooldown_days": 3}}
    path = report_html.render_report(_empty_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert 'id="guide"' in html
    assert "7거래일" in html
    assert "3거래일" in html


def test_why_row_follows_sorted_signal_row_in_browser(tmp_path, cfg):
    """P3.7 4번: 점수 머리글로 정렬해도 설명 줄이 자기 종목 바로 아래를 따라가는지(playwright)."""
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    summary = _empty_summary()
    summary["buy_groups"]["b1"] = [
        {
            "ticker": t, "kr": t, "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
            "key": f"{t}-A1", "is_new_position": True, "limit": 100.0, "stop": 94.0, "qty": 10,
            "amount_krw": 1_000_000, "max_loss_krw": 100_000, "target_qty": 10, "risk_cap_qty": 20,
            "stop_pct": -6.0, "decision": "매수", "note": "", "score": score, "grade": "",
            "explain": _sample_explain(f"{t} · 설명"),
        }
        for t, score in (("AAA", 20), ("BBB", 60), ("CCC", 5))
    ]
    summary["buy_count"] = 3
    path = report_html.render_report(summary, cfg, tmp_path)

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(path.resolve().as_uri(), wait_until="domcontentloaded", timeout=15000)
            table = page.locator("#all table")  # "전체" 탭이 기본으로 보임(b1은 하위 탭이라 숨김)
            table.locator("thead th", has_text="점수").click()  # 오름차순
            rows = table.locator("tbody tr")
            classes = rows.evaluate_all("els => els.map(e => e.className)")
            tickers = rows.evaluate_all("els => els.map(e => e.dataset.key || '')")
            # sig/why가 번갈아 나오고, 각 why 앞의 sig가 같은 종목이어야 한다(정렬 후에도 짝이 안 깨짐).
            assert classes == ["sig", "why"] * 3
            assert [tickers[i] for i in range(0, 6, 2)] == ["CCC-A1", "AAA-A1", "BBB-A1"]  # 점수 5<20<60
            browser.close()
    except Exception as e:  # pragma: no cover - 브라우저 바이너리가 없는 환경 대비
        pytest.skip(f"playwright 브라우저를 쓸 수 없어 건너뜀: {e}")


# ── 브리핑 다이어트 (2026-10-14): 맨 위 오늘 추천 + 나머지는 접기(기본 닫힘) ──────────────


def _compact_summary():
    summary = _empty_summary()
    summary["regime_line"] = "오늘 국면: QQQ 200일선 위(+10.4%) · 전환 장부 QLD·TQQQ 보유 중"
    summary["buy_groups"]["b2"] = [{
        "ticker": "NVDA", "kr": "엔비디아", "stage": "A2", "bucket": "b2", "stage_label": "2차 확인",
        "key": "NVDA-A2", "is_new_position": False, "limit": 120.0, "stop": 110.0, "qty": 3, "amount_krw": 0,
        "max_loss_krw": 0, "stop_pct": -8.3, "decision": "매수", "note": "", "score": 65, "grade": "S", "close": 118.81,
    }]
    summary["buy_count"] = 1
    summary["sell_rows"] = [{
        "티커": "ROP", "종목명": "로퍼", "kind": "E1", "신호": "모멘텀 약화(E1, 데드크로스)", "매도범위": "1차분",
        "수량": 2, "평균단가": 400.0, "종가": 412.3, "손익률": 3.1, "주문안내": "", "비고": "",
    }]
    summary["vbooks"] = {"start": "2026-10-14", "benchmark": "QQQ 그냥 보유", "rows": [
        {"book": "QQQ 그냥 보유", "value": 75000.0, "cum_return_pct": 0.5, "mdd_pct": -1.0, "switches": 0, "holding": "QQQ"},
        {"book": "TQQQ 200일선 전환", "value": 76000.0, "cum_return_pct": 1.6, "mdd_pct": -3.0, "switches": 1, "holding": "SGOV"},
    ]}
    return summary


def test_top_block_comes_before_all_folds_and_folds_are_closed(tmp_path, cfg):
    html = report_html.render_report(_compact_summary(), cfg, tmp_path).read_text(encoding="utf-8")
    body = html[html.index("<header>"):]
    i_head, i_regime = body.index("<header>"), body.index('<div class="regime-line">')
    i_today = body.index('<section class="today"')
    i_fold = body.index('<details class="fold"')
    assert i_head < i_regime < i_today < i_fold
    today = body[i_today:i_fold]
    assert "엔비디아" in today and "NVDA" in today and "$118.81" in today
    assert "MACD 골든크로스 · 2차 확인" in today and "65점 · S등급" in today
    assert "로퍼" in today and "+3.1%" in today and "모멘텀 약화(E1, 데드크로스)" in today
    folds = re.findall(r"<details class=\"fold\"[^>]*>\s*<summary>([^<]+)</summary>", body)
    assert folds == ["▶ 전체 후보 목록 (오늘의 스크리닝)", "▶ 매수·매도 상세 · 보유 현황 · 경고 · 규칙",
                     "▶ FRED 경제 그래프 (시장 온도)", "▶ 돈이 몰리는 종목", "▶ 가상 장부 현황", "▶ 지표 설명 (읽는 법·용어)"]
    assert not re.search(r"<details class=\"fold\"[^>]*\bopen\b", body)  # 기본 닫힘


def test_existing_sections_kept_inside_folds(tmp_path, cfg):
    """기존 내용은 지우지 않고 위치만 옮겼다: 탭 패널·읽는 법·체결 안내가 모두 접기 안에 있다."""
    html = report_html.render_report(_compact_summary(), cfg, tmp_path).read_text(encoding="utf-8")
    first_fold = html.index('<details class="fold"')
    for marker in ['id="buy"', 'id="sell"', 'id="hold"', 'id="watch"', 'id="filtered"', 'id="warn"', 'id="rules"',
                   'id="guide"', "이 프로그램이 쓰는 지표 4가지", "용어 사전", "탭에 한 줄씩 기록하세요",
                   'aria-label="오늘의 스크리닝"', "주문 안내"]:
        assert marker in html and html.index(marker) > first_fold, marker
    assert 'data-tab="guide"' not in html  # 읽는 법은 탭이 아니라 "지표 설명" 접기로
    assert 'class="panel gd"' not in html


def test_no_recommendation_line_when_nothing_today(tmp_path, cfg):
    html = report_html.render_report(_empty_summary(), cfg, tmp_path).read_text(encoding="utf-8")
    today = html[html.index('<section class="today"'):html.index('<details class="fold"')]
    assert "오늘 매수·매도 추천 없음" in today and "<table" not in today
    assert "가상 장부 정보 없음" in html


def test_vbooks_fold_table(tmp_path, cfg):
    html = report_html.render_report(_compact_summary(), cfg, tmp_path).read_text(encoding="utf-8")
    vb = html[html.index('id="fold-vbooks"'):html.index('id="fold-guide"')]
    assert "TQQQ 200일선 전환" in vb and "76,000" in vb and "+1.10%p" in vb and "SGOV" in vb and "시작 2026-10-14" in vb


def test_explanations_are_collapsed_by_default(tmp_path, cfg):
    """설명은 항상 접힌 채로 (2026-10-14 사용자 요청): 어떤 details도 open으로 렌더하지 않고,
    전략 설명·지표 설명 소제목은 각각 접기 안에 있다. 페이지를 열 때 앵커로 자동 펼치지 않는다."""
    html = report_html.render_report(_compact_summary(), cfg, tmp_path).read_text(encoding="utf-8")
    assert not re.search(r"<details[^>]*\bopen\b", html)
    assert '<details class="exp"><summary>① 우리 전략 한눈에 (설명)</summary>' in html
    guide = html[html.index('id="guide"'):html.index("</section>", html.index('id="guide"'))]
    assert guide.count('<details class="exp">') == 7 and "<h2>" not in guide
    assert "openTarget(location.hash)" not in html and "openFoldFor" not in html
