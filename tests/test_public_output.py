"""단체방(투자클럽) 공개 발송 테스트. 네트워크 없음.

- notify.briefing.build_public_briefing_text: 시장 온도 + 오늘의 추천만, 보유·
  수량·평단·손익·계획금액·체결 내역이 새지 않는지.
- notify.report_html.build_public_context / render_public_report: 위와 같은 내용을
  HTML에서도 확인한다.
- notify.telegram.send_group_briefing / group_send_warning_line: TELEGRAM_GROUP_CHAT_ID
  없을 때 건너뛰기, paper 모드 하드 가드, 슈퍼그룹 전환(migrate_to_chat_id) 처리, 중복 발송 방지.
"""

from __future__ import annotations

import pandas as pd
import pytest

from notify import briefing, report_html, telegram
from store import db


# ── 공용 픽스처 ──────────────────────────────────────────────────────────


def _macro_row(slug="fear-greed", value=50, name="공포·탐욕 지수"):
    return {
        "slug": slug,
        "name": name,
        "code": "CNN Fear & Greed",
        "value": value,
        "unit": "/100",
        "as_of": "2026-09-23",
        "change_1w": None,
        "series": [40, 45, value],
        "badge": {"status": "warn", "symbol": "🟡", "label": "탐욕", "text": "🟡주의", "zone": "탐욕", "scale": []},
        "is_stale": False,
        "short_range": False,
        "note": None,
        "ref_values": [],
    }


def _buy_row(ticker="AAPL", stage_label="1차 정찰", limit=190.20, stop=182.10, stop_pct=-4.3, **overrides):
    row = {
        "ticker": ticker,
        "kr": "애플",
        "stage": "A1",
        "bucket": "b1",
        "stage_label": stage_label,
        "key": f"{ticker}-A1",
        "limit": limit,
        "stop": stop,
        "stop_pct": stop_pct,
        "decision": "매수",
        "note": "",
        "condition_summary": "RSI 28.0 → 32.0",
        "score": 20,
        "grade": "",
        # 아래 필드들은 개인 자금 계획에 딸린 값 — 공개용 결과물에는 절대 나오면 안 된다.
        "qty": 987,
        "amount_krw": 4_567_890,
        "max_loss_krw": 111_222,
    }
    row.update(overrides)
    return row


def _sensitive_summary(**overrides):
    """보유·계획금액·체결 관련 민감한 값을 가득 채운 summary (leak 테스트용)."""
    summary = {
        "mode": "live",
        "mode_label": "실전",
        "as_of": pd.Timestamp("2026-09-23"),
        "macro_rows": [_macro_row()],
        "buy_groups": {"b1": [_buy_row()], "b2": [], "b3": [], "b9": []},
        "buy_count": 1,
        "sell_rows": [
            {
                "티커": "ROP",
                "종목명": "로퍼",
                "kind": "STOP",
                "신호": "손절",
                "매도범위": "전량",
                "수량": 777,
                "평균단가": 441.20,
                "종가": 412.30,
                "예상손익_krw": -9_876_543,
                "손익률": -6.6,
            }
        ],
        "hold_rows": [
            {
                "티커": "AMZN",
                "종목명": "아마존",
                "단계": "정찰",
                "수량": 321,
                "평균단가": 221.40,
                "종가": 224.10,
                "평가금액": 71932.40,
                "평가금액_krw": 98_765_432,
                "평가손익_krw": 1_234_567,
                "손익률": 1.2,
            }
        ],
        "funding_plan": {
            "strategy_limit_krw": 60_000_000,
            "slot_krw": 7_500_000,
            "held_krw": 55_555_555,
            "reserved_krw": 3_000_000,
            "new_krw": 2_000_000,
            "remaining_krw": 1_000_000,
            "qqqm_target_krw": 6_666_666,
            "fx_rate": 1400.0,
        },
        "unfilled_rows": [],
    }
    summary.update(overrides)
    return summary


_SENSITIVE_MARKERS = [
    "987",  # buy row qty
    "4,567,890", "4567890",  # buy row amount_krw
    "111,222", "111222",  # buy row max_loss_krw
    "777",  # sell row qty
    "9,876,543", "9876543",  # sell row pnl krw
    "321",  # hold row qty
    "98,765,432", "98765432",  # hold row value krw
    "1,234,567", "1234567",  # hold row pnl krw
    "55,555,555", "55555555",  # funding held_krw
    "아마존",  # 보유 종목명
    "ROP",  # 매도 종목 티커(오늘 추천에 없는 종목)
]


# ── notify.briefing.build_public_briefing_text ─────────────────────────


def test_public_briefing_no_signals_says_none():
    summary = _sensitive_summary(buy_groups={"b1": [], "b2": [], "b3": [], "b9": []}, buy_count=0)
    text = briefing.build_public_briefing_text(summary, {})
    assert "🎯 오늘의 추천" in text
    assert "- 오늘 추천 종목 없음" in text


def test_public_briefing_recommend_line_has_entry_stop_condition():
    summary = _sensitive_summary()
    text = briefing.build_public_briefing_text(summary, {})
    assert "AAPL(1차 정찰) 진입가 $190.20 · 손절가 $182.10 · RSI 28.0 → 32.0" in text


def test_public_briefing_sorts_by_score_descending():
    summary = _sensitive_summary(
        buy_groups={
            "b1": [_buy_row("LOW", score=5), _buy_row("HIGH", score=90)],
            "b2": [],
            "b3": [],
            "b9": [],
        }
    )
    text = briefing.build_public_briefing_text(summary, {})
    assert text.index("HIGH") < text.index("LOW")


def test_public_briefing_excludes_b2_b3_b9_signals_that_reveal_holdings():
    """b2(2차 확인)·b3(3차 확정)·b9(재진입)는 라이브에서 실제 보유했거나 보유
    중인 종목에만 나오는 신호라 종목명만으로도 보유가 드러난다 — 공개용에는
    신규 진입(b1)만 나와야 한다."""
    summary = _sensitive_summary(
        buy_groups={
            "b1": [_buy_row("AAPL", stage_label="1차 정찰")],
            "b2": [_buy_row("NVDA", stage_label="2차 확인", condition_summary="MACD 골든크로스")],
            "b3": [_buy_row("MSFT", stage_label="3차 확정", condition_summary="구름 돌파")],
            "b9": [_buy_row("TSLA", stage_label="재진입", condition_summary="추세 복귀")],
        }
    )
    text = briefing.build_public_briefing_text(summary, {})
    assert "AAPL" in text
    for held_ticker in ("NVDA", "MSFT", "TSLA"):
        assert held_ticker not in text, f"보유를 드러내는 {held_ticker}(b2/b3/b9)가 공개용 텍스트에 나옴"
    for stage_word in ("2차 확인", "3차 확정", "재진입"):
        assert stage_word not in text

    context = report_html.build_public_context(summary, {})
    html_tickers = {row["ticker"] for row in context["buy_rows"]}
    assert html_tickers == {"AAPL"}


def test_public_report_html_excludes_b2_b3_b9_signals_that_reveal_holdings(tmp_path):
    summary = _sensitive_summary(
        buy_groups={
            "b1": [_buy_row("AAPL", stage_label="1차 정찰")],
            "b2": [_buy_row("NVDA", stage_label="2차 확인", condition_summary="MACD 골든크로스")],
            "b3": [_buy_row("MSFT", stage_label="3차 확정", condition_summary="구름 돌파")],
            "b9": [_buy_row("TSLA", stage_label="재진입", condition_summary="추세 복귀")],
        }
    )
    path = report_html.render_public_report(summary, {}, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "AAPL" in html
    for held_ticker in ("NVDA", "MSFT", "TSLA"):
        assert held_ticker not in html, f"보유를 드러내는 {held_ticker}(b2/b3/b9)가 공개용 HTML에 나옴"
    for stage_word in ("2차 확인", "3차 확정", "재진입"):
        assert stage_word not in html


def test_public_briefing_has_market_temp_and_disclaimer():
    text = briefing.build_public_briefing_text(_sensitive_summary(), {})
    assert "📈 시장 온도" in text
    assert briefing.PUBLIC_DISCLAIMER in text


def test_public_briefing_never_leaks_holdings_or_amounts():
    text = briefing.build_public_briefing_text(_sensitive_summary(), {})
    for marker in _SENSITIVE_MARKERS:
        assert marker not in text, f"공개용 텔레그램 본문에 민감 정보 유출: {marker!r}"


def test_public_briefing_condition_uses_condition_summary_not_note():
    """note에는 개인 자금 계획 문구("계획 없음" 등)가 섞일 수 있어 쓰면 안 된다."""
    summary = _sensitive_summary(
        buy_groups={
            "b1": [_buy_row(note="계획 없음 · 남은 한도 부족", condition_summary="MACD 골든크로스")],
            "b2": [], "b3": [], "b9": [],
        }
    )
    text = briefing.build_public_briefing_text(summary, {})
    assert "MACD 골든크로스" in text
    assert "계획 없음" not in text
    assert "남은 한도" not in text


# ── notify.report_html public context / HTML ───────────────────────────


@pytest.fixture
def cfg():
    return {}


def test_render_public_report_creates_file(tmp_path, cfg):
    path = report_html.render_public_report(_sensitive_summary(), cfg, tmp_path)
    assert path.exists()
    assert path.name == "report_public_2026-09-23.html"


def test_public_report_shows_market_temp_and_recommendation(tmp_path, cfg):
    path = report_html.render_public_report(_sensitive_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert "시장 온도" in html
    assert "AAPL" in html
    assert "$190.20" in html
    assert "$182.10" in html
    assert "1차 정찰" in html


def test_public_report_has_disclaimer_footer(tmp_path, cfg):
    path = report_html.render_public_report(_sensitive_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    assert briefing.PUBLIC_DISCLAIMER in html
    footer = html.split("<footer>", 1)[1]
    assert briefing.PUBLIC_DISCLAIMER in footer


def test_public_report_never_leaks_holdings_or_amounts(tmp_path, cfg):
    path = report_html.render_public_report(_sensitive_summary(), cfg, tmp_path)
    html = path.read_text(encoding="utf-8")
    for marker in _SENSITIVE_MARKERS:
        assert marker not in html, f"공개용 HTML에 민감 정보 유출: {marker!r}"


def test_build_public_context_has_no_portfolio_keys():
    """build_context(개인용)에 있는 보유·자금 관련 키가 공개용 context에는 아예 없어야 한다."""
    context = report_html.build_public_context(_sensitive_summary(), {})
    for forbidden_key in ("hold_rows", "hold_totals", "funding", "sell_rows", "total_krw", "live_judgment_rows"):
        assert forbidden_key not in context


def test_build_public_context_buy_rows_exclude_qty_and_amount():
    context = report_html.build_public_context(_sensitive_summary(), {})
    row = context["buy_rows"][0]
    assert "qty" not in row
    assert "amount_krw" not in row
    assert "max_loss_krw" not in row
    assert row["ticker"] == "AAPL"
    assert row["condition"] == "RSI 28.0 → 32.0"


# ── notify.telegram.send_group_briefing ─────────────────────────────────


def _base_group_summary(**overrides):
    summary = {"mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23")}
    summary.update(overrides)
    return summary


def test_send_group_briefing_skips_without_group_chat_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.delenv("TELEGRAM_GROUP_CHAT_ID", raising=False)
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    def _boom(*a, **k):
        raise AssertionError("TELEGRAM_GROUP_CHAT_ID가 없으면 네트워크를 호출하면 안 된다")

    monkeypatch.setattr(telegram, "_send_text_checked", _boom)

    result = telegram.send_group_briefing("공개 본문", None, _base_group_summary(), {}, force_no_send=False)
    assert result["attempted"] is False
    assert result["skipped_reason"] == "no_group_chat_id"
    assert result["out_path"].exists()


def test_send_group_briefing_no_send_flag_never_calls_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    def _boom(*a, **k):
        raise AssertionError("네트워크를 호출하면 안 된다 (--no-send)")

    monkeypatch.setattr(telegram, "_send_text_checked", _boom)

    result = telegram.send_group_briefing("공개 본문", None, _base_group_summary(), {}, force_no_send=True)
    assert result["skipped_reason"] == "no_send"


def test_send_group_briefing_paper_mode_never_calls_network(tmp_path, monkeypatch):
    """paper 모드는 send_briefing과 마찬가지로 --no-send와 무관하게 절대 보내지 않는다(하드 가드)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    def _boom(*a, **k):
        raise AssertionError("paper 모드는 절대 텔레그램 네트워크를 호출하면 안 된다")

    monkeypatch.setattr(telegram, "_send_text_checked", _boom)

    result = telegram.send_group_briefing(
        "공개 본문", None, _base_group_summary(mode="paper"), {}, force_no_send=False
    )
    assert result["skipped_reason"] == "paper_mode"


def test_send_group_briefing_success_records_notification(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    monkeypatch.setattr(telegram, "_send_text_checked", lambda *a, **k: (True, None, None))
    monkeypatch.setattr(telegram, "_send_document_checked", lambda *a, **k: (True, None, None))

    report_path = tmp_path / "report_public_2026-09-23.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    result = telegram.send_group_briefing(
        "공개 본문", report_path, _base_group_summary(), {}, force_no_send=False
    )
    assert result["ok"] is True
    assert result["attempted"] is True

    conn = db.connect(db.db_path_for_mode("live"))
    assert db.has_notified(conn, "2026-09-23:group") is True
    conn.close()


def test_send_group_briefing_skips_when_already_notified(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    calls = []
    monkeypatch.setattr(telegram, "_send_text_checked", lambda *a, **k: calls.append(1) or (True, None, None))

    conn = db.connect(db.db_path_for_mode("live"))
    db.record_notified(conn, "2026-09-23:group", "2026-09-23T07:30:00")
    conn.close()

    result = telegram.send_group_briefing(
        "공개 본문", None, _base_group_summary(), {}, force_no_send=False
    )
    assert calls == []
    assert result["ok"] is True
    assert result["skipped_reason"] == "already_sent"


def test_send_group_briefing_migrate_to_supergroup_reports_new_id(tmp_path, monkeypatch):
    """그룹이 슈퍼그룹으로 전환되면 재시도 없이 새 chat_id를 결과에 담아야 한다."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    calls = []

    def _migrate(*a, **k):
        calls.append(1)
        return False, -1009876543210, "sendMessage: group chat was upgraded to a supergroup chat"

    monkeypatch.setattr(telegram, "_send_text_checked", _migrate)

    result = telegram.send_group_briefing(
        "공개 본문", None, _base_group_summary(), {}, force_no_send=False
    )
    assert len(calls) == 1  # 재시도 없이 한 번만 호출
    assert result["ok"] is False
    assert result["migrate_to_chat_id"] == -1009876543210


def test_send_group_briefing_failure_does_not_record_notification(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    monkeypatch.setattr(telegram, "_send_text_checked", lambda *a, **k: (False, None, "network error"))

    result = telegram.send_group_briefing(
        "공개 본문", None, _base_group_summary(), {}, force_no_send=False
    )
    assert result["ok"] is False
    conn = db.connect(db.db_path_for_mode("live"))
    assert db.has_notified(conn, "2026-09-23:group") is False
    conn.close()


# ── notify.telegram.group_send_warning_line ─────────────────────────────


def test_group_send_warning_line_none_when_ok():
    assert telegram.group_send_warning_line({"ok": True, "attempted": True}) is None


def test_group_send_warning_line_none_when_not_attempted():
    assert telegram.group_send_warning_line({"ok": False, "attempted": False, "skipped_reason": "no_group_chat_id"}) is None


def test_group_send_warning_line_includes_new_chat_id_on_migration():
    line = telegram.group_send_warning_line(
        {"ok": False, "attempted": True, "migrate_to_chat_id": -1009876543210, "error": "..."}
    )
    assert line is not None
    assert "-1009876543210" in line
    assert "TELEGRAM_GROUP_CHAT_ID" in line


def test_group_send_warning_line_generic_error():
    line = telegram.group_send_warning_line(
        {"ok": False, "attempted": True, "migrate_to_chat_id": None, "error": "sendMessage 실패(500): ..."}
    )
    assert line is not None
    assert "단체방 발송 실패" in line


# ── 단체방도 보고서 파일만 (2026-10-01) ─────────────────────────────────────


def test_send_group_briefing_sends_only_public_report_with_title_caption(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    calls = []
    monkeypatch.setattr(telegram, "_send_text_checked", lambda *a, **k: pytest.fail("본문 글을 따로 보내면 안 된다"))
    monkeypatch.setattr(
        telegram, "_send_document_checked",
        lambda token, chat_id, path, filename=None, caption=None: calls.append((path.name, caption)) or (True, None, None),
    )
    report_path = tmp_path / "report_public_2026-09-23.html"
    report_path.write_text("<html></html>", encoding="utf-8")
    text = briefing.build_public_briefing_text(_sensitive_summary(), {})

    result = telegram.send_group_briefing(text, report_path, _base_group_summary(), {}, force_no_send=False)

    assert result["ok"] is True
    assert calls == [("report_public_2026-09-23.html", text.splitlines()[0])]  # 제목 줄만, 추천·시장 온도 본문 없음


def test_send_group_briefing_document_migrate_reports_new_id(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "-100123")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")
    monkeypatch.setattr(
        telegram, "_send_document_checked", lambda *a, **k: (False, -1009876543210, "sendDocument: upgraded")
    )
    report_path = tmp_path / "report_public_2026-09-23.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    result = telegram.send_group_briefing("📊 제목", report_path, _base_group_summary(), {}, force_no_send=False)
    assert result["ok"] is False and result["migrate_to_chat_id"] == -1009876543210
    conn = db.connect(db.db_path_for_mode("live"))
    assert db.has_notified(conn, "2026-09-23:group") is False
    conn.close()
