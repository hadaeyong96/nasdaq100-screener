"""notify/telegram.py, notify/briefing.py 테스트. 네트워크 없음.

- 4096자 분할, 토큰 없을 때 파일로만 저장, 같은 기준일 중복 발송 방지.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from notify import briefing, telegram
from store import db


def test_split_message_returns_single_part_when_short():
    assert telegram.split_message("짧은 글") == ["짧은 글"]


def test_split_message_splits_long_text_under_limit():
    text = "\n".join(f"줄 {i}" for i in range(2000))
    parts = telegram.split_message(text, limit=100)
    assert len(parts) > 1
    assert all(len(p) <= 100 for p in parts)


def test_split_message_force_splits_single_long_line():
    text = "가" * 250
    parts = telegram.split_message(text, limit=100)
    assert len(parts) == 3
    assert all(len(p) <= 100 for p in parts)


def test_send_briefing_without_token_only_saves_file(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    summary = {"mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23")}
    path = telegram.send_briefing("본문 텍스트", summary, {}, force_no_send=False)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == "본문 텍스트"
    assert path.parent.name == "outputs"


def test_send_briefing_no_send_flag_never_calls_network(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    def _boom(*args, **kwargs):
        raise AssertionError("네트워크를 호출하면 안 된다 (--no-send)")

    monkeypatch.setattr(telegram, "_send_text", _boom)
    monkeypatch.setattr(telegram, "_send_document", _boom)

    summary = {"mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23")}
    path = telegram.send_briefing("본문", summary, {}, force_no_send=True)
    assert path.exists()


def test_send_briefing_skips_when_already_notified(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)

    calls = []
    monkeypatch.setattr(telegram, "_send_text", lambda *a, **k: calls.append("text") or True)
    monkeypatch.setattr(telegram, "_send_document", lambda *a, **k: calls.append("doc") or True)

    # 실제 접속하는 DB 경로 대신 tmp_path/data/state.db를 쓰도록 db 모듈을 바꿔치기한다.
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")

    summary = {"mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-23")}
    conn = db.connect(db.db_path_for_mode("live"))
    db.record_notified(conn, "2026-09-23", "2026-09-23T07:30:00")
    conn.close()

    telegram.send_briefing("본문", summary, {}, force_no_send=False)
    assert calls == []  # 이미 발송한 기록이 있어 아무것도 호출하지 않는다


def _base_summary(**overrides):
    summary = {
        "mode": "live",
        "mode_label": "실전",
        "as_of": pd.Timestamp("2026-09-23"),  # 수요일
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []},
        "buy_count": 0,
        "sell_rows": [],
        "hold_rows": [],
        "unfilled_rows": [],
    }
    summary.update(overrides)
    return summary


def test_build_briefing_text_no_signals_is_one_line():
    text = briefing.build_briefing_text(_base_summary(), {})
    assert "오늘 매매 신호 없음" in text
    assert "🟢" not in text and "🔴" not in text
    assert "9/23(수)" in text
    assert "📎" in text


def test_build_briefing_text_buy_only_shows_ticker_and_stage():
    summary = _base_summary(
        buy_groups={"b1": [{"ticker": "CMCSA", "kr": "컴캐스트"}], "b2": [], "b3": [], "b9": [{"ticker": "PLTR", "kr": "팔란티어"}]},
        buy_count=2,
    )
    text = briefing.build_briefing_text(summary, {})
    assert "🟢 매수 2 · CMCSA(1차) PLTR(재진입)" in text
    assert "🔴 매도 0" in text
    assert "컴캐스트" not in text  # 종목은 티커만


def test_build_briefing_text_sell_only_shows_stop_label():
    """P3.5 5번: 텔레그램 매도 줄의 손절은 "예약 체결 확인"까지 붙여 보여준다."""
    summary = _base_summary(
        sell_rows=[{"티커": "ROP", "kind": "STOP", "신호": "손절", "매도범위": "전량"}],
    )
    text = briefing.build_briefing_text(summary, {})
    assert "🟢 매수 0" in text
    assert "🔴 매도 1 · ROP(손절·예약 체결 확인)" in text


def test_build_briefing_text_sell_labels_use_kind_e_numbers():
    """일반 매도 줄 표기는 1차분(E1)/2차분(E2)/3차분(E3)/전량 (기존 지시문 — Telegram 전용)."""
    summary = _base_summary(
        sell_rows=[
            {"티커": "AAA", "kind": "E1", "신호": "모멘텀 약화(E1, 데드크로스)", "매도범위": "1차분 매도(11%)"},
            {"티커": "BBB", "kind": "E2", "신호": "추세 약화(E2, RSI 50 이탈)", "매도범위": "2차분 매도(22%)"},
            {"티커": "CCC", "kind": "E3", "신호": "구조 붕괴(E3)", "매도범위": "3차분(67%) 또는 잔량"},
            {"티커": "DDD", "kind": "A1_EXPIRE", "신호": "A1 만료", "매도범위": "1차분 (전량)"},
        ],
    )
    text = briefing.build_briefing_text(summary, {})
    assert "AAA(1차분(E1))" in text
    assert "BBB(2차분(E2))" in text
    assert "CCC(3차분(E3))" in text
    assert "DDD(전량)" in text


def test_build_briefing_text_stop_alerts_line_only_when_present():
    summary = _base_summary(
        stop_alerts=[
            {"티커": "AMZN", "종목명": "아마존", "type": "근접"},
            {"티커": "TSLA", "종목명": "테슬라", "type": "변경"},
            {"티커": "CMCSA", "종목명": "컴캐스트", "type": "신규"},
        ]
    )
    text = briefing.build_briefing_text(summary, {})
    assert "🛡️ 손절 예약 · AMZN(근접) TSLA(변경) CMCSA(신규)" in text

    text_none = briefing.build_briefing_text(_base_summary(), {})
    assert "🛡️" not in text_none


def test_build_briefing_text_unfilled_line_only_when_present():
    summary = _base_summary(unfilled_rows=[{"티커": "CMCSA", "종목명": "컴캐스트"}])
    text = briefing.build_briefing_text(summary, {})
    assert "⚠️ 미체결 1 · CMCSA" in text

    text_none = briefing.build_briefing_text(_base_summary(), {})
    assert "⚠️" not in text_none


def test_report_attachment_name_uses_korean_title_and_date():
    assert telegram.report_attachment_name("2026-09-23") == "나스닥100_2026-09-23.html"


def test_resend_last_no_send_flag_never_calls_network(tmp_path, monkeypatch):
    text_path = tmp_path / "telegram_2026-09-23.txt"
    text_path.write_text("본문", encoding="utf-8")
    report_path = tmp_path / "report_2026-09-23.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    def _boom(*args, **kwargs):
        raise AssertionError("네트워크를 호출하면 안 된다 (--no-send)")

    monkeypatch.setattr(telegram, "_send_text", _boom)
    monkeypatch.setattr(telegram, "_send_document", _boom)

    assert telegram.resend_last(report_path, text_path, "2026-09-23", force_no_send=True) is True


def test_resend_last_missing_text_file_fails(tmp_path):
    ok = telegram.resend_last(tmp_path / "no_report.html", tmp_path / "no_text.txt", "2026-09-23")
    assert ok is False


def test_env_status_reports_existence_and_length_never_value(tmp_path, monkeypatch):
    """P3.5 보완: .env 로드 진단은 존재 여부·길이만 말하고 값은 절대 포함하지 않는다."""
    monkeypatch.setattr(telegram, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "super-secret-token-value")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")

    status = telegram.env_status()

    assert "super-secret-token-value" not in status
    assert "없음" in status  # tmp_path/.env는 실제로 없다
    assert f"길이 {len('super-secret-token-value')}" in status
    assert "길이 5" in status


def test_load_dotenv_from_real_env_file_populates_environment(tmp_path, monkeypatch):
    """P3.5 보완: .env가 실제로 존재하고 토큰 줄이 있으면 os.environ에 반영돼야 한다
    (P3.4에서 ".env missing"으로 잘못 보고됐던 경로에 대한 회귀 테스트)."""
    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=dummy-token\nTELEGRAM_CHAT_ID=999\n", encoding="utf-8")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    from dotenv import load_dotenv

    ok = load_dotenv(env_file, override=True)

    assert ok is True
    assert os.environ.get("TELEGRAM_BOT_TOKEN") == "dummy-token"
    assert os.environ.get("TELEGRAM_CHAT_ID") == "999"


def test_resend_last_ignores_dedup_record(tmp_path, monkeypatch):
    """--resend는 이미 발송 기록이 있어도(has_notified) 다시 보낸다 — dedup 확인을 하지 않는다."""
    text_path = tmp_path / "telegram_2026-09-23.txt"
    text_path.write_text("본문", encoding="utf-8")
    report_path = tmp_path / "report_2026-09-23.html"
    report_path.write_text("<html></html>", encoding="utf-8")

    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "dummy")
    calls = []
    monkeypatch.setattr(telegram, "_send_text", lambda *a, **k: calls.append("text") or True)
    monkeypatch.setattr(telegram, "_send_document", lambda *a, **k: calls.append("doc") or True)

    ok = telegram.resend_last(report_path, text_path, "2026-09-23", force_no_send=False)
    assert ok is True
    assert calls == ["text", "doc"]
