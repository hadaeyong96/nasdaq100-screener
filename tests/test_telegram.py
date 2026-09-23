"""notify/telegram.py, notify/briefing.py 테스트. 네트워크 없음.

- 4096자 분할, 토큰 없을 때 파일로만 저장, 같은 기준일 중복 발송 방지.
"""

from __future__ import annotations

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


def test_build_briefing_text_lists_empty_sections_as_none():
    summary = {
        "mode": "live",
        "mode_label": "실전",
        "as_of": pd.Timestamp("2026-09-23"),
        "buy_groups": {"b1": [], "b2": [], "b3": [], "b9": []},
        "buy_count": 0,
        "sell_rows": [],
        "hold_rows": [],
        "unfilled_rows": [],
    }
    text = briefing.build_briefing_text(summary, {})
    assert "매수 없음" in text
    assert "매도·손절 없음" in text
    assert "상세는 첨부 보고서" in text
