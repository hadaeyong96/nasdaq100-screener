"""개인 채팅·단체방 같은 보고서 (2026-11-12 사용자 결정) — 네트워크 없음.

두 채팅에 같은 파일, 보고서 맨 위 "검증 중 · 실제 돈 아님 · 투자 권유 아님", 실제 계좌 정보(내 돈·12월 공제·저축일)는 보고서에 없음,
12월 공제 안내는 개인 채팅에만 짧은 글로, 스위치(report.group_same_as_personal)를 끄면 예전 단체방판.
"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from engine import daily
from notify import letter_plain, report_html, telegram
from scripts import report_preview
from store import db

NOTICE = "검증 중 · 실제 돈 아님 · 투자 권유 아님"


@pytest.fixture(autouse=True)
def _allow_real_send_path(monkeypatch):
    """발송 경로(가짜 전송 함수)를 검증한다 — Actions·--send와 같은 발송 허용 상태."""
    monkeypatch.setenv("SCREENER_REAL_SEND", "1")


def _account(s: dict) -> dict:
    """가짜 실제 계좌 값(내 돈·ISA·12월 공제·저축일)."""
    s["my_money"] = {"value_krw": 5_123_456, "pnl_pct": 3.0, "remaining_krw": 7_777_777, "qty": 42, "pnl_krw": 150_000,
                     "cost_krw": 4_973_456}
    s["wallet"] = {"qqqm_state": "ok", "isa_state": "ok",
                   "isa": {"value_krw": 1_000_000, "pnl_pct": 1.0, "qty": 3, "pnl_krw": 10_000, "cost_krw": 990_000,
                           "tax_free_left_krw": 3_333_333}}
    s["dec_tax"] = {"todo": "x", "lines": ["가짜공제줄 남은 공제 250만 원"], "status": "ok", "remaining_krw": 2_500_000,
                    "deadline": None}
    s["savings_day"] = True
    return s


def _summary(cfg, pead: bool):
    s = report_preview.sample_summary(cfg)
    return _account(report_preview.sample_pead(s) if pead else s)


@pytest.mark.parametrize("pead", [True, False])
def test_render_editions_same_file_without_account(cfg, tmp_path, monkeypatch, pead):
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(report_preview, "OUT", tmp_path)
    s = _summary(cfg, pead)
    daily.render_editions(s, cfg)
    assert s["group_report_path"] == s["report_path"]
    html = open(s["report_path"], encoding="utf-8").read()
    assert NOTICE in html
    for mark in ("가짜공제줄", "333만", "777만", "■ 내 돈", "ISA에서"):  # 계좌 값·12월 공제·저축일 할 일 없음
        assert mark not in html, mark
    assert s["dec_tax"]["lines"] and s["my_money"]  # 원본 summary는 그대로(12월 글은 원본에서)


def test_switch_off_renders_separate_group_edition(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(report_preview, "OUT", tmp_path)
    c = copy.deepcopy(cfg)
    c.setdefault("report", {})["group_same_as_personal"] = False
    s = _summary(c, True)
    daily.render_editions(s, c)
    assert s["group_report_path"] != s["report_path"] and str(s["group_report_path"]).endswith("_group.html")


def test_config_default_is_same_report(cfg):
    assert cfg["report"]["group_same_as_personal"] is True


@pytest.fixture
def chats(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "dummy")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "personal")
    monkeypatch.setenv("TELEGRAM_GROUP_CHAT_ID", "group")
    monkeypatch.setattr(telegram, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "data" / "state.db")
    sent = []
    monkeypatch.setattr(telegram, "_send_document",
                        lambda token, chat, path, filename=None, caption=None: sent.append(("file", chat, path)) or True)
    monkeypatch.setattr(telegram, "_send_document_checked",
                        lambda token, chat, path, filename=None, caption=None: (sent.append(("file", chat, path)) or True, None, None))
    monkeypatch.setattr(telegram, "_send_text", lambda token, chat, body: sent.append(("text", chat, body)) or True)
    monkeypatch.setattr(telegram, "_send_text_checked", lambda token, chat, body: (sent.append(("text", chat, body)) or True, None, None))
    return sent


def test_both_chats_get_same_file_and_dec_tax_text_personal_only(cfg, tmp_path, monkeypatch, chats):
    monkeypatch.setattr(daily, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(report_preview, "OUT", tmp_path)
    s = _summary(cfg, True)
    s.update(mode="live", mode_label="실전", rebuilt_from=None)
    daily.render_editions(s, cfg)
    telegram.send_briefing("본문", s, cfg)
    telegram.send_group_briefing("본문", s["group_report_path"], s, cfg)
    assert telegram.send_dec_tax_text(s, cfg) is True
    assert telegram.send_dec_tax_text(s, cfg) is False  # 같은 기준일 두 번째는 건너뜀
    files = [(chat, path) for kind, chat, path in chats if kind == "file"]
    assert [c for c, _ in files] == ["personal", "group"] and files[0][1] == files[1][1]
    texts = [(chat, body) for kind, chat, body in chats if kind == "text"]
    assert len(texts) == 1 and texts[0][0] == "personal" and texts[0][1].startswith("[12월 공제 안내]")
    assert "가짜공제줄" in texts[0][1]


def test_dec_tax_text_none_without_notice_day_and_paper_mode(cfg, chats):
    assert telegram.dec_tax_text({"as_of": pd.Timestamp("2026-11-11")}, cfg) is None
    s = _account({"as_of": pd.Timestamp("2026-12-01"), "mode": "paper"})
    assert telegram.send_dec_tax_text(s, cfg) is False and chats == []
    assert telegram.send_dec_tax_text({**s, "mode": "live"}, cfg, force_no_send=True) is False and chats == []


def test_dec_tax_text_dry_run_locally(cfg, chats, monkeypatch):
    monkeypatch.delenv("SCREENER_REAL_SEND", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    s = _account({"as_of": pd.Timestamp("2026-12-01"), "mode": "live"})
    assert telegram.send_dec_tax_text(s, cfg) is False and chats == []


def test_without_account_drops_only_account_keys():
    s = {"as_of": 1, "my_money": 2, "wallet": 3, "dec_tax": 4, "savings_day": True, "pead_watch": 5}
    assert report_html.without_account(s) == {"as_of": 1, "pead_watch": 5} and "my_money" in s
    assert letter_plain.NOTICE == NOTICE
