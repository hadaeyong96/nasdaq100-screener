"""텔레그램 로그에 봇 토큰이 나오지 않는지 (2026-10-16 보안 작업, 네트워크 없음 — 가짜 토큰)."""

import pytest
import requests

from notify import telegram


@pytest.fixture(autouse=True)
def _allow_real_send_path(monkeypatch):
    """이 파일은 발송 경로(가짜 전송 함수)를 검증한다 — Actions·--send와 같은 발송 허용 상태로 둔다(2026-10-17 관문)."""
    monkeypatch.setenv("SCREENER_REAL_SEND", "1")


FAKE = "1234567890:AAH_fake-TOKEN_for_tests_only_abcdefghijk"
URL = f"https://api.telegram.org/bot{FAKE}/sendMessage"


def test_redact_patterns():
    assert telegram.redact(URL) == "https://api.telegram.org/bot***/sendMessage"
    assert telegram.redact(f"token {FAKE} end") == "token *** end"
    assert telegram.redact("10:30 시각, chat -1001234567890") == "10:30 시각, chat -1001234567890"  # 시각·chat id는 그대로


class _Resp:
    def __init__(self, status=400, text=f"body with {URL}"):
        self.status_code, self.text, self.ok = status, text, status < 400

    def json(self):
        return {}

    def raise_for_status(self):
        err = requests.HTTPError(f"{self.status_code} Client Error: Bad Request for url: {URL}")
        err.response = self
        raise err


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(telegram.time, "sleep", lambda s: None)


def test_retry_wrapper_hides_token_from_http_error(capsys):
    assert telegram._request_with_retry(lambda: _Resp(), "sendMessage") is False
    out = capsys.readouterr().out
    assert FAKE not in out and "HTTPError HTTP 400" in out


def test_checked_wrapper_hides_token_from_connection_error_and_body(capsys):
    def boom():
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: /bot{FAKE}/sendMessage")

    ok, _, err = telegram._post_checked(boom, "sendMessage")
    assert not ok and FAKE not in err and err.endswith("ConnectionError")
    ok, _, err = telegram._post_checked(lambda: _Resp(403), "sendMessage")
    assert not ok and err == "sendMessage 실패(HTTP 403)"  # 응답 본문은 남기지 않는다
    assert FAKE not in capsys.readouterr().out


def test_alert_paths_never_print_or_send_token(monkeypatch, capsys):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", FAKE)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "-1009999")
    sent = []

    def fake_post(url, data=None, timeout=None, **kw):
        sent.append(data["text"])
        raise requests.ConnectionError(f"failed for {url}")

    monkeypatch.setattr(telegram.requests, "post", fake_post)
    assert telegram.notify_ops_error(f"단체방 발송 실패: {URL}") is False
    assert telegram.send_alert_text(f"전환 알림 {FAKE}") is False
    out = capsys.readouterr().out
    assert FAKE not in out
    assert sent and all(FAKE not in t for t in sent)  # 보내는 글에서도 가린다


def test_no_raw_exception_prints_left_in_telegram_module():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "notify" / "telegram.py").read_text(encoding="utf-8")
    assert "{exc}" not in src and "resp.text[:" not in src
    assert src.count("print(") == 1  # _log 한 곳만
