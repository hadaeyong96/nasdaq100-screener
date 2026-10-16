"""scripts/list_telegram_chats.py 테스트 (네트워크 없음 — 가짜 응답·가짜 토큰)."""

import requests

from scripts import list_telegram_chats as lt

FAKE = "1234567890:AAH_fake-TOKEN_for_tests_only_abcdefghijk"


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code, self.ok = payload, status, status < 400

    def json(self):
        return self._p


def test_table_marks_existing_and_hides_token():
    payload = {"result": [
        {"message": {"chat": {"id": 111, "type": "private", "first_name": "개인", "last_name": "사용자"}}},
        {"message": {"chat": {"id": -100222, "type": "supergroup", "title": "투자클럽"}}},
        {"my_chat_member": {"chat": {"id": -100333, "type": "group", "title": "전략분석기"}}},
    ]}
    lines = lt.run(FAKE, "111", "-100222", get=lambda *a, **k: _Resp(payload))
    text = "\n".join(lines)
    assert FAKE not in text and "bot" not in text
    assert "| 개인 사용자 | 개인 | 111 | (기존 개인) |" in text
    assert "| 투자클럽 | 그룹(슈퍼그룹) | -100222 | (투자클럽) |" in text
    assert "| 전략분석기 | 그룹 | -100333 |  |" in text


def test_errors_show_only_kind_and_status():
    def boom(url, **k):
        raise requests.ConnectionError(f"failed for {url}")

    assert lt.run(FAKE, get=boom) == ["getUpdates 연결 실패(ConnectionError)"]
    assert lt.run(FAKE, get=lambda *a, **k: _Resp({}, 401)) == ["getUpdates 실패(HTTP 401)"]
    assert lt.run("") == ["TELEGRAM_BOT_TOKEN 없음 (.env)"]


def test_empty_list_gives_guidance():
    lines = lt.run(FAKE, get=lambda *a, **k: _Resp({"result": []}))
    assert lines[0] == "업데이트 0건, 대화방 0개" and "다시 실행하세요" in lines[-1]
