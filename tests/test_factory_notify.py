"""factory/notify.py 테스트 (네트워크 없음)."""

import pytest

from factory import notify as fn


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in (fn.CHAT_ENV, fn.TOKEN_ENV, fn.TOKEN_FALLBACK_ENV, fn.GROUP_ENV):
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("event,s", [
    ("explore_done", {"name": "강한 업종 첫 눌림", "verdict": "탈락", "cagr": 16.29, "qqq_cagr": 19.98, "random_pct": 68.8,
                      "candidate_id": "explore_a"}),
    ("zone_judged", {"name": "RS26", "verdict": "FAIL", "zone": "검증 ③", "budget_left": "3/5"}),
    ("scorecard_done", {"month": "2026-10", "stop": ["한 달 가속 따라하기"], "review": ["QLD 200일선 전환"]}),
    ("scorecard_done", {"month": "2026-10"}),
    ("error", {"stage": "월간 성적표 계산", "detail": "Traceback 비밀"}),
])
def test_message_format(event, s):
    msg = fn.build_message(event, s)
    lines = msg.split("\n")
    assert lines[0].startswith("[전략공장] ") and lines[-1] == "→ Claude에 결과 보고서를 붙여 넣으세요"
    assert len(lines) <= 6 and lines[-2].startswith("다음 대기 후보: ")
    assert all(ord(ch) < 0x1F000 for ch in msg)  # 이모지 없음
    assert "Traceback" not in msg and "비밀" not in msg  # 오류 상세 금지


def test_next_candidate_follows_catalog_rank():
    assert "탐색 A" in fn.next_candidate()
    assert fn.next_candidate("explore_a") == "주식 수 감소 기업"
    assert "주식 수 감소 기업" in fn.build_message("explore_done", {"candidate_id": "explore_a"})


def test_skips_without_chat_id(capsys):
    sent = []
    assert fn.send("error", {"stage": "x"}, post=lambda *a: sent.append(a)) is False
    assert sent == [] and "건너뜁니다" in capsys.readouterr().out


def test_never_sends_to_group_chat(monkeypatch):
    monkeypatch.setenv(fn.CHAT_ENV, "-100123")
    monkeypatch.setenv(fn.TOKEN_FALLBACK_ENV, "tok")
    monkeypatch.setenv(fn.GROUP_ENV, "-100123")
    sent = []
    with pytest.raises(fn.FactoryNotifyForbidden):
        fn.send("error", {"stage": "x"}, post=lambda *a: sent.append(a))
    assert fn.send_safely("error", {"stage": "x"}, post=lambda *a: sent.append(a)) is False
    assert sent == []


def test_sends_to_factory_chat_and_failure_is_contained(monkeypatch, capsys):
    monkeypatch.setenv(fn.CHAT_ENV, "-100999")
    monkeypatch.setenv(fn.TOKEN_FALLBACK_ENV, "secret-token-value")
    monkeypatch.setenv(fn.GROUP_ENV, "-100123")
    sent = []
    assert fn.send("explore_done", {"name": "a"}, post=lambda t, c, x: sent.append((t, c, x))) is True
    assert sent[0][1] == "-100999"

    def bad(*a):
        raise RuntimeError("https://api.telegram.org/botsecret-token-value/sendMessage 연결 실패")

    assert fn.send_safely("explore_done", {"name": "a"}, post=bad) is False
    out = capsys.readouterr().out
    assert "secret-token-value" not in out and "-100999" not in out  # 값 출력 금지


def test_cli_dry_run(tmp_path, capsys):
    p = tmp_path / "s.json"
    p.write_text('{"name": "탐색 B", "verdict": "탈락", "cagr": 16.29, "qqq_cagr": 19.98, "random_pct": 68.8}',
                 encoding="utf-8")
    assert fn.main(["--event", "explore_done", "--summary-file", str(p), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[전략공장] 탐색 시험 완료" in out and "16.29% vs QQQ 19.98%" in out
