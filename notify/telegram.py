"""텔레그램 발송 (P3, 4번).

토큰(.env의 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)이 없으면 보내지 않고
outputs/telegram_YYYY-MM-DD.txt에 저장만 한다. 에러로 멈추지 않는다.
4096자를 넘으면 나눠 보내고, 실패하면 3번 재시도한다. 같은 기준일 중복 발송은
store.db의 notifications 테이블로 막는다(성공적으로 다 보낸 뒤에만 기록한다).

데이터 지연 모드(P3.2 2번)에는 `send_delay_notice`로 알림 한 통만 보내고
(보고서 첨부 없음), 일반 브리핑(`send_briefing`)은 부르지 않는다.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from store import db

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
# override=True: 이 프로세스의 OS 환경변수에 같은 이름의 빈 값이 이미 있어도
# .env의 값으로 덮어쓴다 — 아니면 .env가 있어도 토큰이 반영되지 않을 수 있다.
load_dotenv(ENV_PATH, override=True)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4096
MAX_RETRIES = 3


def split_message(text: str, limit: int = MAX_LEN) -> list[str]:
    """긴 글을 limit자 이하 여러 통으로 나눈다. 줄 단위로 자르고, 한 줄이 limit보다
    길면 그 줄만 강제로 잘게 나눈다 (순수 함수, 네트워크 없음 — 테스트 가능)."""
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
            current = ""
        while len(line) > limit:
            parts.append(line[:limit])
            line = line[limit:]
        current = line
    if current:
        parts.append(current)
    return parts


def _request_with_retry(request_fn, description: str) -> bool:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = request_fn()
            resp.raise_for_status()
            return True
        except Exception as exc:
            print(f"[telegram] {description} 실패(시도 {attempt}/{MAX_RETRIES}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(1.5 * attempt)
    return False


def _send_text(token: str, chat_id: str, text: str) -> bool:
    url = TELEGRAM_API.format(token=token, method="sendMessage")
    return _request_with_retry(
        lambda: requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20),
        "sendMessage",
    )


def _send_document(token: str, chat_id: str, path: Path, filename: str | None = None) -> bool:
    url = TELEGRAM_API.format(token=token, method="sendDocument")
    display_name = filename or path.name

    def _do():
        with open(path, "rb") as f:
            return requests.post(url, data={"chat_id": chat_id}, files={"document": (display_name, f)}, timeout=60)

    return _request_with_retry(_do, "sendDocument")


def env_status() -> str:
    """텔레그램 토큰 관련 환경변수 진단 문구 (P3.2 0번 — 존재 여부와 길이만, 값은 금지).

    ".env 없음" 오진단(P3.5 보완)을 막기 위해 실제로 확인한 사실만 말한다:
    .env 파일 존재 여부와 각 환경변수의 글자 수. 값 자체는 절대 포함하지 않는다.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    return (
        f".env 파일: {'있음' if ENV_PATH.exists() else '없음'}({ENV_PATH}) · "
        f"TELEGRAM_BOT_TOKEN 길이 {len(token)} · TELEGRAM_CHAT_ID 길이 {len(chat_id)}"
    )


def report_attachment_name(as_of_str: str) -> str:
    """휴대폰 파일 목록에서 알아보기 쉬운 첨부 파일 이름 (P3.4 2번): 나스닥100_YYYY-MM-DD.html"""
    return f"나스닥100_{as_of_str}.html"


def send_briefing(text: str, summary: dict, cfg: dict, force_no_send: bool = False) -> Path:
    """브리핑을 보낸다(토큰 있으면). 항상 outputs/telegram_YYYY-MM-DD.txt에 본문을 남긴다.

    입력: text(본문, notify.briefing.build_briefing_text 결과), summary(engine의 결과 —
         as_of, mode, report_path 포함), cfg, force_no_send(--no-send 플래그)
    출력: 저장한 txt 파일 경로
    """
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"telegram_{as_of_str}.txt"
    out_path.write_text(text, encoding="utf-8")

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""

    if force_no_send:
        return out_path
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 발송하지 않고 파일로만 저장합니다. ({env_status()})")
        return out_path

    mode = summary.get("mode", "live")
    conn = db.connect(db.db_path_for_mode(mode))
    try:
        if db.has_notified(conn, as_of_str):
            print(f"[telegram] {as_of_str} 기준 이미 발송한 기록이 있어 다시 보내지 않습니다.")
            return out_path

        ok = all(_send_text(token, chat_id, chunk) for chunk in split_message(text))
        report_path = summary.get("report_path")
        if report_path and Path(report_path).exists():
            time.sleep(1)
            ok = _send_document(token, chat_id, Path(report_path), report_attachment_name(as_of_str)) and ok

        if ok:
            db.record_notified(conn, as_of_str, datetime.now().isoformat(timespec="seconds"))
        else:
            print("[telegram] 일부 발송에 실패해 발송 기록을 남기지 않습니다 (다음 실행에서 재시도 가능).")
    finally:
        conn.close()
    return out_path


def send_delay_notice(summary: dict, cfg: dict, force_no_send: bool = False) -> Path:
    """데이터 지연 모드(P3.2 2번) 알림 한 통만 보낸다. 보고서는 첨부하지 않는다.

    입력: summary(engine 결과 — mode, expected_date, actual_date 문자열 포함), cfg,
         force_no_send(--no-send 플래그)
    출력: 저장한 txt 파일 경로
    """
    expected = summary.get("expected_date") or "알수없음"
    actual = summary.get("actual_date") or "알수없음"
    text = f"데이터 지연: 기대 기준일 {expected}, 실제 {actual}. 오늘은 매매 신호 없음\n"

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"telegram_{actual}_delay.txt"
    out_path.write_text(text, encoding="utf-8")

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""

    if force_no_send:
        return out_path
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 발송하지 않고 파일로만 저장합니다. ({env_status()})")
        return out_path

    mode = summary.get("mode", "live")
    conn = db.connect(db.db_path_for_mode(mode))
    try:
        dedup_key = f"{actual}:delay"
        if db.has_notified(conn, dedup_key):
            print(f"[telegram] {actual} 지연 알림을 이미 보낸 기록이 있어 다시 보내지 않습니다.")
            return out_path

        if _send_text(token, chat_id, text):
            db.record_notified(conn, dedup_key, datetime.now().isoformat(timespec="seconds"))
        else:
            print("[telegram] 지연 알림 발송에 실패했습니다 (다음 실행에서 재시도 가능).")
    finally:
        conn.close()
    return out_path


def resend_last(report_path: Path, text_path: Path, as_of_str: str, force_no_send: bool = False) -> bool:
    """--resend(P3.4 3번): 상태를 다시 처리하지 않고, 이미 만들어 둔 보고서·글
    파일을 다시 보낸다. 중복 발송 방지 기록(store.db)은 확인하지도, 남기지도 않는다.

    입력: report_path(outputs/report_YYYY-MM-DD.html), text_path(outputs/telegram_YYYY-MM-DD.txt),
         as_of_str(첨부 파일 이름에 쓸 기준일), force_no_send(--no-send 플래그)
    출력: 발송(또는 --no-send 처리) 성공 여부
    """
    if not text_path.exists():
        print(f"[telegram] {text_path}가 없어 다시 보낼 수 없습니다.")
        return False
    text = text_path.read_text(encoding="utf-8")

    if force_no_send:
        print(f"[telegram] --no-send: 재발송하지 않습니다 ({text_path.name}).")
        return True

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 재발송할 수 없습니다. ({env_status()})")
        return False

    ok = all(_send_text(token, chat_id, chunk) for chunk in split_message(text))
    if report_path.exists():
        time.sleep(1)
        ok = _send_document(token, chat_id, report_path, report_attachment_name(as_of_str)) and ok
    else:
        print(f"[telegram] {report_path}가 없어 보고서 없이 글만 보냅니다.")
    return ok
