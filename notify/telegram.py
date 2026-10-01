"""텔레그램 발송 (P3, 4번).

텔레그램에는 HTML 보고서 파일 한 통만 보낸다 — 본문 글(브리핑 텍스트)은 보내지
않고, 제목 줄(+정정본·경고 줄)만 첨부 설명(caption)으로 붙인다(2026-10-01 사용자
확정, `report_caption`). 보고서 파일이 없을 때만 본문 글을 대신 보낸다.

토큰(.env의 TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)이 없으면 보내지 않고
outputs/telegram_{모드}_YYYY-MM-DD.txt에 본문을 저장만 한다. 에러로 멈추지 않는다.
글을 보낼 때 4096자를 넘으면 나눠 보내고, 실패하면 3번 재시도한다. 같은 기준일 중복
발송은 store.db의 notifications 테이블로 막는다(성공적으로 다 보낸 뒤에만 기록한다).

데이터 지연 모드(P3.2 2번)에는 `send_delay_notice`로 지연 배너가 붙은 보고서 한 통만
(지연 문구를 첨부 설명으로) 보내고, 일반 브리핑(`send_briefing`)은 부르지 않는다.

모의(paper) 모드는 두 함수 모두 --no-send 여부와 무관하게 절대 실제로 보내지
않는다(파일 저장까지만 한다) — 모의를 실전과 매일 나란히 돌리기 시작하면서 생긴
하드 가드다.

## 단체방(투자클럽) 공개 발송

`send_group_briefing`은 TELEGRAM_GROUP_CHAT_ID(선택, 없으면 조용히 건너뜀)로
공개용 보고서 파일 한 통만 보낸다(제목 줄 첨부 설명, 본문 글은 보고서가 없을 때만). paper 모드는
send_briefing과 같은 이유로 절대 보내지 않는다(하드 가드). 실패해도 예외를
던지지 않고 결과 dict로 돌려준다 — 호출부(engine.daily)가 개인 채팅에 경고 한
줄만 남기고 실행을 계속하기 위해서다. 그룹이 슈퍼그룹으로 전환돼 텔레그램이
'group chat was upgraded to a supergroup'(migrate_to_chat_id) 오류를 주면
재시도하지 않고 새 chat_id를 결과에 담는다 — `group_send_warning_line`이 그
값을 경고 문구에 포함시킨다.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from notify.briefing import report_titles
from store import db

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
# override=True: 이 프로세스의 OS 환경변수에 같은 이름의 빈 값이 이미 있어도
# .env의 값으로 덮어쓴다 — 아니면 .env가 있어도 토큰이 반영되지 않을 수 있다.
load_dotenv(ENV_PATH, override=True)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4096
MAX_CAPTION_LEN = 1024  # 텔레그램 sendDocument caption 한도
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


def report_caption(text: str, limit: int = MAX_CAPTION_LEN) -> str:
    """브리핑 본문 -> 보고서 첨부에 붙일 짧은 설명 (순수 함수).

    텔레그램은 보고서 파일 한 통만 보낸다(본문 글은 보내지 않는다, 2026-10-01 사용자 확정).
    첨부만 봐도 무슨 파일인지 알 수 있게 본문 첫 줄(📊 제목 · 날짜 · 모드)과, 놓치면
    안 되는 정정본(🔁)·오류 경고(⚠️) 줄만 남긴다. limit자를 넘으면 자른다.
    출력 예: "📊 데이터브리핑 · 9/30(수) 마감 · 실전"
    """
    lines = [line for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return ""
    kept = [lines[0]] + [line for line in lines[1:] if line.startswith(("🔁", "⚠️"))]
    caption = "\n".join(kept)
    return caption if len(caption) <= limit else caption[: limit - 1] + "…"


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


def _document_data(chat_id: str, caption: str | None) -> dict:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
    return data


def _send_document(
    token: str, chat_id: str, path: Path, filename: str | None = None, caption: str | None = None
) -> bool:
    url = TELEGRAM_API.format(token=token, method="sendDocument")
    display_name = filename or path.name

    def _do():
        with open(path, "rb") as f:
            return requests.post(
                url, data=_document_data(chat_id, caption), files={"document": (display_name, f)}, timeout=60
            )

    return _request_with_retry(_do, "sendDocument")


def _extract_migrate_id(resp) -> int | None:
    """텔레그램 오류 응답 본문에서 migrate_to_chat_id(슈퍼그룹 전환 새 chat_id)를 꺼낸다."""
    try:
        body = resp.json()
    except Exception:
        return None
    return (body.get("parameters") or {}).get("migrate_to_chat_id")


def _post_checked(request_fn, description: str) -> tuple[bool, int | None, str | None]:
    """단체방 발송 전용 재시도 래퍼 (P-group 지시문).

    슈퍼그룹 전환 오류(migrate_to_chat_id)는 재시도해도 chat_id가 바뀌지 않는 한
    똑같이 실패하므로 즉시 멈추고 새 chat_id를 돌려준다. 그 밖의 실패는 기존
    _request_with_retry와 같이 MAX_RETRIES번 재시도한다.

    출력: (성공 여부, migrate_to_chat_id 또는 None, 마지막 오류 설명 또는 None)
    """
    last_error: str | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = request_fn()
            if resp.ok:
                return True, None, None
            migrate_id = _extract_migrate_id(resp)
            if migrate_id:
                return False, migrate_id, f"{description}: group chat was upgraded to a supergroup chat"
            last_error = f"{description} 실패({resp.status_code}): {resp.text[:200]}"
        except Exception as exc:
            last_error = f"{description} 실패: {exc}"
        print(f"[telegram] {last_error} (시도 {attempt}/{MAX_RETRIES})")
        if attempt < MAX_RETRIES:
            time.sleep(1.5 * attempt)
    return False, None, last_error


def _send_text_checked(token: str, chat_id: str, text: str) -> tuple[bool, int | None, str | None]:
    url = TELEGRAM_API.format(token=token, method="sendMessage")
    return _post_checked(
        lambda: requests.post(url, data={"chat_id": chat_id, "text": text}, timeout=20), "sendMessage"
    )


def _send_document_checked(
    token: str, chat_id: str, path: Path, filename: str | None = None, caption: str | None = None
) -> tuple[bool, int | None, str | None]:
    url = TELEGRAM_API.format(token=token, method="sendDocument")
    display_name = filename or path.name

    def _do():
        with open(path, "rb") as f:
            return requests.post(
                url, data=_document_data(chat_id, caption), files={"document": (display_name, f)}, timeout=60
            )

    return _post_checked(_do, "sendDocument")


def env_status() -> str:
    """텔레그램 토큰 관련 환경변수 진단 문구 (P3.2 0번 — 존재 여부와 길이만, 값은 금지).

    ".env 없음" 오진단(P3.5 보완)을 막기 위해 실제로 확인한 사실만 말한다:
    .env 파일 존재 여부와 각 환경변수의 글자 수. 값 자체는 절대 포함하지 않는다.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    group_chat_id = os.environ.get("TELEGRAM_GROUP_CHAT_ID") or ""
    return (
        f".env 파일: {'있음' if ENV_PATH.exists() else '없음'}({ENV_PATH}) · "
        f"TELEGRAM_BOT_TOKEN 길이 {len(token)} · TELEGRAM_CHAT_ID 길이 {len(chat_id)} · "
        f"TELEGRAM_GROUP_CHAT_ID 길이 {len(group_chat_id)}"
    )


def notify_ops_error(message: str) -> bool:
    """운영 자동화(예약 작업 등)가 실패했을 때 텔레그램으로 짧은 오류 메시지만 보낸다.

    입력: message(호출한 쪽이 만든 짧은 오류 설명)
    출력: 발송 성공 여부(토큰이 없으면 False — 값은 절대 출력하지 않고 길이만 확인한다)

    live/paper 브리핑과 달리 모드 구분·중복 발송 방지·파일 저장이 없다 — 실행 자체가
    실패했다는 운영 알림이라 매번 보내는 게 맞고, 로그는 호출한 쪽(run_daily.ps1)이
    이미 남긴다.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 오류 알림을 보낼 수 없습니다. ({env_status()})")
        return False
    return all(_send_text(token, chat_id, chunk) for chunk in split_message(message))


def report_attachment_name(as_of_str: str, cfg: dict | None = None) -> str:
    """휴대폰 파일 목록에서 알아보기 쉬운 첨부 파일 이름 (P3.4 2번).

    입력: as_of_str(기준일 YYYY-MM-DD), cfg(config.yaml — report.short_title을 쓴다)
    출력: "{short_title}_YYYY-MM-DD.html" (예: 데이터브리핑_2026-09-23.html)
    """
    return f"{report_titles(cfg)['short_title']}_{as_of_str}.html"


def send_briefing(text: str, summary: dict, cfg: dict, force_no_send: bool = False) -> Path:
    """보고서 HTML 파일 한 통만 보낸다(토큰 있으면). 항상 outputs/telegram_{모드}_YYYY-MM-DD.txt에 본문을 남긴다.

    입력: text(본문, notify.briefing.build_briefing_text 결과), summary(engine의 결과 —
         as_of, mode, report_path 포함), cfg, force_no_send(--no-send 플래그)
    출력: 저장한 txt 파일 경로

    본문 글은 따로 보내지 않고, report_caption(text)(제목 줄 + 정정본·경고 줄)을 첨부
    설명으로만 붙인다(2026-10-01 사용자 확정). 보고서 파일이 없을 때만 본문 글을 대신
    보낸다 — 아무것도 못 받는 날이 없게 하기 위해서다.

    모의(paper) 모드는 --no-send 여부와 관계없이 절대 텔레그램을 보내지 않는다(하드
    가드) — 모의 모드를 실전과 매일 나란히 돌리기 시작하면서, 플래그를 깜빡해도
    안전하도록 모드 자체로 막는다. 파일 저장은 모드와 무관하게 항상 한다.
    """
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    mode = summary.get("mode", "live")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"telegram_{mode}_{as_of_str}.txt"
    out_path.write_text(text, encoding="utf-8")

    if force_no_send or mode == "paper":
        return out_path

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 발송하지 않고 파일로만 저장합니다. ({env_status()})")
        return out_path

    conn = db.connect(db.db_path_for_mode(mode))
    try:
        if db.has_notified(conn, as_of_str) and not summary.get("rebuilt_from"):
            print(f"[telegram] {as_of_str} 기준 이미 발송한 기록이 있어 다시 보내지 않습니다.")
            return out_path

        report_path = summary.get("report_path")
        if report_path and Path(report_path).exists():
            ok = _send_document(
                token, chat_id, Path(report_path), report_attachment_name(as_of_str, cfg), caption=report_caption(text)
            )
        else:
            print("[telegram] 보고서 파일이 없어 본문 글을 대신 보냅니다.")
            ok = all(_send_text(token, chat_id, chunk) for chunk in split_message(text))

        if ok:
            db.record_notified(conn, as_of_str, datetime.now().isoformat(timespec="seconds"))
        else:
            print("[telegram] 일부 발송에 실패해 발송 기록을 남기지 않습니다 (다음 실행에서 재시도 가능).")
    finally:
        conn.close()
    return out_path


def send_delay_notice(summary: dict, cfg: dict, force_no_send: bool = False) -> Path:
    """데이터 지연 모드(P3.2 2번) 알림 한 통만 보낸다.

    지연 배너가 붙은 보고서(summary["report_path"])가 있으면 그 파일을 지연 문구를
    첨부 설명으로 붙여 한 통으로 보내고(보고서 파일만 보내는 방식, 2026-10-01),
    없으면 지연 문구 글만 보낸다.

    입력: summary(engine 결과 — mode, expected_date, actual_date 문자열 포함), cfg,
         force_no_send(--no-send 플래그)
    출력: 저장한 txt 파일 경로

    모의(paper) 모드는 send_briefing과 마찬가지로 --no-send와 무관하게 절대 보내지
    않는다(하드 가드).
    """
    expected = summary.get("expected_date") or "알수없음"
    actual = summary.get("actual_date") or "알수없음"
    text = f"데이터 지연: 기대 기준일 {expected}, 실제 {actual}. 오늘은 매매 신호 없음\n"

    mode = summary.get("mode", "live")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"telegram_{mode}_{actual}_delay.txt"
    out_path.write_text(text, encoding="utf-8")

    if force_no_send or mode == "paper":
        return out_path

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        print(f"[telegram] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID가 없어 발송하지 않고 파일로만 저장합니다. ({env_status()})")
        return out_path

    conn = db.connect(db.db_path_for_mode(mode))
    try:
        dedup_key = f"{actual}:delay"
        if db.has_notified(conn, dedup_key):
            print(f"[telegram] {actual} 지연 알림을 이미 보낸 기록이 있어 다시 보내지 않습니다.")
            return out_path

        report_path = summary.get("report_path")
        if report_path and Path(report_path).exists():
            ok = _send_document(
                token, chat_id, Path(report_path), report_attachment_name(str(actual), cfg), caption=text.strip()
            )
        else:
            ok = _send_text(token, chat_id, text)
        if ok:
            db.record_notified(conn, dedup_key, datetime.now().isoformat(timespec="seconds"))
        else:
            print("[telegram] 지연 알림 발송에 실패했습니다 (다음 실행에서 재시도 가능).")
    finally:
        conn.close()
    return out_path


def send_group_briefing(
    text: str, report_path: Path | None, summary: dict, cfg: dict, force_no_send: bool = False
) -> dict:
    """단체방(투자클럽, TELEGRAM_GROUP_CHAT_ID)에 공개용 브리핑을 보낸다.

    입력: text(notify.briefing.build_public_briefing_text 결과), report_path(공개용
         HTML — engine.daily가 render_public_report로 만든 경로, 없으면 첨부 생략),
         summary, cfg, force_no_send(--no-send 플래그)
    출력: {"attempted": bool, "ok": bool, "skipped_reason": str|None, "error": str|None,
          "migrate_to_chat_id": int|None, "out_path": Path}

    개인 발송과 같이 공개용 보고서 파일 한 통만 보내고 본문 글은 보내지 않는다
    (report_caption(text) = 제목 줄만 첨부 설명으로, 2026-10-01). 보고서 파일이 없을
    때만 본문 글을 대신 보낸다.

    TELEGRAM_GROUP_CHAT_ID가 없으면(Secret 미설정) 조용히 건너뛴다(skipped_reason =
    "no_group_chat_id") — 이 기능을 아직 안 쓰는 사용자에게 영향이 없어야 한다.
    paper 모드는 send_briefing과 같은 이유로 절대 실제 발송을 하지 않는다(하드 가드,
    --no-send 여부와 무관) — 항상 live 전용으로 부르지만(호출부 책임) 방어적으로 한
    번 더 막는다. 실패해도 예외를 던지지 않는다 — 호출부가 개인 채팅에 경고 한 줄만
    남기고 실행을 계속해야 하기 때문이다(P-group 지시문).
    """
    mode = summary.get("mode", "live")
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"telegram_group_{mode}_{as_of_str}.txt"
    out_path.write_text(text, encoding="utf-8")

    result: dict = {
        "attempted": False,
        "ok": False,
        "skipped_reason": None,
        "error": None,
        "migrate_to_chat_id": None,
        "out_path": out_path,
    }

    if force_no_send or mode == "paper":
        result["skipped_reason"] = "no_send" if force_no_send else "paper_mode"
        return result

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    group_chat_id = os.environ.get("TELEGRAM_GROUP_CHAT_ID") or ""
    if not group_chat_id:
        result["skipped_reason"] = "no_group_chat_id"
        return result
    if not token:
        result["skipped_reason"] = "no_token"
        return result

    result["attempted"] = True
    conn = db.connect(db.db_path_for_mode(mode))
    try:
        dedup_key = f"{as_of_str}:group"
        if db.has_notified(conn, dedup_key) and not summary.get("rebuilt_from"):
            print(f"[telegram] {as_of_str} 기준 단체방에 이미 보낸 기록이 있어 다시 보내지 않습니다.")
            result["ok"] = True
            result["skipped_reason"] = "already_sent"
            return result

        ok = True
        migrate_id: int | None = None
        error: str | None = None
        if report_path and Path(report_path).exists():
            ok, migrate_id, error = _send_document_checked(
                token, group_chat_id, Path(report_path), report_attachment_name(as_of_str, cfg),
                caption=report_caption(text),
            )
        else:
            print("[telegram] 공개용 보고서 파일이 없어 본문 글을 대신 보냅니다.")
            for chunk in split_message(text):
                c_ok, c_migrate, c_err = _send_text_checked(token, group_chat_id, chunk)
                ok = ok and c_ok
                migrate_id = migrate_id or c_migrate
                error = error or c_err
                if not c_ok:
                    break  # 슈퍼그룹 전환 등으로 실패하면 뒤 청크를 더 보내도 소용없다

        result["ok"] = ok
        result["migrate_to_chat_id"] = migrate_id
        result["error"] = error
        if ok:
            db.record_notified(conn, dedup_key, datetime.now().isoformat(timespec="seconds"))
        else:
            print(f"[telegram] 단체방 발송에 실패해 발송 기록을 남기지 않습니다: {error}")
    finally:
        conn.close()
    return result


def group_send_warning_line(result: dict | None) -> str | None:
    """send_group_briefing 결과 -> 개인 채팅에 남길 경고 한 줄 (성공·건너뜀이면 None).

    슈퍼그룹 전환 오류(migrate_to_chat_id)면 새 chat_id를 함께 알려줘 사용자가
    TELEGRAM_GROUP_CHAT_ID Secret을 바꿀 수 있게 한다.
    """
    if not result or result.get("ok") or not result.get("attempted"):
        return None
    if result.get("migrate_to_chat_id"):
        return (
            f"⚠️ 단체방 발송 실패: 그룹이 슈퍼그룹으로 전환됐습니다. "
            f"TELEGRAM_GROUP_CHAT_ID를 {result['migrate_to_chat_id']}로 바꿔주세요."
        )
    return f"⚠️ 단체방 발송 실패: {result.get('error') or '알 수 없는 오류'}"


def resend_last(
    report_path: Path, text_path: Path, as_of_str: str, force_no_send: bool = False, cfg: dict | None = None
) -> bool:
    """--resend(P3.4 3번): 상태를 다시 처리하지 않고, 이미 만들어 둔 보고서·글
    파일을 다시 보낸다. 중복 발송 방지 기록(store.db)은 확인하지도, 남기지도 않는다.

    입력: report_path(outputs/report_live_YYYY-MM-DD.html — engine.daily.main은 항상 이 실전
         파일만 넘긴다), text_path(outputs/telegram_live_YYYY-MM-DD.txt),
         as_of_str(첨부 파일 이름에 쓸 기준일), force_no_send(--no-send 플래그),
         cfg(첨부 파일 이름의 report.short_title)
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

    # send_briefing과 같이 보고서 파일 한 통만 (본문 글은 보고서가 없을 때만)
    if report_path.exists():
        return _send_document(
            token, chat_id, report_path, report_attachment_name(as_of_str, cfg), caption=report_caption(text)
        )
    print(f"[telegram] {report_path}가 없어 보고서 없이 글만 보냅니다.")
    return all(_send_text(token, chat_id, chunk) for chunk in split_message(text))
