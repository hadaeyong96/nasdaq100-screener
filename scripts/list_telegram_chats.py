"""봇이 아는 텔레그램 대화방 목록 — 읽기만 (2026-10-16, 전략분석기 chat id 확인용).

    python scripts/list_telegram_chats.py

.env의 TELEGRAM_BOT_TOKEN으로 getUpdates를 한 번 부르고 "제목(또는 이름) · 종류 · chat id" 표만 출력한다.
- 토큰과 기존 Secret 값(TELEGRAM_CHAT_ID·TELEGRAM_GROUP_CHAT_ID)은 출력하지 않는다. 같은 번호면 표에 표시만 한다.
- 오류는 종류·HTTP 상태 코드만 남긴다(요청 URL에 토큰이 있어서).
- 텔레그램은 아직 가져가지 않은 최근 24시간 업데이트만 돌려준다. 이 스크립트는 offset을 넘기지 않아
  업데이트를 "읽음" 처리하지 않는다(다시 실행해도 같은 목록).
- 목록에 원하는 방이 없으면: 그 방에 메시지를 보낸다. 그룹이면 봇이 그 방의 멤버여야 하고, 봇 기본 설정(privacy mode)
  에서는 "/start@봇이름"처럼 봇을 부른 메시지만 보인다. 그 방이 다른 봇과의 대화라면 이 봇에는 절대 보이지 않는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
KIND = {"private": "개인", "group": "그룹", "supergroup": "그룹(슈퍼그룹)", "channel": "채널"}
UPDATE_KEYS = ("message", "edited_message", "channel_post", "edited_channel_post", "my_chat_member", "chat_member")


def collect_chats(updates: list[dict]) -> dict[int, dict]:
    """getUpdates 결과 → {chat id: chat}. 순수 함수."""
    chats: dict[int, dict] = {}
    for u in updates:
        for key in UPDATE_KEYS:
            c = (u.get(key) or {}).get("chat")
            if c:
                chats[c["id"]] = c
    return chats


def table(chats: dict[int, dict], personal: str = "", group: str = "") -> list[str]:
    """대화방 표 줄들 (토큰·Secret 값 없음, chat id는 표 칸에만)."""
    out = ["| 제목(또는 이름) | 종류 | chat id | 표시 |", "| --- | --- | --- | --- |"]
    for cid, c in chats.items():
        name = (c.get("title") or " ".join(x for x in (c.get("first_name"), c.get("last_name")) if x)
                or c.get("username") or "-")
        mark = "(기존 개인)" if personal and str(cid) == personal else ("(투자클럽)" if group and str(cid) == group else "")
        out.append(f"| {name} | {KIND.get(c.get('type'), c.get('type'))} | {cid} | {mark} |")
    return out


def run(token: str, personal: str = "", group: str = "", get=requests.get) -> list[str]:
    """getUpdates 한 번 → 출력할 줄들. 실패하면 오류 종류·상태 코드 한 줄."""
    if not token:
        return ["TELEGRAM_BOT_TOKEN 없음 (.env)"]
    try:
        r = get(f"https://api.telegram.org/bot{token}/getUpdates", params={"limit": 100}, timeout=20)
    except requests.RequestException as exc:
        return [f"getUpdates 연결 실패({type(exc).__name__})"]
    if not r.ok:
        return [f"getUpdates 실패(HTTP {r.status_code})"]
    updates = r.json().get("result", [])
    chats = collect_chats(updates)
    lines = [f"업데이트 {len(updates)}건, 대화방 {len(chats)}개"] + table(chats, personal, group)
    if not chats:
        lines.append("목록이 비어 있습니다 — 원하는 방에 메시지를 하나 보낸 뒤 다시 실행하세요.")
    return lines


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    from dotenv import dotenv_values

    env = dotenv_values(ROOT / ".env")
    lines = run((env.get("TELEGRAM_BOT_TOKEN") or "").strip(), (env.get("TELEGRAM_CHAT_ID") or "").strip(),
                (env.get("TELEGRAM_GROUP_CHAT_ID") or "").strip())
    print("\n".join(lines))


if __name__ == "__main__":
    main()
