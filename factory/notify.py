"""전략공장 알림 — "전략분석기" 텔레그램(예전 이름 데이터탐색기)에만 보낸다 (작업 F2-마무리, 2026-10-16).

받는 곳: 환경변수 TELEGRAM_FACTORY_CHAT_ID (봇 토큰은 TELEGRAM_FACTORY_BOT_TOKEN, 없으면 TELEGRAM_BOT_TOKEN).
- 값이 없으면 알림을 건너뛰고 로그 한 줄만 남긴다(작업은 실패시키지 않는다).
- 투자클럽 단체방(TELEGRAM_GROUP_CHAT_ID)과 같은 chat id면 절대 보내지 않는다(코드에서 고정).
- 값(토큰·chat id)은 어떤 경우에도 출력하지 않는다. 오류도 종류만 남긴다(요청 URL에 토큰이 들어 있어서).

알림 종류(event): explore_done(탐색 시험 완료) · zone_judged(검증 구역 판정 완료) · scorecard_done(월간 성적표 생성)
· error(공장 작업이 오류로 멈춤 — 어느 단계인지만, 오류 상세 금지).
형식: 첫 줄 제목, 결과 2~3줄, 다음 대기 후보(catalog.yaml 순위), 마지막 줄 "→ Claude에 결과 보고서를 붙여 넣으세요".
이모지 없음, 6줄 이내. 발송이 실패해도 공장 작업 결과(파일·커밋)는 그대로 둔다(send_safely는 예외를 내지 않음).

손으로 보내기:
    python -m factory.notify --event explore_done --summary-file <결과 JSON 또는 YAML>
    python -m factory.notify --event explore_done --summary-file x.json --dry-run   # 보내지 않고 글만 출력
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "factory" / "catalog.yaml"
CHAT_ENV, TOKEN_ENV, TOKEN_FALLBACK_ENV = "TELEGRAM_FACTORY_CHAT_ID", "TELEGRAM_FACTORY_BOT_TOKEN", "TELEGRAM_BOT_TOKEN"
GROUP_ENV = "TELEGRAM_GROUP_CHAT_ID"  # 투자클럽 단체방 — 공장 알림은 여기로 절대 보내지 않는다
LAST_LINE = "→ Claude에 결과 보고서를 붙여 넣으세요"
MAX_LINES = 6
EVENTS = ("explore_done", "zone_judged", "scorecard_done", "error")
TITLES = {"explore_done": "[전략공장] 탐색 시험 완료", "zone_judged": "[전략공장] 검증 구역 판정 완료",
          "scorecard_done": "[전략공장] 월간 성적표 생성", "error": "[전략공장] 작업 멈춤"}


class FactoryNotifyForbidden(RuntimeError):
    """단체방으로 보내려 할 때."""


def next_candidate(exclude_id: str | None = None, catalog_path: Path = CATALOG) -> str:
    """catalog.yaml 대기 후보 중 순위가 가장 높은 이름 (exclude_id는 뺀다). 없으면 "없음"."""
    import yaml

    items = yaml.safe_load(catalog_path.read_text(encoding="utf-8")).get("strategies", [])
    cands = sorted((x for x in items if x.get("status") == "대기 후보" and x.get("id") != exclude_id),
                   key=lambda x: x.get("rank", 99))
    return cands[0]["name"] if cands else "없음"


def _num(x, fmt="{:.2f}") -> str:
    try:
        return fmt.format(float(x))
    except (TypeError, ValueError):
        return "-"


def build_message(event: str, s: dict, catalog_path: Path = CATALOG) -> str:
    """알림 글 (6줄 이내, 이모지 없음). s: 결과 요약 dict."""
    if event not in EVENTS:
        raise ValueError(f"모르는 알림 종류: {event}")
    lines = [TITLES[event]]
    if event == "explore_done":
        lines.append(f"후보: {s.get('name', '-')} — {s.get('verdict', '-')}")
        lines.append(f"세후 연복리 {_num(s.get('cagr'))}% vs QQQ {_num(s.get('qqq_cagr'))}% · "
                     f"무작위 {_num(s.get('random_pct'), '{:.1f}')}백분위")
    elif event == "zone_judged":
        lines.append(f"후보: {s.get('name', '-')} — {s.get('verdict', '-')}")
        lines.append(f"{s.get('zone', '구역 -')} 남은 예산 {s.get('budget_left', '-')}")
    elif event == "scorecard_done":
        lines.append(f"{s.get('month', '-')} 성적표를 원본 시트 「월간성적표」에 기록")
        stop, review = s.get("stop") or [], s.get("review") or []
        if stop or review:
            if stop:
                lines.append("중단 제안: " + ", ".join(stop))
            if review:
                lines.append("12개월 검토 제안: " + ", ".join(review))
        else:
            lines.append("제안 없음")
    else:
        lines.append(f"단계: {s.get('stage', '-')}")
    lines.append(f"다음 대기 후보: {next_candidate(s.get('candidate_id'), catalog_path)}")
    lines.append(LAST_LINE)
    if len(lines) > MAX_LINES:  # 결과 줄이 넘치면 결과 쪽을 줄인다(제목·다음 후보·마지막 줄은 남김)
        lines = lines[:MAX_LINES - 2] + lines[-2:]
    return "\n".join(lines)


def _target() -> tuple[str, str] | None:
    """(token, chat_id) 또는 None(값 없음). 단체방과 같은 chat id면 FactoryNotifyForbidden."""
    chat = (os.environ.get(CHAT_ENV) or "").strip()
    token = (os.environ.get(TOKEN_ENV) or os.environ.get(TOKEN_FALLBACK_ENV) or "").strip()
    if not chat or not token:
        return None
    group = (os.environ.get(GROUP_ENV) or "").strip()
    if group and chat == group:
        raise FactoryNotifyForbidden(f"{CHAT_ENV}가 투자클럽 단체방과 같아 보내지 않습니다")
    return token, chat


def _post(token: str, chat_id: str, text: str) -> None:
    import requests

    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=20)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code}")


def send(event: str, s: dict, *, post=None, catalog_path: Path = CATALOG) -> bool:
    """알림 한 통. 출력: 보냈으면 True, 받는 곳이 없어 건너뛰었으면 False. 단체방이면 FactoryNotifyForbidden."""
    text = build_message(event, s, catalog_path)
    target = _target()
    if target is None:
        print(f"[factory.notify] {CHAT_ENV}(또는 봇 토큰)가 없어 전략분석기 알림을 건너뜁니다 ({event})")
        return False
    (post or _post)(target[0], target[1], text)
    print(f"[factory.notify] 전략분석기에 알림 보냄 ({event})")
    return True


def send_safely(event: str, s: dict, **kw) -> bool:
    """send와 같지만 어떤 오류도 밖으로 내지 않는다(공장 작업 결과 보호). 오류는 종류만 로그."""
    try:
        return send(event, s, **kw)
    except Exception as exc:  # noqa: BLE001
        print(f"[factory.notify] 전략분석기 알림 실패({event}): {type(exc).__name__}")
        return False


def _load_summary(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text) or {}
    return json.loads(text)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="전략공장 알림 (전략분석기 텔레그램 전용)")
    p.add_argument("--event", choices=EVENTS, required=True)
    p.add_argument("--summary-file", required=True, help="결과 요약 JSON/YAML (name, verdict, cagr, qqq_cagr, random_pct 등)")
    p.add_argument("--dry-run", action="store_true", help="보내지 않고 글만 출력")
    a = p.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    s = _load_summary(Path(a.summary_file))
    if a.dry_run:
        print(build_message(a.event, s))
        return 0
    send_safely(a.event, s)
    return 0  # 알림 실패로 공장 작업을 실패시키지 않는다


if __name__ == "__main__":
    raise SystemExit(main())
