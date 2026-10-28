"""시장 온도계 독립 실행: python -m thermometer [--dry-run] [--date YYYY-MM-DD]

순서: 일·월(한국 날짜)이면 건너뜀 → 기준일(가장 최근 마감된 미국 거래일) → 기록 표에 그 기준일이 있으면 건너뜀(중복 발송 막기)
→ 온도 계산 → QQQ 자료가 기준일까지 안 왔으면 다음 예약으로 미룸(마지막 예약이면 실패) → 개인방·단체방 발송 → 기록 한 줄
(가상 장부 DB thermometer_history 표 — 드라이브 동기화는 workflow가, 아침 편지와 같은 concurrency 그룹이라 서로 덮어쓰지 않는다).
실제 발송은 GitHub Actions(또는 SCREENER_REAL_SEND=1)에서만 — 로컬·--dry-run은 글만 보여 주고 기록도 남기지 않는다.
실패하면 0이 아닌 종료 코드 → workflow가 개인방에 "시장 온도계 실패" 한 줄을 보낸다(.github/workflows/thermometer.yml).
아침 편지와 따로 돈다(실패해도 편지는 그대로).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, datetime
from zoneinfo import ZoneInfo

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from thermometer import history  # noqa: E402
from thermometer.core import get_temperature, load_config  # noqa: E402
from thermometer.message import build_message  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
US_EASTERN = ZoneInfo("America/New_York")


def skip_weekday(now_kst: datetime, cfg: dict) -> bool:
    """한국 날짜가 일·월(config schedule.skip_weekdays_kst)이면 True — 새 미국 종가가 없는 날."""
    return now_kst.weekday() in set(cfg["schedule"]["skip_weekdays_kst"])


def base_date_for(now_kst: datetime) -> date:
    """실행 시각 → 가장 최근에 마감된 미국 거래일(NYSE 달력)."""
    from data.market_calendar import latest_closed_trading_day

    return latest_closed_trading_day(now_kst.astimezone(US_EASTERN))


def is_final_slot(cfg: dict, schedule_cron: str | None) -> bool:
    """마지막 예약이거나 수동 실행(SCHEDULE_CRON 빈 값)이면 True."""
    return not schedule_cron or schedule_cron == cfg["schedule"]["final_cron"]


def _default_send(text: str) -> None:
    """개인방(필수 — 실패하면 예외)·단체방(실패해도 기록만) 발송. 토큰 값은 출력하지 않는다."""
    from notify import telegram

    token = os.environ.get("TELEGRAM_BOT_TOKEN") or ""
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or ""
    if not token or not chat_id:
        raise RuntimeError(f"텔레그램 토큰·개인방이 없습니다 ({telegram.env_status()})")
    if not telegram._send_text(token, chat_id, text):
        raise RuntimeError("개인방 발송 실패")
    group = os.environ.get("TELEGRAM_GROUP_CHAT_ID") or ""
    if group:
        ok, migrate_id, err = telegram._send_text_checked(token, group, text)
        if not ok:
            print(f"[thermometer] 단체방 발송 실패(개인방은 보냄): {telegram.redact(err or '')}"
                  + (" — 단체방 chat_id가 슈퍼그룹으로 바뀌었습니다" if migrate_id else ""))
    else:
        print("[thermometer] TELEGRAM_GROUP_CHAT_ID가 없어 단체방은 건너뜁니다")


def run(dry_run: bool = False, now: datetime | None = None, prices: dict | None = None, send=None,
        history_path=None, cfg: dict | None = None, schedule_cron: str | None = None, real_send: bool | None = None,
        on_date: str | None = None) -> dict:
    """한 번 실행 → {"status", "base_date", "result"?, "text"?}. 실패할 상황이면 예외.

    status: skip_weekday | already_sent | not_ready | no_storage | dry_run | sent
    """
    from notify.telegram import real_send_allowed

    cfg = cfg or load_config()
    now_kst = (now or datetime.now(KST)).astimezone(KST)
    hpath = history_path or cfg["history"]["path"]
    real = real_send_allowed() if real_send is None else real_send
    real = real and not dry_run
    final = is_final_slot(cfg, schedule_cron)

    if on_date is None and skip_weekday(now_kst, cfg):
        print(f"[thermometer] 한국 {now_kst:%m/%d} {'일월'[0 if now_kst.weekday() == 6 else 1]}요일 — 새 미국 종가가 없어 보내지 않습니다")
        return {"status": "skip_weekday"}
    base = date.fromisoformat(on_date) if on_date else base_date_for(now_kst)
    out = {"base_date": base.isoformat()}
    if real and history.find(base.isoformat(), hpath) is not None:
        print(f"[thermometer] 기준일 {base} 온도계는 이미 보냈습니다 — 건너뜁니다")
        return {**out, "status": "already_sent"}
    if real and not (os.environ.get(cfg["history"]["drive_env"]) or "").strip():
        msg = f"{cfg['history']['drive_env']}가 없어 가상 장부 DB를 받을 수 없습니다(중복 발송 막기 불가) — 보내지 않습니다"
        if final:
            raise RuntimeError(msg)
        print(f"[thermometer] {msg}")
        return {**out, "status": "no_storage"}

    result = get_temperature(base, prices=prices, cfg=cfg)
    for e in (prices or {}).get("errors", []) if prices is not None else []:
        print(f"[thermometer] 경고: {e}")
    out["result"] = result
    if result["data_date"] != base.isoformat() or result["temp"] is None:
        msg = f"기준일 {base} QQQ 자료가 아직 없습니다(마지막 {result['data_date']})"
        if final:
            raise RuntimeError(msg)
        print(f"[thermometer] {msg} — 다음 예약에서 다시 확인합니다")
        return {**out, "status": "not_ready"}

    text = build_message(result, cfg, now_kst.date())
    out["text"] = text
    if not real:
        print("[thermometer] dry-run(전송 없음, 기록 없음) — 견본:\n" + text)
        return {**out, "status": "dry_run"}
    (send or _default_send)(text)
    history.append(result, hpath)
    print(f"[thermometer] 기준일 {base} {result['temp']}도 발송·기록 완료")
    return {**out, "status": "sent"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="시장 온도계 (0~100도)")
    parser.add_argument("--dry-run", action="store_true", help="보내지 않고 견본만 출력(기록도 안 남김)")
    parser.add_argument("--date", help="이 기준일로 계산(견본 확인용, 요일 건너뛰기 없음)")
    args = parser.parse_args(argv)
    try:
        r = run(dry_run=args.dry_run, schedule_cron=os.environ.get("SCHEDULE_CRON") or None, on_date=args.date)
    except Exception as exc:
        from notify.telegram import redact

        print(f"[thermometer] 시장 온도계 실패: {redact(str(exc))}")
        return 1
    return 0 if r["status"] != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
