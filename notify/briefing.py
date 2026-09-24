"""텔레그램 짧은 브리핑 글 (P3.4).

첨부하는 HTML 보고서가 본체이고, 텔레그램 글은 최소로 줄인다. 종목은 티커만
쓴다. 이 모듈은 summary dict(engine/daily.py) + cfg만 받아 문자열을 만든다
(순수 함수, 네트워크·파일 없음 — 테스트가 summary를 직접 만들어 호출할 수 있다).
"""

from __future__ import annotations

_STAGE_ORDER = ["b1", "b2", "b3", "b9"]
_BUY_STAGE_SHORT = {"b1": "1차", "b2": "2차", "b3": "3차", "b9": "재진입"}
_KOREAN_WEEKDAY = ["월", "화", "수", "목", "금", "토", "일"]

# 텔레그램 매도 줄 괄호 표기 (P3.5): 손절은 예약 체결 확인까지 덧붙인다 (P3.5 5번).
_SELL_KIND_SHORT = {
    "STOP": "손절·예약 체결 확인",
    "E1": "1차분(E1)",
    "E2": "2차분(E2)",
    "E3": "3차분(E3)",
    "A1_EXPIRE": "전량",
}


def _sell_short_label(row: dict) -> str:
    """매도 범위를 티커 옆 괄호에 쓸 짧은 표기로 줄인다 (예: "ROP(1차분(E1))", "ROP(손절·예약 체결 확인)")."""
    kind = row.get("kind")
    if kind in _SELL_KIND_SHORT:
        return _SELL_KIND_SHORT[kind]
    # kind가 없는 옛 호출부(테스트 등) 호환: 신호·매도범위 텍스트로 추정한다.
    if row.get("신호") == "손절":
        return _SELL_KIND_SHORT["STOP"]
    범위 = row.get("매도범위", "")
    if "1차분" in 범위:
        return "1차분(E1)"
    if "2차분" in 범위:
        return "2차분(E2)"
    return "전량"


def build_briefing_text(summary: dict, cfg: dict) -> str:
    """summary + cfg -> 텔레그램에 보낼 본문 문자열 (최소 형식, P3.4 2번)."""
    as_of = summary.get("as_of")
    mode_label = summary.get("mode_label", "실전")
    if as_of is not None:
        d = as_of.date()
        date_str = f"{d.month}/{d.day}({_KOREAN_WEEKDAY[as_of.dayofweek]})"
    else:
        date_str = "알수없음"

    lines = [f"📊 나스닥100 · {date_str} 마감 · {mode_label}"]

    buy_groups = summary.get("buy_groups", {})
    buy_items = [
        f"{r['ticker']}({_BUY_STAGE_SHORT[key]})"
        for key in _STAGE_ORDER
        for r in buy_groups.get(key, [])
    ]
    buy_count = summary.get("buy_count", 0)

    sell_rows = summary.get("sell_rows", [])
    sell_items = [f"{r['티커']}({_sell_short_label(r)})" for r in sell_rows]
    sell_count = len(sell_rows)

    if buy_count == 0 and sell_count == 0:
        lines.append("오늘 매매 신호 없음")
    else:
        buy_line = f"🟢 매수 {buy_count}"
        if buy_items:
            buy_line += " · " + " ".join(buy_items)
        lines.append(buy_line)

        sell_line = f"🔴 매도 {sell_count}"
        if sell_items:
            sell_line += " · " + " ".join(sell_items)
        lines.append(sell_line)

    unfilled_rows = summary.get("unfilled_rows", [])
    if unfilled_rows:
        names = " ".join(r["티커"] for r in unfilled_rows)
        lines.append(f"⚠️ 미체결 {len(unfilled_rows)} · {names}")

    stop_alerts = summary.get("stop_alerts", [])
    if stop_alerts:
        items = " ".join(f"{a['티커']}({a['type']})" for a in stop_alerts)
        lines.append(f"🛡️ 손절 예약 · {items}")

    lines.append("📎 보고서를 열어 확인하세요")
    return "\n".join(lines) + "\n"
