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


# 텔레그램 "시장 온도" 한 줄 (P3.8 4번) — 표시 전용, 매매 규칙에는 안 쓴다.
_MACRO_SHORT_NAME = {
    "공포·탐욕 지수": "공포·탐욕", "미국 10년물 국채금리": "10년물", "장단기 금리차 (10년−2년)": "금리차",
    "하이일드 스프레드": "HY", "미국 기준금리 (상단)": "기준금리", "원/달러 환율": "환율",
}


def _macro_item_text(row: dict) -> str:
    name = _MACRO_SHORT_NAME.get(row["name"], row["name"])
    value = row["value"]
    label = row["badge"]["label"]
    if row["name"] == "공포·탐욕 지수":
        text = f"{name} {value:g} {label}"
    elif row["name"] == "장단기 금리차 (10년−2년)":
        text = f"{name} {value:+g}"
    elif row["name"] == "미국 기준금리 (상단)":
        text = f"{name} {value:g}% {label.replace(' 흐름', '')}"
    elif row["name"] == "원/달러 환율":
        text = f"{name} {value:,.0f} {label}"
    else:
        text = f"{name} {value:g}{row.get('unit', '')}"
    return ("⚠" + text) if row["badge"]["status"] in ("warn", "bad") else text


def build_macro_line(macro_rows: list[dict]) -> str | None:
    """summary["macro_rows"] -> "시장 온도: ..." 한 줄 (없으면 None)."""
    if not macro_rows:
        return None
    return "시장 온도: " + " · ".join(_macro_item_text(r) for r in macro_rows)


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


def build_input_error_line(errors: list[str], max_len: int = 80) -> str | None:
    """계획·체결 시트 읽기 오류 목록 -> 텔레그램 한 줄 경고 (없으면 None).

    출력 예: "⚠️ 시트 입력 오류 2건 · 계획 기록 오류: 1번째 줄 - ... (보고서 경고 확인)"
    """
    if not errors:
        return None
    first = errors[0]
    if len(first) > max_len:
        first = first[: max_len - 1] + "…"
    return f"⚠️ 시트 입력 오류 {len(errors)}건 · {first} (보고서 경고 확인)"


def build_briefing_text(summary: dict, cfg: dict) -> str:
    """summary + cfg -> 텔레그램에 보낼 본문 문자열 (최소 형식, P3.4 2번).

    "📈 시장 온도"·"🎯 오늘의 신호" 두 묶음으로 나누고, 각 항목을 "- "로 시작하는
    한 줄씩 쓴다. 경고 이모지(⚠️)는 빼되 내용은 그대로 남긴다(사용자 확정).
    HTML 보고서 첨부는 이 함수와 무관하게 notify.telegram.send_briefing이 그대로 한다.
    """
    as_of = summary.get("as_of")
    mode_label = summary.get("mode_label", "실전")
    if as_of is not None:
        d = as_of.date()
        date_str = f"{d.month}/{d.day}({_KOREAN_WEEKDAY[as_of.dayofweek]})"
    else:
        date_str = "알수없음"

    lines = [f"📊 나스닥100 · {date_str} 마감 · {mode_label}"]
    if summary.get("rebuilt_from"):
        lines.append(f"🔁 체결 기록 변경 반영 정정본 ({summary['rebuilt_from']}부터 재계산)")

    macro_rows = summary.get("macro_rows", [])
    if macro_rows:
        lines.append("")
        lines.append("📈 시장 온도")
        lines.extend(f"- {_macro_item_text(r)}" for r in macro_rows)

    signal_lines: list[str] = []

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
        signal_lines.append("오늘 매매 신호 없음")
    else:
        buy_line = f"🟢 매수 {buy_count}"
        if buy_items:
            buy_line += " · " + " ".join(buy_items)
        signal_lines.append(buy_line)

        sell_line = f"🔴 매도 {sell_count}"
        if sell_items:
            sell_line += " · " + " ".join(sell_items)
        signal_lines.append(sell_line)

    unfilled_rows = summary.get("unfilled_rows", [])
    if unfilled_rows:
        names = " ".join(r["티커"] for r in unfilled_rows)
        signal_lines.append(f"미체결 {len(unfilled_rows)} · {names}")

    # 라이브 어드바이저 1단계 (docs/design/live_advisor.md 3·7번): 판정이 바뀐 종목을
    # 맨 위 근처에 강조하고, 계획금액으로 1주도 못 사는 종목은 항상 경고한다.
    live_judgment_rows = summary.get("live_judgment_rows", [])
    changed_rows = [r for r in live_judgment_rows if r.get("changed")]
    if changed_rows:
        items = " ".join(f"{r['ticker']}({r['judgment']})" for r in changed_rows)
        signal_lines.append(f"🔄 판정 변경 {len(changed_rows)} · {items}")
    warning_rows = [r for r in live_judgment_rows if r.get("one_share_warning")]
    if warning_rows:
        items = " ".join(r["ticker"] for r in warning_rows)
        signal_lines.append(f"계획금액으로 1차 매수 0주 · {items}")

    stop_alerts = summary.get("stop_alerts", [])
    if stop_alerts:
        items = " ".join(f"{a['티커']}({a['type']})" for a in stop_alerts)
        signal_lines.append(f"🛡️ 손절 예약 · {items}")

    lines.append("")
    lines.append("🎯 오늘의 신호")
    lines.extend(f"- {s}" for s in signal_lines)

    input_line = build_input_error_line(summary.get("input_errors", []))
    if input_line:
        lines.append("")
        lines.append(input_line)

    return "\n".join(lines) + "\n"
