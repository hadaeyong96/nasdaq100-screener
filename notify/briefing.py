"""텔레그램 짧은 브리핑 글 (P3.4).

첨부하는 HTML 보고서가 본체이고, 텔레그램 글은 최소로 줄인다. 종목은 티커만
쓴다. 이 모듈은 summary dict(engine/daily.py) + cfg만 받아 문자열을 만든다
(순수 함수, 네트워크·파일 없음 — 테스트가 summary를 직접 만들어 호출할 수 있다).
"""

from __future__ import annotations

# 단체방 공개용 고지 문구 (텔레그램 본문·공개 HTML 맨 아래 모두 이 상수를 쓴다).
PUBLIC_DISCLAIMER = "개인 학습용 참고 자료이며 투자 권유가 아닙니다. 투자 판단과 책임은 본인에게 있습니다."

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


# 텔레그램 "시장 온도" 줄 (P3.8 4번) — 표시 전용, 매매 규칙에는 안 쓴다. 판정 단어는
# 보고서와 같은 core.macro_status.classify_macro 결과(🟢안정·🟡주의·🔴위험)를 그대로 쓴다.
def _macro_value_text(row: dict) -> str:
    """지표별 짧은 이름 + 값. 예: "10년물 5.17%", "금리차 +0.52%p", "환율 1,392원"."""
    slug, v = row.get("slug"), row["value"]
    zone = (row.get("badge") or {}).get("zone")
    if slug == "fear-greed":
        return f"공포·탐욕 {v:g}" + (f"({zone})" if zone else "")
    if slug == "vix":
        return f"VIX {v:.1f}"
    if slug == "dgs10":
        return f"10년물 {v:.2f}%"
    if slug == "t10y2y":
        return f"금리차 {v:+.2f}%p"
    if slug == "hy":
        return f"HY 스프레드 {v:.2f}%"
    if slug == "fed":
        return f"기준금리 {v:g}%" + (f"({zone})" if zone else "")
    if slug == "fx":
        return f"환율 {v:,.0f}원"
    return f"{row['name']} {v:g}{row.get('unit', '')}"


def _macro_item_text(row: dict) -> str:
    """"10년물 5.17% 🔴위험" 형식 한 항목."""
    badge = row.get("badge") or {}
    return f"{_macro_value_text(row)} {badge.get('symbol', '')}{badge.get('label', '')}".rstrip()


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

    출력 예: "⚠️ 시트 오류: 계획 기록 오류: 1번째 줄 - ... 외 1건"
    """
    if not errors:
        return None
    first = errors[0]
    if len(first) > max_len:
        first = first[: max_len - 1] + "…"
    more = f" 외 {len(errors) - 1}건" if len(errors) > 1 else ""
    return f"⚠️ 시트 오류: {first}{more}"


def build_fred_missing_line(missing_codes: list[str]) -> str | None:
    """FRED 지표가 하나라도 빠졌으면 텔레그램 한 줄 경고 (없으면 None).

    출력 예: "⚠️ FRED 지표 수집 실패: DGS10, T10Y2Y"
    """
    if not missing_codes:
        return None
    return f"⚠️ FRED 지표 수집 실패: {', '.join(missing_codes)}"


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

    alert_lines = [
        line
        for line in (
            build_input_error_line(summary.get("input_errors", [])),
            build_fred_missing_line(summary.get("fred_missing", [])),
        )
        if line
    ]
    if alert_lines:
        lines.append("")
        lines.extend(alert_lines)

    return "\n".join(lines) + "\n"


# ── 단체방 공개용 (투자클럽) ──────────────────────────────────────────
# 개인 채팅과 달리 시장 온도 + 오늘의 추천(1차 진입가·손절가·조건)만 보낸다.
# 보유 종목·수량·평단·손익·계획금액·체결 내역은 절대 넣지 않는다.
#
# 오늘의 추천은 신규 진입(b1, A1 정찰) 신호만 보여준다 — b2(2차 확인)·b3(3차
# 확정)·b9(재진입)는 라이브 모드에서 실제 보유 중이거나 과거에 보유했던 종목에만
# 나오는 신호라(core/state.py: A2·A3는 대기 상태에서 A1 이후에만 판정되고, B는
# 이전에 보유했다가 나간 종목의 재진입이다) 종목명만으로도 내 보유가 드러난다.
_PUBLIC_RECOMMEND_BUCKET = "b1"


def _public_recommend_line(r: dict) -> str:
    """buy_groups["b1"] 행 하나 -> "AAPL(1차 정찰) 진입가 $190.20 · 손절가 $182.10 · RSI 28 → 32".

    limit·stop은 지표로만 정해지는 가격(개인 계획금액과 무관)이라 공개해도 된다.
    조건 설명은 r["condition_summary"](engine.daily._build_buy_row가 자금 계획
    문구가 섞이기 전에 따로 남겨 둔 필드)만 쓴다 — r["note"]는 "계획 없음"·
    "남은 한도 부족" 같은 개인 자금 문구가 섞여 있어 쓰지 않는다.
    """
    stop = r.get("stop")
    stop_str = f"${stop:,.2f}" if stop is not None else "미확인"
    head = f"{r['ticker']}({r.get('stage_label', '')}) 진입가 ${r['limit']:,.2f}"
    parts = [head, f"손절가 {stop_str}"]
    condition = r.get("condition_summary")
    if condition:
        parts.append(condition)
    return " · ".join(parts)


def build_public_briefing_text(summary: dict, cfg: dict) -> str:
    """summary + cfg -> 단체방(투자클럽)에 보낼 공개용 본문.

    "📈 시장 온도"·"🎯 오늘의 추천" 두 묶음만 담는다. 오늘의 추천은 오늘 발생한
    신규 진입(b1, A1 정찰) 신호만 점수순으로 보여준다 — 2차·3차·재진입은 보유가
    드러나므로 뺀다(모듈 위 주석 참고). 종목·진입가·손절가·조건뿐이고 수량·
    투입금액·최대손실은 넣지 않는다.
    """
    as_of = summary.get("as_of")
    if as_of is not None:
        d = as_of.date()
        date_str = f"{d.month}/{d.day}({_KOREAN_WEEKDAY[as_of.dayofweek]})"
    else:
        date_str = "알수없음"

    lines = [f"📊 나스닥100 · {date_str} 마감 · 공개용"]

    macro_rows = summary.get("macro_rows", [])
    if macro_rows:
        lines.append("")
        lines.append("📈 시장 온도")
        lines.extend(f"- {_macro_item_text(r)}" for r in macro_rows)

    buy_groups = summary.get("buy_groups", {})
    new_entry_rows = sorted(
        buy_groups.get(_PUBLIC_RECOMMEND_BUCKET, []), key=lambda r: r.get("score", 0), reverse=True
    )

    lines.append("")
    lines.append("🎯 오늘의 추천")
    if not new_entry_rows:
        lines.append("- 오늘 추천 종목 없음")
    else:
        lines.extend(f"- {_public_recommend_line(r)}" for r in new_entry_rows)

    lines.append("")
    lines.append(PUBLIC_DISCLAIMER)

    return "\n".join(lines) + "\n"
