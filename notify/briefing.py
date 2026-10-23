"""텔레그램 짧은 브리핑 글 (P3.4).

첨부하는 HTML 보고서가 본체이고, 텔레그램 글은 최소로 줄인다. 종목은 티커만
쓴다. 이 모듈은 summary dict(engine/daily.py) + cfg만 받아 문자열을 만든다
(순수 함수, 네트워크·파일 없음 — 테스트가 summary를 직접 만들어 호출할 수 있다).
"""

from __future__ import annotations

# 단체방 공개용 고지 문구 (텔레그램 본문·공개 HTML 맨 아래 모두 이 상수를 쓴다).

# 가상 장부 국면 한 줄(notify/virtual_books.regime_line)의 머리말 — 제목 바로 아래에 두고,
# 텔레그램 첨부 설명(notify.telegram.report_caption)에도 남긴다.
REGIME_LINE_PREFIX = "오늘 국면:"

# config.yaml report 섹션이 없을 때(구버전 테스트 fixture 등)만 쓰는 기본값 — 실제 값은 config.yaml에서 읽는다.
_DEFAULT_REPORT_TITLES = {
    "title": "오빠의 아침 편지",
    "subtitle": "미국 대형주 · 미국 종가 기준 · 규칙 기반 분석",
    "short_title": "오빠의 아침 편지",
}
REBUILT_PREFIX = "정정본:"  # 해당하는 날에만 쓰는 줄 머리 (이모지 대신 글자, 2026-10-15)
CHECK_PREFIX = "확인할 것:"


def report_titles(cfg: dict | None) -> dict:
    """cfg -> {title, subtitle, short_title} (config.yaml report 섹션, 빠진 키는 기본값). 다른 report 설정(show_*)은 넣지 않는다."""
    rep = (cfg or {}).get("report") or {}
    return {k: rep.get(k, v) for k, v in _DEFAULT_REPORT_TITLES.items()}

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

    출력 예: "확인할 것: 시트 오류: 계획 기록 오류: 1번째 줄 - ... 외 1건"
    """
    if not errors:
        return None
    first = errors[0]
    if len(first) > max_len:
        first = first[: max_len - 1] + "…"
    more = f" 외 {len(errors) - 1}건" if len(errors) > 1 else ""
    return f"{CHECK_PREFIX} 시트 오류: {first}{more}"


def build_fred_missing_line(missing_codes: list[str]) -> str | None:
    """FRED 지표가 하나라도 빠졌으면 텔레그램 한 줄 경고 (없으면 None).

    출력 예: "확인할 것: FRED 지표 수집 실패: DGS10, T10Y2Y"
    """
    if not missing_codes:
        return None
    return f"{CHECK_PREFIX} FRED 지표 수집 실패: {', '.join(missing_codes)}"


def date_label(as_of) -> str:
    """기준일(미국 종가 기준) -> "10/14(수)". 없으면 "알수없음"."""
    if as_of is None:
        return "알수없음"
    d = as_of.date()
    return f"{d.month}/{d.day}({_KOREAN_WEEKDAY[as_of.dayofweek]})"


def title_line(summary: dict, cfg: dict) -> str:
    """제목 줄(보고서 맨 위·첨부 설명 첫 줄): "오빠의 아침 편지 · 10/14(수)". 이모지·"마감" 없음(2026-10-15).

    모의(paper) 보고서는 보내지 않지만 파일을 헷갈리지 않게 끝에 " · 모의"를 붙인다.
    """
    line = f"{report_titles(cfg)['title']} · {date_label(summary.get('as_of'))}"
    return line + " · 모의" if summary.get("mode") == "paper" else line


def letter_message(summary: dict, cfg: dict, edition: str = "personal") -> str | None:
    """아침 편지 v3(2026-10-24) 본문 문자 메시지(한도 4,096자 — 넘으면 주제 칸부터 줄이고 "나머지는 첨부 파일에 있습니다.").
    보고서 파일보다 먼저 보낸다. 쉬운 문장판(letter.show_books false, 개인판은 report.show_screening false)이 아니면 None."""
    from notify import letter, letter_plain

    if (cfg.get("letter") or {}).get("show_books", False):
        return None
    if edition != letter.GROUP and (cfg.get("report") or {}).get("show_screening", False):
        return None
    return letter_plain.to_text(letter_plain.build(summary, cfg, edition), limit=letter_plain.MESSAGE_LIMIT)


def build_report_caption(summary: dict, cfg: dict, edition: str = "personal") -> str:
    """보고서 첨부 설명.

    짧은 판(2026-10-17, report.show_screening이 false — 기본): 제목 + "오늘 할 일: …"(notify.letter). 개인판만
    정정본·확인할 것 줄을 덧붙인다. 단체방판(edition="group")은 제목 + 단체방 할 일만(티커·개인 항목 없음).
    show_screening이 true인 개인판은 아래 예전 형식(브리핑 다이어트)을 그대로 쓴다.

    예전 형식: 제목 + 국면 줄 + "매수 n · 매도 n" + 티커.

    예: 세 줄 — "오빠의 아침 편지 · 10/13(화)" / "오늘 국면: …" / "매수 2(NVDA, AAPL) · 매도 1(ROP)"
    추천이 없으면 "매수 0 · 매도 0". 티커는 매수는 차수 순서·중복 제거, 매도는 매도 표 순서.
    해당 있는 날에만 "정정본:"(국면 줄 아래)과 "확인할 것:"(시트 입력 오류·FRED 누락, 맨 아래) 줄을 덧붙인다
    (2026-10-14 사용자 확정) — 없는 날은 세 줄 그대로.
    """
    from notify import letter, letter_plain

    plain = not (cfg.get("letter") or {}).get("show_books", False)  # 2026-10-23 쉬운 문장판(새 출발 편지)
    if edition == letter.GROUP:
        if plain:  # 아침 편지 v3(2026-10-24): 본문은 문자 메시지로 먼저(letter_message), 첨부 설명은 머리줄만
            return letter_plain.build(summary, cfg, letter.GROUP)["title"]
        return "\n".join([title_line(summary, cfg), letter.todo_line(summary, cfg, letter.GROUP)])
    if plain and not (cfg.get("report") or {}).get("show_screening", False):
        return letter_plain.build(summary, cfg, letter.PERSONAL)["title"]
    if not (cfg.get("report") or {}).get("show_screening", False):
        lines = [title_line(summary, cfg)]
        if summary.get("letter_check"):  # 아침 편지 v2.1: 머리말 → 점검 → 할 일 (개인판만)
            lines.append(letter.check_line(summary["letter_check"]))
        lines.append(letter.todo_line(summary, cfg, letter.PERSONAL))
        if summary.get("rebuilt_from"):
            lines.append(f"{REBUILT_PREFIX} 체결 기록 변경 반영 ({summary['rebuilt_from']}부터 재계산)")
        lines.extend(line for line in (
            build_input_error_line(summary.get("input_errors", [])),
            build_fred_missing_line(summary.get("fred_missing", [])),
        ) if line)
        return "\n".join(lines)
    lines = [title_line(summary, cfg)]
    if summary.get("regime_line"):
        lines.append(summary["regime_line"])
    if summary.get("rebuilt_from"):
        lines.append(f"{REBUILT_PREFIX} 체결 기록 변경 반영 ({summary['rebuilt_from']}부터 재계산)")
    buy_groups = summary.get("buy_groups", {})
    buys = list(dict.fromkeys(r["ticker"] for key in _STAGE_ORDER for r in buy_groups.get(key, [])))
    sells = list(dict.fromkeys(r["티커"] for r in summary.get("sell_rows", [])))
    buy_n = summary.get("buy_count", sum(len(buy_groups.get(k, [])) for k in _STAGE_ORDER))
    sell_n = len(summary.get("sell_rows", []))
    lines.append(f"매수 {buy_n}" + (f"({', '.join(buys)})" if buys else "")
                 + f" · 매도 {sell_n}" + (f"({', '.join(sells)})" if sells else ""))
    lines.extend(line for line in (
        build_input_error_line(summary.get("input_errors", [])),
        build_fred_missing_line(summary.get("fred_missing", [])),
    ) if line)
    return "\n".join(lines)


def build_briefing_text(summary: dict, cfg: dict) -> str:
    """summary + cfg -> 텔레그램에 보낼 본문 문자열 (최소 형식, P3.4 2번).

    "📈 시장 온도"·"🎯 오늘의 신호" 두 묶음으로 나누고, 각 항목을 "- "로 시작하는
    한 줄씩 쓴다. 경고 이모지(⚠️)는 빼되 내용은 그대로 남긴다(사용자 확정).
    "🎯 오늘의 신호" 아래 각 줄(매수·매도·미체결·판정 변경·손절 예약)은 장식용
    이모지 없이 "제목 N : 항목1, 항목2" 형식으로 통일한다(사용자 확정) — 묶음
    제목(🎯 등)과 시장 온도 줄의 판정 이모지(🟢🟡🔴)는 그대로 둔다.
    HTML 보고서 첨부는 이 함수와 무관하게 notify.telegram.send_briefing이 그대로 한다.
    """
    lines = [title_line(summary, cfg)]
    if summary.get("regime_line"):  # 가상 장부 국면 한 줄 (live, engine.virtual_books — 실패하면 없음)
        lines.append(summary["regime_line"])
    if summary.get("rebuilt_from"):
        lines.append(f"{REBUILT_PREFIX} 체결 기록 변경 반영 ({summary['rebuilt_from']}부터 재계산)")

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
        buy_line = f"매수 {buy_count}"
        if buy_items:
            buy_line += " : " + ", ".join(buy_items)
        signal_lines.append(buy_line)

        sell_line = f"매도 {sell_count}"
        if sell_items:
            sell_line += " : " + ", ".join(sell_items)
        signal_lines.append(sell_line)

    unfilled_rows = summary.get("unfilled_rows", [])
    if unfilled_rows:
        names = ", ".join(r["티커"] for r in unfilled_rows)
        signal_lines.append(f"미체결 {len(unfilled_rows)} : {names}")

    # 라이브 어드바이저 1단계 (docs/design/live_advisor.md 3·7번): 판정이 바뀐 종목을
    # 맨 위 근처에 강조하고, 계획금액으로 1주도 못 사는 종목은 항상 경고한다.
    live_judgment_rows = summary.get("live_judgment_rows", [])
    changed_rows = [r for r in live_judgment_rows if r.get("changed")]
    if changed_rows:
        items = ", ".join(f"{r['ticker']}({r['judgment']})" for r in changed_rows)
        signal_lines.append(f"판정 변경 {len(changed_rows)} : {items}")
    warning_rows = [r for r in live_judgment_rows if r.get("one_share_warning")]
    if warning_rows:
        items = ", ".join(r["ticker"] for r in warning_rows)
        signal_lines.append(f"계획금액으로 1차 매수 0주 : {items}")
    # 기준가 기반 계획(구글 시트 계획 탭 개편, 2026-09-30): 보유 수량이 이미 계획
    # 총 주수 이상이면 추가 매수 추천이 0으로 잘리고 여기 경고로 뜬다.
    limit_exceeded_rows = [r for r in live_judgment_rows if r.get("plan_limit_exceeded")]
    if limit_exceeded_rows:
        items = ", ".join(r["ticker"] for r in limit_exceeded_rows)
        signal_lines.append(f"계획 한도 초과 : {items}")

    stop_alerts = summary.get("stop_alerts", [])
    if stop_alerts:
        items = ", ".join(f"{a['티커']}({a['type']})" for a in stop_alerts)
        signal_lines.append(f"손절 예약 {len(stop_alerts)} : {items}")

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
