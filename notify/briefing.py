"""텔레그램 짧은 브리핑 글 (P3, 4번).

휴대폰에서 읽기 좋게 짧게 쓰고, 자세한 내용은 첨부하는 HTML 보고서를 보라고 안내한다.
이 모듈은 summary dict(engine/daily.py) + cfg만 받아 문자열을 만든다 (순수 함수,
네트워크·파일 없음 — 테스트가 summary를 직접 만들어 호출할 수 있다).
"""

from __future__ import annotations

_STAGE_ORDER = ["b1", "b2", "b3", "b9"]
_STAGE_TITLE = {"b1": "1차 · RSI 30 탈출", "b2": "2차 · 골든크로스", "b3": "3차 · 구름 돌파", "b9": "재진입"}


def build_briefing_text(summary: dict, cfg: dict) -> str:
    """summary + cfg -> 텔레그램에 보낼 본문 문자열."""
    as_of = summary.get("as_of")
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"
    mode_label = summary.get("mode_label", "실전")

    lines = [f"나스닥100 브리핑 ({mode_label})", f"{as_of_str} 미국장 마감", ""]

    buy_count = summary.get("buy_count", 0)
    sell_count = len(summary.get("sell_rows", []))
    lines.append(f"[오늘 할 일] 매수 {buy_count} · 매도·손절 {sell_count}")
    lines.append("")

    buy_groups = summary.get("buy_groups", {})
    any_buy = False
    for key in _STAGE_ORDER:
        rows = buy_groups.get(key, [])
        if not rows:
            continue
        any_buy = True
        lines.append(f"━━━ 매수 · {_STAGE_TITLE[key]} ({len(rows)}) ━━━")
        for r in rows:
            stop_txt = f"{r['stop']:.2f}" if r.get("stop") is not None else "미확정"
            lines.append(f"{r['kr']} {r['limit']:.2f} / {r.get('qty', 0)}주 / 손절 {stop_txt}")
        lines.append("")
    if not any_buy:
        lines.append("매수 없음")
        lines.append("")

    sell_rows = summary.get("sell_rows", [])
    if sell_rows:
        lines.append("━━━ 매도·손절 ━━━")
        for r in sell_rows:
            lines.append(f"{r['종목명']} {r['신호']} · {r['매도범위']} · {r['수량']}주")
        lines.append("")
    else:
        lines.append("매도·손절 없음")
        lines.append("")

    hold_signal_rows = [r for r in summary.get("hold_rows", []) if r.get("오늘신호")]
    if hold_signal_rows:
        lines.append("━━━ 보유 종목 오늘 신호 ━━━")
        for r in hold_signal_rows:
            lines.append(f"{r['종목명']} {r['오늘신호']}")
        lines.append("")

    unfilled_rows = summary.get("unfilled_rows", [])
    if unfilled_rows:
        lines.append("━━━ 미체결 알림 ━━━")
        for r in unfilled_rows:
            lines.append(f"{r['종목명']} {r['단계']} {r['내용']}")
        lines.append("")

    lines.append("상세는 첨부 보고서를 확인하세요.")
    return "\n".join(lines).strip() + "\n"
