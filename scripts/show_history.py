"""검토용 점검 스크립트: 한 종목의 이벤트 이력(신호, blocked, 체결 가정,
상태 변화, 손절가 변화)을 날짜순 표로 출력한다 (P2.1 보완 4번).

data/state.db의 events 테이블을 읽는다 (네트워크 없음). engine.daily 실행으로
쌓인 기록이 있어야 한다.

실행: python scripts/show_history.py TICKER
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from store import db  # noqa: E402


def _format_detail(kind: str, detail: dict) -> str:
    """이벤트 종류별로 상세 내용을 한 줄로 읽기 쉽게 만든다."""
    if kind == "BLOCKED":
        reasons = ", ".join(detail.get("reasons", []))
        return f"{detail.get('stage')} 차단({detail.get('blocked_type')}): {reasons}"
    if kind in ("VIRTUAL_FILL", "FILL"):
        label = "가상체결" if kind == "VIRTUAL_FILL" else "실제체결"
        side = detail.get("side", "buy")
        return f"{label} {detail.get('unit')}묶음 {side} {detail.get('qty')}주 @ {detail.get('price')}"
    if "price" in detail:  # A1/A2/A3/B 진입 신호
        parts = [f"{detail.get('unit')}묶음 신호가 {detail.get('price')}"]
        if detail.get("grade"):
            parts.append(f"등급 {detail['grade']}")
        if "score" in detail:
            parts.append(f"점수 {detail['score']}")
        return " / ".join(parts)
    if "qty" in detail:  # STOP/A1_EXPIRE/E3/E1/E2 매도 신호
        return f"{detail.get('unit')}묶음 {detail.get('qty')}주 매도"
    return json.dumps(detail, ensure_ascii=False)


def show_history(ticker: str, db_path: Path | None = None) -> list[dict]:
    """이 종목의 이벤트 이력을 날짜순 표(dict 목록)로 만든다.

    입력: ticker, db_path(기본 data/state.db)
    출력: [{날짜, 이벤트, 상세, 상태, 손절가}, ...]
    """
    conn = db.connect(db_path or db.DB_PATH)
    rows = conn.execute("SELECT date, kind, detail FROM events WHERE ticker = ? ORDER BY date, id", (ticker,)).fetchall()
    conn.close()

    table = []
    for date, kind, detail_json in rows:
        detail = json.loads(detail_json) if detail_json else {}
        table.append(
            {
                "날짜": date,
                "이벤트": kind,
                "상세": _format_detail(kind, detail),
                "상태": detail.get("state_after", ""),
                "손절가": detail.get("stop_after", ""),
            }
        )
    return table


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(기록 없음)"
    columns = list(rows[0].keys())
    widths = [max(len(str(r[c])) for r in rows + [dict(zip(columns, columns))]) for c in columns]
    header = " | ".join(c.ljust(w) for c, w in zip(columns, widths))
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(str(row[c]).ljust(w) for c, w in zip(columns, widths)))
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) != 2:
        print("사용법: python scripts/show_history.py TICKER")
        sys.exit(1)
    ticker = sys.argv[1].upper()
    table = show_history(ticker)
    print(f"[{ticker} 이벤트 이력]")
    print(format_table(table))


if __name__ == "__main__":
    main()
