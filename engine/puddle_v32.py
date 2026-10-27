"""상승장 대형주 웅덩이 v3.2 (시가총액 상위 10) — 가상 1억 장부 하루 실행 (docs/prereg/puddle_v32_forward.md, 태그 puddle-v32-prereg).

규칙·계산은 웅덩이 v3.1 코드(core/puddle_v31.py·engine/puddle_v31.py) 그대로이고, 대상만 시가총액 상위 10(e31.BOOK_V32, ranking="mcap").
저장: data/virtual_books.db의 meta puddle_v32_state·표 puddle_v32_daily(드라이브 동기화). 시트: 「웅덩이 v3.2 (가상·시총)」 탭(Actions에서만).
첫 저장 때 옛 장부(puddle_v31·puddle_rebound)를 DB에 보관 표시한다(기록은 지우지 않음).
장부 오류는 run_safely가 잡는다 — 편지는 그대로 나간다.

단독 미리보기:  python -m engine.puddle_v32 --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import puddle_v31 as e31  # noqa: E402

BOOK = e31.BOOK_V32


def run(cfg: dict, as_of, **kwargs) -> dict:
    """하루 실행(e31.run, book = v3.2). 입력·출력은 e31.run과 같다."""
    return e31.run(cfg, as_of, book=BOOK, **kwargs)


def run_safely(cfg: dict, as_of, **kwargs) -> dict | None:
    """run을 감싼다 — 예외를 밖으로 내보내지 않는다(브리핑 보호). config puddle_v32.enabled가 아니면 None."""
    return e31.run_safely(cfg, as_of, book=BOOK, **kwargs)


def daily_history(conn) -> list[dict]:
    """일별 기록(표 puddle_v32_daily) 날짜 순."""
    return e31.daily_history(conn, BOOK)


def letter_lines(rows: list[dict]) -> dict:
    """진행한 날들 → 편지 줄(장부 이름 「웅덩이 v3.2」)."""
    return e31.letter_lines(rows, BOOK)


def main(argv=None) -> None:
    e31.main(argv, book=BOOK)


if __name__ == "__main__":
    main()
