"""웅덩이 반등 확인 v2 (시가총액 상위 10) — 가상 1억 장부 하루 실행 (docs/prereg/puddle_rebound_v2_forward.md, 태그 puddle-rebound-v2-prereg).

규칙·계산은 웅덩이 반등 확인 코드(core/puddle_rebound.py·engine/puddle_rebound.py) 그대로이고, 대상만 시가총액 상위 10,
비교 B는 웅덩이 v3.2 일별 기록(puddle_v32_daily, 읽기만)이다(erb.BOOK_V2). v3.2 장부 다음에 돈다.
저장: meta puddle_rebound_v2_state·표 puddle_rebound_v2_daily. 시트: 「웅덩이 반등 확인 v2 (가상·시총)」 탭(Actions에서만).

단독 미리보기:  python -m engine.puddle_rebound_v2 --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import puddle_rebound as erb  # noqa: E402

BOOK = erb.BOOK_V2


def run(cfg: dict, as_of, **kwargs) -> dict:
    """하루 실행(erb.run, book = 반등 확인 v2). 입력·출력은 erb.run과 같다."""
    return erb.run(cfg, as_of, book=BOOK, **kwargs)


def run_safely(cfg: dict, as_of, **kwargs) -> dict | None:
    """run을 감싼다 — 예외를 밖으로 내보내지 않는다(브리핑 보호). config puddle_rebound_v2.enabled가 아니면 None."""
    return erb.run_safely(cfg, as_of, book=BOOK, **kwargs)


def daily_history(conn) -> list[dict]:
    """일별 기록(표 puddle_rebound_v2_daily) 날짜 순."""
    return erb.daily_history(conn, BOOK)


def letter_lines(rows: list[dict]) -> dict:
    """진행한 날들 → 편지 줄(장부 이름 「웅덩이 반등 확인 v2」, 월요일 비교 「웅덩이 v3.2보다」)."""
    return erb.letter_lines(rows, BOOK)


def main(argv=None) -> None:
    erb.main(argv, book=BOOK)


if __name__ == "__main__":
    main()
