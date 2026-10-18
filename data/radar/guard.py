"""신기술 레이더 읽기 가드 — 가격·수익 캐시를 열지 못하게 막는다 (G0 지시: "가격·수익 캐시를 읽지 말 것(코드로 막기)").

install()을 부르면 파이썬 감사 훅(sys.addaudithook)이 모든 open을 본다. data/cache/ 아래는 허용 목록만 열 수 있고,
그 밖(가격·수익·장부·백테스트 캐시)은 RadarGuardError로 멈춘다. *.db(장부 DB)도 막는다. 한 번 설치하면 풀 수 없다(감사 훅의 성질).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = (ROOT / "data" / "cache").resolve()
# data/cache 아래에서 열어도 되는 것: 레이더 자체 캐시 + 시점별 티커 명단(가격 아님)
ALLOWED_DIRS = [CACHE / "arxiv", CACHE / "radar"]
ALLOWED_FILES = [
    CACHE / "sp500" / "fja05680_updated.csv",
    CACHE / "sp500" / "identity.csv",
    CACHE / "sp500" / "final_gap.csv",
    CACHE / "sp500" / "renames.csv",
    CACHE / "sw1" / "cik.csv",
    CACHE / "sw1" / "sp100_monthly.csv",
    CACHE / "nasdaq100_changes.json",
    CACHE / "edgar" / "company_tickers.json",
]
_installed = False


class RadarGuardError(PermissionError):
    """레이더 작업이 가격·수익 캐시(또는 장부 DB)를 열려고 했다."""


def allowed(path) -> bool:
    """이 경로를 레이더 작업이 열어도 되는지 (순수 판정)."""
    try:
        p = Path(os.fsdecode(path)).resolve()
    except (TypeError, ValueError, OSError):
        return True  # 파일 기술자(int) 등 — 경로가 아님
    if p.suffix.lower() == ".db":
        return False
    try:
        p.relative_to(CACHE)
    except ValueError:
        return True  # data/cache 밖(코드·문서·임시 폴더)
    return p in ALLOWED_FILES or any(p == d or d in p.parents for d in ALLOWED_DIRS)


def _hook(event: str, args) -> None:
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) and not allowed(args[0]):
        raise RadarGuardError(f"레이더 가드: 가격·수익 캐시는 열 수 없습니다 — {os.fsdecode(args[0])}")


def install() -> None:
    """감사 훅을 한 번만 건다."""
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
