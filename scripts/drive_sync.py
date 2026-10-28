"""구글 드라이브 상태 저장 동기화 (라이브 어드바이저 1단계 1g,
docs/design/live_advisor.md 5번).

GitHub Actions 러너는 실행마다 새 가상머신이라 data/state.db·
data/paper_state.db가 실행 사이에 안 남는다 — 엔진 실행 전 download로
사용자가 미리 만들어 둔 드라이브 빈 파일(체크리스트 참고)의 내용을 로컬
DB로 내려받고, 실행 후 upload로 같은 파일에 다시 덮어쓴다(파일을 새로
만들지 않는다 — store/drive.py 모듈 설명 참고).

실행:
    python scripts/drive_sync.py download --mode live
    python scripts/drive_sync.py download --mode paper
    python scripts/drive_sync.py upload --mode live
    python scripts/drive_sync.py upload --mode paper
    python scripts/drive_sync.py download --mode vbooks   (가상 장부 data/virtual_books.db)
    python scripts/drive_sync.py upload --mode vbooks
    python scripts/drive_sync.py download --mode thermo   (시장 온도계 기록 data/thermometer/history.csv)
    python scripts/drive_sync.py upload --mode thermo

파일 ID는 환경변수 GOOGLE_DRIVE_STATE_FILE_ID_LIVE / GOOGLE_DRIVE_STATE_FILE_ID_PAPER /
GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS / GOOGLE_DRIVE_STATE_FILE_ID_THERMO에서 읽는다. 로컬 개발처럼 해당 환경변수가 없으면 아무것도 안 하고 조용히
넘어간다 — 드라이브 없이도 지금처럼 로컬 DB 파일로 그대로 돈다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from store import db  # noqa: E402
from store import drive  # noqa: E402
from store import virtual_books as vbstore  # noqa: E402

_FILE_ID_ENV = {
    "live": "GOOGLE_DRIVE_STATE_FILE_ID_LIVE",
    "paper": "GOOGLE_DRIVE_STATE_FILE_ID_PAPER",
    "vbooks": "GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS",
    "thermo": "GOOGLE_DRIVE_STATE_FILE_ID_THERMO",
}
THERMO_HISTORY_PATH = ROOT / "data" / "thermometer" / "history.csv"


def local_path_for_mode(mode: str) -> Path:
    """모드별 로컬 DB 경로 — vbooks는 가상 장부 DB(store/virtual_books.py), thermo는 시장 온도계 기록 CSV."""
    if mode == "thermo":
        return THERMO_HISTORY_PATH
    return vbstore.VBOOKS_DB_PATH if mode == "vbooks" else db.db_path_for_mode(mode)


def sync(direction: str, mode: str) -> bool:
    """direction("download"|"upload")·mode("live"|"paper"|"vbooks") 하나를 동기화한다.

    출력: 실제로 동기화를 수행했으면 True, 파일 ID 환경변수가 없어 건너뛰었으면 False.
    """
    env_key = _FILE_ID_ENV[mode]
    file_id = (os.environ.get(env_key) or "").strip()
    if not file_id:
        print(f"[drive_sync] {env_key}가 없어 {mode} 구글 드라이브 동기화를 건너뜁니다 (로컬 DB 그대로 씀).")
        return False

    local_path = local_path_for_mode(mode)
    if direction == "download":
        drive.download_from_drive(file_id, local_path)
        print(f"[drive_sync] {mode}: 드라이브 -> {local_path} 다운로드 완료")
    else:
        drive.upload_to_drive(file_id, local_path)
        print(f"[drive_sync] {mode}: {local_path} -> 드라이브 업로드 완료")
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("direction", choices=["download", "upload"])
    parser.add_argument("--mode", choices=list(_FILE_ID_ENV), required=True)
    args = parser.parse_args(argv)
    sync(args.direction, args.mode)


if __name__ == "__main__":
    main()
