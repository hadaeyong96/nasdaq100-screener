"""전략공장 시트(GOOGLE_SHEETS_ID) 모든 탭을 탭별 CSV로 내려받는다 (2026-11-13, 읽기만).

사용
    python scripts/sheet_backup.py [--out outputs/sheet_backup_YYYY-MM-DD]

탭 이름·행 수를 출력하고 하나라도 실패하면 종료 코드 1. 인증 정보·시트 ID 값은 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def safe_name(title: str) -> str:
    """탭 이름 → 파일 이름(윈도우에서 못 쓰는 글자만 _로)."""
    return re.sub(r'[\\/:*?"<>|]', "_", title).strip() or "_"


def backup(spreadsheet, out: Path) -> tuple[list[tuple[str, int, Path]], list[str]]:
    """모든 탭 → out/<번호>_<탭 이름>.csv. 출력: ([(탭, 행 수, 파일)], [실패 메시지])."""
    out.mkdir(parents=True, exist_ok=True)
    done, failed = [], []
    for i, ws in enumerate(spreadsheet.worksheets(), 1):
        try:
            values = ws.get_all_values()
            path = out / f"{i:02d}_{safe_name(ws.title)}.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                csv.writer(f).writerows(values)
            done.append((ws.title, len(values), path))
        except Exception as exc:  # noqa: BLE001 — 실패 탭을 모아 보고
            failed.append(f"{ws.title}: {type(exc).__name__}")
    return done, failed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="전략공장 시트 탭별 CSV 백업(읽기만)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / f"sheet_backup_{date.today().isoformat()}"))
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # 값은 출력하지 않는다
    from data import sheets

    sp = sheets.get_client().open_by_key(sheets._sheet_id())
    done, failed = backup(sp, Path(a.out))
    for title, n, path in done:
        print(f"  {title}: {n}행 → {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    for msg in failed:
        print(f"  실패 {msg}")
    print(f"백업 {len(done)}개 탭, 실패 {len(failed)}개")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
