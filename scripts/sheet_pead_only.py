"""전략공장 시트 실적급등 전용 정리 — 1회 실행 (2026-11-13, 사용자 결정).

사용
    python scripts/sheet_pead_only.py            보기만: 지울 탭, 새 탭에 처음 채울 줄 수 (쓰기 0번)
    python scripts/sheet_pead_only.py --apply    ① 새 탭 2개(「일별 성적」·「매매이력」)를 가상 장부 DB 기록으로 처음 채움
                                                 ② 웅덩이·QQQM 탭 삭제 (시트 파일은 그대로)

순서가 채우기 → 삭제인 이유: 구글 시트는 탭이 하나는 있어야 한다(지금 탭 3개가 모두 지울 대상).
삭제 전 확인: 지울 탭마다 백업 CSV(--backup, scripts/sheet_backup.py)가 있어야 한다. 없으면 아무것도 지우지 않는다.
DB는 읽기만(data/virtual_books.db — 먼저 status_check --download로 드라이브 최신본을 받는다). 인증 정보·시트 ID는 출력하지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def tabs_to_delete(titles: list[str]) -> list[str]:
    """지울 탭: 정해 둔 웅덩이·QQQM 탭 + 이름에 「웅덩이」·「QQQM」이 들어간 다른 탭. 실적급등 탭은 절대 넣지 않는다."""
    from data import pead_sheet

    out = [t for t in titles if t in pead_sheet.RETIRED_TABS or "웅덩이" in t or "QQQM" in t]
    return [t for t in out if t not in pead_sheet.TABS]


def backed_up(titles: list[str], backup: Path) -> list[str]:
    """백업 CSV가 없는 탭 목록(scripts/sheet_backup.safe_name 이름 규칙, 앞 번호 무관)."""
    from scripts.sheet_backup import safe_name

    have = {p.stem.split("_", 1)[1] for p in backup.glob("*.csv") if "_" in p.stem} if backup.exists() else set()
    return [t for t in titles if safe_name(t) not in have]


def build_rows(cfg: dict) -> tuple[list[list], list[list]]:
    """가상 장부 DB(읽기만) + 보유 종목·QQQ 가격 → (「일별 성적」 줄, 「매매이력」 줄). 예약 실행과 같은 함수(engine.pead_watch.sheet_rows)."""
    import pandas as pd

    from core import pead_timing as pt
    from data.market_calendar import latest_closed_trading_day
    from engine import pead_watch as ew
    from engine.daily import US_EASTERN
    from store import pead_watch as st
    from store import virtual_books as vbstore

    conn = vbstore.connect(Path(vbstore.VBOOKS_DB_PATH))
    try:
        last = st.get_meta(conn, st.META_LAST)
        if not last:
            raise RuntimeError("실적급등 장부 기록이 없습니다(pead_watch_last)")
        as_of = pd.Timestamp(min(pd.Timestamp(last).date(), latest_closed_trading_day(datetime.now(US_EASTERN))))
        src = ew.live_sources(cfg)
        days = src.calendar((as_of - pd.Timedelta(days=520)).date(), (as_of + pd.Timedelta(days=420)).date())
        trades = [t for t in st.load_trades(conn) if t["ledger"] == "main"]
        want = sorted({t["ticker"] for t in trades if t["status"] in ("open", "closed")} | {"QQQ"})
        al = {t: pt.align(src.prices(t).loc[:as_of], days) for t in want}
        names = ew._name_kr()
        return ew.sheet_rows(conn, al, al["QQQ"], days, as_of, names)
    finally:
        conn.close()


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="전략공장 시트 실적급등 전용 정리(1회)")
    ap.add_argument("--apply", action="store_true", help="실제로 채우고 지움(없으면 보기만)")
    ap.add_argument("--backup", default=str(ROOT / "outputs" / "sheet_backup_2026-11-13"))
    a = ap.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # 값은 출력하지 않는다
    from data import pead_sheet, sheets
    from engine.daily import load_config

    sp = sheets.get_client().open_by_key(sheets._sheet_id())
    titles = [ws.title for ws in sp.worksheets()]
    doomed = tabs_to_delete(titles)
    print(f"지금 탭: {', '.join(titles)}")
    print(f"지울 탭: {', '.join(doomed) or '없음'}")
    missing = backed_up(doomed, Path(a.backup))
    if missing:
        print(f"백업 CSV 없음 → 중단(아무것도 바꾸지 않음): {', '.join(missing)}")
        return 1
    daily, trades = build_rows(load_config())
    print(f"처음 채울 줄: 「{pead_sheet.DAILY_TAB}」 {len(daily)}줄, 「{pead_sheet.TRADE_TAB}」 {len(trades)}줄")
    for r in daily + trades:
        print("  ", r)
    if not a.apply:
        print("보기만 — 바꾼 것 없음(--apply로 실행)")
        return 0
    n = pead_sheet.write_all(sheets.get_client(), sheets._sheet_id(), daily, trades)
    print(f"채움: {n}")
    for t in doomed:
        sp.del_worksheet(sp.worksheet(t))
        print(f"삭제: {t}")
    print(f"남은 탭: {', '.join(ws.title for ws in sp.worksheets())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
