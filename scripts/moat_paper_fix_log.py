"""해자 paper 수정 사유 기록 (사용자 지시 2026-10-02).

진입가를 채우기 전에 고친 두 건을 store/trials.db에 남긴다(scripts/moat_lock.py와 같은 로그):
1. GOOG 유통주식수 이중 계산 — core.moat_paper.aggregate_shares_by_cik
2. 시작 파일 쓰기 방식 — scripts.moat_paper.write_start_file_guarded

실행(한 번만):
    python -u -m scripts.moat_paper_fix_log
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from store import trials as trials_store  # noqa: E402

GOOG_FIX_REASON = (
    "진입가 드라이런(2026-10-02)에서 GOOG 시가총액이 8.57T로 나왔다. GOOG·GOOGL은 CIK가 같아 "
    "둘 다 회사 전체 희석주식수(12.309B, dei 표지 공시 없음 → 우선순위 2번)를 받아 왔는데 "
    "aggregate_shares_by_cik가 티커별 값을 그대로 더해 24.618B가 됐다. CIK마다 서로 다른 값만 "
    "한 번씩 더하도록 고쳤다(종류별로 다른 값은 여전히 합산). 시작 파일의 shares_outstanding은 "
    "덮어쓰지 않고 shares_outstanding_corrections에 보정값을 따로 남겨 P2 시가총액에만 쓴다. "
    "GOOG는 수정 전·후 모두 종목 10% 상한에 걸려 P2 최종 비중은 같다(최대 차이 3e-17)."
)

WRITE_FIX_REASON = (
    "잠긴 설계(moat_paper.md \"진입\")대로 진입가는 시작 파일의 진입 대기 칸에 채우되, 파일 전체를 "
    "다시 쓰는 방식이라 안전장치를 더했다: 쓰기 전 원본을 보관하고 쓴 뒤 다시 읽어 "
    "core.moat_paper.find_protected_changes로 비교 — 바뀌어도 되는 칸(진입 대기였던 entries, "
    "P1·P2 확정 비중, p2_market_caps, entered_at, shares_outstanding_corrections, status 진입 대기→체결 완료) "
    "밖이 하나라도 바뀌면 원본으로 되돌리고 멈춘다. 디스크 파일이 이미 체결 완료면 쓰기 자체를 거부한다."
)


def main() -> None:
    run_at = datetime.now().isoformat(timespec="seconds")
    conn = trials_store.connect()
    try:
        goog_id = trials_store.record_trial(
            conn, run_at=run_at, label="해자 paper GOOG 유통주식수 이중 계산 수정",
            metrics={
                "tickers": ["GOOG", "GOOGL"], "cik": 1652044,
                "shares_before": 24_618_000_000, "shares_after": 12_309_000_000,
                "goog_market_cap_before_b": 8565, "goog_market_cap_after_b": 4283,
                "p2_weight_max_diff": 3.47e-17,
            },
            notes=GOOG_FIX_REASON,
        )
        write_id = trials_store.record_trial(
            conn, run_at=run_at, label="해자 paper 시작 파일 쓰기 방식 보호",
            metrics={"start_file": "docs/paper/moat_paper_start.json", "guard": "write_start_file_guarded"},
            notes=WRITE_FIX_REASON,
        )
    finally:
        conn.close()
    print(f"기록 완료: GOOG 수정 trial_id={goog_id}, 시작 파일 쓰기 방식 trial_id={write_id}", flush=True)


if __name__ == "__main__":
    main()
