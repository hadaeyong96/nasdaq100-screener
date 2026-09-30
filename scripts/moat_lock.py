"""해자 분석 기준 잠금 기록 (사용자 지시 2026-10-01).

두 가지를 store/trials.db(N_raw 시도 로그, docs/design/fund_sim.md 2.2 "시도 기록"과 같은
DB — 해자 분석도 같은 로그를 쓴다)에 남긴다:
1. M4 지표를 "영업현금흐름÷순이익"(기존 "(영업현금흐름-설비투자)÷순이익"에서)으로 바꾼 이유
   — 결과를 본 뒤 정한 변경이라 사유를 기록해야 한다.
2. 기준 잠금 자체 — config.yaml의 moat: 섹션 해시와 그 내용을 커밋한 git 커밋 해시.

실행 순서(반드시 이 순서):
    python -u -m scripts.moat_lock --step m4        # 커밋 전에 먼저
    git commit ...; git tag moat-v1.1-locked
    python -u -m scripts.moat_lock --step lock --commit <해시>   # 커밋 뒤에
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import moat  # noqa: E402
from store import trials as trials_store  # noqa: E402

M4_CHANGE_REASON = (
    "라운드3(2026-10-01)에서 M4 주 지표(기존식, (영업현금흐름-설비투자)/순이익, 0.8 기준)와 "
    "참고 비율(영업현금흐름/순이익, 0.9 기준)을 나란히 비교한 결과 MSFT·CPRT·TXN·TSLA·FER·"
    "ALAB·SNDK·CRWV 8개 종목의 등급이 갈렸다. 설비투자는 연도별로 들쭉날쭉해(대형 투자 시기 "
    "vs 아닌 시기) 자본배분 방식이 달라도 '이익이 진짜 현금인가'라는 M4의 원래 질문과는 별개로 "
    "지표가 출렁인다고 판단 — 영업현금흐름/순이익을 주 지표로, 기존식은 참고 열로 확정했다."
)


def record_m4_change(run_at: str) -> int:
    """M4 지표 변경 사유를 trials.db에 기록한다."""
    conn = trials_store.connect()
    try:
        return trials_store.record_trial(
            conn, run_at=run_at, label="해자 M4 지표 변경",
            metrics={
                "old_main_formula": "(영업현금흐름-설비투자)/순이익, good_min_ratio=0.8",
                "new_main_formula": "영업현금흐름/순이익, good_min_ratio=0.9",
                "reference_formula": "(영업현금흐름-설비투자)/순이익, reference_good_min_ratio=0.8",
                "changed_tickers": ["MSFT", "CPRT", "TXN", "TSLA", "FER", "ALAB", "SNDK", "CRWV"],
            },
            notes=M4_CHANGE_REASON,
        )
    finally:
        conn.close()


def record_lock(run_at: str, config_hash: str, commit_hash: str, tag_name: str) -> int:
    """기준 잠금(config.yaml moat: 섹션 해시 + git 커밋 해시)을 trials.db에 기록한다."""
    conn = trials_store.connect()
    try:
        return trials_store.record_trial(
            conn, run_at=run_at, label="해자 지표 v1.1 확정 및 잠금",
            metrics={"config_hash": config_hash, "commit_hash": commit_hash, "tag": tag_name},
            config_hash=config_hash,
            notes=(
                f"docs/design/moat_plan.md v1.1 + config.yaml moat: 섹션을 커밋 {commit_hash}로 잠금, "
                f"태그 {tag_name}. 이후 core.moat.check_lock(cfg['moat'], '{config_hash}')로 설정이 "
                "이 기준과 같은지 확인할 수 있다 — H4(과거 검증)는 실행 전에 이 확인을 거쳐야 한다."
            ),
        )
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["m4", "lock"], required=True)
    parser.add_argument("--commit", help="'lock' 단계에서 필요 — 기준을 잠근 git 커밋 해시")
    parser.add_argument("--tag", default="moat-v1.1-locked")
    args = parser.parse_args()

    run_at = datetime.now().isoformat(timespec="seconds")

    if args.step == "m4":
        trial_id = record_m4_change(run_at)
        print(f"M4 지표 변경 사유 기록 완료 (trial_id={trial_id})", flush=True)
        return

    if not args.commit:
        raise SystemExit("--step lock에는 --commit <커밋 해시>가 필요합니다.")

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    config_hash = moat.compute_config_hash(cfg["moat"])
    trial_id = record_lock(run_at, config_hash, args.commit, args.tag)
    print(f"기준 잠금 기록 완료 (trial_id={trial_id})", flush=True)
    print(f"config_hash={config_hash}", flush=True)
    print(f"commit={args.commit} tag={args.tag}", flush=True)


if __name__ == "__main__":
    main()
