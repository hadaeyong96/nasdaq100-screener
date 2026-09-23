"""P3.3 5번: 캐시 전체 점검 스크립트 — 누락·NaN 거래일을 찾아 복구하고 표로 보여준다.

일회성 점검용. engine.daily.run()이 쓰는 것과 같은 data.prices.fetch_universe_prices를
그대로 호출한다 — 캐시 우선이고, 구멍이 있으면 그 안에서 자동으로 복구를 시도한 뒤
(일봉 재조회 -> 60분봉 집계) 캐시 파일도 함께 갱신한다. 복구 못 한 구멍만 남는다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import yaml  # noqa: E402

from data.prices import fetch_universe_prices  # noqa: E402
from data.universe import get_universe  # noqa: E402


def main() -> None:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    universe = get_universe()
    tickers = universe["ticker"].tolist()
    print(f"캐시 점검 대상: {len(tickers)}종목\n")

    result = fetch_universe_prices(tickers, cfg)

    rows = []
    for ticker, msgs in result.warnings.items():
        recovered_msgs = [m for m in msgs if "누락 거래일 복구함" in m]
        failed_msgs = [m for m in msgs if "복구 실패" in m or "범위를 벗어나 버림" in m]
        if not recovered_msgs and not failed_msgs:
            continue
        rows.append(
            {
                "티커": ticker,
                "복구함": len(recovered_msgs),
                "복구실패": len(failed_msgs),
                "비고": "; ".join(recovered_msgs + failed_msgs),
            }
        )

    print(f"누락·NaN 거래일이 있던 종목 수: {len(rows)}")
    print(f"복구 성공(종목x건수): {sum(r['복구함'] for r in rows)}건")
    print(f"끝내 복구 실패(종목x건수): {sum(r['복구실패'] for r in rows)}건\n")

    print("| 티커 | 복구함 | 복구실패 | 비고 |")
    print("| --- | --- | --- | --- |")
    for r in rows:
        print(f"| {r['티커']} | {r['복구함']} | {r['복구실패']} | {r['비고']} |")

    if result.failed:
        print(f"\n종목 자체 조회 실패({len(result.failed)}개): {result.failed}")


if __name__ == "__main__":
    main()
