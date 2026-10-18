"""8번 「가치평가 따라하기」 미리 보기 (dry-run) — 그 월말 점수로 시작일에 고를 10종목. DB·시트·텔레그램에 쓰지 않는다.

    python scripts/value_book_preview.py 2026-09-30 [--env 경로]

--env: SEC User-Agent를 읽을 .env 경로(값은 출력하지 않는다). 출력은 개인 확인용(단체방에 올리지 않는다).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import value_book as vbk  # noqa: E402
from core import value_score as vs  # noqa: E402


def main() -> None:
    for s in (sys.stdout, sys.stderr):
        s.reconfigure(encoding="utf-8")
    me = pd.Timestamp(sys.argv[1])
    if "--env" in sys.argv:
        from dotenv import load_dotenv

        load_dotenv(sys.argv[sys.argv.index("--env") + 1])
    from data import value_live

    t0 = time.time()
    df, warns = value_live.compute_scores(me)
    bd = vbk.board(df)
    u = df[df["universe"]]
    comp = u[u["total"].notna()]
    print(f"월말 {me.date()} · 명단 점수 {len(df)} · 대상 {len(u)} · 점수 완전 {len(comp)} · 실패 {len(warns)} · {time.time() - t0:.0f}초")
    p = vbk.ValueParams(name="미리 보기")
    order = [t for t in bd["order"] if bd["total"][t] >= p.min_pick]
    picks = vs.pick(order, p.slots, bd["sic2"], [], p.sector_cap)
    print(f"49점(표시 70) 이상 {len(order)}종목 → 시작 후보 {len(picks)}종목 (업종 최대 3):")
    for i, t in enumerate(picks, 1):
        r = df.loc[t]
        print(f"  {i:2d}. {t:6s} {vbk.display_score(r['total']):5.1f}점(100점) · SIC {r['sic2']} · PEG {r['peg_value']:.2f}")
    if warns:
        print("점수 못 낸 종목(앞 15): " + "; ".join(warns[:15]))
    df.to_csv(ROOT / "outputs" / f"value_preview_{me.date()}.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    main()
