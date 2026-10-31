"""보고서 다이어트 견본 (일회성 점검, 2026-10-31) — 실제 발송·DB·시트 쓰기 없음(dry-run).

오늘 확정 봉으로 engine.daily.run(dry_run=True) → 편지 재료(시장 지수·시장 온도·공포 매수판·내 돈·뉴스, 읽기만) →
웅덩이 v3.2·반등 확인 v2 장부는 --vbooks-db(드라이브에서 받은 사본 등)로 dry_run 실행(쓰지 않음) → 견본 4개를 그린다.
  평일: 오늘 그대로 · 월요일: 편지 날짜만 다음 월요일 · 저축일: 편지 날짜 11/17 + 저축일 · 공포: 공포 매수판만 가짜 값(4/5)
각 견본의 개인판·단체방판 HTML을 --out(기본 outputs/report_diet_samples/)에 쓰고, 접기를 닫은 상태의 본문을 출력한다.

실행: python scripts/report_diet_samples.py --vbooks-db <virtual_books.db 사본> [--out 폴더]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import daily  # noqa: E402
from notify import briefing, report_html  # noqa: E402
from scripts.letter_preview import visible_text  # noqa: E402
from thermometer.core import load_config as load_thermo  # noqa: E402
from thermometer.fear_board import board  # noqa: E402


def base_summary(cfg: dict, vbooks_db: Path, out: Path) -> dict:
    """오늘 편지 재료(읽기만) — engine.daily.main과 같은 순서, 장부는 사본 DB로 dry_run."""
    from engine import puddle_rebound_v2, puddle_v32

    daily.OUTPUT_DIR = out
    s = daily.run(cfg, "live", do_replay=False, dry_run=True)
    if s.get("skipped"):
        raise SystemExit("기준일이 이미 처리돼 요약이 비었습니다")
    daily._attach_market_glance(s, cfg)
    daily._attach_thermometer(s)  # record=False — 기록 안 남김
    daily._attach_fear_board(s)
    holdings = daily._read_holdings(cfg)
    daily._attach_dec_tax(s, cfg, holdings=holdings)
    daily._attach_letter_v3(s, cfg, holdings)
    daily._attach_puddle_v31(s, cfg, run_fn=lambda c, d: puddle_v32.run_safely(c, d, db_path=vbooks_db, dry_run=True,
                                                                                 sheet_write=False))
    daily._attach_puddle_rebound(s, cfg, run_fn=lambda c, d: puddle_rebound_v2.run_safely(c, d, db_path=vbooks_db, dry_run=True,
                                                                                          sheet_write=False))
    daily._attach_curve_chart(s, cfg, out_dir=out, db_path=vbooks_db)
    daily._attach_candidate_charts(s, out_dir=out)
    daily._attach_news(s, cfg)
    return s


def variants(s: dict, cfg: dict) -> dict:
    """견본 4개 — 평일(오늘 그대로)·월요일·저축일(11/17)·공포 구간(공포 매수판만 가짜 값)."""
    from notify import letter

    as_of = pd.Timestamp(s["as_of"])
    monday = as_of + pd.offsets.Week(weekday=0)
    mon = {**copy.copy(s), "letter_today_kst": monday, "research_news": letter.research_lines(cfg, monday)}
    sav = {**copy.copy(s), "letter_today_kst": pd.Timestamp("2026-11-17"), "savings_day": True}
    fake = {"fear_greed": 18, "rsi": 27.4, "close": 600.0, "bb_lower": 606.2, "sma_long": 640.0, "sma_short": 625.0,
            "sma_short_lag": 641.0, "vix": 26.5}  # 가짜 값: ① ② ③ ④ 켜짐, ⑤ 꺼짐 → 4/5 공포
    day = as_of.date().isoformat()
    fear = {**copy.copy(s), "fear_board": {"date": day, "data_date": day, **board(fake, load_thermo())}}
    return {"weekday": s, "monday": mon, "savings": sav, "fear": fear}


def main() -> None:
    for st in (sys.stdout, sys.stderr):
        st.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vbooks-db", required=True, help="가상 장부 DB 사본(쓰지 않음)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "report_diet_samples"))
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    cfg = daily.load_config()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    s = base_summary(cfg, Path(a.vbooks_db), out)
    for name, v in variants(s, cfg).items():
        d = out / name
        d.mkdir(exist_ok=True)
        for edition in ("personal", "group"):
            p = report_html.render_report(v, cfg, d, edition=edition)
            print(f"\n===== {name} · {edition} — {p} =====")
            print("[설명 한 줄] " + briefing.build_report_caption(v, cfg, edition=edition))
            print(visible_text(p.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
