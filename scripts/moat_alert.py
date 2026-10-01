"""해자 약화 경보 B3 보고서 (docs/p6_2a_instructions.md 3장, 사용자 지시 2026-10-01).

docs/paper/moat_paper_start.json의 P1·P2 보유 종목(둘 다 "넓음 전부"라 같은 종목 집합)을
대상으로 core/moat_alert.py의 경보 3종(이익률·수익성·등급)을 평가해
outputs/moat_alert_YYYY-MM.md로만 남긴다 — 텔레그램 연결 없음.

실행(.env에 SEC_USER_AGENT 필요, scripts/moat_paper.py가 먼저 시작 파일을 만들어 둬야 함):
    python -u -m scripts.moat_alert
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import moat, moat_alert  # noqa: E402
from data import edgar  # noqa: E402
from scripts.moat_paper import START_PATH, load_config  # noqa: E402
from store import moat_paper as store_mp  # noqa: E402

OUT_DIR = ROOT / "outputs"


def main() -> None:
    if not START_PATH.exists():
        raise SystemExit(f"{START_PATH}가 없습니다 — scripts/moat_paper.py를 먼저 실행해 보유 종목을 만드세요.")

    import json

    start = json.loads(START_PATH.read_text(encoding="utf-8"))
    cfg = load_config()
    user_agent = edgar.get_user_agent()
    as_of = date.today()
    period = as_of.strftime("%Y-%m")

    tickers = sorted(start["wide_tickers"])
    print(f"B3 경보 대상(P1·P2 보유, 넓음 전부) {len(tickers)}개")

    conn = store_mp.connect()
    results = []
    try:
        for t in tickers:
            cik = start["cik"].get(t)
            previous_grade = store_mp.latest_grade_before(conn, t, period) or start["grades"].get(t, "넓음")
            if cik is None:
                results.append({"ticker": t, "error": "CIK 없음", "previous_grade": previous_grade})
                continue
            try:
                facts = edgar.fetch_company_facts(cik, user_agent=user_agent)
            except edgar.EdgarFetchError as exc:
                results.append({"ticker": t, "error": f"companyfacts 요청 실패: {exc}", "previous_grade": previous_grade})
                continue
            profile = moat.analyze_company(facts, t, cfg, as_of)
            alerts = moat_alert.evaluate_company_alerts(t, facts, cfg, as_of, previous_grade, profile)
            results.append(alerts)
    finally:
        conn.close()

    triggered = [r for r in results if r.get("any_triggered")]
    lines = [
        f"# 해자 약화 경보 {period}",
        "",
        f"평가 시각: {datetime.now().isoformat(timespec='seconds')} · 평가 기준일: {as_of.isoformat()}",
        f"대상 {len(tickers)}개 종목 · 경보 발생 {len(triggered)}개",
        "",
        "## 경보 발생 종목" if triggered else "## 경보 발생 종목 — 없음",
    ]
    for r in triggered:
        lines.append(f"### {r['ticker']}")
        lines.append(f"- 이익률 경보: {r['gross_margin']['status']} — {r['gross_margin']['detail']}")
        lines.append(f"- 수익성 경보: {r['profitability']['status']} — {r['profitability']['detail']}")
        lines.append(f"- 등급 경보: {r['grade']['status']} — {r['grade']['detail']}")
        lines.append("")

    lines.append("## 전체 종목 상세")
    for r in results:
        if "error" in r:
            lines.append(f"- {r['ticker']}: 평가 불가 — {r['error']}")
            continue
        lines.append(
            f"- {r['ticker']}: 이익률={r['gross_margin']['status']}, 수익성={r['profitability']['status']}, "
            f"등급={r['grade']['status']}({r['grade']['detail']})"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"moat_alert_{period}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out_path}")
    print(f"경보 발생 {len(triggered)}개: {[r['ticker'] for r in triggered]}")


if __name__ == "__main__":
    main()
