"""보고서·텔레그램 글을 상태를 다시 계산하지 않고 다시 만드는 스크립트.

state.db(또는 --mode paper의 paper_state.db)에 이미 저장된 포지션과 그 날짜의
이벤트(store.db.get_events_for_date)만 읽어 outputs/report_YYYY-MM-DD.html과
outputs/telegram_YYYY-MM-DD.txt를 다시 만든다. engine.daily.simulate_since를
다시 돌리지 않으므로 DB에는 전혀 쓰지 않는다(읽기 전용) — last_processed_date도
그대로다.

쓰는 경우: 보고서 렌더링 코드(notify/report_html.py 등)만 고쳤을 때, 또는
outputs/의 보고서 파일이 다른 모드 dry-run 등으로 덮어써졌을 때 실제 운용
상태를 다시 보여줄 때. 텔레그램 발송은 하지 않는다 — 다시 보내려면 이 스크립트
실행 후 `python -m engine.daily --resend`를 쓴다.

실행:
    python scripts/regenerate_report.py                 last_processed_date 기준일로 재생성 (live)
    python scripts/regenerate_report.py --mode paper     paper 모드
    python scripts/regenerate_report.py --date 2026-09-23  이 DB의 last_processed_date와
                                                          같아야 한다(다르면 경고만 하고 진행)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import state as st  # noqa: E402
from core.indicators import compute_indicators  # noqa: E402
from data import fx  # noqa: E402
from data.earnings import get_earnings_dates  # noqa: E402
from data.fills import load_fills  # noqa: E402
from data.prices import fetch_universe_prices  # noqa: E402
from data.universe import get_universe  # noqa: E402
from engine.daily import OUTPUT_DIR, build_report_summary, load_config, resolve_mode  # noqa: E402
from engine.daily import _write_outputs  # noqa: E402
from notify import briefing, report_html, telegram  # noqa: E402
from store import db  # noqa: E402


def regenerate(cfg: dict, mode: str, date_str: str | None = None) -> dict:
    """state.db를 건드리지 않고 summary를 다시 만들고 보고서·텔레그램 글을 저장한다."""
    conn = db.connect(db.db_path_for_mode(mode))
    try:
        last_processed = db.get_meta(conn, "last_processed_date")
        target_date = date_str or last_processed
        if target_date is None:
            raise SystemExit(f"[regenerate] {mode} DB에 처리된 기준일(last_processed_date)이 없습니다.")
        if last_processed is not None and target_date != last_processed:
            print(
                f"[regenerate] 경고: 지정한 날짜({target_date})가 이 DB의 마지막 처리일"
                f"({last_processed})과 다릅니다 — 저장된 포지션은 {last_processed} 기준입니다."
            )

        positions = db.load_all_positions(conn)
        today_events = db.get_events_for_date(conn, target_date)
    finally:
        conn.close()  # 읽기 전용 — 이후로는 아무것도 쓰지 않는다

    print(f"[regenerate] {mode} DB, 기준일 {target_date} — state.db는 다시 쓰지 않습니다.")
    print("나스닥 100 구성 종목 목록을 가져오는 중...")
    universe = get_universe()
    name_map = universe.set_index("ticker")["name_kr"].to_dict()

    print("일봉 시세를 받는 중... (캐시가 있으면 재사용, DB에는 쓰지 않음)")
    price_result = fetch_universe_prices(universe["ticker"].tolist(), cfg)
    indicator_map = {t: compute_indicators(df, cfg) for t, df in price_result.prices.items()}

    for ticker in indicator_map:  # 처리 이후 새로 추가된 종목 등 방어적 기본값
        positions.setdefault(ticker, st.init_state(ticker, name_map.get(ticker, "")))

    print("실적 발표일을 확인하는 중...")
    earnings_map = get_earnings_dates(list(indicator_map.keys()))
    fills_result = load_fills()

    target_ts = pd.Timestamp(target_date)
    as_of_by_ticker = {t: target_ts for t, df in indicator_map.items() if target_ts in df.index}
    missing = set(indicator_map) - set(as_of_by_ticker)
    if missing:
        print(f"[regenerate] 경고: {len(missing)}개 종목의 시세에 {target_date} 봉이 없어 그 종목은 표시에서 빠집니다.")
    data_gap_tickers = [t for t, gaps in price_result.data_gap.items() if target_ts in {pd.Timestamp(d) for d in gaps}]

    max_concurrent = cfg["risk"]["max_concurrent_positions"]
    run_warnings = list(fills_result.errors)

    fx_result = fx.get_usd_krw_rate(target_date)
    if fx_result.warning:
        run_warnings.append(fx_result.warning)
        print(f"[regenerate] {fx_result.warning}")

    summary = build_report_summary(
        mode, cfg, indicator_map, name_map, earnings_map, positions, today_events,
        as_of_by_ticker, data_gap_tickers, fills_result, run_warnings, max_concurrent,
        fx_result,
    )

    _write_outputs(summary)
    report_path = report_html.render_report(summary, cfg, OUTPUT_DIR)
    summary["report_path"] = report_path

    text = briefing.build_briefing_text(summary, cfg)
    as_of = summary["as_of"]
    as_of_str = as_of.date().isoformat() if as_of is not None else target_date
    OUTPUT_DIR.mkdir(exist_ok=True)
    text_path = OUTPUT_DIR / f"telegram_{as_of_str}.txt"
    text_path.write_text(text, encoding="utf-8")
    summary["telegram_text"] = text
    summary["telegram_path"] = text_path

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="상태를 다시 계산하지 않고 보고서·텔레그램 글을 다시 만든다")
    parser.add_argument("--mode", choices=["live", "paper"], default=None, help="config.yaml의 mode보다 우선")
    parser.add_argument("--date", default=None, help="재생성할 기준일(YYYY-MM-DD). 생략하면 last_processed_date")
    args = parser.parse_args()

    cfg = load_config()
    mode = resolve_mode(cfg, args.mode)

    summary = regenerate(cfg, mode, args.date)

    as_of = summary["as_of"]
    print(f"\n기준일: {as_of.date().isoformat() if as_of is not None else '알수없음'} / 모드: {summary['mode_label']}")
    print(f"오늘 매수 신호 {summary['buy_count']}건, 매도·손절 신호 {len(summary['sell_rows'])}건, 보유 {summary['held_tickers_count']}종목")
    print(f"보고서: {summary['report_path']}")
    print(f"텔레그램 글: {summary['telegram_path']}")
    print("텔레그램 미리보기:")
    print(summary["telegram_text"])
    print(f"[regenerate] {telegram.env_status()}")  # 값은 절대 출력하지 않는다


if __name__ == "__main__":
    main()
