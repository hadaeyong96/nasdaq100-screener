"""오늘 확정 기준일 하루를 처리해 매수·매도·손절 신호와 추천 수량을 뽑는다.

실행:
    python -m engine.daily              오늘 확정 기준일 하루를 처리, state.db에 반영
    python -m engine.daily --replay      state.db가 비어 있으면 최근 replay.lookback_days
                                          거래일을 하루씩 되돌려 처리해 상태를 만든 뒤 오늘까지 진행
    python -m engine.daily --dry-run     DB에 쓰지 않고 결과만 출력

라이브 실행과 되돌려 보기는 core/의 같은 함수(core.state.process_day 등)를 쓴다.
되돌려 보기 중에는 추천 수량대로 체결됐다고 가정한 "가상 보유" 상태로 만든다
(core.state.apply_fill을 그 자리에서 호출해, 실제 체결 기록과 같은 경로로 반영한다).

매매 금지 구간·동시 보유 한도(P2.1 보완 1번): 진입 신호가 매매 금지에 걸리거나
동시 보유 8개 한도를 넘으면 core.state.process_day가 상태를 바꾸지 않고
"BLOCKED" 이벤트만 남긴다. 동시 보유 한도는 종목을 가로질러 점수를 비교해야
해서, 날짜마다 먼저 모든 종목의 새 진입 후보(core.state.preview_new_entry)를
모아 점수 순으로 추려낸 뒤에야 각 종목을 실제로 처리한다 — 라이브 실행과
되돌려 보기가 날짜 단위로 같은 순서를 따른다.

결과는 outputs/signals_YYYY-MM-DD.{md,csv}에 저장한다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import sizing  # noqa: E402
from core import signals as sig  # noqa: E402
from core import state as st  # noqa: E402
from core.indicators import compute_indicators  # noqa: E402
from data.earnings import get_earnings_dates  # noqa: E402
from data.fills import fills_for, load_fills  # noqa: E402
from data.prices import fetch_universe_prices  # noqa: E402
from data.universe import get_universe  # noqa: E402
from store import db  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"

_BUY_KINDS = ("A1", "A2", "A3", "B")
_SELL_KINDS = ("STOP", "A1_EXPIRE", "E3", "E1", "E2")
_SELL_REASON = {
    "STOP": "손절",
    "A1_EXPIRE": "A1 만료",
    "E3": "구조 붕괴(E3)",
    "E1": "모멘텀 약화(E1, 데드크로스)",
    "E2": "추세 약화(E2, RSI 50 이탈)",
}
_STAGE_LABEL = {"A1": "1차 · RSI 30 탈출", "A2": "2차 · 골든크로스", "A3": "3차 · 구름 돌파", "B": "추세 재진입"}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dates_to_process(df: pd.DataFrame, do_replay: bool, lookback_days: int) -> list:
    """이 종목에 대해 처리할 날짜 목록을 정한다 (오늘만, 또는 되돌려 보기 포함)."""
    if not do_replay:
        return [df.index[-1]]
    n = min(lookback_days + 1, len(df))
    return list(df.index[-n:])


def _average_entry_price(state: dict) -> float | None:
    pairs = [(p, state["units"].get(u, 0)) for u, p in state["entries"].items() if p is not None]
    pairs = [(p, q) for p, q in pairs if q > 0]
    if not pairs:
        return None
    total_qty = sum(q for _, q in pairs)
    return sum(p * q for p, q in pairs) / total_qty


def _held_count(states: dict) -> int:
    return sum(1 for s in states.values() if any(q > 0 for q in s["units"].values()))


def _size_and_price(event: dict, df: pd.DataFrame, state_after: dict, cfg: dict) -> dict:
    """진입 이벤트에 지정가·손절가·추천수량을 붙인다 (매매 금지·한도는 core.state가 이미 판정했다).

    입력: event({date,kind,unit,price,ticker,score,grade?}), df(그 종목 지표 DataFrame),
         state_after(이 이벤트가 반영된 뒤의 상태 — stop이 이미 갱신돼 있다), cfg
    출력: {entry_price, stop_price, qty}
    """
    equity = cfg["account"]["equity_usd"]
    entry_price = sig.entry_limit_price(event["price"], cfg)
    stop_price = state_after.get("stop")
    if stop_price is None:
        qty = 0
    else:
        qty = sizing.position_size(event["kind"], entry_price, stop_price, equity, cfg)
        qty = sizing.cap_qty_by_position_limit(qty, entry_price, equity, cfg)
    return {"entry_price": entry_price, "stop_price": stop_price, "qty": qty}


def run(cfg: dict, do_replay: bool, dry_run: bool) -> dict:
    """엔진을 한 번 실행한다. 결과 요약 dict를 반환한다 (완료 보고용)."""
    print("나스닥 100 구성 종목 목록을 가져오는 중...")
    universe = get_universe()
    name_map = universe.set_index("ticker")[["name_kr"]]

    print("일봉 시세를 받는 중... (캐시가 있으면 재사용)")
    price_result = fetch_universe_prices(universe["ticker"].tolist(), cfg)
    print(f"  성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")

    print("지표를 계산하는 중...")
    indicator_map = {t: compute_indicators(df, cfg) for t, df in price_result.prices.items()}

    print("실적 발표일을 확인하는 중...")
    earnings_map = get_earnings_dates(list(indicator_map.keys()))

    fills_df = load_fills()

    # --dry-run은 DB에 쓰지 않을 뿐, 기존 상태는 그대로 읽어서 이어간다.
    conn = db.connect()
    positions = db.load_all_positions(conn)
    replay_needed = do_replay and len(positions) == 0
    lookback_days = cfg["replay"]["lookback_days"]
    max_concurrent = cfg["risk"]["max_concurrent_positions"]

    states: dict[str, dict] = {}
    for ticker in indicator_map:
        name_kr = name_map["name_kr"].get(ticker, "") if ticker in name_map.index else ""
        state_ = positions.get(ticker) or st.init_state(ticker, name_kr)
        state_["name_kr"] = name_kr or state_.get("name_kr", "")
        states[ticker] = state_

    per_ticker_dates = {t: _dates_to_process(df, replay_needed, lookback_days) for t, df in indicator_map.items()}
    gap_dates_by_ticker = {t: {pd.Timestamp(d) for d in price_result.data_gap.get(t, [])} for t in indicator_map}
    master_dates = sorted(set().union(*per_ticker_dates.values())) if per_ticker_dates else []

    all_run_events: list[dict] = []
    today_events: list[dict] = []
    run_warnings: list[str] = []
    data_gap_tickers: list[str] = []
    as_of_by_ticker: dict[str, pd.Timestamp] = {}
    replay_start_by_ticker: dict[str, pd.Timestamp] = {}

    for date in master_dates:
        active = [t for t in indicator_map if date in per_ticker_dates[t]]

        # ── Pass 1: data_gap 제외 + 오늘 새 진입(A1·B) 후보를 모두 모아 점수로 추린다 ──
        skip_today: set[str] = set()
        candidates = []
        for ticker in active:
            if date in gap_dates_by_ticker[ticker]:
                skip_today.add(ticker)
                run_warnings.append(f"{ticker} {date.date()} data_gap - 신호 판정에서 제외")
                if ticker not in data_gap_tickers:
                    data_gap_tickers.append(ticker)
                continue
            cand = st.preview_new_entry(indicator_map[ticker], date, states[ticker], cfg, earnings_map.get(ticker))
            if cand:
                candidates.append({**cand, "ticker": ticker})

        candidates.sort(key=lambda c: c["score"], reverse=True)
        slots = max(max_concurrent - _held_count(states), 0)
        admitted = {c["ticker"] for c in candidates[:slots]}

        # ── Pass 2: 실제 처리 (같은 core 함수를 라이브·되돌려 보기 모두에 쓴다) ──
        for ticker in active:
            if ticker in skip_today:
                continue
            df = indicator_map[ticker]
            events, states[ticker] = st.process_day(
                df, date, states[ticker], cfg, earnings_date=earnings_map.get(ticker), new_entry_allowed=(ticker in admitted)
            )
            for event in events:
                event["ticker"] = ticker

            if replay_needed:
                for event in list(events):
                    if event["kind"] in _BUY_KINDS:
                        sized = _size_and_price(event, df, states[ticker], cfg)
                        if sized["qty"] > 0:
                            fill = {"unit": event["unit"], "side": "buy", "price": sized["entry_price"], "qty": sized["qty"]}
                            states[ticker] = st.apply_fill(states[ticker], fill, cfg)
                            events.append(
                                {"date": date, "ticker": ticker, "kind": "VIRTUAL_FILL", "unit": event["unit"], **fill}
                            )

            for fill in fills_for(fills_df, ticker, date):
                states[ticker] = st.apply_fill(states[ticker], fill, cfg)
                events.append({"date": date, "ticker": ticker, "kind": "FILL", **fill})

            for event in events:
                event["state_after"] = states[ticker]["state"]
                event["stop_after"] = states[ticker].get("stop")

            all_run_events.extend(events)
            if date == per_ticker_dates[ticker][-1]:
                today_events.extend(events)
                as_of_by_ticker[ticker] = date
            if date == per_ticker_dates[ticker][0]:
                replay_start_by_ticker[ticker] = date

    for ticker, state_ in states.items():
        positions[ticker] = state_
        if not dry_run:
            db.save_position(conn, state_)

    if not dry_run:
        db.record_events(conn, all_run_events)

    as_of = max(as_of_by_ticker.values()) if as_of_by_ticker else None
    replay_start = min(replay_start_by_ticker.values()) if replay_needed and replay_start_by_ticker else None

    # ── 재현성 확인용 가격 스냅샷 (P2.1 보완 3번): dry-run에도 남긴다(진단 목적) ──
    run_at = datetime.now().isoformat(timespec="seconds")
    snapshot_rows = []
    for ticker, df in price_result.prices.items():
        last = df.iloc[-1]
        snapshot_rows.append(
            {
                "ticker": ticker,
                "date": str(df.index[-1].date()),
                "close": float(last["close"]) if pd.notna(last["close"]) else None,
                "close_source": last.get("close_source"),
                "meta_time": df.attrs.get("close_meta_time"),
            }
        )
    db.record_price_snapshots(conn, run_at, snapshot_rows)

    # ── 오늘 매수 신호: 지정가·손절가·수량을 붙인다 (등급·점수는 core.state가 이미 계산) ──
    buy_rows = []
    earnings_unknown_count = sum(1 for d in earnings_map.values() if d is None)
    for event in today_events:
        if event["kind"] not in _BUY_KINDS:
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        sized = _size_and_price(event, df, positions[ticker], cfg)
        earnings_date = earnings_map.get(ticker)

        stop_cell = round(sized["stop_price"], 2) if sized["stop_price"] is not None else "미확정(swing_low 데이터 부족)"
        qty_cell = sized["qty"] if sized["stop_price"] is not None else 0
        note = "" if sized["stop_price"] is not None else "손절가 계산 불가로 수량 미산정"

        buy_rows.append(
            {
                "티커": ticker,
                "종목명": name_map["name_kr"].get(ticker, "") or ticker,
                "단계": _STAGE_LABEL.get(event["kind"], event["kind"]),
                "등급": event.get("grade") or "",
                "점수": event.get("score", 0),
                "지정가": round(sized["entry_price"], 2),
                "손절가": stop_cell,
                "추천수량": qty_cell,
                "실적발표일": earnings_date.isoformat() if earnings_date else "확인불가",
                "비고": note,
            }
        )
    buy_rows.sort(key=lambda r: r["점수"], reverse=True)

    # ── 오늘 blocked 신호 (매매 금지·한도 초과로 막힌 진입) ────────────────────
    blocked_rows = [
        {
            "티커": event["ticker"],
            "종목명": name_map["name_kr"].get(event["ticker"], "") or event["ticker"],
            "단계": _STAGE_LABEL.get(event["stage"], event["stage"]),
            "유형": "동시보유한도" if event["blocked_type"] == "limit" else "매매금지",
            "사유": ", ".join(event["reasons"]),
            "점수": event.get("score", 0),
        }
        for event in today_events
        if event["kind"] == "BLOCKED"
    ]
    blocked_rows.sort(key=lambda r: r["점수"], reverse=True)

    # ── 오늘 매도·손절 신호 ─────────────────────────────────────────────
    sell_rows = [
        {
            "티커": event["ticker"],
            "종목명": name_map["name_kr"].get(event["ticker"], "") or event["ticker"],
            "묶음": event["unit"],
            "수량": event["qty"],
            "사유": _SELL_REASON[event["kind"]],
        }
        for event in today_events
        if event["kind"] in _SELL_KINDS
    ]

    # ── 경고 (알림만) ───────────────────────────────────────────────────
    alert_rows = []
    for ticker, df in indicator_map.items():
        state_ = positions[ticker]
        if not any(q > 0 for q in state_["units"].values()):
            continue
        date = as_of_by_ticker.get(ticker)
        if date is None:
            continue
        row = df.loc[date]
        idx = df.index.get_loc(date)
        prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")
        name_kr = name_map["name_kr"].get(ticker, "") or ticker

        if sig.kijun_breach(row.get("close"), row.get("kijun")):
            alert_rows.append({"티커": ticker, "종목명": name_kr, "내용": "기준선 이탈 (매도 아님)"})
        if sig.rsi_overheat_relief(prev_rsi, row.get("rsi")):
            alert_rows.append({"티커": ticker, "종목명": name_kr, "내용": "RSI 과열 해소 (매도 아님)"})
        avg_entry = _average_entry_price(state_)
        if avg_entry is not None and sig.target_reached(avg_entry, row.get("close"), state_.get("stop")):
            alert_rows.append({"티커": ticker, "종목명": name_kr, "내용": "목표 도달 (손익비 2배, 매도 아님)"})

    for ticker in data_gap_tickers:
        alert_rows.append(
            {"티커": ticker, "종목명": name_map["name_kr"].get(ticker, "") or ticker, "내용": "data_gap - 신호 판정 제외"}
        )

    # ── 단계별 종목 수 ───────────────────────────────────────────────────
    stage_counts = {s: 0 for s in st.STATES}
    for state_ in positions.values():
        stage_counts[state_["state"]] = stage_counts.get(state_["state"], 0) + 1
    held_tickers_count = _held_count(positions)

    summary = {
        "as_of": as_of,
        "replay_needed": replay_needed,
        "replay_start": replay_start,
        "buy_rows": buy_rows,
        "blocked_rows": blocked_rows,
        "sell_rows": sell_rows,
        "alert_rows": alert_rows,
        "stage_counts": stage_counts,
        "held_tickers_count": held_tickers_count,
        "max_concurrent": max_concurrent,
        "warnings": run_warnings,
        "data_gap_tickers": data_gap_tickers,
        "earnings_unknown_count": earnings_unknown_count,
        "all_events": all_run_events,
    }

    if not dry_run:
        db.record_run(
            conn,
            run_at=run_at,
            as_of_date=str(as_of.date()) if as_of is not None else "",
            ticker_count=len(indicator_map),
            warning_count=len(run_warnings),
        )
    conn.close()

    _write_outputs(summary)
    return summary


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _to_markdown_table(rows: list[dict]) -> str:
    """의존성(tabulate) 없이 표를 마크다운으로 바꾼다 (scripts/p1_report.py와 같은 방식)."""
    if not rows:
        return "없음"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _write_outputs(summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    as_of = summary["as_of"]
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"

    pd.DataFrame(summary["buy_rows"]).to_csv(OUTPUT_DIR / f"signals_{as_of_str}_buy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["sell_rows"]).to_csv(OUTPUT_DIR / f"signals_{as_of_str}_sell.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["blocked_rows"]).to_csv(
        OUTPUT_DIR / f"signals_{as_of_str}_blocked.csv", index=False, encoding="utf-8-sig"
    )

    lines = [f"# 나스닥100 신호 — 기준일 {as_of_str}", ""]
    lines.append(f"보유 종목 수: {summary['held_tickers_count']} / 동시 보유 한도: {summary['max_concurrent']}")
    lines.append("")
    lines.append("## 매수 신호 (점수 순)")
    lines.append(_to_markdown_table(summary["buy_rows"]))
    lines.append("")
    lines.append("## blocked 신호 (매매 금지·한도 초과)")
    lines.append(_to_markdown_table(summary["blocked_rows"]))
    lines.append("")
    lines.append("## 매도·손절 신호")
    lines.append(_to_markdown_table(summary["sell_rows"]))
    lines.append("")
    lines.append("## 경고")
    lines.append(_to_markdown_table(summary["alert_rows"]))
    lines.append("")
    lines.append("## 단계별 종목 수")
    for stage, count in summary["stage_counts"].items():
        lines.append(f"- {stage}: {count}")
    lines.append("")
    lines.append(f"data_gap 종목: {', '.join(summary['data_gap_tickers']) or '없음'}")
    lines.append(f"실적일 확인불가 종목 수: {summary['earnings_unknown_count']}")
    (OUTPUT_DIR / f"signals_{as_of_str}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="나스닥 100 MACD 스크리너 — 일일 신호 판정")
    parser.add_argument("--replay", action="store_true", help="state.db가 비어 있으면 되돌려 보기로 상태를 만든다")
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 결과만 출력한다")
    args = parser.parse_args()

    cfg = load_config()
    summary = run(cfg, do_replay=args.replay, dry_run=args.dry_run)

    as_of = summary["as_of"]
    print(f"\n기준일: {as_of.date().isoformat() if as_of is not None else '알수없음'}")
    if summary["replay_needed"] and summary["replay_start"] is not None:
        print(f"되돌려 보기 기간: {summary['replay_start'].date()} ~ {as_of.date()}")
    print(f"단계별 종목 수: {summary['stage_counts']} (보유 {summary['held_tickers_count']} / 한도 {summary['max_concurrent']})")
    print(f"오늘 매수 신호 {len(summary['buy_rows'])}건, blocked {len(summary['blocked_rows'])}건, 매도·손절 신호 {len(summary['sell_rows'])}건")
    print(f"경고 {len(summary['warnings'])}건, data_gap 종목 {len(summary['data_gap_tickers'])}개")
    print(f"실적일 확인불가 종목 수: {summary['earnings_unknown_count']}")


if __name__ == "__main__":
    main()
