"""오늘 확정 기준일 하루를 처리해 매수·매도·손절 신호와 추천 수량을 뽑고,
HTML 보고서·텔레그램 브리핑까지 만든다.

실행:
    python -m engine.daily              live/paper 모드로 오늘까지 처리, 보고서+텔레그램(토큰 있으면)
    python -m engine.daily --no-send     보고서만 만들고 텔레그램은 보내지 않음
    python -m engine.daily --mode paper  config.yaml의 mode보다 이 값을 우선
    python -m engine.daily --replay      state.db가 비어 있으면 최근 replay.lookback_days
                                          거래일을 하루씩 되돌려 처리해 상태를 만든 뒤 오늘까지 진행
                                          (테스트·백테스트(P5)용. 운용 시작 기본 경로가 아니다)
    python -m engine.daily --dry-run     DB에 쓰지 않고 결과만 출력

## 운용 모드 (P3)

live 모드는 보유를 오직 data/fills.csv의 실제 체결 기록으로만 만든다(가상 체결
없음). paper 모드는 추천대로 체결됐다고 가정하는 모의 운용이고, DB를
data/paper_state.db로 완전히 분리한다.

두 모드 모두 "오늘"부터 시작한다. 각 DB는 이 모드로 처음 실행한 날짜를
`meta` 테이블에 `start_date`로 저장해 두고, 매 실행마다 상태를 항상
init_state에서부터 start_date~오늘까지 다시 계산한다(`simulate_since`).
그래야 사용자가 체결 기록을 며칠 늦게 넣어도(fills.csv에 지난 날짜로 한 줄
추가) 다음 실행에서 그 날짜부터 다시 계산돼 올바른 단계·손절가로 반영된다.
DB에 저장된 상태를 이어받아 증분으로만 갱신하면 이미 지나간 날짜의 체결
기록이 반영될 기회가 없어져 이 성질이 깨진다.

--replay(레거시, P2)는 이 규칙과 무관하게 예전 그대로 남겨 둔다: state.db가
비어 있을 때만 replay.lookback_days거래일을 가상 체결로 되돌려 처리한다.
테스트·백테스트(P5) 전용이고 운용 시작의 기본 경로가 아니다.
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
from notify import briefing, report_html, telegram  # noqa: E402
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
# 표기 규칙: "묶음" 대신 매도 범위를 이렇게 쓴다.
_SELL_RANGE_LABEL = {
    "STOP": "전량",
    "E3": "전량",
    "A1_EXPIRE": "전량",
    "E1": "1차분 매도(11%)",
    "E2": "2차분 매도(22%)",
}
_STAGE_LABEL = {"A1": "1차 · RSI 30 탈출", "A2": "2차 · 골든크로스", "A3": "3차 · 구름 돌파", "B": "추세 재진입"}
_STAGE_TO_BUCKET = {"A1": "b1", "A2": "b2", "A3": "b3", "B": "b9"}
_STATE_TO_KIND = {"정찰": "A1", "확인": "A2", "확정": "A3", "추세보유": "B"}
_STAGE_UNIT = {"정찰": "1", "확인": "2", "확정": "6", "추세보유": "9"}
_MODE_LABEL = {"live": "실전", "paper": "모의"}

# 표기 규칙: "휩소" 대신 "잦은 교차 (횡보)". E1·E2·E3는 "모멘텀 약화·추세 약화·구조 붕괴"와 함께 표기.
_FILTER_REASON_LABEL = {
    "골든크로스 당일 RSI 70 이상": "과열 (RSI 70 이상)",
}


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_mode(cfg: dict, cli_mode: str | None) -> str:
    """운용 모드를 정한다. --mode가 config.yaml의 mode보다 우선한다."""
    mode = cli_mode or cfg.get("mode", "live")
    if mode not in ("live", "paper"):
        raise ValueError(f"알 수 없는 모드: {mode!r} (live 또는 paper만 허용)")
    return mode


def _label_filter_reason(reason: str) -> str:
    """매매 금지 이유를 보고서 표기 규칙에 맞게 바꾼다."""
    if reason in _FILTER_REASON_LABEL:
        return _FILTER_REASON_LABEL[reason]
    if "휩소" in reason:
        return "잦은 교차 (횡보)"
    return reason


def _dates_since_start(df: pd.DataFrame, start_date) -> list:
    """이 종목에 대해 start_date(포함)부터 오늘까지 처리할 날짜 목록을 정한다.

    종목이 start_date 이후 상장했으면(신규 상장) 그 종목의 첫 거래일부터 쓴다.
    start_date가 df 범위보다 미래면(데이터 지연 등) 마지막 날짜 하나만 쓴다.
    """
    mask = df.index >= pd.Timestamp(start_date)
    dates = list(df.index[mask])
    return dates if dates else [df.index[-1]]


def _dates_to_process(df: pd.DataFrame, do_replay: bool, lookback_days: int) -> list:
    """레거시 --replay 경로 전용: 오늘만, 또는 되돌려 보기 포함 날짜 목록."""
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


def simulate_since(
    indicator_map: dict,
    per_ticker_dates: dict,
    states: dict,
    cfg: dict,
    earnings_map: dict,
    gap_dates_by_ticker: dict,
    fills_df: pd.DataFrame,
    virtual_fill: bool,
    max_concurrent: int,
) -> dict:
    """여러 종목의 날짜별 상태 전이를 한 번에 처리한다 (네트워크·DB 없음, 테스트 가능).

    매매 금지 구간·동시 보유 한도는 종목을 가로질러 점수를 비교해야 해서, 날짜마다
    먼저 모든 종목의 새 진입 후보(core.state.preview_new_entry)를 모아 점수 순으로
    추려낸 뒤에야 각 종목을 실제로 처리한다(engine/daily.py 모듈 설명 참고).

    입력: indicator_map({ticker: df}), per_ticker_dates({ticker: [처리할 날짜, ...]}),
         states({ticker: 시작 상태}), cfg, earnings_map({ticker: 실적일 또는 None}),
         gap_dates_by_ticker({ticker: {data_gap 날짜, ...}}), fills_df(실제 체결 기록),
         virtual_fill(True면 추천대로 체결됐다고 가정하는 가상 체결을 쓴다 — paper 모드·
         레거시 --replay용. False면 fills_df의 실제 체결 기록만 쓴다 — live 모드),
         max_concurrent(동시 보유 한도)
    출력: {states, all_events, today_events, warnings, data_gap_tickers,
          as_of_by_ticker, start_by_ticker}
    """
    master_dates = sorted(set().union(*per_ticker_dates.values())) if per_ticker_dates else []

    all_events: list[dict] = []
    today_events: list[dict] = []
    warnings: list[str] = []
    data_gap_tickers: list[str] = []
    as_of_by_ticker: dict = {}
    start_by_ticker: dict = {}

    for date in master_dates:
        active = [t for t in indicator_map if date in per_ticker_dates[t]]

        # ── Pass 1: data_gap 제외 + 오늘 새 진입(A1·B) 후보를 모두 모아 점수로 추린다 ──
        skip_today: set = set()
        candidates = []
        for ticker in active:
            if date in gap_dates_by_ticker.get(ticker, set()):
                skip_today.add(ticker)
                warnings.append(f"{ticker} {date.date()} data_gap - 신호 판정에서 제외")
                if ticker not in data_gap_tickers:
                    data_gap_tickers.append(ticker)
                continue
            cand = st.preview_new_entry(indicator_map[ticker], date, states[ticker], cfg, earnings_map.get(ticker))
            if cand:
                candidates.append({**cand, "ticker": ticker})

        candidates.sort(key=lambda c: c["score"], reverse=True)
        slots = max(max_concurrent - _held_count(states), 0)
        admitted = {c["ticker"] for c in candidates[:slots]}

        # ── Pass 2: 실제 처리 (같은 core 함수를 라이브·paper·되돌려 보기 모두에 쓴다) ──
        for ticker in active:
            if ticker in skip_today:
                continue
            df = indicator_map[ticker]
            events, states[ticker] = st.process_day(
                df, date, states[ticker], cfg, earnings_date=earnings_map.get(ticker), new_entry_allowed=(ticker in admitted)
            )
            for event in events:
                event["ticker"] = ticker

            if virtual_fill:
                for event in list(events):
                    if event["kind"] in _BUY_KINDS:
                        sized = _size_and_price(event, df, states[ticker], cfg)
                        if sized["qty"] > 0:
                            fill = {"unit": event["unit"], "side": "buy", "price": sized["entry_price"], "qty": sized["qty"]}
                            states[ticker] = st.apply_fill(states[ticker], fill, cfg)
                            events.append(
                                {"date": date, "ticker": ticker, "kind": "VIRTUAL_FILL", "unit": event["unit"], **fill}
                            )
            else:
                for fill in fills_for(fills_df, ticker, date):
                    states[ticker] = st.apply_fill(states[ticker], fill, cfg)
                    events.append({"date": date, "ticker": ticker, "kind": "FILL", **fill})

            for event in events:
                event["state_after"] = states[ticker]["state"]
                event["stop_after"] = states[ticker].get("stop")

            all_events.extend(events)
            if date == per_ticker_dates[ticker][-1]:
                today_events.extend(events)
                as_of_by_ticker[ticker] = date
            if date == per_ticker_dates[ticker][0]:
                start_by_ticker[ticker] = date

    return {
        "states": states,
        "all_events": all_events,
        "today_events": today_events,
        "warnings": warnings,
        "data_gap_tickers": data_gap_tickers,
        "as_of_by_ticker": as_of_by_ticker,
        "start_by_ticker": start_by_ticker,
    }


def _unfilled_rows(states: dict, name_map, mode: str) -> list[dict]:
    """live 모드에서 신호는 났지만 체결 기록이 없는 종목을 찾는다 (보유로 잡지 않는다).

    process_day는 새 단계에 들어갈 때 그 묶음을 units[unit]=0으로 자리만 만들어 두고
    (entries[unit]=None), engine이 체결 기록으로 채워야 한다. 기록이 전혀 없으면
    그 자리가 그대로 0으로 남는다 — 이 상태를 "미체결 (기록 없음)"으로 본다.
    """
    if mode != "live":
        return []
    rows = []
    for ticker, state_ in states.items():
        stage = state_["state"]
        unit = _STAGE_UNIT.get(stage)
        if unit is None or state_["units"].get(unit, 0) > 0:
            continue
        rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "단계": _STAGE_LABEL.get(_STATE_TO_KIND.get(stage, ""), stage),
                "내용": "미체결 (기록 없음)",
            }
        )
    return rows


def _compute_funnel(today_events: list[dict], indicator_map: dict, as_of_by_ticker: dict) -> dict:
    """통과 현황(1차 RSI 30 돌파 → 2차 골든크로스 → 3차 구름 4요소 → 4차 매매금지 → 5차 보유한도).

    보고서에는 표시하지 않고 outputs/funnel_YYYY-MM-DD.csv와 events에만 남긴다 (지시문 3번).
    """
    stage1 = stage2 = 0
    for ticker, df in indicator_map.items():
        date = as_of_by_ticker.get(ticker)
        if date is None or date not in df.index:
            continue
        idx = df.index.get_loc(date)
        row = df.loc[date]
        prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")
        if sig.check_a1(prev_rsi, row.get("rsi")):
            stage1 += 1
        if bool(row.get("gc")) if not pd.isna(row.get("gc")) else False:
            stage2 += 1

    stage3 = sum(
        1
        for e in today_events
        if e["kind"] == "A3" or (e["kind"] == "BLOCKED" and e.get("stage") == "A3")
    )
    stage4 = sum(1 for e in today_events if e["kind"] == "BLOCKED" and e.get("blocked_type") == "ban")
    stage5 = sum(1 for e in today_events if e["kind"] == "BLOCKED" and e.get("blocked_type") == "limit")

    return {
        "1차 RSI 30 돌파": stage1,
        "2차 MACD 골든크로스": stage2,
        "3차 일목구름 4요소": stage3,
        "4차 매매금지 필터": stage4,
        "5차 보유한도": stage5,
    }


def _build_buy_row(event: dict, df: pd.DataFrame, states_after: dict, cfg: dict, name_map, earnings_map) -> dict:
    """매수 이벤트 하나를 보고서·텔레그램에 쓸 행 dict로 만든다 (탭별 조건 열 포함)."""
    ticker = event["ticker"]
    date = event["date"]
    row = df.loc[date]
    sized = _size_and_price(event, df, states_after, cfg)
    earnings_date = earnings_map.get(ticker)
    stop_ok = sized["stop_price"] is not None
    stop_pct = (sized["stop_price"] - sized["entry_price"]) / sized["entry_price"] * 100 if stop_ok else None

    idx = df.index.get_loc(date)
    prev_rsi = df.iloc[idx - 1]["rsi"] if idx > 0 else float("nan")

    base = {
        "ticker": ticker,
        "kr": name_map.get(ticker, "") or ticker,
        "stage": event["kind"],
        "bucket": _STAGE_TO_BUCKET[event["kind"]],
        "limit": sized["entry_price"],
        "stop": sized["stop_price"] if stop_ok else None,
        "qty": sized["qty"],
        "stop_pct": round(stop_pct, 1) if stop_pct is not None else None,
        "stop_basis": {"A1": "10일 최저가", "A2": "10일 최저가", "A3": "10일 최저가·구름 하단 중 높은 값", "B": "진입일 10일 최저가"}[
            event["kind"]
        ],
        "earnings": earnings_date.isoformat() if earnings_date else "확인불가",
        "decision": "매수" if stop_ok else "보류",
        "note": "" if stop_ok else "손절가 계산 불가로 수량 미산정 — 매수 보류",
        "score": event.get("score", 0),
        "grade": event.get("grade") or "",
    }

    if event["kind"] == "A1":
        base.update(
            {
                "rsi_prev": round(prev_rsi, 1) if not pd.isna(prev_rsi) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
                "vol_ratio": round(row.get("vol_ratio"), 1) if not pd.isna(row.get("vol_ratio")) else None,
            }
        )
    elif event["kind"] == "A2":
        base.update(
            {
                "a1_date": str(states_after.get("a1_date").date()) if states_after.get("a1_date") is not None else "",
                "macd_norm": round(row.get("macd_norm"), 2) if not pd.isna(row.get("macd_norm")) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
            }
        )
    elif event["kind"] == "A3":
        gap_pct = None
        prev_close = df.iloc[idx - 1]["close"] if idx > 0 else float("nan")
        if not pd.isna(row.get("open")) and not pd.isna(prev_close) and prev_close:
            gap_pct = round((row["open"] / prev_close - 1) * 100, 1)
        base.update(
            {
                "cloud_ok": bool(row.get("close") > row.get("cloud_top")) if not pd.isna(row.get("cloud_top")) else False,
                "future_yang_ok": bool(row.get("future_yang")) if not pd.isna(row.get("future_yang")) else False,
                "chikou_ok": bool(row.get("chikou_ok")) if not pd.isna(row.get("chikou_ok")) else False,
                "momentum_ok": bool(row.get("macd") > row.get("signal") and row.get("rsi") >= 50)
                if not (pd.isna(row.get("macd")) or pd.isna(row.get("signal")) or pd.isna(row.get("rsi")))
                else False,
                "gap_pct": gap_pct,
            }
        )
    else:  # B
        base.update(
            {
                "trend_ok": True,
                "macd_norm": round(row.get("macd_norm"), 2) if not pd.isna(row.get("macd_norm")) else None,
                "rsi_now": round(row.get("rsi"), 1) if not pd.isna(row.get("rsi")) else None,
            }
        )
    return base


def run(cfg: dict, mode: str, do_replay: bool, dry_run: bool) -> dict:
    """엔진을 한 번 실행한다. 결과 요약 dict를 반환한다 (완료 보고·보고서·텔레그램용)."""
    print(f"[모드: {_MODE_LABEL[mode]} ({mode})]")
    print("나스닥 100 구성 종목 목록을 가져오는 중...")
    universe = get_universe()
    name_map_df = universe.set_index("ticker")[["name_kr"]]
    name_map = name_map_df["name_kr"].to_dict()

    print("일봉 시세를 받는 중... (캐시가 있으면 재사용)")
    price_result = fetch_universe_prices(universe["ticker"].tolist(), cfg)
    print(f"  성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")

    print("지표를 계산하는 중...")
    indicator_map = {t: compute_indicators(df, cfg) for t, df in price_result.prices.items()}

    print("실적 발표일을 확인하는 중...")
    earnings_map = get_earnings_dates(list(indicator_map.keys()))

    fills_result = load_fills()
    fills_df = fills_result.df
    fills_errors = fills_result.errors
    for line in fills_errors:
        print(f"  {line}")

    conn = db.connect(db.db_path_for_mode(mode))
    max_concurrent = cfg["risk"]["max_concurrent_positions"]
    run_warnings: list[str] = list(fills_errors)

    gap_dates_by_ticker = {t: {pd.Timestamp(d) for d in price_result.data_gap.get(t, [])} for t in indicator_map}

    if do_replay:
        # ── 레거시 경로(P2): state.db가 비어 있을 때만 가상 체결로 되돌려 본다 ──
        positions = db.load_all_positions(conn)
        replay_needed = len(positions) == 0
        lookback_days = cfg["replay"]["lookback_days"]
        states = {
            t: positions.get(t) or st.init_state(t, name_map.get(t, "")) for t in indicator_map
        }
        per_ticker_dates = {t: _dates_to_process(df, replay_needed, lookback_days) for t, df in indicator_map.items()}
        sim = simulate_since(
            indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker, fills_df,
            virtual_fill=replay_needed, max_concurrent=max_concurrent,
        )
        events_to_persist = sim["all_events"]
    else:
        # ── P3 기본 경로: 이 모드로 처음 실행한 날짜(start_date)부터 오늘까지 매번 다시 계산한다 ──
        start_date_str = db.get_meta(conn, "start_date")
        if start_date_str is None:
            as_of_today = max(df.index[-1] for df in indicator_map.values())
            start_date_str = str(as_of_today.date())
            if not dry_run:
                db.set_meta(conn, "start_date", start_date_str)
        start_date = pd.Timestamp(start_date_str)

        states = {t: st.init_state(t, name_map.get(t, "")) for t in indicator_map}
        per_ticker_dates = {t: _dates_since_start(df, start_date) for t, df in indicator_map.items()}
        sim = simulate_since(
            indicator_map, per_ticker_dates, states, cfg, earnings_map, gap_dates_by_ticker, fills_df,
            virtual_fill=(mode == "paper"), max_concurrent=max_concurrent,
        )
        replay_needed = False
        events_to_persist = sim["today_events"]  # 매 실행마다 전체 이력을 다시 넣지 않는다 (중복 방지)

    states = sim["states"]
    today_events = sim["today_events"]
    run_warnings.extend(sim["warnings"])
    data_gap_tickers = sim["data_gap_tickers"]
    as_of_by_ticker = sim["as_of_by_ticker"]

    positions = states
    for ticker, state_ in states.items():
        if not dry_run:
            db.save_position(conn, state_)

    if not dry_run:
        db.record_events(conn, events_to_persist)

    as_of = max(as_of_by_ticker.values()) if as_of_by_ticker else None

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
    if not dry_run:
        db.record_price_snapshots(conn, run_at, snapshot_rows)

    # ── 오늘 매수 신호: 지정가·손절가·수량·탭별 조건 열을 붙인다 ──────────────
    buy_groups: dict[str, list] = {"b1": [], "b2": [], "b3": [], "b9": []}
    earnings_unknown_count = sum(1 for d in earnings_map.values() if d is None)
    for event in today_events:
        if event["kind"] not in _BUY_KINDS:
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        row = _build_buy_row(event, df, positions[ticker], cfg, name_map, earnings_map)
        buy_groups[row["bucket"]].append(row)
    for bucket in buy_groups:
        buy_groups[bucket].sort(key=lambda r: r["score"], reverse=True)
    buy_count = sum(len(v) for v in buy_groups.values())

    # ── 오늘 걸러진 신호 (매매 금지·동시 보유 한도) ────────────────────────────
    filtered_rows = [
        {
            "티커": event["ticker"],
            "종목명": name_map.get(event["ticker"], "") or event["ticker"],
            "단계": _STAGE_LABEL.get(event["stage"], event["stage"]),
            "유형": "동시보유한도" if event["blocked_type"] == "limit" else "매매금지",
            "사유": ", ".join(_label_filter_reason(r) for r in event["reasons"]),
            "점수": event.get("score", 0),
        }
        for event in today_events
        if event["kind"] == "BLOCKED"
    ]
    filtered_rows.sort(key=lambda r: r["점수"], reverse=True)

    # ── 오늘 매도·손절 신호 ─────────────────────────────────────────────
    sell_rows = []
    for event in today_events:
        if event["kind"] not in _SELL_KINDS:
            continue
        ticker = event["ticker"]
        df = indicator_map[ticker]
        date = event["date"]
        close = float(df.loc[date, "close"]) if date in df.index and not pd.isna(df.loc[date, "close"]) else None
        entry_price = event.get("entry_price")
        pnl_pct = round((close - entry_price) / entry_price * 100, 1) if (close and entry_price) else None
        sell_rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "신호": _SELL_REASON[event["kind"]],
                "매도범위": _SELL_RANGE_LABEL[event["kind"]],
                "수량": event["qty"],
                "평균단가": round(entry_price, 2) if entry_price is not None else None,
                "종가": round(close, 2) if close is not None else None,
                "손익률": pnl_pct,
                "비고": "",
            }
        )

    # ── 경고 (보유 종목만, 매도 아님) + 미체결(live) ────────────────────────
    warn_rows = []
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
        name_kr = name_map.get(ticker, "") or ticker

        if sig.kijun_breach(row.get("close"), row.get("kijun")):
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "기준선 이탈 (매도 아님)"})
        if sig.rsi_overheat_relief(prev_rsi, row.get("rsi")):
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "RSI 과열 해소 (매도 아님)"})
        avg_entry = _average_entry_price(state_)
        if avg_entry is not None and sig.target_reached(avg_entry, row.get("close"), state_.get("stop")):
            warn_rows.append({"티커": ticker, "종목명": name_kr, "내용": "목표 도달 (손익비 2배, 매도 아님)"})

    unfilled_rows = _unfilled_rows(positions, name_map, mode)

    data_status_rows = []
    for ticker in data_gap_tickers:
        data_status_rows.append(
            {"티커": ticker, "종목명": name_map.get(ticker, "") or ticker, "내용": "data_gap - 신호 판정 제외"}
        )
    if earnings_unknown_count:
        data_status_rows.append({"티커": "", "종목명": "", "내용": f"실적일 확인불가 종목 {earnings_unknown_count}개"})
    for line in fills_errors:
        data_status_rows.append({"티커": "", "종목명": "", "내용": line})
    data_status_rows.extend(unfilled_rows)

    # ── 보유 현황 / 내 보유 종목 (같은 데이터) ──────────────────────────────
    today_signal_by_ticker: dict[str, str] = {}
    for row in sell_rows:
        today_signal_by_ticker.setdefault(row["티커"], f"{row['매도범위']} 매도 ({row['신호']})")
    for row in warn_rows:
        if row["티커"]:
            today_signal_by_ticker.setdefault(row["티커"], row["내용"])

    hold_rows = []
    for ticker, state_ in positions.items():
        if not any(q > 0 for q in state_["units"].values()):
            continue
        df = indicator_map[ticker]
        date = as_of_by_ticker.get(ticker) or df.index[-1]
        close = float(df.loc[date, "close"]) if not pd.isna(df.loc[date, "close"]) else None
        avg_entry = _average_entry_price(state_)
        qty = sum(q for q in state_["units"].values() if q > 0)
        pnl_pct = round((close - avg_entry) / avg_entry * 100, 1) if (close and avg_entry) else None
        stop = state_.get("stop")
        stop_dist_pct = round((stop - close) / close * 100, 1) if (stop is not None and close) else None
        hold_rows.append(
            {
                "티커": ticker,
                "종목명": name_map.get(ticker, "") or ticker,
                "단계": state_["state"],
                "수량": qty,
                "평균단가": round(avg_entry, 2) if avg_entry is not None else None,
                "종가": round(close, 2) if close is not None else None,
                "평가금액": round(qty * close, 2) if close is not None else None,
                "손익률": pnl_pct,
                "손절가": round(stop, 2) if stop is not None else None,
                "손절까지": stop_dist_pct,
                "오늘신호": today_signal_by_ticker.get(ticker, ""),
            }
        )
    hold_rows.sort(key=lambda r: r["티커"])

    # ── 관찰 목록: 다음 단계를 기다리는 종목 ────────────────────────────────
    watch_rows = []
    for ticker, state_ in positions.items():
        df = indicator_map[ticker]
        date = as_of_by_ticker.get(ticker)
        if date is None or date not in df.index:
            continue
        if state_["state"] == "정찰" and state_["units"].get("1", 0) > 0 and state_.get("a1_date") in df.index:
            bars_since = df.index.get_loc(date) - df.index.get_loc(state_["a1_date"])
            expiry_days = cfg["assumptions"]["a1_to_a2_expiry_days"]
            remaining = max(expiry_days - bars_since - 1, 0)
            watch_rows.append(
                {
                    "티커": ticker,
                    "종목명": name_map.get(ticker, "") or ticker,
                    "현재단계": "1차 (체결 시)" if state_["units"].get("1", 0) else "1차 (미체결)",
                    "기다리는신호": "2차 · MACD 골든크로스",
                    "남은거래일": remaining,
                }
            )
        elif state_["state"] == "확인" and state_["units"].get("2", 0) > 0:
            watch_rows.append(
                {
                    "티커": ticker,
                    "종목명": name_map.get(ticker, "") or ticker,
                    "현재단계": "2차 확인",
                    "기다리는신호": "3차 · 구름 4요소",
                    "남은거래일": None,
                }
            )

    # ── 단계별 종목 수 ───────────────────────────────────────────────────
    stage_counts = {s: 0 for s in st.STATES}
    for state_ in positions.values():
        stage_counts[state_["state"]] = stage_counts.get(state_["state"], 0) + 1
    held_tickers_count = _held_count(positions)

    # ── 통과 현황(5단계 funnel): 보고서에는 안 쓰고 CSV·events에만 남긴다 ───────
    funnel = _compute_funnel(today_events, indicator_map, as_of_by_ticker)
    if as_of is not None:
        _write_funnel(funnel, as_of)
        if not dry_run:
            db.record_events(conn, [{"date": str(as_of.date()), "ticker": "", "kind": "FUNNEL", **funnel}])

    if not dry_run:
        db.record_run(
            conn,
            run_at=run_at,
            as_of_date=str(as_of.date()) if as_of is not None else "",
            ticker_count=len(indicator_map),
            warning_count=len(run_warnings),
        )
    conn.close()

    summary = {
        "mode": mode,
        "mode_label": _MODE_LABEL[mode],
        "as_of": as_of,
        "replay_needed": replay_needed,
        "buy_groups": buy_groups,
        "buy_count": buy_count,
        "filtered_rows": filtered_rows,
        "sell_rows": sell_rows,
        "warn_rows": warn_rows,
        "data_status_rows": data_status_rows,
        "unfilled_rows": unfilled_rows,
        "hold_rows": hold_rows,
        "watch_rows": watch_rows,
        "stage_counts": stage_counts,
        "held_tickers_count": held_tickers_count,
        "max_concurrent": max_concurrent,
        "warnings": run_warnings,
        "data_gap_tickers": data_gap_tickers,
        "earnings_unknown_count": earnings_unknown_count,
        "all_events": sim["all_events"],
        "funnel": funnel,
    }

    _write_outputs(summary)
    report_path = report_html.render_report(summary, cfg, OUTPUT_DIR)
    summary["report_path"] = report_path
    return summary


def _format_cell(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _to_markdown_table(rows: list[dict]) -> str:
    """의존성(tabulate) 없이 표를 마크다운으로 바꾼다."""
    if not rows:
        return "없음"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row.get(c)) for c in columns) + " |")
    return "\n".join(lines)


def _write_funnel(funnel: dict, as_of) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    as_of_str = as_of.date().isoformat()
    pd.DataFrame([{"단계": k, "건수": v} for k, v in funnel.items()]).to_csv(
        OUTPUT_DIR / f"funnel_{as_of_str}.csv", index=False, encoding="utf-8-sig"
    )


def _write_outputs(summary: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    as_of = summary["as_of"]
    as_of_str = as_of.date().isoformat() if as_of is not None else "알수없음"

    all_buy_rows = [r for rows in summary["buy_groups"].values() for r in rows]
    pd.DataFrame(all_buy_rows).to_csv(OUTPUT_DIR / f"signals_{as_of_str}_buy.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["sell_rows"]).to_csv(OUTPUT_DIR / f"signals_{as_of_str}_sell.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary["filtered_rows"]).to_csv(
        OUTPUT_DIR / f"signals_{as_of_str}_blocked.csv", index=False, encoding="utf-8-sig"
    )

    lines = [f"# 나스닥100 신호 — 기준일 {as_of_str} ({summary['mode_label']})", ""]
    lines.append(f"보유 종목 수: {summary['held_tickers_count']} / 동시 보유 한도: {summary['max_concurrent']}")
    lines.append("")
    lines.append("## 매수 신호")
    lines.append(_to_markdown_table(all_buy_rows))
    lines.append("")
    lines.append("## 걸러진 신호 (매매 금지·한도 초과)")
    lines.append(_to_markdown_table(summary["filtered_rows"]))
    lines.append("")
    lines.append("## 매도·손절 신호")
    lines.append(_to_markdown_table(summary["sell_rows"]))
    lines.append("")
    lines.append("## 경고")
    lines.append(_to_markdown_table(summary["warn_rows"]))
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
    parser.add_argument("--mode", choices=["live", "paper"], default=None, help="config.yaml의 mode보다 우선")
    parser.add_argument("--replay", action="store_true", help="레거시(P2): state.db가 비어 있으면 되돌려 보기로 상태를 만든다")
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 결과만 출력한다")
    parser.add_argument("--no-send", action="store_true", help="보고서만 만들고 텔레그램은 보내지 않는다")
    args = parser.parse_args()

    cfg = load_config()
    mode = resolve_mode(cfg, args.mode)
    summary = run(cfg, mode, do_replay=args.replay, dry_run=args.dry_run)

    as_of = summary["as_of"]
    print(f"\n기준일: {as_of.date().isoformat() if as_of is not None else '알수없음'} / 모드: {summary['mode_label']}")
    print(f"단계별 종목 수: {summary['stage_counts']} (보유 {summary['held_tickers_count']} / 한도 {summary['max_concurrent']})")
    print(f"오늘 매수 신호 {summary['buy_count']}건, 걸러진 신호 {len(summary['filtered_rows'])}건, 매도·손절 신호 {len(summary['sell_rows'])}건")
    print(f"경고 {len(summary['warn_rows'])}건, data_gap 종목 {len(summary['data_gap_tickers'])}개")
    print(f"보고서: {summary['report_path']}")

    if not args.dry_run:
        text = briefing.build_briefing_text(summary, cfg)
        sent_path = telegram.send_briefing(text, summary, cfg, force_no_send=args.no_send)
        print(f"텔레그램 글: {sent_path}")


if __name__ == "__main__":
    main()
