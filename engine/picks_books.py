"""추천 따라하기 장부 2개 — 매일 실행·DB 저장·시트 복구 (docs/design/virtual_books.md).

engine/virtual_books.py의 run 안에서 기존 5개 장부를 계산한 뒤 불린다. 장부마다 따로 try로 감싸
한 장부가 실패해도 다른 장부·기존 브리핑은 그대로 나간다(오류는 errors로 모아 따로 알림).

- 「추천 종목 따라하기」: 지금 브리핑과 같은 신호(core.state.ICHIMOKU_RULES)
- 「볼린저·RSI·MACD 따라하기」: 일목 자리만 볼린저(core.bollinger_signals.BOLLINGER_RULES), 브리핑에 섞지 않는다

장부는 하루씩 이어 간다: DB(picks_state)에 저장된 마지막 처리일 다음 거래일부터 기준일까지
core.picks_book.process_day를 부르고, 장부 상태·일별·매매·신호 행을 쌓는다. 같은 날 다시 실행하면
처리할 날이 없어 아무것도 바뀌지 않는다. DB를 잃었으면(시작일은 고정돼 있는데 상태가 없음) 새로
시작하지 않고 오류로 알린다 — `python -m engine.virtual_books --recover-picks`로 시트에서 다시 만든다.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from core import picks_book as pb
from core import state as st
from core.bollinger_signals import BOLLINGER_RULES, add_bollinger_columns
from data import virtual_books_data as vbdata
from store import virtual_books as vbstore

RULES = {"ichimoku": st.ICHIMOKU_RULES, "bollinger": BOLLINGER_RULES}


class PicksStateMissingError(RuntimeError):
    """시작일은 정해졌는데 DB에 장부 상태가 없을 때 — 새로 시작하지 않고 복구를 요구한다."""


def default_market(cfg: dict, extra_tickers: list[str]) -> dict:
    """나스닥100 명단 + 보유·체결 대기 종목의 지표·실적일 (매일 실행이 먼저 받아 둔 캐시를 그대로 쓴다).

    출력: {"universe": [티커], "frames": {티커: 지표 DataFrame}, "earnings": {티커: 날짜}, "warnings": [str]}
    예외: 보유·체결 대기 종목 가격을 못 받으면 VirtualBooksDataError (명단 종목 실패는 경고만)
    """
    from core.indicators import compute_indicators
    from data.earnings import get_earnings_dates
    from data.prices import fetch_universe_prices
    from data.universe import get_universe

    universe = get_universe()["ticker"].tolist()
    res = fetch_universe_prices(list(dict.fromkeys(universe + extra_tickers)), cfg, sleep_sec=0.0)
    bad = [t for t in extra_tickers if t in res.failed]
    if bad:
        raise vbdata.VirtualBooksDataError(f"보유 종목 가격 받기 실패: {', '.join(f'{t}({res.failed[t]})' for t in bad)}")
    frames = {t: compute_indicators(df, cfg) for t, df in res.prices.items()}
    return {"universe": universe, "frames": frames, "earnings": get_earnings_dates(list(frames)),
            "warnings": [f"{t} 가격 받기 실패: {e}" for t, e in res.failed.items()]}


def params_for(cfg: dict, name: str) -> pb.PicksParams:
    vb, pk = cfg["virtual_books"], cfg["virtual_books"]["picks"]
    return pb.PicksParams(
        name=name, slots=int(pk["slots"]), commission_pct=float(vb["commission_pct"]),
        slippage_pct=float(vb["slippage_pct"]), tax_rate_pct=float(vb["capital_gains_rate_pct"]),
        dividend_withholding_pct=float(vb["dividend_withholding_pct"]), cash_ticker=str(vb["defensive_ticker"]),
        progress_on_skipped_adds=bool(pk["progress_on_skipped_adds"]),
    )


def _frames_for(market: dict, rules_name: str, bb_cfg: dict) -> dict:
    if rules_name != "bollinger":
        return market["frames"]
    cache = market.setdefault("_bb_frames", {})
    if not cache:
        cache.update({t: add_bollinger_columns(df, bb_cfg) for t, df in market["frames"].items()})
    return cache


def _actions_for_day(actions: dict, d: pd.Timestamp) -> tuple[dict, dict]:
    divs, splits = {}, {}
    for t, a in actions.items():
        if d in a.index:
            if a.at[d, "dividend"] > 0:
                divs[t] = float(a.at[d, "dividend"])
            if a.at[d, "split"] > 0:
                splits[t] = float(a.at[d, "split"])
    return divs, splits


def _last_or(daily: pd.DataFrame, col: str, default: float) -> float:
    """마지막 줄 값 (시트에서 복구한 줄처럼 비어 있으면 default)."""
    v = daily[col].iat[-1] if col in daily.columns else None
    return default if v is None or pd.isna(v) else float(v)


def summarize(book: dict, daily: pd.DataFrame, trades: list[dict], min_days_for_annualized: int) -> dict:
    """전략요약 한 줄 (기존 장부 요약과 같은 키 + 완료 거래 수·승률·평균 보유일·SGOV 보유액)."""
    start, end = daily.index[0], daily.index[-1]
    value = float(daily["value"].iat[-1])
    days = (end - start).days
    cagr = ((value / book["initial_usd"]) ** (365.25 / days) - 1.0) * 100.0 if days >= min_days_for_annualized and days > 0 else None
    return {
        "book": book["name"], "start": start, "end": end, "initial_usd": book["initial_usd"], "value": value,
        "cum_return_pct": float(daily["cum_return_pct"].iat[-1]), "cagr_pct": cagr,
        "mdd_pct": float(daily["drawdown_pct"].min()), "mdd_date": daily["drawdown_pct"].idxmin(), "switches": "-",
        "holding": daily["holding"].iat[-1], "tax_cum": float(book["tax_cum"]),
        "value_after_tax_liq": _last_or(daily, "value_after_tax_liq", value),
        "sgov_value": _last_or(daily, "sgov_value", float("nan")), **pb.trade_stats(trades),
    }


def _db_trades(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        t = dict(r)
        for k in ("signal_date", "fill_date", "entry_date"):
            t[k] = pd.Timestamp(t[k]) if t[k] else None
        t["in_briefing"] = None if t["in_briefing"] is None else bool(t["in_briefing"])
        out.append(t)
    return out


def _db_daily(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def run_picks(cfg: dict, conn, ctx: dict, *, briefing: dict | None, dry_run: bool, market_fn=None,
              actions_fn=None, fx_fn=None, year_end_fn=None) -> dict:
    """추천 따라하기 장부들을 기준일까지 이어 간다.

    입력: cfg, conn(가상 장부 DB), ctx(engine.virtual_books.compute 결과 — QQQ 거래일·SGOV 종가·분배금·기준일),
         briefing(오늘 브리핑 추천 {"buy": [티커], "sell": [티커]} — 없으면 None), dry_run(쓰지 않음),
         market_fn(cfg, extra_tickers)·actions_fn(tickers, start)·fx_fn(date)·year_end_fn(첫 해, 끝 해)(테스트 주입)
    출력: {"extra": [시트·보고서용 장부 묶음], "errors": [str], "warnings": [str], "today": {장부: 오늘 신호 행}}
    """
    vb = cfg["virtual_books"]
    pk = vb.get("picks") or {}
    out = {"extra": [], "errors": [], "warnings": [], "today": {}}
    if not pk.get("enabled", False):  # 설정이 없거나 꺼져 있으면 기존 5개 장부만
        return out
    base = ctx["base_date"]
    base_str = base.date().isoformat()
    if briefing is not None and not dry_run:
        vbstore.save_rec_log(conn, base_str, list(briefing.get("buy", [])), list(briefing.get("sell", [])))
    recs = vbstore.load_rec_log(conn)
    if briefing is not None:
        recs[base_str] = {"buy": set(briefing.get("buy", [])), "sell": set(briefing.get("sell", []))}

    closes_all, divs_all = ctx["closes"], ctx["divs"]
    cash_t = str(vb["defensive_ticker"])
    days_index = closes_all.index
    market = None
    for spec in pk["books"]:
        name = spec["name"]
        try:
            book = vbstore.load_picks_book(conn, name)
            start_cfg = pk.get("start_date")
            if book is None:
                if start_cfg and pd.Timestamp(start_cfg) < base:
                    raise PicksStateMissingError(
                        f"{name}: 시작일 {start_cfg}이 고정돼 있는데 DB에 장부 상태가 없습니다 — 새로 시작하지 않습니다. "
                        "`python -m engine.virtual_books --recover-picks`로 시트에서 다시 만드세요")
                start = pd.Timestamp(start_cfg) if start_cfg else base
                if start > base:
                    out["warnings"].append(f"{name}: 시작일 {start.date()} 전이라 계산하지 않았습니다")
                    continue
                start_fx, fx_date = _picks_fx(conn, pk, start, fx_fn)
                book = pb.new_book(name, start, float(vb["start_krw"]) / start_fx)
                book["start_fx"], book["fx_date"] = start_fx, fx_date
                days = list(days_index[(days_index >= start) & (days_index <= base)])
            else:
                last = pd.Timestamp(book["last_date"])
                days = list(days_index[(days_index > last) & (days_index <= base)])
            extra = sorted(set(book["positions"]) | {o["ticker"] for o in book["pending_buys"] + book["pending_sells"]})
            if days:
                if market is None:
                    market = (market_fn or default_market)(cfg, extra)
                    out["warnings"].extend(market.get("warnings", []))
                missing = [t for t in extra if t not in market["frames"]]
                if missing:  # 다른 장부가 먼저 market을 받아 이 장부의 보유 종목이 빠졌으면 더 받는다
                    more = (market_fn or default_market)(cfg, missing)
                    market["frames"].update({t: more["frames"][t] for t in missing if t in more["frames"]})
                    market.pop("_bb_frames", None)
                frames = _frames_for(market, spec["rules"], pk["bollinger"])
                actions = (actions_fn or vbdata.fetch_actions)(list(book["positions"]) + [o["ticker"] for o in book["pending_buys"]],
                                                                (days[0] - timedelta(days=10)).date()) if extra else {}
                year_ends = (year_end_fn or vbdata.year_end_trading_days)(days[0].year, base.year)
            new_daily, new_trades, new_signals = [], [], []
            for d in days:
                closes = {t: float(df.at[d, "close"]) for t, df in frames.items() if d in df.index and not pd.isna(df.at[d, "close"])}
                divs, splits = _actions_for_day(actions, d)
                if d in divs_all.index and divs_all.at[d, cash_t] > 0:
                    divs[cash_t] = float(divs_all.at[d, cash_t])
                scan = list(dict.fromkeys(market["universe"] + sorted(book["positions"]) + [o["ticker"] for o in book["pending_sells"]]))
                book, row, trades, signals = pb.process_day(
                    book, d, frames=frames, closes=closes, sgov_close=float(closes_all.at[d, cash_t]), dividends=divs,
                    splits=splits, scan_tickers=scan, earnings_map=market["earnings"], rules=RULES[spec["rules"]],
                    params=params_for(cfg, name), cfg=cfg, year_end=d in year_ends,
                    briefing=recs.get(d.date().isoformat()),
                )
                new_daily.append(row)
                new_trades.extend(trades)
                new_signals.extend(signals)
            if not dry_run and days:
                vbstore.save_picks_day(conn, book, new_daily, new_trades, new_signals)
            daily_rows = vbstore.picks_rows(conn, "picks_daily", name)
            trade_rows = _db_trades(vbstore.picks_rows(conn, "picks_trades", name))
            if dry_run:  # 저장하지 않았으므로 이번 계산분을 덧붙여 보여 준다
                daily_rows = daily_rows + new_daily
                trade_rows = trade_rows + new_trades
            daily = _db_daily(daily_rows)
            if daily.empty:
                continue
            out["today"][name] = [s for s in new_signals if pd.Timestamp(s["date"]) == base]
            out["extra"].append({
                "name": name, "start": pd.Timestamp(book["start_date"]), "daily": daily, "trades": trade_rows,
                "summary": summarize(book, daily, trade_rows, int(vb["min_days_for_annualized"])),
                "meta": {"start_krw": float(vb["start_krw"]), "start_fx": book.get("start_fx"),
                         "fx_date": book.get("fx_date", ""), "initial_usd": book["initial_usd"]},
                "positions": book["positions"], "pending_buys": book["pending_buys"], "pending_sells": book["pending_sells"],
            })
        except Exception as exc:  # 장부 하나의 오류가 다른 장부·브리핑을 막지 않게
            out["errors"].append(f"{name} 오류: {type(exc).__name__}: {exc}")
    return out


def _picks_fx(conn, pk: dict, start: pd.Timestamp, fx_fn) -> tuple[float, str]:
    """추천 따라하기 장부 시작 환율: 설정값 → 시작일 원/달러 종가."""
    if pk.get("start_fx"):
        return float(pk["start_fx"]), "설정값"
    if fx_fn is None:
        from engine.virtual_books import _default_fx

        fx_fn = _default_fx
    rate, fx_date, warning = fx_fn(start.date())
    if rate is None:
        raise RuntimeError(f"추천 따라하기 시작일 {start.date()} 원/달러 환율을 구하지 못했습니다 ({warning})")
    return float(rate), str(fx_date or "")


# ── DB 복구 보조 경로 (DB를 잃었을 때 시트 세 탭에서 장부를 다시 만든다) ──────────────────────

_LABEL_KIND = {v: k for k, v in pb.KIND_LABEL.items()}


def _num(v):
    try:
        return float(str(v).replace(",", "")) if str(v).strip() != "" else None
    except ValueError:
        return None


def _kind_from_reason(reason: str) -> str:
    if "세금 납부" in reason:
        return "TAX"
    for label, kind in sorted(_LABEL_KIND.items(), key=lambda kv: -len(kv[0])):
        if reason.startswith(label):
            return kind
    return ""


def recover_from_tables(cfg: dict, name: str, tables: dict[str, list[list]], frames: dict, sgov_close: pd.Series) -> dict:
    """시트 세 탭 값으로 장부 하나를 다시 만든다 (근사 — 문서 docs/design/virtual_books.md "DB 복구" 참고).

    - 보유: 매매기록의 매수에서 신호 매도·세금 납부 매도를 뺀 나머지 (수량·취득가·진입일·신호 종류)
    - 현금+SGOV: 전략비교_일일 마지막 날 총자산 − 보유 종목 평가액 → SGOV 정수 주식(나머지 현금), 취득가 = 그날 평가액
    - 신호 상태: 보유 종목은 진입 단계(A1→정찰, B→추세보유)·진입일 손절가(그날 swing_low)로, 판 종목은 재진입 대기만
      (2·3차 진행 단계와 체결 대기 주문은 되살리지 못한다)
    - 올해 실현 손익: 매매기록 실현손익 합(SGOV 손익은 빠짐), 납부 세금 누적: 전략요약
    출력: {"book": 장부 상태, "daily": [일별 행], "trades": [매매 행]}
    예외: 시트에 이 장부 기록이 없으면 ValueError
    """
    daily_v, trades_v, summary_v = tables["전략비교_일일"], tables["매매기록"], tables["전략요약"]
    head = daily_v[0]
    try:
        iv, ic, idd = head.index(f"{name} 총자산($)"), head.index(f"{name} 누적(%)"), head.index(f"{name} 낙폭(%)")
    except ValueError as exc:
        raise ValueError(f"{name}: 전략비교_일일에 이 장부 열이 없습니다") from exc
    daily = []
    for r in daily_v[1:]:
        v = _num(r[iv]) if len(r) > iv else None
        if v is None:
            continue
        daily.append({"book": name, "date": pd.Timestamp(r[0]), "holding": "", "value": v,
                      "cum_return_pct": _num(r[ic]), "drawdown_pct": _num(r[idd])})
    if not daily:
        raise ValueError(f"{name}: 전략비교_일일에 기록이 없습니다")
    last = daily[-1]["date"]

    srow = next((r for r in summary_v[1:] if r and r[0] == name), None)
    if srow is None:
        raise ValueError(f"{name}: 전략요약에 이 장부 줄이 없습니다")
    sh = summary_v[0]
    start = pd.Timestamp(srow[sh.index("시작일")])
    initial = _num(srow[sh.index("시작 자금($)")])
    tax_cum = _num(srow[sh.index("납부 세금 누적($)")]) or 0.0
    fx_row = next((r for r in summary_v if r and r[0] == "config 고정값(picks)"), None)
    start_fx = float(fx_row[1].split("start_fx:")[1].strip()) if fx_row else None

    th = trades_v[0]
    col = {k: th.index(k) for k in th}
    trades, positions, last_sell = [], {}, {}
    for r in trades_v[1:]:
        if not r or r[col["장부"]] != name:
            continue
        reason, side, t = r[col["사유"]], r[col["구분"]], r[col["종목"]]
        kind = _kind_from_reason(reason)
        yn = r[col["브리핑 추천에도 있었음"]] if "브리핑 추천에도 있었음" in col else ""
        tr = {"book": name, "signal_date": pd.Timestamp(r[col["신호일"]]) if r[col["신호일"]] else None,
              "fill_date": pd.Timestamp(r[col["체결일"]]), "side": side, "ticker": t, "qty": int(_num(r[col["수량"]]) or 0),
              "price": _num(r[col["가격($)"]]) or 0.0, "commission": _num(r[col["수수료($)"]]) or 0.0,
              "realized_pnl": _num(r[col["실현손익($)"]]), "reason": reason, "kind": kind,
              "in_briefing": None if yn == "" else yn == "예", "entry_date": None}
        tr["amount"] = tr["qty"] * tr["price"] + (tr["commission"] if side == pb.SIDE_BUY else -tr["commission"])
        if side == pb.SIDE_BUY:
            positions[t] = {"qty": tr["qty"], "basis": tr["amount"], "entry_date": tr["fill_date"],
                            "signal_date": tr["signal_date"], "kind": kind, "last_close": tr["price"]}
        elif side == pb.SIDE_SELL and t in positions:
            p = positions[t]
            tr["entry_date"] = p["entry_date"]
            if kind == "TAX":
                p["basis"] -= p["basis"] * tr["qty"] / p["qty"]
                p["qty"] -= tr["qty"]
                if p["qty"] <= 0:
                    positions.pop(t)
            else:
                positions.pop(t)
                last_sell[t] = tr["signal_date"]
        trades.append(tr)

    stock_value = 0.0
    for t, p in positions.items():
        df = frames.get(t)
        if df is not None and last in df.index:
            p["last_close"] = float(df.at[last, "close"])
        stock_value += p["qty"] * p["last_close"]
    sg = float(sgov_close.at[last])
    rest = daily[-1]["value"] - stock_value
    sgov_qty = max(int(rest // sg), 0)

    book = pb.new_book(name, start, initial)
    book.update({
        "cash": rest - sgov_qty * sg, "sgov_qty": sgov_qty, "sgov_basis": sgov_qty * sg, "positions": positions,
        "realized_ytd": sum((tr["realized_pnl"] or 0.0) for tr in trades if tr["fill_date"].year == last.year
                            and tr["side"] == pb.SIDE_SELL),
        "tax_cum": tax_cum, "peak": max(r["value"] for r in daily), "last_date": last,
        "start_fx": start_fx, "fx_date": "복구",
    })
    for t, p in positions.items():
        sst = st.init_state(t)
        unit = "1" if p["kind"] == "A1" else "9"
        sst.update({"state": "정찰" if unit == "1" else "추세보유", "units": {unit: p["qty"]},
                    "entries": {unit: p["basis"] / p["qty"]}})
        df = frames.get(t)
        sd = p["signal_date"]
        if df is not None and sd in df.index and not pd.isna(df.at[sd, "swing_low"]):
            sst["stop"] = float(df.at[sd, "swing_low"])
        if unit == "1":
            sst["a1_date"] = sd
        else:
            sst["b_entry_date"], sst["b_total_qty"] = sd, p["qty"]
        book["signal_states"][t] = sst
    for t, sd in last_sell.items():
        if t not in positions and sd is not None:
            book["signal_states"][t] = st.force_flat(st.init_state(t), sd, cfg)
    return {"book": book, "daily": daily, "trades": trades}
