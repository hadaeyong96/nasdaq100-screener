"""가상 장부 8번 「가치평가 따라하기」 실행 — DB 상태를 읽어 기준일까지 하루씩 이어 간다 (core.value_book.process_day).

- 시작일·시작 환율: config virtual_books.value_score.start_date(고정) / start_fx(시작 뒤 시트 "config 고정값(value_score)"대로 고정).
  start_fx가 고정돼 있는데 DB에 장부가 없으면 새로 시작하지 않는다(DB를 잃은 것 — 다른 상태 장부와 같은 보호).
- 점수: 월말 점수표를 처음 필요할 때 data.value_live.compute_scores로 계산해 DB(value_scores)에 남기고 다시 쓴다.
  필요한 월말 = 시작일 전 마지막 월말(첫 선정), 분기 말(수시 매도 판정), 11월 말(12월 실행일).
- 가격: 보유 종목은 하나라도 못 받으면 그날 처리를 멈춘다(다음 실행에서 이어 감 — 가격 끝 판정이 틀리지 않게).
  첫 선정·12월 후보는 못 받으면 경고만 남기고 그 후보를 건너뛴다.
- 오류가 나도 다른 장부·브리핑은 그대로(engine.virtual_books가 따로 잡는다). dry_run이면 DB에 쓰지 않는다.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from core import tax_harvest as th
from core import value_book as vbk
from data import virtual_books_data as vbdata
from data.market_calendar import trading_days_between
from store import virtual_books as vbstore

KIND = "value_score"


class ValueStateMissingError(RuntimeError):
    """시작 환율이 고정돼 있는데 DB에 장부 상태가 없을 때 (새로 시작하지 않는다)."""


def value_params(cfg: dict) -> vbk.ValueParams:
    vb, c = cfg["virtual_books"], cfg["virtual_books"]["value_score"]
    return vbk.ValueParams(
        name=str(c["name"]), slots=int(c["slots"]), min_pick=float(c["min_pick"]), min_keep=float(c["min_keep"]),
        min_swap=float(c["min_swap"]), swap_gap=float(c["swap_gap"]), sector_cap=int(c["sector_cap"]),
        start_krw=float(vb["start_krw"]), commission_pct=float(c["commission_pct"]), slippage_pct=float(c["slippage_pct"]),
        fx_spread_pct=float(c["fx_spread_pct"]), tax_rate_pct=float(c["tax_rate_pct"]), deduction_krw=float(c["deduction_krw"]),
        dividend_withholding_pct=float(vb["dividend_withholding_pct"]))


def calendar(first_year: int, last_year: int) -> dict:
    """{month_ends, quarter_ends, dec_exec{해: 날}, nov_end{해: 날}, may_first, year_ends} — NYSE 달력(미리 정해진 일정)."""
    me = vbdata.month_end_trading_days(first_year, last_year)
    out = {"month_ends": me, "quarter_ends": {d for d in me if d.month in (3, 6, 9, 12)}, "dec_exec": {}, "nov_end": {},
           "may_first": set(), "year_ends": vbdata.year_end_trading_days(first_year, last_year)}
    for y in range(first_year, last_year + 1):
        days = trading_days_between(date(y, 1, 1), date(y, 12, 31))
        out["dec_exec"][y] = pd.Timestamp(th.deadline(days, y))
        out["nov_end"][y] = max(d for d in me if d.year == y and d.month == 11)
        may = [d for d in days if d.month == 5]
        if may:
            out["may_first"].add(pd.Timestamp(may[0]))
    return out


def last_month_end_before(d: pd.Timestamp) -> pd.Timestamp:
    me = vbdata.month_end_trading_days(d.year - 1, d.year)
    return max(x for x in me if x < d)


def default_history(t: str, start: date) -> pd.DataFrame:
    """yfinance 일봉(auto_adjust=False Close, 분배금·분할). 확정 봉만은 기준일로 자른다(호출부)."""
    return vbdata._yf_history(t, start)


def fetch_frames(tickers: list[str], start: date, base: pd.Timestamp, history_fn=None) -> tuple[dict, list[str]]:
    """{티커: DataFrame(close, dividend, split)}, 실패 목록."""
    history_fn = history_fn or default_history
    out, failed = {}, []
    for t in sorted(set(tickers)):
        try:
            raw = history_fn(t, start)
            if raw is None or raw.empty or "Close" not in raw.columns:
                raise ValueError("가격 없음")
            idx = pd.DatetimeIndex(raw.index)
            raw = raw.set_axis((idx.tz_localize(None) if idx.tz is not None else idx).normalize())
            raw = raw[(raw.index <= base) & ~raw.index.duplicated(keep="last")]
            out[t] = pd.DataFrame({"close": raw["Close"].astype(float),
                                   "dividend": raw.get("Dividends", pd.Series(0.0, index=raw.index)).astype(float).fillna(0.0),
                                   "split": raw.get("Stock Splits", pd.Series(0.0, index=raw.index)).astype(float).fillna(0.0)})
        except Exception as exc:  # noqa: BLE001 — 모아서 보고
            failed.append(f"{t}: {type(exc).__name__}: {str(exc)[:60]}")
    return out, failed


def run_value_score(cfg: dict, conn, ctx: dict, *, dry_run: bool, history_fn=None, fx_range_fn=None, score_fn=None) -> dict:
    """8번 장부를 기준일까지 이어 간다. 출력: {"extra": [시트·편지용 장부 묶음 0~1개], "errors", "warnings"}."""
    from engine import picks_books as pb

    vbc = cfg["virtual_books"]
    conf = vbc.get("value_score") or {}
    out = {"extra": [], "errors": [], "warnings": []}
    if not conf.get("enabled", False):
        return out
    p = value_params(cfg)
    name = p.name
    try:
        base = pd.Timestamp(ctx["base_date"]).normalize()
        days_index = ctx["closes"].index
        start = pd.Timestamp(conf["start_date"])
        book = vbstore.load_picks_book(conn, name)
        mismatch = pb._start_mismatch_warning(name, conf, book)
        if mismatch:
            out["warnings"].append(mismatch)
        if book is None and conf.get("start_fx"):
            raise ValueStateMissingError(f"{name}: 시작 환율이 고정돼 있는데 DB에 장부 상태가 없습니다 — 새로 시작하지 않습니다")
        if book is None:
            book = vbk.new_book(name, start)
        last = book.get("last_date")
        days = [d for d in days_index if start <= d <= base and (last is None or d > pd.Timestamp(last))]
        cal = calendar(start.year, base.year + 1)
        boards: dict = {}

        def board_for(me: pd.Timestamp, held: list[str]) -> dict:
            key = me.date().isoformat()
            if key not in boards:
                df = vbstore.load_value_scores(conn, name, key)
                if df is None:
                    sic = vbstore.load_sic_cache(conn)
                    if score_fn is not None:
                        df, warns = score_fn(me, held)
                    else:
                        from data import value_live

                        df, warns = value_live.compute_scores(me, held, sic_cache=sic)
                    if warns:
                        out["warnings"].append(f"{name}: {key} 점수 못 낸 종목 {len(warns)}개 — " + "; ".join(warns[:5]))
                    if not dry_run:
                        vbstore.save_value_scores(conn, name, key, df)
                        vbstore.save_sic_cache(conn, sic)
                boards[key] = (df, vbk.board(df))
            return boards[key][1]

        new_daily, new_trades = [], []
        if days:
            held = sorted(book["positions"])
            need: dict = {}
            if book["status"] == vbk.WAITING:
                need[days[0]] = ("start", last_month_end_before(days[0]))
            for d in days:
                if d in cal["quarter_ends"]:
                    need.setdefault(d, ("quarter", d))
                if d == cal["dec_exec"].get(d.year):
                    need[d] = ("dec", cal["nov_end"][d.year])
            cands: set = set()
            for d, (kind, me) in need.items():
                bd = board_for(me, held)
                if kind in ("start", "dec"):
                    cands |= set(bd["order"])
            frames, failed = fetch_frames(held + sorted(cands), (days[0] - timedelta(days=10)).date(), base, history_fn)
            bad_held = [f for f in failed if f.split(":")[0] in held]
            if bad_held:
                raise vbdata.VirtualBooksDataError(f"{name}: 보유 종목 가격을 받지 못했습니다 — " + "; ".join(bad_held))
            if failed:
                out["warnings"].append(f"{name}: 후보 가격 못 받음 {len(failed)}개 — " + "; ".join(failed[:5]))
            days, defer = pb.defer_incomplete_last_day(days, frames, held, days_index)
            if defer:
                out["warnings"].append(f"{name}: {defer}")
            fx = pd.Series((fx_range_fn or _fx_range)(days[0] - timedelta(days=10), base)) if days else pd.Series(dtype=float)
            if days:
                fx.index = pd.to_datetime(fx.index)
                fx = fx.sort_index()
            for d in days:
                f = fx[fx.index <= d]
                if f.empty:
                    raise RuntimeError(f"{name}: {d.date()} 원/달러 환율이 없습니다")
                closes = {t: float(df.at[d, "close"]) for t, df in frames.items() if d in df.index and not pd.isna(df.at[d, "close"])}
                divs = {t: float(df.at[d, "dividend"]) for t, df in frames.items() if d in df.index and df.at[d, "dividend"] > 0}
                splits = {t: float(df.at[d, "split"]) for t, df in frames.items() if d in df.index and df.at[d, "split"] > 0}
                kind, me = need.get(d, (None, None))
                kw = {}
                if kind == "start":
                    kw["start_scores"] = board_for(me, [])
                if d in cal["quarter_ends"]:
                    kw["quarter_scores"] = board_for(d, sorted(book["positions"]))
                if kind == "dec":
                    kw["nov_scores"] = board_for(me, sorted(book["positions"]))
                book, row, trades = vbk.process_day(book, d, closes=closes, fx=float(f.iloc[-1]), dividends=divs, splits=splits, p=p,
                                                    is_dec_exec=kind == "dec", is_may_first=d in cal["may_first"],
                                                    is_year_end=d in cal["year_ends"], **kw)
                if row is not None:
                    new_daily.append(row)
                new_trades.extend(trades)
        if not dry_run and (days or vbstore.load_picks_book(conn, name) is None):
            vbstore.save_picks_day(conn, book, new_daily, new_trades, [])
        if book["status"] != vbk.RUNNING:
            out["warnings"].append(f"{name}: 시작일 {start.date()} 대기 중")
            return out
        daily_rows = vbstore.picks_rows(conn, "picks_daily", name) + (new_daily if dry_run else [])
        trade_rows = pb._db_trades(vbstore.picks_rows(conn, "picks_trades", name)) + (new_trades if dry_run else [])
        daily = pb._db_daily(daily_rows)
        if daily.empty:
            return out
        summary = pb.summarize_momentum(book, daily, int(vbc["min_days_for_annualized"]))
        latest = max((k for k in boards), default=None) or vbstore.latest_value_scores_date(conn, name)
        bd = boards[latest][1] if latest in boards else (vbk.board(vbstore.load_value_scores(conn, name, latest)) if latest else None)
        if bd is not None:  # 보유 종목 옆에 100점 만점 점수(그 월말)
            summary["holdings_list"] = ", ".join(
                f"{t}({vbk.display_score(bd['total'].get(t)):.0f})" if bd["total"].get(t) is not None else f"{t}(-)"
                for t in sorted(book["positions"]))
        summary["rules"] = f"재무 점수 상위 10종목 오래 보유, 12월에 세금 없는 만큼만 갈아타기 · 문턱 {vbk.threshold_text(p)}"
        out["extra"].append({
            "name": name, "kind": KIND, "start": pd.Timestamp(book["start_date"]), "daily": daily, "trades": trade_rows,
            "summary": summary, "positions": book["positions"],
            "meta": {"start_krw": p.start_krw, "start_fx": book.get("start_fx"), "fx_date": str(pd.Timestamp(book["start_date"]).date()),
                     "initial_usd": book["initial_usd"]},
        })
    except Exception as exc:  # 이 장부의 오류가 다른 장부·브리핑을 막지 않게
        out["errors"].append(f"{name} 오류: {type(exc).__name__}: {exc}")
    return out


def _fx_range(start, end) -> dict:
    from data import fx

    return fx.fetch_usd_krw_range(pd.Timestamp(start).date(), pd.Timestamp(end).date())
