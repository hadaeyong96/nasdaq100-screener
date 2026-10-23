"""상승장 대형주 웅덩이 v3.1 — 가상 1억 장부 하루 실행 (docs/prereg/puddle_v31_forward.md, 태그 puddle-v31-prereg).

실제 주문 없음. 장부 4종(v3.1·v3·QQQ 보유·무작위 200경로)을 data/virtual_books.db(드라이브 동기화)에 상태로 저장해 이어 쓴다.
흐름 (engine/daily.py live 실행, 편지 만들기 전):
  1) QQQ·현재 나스닥100·보유 종목 확정 종가(data.prices.fetch_one 캐시) → 지표(core.puddle_v31.indicators)
  2) 지난 기준일 다음 거래일부터 오늘 기준일까지 날짜 순서대로 장부 진행(놓친 날 따라잡기)
  3) DB 저장, 사용자 시트 「웅덩이 v3.1 (가상)」 탭에만 기준일마다 한 줄(GitHub Actions에서만)
  4) 편지 줄(개인판·단체방판)·점검 메모를 돌려준다
시작일은 main 병합 뒤 첫 정규 실행(GitHub Actions)의 기준일. 로컬 실행은 저장·시트 쓰기 없이 미리보기만 한다.
장부 오류는 run_safely가 잡는다 — 기존 브리핑은 그대로 나간다.

단독 미리보기:  python -m engine.puddle_v31 --dry-run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import puddle_v31 as pv  # noqa: E402
from store import virtual_books as vbstore  # noqa: E402

META_STATE = "puddle_v31_state"
META_JUDGMENT = "puddle_v31_judgment"
MAIN, V3, QQQ = "v31", "v3", "qqq"


# ── 저장 ─────────────────────────────────────────────────────────────────────


def _ensure_schema(conn) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS puddle_v31_daily (date TEXT PRIMARY KEY, row_json TEXT);
        CREATE TABLE IF NOT EXISTS puddle_v31_trades (ledger TEXT, exit_date TEXT, tk TEXT, row_json TEXT);
    """)
    conn.commit()


def load_state(conn) -> dict | None:
    raw = vbstore.get_meta(conn, META_STATE)
    return json.loads(raw) if raw else None


def save(conn, state: dict, daily_rows: list[dict], trades: list[tuple[str, dict]]) -> None:
    _ensure_schema(conn)
    vbstore.set_meta(conn, META_STATE, json.dumps(state, ensure_ascii=False))
    for r in daily_rows:
        conn.execute("INSERT OR REPLACE INTO puddle_v31_daily VALUES (?, ?)", (r["date"], json.dumps(r, ensure_ascii=False)))
    for ledger, t in trades:
        conn.execute("INSERT INTO puddle_v31_trades VALUES (?, ?, ?, ?)", (ledger, t["exit_date"], t["tk"], json.dumps(t, ensure_ascii=False)))
    conn.commit()


def daily_history(conn) -> list[dict]:
    _ensure_schema(conn)
    return [json.loads(r[0]) for r in conn.execute("SELECT row_json FROM puddle_v31_daily ORDER BY date")]


# ── 입력 ─────────────────────────────────────────────────────────────────────


def _default_history(ticker: str, cfg: dict) -> pd.DataFrame:
    from data.prices import fetch_one

    return fetch_one(ticker, cfg)[["close", "volume"]]


def _default_universe() -> list[str]:
    from data.universe import get_universe

    return list(get_universe()["ticker"])


def _default_fx(dates: list) -> dict:
    from data.fx import get_usd_krw_rate_map

    return get_usd_krw_rate_map(dates)


def _default_year_days(year: int) -> pd.DatetimeIndex:
    from data.market_calendar import trading_days_between

    return trading_days_between(pd.Timestamp(year, 1, 1).date(), pd.Timestamp(year, 12, 31).date())


class FilingChecker:
    """실행마다 종목당 한 번 EDGAR 접수 목록을 받아 두고, 신호 날마다 실적 접수 여부를 본다(실패는 None)."""

    def __init__(self, filings_fn=None):
        self._fn = filings_fn or self._edgar
        self._cache: dict[str, list | None] = {}
        self._cik = None

    def _edgar(self, ticker: str) -> list[dict]:
        from data import edgar

        if self._cik is None:
            self._cik = edgar.fetch_ticker_to_cik()
        cik = self._cik.get(ticker.upper().replace("-", "."), self._cik.get(ticker.upper()))
        if cik is None:
            raise LookupError(f"CIK 없음: {ticker}")
        return edgar.fetch_recent_filings(cik)

    def filings(self, ticker: str) -> list | None:
        if ticker not in self._cache:
            try:
                self._cache[ticker] = self._fn(ticker)
            except Exception as exc:  # noqa: BLE001 — 조회 실패는 그 종목 새 매수만 막는다
                print(f"[puddle_v31] 실적 접수 조회 실패 {ticker}: {type(exc).__name__}")
                self._cache[ticker] = None
        return self._cache[ticker]

    def check(self, ticker: str, window_start, signal_date) -> bool | None:
        f = self.filings(ticker)
        return None if f is None else pv.has_earnings_filing(f, window_start, signal_date)

    def last_filing(self, ticker: str, upto) -> str | None:
        """upto 이전 마지막 실적 접수일(표시용)."""
        f = self.filings(ticker)
        if f is None:
            return None
        ds = [x["filingDate"] for x in f if (x["form"] in ("10-Q", "10-K") or (x["form"] == "8-K" and "2.02" in str(x.get("items"))))
              and pd.Timestamp(x["filingDate"]) <= pd.Timestamp(upto)]
        return max(ds) if ds else None


class World:
    """가격·지표·달력 — 실행 한 번 동안 고정."""

    def __init__(self, cfg, universe: list[str], held: set[str], history_fn, p: pv.Params):
        self.p = p
        q = history_fn("QQQ", cfg)
        self.cal = pd.DatetimeIndex(q.index).normalize()
        self.qqq = q["close"].set_axis(self.cal)
        self.qqq_sma200 = self.qqq.rolling(200, min_periods=200).mean()
        self.universe = list(dict.fromkeys(universe))
        closes, vols, self.missing = {}, {}, []
        for t in dict.fromkeys(self.universe + sorted(held)):
            try:
                df = history_fn(t, cfg)
            except Exception as exc:  # noqa: BLE001
                if t in held:
                    raise RuntimeError(f"보유 종목 {t} 가격을 받지 못했습니다: {type(exc).__name__}") from None
                self.missing.append(t)
                continue
            idx = pd.DatetimeIndex(df.index).normalize()
            closes[t], vols[t] = df["close"].set_axis(idx), df["volume"].set_axis(idx)
        self.close = pd.DataFrame(closes).reindex(self.cal)
        self.volume = pd.DataFrame(vols).reindex(self.cal)
        self.ind = pv.indicators(self.close, self.volume, p)
        self.last_valid = {t: s.last_valid_index() for t, s in self.close.items()}

    def pos(self, d) -> int:
        return int(self.cal.get_loc(pd.Timestamp(d)))

    def month_first(self, d) -> bool:
        i = self.pos(d)
        return i == 0 or self.cal[i - 1].month != self.cal[i].month

    def top10(self, d) -> list[str]:
        row = self.ind["dv"].loc[pd.Timestamp(d)]
        return pv.top_by_dollar_volume({t: row.get(t) for t in self.universe if t in row.index}, self.universe, self.p.top_n)

    def tickers_ind(self, d, tickers) -> tuple[dict, dict]:
        d = pd.Timestamp(d)
        closes, ind = {}, {}
        for t in tickers:
            if t not in self.close.columns:
                closes[t] = float("nan")
                continue
            lv = self.last_valid.get(t)
            closes[t] = float(self.close.at[d, t]) if lv is not None and d <= lv else float("nan")
            ind[t] = {k: float(v.at[d, t]) for k, v in self.ind.items() if k != "dv"}
        return closes, ind


# ── 하루 진행 ─────────────────────────────────────────────────────────────────


def _all_ledgers(state):
    return list((state.get("ledgers") or {}).values()) + list(state.get("randoms") or [])


def build_day(state, world: World, d, fx: float, year_days_fn) -> pv.Day:
    d = pd.Timestamp(d)
    i = world.pos(d)
    held = {tk for L in _all_ledgers(state) for tk in L["pos"]}
    if world.month_first(d) or not state.get("top10"):
        state["top10"] = {"month": d.strftime("%Y-%m"), "date": str(d.date()), "list": world.top10(d)}
    top = state["top10"]["list"]
    closes, ind = world.tickers_ind(d, set(top) | held)
    yd = year_days_fn(d.year)
    may = d.month == 5 and (i == 0 or world.cal[i - 1].month != 5)
    harvest = len(yd) >= 3 and d == pd.Timestamp(yd[-3]).normalize()
    q20 = float(world.qqq.iloc[i] / world.qqq.iloc[i - 20] - 1) if i >= 20 else float("nan")
    return pv.Day(date=d, idx=i - world.pos(state["start_date"]), fx=fx, qqq=float(world.qqq.iloc[i]),
                  qqq_sma200=float(world.qqq_sma200.iloc[i]), qqq_ret20=q20, closes=closes, ind=ind, top10=top,
                  may_first=may, harvest=harvest)


def init_state(world: World, start, fx: float, year_days_fn, p: pv.Params) -> dict:
    state = {"start_date": str(pd.Timestamp(start).date()), "start_fx": fx, "last_date": None, "top10": None,
             "ledgers": {}, "randoms": [], "rng": None}
    day = build_day(state, world, start, fx, year_days_fn)
    state["ledgers"] = {MAIN: pv.new_ledger(MAIN, day, p, earnings_filter=True),
                        V3: pv.new_ledger(V3, day, p, earnings_filter=False),
                        QQQ: pv.new_ledger(QQQ, day, p, earnings_filter=False, stocks=False)}
    state["randoms"] = [pv.new_ledger(f"r{k}", day, p, earnings_filter=True, random=True) for k in range(p.n_random)]
    state["rng"] = np.random.default_rng(p.seed).bit_generator.state
    return state


def advance_day(state, day: pv.Day, world: World, checker: FilingChecker, p: pv.Params) -> dict:
    """하루 진행 → 그날 기록 한 줄(dict)."""
    win_start = world.cal[max(0, world.pos(day.date) - (p.filing_window - 1))]

    def filing(tk):
        return checker.check(tk, win_start, day.date)

    L = state["ledgers"]
    ev = pv.step(L[MAIN], day, p, filing=filing)
    ev3 = pv.step(L[V3], day, p)
    pv.step(L[QQQ], day, p)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    for R in state["randoms"]:
        pv.step(R, day, p, filing=filing, random_n=ev["entries"], rng=rng)
    state["rng"] = rng.bit_generator.state
    state["last_date"] = str(day.date.date())
    rv = [R["last_value"] for R in state["randoms"]]
    dist = pv.percentiles(rv)
    w = pv.weights(L[MAIN], day)
    nav = pv.nav_usd(L[MAIN], day)
    return {"date": str(day.date.date()), "v31": L[MAIN]["last_value"], "qqq": L[QQQ]["last_value"], "v3": L[V3]["last_value"],
            "random": dist, "weights": w, "qqq_weight": L[MAIN]["q_units"] * day.qqq / nav if nav else 1.0,
            "fills": ev["fills"], "sells": ev["sells"], "skipped": ev["skipped"], "failed": sorted(set(ev["failed"])),
            "v3_fills": ev3["fills"], "v3_sells": ev3["sells"], "realized_year": L[MAIN]["realized"].get(str(day.date.year), 0.0),
            "mdd": {MAIN: L[MAIN]["mdd"], QQQ: L[QQQ]["mdd"], V3: L[V3]["mdd"]}, "top10": list(day.top10)}


# ── 표시 ─────────────────────────────────────────────────────────────────────


def _pp(v_krw: float) -> str:
    from notify import letter

    return letter._pp(v_krw / 100_000_000.0 * 100)


def _hold_text(w: dict, personal: bool) -> str:
    """보유 한 줄(시트·예전 표시용)."""
    if not w:
        return "없음"
    if personal:
        return ", ".join(f"{t} {x * 100:.1f}%" for t, x in sorted(w.items(), key=lambda kv: -kv[1]))
    return f"{len(w)}종목·합계 {sum(w.values()) * 100:.1f}%"


def _trade_text(rows: list[dict], personal: bool) -> str | None:
    """시트 「오늘 매매」 칸(짧은 표기)."""
    fills = [f for r in rows for f in r["fills"]]
    sells = [s for r in rows for s in r["sells"]]
    if not fills and not sells:
        return None
    parts = [f"{f['tk']} {f['tranche']}차 매수" for f in fills] + [f"{s['tk']} 매도({s['reason']})" for s in sells]
    return "오늘 가상 매매: " + " · ".join(parts)


def _plain_lines(rows: list[dict], personal: bool) -> list[str]:
    """편지 [가상 실험] 줄 — 쉬운 문장(2026-10-23 feat/letter-plain). 단체방판은 종목명 없이."""
    from notify import letter_plain as lp

    last = rows[-1]
    lines = [f"웅덩이 v3.1: 가상 1억이 {lp.vs_qqq((last['v31'] - last['qqq']) / 1_000_000)}."]
    w = last["weights"]
    if not w:
        lines.append("지금은 종목 없이 QQQ만 들고 있어요.")
    elif personal:
        items = sorted(w.items(), key=lambda kv: -kv[1])
        if len(items) == 1:
            lines.append(f"지금 {lp.josa(items[0][0], ('을', '를'))} {items[0][1] * 100:.0f}% 담고 있어요.")
        else:
            lines.append("지금 " + ", ".join(f"{t} {x * 100:.0f}%" for t, x in items) + "를 담고 있어요.")
    else:
        tot = sum(w.values()) * 100
        lines.append(f"지금 1종목을 {tot:.0f}% 담고 있어요." if len(w) == 1 else f"지금 {len(w)}종목을 합계 {tot:.0f}% 담고 있어요.")
    fills = [f for r in rows for f in r["fills"]]
    sells = [s for r in rows for s in r["sells"]]
    skipped = [s for r in rows for s in r["skipped"]]
    if personal:
        for f in fills:
            lines.append(f"오늘 {lp.josa(f['tk'], ('을', '를'))} {'처음' if f['tranche'] == 1 else '한 번 더'} 샀어요({f['tranche']}차).")
        for s in sells:
            lines.append(f"오늘 {lp.josa(s['tk'], ('을', '를'))} 모두 팔았어요({s['reason']}).")
        for s in skipped:
            lines.append(f"오늘은 {lp.josa(s['tk'], ('이', '가'))} 내려왔지만 실적 발표 직후라 사지 않았어요.")
    else:
        if fills:
            lines.append(f"오늘 가상으로 {len({f['tk'] for f in fills})}종목을 샀어요.")
        if sells:
            lines.append(f"오늘 가상으로 {len(sells)}종목을 팔았어요.")
        if skipped:
            lines.append(f"오늘 {len(skipped)}종목이 내려왔지만 실적 발표 직후라 사지 않았어요.")
    return lines


def letter_lines(rows: list[dict]) -> dict:
    """이번 실행에서 진행한 날들의 기록 → 편지 줄 {personal: [...], group: [...], check_personal, check_group}."""
    if not rows:
        return {"personal": [], "group": [], "check_personal": None, "check_group": None}
    out = {"personal": _plain_lines(rows, True), "group": _plain_lines(rows, False)}
    failed = sorted({t for r in rows for t in r["failed"]})
    out["check_personal"] = f"실적 확인 실패({', '.join(failed)})" if failed else None
    out["check_group"] = f"실적 확인 실패({len(failed)}종목)" if failed else None
    return out


def sheet_row(r: dict) -> list:
    hold = ", ".join(f"{t} {x * 100:.1f}%" for t, x in sorted(r["weights"].items(), key=lambda kv: -kv[1])) or "없음"
    tr = _trade_text([r], True)
    return [r["date"], round(r["v31"]), round(r["qqq"]), round(r["v3"]),
            round((r["v31"] - r["qqq"]) / 1_000_000, 2), round((r["v31"] - r["random"]["50"]) / 1_000_000, 2),
            f"{r['qqq_weight'] * 100:.1f}%", hold, tr.replace("오늘 가상 매매: ", "") if tr else "", round(r["realized_year"])]


# ── 실행 ─────────────────────────────────────────────────────────────────────


def judgment_date(start) -> pd.Timestamp:
    return pd.Timestamp(start) + pd.DateOffset(months=12)


def run(cfg: dict, as_of, *, db_path=None, dry_run: bool = False, history_fn=None, universe_fn=None, fx_fn=None,
        year_days_fn=None, filings_fn=None, sheets_client=None, sheet_write: bool | None = None, p: pv.Params = pv.Params()) -> dict:
    """하루 실행. as_of = 오늘 기준일(직전 확정 미국 종가일).

    dry_run=True면 DB·시트에 쓰지 않는다. sheet_write 기본값은 GitHub Actions에서만 True.
    출력: {status, start_date, rows(이번에 진행한 날들), letter, warnings, errors, diag}"""
    pc = cfg.get("puddle_v31") or {}
    as_of = pd.Timestamp(as_of).normalize()
    history_fn = history_fn or _default_history
    year_days_fn = year_days_fn or _default_year_days
    fx_fn = fx_fn or _default_fx
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    if sheet_write is None:
        sheet_write = in_actions
    db_path = Path(db_path or vbstore.VBOOKS_DB_PATH)
    conn = vbstore.connect(db_path)
    result = {"status": "ok", "rows": [], "warnings": [], "errors": [], "letter": letter_lines([]), "diag": None}
    try:
        state = load_state(conn)
        if state is None and in_actions and not pc.get("start_date") and not (os.environ.get("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS") or "").strip():
            raise RuntimeError("드라이브 state 파일도 config 시작일도 없어 매 실행이 새 장부가 됩니다 — 웅덩이 v3.1을 건너뜁니다")
        held = {tk for L in (_all_ledgers(state) if state else []) for tk in L["pos"]}
        universe = (universe_fn or _default_universe)()
        world = World(cfg, universe, held, history_fn, p)
        if as_of not in world.cal:
            as_of = world.cal[world.cal <= as_of][-1]
        if world.missing:
            result["warnings"].append(f"가격 없는 구성 종목 {len(world.missing)}개: {', '.join(world.missing[:10])}")
        checker = FilingChecker(filings_fn)
        fresh = state is None
        if fresh:
            start = pd.Timestamp(pc.get("start_date") or as_of).normalize()
            fx0 = float(pc["start_fx"]) if pc.get("start_fx") else fx_fn([start.date()]).get(str(start.date()))
            if fx0 is None:
                raise RuntimeError(f"시작일 {start.date()} 환율이 없습니다")
            state = init_state(world, start, float(fx0), year_days_fn, p)
            todo = [start]
        else:
            last = pd.Timestamp(state["last_date"])
            todo = list(world.cal[(world.cal > last) & (world.cal <= as_of)])
        result["start_date"] = state["start_date"]
        if not todo:
            result["status"] = "already"
            hist = daily_history(conn)
            result["letter"] = letter_lines(hist[-1:]) if hist else letter_lines([])
            result["rows_all"] = hist
            return result
        fxm = fx_fn([d.date() for d in todo])
        rows, trades = [], []
        prev_fx = state["start_fx"] if fresh else None
        for d in todo:
            fx = fxm.get(str(d.date())) or prev_fx or state.get("last_fx")
            if fx is None:
                raise RuntimeError(f"{d.date()} 환율이 없습니다")
            prev_fx = fx
            state["last_fx"] = fx
            day = build_day(state, world, d, float(fx), year_days_fn)
            r = advance_day(state, day, world, checker, p)
            rows.append(r)
            trades += [(MAIN, t) for t in r["sells"]] + [(V3, t) for t in r["v3_sells"]]
            if pd.Timestamp(d) >= judgment_date(state["start_date"]) and not state.get("judged"):
                p80 = r["random"]["80"]
                state["judged"] = {"date": r["date"], **pv.judge(r["v31"], r["qqq"], r["v3"], p80, r["mdd"][MAIN], r["mdd"][QQQ])}
        result["rows"] = rows
        result["letter"] = letter_lines(rows)
        result["diag"] = diagnose(world, checker, as_of, state)
        if dry_run or not in_actions:
            result["status"] = "preview"
            return result
        save(conn, state, rows, trades)
        if state.get("judged"):
            vbstore.set_meta(conn, META_JUDGMENT, json.dumps(state["judged"], ensure_ascii=False))
        if sheet_write:
            sheet_id = (os.environ.get(pc.get("sheet_id_env", "GOOGLE_SHEETS_ID")) or "").strip()
            if not sheet_id:
                result["warnings"].append("GOOGLE_SHEETS_ID가 없어 웅덩이 v3.1 탭 기록을 건너뜁니다")
            else:
                try:
                    from data import puddle_v31_sheet as psheet

                    if sheets_client is None:
                        from data import sheets

                        sheets_client = sheets.get_client()
                    psheet.write_rows(sheets_client, sheet_id, [sheet_row(r) for r in rows])
                except Exception as exc:  # noqa: BLE001 — 장부는 저장됨, 시트만 실패
                    result["errors"].append(f"웅덩이 v3.1 시트 기록 실패: {type(exc).__name__}")
        return result
    finally:
        conn.close()


def diagnose(world: World, checker: FilingChecker, d, state) -> list[dict]:
    """오늘 상위 10 진단(미리보기·보고용): 자격·웅덩이 상태·최근 실적 접수일."""
    d = pd.Timestamp(d)
    top = state["top10"]["list"] if state.get("top10") else world.top10(d)
    closes, ind = world.tickers_ind(d, top)
    out = []
    for t in top:
        x, i = closes.get(t), ind.get(t) or {}
        qual = pv._qualified(i, x)
        below60 = x < i.get("sma60", np.nan)
        out.append({"ticker": t, "close": x, "qualified": qual, "below_sma60": bool(below60),
                    "above_prev20": i.get("above_prev") == 1.0, "first_cross_60": bool(below60 and i.get("above_prev") == 1.0),
                    "below_sma120": bool(x < i.get("sma120", np.nan)), "last_filing": checker.last_filing(t, d)})
    return out


def run_safely(cfg: dict, as_of, **kwargs) -> dict | None:
    """run을 감싼다 — 예외를 밖으로 내보내지 않는다(브리핑 보호)."""
    if not (cfg.get("puddle_v31") or {}).get("enabled", False):
        return None
    try:
        res = run(cfg, as_of, **kwargs)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[puddle_v31] 장부 오류(브리핑은 그대로): {type(exc).__name__}: {exc}")
        return {"status": "error", "error": f"{type(exc).__name__}", "letter": letter_lines([]), "rows": [], "warnings": [], "errors": [str(exc)]}
    for w in res.get("warnings", []):
        print(f"[puddle_v31] 경고: {w}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="웅덩이 v3.1 가상 장부 미리보기")
    ap.add_argument("--dry-run", action="store_true", help="DB·시트에 쓰지 않음(로컬은 늘 미리보기)")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args(argv)
    from data.market_calendar import latest_closed_trading_day
    from engine.daily import US_EASTERN, load_config

    cfg = load_config()
    as_of = a.as_of or latest_closed_trading_day(datetime.now(US_EASTERN))
    res = run(cfg, as_of, dry_run=True)
    print(json.dumps({k: res[k] for k in ("status", "start_date", "letter", "warnings")}, ensure_ascii=False, indent=1, default=str))
    for r in res.get("diag") or []:
        print(r)


if __name__ == "__main__":
    main()
