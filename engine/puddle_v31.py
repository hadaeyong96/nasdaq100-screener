"""상승장 대형주 웅덩이 v3.1 — 가상 1억 장부 하루 실행 (docs/prereg/puddle_v31_forward.md, 태그 puddle-v31-prereg).

2026-10-27: v3.1 장부는 보관(config enabled false, DB 기록은 그대로)하고, 같은 코드를 Book(BOOK_V32)으로 불러
웅덩이 v3.2(시가총액 상위 10, docs/prereg/puddle_v32_forward.md)를 돌린다 — engine/puddle_v32.py. 규칙 코드는 같고 대상 고르기·저장 위치만 다르다.

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
import importlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import puddle_v31 as pv  # noqa: E402
from store import virtual_books as vbstore  # noqa: E402

MAIN, V3, QQQ = "v31", "v3", "qqq"  # 장부 이름(v3.2도 같은 이름 — 일별 줄의 값 열 이름 "v31"·"v3"도 그대로 쓴다)


@dataclass(frozen=True)
class Book:
    """장부 하나의 이름·저장 위치·대상 고르기(2026-10-27). 규칙은 같고 이것만 다르다.

    ranking: "dv"(20일 평균 거래대금 상위 10 — v3.1) | "mcap"(시가총액 상위 10 — v3.2, docs/prereg/puddle_v32_forward.md 2-2)."""

    cfg_key: str
    label: str  # 편지·로그 이름
    name_key: str  # config/letter_phrases.yaml lab.names 키
    table: str  # 일별 기록 표 = {table}_daily, 매매 = {table}_trades, meta = {table}_state·{table}_judgment
    sheet_module: str
    ranking: str = "dv"
    archives: tuple = ()  # 이 장부가 처음 저장될 때 DB에 보관 표시할 옛 장부 표 이름(기록은 지우지 않음)

    @property
    def meta_state(self) -> str:
        return f"{self.table}_state"

    @property
    def meta_judgment(self) -> str:
        return f"{self.table}_judgment"


BOOK = Book("puddle_v31", "웅덩이 v3.1", "v31", "puddle_v31", "data.puddle_v31_sheet")  # 2026-10-27 보관(매일 실행에서 끔)
BOOK_V32 = Book("puddle_v32", "웅덩이 v3.2", "v32", "puddle_v32", "data.puddle_v32_sheet", ranking="mcap",
                archives=("puddle_v31", "puddle_rebound"))
META_STATE, META_JUDGMENT = BOOK.meta_state, BOOK.meta_judgment
META_ARCHIVED = "puddle_archived"  # 보관 표시(2026-10-27): {장부 표 이름: {date, tag, reason}} — 옛 기록은 지우지 않는다
ARCHIVE_TAG = "archive-puddle-v31-20261027"


# ── 저장 ─────────────────────────────────────────────────────────────────────


def _ensure_schema(conn, book: Book = BOOK) -> None:
    conn.executescript(f"""
        CREATE TABLE IF NOT EXISTS {book.table}_daily (date TEXT PRIMARY KEY, row_json TEXT);
        CREATE TABLE IF NOT EXISTS {book.table}_trades (ledger TEXT, exit_date TEXT, tk TEXT, row_json TEXT);
    """)
    conn.commit()


def load_state(conn, book: Book = BOOK) -> dict | None:
    raw = vbstore.get_meta(conn, book.meta_state)
    return json.loads(raw) if raw else None


def save(conn, state: dict, daily_rows: list[dict], trades: list[tuple[str, dict]], book: Book = BOOK) -> None:
    _ensure_schema(conn, book)
    vbstore.set_meta(conn, book.meta_state, json.dumps(state, ensure_ascii=False))
    for r in daily_rows:
        conn.execute(f"INSERT OR REPLACE INTO {book.table}_daily VALUES (?, ?)", (r["date"], json.dumps(r, ensure_ascii=False)))
    for ledger, t in trades:
        conn.execute(f"INSERT INTO {book.table}_trades VALUES (?, ?, ?, ?)", (ledger, t["exit_date"], t["tk"], json.dumps(t, ensure_ascii=False)))
    conn.commit()


def daily_history(conn, book: Book = BOOK) -> list[dict]:
    _ensure_schema(conn, book)
    return [json.loads(r[0]) for r in conn.execute(f"SELECT row_json FROM {book.table}_daily ORDER BY date")]


def mark_archived(conn, tables: list[str], date: str, reason: str) -> list[str]:
    """옛 장부 보관 표시(meta puddle_archived). 기록·상태는 지우지 않는다. 이미 표시된 표는 그대로 → 새로 표시한 표 목록."""
    raw = vbstore.get_meta(conn, META_ARCHIVED)
    cur = json.loads(raw) if raw else {}
    new = [t for t in tables if t not in cur]
    for t in new:
        cur[t] = {"date": date, "tag": ARCHIVE_TAG, "reason": reason}
    if new:
        vbstore.set_meta(conn, META_ARCHIVED, json.dumps(cur, ensure_ascii=False))
    return new


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


def _default_shares(tickers: list[str]):
    from data.shares import fetch_shares

    return fetch_shares(tickers)


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

    def __init__(self, cfg, universe: list[str], held: set[str], history_fn, p: pv.Params, ranking: str = "dv", shares_fn=None):
        self.p = p
        self.ranking = ranking
        self._shares_fn = shares_fn or _default_shares
        self._shares = None  # (주식 수, 회사 전체 티커, 실패 티커) — 시총 판 순위 날에만 한 번 받는다
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
        return self.rank(d)["list"]

    def rank(self, d) -> dict:
        """그날 상위 10 → {list, caps(시총 판만: 거래 티커 → 회사 시가총액 달러), missing(주식 수 없는 티커)}."""
        row = self.ind["dv"].loc[pd.Timestamp(d)]
        dv = {t: row.get(t) for t in self.universe if t in row.index}
        if self.ranking != "mcap":
            return {"list": pv.top_by_dollar_volume(dv, self.universe, self.p.top_n), "caps": None, "missing": []}
        if self._shares is None:
            self._shares = self._shares_fn([t for t in self.universe if t in self.close.columns])
        shares, whole, _ = self._shares
        crow = self.close.loc[pd.Timestamp(d)]
        closes = {t: float(crow[t]) for t in self.universe if t in crow.index}
        top, caps, missing = pv.top_by_market_cap(closes, shares, dv, self.universe, self.p.top_n, whole=set(whole))
        return {"list": top, "caps": caps, "missing": missing}

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
        rk = world.rank(d)
        state["top10"] = {"month": d.strftime("%Y-%m"), "date": str(d.date()), "list": rk["list"]}
        if rk["caps"] is not None:  # 시총 판(v3.2·반등 확인 v2): 순위 날 시가총액·주식 수 없는 종목도 남긴다
            state["top10"].update(caps=rk["caps"], missing=rk["missing"])
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
            "mdd": {MAIN: L[MAIN]["mdd"], QQQ: L[QQQ]["mdd"], V3: L[V3]["mdd"]}, "top10": list(day.top10), **rank_fields(state, day)}


def rank_fields(state, day: pv.Day) -> dict:
    """순위 날(그날 상위 10을 새로 정한 날) 시총 판 줄에만: caps(시가총액 달러)·shares_missing(주식 수 없는 티커)."""
    t = state.get("top10") or {}
    if t.get("date") != str(day.date.date()) or "caps" not in t:
        return {}
    return {"caps": t["caps"], "shares_missing": list(t.get("missing") or [])}


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


def _plain_lines(rows: list[dict], personal: bool, phrases: dict | None = None) -> list[str]:
    """편지 ■ 가상 실험 줄 — config/letter_phrases.yaml lab 문구(경어체, 2026-10-24 아침 편지 v3). 단체방판은 종목명 없이."""
    from notify import letter_plain as lp

    ph = phrases if phrases is not None else lp.load_phrases()
    day = rows[-1].get("date")
    last = rows[-1]
    key, d = lp.vs_qqq_parts((last["v31"] - last["qqq"]) / 1_000_000)
    lines = [lp.phrase(ph, f"lab.{key}", day, d=d)]
    w = last["weights"]
    if not w:
        lines.append(lp.phrase(ph, "lab.none", day))
    elif personal:
        items = sorted(w.items(), key=lambda kv: -kv[1])
        head = ", ".join(f"{t} {x * 100:.0f}%" for t, x in items[:-1])
        tk = (head + ", " if head else "") + items[-1][0]
        lines.append(lp.phrase(ph, "lab.hold", day, 종목=tk, w=f"{items[-1][1] * 100:.0f}"))
    else:
        lines.append(lp.phrase(ph, "lab.hold_group", day, n=len(w), w=f"{sum(w.values()) * 100:.0f}"))
    fills = [f for r in rows for f in r["fills"]]
    sells = [s for r in rows for s in r["sells"]]
    skipped = [s for r in rows for s in r["skipped"]]
    if personal:
        for f in fills:
            lines.append(lp.phrase(ph, "lab.buy1" if f["tranche"] == 1 else "lab.buy2", day, 종목=f["tk"]))
        for s in sells:
            lines.append(lp.phrase(ph, f"lab.sell_{_REASON_KEY.get(s['reason'], 'gone')}", day, 종목=s["tk"]))
        for s in skipped:
            lines.append(lp.phrase(ph, "lab.skip_earnings", day, 종목=s["tk"]))
    else:
        if fills:
            lines.append(lp.phrase(ph, "lab.group_buy", day, n=len({f["tk"] for f in fills})))
        if sells:
            lines.append(lp.phrase(ph, "lab.group_sell", day, n=len(sells)))
        if skipped:
            lines.append(lp.phrase(ph, "lab.skip_earnings_group", day, n=len({s["tk"] for s in skipped})))
    return lines


_REASON_KEY = {v: k for k, v in pv.REASONS.items()}  # 매도 이유 글 → 문구 키(high·rank·max_hold·gone)


def shares_checks(rows: list[dict]) -> dict:
    """주식 수를 못 받아 시가총액 순위에서 뺀 종목(사전 등록 v3.2 2-2-7) → {shares_personal: "(티커, …)", shares_group: "(n종목)"}."""
    miss = sorted({t for r in rows for t in r.get("shares_missing") or []})
    return {"shares_personal": f"({', '.join(miss)})" if miss else None, "shares_group": f"({len(miss)}종목)" if miss else None}


def letter_lines(rows: list[dict], book: Book = BOOK) -> dict:
    """이번 실행에서 진행한 날들의 기록 → 편지 줄 {personal: [...], group: [...], check_personal, check_group, shares_*}."""
    if not rows:
        return {"personal": [], "group": [], "check_personal": None, "check_group": None, "shares_personal": None, "shares_group": None}
    from notify import letter_plain as lp

    ph = lp.load_phrases()
    day = rows[-1].get("date")  # 장부 한 줄(2026-10-25): "웅덩이 v3.1: QQQ와 같습니다. QQQ만 들고 있습니다."
    out = {"personal": [lp.ledger_line(ph, book.name_key, _plain_lines(rows, True, ph), day)],
           "group": [lp.ledger_line(ph, book.name_key, _plain_lines(rows, False, ph), day)]}
    failed = sorted({t for r in rows for t in r["failed"]})
    out["check_personal"] = f"실적 확인 실패({', '.join(failed)})" if failed else None
    out["check_group"] = f"실적 확인 실패({len(failed)}종목)" if failed else None
    out.update(shares_checks(rows))
    last = rows[-1]  # 표시 전용(2026-10-24): 월요일 「필터 없는 판(v3)보다 x%p」 — 이미 기록하는 v3 값을 읽기만
    out["filter_gap_pp"] = (last["v31"] - last["v3"]) / 1_000_000 if last.get("v3") is not None else None
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
        year_days_fn=None, filings_fn=None, sheets_client=None, sheet_write: bool | None = None, p: pv.Params = pv.Params(),
        book: Book = BOOK, shares_fn=None) -> dict:
    """하루 실행. as_of = 오늘 기준일(직전 확정 미국 종가일). book = 장부(v3.1 보관본·v3.2), shares_fn = 주식 수(시총 판, 테스트 주입).

    dry_run=True면 DB·시트에 쓰지 않는다. sheet_write 기본값은 GitHub Actions에서만 True.
    출력: {status, start_date, rows(이번에 진행한 날들), letter, warnings, errors, diag}"""
    pc = cfg.get(book.cfg_key) or {}
    as_of = pd.Timestamp(as_of).normalize()
    history_fn = history_fn or _default_history
    year_days_fn = year_days_fn or _default_year_days
    fx_fn = fx_fn or _default_fx
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    if sheet_write is None:
        sheet_write = in_actions
    db_path = Path(db_path or vbstore.VBOOKS_DB_PATH)
    conn = vbstore.connect(db_path)
    result = {"status": "ok", "rows": [], "warnings": [], "errors": [], "letter": letter_lines([], book), "diag": None}
    try:
        state = load_state(conn, book)
        if state is None and in_actions and not pc.get("start_date") and not (os.environ.get("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS") or "").strip():
            raise RuntimeError(f"드라이브 state 파일도 config 시작일도 없어 매 실행이 새 장부가 됩니다 — {book.label}을 건너뜁니다")
        held = {tk for L in (_all_ledgers(state) if state else []) for tk in L["pos"]}
        universe = (universe_fn or _default_universe)()
        world = World(cfg, universe, held, history_fn, p, ranking=book.ranking, shares_fn=shares_fn)
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
            hist = daily_history(conn, book)
            result["letter"] = letter_lines(hist[-1:], book) if hist else letter_lines([], book)
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
        result["letter"] = letter_lines(rows, book)
        result["diag"] = diagnose(world, checker, as_of, state)
        if dry_run or not in_actions:
            result["status"] = "preview"
            return result
        save(conn, state, rows, trades, book)
        if book.archives:
            mark_archived(conn, list(book.archives), rows[-1]["date"], f"{book.label}(시가총액 상위 10)로 재시작 — 2026-10-27 사용자 지시")
        if state.get("judged"):
            vbstore.set_meta(conn, book.meta_judgment, json.dumps(state["judged"], ensure_ascii=False))
        if sheet_write:
            sheet_id = (os.environ.get(pc.get("sheet_id_env", "GOOGLE_SHEETS_ID")) or "").strip()
            if not sheet_id:
                result["warnings"].append(f"GOOGLE_SHEETS_ID가 없어 {book.label} 탭 기록을 건너뜁니다")
            else:
                try:
                    psheet = importlib.import_module(book.sheet_module)

                    if sheets_client is None:
                        from data import sheets

                        sheets_client = sheets.get_client()
                    psheet.write_rows(sheets_client, sheet_id, [sheet_row(r) for r in rows])
                except Exception as exc:  # noqa: BLE001 — 장부는 저장됨, 시트만 실패
                    result["errors"].append(f"{book.label} 시트 기록 실패: {type(exc).__name__}")
        return result
    finally:
        conn.close()


def diagnose(world: World, checker: FilingChecker, d, state) -> list[dict]:
    """오늘 상위 10 진단(미리보기·보고용): 자격·웅덩이 상태·최근 실적 접수일."""
    d = pd.Timestamp(d)
    top = state["top10"]["list"] if state.get("top10") else world.top10(d)
    closes, ind = world.tickers_ind(d, top)
    caps = (state.get("top10") or {}).get("caps") or {}
    out = []
    for t in top:
        x, i = closes.get(t), ind.get(t) or {}
        qual = pv._qualified(i, x)
        below60 = x < i.get("sma60", np.nan)
        out.append({"ticker": t, "close": x, "cap": caps.get(t), "qualified": qual, "below_sma60": bool(below60),
                    "above_prev20": i.get("above_prev") == 1.0, "first_cross_60": bool(below60 and i.get("above_prev") == 1.0),
                    "below_sma120": bool(x < i.get("sma120", np.nan)), "last_filing": checker.last_filing(t, d)})
    return out


def run_safely(cfg: dict, as_of, book: Book = BOOK, **kwargs) -> dict | None:
    """run을 감싼다 — 예외를 밖으로 내보내지 않는다(브리핑 보호). config enabled가 false(보관된 장부)면 돌지 않는다."""
    if not (cfg.get(book.cfg_key) or {}).get("enabled", False):
        return None
    try:
        res = run(cfg, as_of, book=book, **kwargs)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[{book.cfg_key}] 장부 오류(브리핑은 그대로): {type(exc).__name__}: {exc}")
        return {"status": "error", "error": f"{type(exc).__name__}", "letter": letter_lines([], book), "rows": [], "warnings": [], "errors": [str(exc)]}
    for w in res.get("warnings", []):
        print(f"[{book.cfg_key}] 경고: {w}")
    return res


def main(argv=None, book: Book = BOOK) -> None:
    ap = argparse.ArgumentParser(description=f"{book.label} 가상 장부 미리보기")
    ap.add_argument("--dry-run", action="store_true", help="DB·시트에 쓰지 않음(로컬은 늘 미리보기)")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args(argv)
    from data.market_calendar import latest_closed_trading_day
    from engine.daily import US_EASTERN, load_config

    cfg = load_config()
    as_of = a.as_of or latest_closed_trading_day(datetime.now(US_EASTERN))
    res = run(cfg, as_of, dry_run=True, book=book)
    print(json.dumps({k: res[k] for k in ("status", "start_date", "letter", "warnings")}, ensure_ascii=False, indent=1, default=str))
    for r in res.get("diag") or []:
        print(r)


if __name__ == "__main__":
    main()
