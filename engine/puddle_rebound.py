"""웅덩이 반등 확인 전략 — 가상 1억 장부 하루 실행 (docs/prereg/puddle_rebound_forward.md, 태그 puddle-rebound-prereg).

실제 주문 없음. 장부(본 장부·QQQ 보유·무작위 200경로)를 data/virtual_books.db(드라이브 동기화)에 상태로 저장해 이어 쓴다.
비교 B(웅덩이 v3.1)는 v3.1 장부가 저장한 puddle_v31_daily를 같은 날짜로 읽기만 한다(v3.1 규칙·기록은 건드리지 않는다).
가격·상위 10·실적 접수 조회·달력은 engine/puddle_v31의 World·FilingChecker·build_day를 그대로 쓴다.
흐름 (engine/daily.py live 실행, 편지 만들기 전): 지표 → 놓친 날 따라잡기 → DB 저장·시트 「웅덩이 반등 확인 (가상)」 탭(Actions에서만)
→ 편지 줄(■ 웅덩이 후보·■ 가상 실험)·점검 메모.

단독 미리보기:  python -m engine.puddle_rebound --dry-run

2026-10-27: 이 장부는 보관(config enabled false, DB 기록은 그대로)하고, 같은 코드를 RBook(BOOK_V2)으로 불러
웅덩이 반등 확인 v2(시가총액 상위 10, 비교 B = 웅덩이 v3.2, docs/prereg/puddle_rebound_v2_forward.md)를 돌린다 — engine/puddle_rebound_v2.py.
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

from core import puddle_rebound as pr  # noqa: E402
from core import puddle_v31 as pv  # noqa: E402
from engine import puddle_v31 as e31  # noqa: E402
from store import virtual_books as vbstore  # noqa: E402

MAIN, QQQ = "rebound", "qqq"


@dataclass(frozen=True)
class RBook(e31.Book):
    """반등 확인 장부 하나의 이름·저장 위치·대상·비교 B 표(2026-10-27). b_table = 비교 B 장부의 일별 기록 표(읽기만)."""

    b_table: str = "puddle_v31_daily"
    b_label: str = "v3.1"


BOOK = RBook("puddle_rebound", "웅덩이 반등 확인", "rebound", "puddle_rebound", "data.puddle_rebound_sheet")  # 2026-10-27 보관
BOOK_V2 = RBook("puddle_rebound_v2", "웅덩이 반등 확인 v2", "rebound_v2", "puddle_rebound_v2", "data.puddle_rebound_v2_sheet",
                ranking="mcap", b_table="puddle_v32_daily", b_label="v3.2")
META_STATE, META_JUDGMENT = BOOK.meta_state, BOOK.meta_judgment
CFG_KEY = BOOK.cfg_key


# ── 저장 ─────────────────────────────────────────────────────────────────────


def _ensure_schema(conn, book: RBook = BOOK) -> None:
    e31._ensure_schema(conn, book)


def load_state(conn, book: RBook = BOOK) -> dict | None:
    return e31.load_state(conn, book)


def save(conn, state: dict, daily_rows: list[dict], trades: list[tuple[str, dict]], book: RBook = BOOK) -> None:
    e31.save(conn, state, daily_rows, trades, book)


def daily_history(conn, book: RBook = BOOK) -> list[dict]:
    return e31.daily_history(conn, book)


def v31_values(conn, book: RBook = BOOK) -> dict[str, float]:
    """비교 B: 웅덩이 v3.1(v2 장부는 v3.2) 일별 세후 가치 {날짜: 원}(읽기만). 표가 없으면 빈 dict."""
    try:
        rows = conn.execute(f"SELECT date, row_json FROM {book.b_table}").fetchall()
    except Exception:  # noqa: BLE001 — 표 없음(비교 장부 첫 저장 전)
        return {}
    return {d: float(json.loads(j)["v31"]) for d, j in rows}


# ── 세계(가격·지표) ──────────────────────────────────────────────────────────


class World(e31.World):
    """v3.1 World + RSI(14)·전날 RSI·최근 3거래일 골든크로스(종목별)."""

    def __init__(self, cfg, universe, held, history_fn, p: pr.Params, ranking: str = "dv", shares_fn=None):
        super().__init__(cfg, universe, held, history_fn, p, ranking=ranking, shares_fn=shares_fn)
        self.ind.update(pr.rebound_indicators(self.close, p))


# ── 하루 진행 ─────────────────────────────────────────────────────────────────


def init_state(world: World, start, fx: float, year_days_fn, p: pr.Params) -> dict:
    state = {"start_date": str(pd.Timestamp(start).date()), "start_fx": fx, "last_date": None, "top10": None,
             "ledgers": {}, "randoms": [], "rng": None}
    day = e31.build_day(state, world, start, fx, year_days_fn)
    state["ledgers"] = {MAIN: pr.new_ledger(MAIN, day, p), QQQ: pr.new_qqq_ledger(QQQ, day, p)}
    state["randoms"] = [pr.new_ledger(f"r{k}", day, p, random=True) for k in range(p.n_random)]
    state["rng"] = np.random.default_rng(p.seed).bit_generator.state
    return state


def _v31_rebased(v31: dict, start: str, date: str) -> float | None:
    """B = 같은 기준일의 v3.1 세후 평가금액(사전 등록 2-2 비교 B).
    두 장부 시작일이 같으면 그대로 읽고, 다르면 v3.1 세후(date) × 1억 ÷ v3.1 세후(이 장부 시작일). 없으면 None."""
    b = v31.get(date)
    if b is None:
        return None
    if v31 and min(v31) == start:  # v3.1 첫 기록일 = v3.1 시작일
        return b
    a = v31.get(start)
    if not a:
        return None
    return b * pv.Params().start_krw / a


def refresh_v31(rows: list[dict], v31: dict, start: str) -> list[dict]:
    """저장된 줄들의 비교 B(v31)를 지금 v3.1 기록으로 다시 계산 → 값이 바뀐 줄만 고친 사본으로 반환."""
    fixed = []
    for r in rows:
        b = _v31_rebased(v31, start, r["date"])
        if b != r.get("v31"):
            fixed.append({**r, "v31": b})
    return fixed


def advance_day(state, day: pv.Day, world: World, checker: e31.FilingChecker, p: pr.Params, v31: dict) -> dict:
    """하루 진행 → 그날 기록 한 줄(dict)."""
    win_start = world.cal[max(0, world.pos(day.date) - (p.filing_window - 1))]

    def filing(tk):
        return checker.check(tk, win_start, day.date)

    L = state["ledgers"]
    ev = pr.step(L[MAIN], day, p, filing=filing)
    pr.step(L[QQQ], day, p)
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    for R in state["randoms"]:
        pr.step(R, day, p, filing=filing, random_n=ev["entries"], rng=rng)
    state["rng"] = rng.bit_generator.state
    state["last_date"] = str(day.date.date())
    dist = pv.percentiles([R["last_value"] for R in state["randoms"]])
    nav = pv.nav_usd(L[MAIN], day)
    date = str(day.date.date())
    return {"date": date, "value": L[MAIN]["last_value"], "qqq": L[QQQ]["last_value"],
            "v31": _v31_rebased(v31, state["start_date"], date), "random": dist, "weights": pv.weights(L[MAIN], day),
            "qqq_weight": L[MAIN]["q_units"] * day.qqq / nav if nav else 1.0,
            "fills": ev["fills"], "sells": ev["sells"], "skipped": ev["skipped"], "failed": sorted(set(ev["failed"])),
            "buys": ev["buys"], "waiting": ev["waiting"], "opened": ev["opened"], "unqualified": ev["unqualified"],
            "realized_year": L[MAIN]["realized"].get(str(day.date.year), 0.0),
            "mdd": {MAIN: L[MAIN]["mdd"], QQQ: L[QQQ]["mdd"]}, "top10": list(day.top10), **e31.rank_fields(state, day)}


# ── 표시 ─────────────────────────────────────────────────────────────────────


def _rsi(v) -> str:
    return f"{v:.1f}" if v is not None else "?"


def candidate_lines(r: dict, personal: bool, phrases: dict | None = None) -> list[str]:
    """■ 웅덩이 후보 — 그날 이 장부의 단계별 한 줄(개인판), 단체방판은 개수만(종목명 없음)."""
    from notify import letter_plain as lp

    ph = phrases if phrases is not None else lp.load_phrases()
    day = r.get("date")
    buys, waiting, opened = r.get("buys") or [], r.get("waiting") or [], r.get("opened") or []
    skipped, unq = r.get("skipped") or [], r.get("unqualified") or []
    if not personal:
        if not (buys or waiting or opened):
            return [lp.phrase(ph, "candidates.none", day)]
        return [lp.phrase(ph, "candidates.group", day, a=len(buys), b=len(waiting), c=len(opened))]
    lines = [lp.phrase(ph, "candidates.buy", day, 종목=b["tk"] + ("(2차)" if b["tranche"] == 2 else ""), rsi=_rsi(b["rsi"]))
             for b in buys]
    lines += [lp.phrase(ph, "candidates.waiting", day, 종목=w["tk"] + ("(2차)" if w["tranche"] == 2 else ""), n=w["day"],
                        rsi=_rsi(w["rsi"])) for w in waiting if w["tk"] not in opened]
    lines += [lp.phrase(ph, "candidates.opened", day, 종목=t) for t in opened]
    if skipped:
        lines.append(lp.phrase(ph, "candidates.skipped", day, 종목=", ".join(sorted({s["tk"] for s in skipped}))))
    if unq:
        lines.append(lp.phrase(ph, "candidates.unqualified", day, 종목=", ".join(unq)))
    return lines or [lp.phrase(ph, "candidates.none", day)]


def lab_lines(rows: list[dict], personal: bool, phrases: dict | None = None, book: RBook = BOOK) -> list[str]:
    """■ 가상 실험 칸(v3.1 줄 아래) 장부 한 줄 — "웅덩이 반등 확인: QQQ와 같습니다. …"(보유·그날 매매는 v3.1과 같은 문구 규칙)."""
    from notify import letter_plain as lp

    ph = phrases if phrases is not None else lp.load_phrases()
    parts = e31._plain_lines([{**r, "v31": r["value"], "skipped": r.get("skipped", [])} for r in rows], personal, ph)
    return [lp.ledger_line(ph, book.name_key, parts, rows[-1].get("date"))]


def letter_lines(rows: list[dict], book: RBook = BOOK) -> dict:
    """진행한 날들 → {personal, group(■ 가상 실험 줄), cand_personal, cand_group(■ 웅덩이 후보), vs_v31_pp(비교 B 대비), b_label, check_*, shares_*}."""
    if not rows:
        return {"personal": [], "group": [], "cand_personal": [], "cand_group": [], "vs_v31_pp": None, "b_label": book.b_label,
                "check_personal": None, "check_group": None, "shares_personal": None, "shares_group": None}
    last = rows[-1]
    failed = sorted({t for r in rows for t in r["failed"]})
    return {"personal": lab_lines(rows, True, book=book), "group": lab_lines(rows, False, book=book),
            "cand_personal": candidate_lines(last, True), "cand_group": candidate_lines(last, False),
            "vs_v31_pp": (last["value"] - last["v31"]) / 1_000_000 if last.get("v31") is not None else None, "b_label": book.b_label,
            "check_personal": f"실적 확인 실패({', '.join(failed)})" if failed else None,
            "check_group": f"실적 확인 실패({len(failed)}종목)" if failed else None, **e31.shares_checks(rows)}


def sheet_row(r: dict) -> list:
    hold = ", ".join(f"{t} {x * 100:.1f}%" for t, x in sorted(r["weights"].items(), key=lambda kv: -kv[1])) or "없음"
    tr = e31._trade_text([r], True)
    waiting = ", ".join(f"{w['tk']}{'(2차)' if w['tranche'] == 2 else ''} {w['day']}일째" for w in r.get("waiting") or [])
    v31 = r.get("v31")
    return [r["date"], round(r["value"]), round(r["qqq"]), round(v31) if v31 is not None else "",
            round((r["value"] - r["qqq"]) / 1_000_000, 2), round((r["value"] - v31) / 1_000_000, 2) if v31 is not None else "",
            round((r["value"] - r["random"]["50"]) / 1_000_000, 2), f"{r['qqq_weight'] * 100:.1f}%", hold,
            tr.replace("오늘 가상 매매: ", "") if tr else "", waiting, round(r["realized_year"])]


# ── 후보 종목 차트 자료 (2026-10-26, 개인판 HTML 전용) ────────────────────────


def chart_targets(rows: list[dict], most: int = 4) -> list[dict]:
    """마지막 날 「반등 확인 대기」·「매수」 종목 → [{tk, tranche, stage, day, rsi}] 대기일 짧은 순 최대 most개.
    매수 종목의 대기일 = 전날 대기 줄의 day + 1(전날 기록이 없으면 1)."""
    if not rows:
        return []
    last, prev = rows[-1], (rows[-2] if len(rows) > 1 else {})
    before = {(w["tk"], w["tranche"]): w["day"] for w in prev.get("waiting") or []}
    out = [{"tk": w["tk"], "tranche": w["tranche"], "stage": "wait", "day": w["day"], "rsi": w.get("rsi")}
           for w in last.get("waiting") or []]
    out += [{"tk": b["tk"], "tranche": b["tranche"], "stage": "buy", "day": before.get((b["tk"], b["tranche"]), 0) + 1,
             "rsi": b.get("rsi")} for b in last.get("buys") or []]
    return sorted(out, key=lambda t: (t["day"], t["tk"]))[:most]


def candidate_data(world: World, as_of, rows: list[dict], p: pr.Params, lookback: int = 120, most: int = 4) -> list[dict]:
    """후보 차트용 자료(그리기만): 종목마다 최근 lookback거래일 종가·60·120·200일선·RSI(장부 World 지표 그대로)·
    MACD·시그널·골든크로스(pr.macd_cross — 장부와 같은 함수)·RSI 40 재돌파 날(pr.rsi_cross_up)·웅덩이 칠할 구간·이 장부 매매."""
    as_of = pd.Timestamp(as_of)
    i = world.pos(as_of)
    win = world.cal[max(0, i - lookback + 1): i + 1]
    out = []
    for t in chart_targets(rows, most):
        tk = t["tk"]
        if tk not in world.close:
            continue
        s = world.close[tk].dropna()
        m, sig, cross = pr.macd_cross(s[s.index <= as_of])
        rsi, rsi_prev = world.ind["rsi"][tk].reindex(win), world.ind["rsi_prev"][tk].reindex(win)
        recross = [d for d in win if pr.rsi_cross_up(rsi_prev.get(d), rsi.get(d), p.rsi_level)]
        trades = [{"date": pd.Timestamp(f.get("date") or r["date"]), "kind": "buy2" if f.get("tranche") == 2 else "buy1"}
                  for r in rows for f in r.get("fills") or [] if f["tk"] == tk]
        trades += [{"date": pd.Timestamp(x.get("exit_date") or r["date"]), "kind": "sell"}
                   for r in rows for x in r.get("sells") or [] if x["tk"] == tk]
        out.append({**t, "as_of": as_of, "level": p.rsi_level,
                    "close": world.close[tk].reindex(win), "sma60": world.ind["sma60"][tk].reindex(win),
                    "sma120": world.ind["sma120"][tk].reindex(win), "sma200": world.ind["sma200"][tk].reindex(win),
                    "rsi_line": rsi, "macd": m.reindex(win), "signal": sig.reindex(win),
                    "golden": [d for d in win if cross.get(d) == 1.0], "recross": recross,
                    "puddle": (world.cal[max(0, i - t["day"] + 1)], as_of),
                    "trades": [x for x in trades if win[0] <= x["date"] <= win[-1]]})
    return out


# ── 실행 ─────────────────────────────────────────────────────────────────────


def run(cfg: dict, as_of, *, db_path=None, dry_run: bool = False, history_fn=None, universe_fn=None, fx_fn=None,
        year_days_fn=None, filings_fn=None, sheets_client=None, sheet_write: bool | None = None, p: pr.Params = pr.Params(),
        book: RBook = BOOK, shares_fn=None) -> dict:
    """하루 실행. dry_run=True면 DB·시트에 쓰지 않는다(로컬은 늘 미리보기). sheet_write 기본값은 GitHub Actions에서만 True.
    book = 장부(반등 확인 보관본·v2), shares_fn = 주식 수(시총 판, 테스트 주입).
    출력: {status, start_date, rows, letter, warnings, errors, diag}"""
    pc = cfg.get(book.cfg_key) or {}
    as_of = pd.Timestamp(as_of).normalize()
    history_fn = history_fn or e31._default_history
    year_days_fn = year_days_fn or e31._default_year_days
    fx_fn = fx_fn or e31._default_fx
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
        held = {tk for L in (e31._all_ledgers(state) if state else []) for tk in L["pos"]}
        universe = (universe_fn or e31._default_universe)()
        world = World(cfg, universe, held, history_fn, p, ranking=book.ranking, shares_fn=shares_fn)
        if as_of not in world.cal:
            as_of = world.cal[world.cal <= as_of][-1]
        if world.missing:
            result["warnings"].append(f"가격 없는 구성 종목 {len(world.missing)}개: {', '.join(world.missing[:10])}")
        checker = e31.FilingChecker(filings_fn)
        fresh = state is None
        if fresh:
            start = pd.Timestamp(pc.get("start_date") or as_of).normalize()
            fx0 = float(pc["start_fx"]) if pc.get("start_fx") else fx_fn([start.date()]).get(str(start.date()))
            if fx0 is None:
                raise RuntimeError(f"시작일 {start.date()} 환율이 없습니다")
            state = init_state(world, start, float(fx0), year_days_fn, p)
            todo = [start]
        else:
            todo = list(world.cal[(world.cal > pd.Timestamp(state["last_date"])) & (world.cal <= as_of)])
        result["start_date"] = state["start_date"]
        v31 = v31_values(conn, book)
        fixed = [] if fresh else refresh_v31(daily_history(conn, book), v31, state["start_date"])
        if not todo:
            result["status"] = "already"
            if fixed and not dry_run and in_actions:
                save(conn, state, fixed, [], book)
                _write_sheet(result, pc, sheets_client, sheet_write, fixed, book)
            hist = _merge(daily_history(conn, book), fixed)
            result["letter"] = letter_lines(hist[-1:], book) if hist else letter_lines([], book)
            result["candidates"] = _safe_candidates(world, as_of, hist, p)
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
            day = e31.build_day(state, world, d, float(fx), year_days_fn)
            r = advance_day(state, day, world, checker, p, v31)
            rows.append(r)
            trades += [(MAIN, t) for t in r["sells"]]
            if pd.Timestamp(d) >= e31.judgment_date(state["start_date"]) and not state.get("judged"):
                state["judged"] = {"date": r["date"], **pr.judge(r["value"], r["qqq"], r["v31"], r["random"]["80"],
                                                                  r["mdd"][MAIN], r["mdd"][QQQ])}
        result["rows"] = rows
        result["letter"] = letter_lines(rows, book)
        result["diag"] = diagnose(world, checker, as_of, state, rows[-1])
        result["candidates"] = _safe_candidates(world, as_of, _merge(daily_history(conn, book), fixed + rows), p)
        if dry_run or not in_actions:
            result["status"] = "preview"
            return result
        save(conn, state, fixed + rows, trades, book)
        if state.get("judged"):
            vbstore.set_meta(conn, book.meta_judgment, json.dumps(state["judged"], ensure_ascii=False))
        _write_sheet(result, pc, sheets_client, sheet_write, fixed + rows, book)
        return result
    finally:
        conn.close()


def _safe_candidates(world, as_of, rows, p) -> list[dict]:
    """후보 차트 자료 — 실패해도 장부·편지는 그대로(빈 목록)."""
    try:
        return candidate_data(world, as_of, rows, p)
    except Exception as exc:  # noqa: BLE001
        print(f"[puddle_rebound] 후보 차트 자료를 만들지 못했습니다: {type(exc).__name__}: {exc}")
        return []


def _merge(hist: list[dict], fixed: list[dict]) -> list[dict]:
    """저장된 줄에 고친 줄을 날짜로 덮어쓴 목록(미리보기용)."""
    by = {r["date"]: r for r in hist}
    by.update({r["date"]: r for r in fixed})
    return [by[d] for d in sorted(by)]


def _write_sheet(result: dict, pc: dict, sheets_client, sheet_write: bool, rows: list[dict], book: RBook = BOOK) -> None:
    """장부 탭(「웅덩이 반등 확인 (가상)」·v2는 「웅덩이 반등 확인 v2 (가상·시총)」)에 rows를 날짜 줄로 덮어쓰기/붙이기. 실패는 result에 적는다."""
    if not (sheet_write and rows):
        return
    sheet_id = (os.environ.get(pc.get("sheet_id_env", "GOOGLE_SHEETS_ID")) or "").strip()
    if not sheet_id:
        result["warnings"].append(f"GOOGLE_SHEETS_ID가 없어 {book.label} 탭 기록을 건너뜁니다")
        return
    try:
        psheet = importlib.import_module(book.sheet_module)

        if sheets_client is None:
            from data import sheets

            sheets_client = sheets.get_client()
        psheet.write_rows(sheets_client, sheet_id, [sheet_row(r) for r in rows])
    except Exception as exc:  # noqa: BLE001 — 장부는 저장됨, 시트만 실패
        result["errors"].append(f"{book.label} 시트 기록 실패: {type(exc).__name__}")


def diagnose(world: World, checker: e31.FilingChecker, d, state, row: dict) -> list[dict]:
    """오늘 상위 10 진단(미리보기·보고용): 자격·웅덩이 단계·RSI·MACD·최근 실적 접수일."""
    d = pd.Timestamp(d)
    top = state["top10"]["list"] if state.get("top10") else world.top10(d)
    closes, ind = world.tickers_ind(d, top)
    buys = {b["tk"] for b in row.get("buys") or []}
    waiting = {w["tk"]: w for w in row.get("waiting") or []}
    opened = set(row.get("opened") or [])
    skipped = {s["tk"] for s in row.get("skipped") or []}
    caps = (state.get("top10") or {}).get("caps") or {}
    out = []
    for t in top:
        x, i = closes.get(t), ind.get(t) or {}
        qual = pv._qualified(i, x)
        if t in buys:
            stage = "반등 확인 — 가상 매수 신호"
        elif t in opened:
            stage = "웅덩이 진입(오늘 열림)"
        elif t in waiting:
            stage = f"반등 확인 대기 {waiting[t]['day']}일째"
        elif t in skipped:
            stage = "실적 발표 직후라 제외"
        elif not qual:
            stage = "자격 미달"
        else:
            stage = "웅덩이 아님(60일선 위)" if x >= i.get("sma60", np.nan) else "60일선 아래(열림 조건 아님)"
        out.append({"ticker": t, "stage": stage, "close": x, "cap": caps.get(t), "qualified": qual, "sma60": i.get("sma60"), "sma200": i.get("sma200"),
                    "rsi": i.get("rsi"), "rsi_prev": i.get("rsi_prev"), "gc3": i.get("gc3"), "last_filing": checker.last_filing(t, d)})
    return out


def run_safely(cfg: dict, as_of, book: RBook = BOOK, **kwargs) -> dict | None:
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


def main(argv=None, book: RBook = BOOK) -> None:
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
