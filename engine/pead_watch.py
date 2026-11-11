"""실적급등 추천 장부(pead_watch) 하루 실행 — 앞으로 기록, 실제 돈 없음 (사전 등록: docs/prereg/pead_watch.md, 태그 pead-watch-prereg).

순서: 사전 등록 해시 확인 → 달 말 명단 → EDGAR 8-K 2.02(직전 발표 60일 뒤) → 색인 Accepted → 반응일 → 거래일마다
(매수 체결 → 매도 → 그날 반응일 신호 판정·대기 매수) → 환율 → 보유 평가 → 편지 칸·그래프 자료.
쓰기는 GitHub Actions 예약 실행만(dry_run·로컬·수동 실행은 미리보기). run_safely는 어떤 예외도 밖으로 내보내지 않는다(편지 보호).

미리보기: python -m engine.pead_watch --from 2026-10-01 --as-of 2026-11-10   (DB에 쓰지 않음)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from core import pead_timing as pt
from core import pead_watch as pw
from store import pead_watch as st
from store import virtual_books as vbstore

ROOT = Path(__file__).resolve().parents[1]
PREREG_YAML = "configs/pead_watch.yaml"
PREREG_FILES = {  # LF 정규화 sha256 — 태그 pead-watch-prereg 커밋 내용
    PREREG_YAML: "82e8f331809dae25b2e0dbfc4eea6b913a564432e594801db438134aafe41b22",
    "docs/prereg/pead_watch.md": "0c4f03cc58983bedaa715c8b6398ea2fcfb72b63284d461affa4a73dc3462fe9",
}
FILING_LOOKBACK_DAYS = 7  # 진행할 첫 날보다 이만큼 앞선 접수부터 반응일을 본다(주말·연휴 장후 접수)
CHART_LOOKBACK_CLOSED = 40  # 끝난 매매도 매도일이 기준일에서 이 거래일 안이면 그래프에 남긴다
NAME_KR_CSV = ROOT / "data" / "name_kr.csv"


def verify_prereg(root: Path = ROOT) -> dict:
    """사전 등록 파일 해시 확인(작업 폴더 = 기록값). 다르면 RuntimeError. 출력: configs/pead_watch.yaml 내용."""
    for rel, want in PREREG_FILES.items():
        got = hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            raise RuntimeError(f"사전 등록 파일이 태그와 다름: {rel}")
    return yaml.safe_load((root / PREREG_YAML).read_text(encoding="utf-8"))


def write_allowed(dry_run: bool) -> bool:
    """쓰기 = GitHub Actions 예약(schedule) 실행만."""
    return (not dry_run and os.environ.get("GITHUB_ACTIONS") == "true"
            and os.environ.get("GITHUB_EVENT_NAME") == "schedule")


# ── 입력 ─────────────────────────────────────────────────────────────────────


@dataclass
class Sources:
    """네트워크 입력 묶음(테스트에서 가짜로 바꾼다)."""

    current_ndx: Callable[[], pd.DataFrame]          # ticker, name, name_kr
    ndx_at: Callable[[pd.Timestamp, set], tuple]     # (집합, 출처)
    sp100_at: Callable[[pd.Timestamp], tuple]
    cik_map: Callable[[], dict]
    recent_8k: Callable[[int], list]
    accepted: Callable[[int, str], pd.Timestamp]
    prices: Callable[[str], pd.DataFrame]            # open, close, volume
    fx: Callable[[date, date], dict]
    earnings: Callable[[list], dict]
    calendar: Callable[[date, date], pd.DatetimeIndex]


def live_sources(cfg: dict) -> Sources:
    from data import earnings as dearn
    from data import edgar
    from data import fx as dfx
    from data import market_calendar as mc
    from data import pead_watch_data as pd_
    from data.prices import fetch_one
    from data.universe import get_universe

    ua_box: list[str] = []

    def ua() -> str:
        if not ua_box:
            ua_box.append(edgar.get_user_agent())
        return ua_box[0]

    def fx(a: date, b: date) -> dict:
        h = dfx.fetch_usd_krw_range(a, b)
        if not h:
            from data.fx import fetch_fred_series_range
            h = fetch_fred_series_range("DEXKOUS", a.isoformat(), b.isoformat())
        return h

    return Sources(
        current_ndx=get_universe,
        ndx_at=pd_.ndx_at,
        sp100_at=pd_.sp100_at,
        cik_map=lambda: edgar.fetch_ticker_to_cik(ua()),
        recent_8k=lambda cik: pd_.recent_8k(cik, ua()),
        accepted=lambda cik, accn: pd_.accepted_et(cik, accn, ua()),
        prices=lambda t: fetch_one(t, cfg),
        fx=fx,
        earnings=lambda tick: dearn.get_earnings_dates(tick, max_workers=8),
        calendar=mc.trading_days_between,
    )


def _name_kr() -> dict[str, str]:
    if not NAME_KR_CSV.exists():
        return {}
    d = pd.read_csv(NAME_KR_CSV, dtype=str).fillna("")
    return {t.strip(): n.strip() for t, n in zip(d["ticker"], d["name_kr"]) if n.strip()}


def _parallel(jobs: dict, workers: int = 6) -> dict:
    """{키: 인자 없는 함수} → {키: 결과 또는 예외}. SEC 요청 간격은 data/edgar 속도 제한(스레드 안전)이 지킨다."""
    from concurrent.futures import ThreadPoolExecutor

    def one(fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — 모아 보고
            return exc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return dict(zip(jobs, ex.map(one, jobs.values())))


def _cik_key(t: str) -> str:
    return t.replace("-", ".")  # SEC 티커 표(BRK.B) ↔ yfinance(BRK-B)


# ── 실행 ─────────────────────────────────────────────────────────────────────


def _empty_letter() -> dict:
    return {"actions": {"buy": [], "sell": []}, "new": [], "watching": [], "closed": [], "scorecards": {},
            "earnings": {"rows": [], "failed": 0}, "chart": None, "counts": {"new": 0, "watching": 0}}


def run(cfg: dict, as_of, *, db_path=None, dry_run: bool = False, src: Sources | None = None, start: str | None = None,
        today_kst=None, root: Path = ROOT) -> dict:
    """하루 실행. as_of = 직전 확정 미국 종가일. start = 미리보기 시작일(주면 DB 메타 대신 이 날부터 진행).

    출력: {status('ok'|'preview'|'already'), start_date, as_of, letter{…}, judged[판정 행], warnings, errors, timing}
    """
    t0 = time.monotonic()
    pre = verify_prereg(root)
    p = pw.params_from(pre)
    src = src or live_sources(cfg)
    write = write_allowed(dry_run)
    conn = vbstore.connect(Path(db_path or vbstore.VBOOKS_DB_PATH))
    st.ensure_schema(conn)
    res = {"status": "ok", "warnings": [], "errors": [], "judged": [], "letter": _empty_letter(), "timing": {}}
    try:
        as_of = pd.Timestamp(as_of).normalize()
        days = src.calendar((as_of - pd.Timedelta(days=520)).date(), (as_of + pd.Timedelta(days=420)).date())
        as_of = days[days <= as_of][-1]
        ia = int(days.get_loc(as_of))
        if start:
            start_d, last = pd.Timestamp(start), None
        else:
            start_d = pd.Timestamp(st.get_meta(conn, st.META_START) or (cfg.get("pead_watch") or {}).get("start_date") or as_of)
            last = st.get_meta(conn, st.META_LAST)
        lo = pd.Timestamp(last) if last else start_d - pd.Timedelta(days=1)
        todo = list(days[(days > lo) & (days >= start_d) & (days <= as_of)])
        res.update(start_date=start_d.date().isoformat(), as_of=as_of.date().isoformat(), todo=[d.date().isoformat() for d in todo])
        if not todo:
            res["status"] = "already"

        # 1) 달 말 명단 (진행할 날들의 전 달 말 + 기준일 다음 거래일의 전 달 말 — 실적 예정 칸)
        need = sorted({pw.prev_month_end(d) for d in todo} | {pw.prev_month_end(days[ia + 1])})
        members, new_members, cur = {}, [], None
        for me in need:
            key = me.date().isoformat()
            m = st.get_members(conn, key)
            if m is None:
                if cur is None:
                    u = src.current_ndx()
                    cur = (set(u["ticker"]), u)
                ndx, s1 = src.ndx_at(me, cur[0])
                sp, s2 = src.sp100_at(me)
                m = {"ndx": ndx, "sp100": sp}
                new_members.append((key, ndx, s1, sp, s2))
            members[key] = m
        res["timing"]["members"] = round(time.monotonic() - t0, 1)
        names = _name_kr()
        if cur is not None:
            for t, n, k in zip(cur[1]["ticker"], cur[1].get("name", cur[1]["ticker"]), cur[1].get("name_kr", [""] * len(cur[1]))):
                names.setdefault(t, k or n)

        # 2) 실적 발표 → 반응일 후보
        t1 = time.monotonic()
        trades = st.load_trades(conn)
        todo_set = {d.date().isoformat() for d in todo}
        cands, new_filings, gaps = [], [], {"cik": [], "edgar": [], "accepted": []}
        if todo:
            tick = sorted(set().union(*[m["ndx"] | m["sp100"] for k, m in members.items()
                                        if k in {pw.prev_month_end(d).date().isoformat() for d in todo}]))
            ciks = src.cik_map()
            since = (todo[0] - pd.Timedelta(days=FILING_LOOKBACK_DAYS)).date().isoformat()
            cik_of = {}
            for t in tick:
                cik = ciks.get(_cik_key(t)) or ciks.get(t)
                if cik is None:
                    gaps["cik"].append(t)
                else:
                    cik_of[t] = int(cik)
            lists = _parallel({t: (lambda c=c: src.recent_8k(c)) for t, c in cik_of.items()})
            need_acc = {}
            for t, fl in lists.items():
                if isinstance(fl, Exception):
                    gaps["edgar"].append(f"{t}({type(fl).__name__})")
                    continue
                for f in pw.counted_announcements(fl, p.min_gap_days):
                    if f["filing_date"] >= since:
                        need_acc[(t, f["accn"])] = f
            cached = {k: st.get_accepted(conn, k[1]) for k in need_acc}
            fetched = _parallel({k: (lambda c=cik_of[k[0]], a=k[1]: src.accepted(c, a)) for k, v in cached.items() if v is None})
            for (t, accn), f in sorted(need_acc.items(), key=lambda kv: (kv[1]["filing_date"], kv[0])):
                acc = cached[(t, accn)]
                if acc is None:
                    got = fetched[(t, accn)]
                    if isinstance(got, Exception):
                        gaps["accepted"].append(f"{t} {accn}({type(got).__name__})")
                        continue
                    acc = str(got)
                    new_filings.append({"accn": accn, "ticker": t, "cik": cik_of[t], "filing_date": f["filing_date"],
                                        "accepted_et": acc})
                d0 = pw.reaction_day(pd.Timestamp(acc), days)
                if d0 is None or d0.date().isoformat() not in todo_set:
                    continue
                for nf in new_filings:
                    if nf["accn"] == accn:
                        nf["d0"] = d0.date().isoformat()
                m = members[pw.prev_month_end(d0).date().isoformat()]
                label = pw.membership_label(t, m["ndx"], m["sp100"])
                if label:
                    cands.append({"ticker": t, "d0": d0.date().isoformat(), "accn": accn, "label": label, "accepted_et": acc})
        for k, v in gaps.items():
            if v:
                res["warnings"].append({"cik": "CIK 없음", "edgar": "EDGAR 접수 목록 실패", "accepted": "Accepted 실패"}[k]
                                       + f" {len(v)}개: {', '.join(v[:8])}")
        res["timing"]["edgar"] = round(time.monotonic() - t1, 1)

        # 3) 가격
        t2 = time.monotonic()
        held = {t["ticker"] for t in trades if t["status"] in ("pending", "open")}
        want = sorted({c["ticker"] for c in cands} | held | {"QQQ"})
        raw, failed = {}, []
        for t, v in _parallel({t: (lambda t=t: src.prices(t)) for t in want}, workers=4).items():
            if isinstance(v, Exception):
                failed.append(f"{t}({type(v).__name__})")
            else:
                raw[t] = v
        if "QQQ" not in raw:
            raise RuntimeError("QQQ 가격 없음")
        if failed:
            res["warnings"].append(f"가격 실패 {len(failed)}개: {', '.join(failed[:8])}")
        raw = {t: d.loc[:as_of] for t, d in raw.items()}  # 기준일 뒤 행은 쓰지 않는다
        al = {t: pt.align(d, days) for t, d in raw.items()}
        bench = al["QQQ"]
        res["timing"]["prices"] = round(time.monotonic() - t2, 1)

        # 4) 거래일마다 진행
        judged = []
        by_ledger = {lg: [t for t in trades if t["ledger"] == lg] for lg, _ in p.ledgers}
        old_closed = {(t["ledger"], t["ticker"], t["d0"]): t for t in trades if t["status"] in ("closed", "skipped")}
        for d in todo:
            ds = d.date().isoformat()
            today_sig = []
            for c in (c for c in cands if c["d0"] == ds):
                px = raw.get(c["ticker"])
                j = pw.judge(px, raw["QQQ"]["close"], d, p) if px is not None else None
                row = {**c, "ret": None, "excess": None, "vol_ratio": None, "signal": None, "judged_as_of": res["as_of"],
                       "status": "판정 불가"}
                if j:
                    row.update(ret=j["ret"], excess=j["excess"], vol_ratio=j["vol_ratio"], signal=int(j["signal"]),
                               status="신호" if j["signal"] else "조건 미달")
                judged.append(row)
                if j and j["signal"]:
                    today_sig.append(row)
            for lg, hold in p.ledgers:
                by_ledger[lg], w = pw.step_day(by_ledger[lg], d, today_sig, al, bench, days, lg, hold)
                if d == todo[-1]:
                    res["warnings"] += sorted(set(w))
        res["judged"] = judged

        # 5) 환율 (새로 체결·매도된 매매)
        allt = [t for lg in by_ledger.values() for t in lg]
        need_fx = sorted({t["buy_day"] for t in allt if t["status"] in ("open", "closed") and not t.get("fx_buy")}
                         | {t["sell_day"] for t in allt if t["status"] == "closed" and not t.get("fx_sell")})
        if need_fx:
            try:
                from data.fx import pick_rate_for_date

                h = src.fx(date.fromisoformat(need_fx[0]) - timedelta(days=10), as_of.date())
                for t in allt:
                    if t["status"] in ("open", "closed") and not t.get("fx_buy"):
                        r = pick_rate_for_date(h, t["buy_day"])
                        t["fx_buy"] = r[0] if r else None
                    if t["status"] == "closed" and not t.get("fx_sell"):
                        r = pick_rate_for_date(h, t["sell_day"])
                        t["fx_sell"] = r[0] if r else None
            except Exception as exc:  # noqa: BLE001 — 환율은 다음 실행에 다시
                res["warnings"].append(f"환율 실패({type(exc).__name__}) — 다음 실행에서 채움")
        for t in allt:  # 끝난 매매는 한 번 정해지면 그대로
            k = (t["ledger"], t["ticker"], t["d0"])
            if k in old_closed:
                o = old_closed[k]
                if (o["status"], o.get("sell_day"), o.get("buy_px")) != (t["status"], t.get("sell_day"), t.get("buy_px")):
                    res["warnings"].append(f"끝난 매매 값이 달라짐 {k} — 기록값 유지")
                t.update({c: o.get(c) for c in st.TRADE_COLS if o.get(c) is not None})

        # 6) 편지 칸 (카드의 "새 추천" 태그는 지난 실행의 판정값도 쓴다 — 읽기만)
        res["signal_hist"] = [x for x in st.load_signals(conn) if x.get("signal")]
        res["letter"] = build_letter(by_ledger, judged, al, raw, bench, days, as_of, names, p, members, src, today_kst,
                                     need[-1].date().isoformat(), res)
        res["timing"]["total"] = round(time.monotonic() - t0, 1)

        if not write:
            if res["status"] != "already":
                res["status"] = "preview"
            return res
        for key, ndx, s1, sp, s2 in new_members:
            st.put_members(conn, key, "ndx", ndx, s1)
            st.put_members(conn, key, "sp100", sp, s2)
        for f in new_filings:
            st.put_filing(conn, f)
        st.put_signals(conn, judged)
        st.replace_trades(conn, allt)
        if not st.get_meta(conn, st.META_START):
            st.set_meta(conn, st.META_START, start_d.date().isoformat())
        if todo:
            st.set_meta(conn, st.META_LAST, res["as_of"])
            st.put_daily(conn, res["as_of"], {"todo": res["todo"], "judged": len(judged),
                                              "signals": [j["ticker"] for j in judged if j["signal"]],
                                              "counts": res["letter"]["counts"], "warnings": res["warnings"][:10]})
        conn.commit()
        return res
    finally:
        conn.close()


def _row_name(t: str, names: dict) -> str:
    return names.get(t) or t


def build_letter(by_ledger, judged, al, raw, bench, days, as_of, names, p, members, src, today_kst, me_key, res) -> dict:
    """편지 칸 자료: 오늘 할 일, 새 추천, 관찰 중, 실적 예정, 끝난 종목·성적표, 그래프 자료, 개수(단체방판)."""
    ia = int(days.get_loc(as_of))
    nxt = days[ia + 1]
    main_hold = dict(p.ledgers)["main"]
    cmp_hold = dict(p.ledgers).get("compare", 60)
    main = by_ledger["main"]
    L = _empty_letter()
    L["next_day"] = nxt.date().isoformat()
    L["as_of"] = as_of.date().isoformat()

    def info(t):
        return {"ticker": t["ticker"], "name": _row_name(t["ticker"], names), "label": t.get("label")}

    for t in main:
        if t["status"] == "pending" and t["buy_day"] == L["next_day"]:
            L["actions"]["buy"].append(info(t))
        if t["status"] == "open" and t["plan_day"] == L["next_day"]:
            L["actions"]["sell"].append(info(t))
    for j in judged:
        if not j["signal"]:
            continue
        mt = next((t for t in main if t["ticker"] == j["ticker"] and t["d0"] == j["d0"]), None)
        ib = days.searchsorted(pd.Timestamp(j["d0"]), side="right")
        L["new"].append({"ticker": j["ticker"], "name": _row_name(j["ticker"], names), "label": j["label"], "d0": j["d0"],
                         "ret": j["ret"], "excess": j["excess"], "vol_ratio": j["vol_ratio"],
                         "buy_day": days[ib].date().isoformat(),
                         "check_day": days[ib + cmp_hold].date().isoformat(),
                         "sell_day": days[ib + main_hold].date().isoformat(),
                         "late": j["d0"] < L["as_of"], "skip": (mt or {}).get("skip")})
    for t in main:
        if t["status"] != "open":  # 매수 대기는 ② 새 추천에만(같은 종목 두 번 금지)
            continue
        ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
        row = {**info(t), "d0": t["d0"], "buy_day": t["buy_day"], "plan_day": t["plan_day"],
               "check_day": days[ib + cmp_hold].date().isoformat(), "status": t["status"], "hold": main_hold,
               "held": 0, "ret": None, "vs_qqq": None}
        if t["status"] == "open":
            row.update({k: v for k, v in pw.mark(t, al.get(t["ticker"]), bench, days, as_of).items() if k in ("held", "ret", "vs_qqq")})
        L["watching"].append(row)
    closed = sorted((t for t in main if t["status"] == "closed"), key=lambda t: (t["sell_day"], t["ticker"]), reverse=True)
    for t in closed[:5]:
        x = pw.closed_excess(t)
        L["closed"].append({**info(t), "buy_day": t["buy_day"], "sell_day": t["sell_day"], "ret": x["ret"], "vs_qqq": x["ex_pre"]})
    for lg, hold in p.ledgers:
        if lg in ("main", "compare"):
            L["scorecards"][lg] = {"hold": hold, **pw.scorecard(by_ledger[lg])}
    L["counts"] = {"new": len(L["new"]), "watching": len(L["watching"])}

    # 실적 예정 (이번 주·다음 주, 기준일 다음 거래일의 전 달 말 명단)
    m = members.get(me_key) or {}
    tick = sorted((m.get("ndx") or set()) | (m.get("sp100") or set()))
    today = pd.Timestamp(today_kst or datetime.now().date()).normalize()
    wk0 = today - pd.Timedelta(days=today.dayofweek)
    wk2 = wk0 + pd.Timedelta(days=13)
    te = time.monotonic()
    try:
        ed = src.earnings(tick) if tick else {}
        rows = [{"ticker": t, "name": _row_name(t, names), "label": pw.membership_label(t, m.get("ndx", set()), m.get("sp100", set())),
                 "date": pd.Timestamp(d).date().isoformat()} for t, d in ed.items()
                if d is not None and wk0 <= pd.Timestamp(d) <= wk2]
        L["earnings"] = {"rows": sorted(rows, key=lambda r: (r["date"], r["ticker"])), "failed": sum(1 for d in ed.values() if d is None),
                         "week0": wk0.date().isoformat(), "week2": wk2.date().isoformat()}
    except Exception as exc:  # noqa: BLE001
        L["earnings"] = {"rows": [], "failed": len(tick), "error": type(exc).__name__}
    res["timing"]["earnings"] = round(time.monotonic() - te, 1)

    sigs = {(j["ticker"], j["d0"]): j for j in (res.get("signal_hist") or [])}
    sigs.update({(j["ticker"], j["d0"]): j for j in judged})
    L["chart"] = chart_data(main, al, bench, days, ia, names, raw, sigs, main_hold, cmp_hold)
    return L


CARD_PAD_BEFORE = 20  # 카드 기간: 매수일 이만큼 거래일 전부터 기준일까지
CARD_NEW_DAYS = 5  # 매수일이 기준일 또는 이 거래일 안이면 "새 추천" 카드(맨 앞)
CARD_SOON_DAYS = 5  # 매도일까지 이 거래일 이하면 새 추천 다음 자리


def _qqq_ohlc(raw_q: pd.DataFrame | None, ds: pd.DatetimeIndex) -> dict:
    """QQQ 일봉(시가·고가·저가·종가) — 고가·저가 칸이 없으면 시가·종가의 큰 값·작은 값. 없는 날은 None."""
    d = (raw_q if raw_q is not None else pd.DataFrame(columns=["open", "close"])).reindex(ds)
    o, c = d["open"].astype(float), d["close"].astype(float)
    h = d["high"].astype(float) if "high" in d else pd.concat([o, c], axis=1).max(axis=1)
    lo = d["low"].astype(float) if "low" in d else pd.concat([o, c], axis=1).min(axis=1)
    f = lambda x: [None if not np.isfinite(v) else float(v) for v in x.to_numpy(dtype=float)]  # noqa: E731
    return {"open": f(o), "high": f(h), "low": f(lo), "close": f(c)}


def chart_data(main: list[dict], al: dict, bench: pt.Aligned, days: pd.DatetimeIndex, ia: int, names: dict,
               raw: dict | None = None, signals: dict | None = None, hold: int = 90, check: int = 60,
               labels: dict | None = None) -> dict | None:
    """그래프 자료(계산은 여기서만, notify/pead_chart는 그리기만).

    lines: 보유 중 + 최근 끝난(CHART_LOOKBACK_CLOSED) 매매. 선마다 매수일부터 날짜별 매수가 대비 %(values)와
      QQQ 대비 초과 %p(excess — 그 종목 매수일 QQQ 시가 기준, 마지막 값 = vs_qqq), 60거래일 점검 위치(check_at, dates 안 칸 번호),
      보유 거래일 수(held)/hold.
    qqq: 가장 이른 매수일부터 QQQ % 선, qqq_pct: 같은 기간 QQQ %(가장 이른 보유 중 매수일 시가 → 기준일 종가, 보유 없으면 전체 선).
    cards: 보유 중인 모든 종목(끝난 종목 없음). 순서 = 새 추천(매수일이 기준일 또는 CARD_NEW_DAYS 거래일 안) →
      매도일까지 CARD_SOON_DAYS 거래일 이하 → 나머지(매도일 가까운 순). 기간 = 매수일 CARD_PAD_BEFORE 거래일 전 ~ 기준일.
    선이 하나도 없으면 None.
    """
    raw, signals, labels = raw or {}, signals or {}, labels or {}
    lines, cards = [], []
    for t in main:
        if t["status"] not in ("open", "closed") or t.get("buy_px") is None:
            continue
        a = al.get(t["ticker"])
        if a is None:
            continue
        ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
        ie = int(days.get_loc(pd.Timestamp(t["sell_day"]))) if t["status"] == "closed" else ia
        if t["status"] == "closed" and ia - ie > CHART_LOOKBACK_CLOSED:
            continue
        idx = np.arange(ib, ie + 1)
        v = a.close[a.lastv[idx]] / t["buy_px"] - 1
        q0 = t.get("q_buy") or bench.open[ib]  # 이 종목 매수일 QQQ 시가 (이름표의 QQQ 대비)
        q = bench.close[bench.lastv[idx]] / q0 - 1
        ex = v - q
        name = _row_name(t["ticker"], names)
        lines.append({"ticker": t["ticker"], "name": name, "status": t["status"],
                      "dates": [days[i].date().isoformat() for i in idx], "values": [float(x) * 100 for x in v],
                      "excess": [float(x) * 100 for x in ex],
                      "vs_qqq": float(ex[-1]) * 100 if np.isfinite(ex[-1]) else None,
                      "check_at": check if ib + check <= ie else None, "held": int(ie - ib), "hold": hold})
        if t["status"] != "open":
            continue
        ip = int(days.get_loc(pd.Timestamp(t["plan_day"])))
        i0 = max(0, ib - CARD_PAD_BEFORE)
        cidx = np.arange(i0, ia + 1)
        ds = days[i0:ia + 1]
        sig = signals.get((t["ticker"], t["d0"])) or {}
        new = ia - ib <= CARD_NEW_DAYS
        last_c = float(a.close[a.lastv[ia]])
        cards.append({
            "ticker": t["ticker"], "name": name, "label": t.get("label") or labels.get(t["ticker"]),
            "kind": "new" if new else ("soon" if ip - ia <= CARD_SOON_DAYS else "hold"),
            "d0": t["d0"], "sig_ret": sig.get("ret"), "sig_excess": sig.get("excess"), "sig_vol": sig.get("vol_ratio"),
            "dates": [d.date().isoformat() for d in ds], "buy_at": int(ib - i0), "buy_day": t["buy_day"],
            "buy_px": float(t["buy_px"]), "q_buy": float(q0), "qqq": _qqq_ohlc(raw.get("QQQ"), ds),
            "close": [None if a.lastv[i] < 0 else float(a.close[a.lastv[i]]) for i in cidx],
            "ret": float(v[-1]) * 100, "qqq_ret": float(q[-1]) * 100,
            "vs_qqq": float(ex[-1]) * 100 if np.isfinite(ex[-1]) else None, "last_close": last_c,
            "last_qqq": float(bench.close[bench.lastv[ia]]),
            "held": int(ia - ib), "hold": hold, "plan_day": t["plan_day"], "sell_in": int(ip - ia),
        })
    if not lines:
        return None
    rank = {"new": 0, "soon": 1, "hold": 2}
    cards.sort(key=lambda c: (rank[c["kind"]], -c["buy_at"] if c["kind"] == "new" else 0, c["plan_day"], c["ticker"]))
    first = min(int(days.get_loc(pd.Timestamp(x["dates"][0]))) for x in lines)
    idx = np.arange(first, ia + 1)
    q = bench.close[idx] / bench.open[first] - 1
    opens = [int(days.get_loc(pd.Timestamp(x["dates"][0]))) for x in lines if x["status"] == "open"]
    f0 = min(opens) if opens else first
    qp = bench.close[bench.lastv[ia]] / bench.open[f0] - 1
    return {"lines": lines, "cards": cards, "qqq_pct": float(qp) * 100 if np.isfinite(qp) else None,
            "qqq": {"dates": [days[i].date().isoformat() for i in idx], "values": [float(x) * 100 for x in q]}}


def run_safely(cfg: dict, as_of, **kwargs) -> dict | None:
    """run을 감싼다 — 예외를 밖으로 내보내지 않는다(편지 보호). config pead_watch.enabled가 false면 돌지 않는다."""
    if not (cfg.get("pead_watch") or {}).get("enabled", False):
        return None
    try:
        res = run(cfg, as_of, **kwargs)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[pead_watch] 오류(편지는 그대로): {type(exc).__name__}: {exc}")
        return {"status": "error", "error": type(exc).__name__, "letter": _empty_letter(), "warnings": [], "errors": [str(exc)]}
    for w in res.get("warnings", []):
        print(f"[pead_watch] 경고: {w}")
    print(f"[pead_watch] {res['status']} · 진행 {len(res.get('todo') or [])}일 · 판정 {len(res.get('judged') or [])} · 시간 {res['timing']}")
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="실적급등 추천 장부 미리보기(DB에 쓰지 않음)")
    ap.add_argument("--from", dest="start", default=None, help="이 날부터 진행(빈 임시 DB) — 예: 2026-10-01")
    ap.add_argument("--as-of", default=None)
    a = ap.parse_args(argv)
    from data.market_calendar import latest_closed_trading_day
    from engine.daily import US_EASTERN, load_config

    cfg = load_config()
    as_of = a.as_of or latest_closed_trading_day(datetime.now(US_EASTERN))
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "preview.db" if a.start else None
        res = run(cfg, as_of, db_path=db, dry_run=True, start=a.start)
    sig = [j for j in res["judged"] if j["signal"]]
    print(f"[pead_watch] {res['status']} · {res['start_date']} ~ {res['as_of']} · 판정 {len(res['judged'])} · 신호 {len(sig)} · 시간 {res['timing']}")
    for w in res["warnings"]:
        print("  경고:", w)
    L = res["letter"]
    for n in L["new"]:
        print(f"  {n['ticker']:6} {n['name']} [{n['label']}] 반응일 {n['d0']} {n['ret']:+.1%} QQQ대비 {n['excess']:+.1%} "
              f"거래량 {n['vol_ratio']:.1f}배 · 매수 {n['buy_day']} · 점검 {n['check_day']} · 매도 {n['sell_day']}")


if __name__ == "__main__":
    main()
