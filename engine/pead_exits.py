"""실적급등 매도 조건 비교(pead_exits) 하루 진행 — engine/pead_watch.run 안에서 V1 장부를 진행한 뒤 부른다.

사전 등록: docs/prereg/pead_exits.md, configs/pead_exits.yaml (태그 pead-exits-prereg). V1.0 규칙·편지는 그대로.
순서: 사전 등록 해시 확인 → 시작일(DB 고정) → 새 신호의 A 예정일(yfinance → 8-K 간격 추정 → 기본 60일, 처음 값 고정)·
가이던스·전날 200일선 → 거래일마다 A·B 진행 → 환율 → 15칸 포트폴리오 V1·A·B.
실패는 밖으로 내보내지 않고 경고로(V1 장부·편지 보호). 쓰기는 save()가 예약 실행에서만(engine/pead_watch가 판단).
"""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

from core import pead_exits as px
from core import pead_tags as ptag
from core import pead_watch as pw
from store import pead_exits as xst
from store import pead_watch as st

ROOT = Path(__file__).resolve().parents[1]
PREREG_YAML = "configs/pead_exits.yaml"
PREREG_FILES = {  # LF 정규화 sha256 — 태그 pead-exits-prereg 커밋 내용
    PREREG_YAML: "9f0fceae3fd9edf89b41d6968414e9f27596b40a11483519fa881ebdcad8115d",
    "docs/prereg/pead_exits.md": "739b556990582c7250d67d4e7ca2f539ce111ccc0cfaaa27b9f097e0621ce265",
}
RULES = ("exit_a", "exit_b")


def verify_prereg(root: Path = ROOT) -> dict:
    """사전 등록 파일 해시 확인. 다르면 RuntimeError. 출력: configs/pead_exits.yaml 내용."""
    for rel, want in PREREG_FILES.items():
        got = hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if got != want:
            raise RuntimeError(f"사전 등록 파일이 태그와 다름: {rel}")
    return yaml.safe_load((root / PREREG_YAML).read_text(encoding="utf-8"))


def _cached(cache: dict, key, fn):
    from engine.pead_watch import _cached as c

    return c(cache, key, fn)


def gap_reactions(fl: list[dict], accn: str, cik: int, days: pd.DatetimeIndex, conn, src, min_gap: int,
                  max_gaps: int) -> list[pd.Timestamp]:
    """접수 목록 → 이번 발표(accn)와 그 앞 센 발표 최대 max_gaps건의 반응일(V1 3장 규칙). 달력 밖·Accepted 실패는 뺀다."""
    k = pw.counted_announcements(fl, min_gap)
    pos = next((i for i, f in enumerate(k) if f["accn"] == accn), None)
    if pos is None:
        return []
    out = []
    for f in k[max(0, pos - max_gaps):pos + 1]:
        acc = st.get_accepted(conn, f["accn"])
        if acc is None:
            try:
                acc = str(src.accepted(cik, f["accn"]))
            except Exception:  # noqa: BLE001 — 그 건만 뺀다
                continue
        t = pd.Timestamp(acc)
        if t.normalize() < days[0]:
            continue
        d = pw.reaction_day(t, days)
        if d is not None:
            out.append(d)
    return out


def make_plan(c: dict, j: dict, raw_px: pd.DataFrame | None, days, conn, src, lists: dict, cache: dict, p: px.Params,
              as_of: str, today_kst: str, min_gap: int) -> dict:
    """새 신호 하나의 A 예정일·출처와 꼬리표(가이던스·전날 200일선). 받기 실패는 다음 단계로 넘기고 errors에 사유."""
    t, d0 = c["ticker"], c["d0"]
    row = {"ticker": t, "d0": d0, "next_date": None, "source": px.SRC_DEFAULT, "n_gaps": None, "yf_dates": None,
           "fetched_as_of": as_of, "fetched_kst": today_kst, "guid_raise": None, "guid_text": None,
           "prev_below_sma200": None, "errors": None}
    errs = []
    if getattr(src, "earnings_dates", None) is None:
        errs.append("예정일: 출처 없음")
    else:
        try:
            ds = src.earnings_dates(t)
            row["yf_dates"] = json.dumps(list(ds))
            e = px.pick_next_date(ds, d0, p.min_days_after_d0)
            if e is not None:
                row.update(next_date=e.date().isoformat(), source=px.SRC_YF)
            else:
                errs.append("예정일: yfinance에 반응일+30일 뒤 날짜 없음")
        except Exception as exc:  # noqa: BLE001
            errs.append(f"예정일: yfinance {type(exc).__name__}")
    if row["next_date"] is None:
        fl = lists.get(t)
        if isinstance(fl, list) and c.get("cik") is not None:
            rd = gap_reactions(fl, c["accn"], int(c["cik"]), days, conn, src, min_gap, p.max_gaps)
            e, n = px.estimate_from_gaps(rd, p.max_gaps)
            if e is not None:
                row.update(next_date=e.date().isoformat(), source=f"{px.SRC_GAP}(간격 {n}개)", n_gaps=n)
            else:
                errs.append("8-K 간격: 앞 발표 없음")
        else:
            errs.append("8-K 간격: 접수 목록 없음")
    if raw_px is not None:
        row["prev_below_sma200"] = px.prev_below_sma200(raw_px["close"], d0)
    if src.index_html is None or src.fetch_text is None or c.get("cik") is None:
        errs.append("가이던스: 출처 없음")
    else:
        try:
            from data import pead_tags_data as tg

            cik, accn = int(c["cik"]), c["accn"]
            docs = tg.parse_index_docs(_cached(cache, ("index", cik, accn), lambda: src.index_html(cik, accn)))
            ex = ptag.pick_press_release(docs)
            if ex is None:
                errs.append("가이던스: EX-99.1 없음")
            else:
                g, s = px.guidance_raise(_cached(cache, ("text", ex["url"]), lambda: src.fetch_text(ex["url"])))
                row.update(guid_raise=g, guid_text=s)
                if g is None:
                    errs.append("가이던스: 단어 0개")
        except Exception as exc:  # noqa: BLE001
            errs.append(f"가이던스: {type(exc).__name__}")
    row["errors"] = "; ".join(errs) or None
    return row


def advance(conn, cfg: dict, *, judged: list[dict], cands: list[dict], todo: list, days: pd.DatetimeIndex, al: dict,
            raw: dict, bench, src, lists: dict, cache: dict, main: list[dict], as_of: str, today_kst: str | None,
            start_override: str | None, min_gap: int, root: Path = ROOT) -> dict:
    """A·B 하루 진행(쓰기 없음). 출력: {start, trades{규칙: 매매}, plans_new[], pf{장부: portfolio_days 결과}, counts, warnings}."""
    from engine import pead_watch as ew

    pre = verify_prereg(root)
    p = px.params_from(pre)
    xst.ensure_schema(conn)
    warn: list[str] = []
    start = start_override or xst_meta(conn) or as_of
    kst = today_kst or datetime.now().date().isoformat()
    trades = {r: xst.load_trades(conn, r) for r in RULES}
    old_closed = {(r, t["ticker"], t["d0"]): t for r in RULES for t in trades[r] if t["status"] in ("closed", "skipped")}
    plans = xst.load_plans(conn)
    cmap = {(c["ticker"], c["d0"]): c for c in cands}
    new_sig = [j for j in judged if j.get("signal") and j["d0"] >= start]
    plans_new = []
    for j in new_sig:
        k = (j["ticker"], j["d0"])
        if k in plans:
            continue
        row = make_plan(cmap.get(k, j), j, raw.get(j["ticker"]), days, conn, src, lists, cache, p, as_of, kst, min_gap)
        plans[k] = row
        plans_new.append(row)
        if row["errors"]:
            warn.append(f"매도 조건 기록 빈칸 {k[0]} {k[1]}: {row['errors']}")
    for d in todo:
        ds = d.date().isoformat()
        today = [j for j in new_sig if j["d0"] == ds]
        for r in RULES:
            trades[r], w = px.step_exit_day(trades[r], d, today, al, bench, days, r, plans if r == "exit_a" else None, p)
            if d == todo[-1]:
                warn += sorted(set(w))
    allt = [t for r in RULES for t in trades[r]]
    need_fx = sorted({t["buy_day"] for t in allt if t["status"] in ("open", "closed") and not t.get("fx_buy")}
                     | {t["sell_day"] for t in allt if t["status"] == "closed" and not t.get("fx_sell")})
    if need_fx:
        try:
            from data.fx import pick_rate_for_date

            h = src.fx(date.fromisoformat(need_fx[0]) - timedelta(days=10), date.fromisoformat(as_of))
            for t in allt:
                if t["status"] in ("open", "closed") and not t.get("fx_buy"):
                    x = pick_rate_for_date(h, t["buy_day"])
                    t["fx_buy"] = x[0] if x else None
                if t["status"] == "closed" and not t.get("fx_sell"):
                    x = pick_rate_for_date(h, t["sell_day"])
                    t["fx_sell"] = x[0] if x else None
        except Exception as exc:  # noqa: BLE001
            warn.append(f"매도 조건 환율 실패({type(exc).__name__}) — 다음 실행에서 채움")
    for r in RULES:
        for t in trades[r]:
            k = (r, t["ticker"], t["d0"])
            if k in old_closed:
                o = old_closed[k]
                if (o["status"], o.get("sell_day"), o.get("buy_px")) != (t["status"], t.get("sell_day"), t.get("buy_px")):
                    warn.append(f"끝난 매매 값이 달라짐 {k} — 기록값 유지")
                t.update({c: o.get(c) for c in xst.EXIT_COLS if o.get(c) is not None})
    books = {"main": [t for t in main if t["d0"] >= start], **trades}
    pf = {}
    pp = ew.pf_params(cfg)
    step = functools.partial(px.pf_step_slots, slots=p.slots)
    for b, tr in books.items():
        state = json.loads(xst_meta(conn, xst.pf_state_key(b)) or "null")
        try:
            pf[b] = ew.portfolio_days(cfg, tr, todo, al, bench, days, src, state, start=pd.Timestamp(start), step=step, p=pp)
        except Exception as exc:  # noqa: BLE001
            warn.append(f"15칸 포트폴리오({b}) 실패({type(exc).__name__}) — 다음 실행에서 다시")
            pf[b] = {"rows": [], "events": [], "state": None, "warnings": []}
        warn += [f"15칸({b}) {w}" for w in pf[b].get("warnings", [])]
    counts = {r: sum(1 for t in trades[r] if t["status"] == "closed" and t["d0"] >= start) for r in RULES}
    reports = milestone_reports(conn, pre, p, start, main, trades, judged, plans, as_of)
    return {"start": start, "trades": trades, "plans_new": plans_new, "plans": plans, "pf": pf, "counts": counts,
            "reports": reports, "warnings": warn}


def milestone_reports(conn, pre: dict, p: px.Params, start: str, main: list[dict], trades: dict, judged: list[dict],
                      plans: dict, as_of: str) -> list[dict]:
    """A·B 모두 끝난 매매가 50건(첫 성적)·120건(확정 판단) 이상이 된 뒤 첫 실행에서 한 번씩 계산(사전 등록 7장)."""
    have = xst.load_reports(conn)
    k = min(sum(1 for t in trades[r] if t["status"] == "closed" and t["d0"] >= start) for r in RULES)
    out = []
    for name, need in (("first", int(pre["report"]["milestones"]["first"])), ("final", int(pre["report"]["milestones"]["final"]))):
        if k < need or name in have:
            continue
        sigs = {(s["ticker"], s["d0"]): s for s in st.load_signals(conn)}
        sigs.update({(j["ticker"], j["d0"]): j for j in judged if j.get("signal")})
        rep = px.report({"main": main, **trades}, list(sigs.values()), plans, start, p)
        out.append({"milestone": name, "as_of": as_of, "counts": rep["counts"], "report": rep})
    return out


def xst_meta(conn, key: str = xst.META_START) -> str | None:
    return st.get_meta(conn, key)


def save(conn, ex: dict) -> None:
    """A·B 기록 저장(예약 실행만 — 부르는 쪽이 판단). 커밋은 부르는 쪽."""
    for r in RULES:
        xst.replace_trades(conn, r, ex["trades"][r])
    xst.put_plans(conn, ex["plans_new"])
    for r in ex.get("reports") or []:
        xst.put_report(conn, r["milestone"], r["as_of"], r["counts"], r["report"])
    for b, pf in ex["pf"].items():
        if pf.get("state") is not None:
            xst.put_portfolio_rows(conn, b, pf["rows"])
            xst.put_pf_events(conn, b, pf["events"])
            _meta(conn, xst.pf_state_key(b), json.dumps(pf["state"]))
    if not st.get_meta(conn, xst.META_START):
        _meta(conn, xst.META_START, ex["start"])


def _meta(conn, key: str, value: str) -> None:
    """메타 쓰기(커밋 없음 — engine/pead_watch의 SAVEPOINT 안에서 V1과 따로 되돌릴 수 있게)."""
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))
