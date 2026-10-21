"""가상 장부 10번 「동전주 거래량 수렴 전략」 + 그림자 「무작위 동전주 20」 매일 실행 (docs/prereg/penny_volume_squeeze.md).

- 두 장부는 같은 날 시작(첫 실행 기준일 = 첫 선정일)하고 같은 날 다시 고른다. DB 상태를 읽어 기준일까지 하루씩 이어 간다.
- 선정일(1·4·7·10월 첫 거래일, 미뤄진 선정일, 시작일)에만 Nasdaq Trader 목록 전체를 yfinance로 묶어 받는다(약 5분).
  평일에는 두 장부 보유 종목 + BIL만 받는다(종목당 yfinance 일봉 1번).
- 목록을 못 받거나 가격을 받은 종목이 절반 미만이면 그날은 선정하지 않고(다음 실행으로 미룸) notes에 남긴다 → 편지 「점검」 줄.
- 보유 종목 가격을 못 받으면 그 종목은 그날 가격 없음(5거래일 연속이면 0원). 보유 종목 절반 이상이 실패하면 네트워크 문제로 보고 멈춘다.
- 알림(10번 보유 종목, 기준일만): 하루 +30% 이상 또는 거래량 20일 평균 5배 이상 → 전략분석기 텔레그램(engine.virtual_books.send_alerts).
- 판정(시작 후 12개월 1회): 판정일 종가 기준 세 조건 — 결과를 DB meta에 남기고 전략요약 규칙 줄·경고로 보인다.
오류가 나도 다른 장부·브리핑은 그대로(장부 묶음 단위로 잡아 errors로). dry_run이면 DB에 쓰지 않는다.
선정 명단(종목명)은 이 DB(penny_picks)와 시트에만 — 저장소·편지 단체방판·보고에는 쓰지 않는다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta

import pandas as pd

from core import penny_book as pcore
from data import virtual_books_data as vbdata
from store import virtual_books as vbstore

KIND, KIND_RANDOM = "penny", "penny_random"
ALERT_KIND = "penny"
JUDGE_KEY = "penny_judgement"


class PennyStateMissingError(RuntimeError):
    """시작값이 고정돼 있는데 DB에 장부 상태가 없을 때 (새로 시작하지 않는다)."""


def penny_params(cfg: dict) -> tuple[pcore.PennyParams, pcore.PennyParams]:
    """(10번, 그림자) 설정 — 사전 등록값(config virtual_books.penny)."""
    vb, c = cfg["virtual_books"], cfg["virtual_books"]["penny"]
    p = pcore.PennyParams(
        name=str(c["name"]), slots=int(c["slots"]), min_price=float(c["min_price"]), max_price=float(c["max_price"]),
        min_dollar_volume=float(c["min_dollar_volume"]), min_history=int(c["min_history"]), short_days=int(c["short_days"]),
        long_days=int(c["long_days"]), max_ratio=float(c["max_ratio"]), max_width=float(c["max_width"]),
        half_gain=float(c["half_gain"]), gone_days=int(c["gone_days"]), start_krw=float(vb["start_krw"]),
        commission_pct=float(c["commission_pct"]), slippage_pct=float(c["slippage_pct"]), fx_spread_pct=float(c["fx_spread_pct"]),
        tax_rate_pct=float(c["tax_rate_pct"]), deduction_krw=float(c["deduction_krw"]),
        dividend_withholding_pct=float(vb["dividend_withholding_pct"]), cash_ticker=str(c["cash_ticker"]), seed=int(c["seed"]))
    return p, replace(p, name=str(c["shadow_name"]))


def _prev_day(days_index: pd.DatetimeIndex, d: pd.Timestamp) -> pd.Timestamp | None:
    pos = days_index.get_loc(d)
    return days_index[pos - 1] if pos > 0 else None


def _fx_range(start, end) -> dict:
    from data import fx

    return fx.fetch_usd_krw_range(pd.Timestamp(start).date(), pd.Timestamp(end).date())


def _select_days(book: dict, days: list[pd.Timestamp], days_index) -> set:
    """이 장부가 고를 날: 시작 전이면 첫날, 시작 뒤면 1·4·7·10월 첫 거래일 + 미뤄진 선정(select_due)이면 첫날."""
    out = set()
    if not days:
        return out
    if book["status"] == pcore.WAITING or book.get("select_due"):
        out.add(days[0])
    for d in days:
        prev = _prev_day(days_index, d)
        if book["status"] == pcore.RUNNING and prev is not None and pcore.is_select_day(d, prev):
            out.add(d)
    return out


def _alert_text(hit: dict, d: pd.Timestamp, label: str) -> str:
    parts = [f"하루 {hit['ret'] * 100:+.1f}%"]
    if hit["vol_x"] is not None:
        parts.append(f"거래량 20일 평균의 {hit['vol_x']:.1f}배")
    return "\n".join([f"[10번 {label}] {hit['ticker']} — " + " · ".join(parts) + f" ({d.month}/{d.day} 마감)",
                      f"종가 ${hit['close']:,.2f} · 가상 장부 관찰 알림, 실제 주문 아님"])


def _summary(book: dict, daily: pd.DataFrame, p: pcore.PennyParams, min_days: int, rules: str) -> dict:
    from engine import picks_books as pb

    stocks = {t: v for t, v in book["positions"].items() if t != p.cash_ticker}
    s = pb.summarize_momentum({**book, "rebalances": book["selections"], "positions": stocks}, daily, min_days)
    s["rules"] = rules
    return s


def run_penny(cfg: dict, conn, ctx: dict, *, dry_run: bool, list_fn=None, batch_fn=None, history_fn=None,
              fx_range_fn=None, calendar_fn=None) -> dict:
    """10번·그림자 장부를 기준일까지 이어 간다.

    입력: cfg, conn(가상 장부 DB), ctx(engine.virtual_books.compute 결과 — QQQ 거래일·종가·분배금·기준일), dry_run,
         list_fn() → 티커 목록(예외 PennyListError) / batch_fn(티커, 시작, 끝) → ({티커: close·volume}, {실패: 사유}, 초) /
         history_fn(티커, 시작) → DataFrame(close·volume·dividend·split) / fx_range_fn(시작, 끝) → {날짜: 원/달러} /
         calendar_fn(첫 해, 끝 해) → engine.value_book.calendar 모양 (모두 테스트 주입)
    출력: {"extra": [10번, 그림자 묶음], "errors", "warnings", "notes"(점검 줄에 붙일 말), "alerts"([{kind, key, text}]),
          "timing"({단계: 초}), "selection"({날짜: {"main": n, "shadow": n, "eligible": n, "listed": n}})}
    """
    from data import penny_universe as pu
    from engine import picks_books as pb
    from engine import value_book

    vbc = cfg["virtual_books"]
    conf = vbc.get("penny") or {}
    out = {"extra": [], "errors": [], "warnings": [], "notes": [], "alerts": [], "timing": {}, "selection": {}}
    if not conf.get("enabled", False):
        return out
    p, ps = penny_params(cfg)
    params = {p.name: p, ps.name: ps}
    try:
        import time

        base = pd.Timestamp(ctx["base_date"]).normalize()
        days_index = ctx["closes"].index
        books = {}
        for name in params:
            b = vbstore.load_picks_book(conn, name)
            mismatch = pb._start_mismatch_warning(name, conf, b)
            if mismatch:
                out["warnings"].append(mismatch)
            if b is None and conf.get("start_fx"):
                raise PennyStateMissingError(f"{name}: 시작값이 고정돼 있는데 DB에 장부 상태가 없습니다 — 새로 시작하지 않습니다")
            books[name] = b or pcore.new_book(name)
        days_of = {}
        for name, b in books.items():
            if b["status"] == pcore.WAITING:
                days_of[name] = [base] if base in days_index else []
            else:
                last = pd.Timestamp(b["last_date"])
                days_of[name] = [d for d in days_index if last < d <= base]
        all_days = sorted(set().union(*map(set, days_of.values())))
        if not all_days:
            return _finish(cfg, conn, ctx, out, books, params, {}, {}, dry_run, calendar_fn, fx_range_fn)

        # 1) 선정일 명단 (목록 전체 — 선정일에만)
        sel_days = {n: _select_days(books[n], days_of[n], days_index) for n in books}
        need = sorted(set().union(*sel_days.values()))
        sel: dict = {}
        if need:
            t0 = time.time()
            try:
                listed = (list_fn or pu.fetch_lists)()
                out["timing"]["목록"] = round(time.time() - t0, 1)
                ps_days = {s: _prev_day(days_index, s) for s in need}
                start = min(ps_days.values()) - timedelta(days=370)
                frames_u, failed_u, secs = (batch_fn or pu.download_batch)(listed, start.date(), max(ps_days.values()).date())
                out["timing"]["전체 가격"] = round(secs, 1)
                if len(frames_u) < len(listed) / 2:
                    raise pu.PennyListError(f"가격 받은 종목 {len(frames_u)}/{len(listed)}개")
                for s in need:
                    elig = pcore.eligible(frames_u, ps_days[s], p)
                    main_t, shadow_t = pcore.squeeze_pick(elig, p), pcore.random_pick(elig, s, ps)
                    sel[s] = {p.name: main_t, ps.name: shadow_t, "elig": elig}
                    out["selection"][s.date().isoformat()] = {"main": len(main_t), "shadow": len(shadow_t), "eligible": len(elig),
                                                               "listed": len(listed), "priced": len(frames_u)}
            except pu.PennyListError as exc:
                out["notes"].append("동전주 선정 못함(목록 못 받음)")
                out["warnings"].append(f"{p.name}: 선정을 다음 실행으로 미룹니다 — {exc}")

        if not sel and all(b["status"] == pcore.WAITING for b in books.values()):  # 첫 선정을 못 함 — 가격은 받지 않는다
            if not dry_run:
                for name in books:
                    vbstore.save_picks_day(conn, books[name], [], [], [])
            return _finish(cfg, conn, ctx, out, books, params, {}, {}, dry_run, calendar_fn, fx_range_fn)

        # 2) 보유·새 명단·BIL 일봉 (종목마다)
        held = set()
        for b in books.values():
            held |= {t for t in b["positions"] if t != p.cash_ticker}
        new = set()
        for s in sel.values():
            new |= set(s[p.name]) | set(s[ps.name])
        t0 = time.time()
        frames, failed = {}, []
        hist_start = (all_days[0] - timedelta(days=45)).date()
        for t in sorted(held | new | {p.cash_ticker}):
            try:
                f = (history_fn or pu.history)(t, hist_start)
                frames[t] = f[f.index <= base]
            except Exception as exc:  # noqa: BLE001 — 모아서 보고
                failed.append(f"{t}: {type(exc).__name__}")
        out["timing"]["보유 가격"] = round(time.time() - t0, 1)
        if p.cash_ticker not in frames:
            raise vbdata.VirtualBooksDataError(f"{p.name}: {p.cash_ticker} 가격을 받지 못했습니다")
        bad_held = [f for f in failed if f.split(":")[0] in held]
        if held and len(bad_held) * 2 >= len(held) and len(held) >= 2:
            raise vbdata.VirtualBooksDataError(f"{p.name}: 보유 종목 가격을 절반 넘게 받지 못했습니다({len(bad_held)}/{len(held)})")
        if failed:
            out["warnings"].append(f"{p.name}: 가격 못 받은 종목 {len(failed)}개(그날 가격 없음으로 처리)")
        kept, defer = pb.defer_incomplete_last_day(all_days, frames, sorted(held | {p.cash_ticker}), days_index)
        if defer:
            out["warnings"].append(f"{p.name}: {defer}")
            days_of = {n: [d for d in ds if d in kept] for n, ds in days_of.items()}
        fx = pd.Series((fx_range_fn or _fx_range)(all_days[0] - timedelta(days=10), base), dtype=float)
        fx.index = pd.to_datetime(fx.index)
        fx = fx.sort_index()
        cal = (calendar_fn or value_book.calendar)(all_days[0].year, base.year + 1)

        # 3) 하루씩
        new_rows: dict = {n: [] for n in books}
        new_trades: dict = {n: [] for n in books}
        for name, pp in params.items():
            b = books[name]
            for d in days_of[name]:
                f = fx[fx.index <= d]
                if f.empty:
                    raise RuntimeError(f"{name}: {d.date()} 원/달러 환율이 없습니다")
                closes = {t: float(df.at[d, "close"]) for t, df in frames.items() if d in df.index and not pd.isna(df.at[d, "close"])}
                divs = {t: float(df.at[d, "dividend"]) for t, df in frames.items() if d in df.index and df.at[d, "dividend"] > 0}
                splits = {t: float(df.at[d, "split"]) for t, df in frames.items() if d in df.index and df.at[d, "split"] > 0}
                targets = sel[d][name] if (d in sel_days[name] and d in sel) else None
                if d in sel_days[name] and d not in sel:
                    b["select_due"] = True  # 목록 실패 — 다음 실행일이 새 선정일
                b, row, trades = pcore.process_day(b, d, closes=closes, fx=float(f.iloc[-1]), dividends=divs, splits=splits, p=pp,
                                                   targets=targets, is_dec_exec=d == cal["dec_exec"].get(d.year),
                                                   is_may_first=d in cal["may_first"], is_year_end=d in cal["year_ends"])
                if row is not None:
                    new_rows[name].append(row)
                new_trades[name].extend(trades)
            books[name] = b

        # 4) 알림 (10번 보유, 기준일만)
        main = books[p.name]
        if main["status"] == pcore.RUNNING and base in days_of[p.name]:
            label = cfg["virtual_books"].get("display_names", {}).get(p.name, p.name)
            stocks = [t for t in main["positions"] if t != p.cash_ticker]
            for hit in pcore.alert_hits(frames, stocks, base):
                out["alerts"].append({"kind": ALERT_KIND, "key": f"{base.date()}:{hit['ticker']}", "text": _alert_text(hit, base, label)})

        if not dry_run:
            for name in books:
                vbstore.save_picks_day(conn, books[name], new_rows[name], new_trades[name], [])
            for s, picks in sel.items():
                if any(s in days_of[n] and books[n]["status"] == pcore.RUNNING for n in books):
                    for name in params:
                        rows = [{"rank": i + 1, "ticker": t, **{k: picks["elig"][t][k] for k in ("ratio", "width", "close")},
                                 "dollar_volume": picks["elig"][t]["dollar_volume"]} for i, t in enumerate(picks[name])]
                        vbstore.save_penny_picks(conn, name, s.date().isoformat(), rows)
        return _finish(cfg, conn, ctx, out, books, params, new_rows, new_trades, dry_run, calendar_fn, fx_range_fn)
    except Exception as exc:  # 이 장부들의 오류가 다른 장부·브리핑을 막지 않게
        out["errors"].append(f"{p.name} 오류: {type(exc).__name__}: {exc}")
        return out


def _rules(p: pcore.PennyParams, shadow: bool, judge_text: str) -> str:
    if shadow:
        return f"10번과 같은 대상·같은 날·같은 규칙, 수렴 조건 없이 무작위 {p.slots}개(시드 {p.seed}) · 판정 비교용"
    return (f"선정 전날 종가 {p.min_price:g}~{p.max_price:g}달러·20일 거래대금 {p.min_dollar_volume / 1e6:g}백만 달러·"
            f"{p.min_history}거래일 → 수렴 비율(20/120일 거래량) ≤ {p.max_ratio:g}·가격 폭 ≤ {p.max_width * 100:.0f}% 낮은 순 "
            f"{p.slots}개 · 1·4·7·10월 첫 거래일 다시 고르기 · +100% 절반 매도 · {judge_text}")


def _finish(cfg, conn, ctx, out, books, params, new_rows, new_trades, dry_run, calendar_fn, fx_range_fn) -> dict:
    """장부 묶음(시트·편지용)·판정. 시작 전 장부는 묶음 없이 경고만."""
    from engine import picks_books as pb

    vbc = cfg["virtual_books"]
    names = list(params)
    main_name, shadow_name = names[0], names[1]
    main = books[main_name]
    if main["status"] != pcore.RUNNING:
        out["warnings"].append(f"{main_name}: 첫 선정 대기 중(첫 실행 기준일에 시작)")
        return out
    daily = {}
    trades = {}
    for name in names:
        rows = vbstore.picks_rows(conn, "picks_daily", name) + (new_rows.get(name, []) if dry_run else [])
        daily[name] = pb._db_daily(rows)
        trades[name] = pb._db_trades(vbstore.picks_rows(conn, "picks_trades", name)) + (new_trades.get(name, []) if dry_run else [])
    judge_text = _judge(cfg, conn, ctx, out, books, params, daily, dry_run, fx_range_fn, calendar_fn)
    for name, kind in ((main_name, KIND), (shadow_name, KIND_RANDOM)):
        b = books[name]
        if b["status"] != pcore.RUNNING or daily[name].empty:
            continue
        pp = params[name]
        out["extra"].append({
            "name": name, "kind": kind, "start": pd.Timestamp(b["start_date"]), "daily": daily[name], "trades": trades[name],
            "summary": _summary(b, daily[name], pp, int(vbc["min_days_for_annualized"]), _rules(pp, kind == KIND_RANDOM, judge_text)),
            "positions": {t: v for t, v in b["positions"].items() if t != pp.cash_ticker},
            "meta": {"start_krw": pp.start_krw, "start_fx": b.get("start_fx"), "fx_date": str(pd.Timestamp(b["start_date"]).date()),
                     "initial_usd": b["initial_usd"]},
        })
    return out


def _judge(cfg, conn, ctx, out, books, params, daily, dry_run, fx_range_fn, calendar_fn=None) -> str:
    """판정일이 지났으면 한 번 판정해 DB meta에 남긴다. 출력: 전략요약 규칙 줄에 붙일 판정 문구."""
    names = list(params)
    main = books[names[0]]
    start = pd.Timestamp(main["start_date"])
    saved = vbstore.get_meta(conn, JUDGE_KEY)
    if saved:
        r = json.loads(saved)
        return (f"판정 {r['date']}: {'통과' if r['pass'] else '불통과'} (세후 {r['main_cum']:+.1f}% · QQQ {r['qqq_cum']:+.1f}% · "
                f"무작위 {r['shadow_cum']:+.1f}% · 최대 낙폭 {r['main_mdd']:.1f}%)")
    try:
        target = start.replace(year=start.year + 1)
    except ValueError:
        target = start.replace(year=start.year + 1, day=28)
    jd = pcore.judge_date(start, ctx["closes"].index)
    if jd is None or any(daily[n].empty or jd not in daily[n].index for n in names):
        return f"판정 예정 {target.date()} 이후 첫 거래일(시작 후 12개월 1회)"
    p = params[names[0]]
    m_row, s_row = daily[names[0]].loc[jd], daily[names[1]].loc[jd]
    main_cum = pcore.after_tax_cum_pct(m_row, main["initial_usd"])
    shadow_cum = pcore.after_tax_cum_pct(s_row, books[names[1]]["initial_usd"])
    mdd = float(daily[names[0]].loc[:jd, "drawdown_pct"].min())
    qqq_cum = qqq_compare(cfg, ctx, start, jd, p, fx_range_fn, calendar_fn)
    r = pcore.judge(main_cum, shadow_cum, qqq_cum, mdd)
    r["date"] = jd.date().isoformat()
    if not dry_run:
        vbstore.set_meta(conn, JUDGE_KEY, json.dumps(r, ensure_ascii=False))
    out["warnings"].append(f"{names[0]}: 12개월 판정 {r['date']} — {'통과' if r['pass'] else '불통과'}")
    return (f"판정 {r['date']}: {'통과' if r['pass'] else '불통과'} (세후 {main_cum:+.1f}% · QQQ {qqq_cum:+.1f}% · "
            f"무작위 {shadow_cum:+.1f}% · 최대 낙폭 {mdd:.1f}%)")


def qqq_compare(cfg: dict, ctx: dict, start: pd.Timestamp, jd: pd.Timestamp, p: pcore.PennyParams, fx_range_fn=None,
                calendar_fn=None) -> float:
    """판정용 QQQ 비교: 같은 시작일·같은 1억 원·같은 비용(슬리피지 0)·공용 세금으로 QQQ 일시 매수 → 판정일 청산 가정 세후 누적(%)."""
    from engine import value_book

    sig = str(cfg["virtual_books"]["signal_ticker"])
    q = ctx["closes"][sig]
    dv = ctx["divs"][sig] if sig in ctx["divs"] else pd.Series(0.0, index=q.index)
    fx = pd.Series((fx_range_fn or _fx_range)(start - timedelta(days=10), jd), dtype=float)
    fx.index = pd.to_datetime(fx.index)
    fx = fx.sort_index()
    cal = (calendar_fn or value_book.calendar)(start.year, jd.year)
    qp = pcore.qqq_compare_params(p, sig)
    b = pcore.new_book(qp.name)
    row = None
    for d in [d for d in q.index if start <= d <= jd]:
        f = fx[fx.index <= d]
        b, row, _ = pcore.process_day(b, d, closes={sig: float(q.at[d])}, fx=float(f.iloc[-1]),
                                      dividends={sig: float(dv.at[d])} if d in dv.index and dv.at[d] > 0 else {}, splits={},
                                      p=qp, targets=[sig] if b["status"] == pcore.WAITING else None,
                                      is_dec_exec=d == cal["dec_exec"].get(d.year), is_may_first=d in cal["may_first"],
                                      is_year_end=d in cal["year_ends"])
    return pcore.after_tax_cum_pct(row, b["initial_usd"])
