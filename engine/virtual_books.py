"""가상 장부 5개 — 매일 실행 (docs/design/virtual_books.md).

실제 돈은 QQQM만. QQQ·QLD·TQQQ 그냥 보유와 QLD·TQQQ 200일선 전환(아래면 SGOV)을 각 가상 1억 원으로
같은 날 시작해 관찰만 한다. 실제 주문 기능은 없다.

흐름 (engine/daily.py live 실행 뒤, 브리핑 만들기 전):
  1) QQQ·QLD·TQQQ·SGOV 확정 종가·분배금 받기 (data/virtual_books_data.py)
  2) 시작일부터 다시 계산 (core/virtual_books.py — 순수 함수, 같은 입력이면 같은 결과)
  3) data/virtual_books.db에 일별·체결 표를 통째로 바꿔 쓰고 run_id·기준일로 실행 기록
  4) 구글 시트 세 탭 통째로 다시 쓰기 (data/virtual_books_sheet.py)
  5) 브리핑 맨 위 국면 한 줄을 돌려준다 → 브리핑 발송 뒤 전환 신호·낙폭 알림(한 번만)

같은 날 두 번 실행해도 중복이 없다: 표는 통째로 바꾸고, 알림은 alerts 표로 한 번만 보낸다.
장부 오류는 run_safely가 잡아 따로 알림만 보내고 None을 돌려준다 — 기존 브리핑은 그대로 나간다.

단독 실행 (사본 시트 시험 등):
    python -m engine.virtual_books              계산·DB·시트·알림
    python -m engine.virtual_books --no-send    텔레그램 알림만 보내지 않음
    python -m engine.virtual_books --dry-run    DB·시트에 쓰지 않고 결과만 출력
    python -m engine.virtual_books --db PATH    다른 DB 파일 사용
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import virtual_books as vbc  # noqa: E402
from data import virtual_books_data as vbdata  # noqa: E402
from data import virtual_books_sheet as vbsheet  # noqa: E402
from notify import virtual_books as vbtext  # noqa: E402
from store import virtual_books as vbstore  # noqa: E402

ALERT_SWITCH = "switch"
ALERT_DRAWDOWN = "drawdown"
SWITCH_ALERT_WINDOW = 5  # 실행을 며칠 걸렀을 때 늦게라도 보낼 전환 신호 범위(최근 거래일 수)

NOTES = [
    "가격: 분할 반영 종가(배당 미반영). 운용보수는 ETF 가격에 반영돼 따로 빼지 않음",
    "체결: 정수 주식, 수수료 편도 0.07%, 슬리피지 0. 전환 신호는 QQQ t일 종가 vs 200일 단순이동평균 → t+1 거래일 종가 체결",
    "분배금: 원천징수 15% 뒤 현금으로 쌓음(재투자 안 함)",
    "세금: 그해 마지막 거래일 실현 순이익 × 22%(공제·이월결손 무시), 장부 현금에서 빼지 않고 따로 누적",
    "청산 가정 세후 추정: 오늘 종가로 전량 판다고 보고(매도 수수료 포함) 올해 실현 + 미실현 이익에 22%",
]


def _now_kst() -> datetime:
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("Asia/Seoul"))


def _default_fx(start_date: date) -> tuple[float | None, str | None, str | None]:
    from data import fx

    r = fx.get_usd_krw_rate(start_date)
    return r.rate, r.rate_date, r.warning


def _next_trading_day(d: pd.Timestamp) -> pd.Timestamp | None:
    from data.market_calendar import trading_days_between

    days = trading_days_between((d + timedelta(days=1)).date(), (d + timedelta(days=10)).date())
    return pd.Timestamp(days[0]) if len(days) else None


def compute(cfg: dict, conn, *, now_et: datetime | None = None, history_fn=None, fx_fn=None,
            year_end_fn=None) -> dict:
    """가격을 받아 장부를 계산한다 (DB 읽기만 — 시작일·환율 기억값).

    출력: {status("ok"|"not_started"), books, ledger, regime, start, base_date, start_fx, fx_date,
          initial_usd, params, warnings}
    """
    vb = cfg["virtual_books"]
    books = vbc.books_from_config(vb)
    tickers = vbc.required_tickers(books, vbc.params_from_config(vb, 1.0))
    warnings: list[str] = []

    start_cfg = vb.get("start_date")
    start_meta = vbstore.get_meta(conn, "start_date")
    if start_cfg and start_meta and str(start_cfg) != start_meta:
        warnings.append(f"설정 시작일 {start_cfg}이 DB 기억값 {start_meta}과 달라 설정값을 씁니다")
    known_start = str(start_cfg) if start_cfg else start_meta

    anchor = pd.Timestamp(known_start).date() if known_start else (now_et or datetime.now()).date()
    closes, divs = vbdata.fetch_closes_and_dividends(
        tickers, anchor - timedelta(days=vbdata.WARMUP_CALENDAR_DAYS), now_et=now_et, history_fn=history_fn
    )
    base_date = closes.index[-1]
    start = pd.Timestamp(known_start) if known_start else base_date
    if start > base_date:
        return {"status": "not_started", "start": start, "base_date": base_date, "warnings": warnings, "books": books}

    if vb.get("start_fx"):
        start_fx, fx_date = float(vb["start_fx"]), "설정값"
    elif vbstore.get_meta(conn, "start_fx"):
        start_fx, fx_date = float(vbstore.get_meta(conn, "start_fx")), vbstore.get_meta(conn, "start_fx_date") or ""
    else:
        rate, fx_date, fx_warning = (fx_fn or _default_fx)(start.date())
        if rate is None:
            raise RuntimeError(f"시작일 {start.date()} 원/달러 환율을 구하지 못했습니다 ({fx_warning})")
        if fx_warning:
            warnings.append(fx_warning)
        start_fx = float(rate)

    initial_usd = float(vb["start_krw"]) / start_fx
    params = vbc.params_from_config(vb, initial_usd)
    year_ends = (year_end_fn or vbdata.year_end_trading_days)(start.year, base_date.year)
    ledger = vbc.run_ledger(books, closes, divs, start, params, year_ends, int(vb["min_days_for_annualized"]))
    return {
        "status": "ok", "books": books, "ledger": ledger, "regime": vbc.regime_today(ledger, books),
        "start": ledger.start, "base_date": ledger.base_date, "start_fx": start_fx, "fx_date": fx_date,
        "initial_usd": initial_usd, "params": params, "warnings": warnings,
    }


def pending_alerts(conn, cfg: dict, ctx: dict, next_day_fn=None) -> list[dict]:
    """아직 안 보낸 알림 목록: 최근 전환 신호(SWITCH_ALERT_WINDOW 거래일 안), 지정 장부 첫 낙폭 돌파.

    출력: [{kind, key, text}]
    """
    vb = cfg["virtual_books"]
    ledger, books, params = ctx["ledger"], ctx["books"], ctx["params"]
    out = []
    idx = ledger.regime.index
    recent = set(idx[max(0, len(idx) - SWITCH_ALERT_WINDOW):])
    switch_books = [b for b in books if b.kind == vbc.SMA_SWITCH]
    for sig in ledger.switches:
        key = sig["signal_date"].date().isoformat()
        if sig["signal_date"] not in recent or vbstore.has_alert(conn, ALERT_SWITCH, key):
            continue
        moves = []
        for b in switch_books:
            held = ledger.books[b.name].daily.at[sig["signal_date"], "holding"]
            moves.append((b.name, held, b.asset if sig["above"] else params.defensive_ticker))
        next_day = sig["fill_date"] or (next_day_fn or _next_trading_day)(sig["signal_date"])
        out.append({"kind": ALERT_SWITCH, "key": key,
                    "text": vbtext.switch_alert_text(sig, moves, next_day, params.signal_ticker)})

    book, pct = vb.get("drawdown_alert_book"), vb.get("drawdown_alert_pct")
    if book and pct is not None and book in ledger.books:
        daily = ledger.books[book].daily
        breach = vbc.first_drawdown_breach(daily, float(pct))
        key = f"{book}|{float(pct):g}"
        if breach is not None and not vbstore.has_alert(conn, ALERT_DRAWDOWN, key):
            out.append({"kind": ALERT_DRAWDOWN, "key": key,
                        "text": vbtext.drawdown_alert_text(book, daily, breach, float(pct))})
    return out


def sheet_tables(cfg: dict, ctx: dict, updated_at: str) -> dict[str, list[list]]:
    """시트 세 탭 값."""
    vb = cfg["virtual_books"]
    ledger, books = ctx["ledger"], ctx["books"]
    meta = {
        "start_krw": float(vb["start_krw"]), "start_fx": ctx["start_fx"], "fx_date": ctx["fx_date"],
        "initial_usd": ctx["initial_usd"], "updated_at": updated_at,
        "min_days_for_annualized": int(vb["min_days_for_annualized"]), "notes": NOTES,
    }
    return {
        vbsheet.TAB_DAILY: vbtext.daily_table(ledger, books, vb["benchmark_book"]),
        vbsheet.TAB_TRADES: vbtext.trades_table(ledger, books),
        vbsheet.TAB_SUMMARY: vbtext.summary_table(ledger, meta),
    }


def run(cfg: dict, *, db_path: Path | None = None, dry_run: bool = False, sheets_client=None,
        now_et: datetime | None = None, history_fn=None, fx_fn=None, year_end_fn=None, next_day_fn=None) -> dict:
    """가상 장부 하루 실행 (알림은 아직 보내지 않는다 — send_alerts가 브리핑 뒤에 보낸다).

    출력: {run_id, status, base_date, start, regime_line, alerts, errors(시트 등 부분 실패), warnings,
          ledger, sheet_rows, db_path}
    예외: 가격·계산 실패는 그대로 올린다(run_safely가 잡는다). 실행 기록에는 error로 남긴다.
    """
    vb = cfg["virtual_books"]
    db_path = Path(db_path or vbstore.VBOOKS_DB_PATH)
    if (os.environ.get("GITHUB_ACTIONS") == "true" and not vb.get("start_date")
            and not (os.environ.get("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS") or "").strip()):
        raise RuntimeError("GOOGLE_DRIVE_STATE_FILE_ID_VBOOKS가 없고 시작일도 고정돼 있지 않아 "
                           "매 실행이 새 장부로 시작됩니다 — 드라이브 state 파일을 연결하거나 config 시작일을 고정하세요")

    run_id = uuid.uuid4().hex[:12]
    run_at = _now_kst().isoformat(timespec="seconds")
    conn = vbstore.connect(db_path)
    try:
        try:
            ctx = compute(cfg, conn, now_et=now_et, history_fn=history_fn, fx_fn=fx_fn, year_end_fn=year_end_fn)
        except Exception as exc:
            if not dry_run:
                vbstore.record_run(conn, run_id, "", run_at, "error", f"{type(exc).__name__}: {exc}")
            raise
        base_str = ctx["base_date"].date().isoformat()
        result = {"run_id": run_id, "status": ctx["status"], "base_date": ctx["base_date"], "start": ctx["start"],
                  "regime_line": None, "alerts": [], "errors": [], "warnings": list(ctx["warnings"]),
                  "ledger": ctx.get("ledger"), "sheet_rows": {}, "db_path": db_path, "dry_run": dry_run,
                  "same_day_runs": len(vbstore.runs_for_base_date(conn, base_str))}
        if ctx["status"] == "not_started":
            result["warnings"].append(f"가상 장부 시작일 {ctx['start'].date()} 전이라 계산하지 않았습니다")
            if not dry_run:
                vbstore.record_run(conn, run_id, base_str, run_at, "not_started")
            return result

        params = ctx["params"]
        result["regime_line"] = vbtext.regime_line(
            ctx["regime"], params.signal_ticker, params.defensive_ticker,
            [b.asset for b in ctx["books"] if b.kind == vbc.SMA_SWITCH],
        )
        result["alerts"] = pending_alerts(conn, cfg, ctx, next_day_fn)
        tables = sheet_tables(cfg, ctx, run_at)
        result["tables"] = tables

        if dry_run:
            return result

        if vbstore.get_meta(conn, "start_date") is None or vb.get("start_date"):
            vbstore.set_meta(conn, "start_date", ctx["start"].date().isoformat())
        if vbstore.get_meta(conn, "start_fx") is None or vb.get("start_fx"):
            vbstore.set_meta(conn, "start_fx", repr(ctx["start_fx"]))
            vbstore.set_meta(conn, "start_fx_date", str(ctx["fx_date"] or ""))
        daily_rows = [{**row, "date": d} for b in ctx["books"]
                      for d, row in ctx["ledger"].books[b.name].daily.to_dict("index").items()]
        trades = [t for b in ctx["books"] for t in ctx["ledger"].books[b.name].trades]
        vbstore.replace_ledger(conn, daily_rows, trades)

        sheet_id = vbsheet.sheet_id_from_env(vb["sheet_id_env"])
        if sheet_id is None:
            result["warnings"].append(f"{vb['sheet_id_env']}가 없어 가상 장부 시트 기록을 건너뜁니다")
        else:
            try:
                if sheets_client is None:
                    from data import sheets

                    sheets_client = sheets.get_client()
                result["sheet_rows"] = vbsheet.write_tables(sheets_client, sheet_id, tables)
            except Exception as exc:  # 장부·DB는 이미 갱신 — 시트만 실패로 따로 알린다
                result["errors"].append(f"가상 장부 시트 기록 실패: {type(exc).__name__}: {exc}")

        vbstore.record_run(conn, run_id, base_str, run_at, "ok" if not result["errors"] else "partial",
                           "; ".join(result["errors"]))
        return result
    finally:
        conn.close()


def send_alerts(result: dict, *, force_no_send: bool = False, sender=None) -> list[dict]:
    """run 결과의 알림을 보낸다. 실제로 보낸 것만 alerts 표에 기록한다(같은 날 다시 실행해도 한 번만).

    출력: 보낸 알림 목록
    """
    if not result or not result.get("alerts") or result.get("dry_run"):
        return []
    if sender is None:
        from notify import telegram

        sender = telegram.send_alert_text
    sent = []
    conn = vbstore.connect(result["db_path"])
    try:
        for a in result["alerts"]:
            if vbstore.has_alert(conn, a["kind"], a["key"]):
                continue
            if sender(a["text"], force_no_send=force_no_send):
                vbstore.record_alert(conn, a["kind"], a["key"], _now_kst().isoformat(timespec="seconds"))
                sent.append(a)
    finally:
        conn.close()
    return sent


def _ops_alert(message: str, force_no_send: bool, ops_sender=None) -> None:
    print(f"[virtual_books] {message}")
    if force_no_send:
        return
    try:
        if ops_sender is None:
            from notify import telegram

            ops_sender = telegram.notify_ops_error
        ops_sender(message)
    except Exception as exc:  # 알림 실패가 브리핑을 막지 않게
        print(f"[virtual_books] 오류 알림도 보내지 못했습니다: {type(exc).__name__}: {exc}")


def run_safely(cfg: dict, *, force_no_send: bool = False, ops_sender=None, **kwargs) -> dict | None:
    """run을 감싼다 — 어떤 예외도 밖으로 내보내지 않는다(기존 브리핑 보호). 오류는 따로 알림.

    출력: run 결과, 실패하면 None
    """
    if not (cfg.get("virtual_books") or {}).get("enabled", False):
        return None
    try:
        result = run(cfg, **kwargs)
    except Exception as exc:
        traceback.print_exc()
        _ops_alert(f"가상 장부 오류 (기존 브리핑은 그대로 나감): {type(exc).__name__}: {exc}", force_no_send, ops_sender)
        return None
    for w in result.get("warnings", []):
        print(f"[virtual_books] 경고: {w}")
    if result.get("errors"):
        _ops_alert("가상 장부 부분 오류 (장부·브리핑은 정상): " + "; ".join(result["errors"]), force_no_send, ops_sender)
    return result


def send_alerts_safely(result: dict | None, *, force_no_send: bool = False, sender=None, ops_sender=None) -> list[dict]:
    """send_alerts를 감싼다 — 예외를 밖으로 내보내지 않는다."""
    try:
        return send_alerts(result, force_no_send=force_no_send, sender=sender)
    except Exception as exc:
        _ops_alert(f"가상 장부 알림 발송 오류: {type(exc).__name__}: {exc}", force_no_send, ops_sender)
        return []


def main(argv: list[str] | None = None) -> None:
    for stream in (sys.stdout, sys.stderr):  # 윈도우 콘솔 cp949 UnicodeEncodeError 방지
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="가상 장부 5개 하루 실행")
    parser.add_argument("--no-send", action="store_true", help="텔레그램 알림을 보내지 않는다")
    parser.add_argument("--dry-run", action="store_true", help="DB·시트에 쓰지 않는다")
    parser.add_argument("--db", default=None, help="DB 파일 경로 (기본 data/virtual_books.db)")
    args = parser.parse_args(argv)

    from engine.daily import load_config

    cfg = load_config()
    result = run_safely(cfg, force_no_send=args.no_send, dry_run=args.dry_run,
                        db_path=Path(args.db) if args.db else None)
    if result is None:
        sys.exit(1)
    print(f"run_id {result['run_id']} · 기준일 {result['base_date'].date()} · 시작일 {result['start'].date()} "
          f"· 같은 기준일 이전 실행 {result['same_day_runs']}회")
    print(result["regime_line"] or "(국면 줄 없음)")
    if result.get("ledger") is not None:
        for s in result["ledger"].summaries:
            print(f"  {s['book']}: ${s['value']:,.2f} ({s['cum_return_pct']:+.2f}%) 보유 {s['holding']}")
    print(f"시트: {result['sheet_rows'] or '안 씀'}")
    sent = send_alerts_safely(result, force_no_send=args.no_send)
    for a in result["alerts"]:
        print(f"알림({a['kind']}, {'보냄' if a in sent else '안 보냄'}):\n{a['text']}")


if __name__ == "__main__":
    main()
