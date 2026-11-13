"""전략공장 시트 실적급등 탭 2개(「일별 성적」·「매매이력」)의 줄 만들기 — 순수 함수 (2026-11-13, 사용자 지시).

네트워크·파일·DB·현재 시각 접근 없음. 입력은 가상 장부 DB에서 읽은 행(포트폴리오·사건·매매·신호)과 달력 배열,
출력은 시트에 그대로 쓸 줄(list). 판정·매매 규칙(pead-watch-prereg)은 바꾸지 않고 기록을 보여 주기만 한다.
실제 계좌 정보는 입력에도 출력에도 없다.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from core import pead_timing as pt

DAILY_HEADER = ["날짜", "평가금액(원)", "QQQ 보유(원)", "QQQ 대비(%p)", "QQQ 비중", "보유 종목 수", "보유 종목", "오늘 매매",
                "올해 실현 이익(원)",
                # 매도 조건 비교 15칸 포트폴리오(docs/prereg/pead_exits.md 6장, 2026-11-13) — 시작일 전은 빈칸
                "15칸 V1 평가(원)", "15칸 V1 QQQ 대비(%p)", "15칸 A 평가(원)", "15칸 A QQQ 대비(%p)",
                "15칸 B 평가(원)", "15칸 B QQQ 대비(%p)"]
SLOT_BOOKS = ("main", "exit_a", "exit_b")
TRADE_HEADER = ["반응일", "티커", "회사명", "소속", "반응일 상승률", "QQQ 대비(%p)", "거래량 배수", "매수일", "매수가($)",
                "중간 점검일(60)", "매도 예정일(90)", "매도일", "매도가($)", "수익률", "같은 기간 QQQ", "QQQ 대비(%p)",
                "60일 시점 QQQ 대비(%p)", "상태", "꼬리표 요약",
                # 매도 조건 비교 A·B(docs/prereg/pead_exits.md, 2026-11-13) — A·B 시작 전 신호는 빈칸
                "A 매도 예정일", "A 결과", "B 목표가($)", "B 매도일", "B 결과", "B 목표 도달"]
TRADE_KEY = (1, 7)  # 티커 + 매수일 — 같은 매매는 같은 줄을 갱신
STATUS = {"pending": "매수 대기", "open": "보유 중", "closed": "끝", "skipped": "건너뜀"}
ACTION = {"buy": "매수", "sell": "매도", "skip": "건너뜀"}


def _pct(x) -> str:
    """비율(0.1234) → "+12.34%". 없으면 빈칸."""
    return "" if x is None or not np.isfinite(x) else f"{x * 100:+.2f}%"


def _pp(x) -> float | str:
    """비율 차이(0.1234) → 12.34(%p, 숫자). 없으면 빈칸."""
    return "" if x is None or not np.isfinite(x) else round(float(x) * 100, 2)


def _num(x, nd: int = 2) -> float | str:
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


# ── 「일별 성적」 ────────────────────────────────────────────────────────────


def _event_text(e: dict) -> str:
    """사건 하나 → "매수 CRWV 144주 $50.90" / "매도 … (손익 …)" / "건너뜀 X(사유)"."""
    a = ACTION.get(e["action"], e["action"])
    if e["action"] == "skip":
        return f"{a} {e['ticker']}({e.get('note') or ''})"
    s = f"{a} {e['ticker']} {int(e['shares'])}주 ${float(e['px']):,.2f}"
    return s + (f" ({e['note']})" if e["action"] == "sell" and e.get("note") else "")


def daily_rows(portfolio: list[dict], events: list[dict], slots: dict[str, list[dict]] | None = None) -> list[list]:
    """포트폴리오 하루 줄(pead_watch_portfolio, row_json 포함)과 사건(pead_watch_pf_events) → 「일별 성적」 줄(날짜 순).
    slots = 매도 조건 비교 15칸 포트폴리오 {장부(main·exit_a·exit_b): 하루 줄} — 그날 줄이 없으면 빈칸.

    QQQ 비중 = 남은 현금(QQQ로 간주) ÷ 평가금액. 보유 종목은 그날까지 사건으로 되짚는다.
    올해 실현 이익(원) = 그해 매도마다 (받은 돈 × 매도일 환율 − 들어간 돈 × 매수일 환율), 수수료 뒤·세금 전.
    """
    fx = {r["date"]: r.get("fx") for r in portfolio}
    by_day: dict[str, list[dict]] = {}
    for e in events:
        by_day.setdefault(e["date"], []).append(e)
    held: dict[tuple, str] = {}
    cost: dict[tuple, tuple[float, str]] = {}
    realized: dict[int, float] = {}
    slot_by = {b: {r["date"]: r for r in rows} for b, rows in (slots or {}).items()}
    out = []
    for r in sorted(portfolio, key=lambda r: r["date"]):
        ds = r["date"]
        today = sorted(by_day.get(ds, []), key=lambda e: ({"buy": 0, "sell": 1, "skip": 2}.get(e["action"], 3), e["ticker"]))
        for e in today:
            k = (e["ticker"], e["d0"])
            if e["action"] == "buy":
                held[k] = e["ticker"]
                cost[k] = (float(e["amount_usd"]), ds)
            elif e["action"] == "sell":
                held.pop(k, None)
                c = cost.pop(k, None)
                f_sell, f_buy = fx.get(ds), fx.get(c[1]) if c else None
                if c and f_sell and f_buy:
                    y = int(ds[:4])
                    realized[y] = realized.get(y, 0.0) + float(e["amount_usd"]) * f_sell - c[0] * f_buy
        x = r.get("row_json")
        x = json.loads(x) if isinstance(x, str) else (x or r)
        value, cash = x.get("value_usd", r.get("value_usd")), x.get("cash_usd")
        out.append([ds,
                    round(r["value_krw"]) if r.get("value_krw") is not None else "",
                    round(r["qqq100_krw"]) if r.get("qqq100_krw") is not None else "",
                    _num(r.get("vs_qqq_pct")),
                    f"{cash / value * 100:.1f}%" if cash is not None and value else "",
                    len(held),
                    ", ".join(sorted(set(held.values()))) or "없음",
                    " · ".join(_event_text(e) for e in today),
                    round(realized.get(int(ds[:4]), 0.0))] + _slot_cells(slot_by, ds))
    return out


def _slot_cells(slot_by: dict[str, dict[str, dict]], ds: str) -> list:
    cells = []
    for b in SLOT_BOOKS:
        r = slot_by.get(b, {}).get(ds)
        cells += ["", ""] if r is None else [round(r["value_krw"]) if r.get("value_krw") is not None else "",
                                             _num(r.get("vs_qqq_pct"))]
    return cells


# ── 「매매이력」 ────────────────────────────────────────────────────────────


def tag_summary(s: dict | None) -> str:
    """신호 행의 기록 꼬리표(docs/prereg/pead_watch_tags.md) → 한 칸 요약. 꼬리표가 하나도 없으면 "기록 없음"(+빈칸 사유)."""
    s = s or {}
    parts = []
    if s.get("tag_sma200_above") is not None:
        dist = s.get("tag_sma200_dist")
        parts.append(("200일선 위" if s["tag_sma200_above"] else "200일선 아래") + (f"({dist:+.1f}%)" if dist is not None else ""))
    if s.get("tag_hi52") is not None:
        parts.append(f"52주 고점 대비 {s['tag_hi52']:+.1f}%")
    if s.get("tag_close_pos") is not None:
        parts.append(f"종가 위치 {s['tag_close_pos']:.2f}")
    if s.get("tag_n202_d0") is not None:
        parts.append(f"같은 날 발표 {int(s['tag_n202_d0'])}곳")
    if s.get("tag_q_before_close") is not None:
        parts.append("10-Q 동시 제출 " + ("예" if s["tag_q_before_close"] else "아니오"))
    if s.get("tag_lm_pos") is not None and s.get("tag_lm_neg") is not None:
        parts.append(f"어조 {s['tag_lm_pos'] - s['tag_lm_neg']:+.2f}(긍정−부정 %)")
    if not parts:
        return "기록 없음" + (f"({s['tag_errors']})" if s.get("tag_errors") else "")
    return " · ".join(parts) + (f" · 빈칸: {s['tag_errors']}" if s.get("tag_errors") else "")


def _exit_result(t: dict | None, a: pt.Aligned | None, bench: pt.Aligned, days: pd.DatetimeIndex, ia: int) -> str:
    """A·B 매매 → 결과 한 칸: 끝 "+12.34% (QQQ 대비 +3.21%p)" / 보유 중 "보유 중 …"(기준일 종가, 비용 전) / 매수 대기 / 건너뜀."""
    if t is None:
        return ""
    if t["status"] == "pending":
        return STATUS["pending"]
    if t["status"] == "skipped":
        return STATUS["skipped"] + (f"({t['skip']})" if t.get("skip") else "")
    if t["status"] == "closed" and t.get("buy_px") and t.get("sell_px") and t.get("q_buy") and t.get("q_sell"):
        r, q = t["sell_px"] / t["buy_px"] - 1, t["q_sell"] / t["q_buy"] - 1
        return f"{_pct(r)} (QQQ 대비 {(r - q) * 100:+.2f}%p)"
    if t["status"] == "open" and t.get("buy_px") and t.get("q_buy") and a is not None:
        ib = int(days.get_loc(pd.Timestamp(t["buy_day"])))
        if a.lastv[ia] >= ib:
            r = a.close[a.lastv[ia]] / t["buy_px"] - 1
            q = bench.close[bench.lastv[ia]] / t["q_buy"] - 1
            return f"보유 중 {_pct(r)} (QQQ 대비 {(r - q) * 100:+.2f}%p)"
    return STATUS.get(t["status"], t["status"])


def exit_cells(ta: dict | None, tb: dict | None, a: pt.Aligned | None, bench: pt.Aligned, days: pd.DatetimeIndex,
               ia: int) -> list:
    """「매매이력」 A·B 칸 6개: A 매도 예정일(출처), A 결과, B 목표가, B 매도일, B 결과, B 목표 도달."""
    a_plan = ""
    if ta is not None and ta.get("plan_day"):
        a_plan = f"{ta['plan_day']} ({ta.get('next_source') or ''})"
    elif ta is not None:
        a_plan = f"({ta.get('next_source') or ''})"
    hit = ""
    if tb is not None:
        if tb.get("hit_day"):
            hit = f"예 ({tb['hit_day']}, {int(tb['hit_days'])}거래일)"
        elif tb["status"] == "closed":
            hit = "아니오"
        elif tb["status"] in ("open", "pending"):
            hit = "아직"
    return [a_plan, _exit_result(ta, a, bench, days, ia),
            _num(tb.get("target_px")) if tb is not None else "",
            (tb.get("sell_day") or "") if tb is not None and tb["status"] == "closed" else "",
            _exit_result(tb, a, bench, days, ia), hit]


def trade_rows(main: list[dict], signals: list[dict], al: dict[str, pt.Aligned], bench: pt.Aligned,
               days: pd.DatetimeIndex, as_of, names: dict[str, str], check: int = 60,
               exits: dict[str, list[dict]] | None = None) -> list[list]:
    """주 장부(main) 매매 → 「매매이력」 줄(반응일·티커 순).

    입력: main(ledger='main' 매매 dict), signals(pead_watch_signals 행 — 반응일 숫자·꼬리표), al(종목 달력 배열, 보유 중 평가용),
          bench(QQQ), days(달력), as_of(기준일), names(티커 → 회사명).
    끝난 매매는 기록된 매도가·QQQ 매도가로, 보유 중이면 기준일 종가로(같은 시가에 산 QQQ와 차이, 비용 전).
    60일 시점 = 매수일부터 check 거래일째 종가(그날이 기준일 이하일 때만).
    exits = 매도 조건 비교 {'exit_a': 매매, 'exit_b': 매매} — 같은 (티커, 반응일) 줄 끝에 A·B 칸(없으면 빈칸).
    """
    sig = {(s["ticker"], str(s["d0"])[:10]): s for s in signals}
    xa = {(t["ticker"], str(t["d0"])[:10]): t for t in (exits or {}).get("exit_a", [])}
    xb = {(t["ticker"], str(t["d0"])[:10]): t for t in (exits or {}).get("exit_b", [])}
    ia = int(days.get_loc(pd.Timestamp(as_of)))
    out = []
    for t in sorted(main, key=lambda t: (str(t["d0"]), t["ticker"])):
        s = sig.get((t["ticker"], str(t["d0"])[:10])) or {}
        ib = int(days.get_loc(pd.Timestamp(t["buy_day"]))) if t.get("buy_day") else None
        chk = days[ib + check].date().isoformat() if ib is not None and ib + check < len(days) else ""
        ret = qret = ex60 = None
        a = al.get(t["ticker"])
        if t["status"] == "closed" and t.get("buy_px") and t.get("sell_px") and t.get("q_buy") and t.get("q_sell"):
            ret = t["sell_px"] / t["buy_px"] - 1
            qret = t["q_sell"] / t["q_buy"] - 1
        elif t["status"] == "open" and t.get("buy_px") and t.get("q_buy") and a is not None and a.lastv[ia] >= ib:
            ret = a.close[a.lastv[ia]] / t["buy_px"] - 1
            qret = bench.close[bench.lastv[ia]] / t["q_buy"] - 1
        if (t["status"] in ("open", "closed") and t.get("buy_px") and t.get("q_buy") and a is not None
                and ib + check <= ia and a.lastv[ib + check] >= ib):
            i60 = ib + check
            ex60 = (a.close[a.lastv[i60]] / t["buy_px"] - 1) - (bench.close[bench.lastv[i60]] / t["q_buy"] - 1)
        status = STATUS.get(t["status"], t["status"])
        if t["status"] == "skipped" and t.get("skip"):
            status += f"({t['skip']})"
        out.append([str(t["d0"])[:10], t["ticker"], names.get(t["ticker"], t["ticker"]), t.get("label") or s.get("label") or "",
                    _pct(s.get("ret")), _pp(s.get("excess")), _num(s.get("vol_ratio"), 1),
                    t.get("buy_day") or "", _num(t.get("buy_px")), chk, t.get("plan_day") or "",
                    (t.get("sell_day") or "") if t["status"] == "closed" else "",
                    _num(t.get("sell_px")) if t["status"] == "closed" else "",
                    _pct(ret), _pct(qret), _pp(ret - qret) if ret is not None and qret is not None else "",
                    _pp(ex60), status, tag_summary(s)]
                   + exit_cells(xa.get((t["ticker"], str(t["d0"])[:10])), xb.get((t["ticker"], str(t["d0"])[:10])),
                                a, bench, days, ia))
    return out
