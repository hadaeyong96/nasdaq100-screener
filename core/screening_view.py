"""보고서 맨 위 "오늘의 스크리닝" 구역과 매수 "왜?" 설명에 쓰는 표시용 값을 만드는 순수 함수 모음.

전략 판정은 하지 않는다. 오늘 실제로 무엇을 샀고 무엇이 막혔는지는 core/state.py가 낸
이벤트(A1·A2·A3·B·BLOCKED·STOP)를 그대로 따르고, 이 모듈은 그 결과를 차수별 "깔때기"
(단계별 종목 수 + 종목 표)로 다시 묶고, 각 종목의 오늘 지표 값을 표시용으로 모을 뿐이다.
조건 판정이 필요한 곳은 core/signals.py·core/filters.py의 함수를 그대로 불러 쓴다
(같은 조건을 두 번 구현하지 않는다).

core/ 규칙: 네트워크·파일·DB·현재 시각에 접근하지 않는다. 오늘(as_of) 이후의 지표 행은
읽지 않는다 — 날짜 계산(2차 기한 등)에서 지표 표 밖의 미래 거래일이 필요하면 호출부가
거래소 달력(future_trading_days)을 넘겨준다.

차수별 후보 정의(core/state.py process_day ⑤와 같다):
- 판정 시점 상태(entry_state): 오늘 손절(STOP)이 났으면 None(진입 판정 없음). 오늘 매수
  신호로 "주문대기"가 됐으면 신호 전 상태(pending.prev_state). 그 밖에는 오늘 처리 후 상태.
- 1차(A1): 판정 시점 상태가 "대기"이고 재진입 대기(쿨다운)가 아닌 종목 중 RSI 30 상향 돌파.
- 2차(A2): 판정 시점 상태가 "정찰"(1차 체결 확정, a1_date 있음)인 종목만.
- 3차(A3): 판정 시점 상태가 "확인"(2차 보유)인 종목만.
- 재진입(B): 판정 시점 상태가 "대기"인 종목 중 추세 확인(종가 > 구름 상단, 앞구름 양운,
  후행스팬 돌파). 쿨다운은 보지 않는다(core/state.py는 쿨다운을 A1에만 적용한다).
"""

from __future__ import annotations

import pandas as pd

from core import filters
from core import signals as sig

BUY_KINDS = ("A1", "A2", "A3", "B")
_HELD_LABEL = {
    "정찰": "1차 보유 중", "확인": "2차 보유 중", "확정": "3차 보유 중",
    "추세보유": "재진입 보유 중", "청산중": "청산 중", "주문대기": "주문 대기 중",
}
FOLD_AFTER = 10  # 2차 "대기" 줄이 이보다 많으면 나머지는 접는다


# ── 작은 도우미 ─────────────────────────────────────────────────────────


def _num(v, digits: int = 2):
    """NaN·None -> None, 그 밖에는 반올림한 float."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(v), digits)


def _flag(v):
    """NaN·None -> None(확인 불가), 그 밖에는 bool."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return bool(v)


def _mmdd(d) -> str | None:
    if d is None:
        return None
    ts = pd.Timestamp(d)
    return f"{ts.month}/{ts.day}"


def _gt(a, b):
    """a > b. 둘 중 하나라도 없으면 None."""
    if a is None or b is None:
        return None
    return a > b


def trading_day_offset(index: pd.DatetimeIndex, base, n: int, as_of, future_trading_days=None):
    """base 거래일에서 n거래일 뒤의 날짜.

    입력: index(종목 지표 표의 날짜), base(index 안의 날짜), n(0 이상), as_of(오늘),
         future_trading_days(as_of 다음 거래일부터의 거래소 달력, 없으면 주말만 빼는 근사)
    출력: pd.Timestamp 또는 None(base가 index에 없을 때)
    오늘(as_of) 이후의 index 값은 읽지 않는다 — 미래 데이터 방지.
    """
    past = index[index <= pd.Timestamp(as_of)]
    if pd.Timestamp(base) not in past:
        return None
    pos = past.get_loc(pd.Timestamp(base)) + n
    if pos < len(past):
        return past[pos]
    ahead = pos - (len(past) - 1)  # as_of 다음 몇 번째 거래일인지 (1부터)
    if future_trading_days is not None:
        future = [pd.Timestamp(d) for d in future_trading_days if pd.Timestamp(d) > pd.Timestamp(as_of)]
        if ahead <= len(future):
            return future[ahead - 1]
    return (pd.Timestamp(as_of) + pd.tseries.offsets.BDay(ahead)).normalize()


def a2_deadline(index, a1_date, cfg: dict, as_of, future_trading_days=None):
    """2차를 살 수 있는 마지막 날 = A1 당일 포함 a1_to_a2_expiry_days번째 거래일.

    core/state.py: A1일 오프셋 0부터 (expiry_days − 1)까지 A2를 판정하고, 오프셋
    expiry_days가 되는 날 ②에서 먼저 만료시킨다.
    """
    return trading_day_offset(index, a1_date, cfg["assumptions"]["a1_to_a2_expiry_days"] - 1, as_of, future_trading_days)


def entry_state(state_after: dict, ticker_events: list[dict], as_of) -> str | None:
    """오늘 진입 신호(⑤)를 판정하던 시점의 상태. 오늘 손절이 났으면 None (state.py ①이 바로 끝낸다)."""
    if any(e["kind"] == "STOP" for e in ticker_events):
        return None
    pending = state_after.get("pending")
    if state_after.get("state") == "주문대기" and pending and pd.Timestamp(pending["date"]) == pd.Timestamp(as_of):
        return pending.get("prev_state")
    return state_after.get("state")


def _in_cooldown(state_after: dict, as_of) -> bool:
    cu = state_after.get("cooldown_until")
    return bool(cu) and pd.Timestamp(as_of) < pd.Timestamp(cu)


def display_score(df: pd.DataFrame, date, with_grade: bool, cfg: dict) -> tuple[str | None, int]:
    """core/state.py _entry_score와 같은 계산(등급은 A2·B형만) — 표시용.

    입력: df, date(df.index의 값), with_grade(등급을 매길지), cfg
    출력: (등급 또는 None, 우선순위 점수)
    """
    row = df.loc[date]
    grade_letter = filters.grade(row.get("macd_norm"), filters.macd_cross_count(df, date), cfg) if with_grade else None
    thickness = filters.cloud_thickness_pct(row.get("cloud_top"), row.get("cloud_bot"), row.get("close"))
    return grade_letter, filters.priority_score(grade_letter, row.get("vol_ratio"), thickness, row.get("bb_width_pct"), cfg)


def _reason_label(reason: str) -> str:
    """core/filters.py ban_reasons 원문 -> 표의 짧은 제외 사유."""
    if "RSI 70" in reason:
        return "RSI 70 이상"
    if "휩소" in reason:
        return "횡보(교차 잦음)"
    if "구름 안" in reason:
        return "구름 안"
    if "음운" in reason:
        return "앞구름 음운"
    if "갭" in reason:
        return "시가 갭"
    if "실적" in reason:
        return "실적 3거래일 이내"
    if "한도" in reason:
        return "동시 보유 한도"
    return reason


def _blocked_label(event: dict) -> str:
    if event.get("blocked_type") == "limit":
        return "동시 보유 한도"
    return " · ".join(_reason_label(r) for r in event.get("reasons", []))


def _a3_mode(cfg: dict) -> str:
    """A3 진입 방식(config a3.mode). 설정이 없으면 core/signals.py 기본값(돌파형)과 같게 본다."""
    return (cfg.get("a3") or {}).get("mode", "breakout")


def _check_a3(row, cfg: dict) -> bool:
    """core.signals.check_a3 — a3 설정이 없는 cfg에서는 돌파형으로 판정한다(check_a3의 기본 분기와 같음)."""
    return sig.check_a3(row, cfg) if cfg.get("a3") else sig.check_a3_breakout(row, cfg)


def _rank(rows: list[dict]) -> None:
    """추천 줄에만 점수 순 순위를 매긴다(같은 점수면 표의 원래 순서)."""
    recommended = sorted((r for r in rows if r["result"] == "추천"), key=lambda r: -r["score"])
    for i, r in enumerate(recommended, start=1):
        r["rank"] = i
    for r in rows:
        r.setdefault("rank", None)


def _base_row(ticker: str, name_map: dict, df: pd.DataFrame, date) -> dict:
    row = df.loc[date]
    thickness = filters.cloud_thickness_pct(row.get("cloud_top"), row.get("cloud_bot"), row.get("close"))
    return {
        "ticker": ticker,
        "kr": (name_map or {}).get(ticker) or ticker,
        "rsi": _num(row.get("rsi"), 1),
        "vol_ratio": _num(row.get("vol_ratio"), 1),
        "cloud_thickness_pct": _num(thickness, 1),
    }


# ── 메인: 오늘의 깔때기 ─────────────────────────────────────────────────


def build_screening_view(
    indicator_map: dict,
    as_of_by_ticker: dict,
    positions: dict,
    today_events: list[dict],
    cfg: dict,
    *,
    name_map: dict | None = None,
    recent_events: list[dict] | None = None,
    future_trading_days=None,
) -> dict:
    """차수별 깔때기(단계별 숫자 + 종목 표)를 만든다.

    입력: indicator_map({ticker: 지표 df}), as_of_by_ticker({ticker: 오늘 처리한 날짜}),
         positions({ticker: 오늘 처리 후 상태}), today_events(오늘 이벤트, 각자 "ticker" 포함),
         cfg, name_map({ticker: 한글명}), recent_events(최근 a1_to_a2_expiry_days 거래일의
         이벤트 — DB에서 engine이 읽어 넘긴다. A1 신호가 났지만 2차 대상이 아닌 종목 표시용),
         future_trading_days(오늘 다음 거래일 목록 — 2차 기한이 지표 표 밖일 때 쓴다)
    출력: {"as_of", "scan_count", "lanes": {"a1","a2","a3","b"}, "funnel_check", "facts_by_key"}
         각 레인: {"key","title","steps":[{"label","count"}],"rows":[...], ...}
         facts_by_key: {"{ticker}-{차수}": {"lane_counts", "rank", "scan_count"}} — 매수 "왜?" ②용
    """
    name_map = name_map or {}
    events_by_ticker: dict[str, list[dict]] = {}
    for e in today_events:
        events_by_ticker.setdefault(e.get("ticker", ""), []).append(e)

    scanned = [
        t for t, df in indicator_map.items()
        if as_of_by_ticker.get(t) is not None and as_of_by_ticker[t] in df.index
    ]
    as_of = max((as_of_by_ticker[t] for t in scanned), default=None)

    def ev(ticker: str, kind: str, stage: str | None = None):
        for e in events_by_ticker.get(ticker, []):
            if e["kind"] == kind and (stage is None or e.get("stage") == stage):
                return e
        return None

    a1_rows, a2_rows, a3_rows, b_rows = [], [], [], []
    a2_holders: list[dict] = []
    trend_count = gc_total = gc_a2 = gc_b = 0
    rsi_cross = a3_cond = 0
    today_a1: list[str] = []
    s_min = cfg["assumptions"]["s_grade_macd_norm_min_pct"]

    for ticker in scanned:
        df = indicator_map[ticker]
        date = as_of_by_ticker[ticker]
        idx = df.index.get_loc(date)
        row = df.loc[date]
        prev_rsi = df.iloc[idx - 1].get("rsi") if idx > 0 else float("nan")
        state_after = positions.get(ticker) or {"state": "대기"}
        tev = events_by_ticker.get(ticker, [])
        pre = entry_state(state_after, tev, date)
        gc = _flag(row.get("gc")) is True
        if gc:
            gc_total += 1
        if ev(ticker, "A1"):
            today_a1.append(ticker)

        # ── 1차: RSI 30 상향 돌파한 모든 종목 ──
        if sig.check_a1(prev_rsi, row.get("rsi")):
            rsi_cross += 1
            _, score = display_score(df, date, False, cfg)
            r = {**_base_row(ticker, name_map, df, date), "rsi_prev": _num(prev_rsi, 1), "score": score}
            blocked = ev(ticker, "BLOCKED", "A1")
            if ev(ticker, "A1"):
                r.update(result="추천", tone="ok")
            elif blocked:
                r.update(result="제외", reason=_blocked_label(blocked), tone="no", passed_checks=blocked.get("blocked_type") == "limit")
            elif pre is None:
                r.update(result="제외", reason="오늘 손절", tone="no")
            elif pre != "대기":
                r.update(result="제외", reason=_HELD_LABEL.get(pre, pre), tone="inf")
            elif _in_cooldown(state_after, date):
                r.update(result="제외", reason=f"재진입 대기 ~{_mmdd(state_after['cooldown_until'])}", tone="inf")
            else:
                r.update(result="확인 필요", reason="오늘 신호 기록 없음", tone="inf")
            a1_rows.append(r)

        # ── 2차: 판정 시점에 정찰(1차 보유) ──
        if pre == "정찰":
            a1_date = state_after.get("a1_date")
            a1_date = pd.Timestamp(a1_date) if a1_date is not None else None
            elapsed = (idx - df.index.get_loc(a1_date) + 1) if (a1_date is not None and a1_date in df.index) else None
            deadline = a2_deadline(df.index, a1_date, cfg, date, future_trading_days) if a1_date is not None else None
            grade_letter, score = display_score(df, date, gc, cfg)
            r = {
                **_base_row(ticker, name_map, df, date),
                "a1_date": a1_date, "a1_date_str": _mmdd(a1_date), "elapsed": elapsed,
                "expiry_days": cfg["assumptions"]["a1_to_a2_expiry_days"],
                "deadline": deadline, "deadline_str": _mmdd(deadline),
                "gc": gc, "grade": grade_letter, "score": score,
            }
            a2_holders.append({"ticker": ticker, "kr": r["kr"], "a1_date": a1_date, "deadline": deadline})
            blocked = ev(ticker, "BLOCKED", "A2")
            rsi = r["rsi"]
            if gc:
                gc_a2 += 1
            if ev(ticker, "A2"):
                r.update(result="추천", tone="ok")
            elif blocked:
                r.update(result="제외", reason=_blocked_label(blocked), tone="no")
            elif gc and rsi is not None and rsi >= 70:
                r.update(result="제외", reason="RSI 70 이상", tone="no")
            elif gc and rsi is not None and rsi < 30:
                r.update(result="제외", reason="RSI 30 미만", tone="no")
            elif gc:
                r.update(result="확인 필요", reason="오늘 신호 기록 없음", tone="inf")
            else:
                r.update(result="대기", reason=f"기한 {r['deadline_str']}" if r["deadline_str"] else "기한 확인 불가", tone="inf")
            a2_rows.append(r)

        # ── 3차: 판정 시점에 확인(2차 보유) ──
        if pre == "확인":
            close, top = _num(row.get("close")), _num(row.get("cloud_top"))
            conds = {
                "cloud": _gt(close, top),
                "yang": _flag(row.get("future_yang")),
                "chikou": _flag(row.get("chikou_ok")),
                "momentum": (
                    None if _num(row.get("macd"), 4) is None or _num(row.get("signal"), 4) is None or _num(row.get("rsi")) is None
                    else bool(row["macd"] > row["signal"] and row["rsi"] >= 50)
                ),
            }
            met = sum(1 for v in conds.values() if v)
            prev_close = df.iloc[idx - 1].get("close") if idx > 0 else float("nan")
            gap = _num((row["open"] / prev_close - 1) * 100, 1) if (_num(row.get("open")) and _num(prev_close)) else None
            _, score = display_score(df, date, False, cfg)
            ok = _check_a3(row, cfg)
            a3_cond += int(ok)
            r = {**_base_row(ticker, name_map, df, date), **conds, "met": met, "gap_pct": gap, "score": score,
                 "gap_ok": None if gap is None else gap < cfg["assumptions"]["gap_filter_pct"]}
            blocked = ev(ticker, "BLOCKED", "A3")
            if ev(ticker, "A3"):
                r.update(result="추천", tone="ok")
            elif blocked:
                r.update(result="제외", reason=_blocked_label(blocked), tone="no")
            elif met == 4:
                r.update(result="대기", reason="되돌림 기다림" if _a3_mode(cfg) == "pullback" else "오늘 신호 기록 없음", tone="inf")
            else:
                r.update(result="대기", reason=f"{met}/4", tone="inf")
            a3_rows.append(r)

        # ── 재진입: 판정 시점에 대기 + 추세 확인 ──
        if pre == "대기":
            trend = (
                _gt(_num(row.get("close")), _num(row.get("cloud_top"))) is True
                and _flag(row.get("future_yang")) is True
                and _flag(row.get("chikou_ok")) is True
            )
            if trend:
                trend_count += 1
                if gc:
                    gc_b += 1
                    grade_letter, score = display_score(df, date, True, cfg)
                    r = {**_base_row(ticker, name_map, df, date), "macd_norm": _num(row.get("macd_norm")), "grade": grade_letter, "score": score}
                    blocked = ev(ticker, "BLOCKED", "B")
                    rsi, mn = r["rsi"], r["macd_norm"]
                    if ev(ticker, "B"):
                        r.update(result="추천", tone="ok")
                    elif blocked:
                        r.update(result="제외", reason=_blocked_label(blocked), tone="no")
                    elif ev(ticker, "A1") or ev(ticker, "BLOCKED", "A1"):
                        r.update(result="제외", reason="오늘 1차 신호 우선", tone="inf")
                    elif mn is not None and mn < s_min:
                        r.update(result="제외", reason=f"정규화 MACD {s_min:g}% 미만", tone="no")
                    elif rsi is not None and rsi < 50:
                        r.update(result="제외", reason="RSI 50 미만", tone="no")
                    elif rsi is not None and rsi >= 70:
                        r.update(result="제외", reason="RSI 70 이상", tone="no")
                    else:
                        r.update(result="확인 필요", reason="오늘 신호 기록 없음", tone="inf")
                    b_rows.append(r)

    # 정렬: 추천(점수 순) → 제외 → 대기
    order = {"추천": 0, "제외": 1, "확인 필요": 2, "대기": 3}
    for rows in (a1_rows, a2_rows, a3_rows, b_rows):
        rows.sort(key=lambda r: (order.get(r["result"], 9), -r["score"], r["ticker"]))
        _rank(rows)
    a2_rows.sort(key=lambda r: (order.get(r["result"], 9), -r["score"] if r["result"] != "대기" else 0,
                                r["deadline"] or pd.Timestamp.max, r["ticker"]))
    waiting = 0
    for r in a2_rows:
        if r["result"] == "대기":
            waiting += 1
            r["folded"] = waiting > FOLD_AFTER
        else:
            r["folded"] = False

    def count(rows, *results):
        return sum(1 for r in rows if r["result"] in results)

    a1_pass = count(a1_rows, "추천") + sum(1 for r in a1_rows if r.get("passed_checks"))
    a1_rec = count(a1_rows, "추천")
    a2_rec, a3_rec, b_rec = count(a2_rows, "추천"), count(a3_rows, "추천"), count(b_rows, "추천")

    # 2차 "1차 보유 명단": 1차 매수일별로 묶는다
    roster_map: dict = {}
    for h in sorted(a2_holders, key=lambda h: (h["a1_date"] or pd.Timestamp.min, h["ticker"])):
        roster_map.setdefault(h["a1_date"], {"a1_date": h["a1_date"], "a1_date_str": _mmdd(h["a1_date"]),
                                             "deadline_str": _mmdd(h["deadline"]), "names": []})["names"].append(f"{h['kr']}({h['ticker']})")
    today_deadline = None
    if as_of is not None and scanned:
        ref_df = indicator_map[scanned[0]]
        if as_of in ref_df.index:
            today_deadline = a2_deadline(ref_df.index, as_of, cfg, as_of, future_trading_days)

    # 참고: 최근 A1 신호가 났지만 지금 2차 대상이 아닌 종목
    holder_set = {h["ticker"] for h in a2_holders}
    not_bought: dict[str, dict] = {}
    recent = sorted(recent_events or [], key=lambda e: pd.Timestamp(e["date"]))
    for e in recent:
        t = e.get("ticker")
        if e["kind"] != "A1" or t in holder_set or t in today_a1:
            continue
        if as_of is not None and pd.Timestamp(e["date"]) >= pd.Timestamp(as_of):
            continue
        st_after = positions.get(t) or {}
        if st_after.get("a1_date") is not None and pd.Timestamp(st_after["a1_date"]) == pd.Timestamp(e["date"]):
            continue  # 그 1차로 이미 2차 이상 진행 중 — 산 종목이다
        later = [x["kind"] for x in recent if x.get("ticker") == t and pd.Timestamp(x["date"]) > pd.Timestamp(e["date"])]
        if "UNFILLED" in later:
            why = "체결 기록 없음"
        elif "STOP" in later:
            why = "손절로 정리"
        elif "A1_EXPIRE" in later:
            why = "2차 기한 만료로 정리"
        else:
            why = _HELD_LABEL.get(st_after.get("state"), "체결 기록 없음")
        not_bought[t] = {"ticker": t, "kr": name_map.get(t) or t, "a1_date_str": _mmdd(e["date"]), "why": why}

    lanes = {
        "a1": {
            "key": "a1", "title": "1차 정찰 · RSI 30 탈출",
            "steps": [
                {"label": "스캔", "count": len(scanned)},
                {"label": "RSI 30 탈출", "count": rsi_cross},
                {"label": "검사 통과", "count": a1_pass},
                {"label": "오늘 추천", "count": a1_rec},
            ],
            "rows": a1_rows,
        },
        "a2": {
            "key": "a2", "title": "2차 확인 · 1차를 산 종목 중 오늘 MACD 골든크로스",
            "steps": [
                {"label": "1차 보유 종목", "count": len(a2_rows)},
                {"label": "오늘 골든크로스", "count": gc_a2},
                {"label": "금지 구간 통과", "count": a2_rec},
            ],
            "rows": a2_rows,
            "roster": list(roster_map.values()),
            "today_a1_count": len(today_a1),
            "today_a1_deadline_str": _mmdd(today_deadline),
            "as_of_str": _mmdd(as_of),
            "not_bought": sorted(not_bought.values(), key=lambda x: x["ticker"]),
            "folded_count": sum(1 for r in a2_rows if r.get("folded")),
        },
        "a3": {
            "key": "a3", "title": "3차 확정 · 2차 보유 종목의 구름 4조건",
            "steps": [
                {"label": "2차 보유", "count": len(a3_rows)},
                {"label": "4조건 충족" if _a3_mode(cfg) != "pullback" else "4조건+되돌림 충족", "count": a3_cond},
                {"label": "갭·금지 구간 통과", "count": a3_rec},
            ],
            "rows": a3_rows,
            "mode": _a3_mode(cfg),
        },
        "b": {
            "key": "b", "title": "재진입 · 상승 추세 종목의 골든크로스",
            "steps": [
                {"label": "구름 위 추세 종목", "count": trend_count},
                {"label": "오늘 골든크로스", "count": gc_b},
                {"label": "조건·금지 구간 통과", "count": b_rec},
            ],
            "rows": b_rows,
        },
    }

    blocked_events = [e for e in today_events if e["kind"] == "BLOCKED"]
    funnel_check = {
        "rsi_cross": rsi_cross,
        "gc_total": gc_total,
        "gc_a2": gc_a2,
        "gc_b": gc_b,
        "gc_other": gc_total - gc_a2 - gc_b,
        "a3_cond": a3_cond,
        "ban_total": sum(1 for e in blocked_events if e.get("blocked_type") == "ban"),
        "limit_total": sum(1 for e in blocked_events if e.get("blocked_type") == "limit"),
    }

    stage_to_lane = {"A1": "a1", "A2": "a2", "A3": "a3", "B": "b"}
    facts_by_key: dict[str, dict] = {}
    for stage, lane_key in stage_to_lane.items():
        lane = lanes[lane_key]
        for r in lane["rows"]:
            if r["result"] == "추천":
                facts_by_key[f"{r['ticker']}-{stage}"] = {
                    "rank": r["rank"], "steps": lane["steps"], "rec_count": sum(1 for x in lane["rows"] if x["result"] == "추천"),
                }

    return {
        "as_of": as_of,
        "as_of_str": _mmdd(as_of),
        "scan_count": len(scanned),
        "lanes": lanes,
        "funnel_check": funnel_check,
        "facts_by_key": facts_by_key,
    }


# ── 매수 한 건의 "왜?" ① 규칙 확인용 오늘 값 ─────────────────────────────


def buy_facts(
    df: pd.DataFrame,
    date,
    stage: str,
    state_after: dict,
    cfg: dict,
    *,
    earnings_date=None,
    future_trading_days=None,
) -> dict:
    """매수 신호 한 건(stage)의 규칙 확인용 오늘 값을 모은다 (core/explain.py explain_buy의 ① ③용).

    입력: df(지표), date(신호일 = 오늘), stage(A1|A2|A3|B), state_after(오늘 처리 후 상태 —
         주문대기의 pending에 신호 전 상태가 있다), cfg, earnings_date(다음 실적일, 모르면 None),
         future_trading_days(2차 기한 계산용 거래소 달력)
    출력: dict — 값이 없으면 None(설명에서 "확인 불가"). 오늘(date) 이후 행은 읽지 않는다.
    """
    idx = df.index.get_loc(date)
    row = df.loc[date]
    prev = df.iloc[idx - 1] if idx > 0 else None
    a = cfg["assumptions"]
    D = a.get("ichimoku_shift", (cfg.get("indicators") or {}).get("ichimoku_shift"))
    n_swing = a.get("swing_low_period", ((cfg.get("indicators") or {}).get("swing_low") or {}).get("period"))
    pending = state_after.get("pending") or {}
    pre_state = pending.get("prev_state") if state_after.get("state") == "주문대기" and pending else state_after.get("state")

    facts: dict = {
        "date": pd.Timestamp(date),
        "close": _num(row.get("close")),
        "open": _num(row.get("open")),
        "prev_close": _num(prev.get("close")) if prev is not None else None,
        "rsi_now": _num(row.get("rsi"), 1),
        "rsi_prev": _num(prev.get("rsi"), 1) if prev is not None else None,
        "macd": _num(row.get("macd"), 3),
        "signal": _num(row.get("signal"), 3),
        "macd_prev": _num(prev.get("macd"), 3) if prev is not None else None,
        "signal_prev": _num(prev.get("signal"), 3) if prev is not None else None,
        "gc": _flag(row.get("gc")),
        "macd_norm": _num(row.get("macd_norm")),
        "cloud_top": _num(row.get("cloud_top")),
        "cloud_bot": _num(row.get("cloud_bot")),
        "span_a": _num(row.get("span_a")),
        "span_b": _num(row.get("span_b")),
        "future_yang": _flag(row.get("future_yang")),
        "high_d_ago": _num(df.iloc[idx - D]["high"]) if D is not None and idx - D >= 0 and "high" in df.columns else None,
        "chikou_ok": _flag(row.get("chikou_ok")),
        "ichimoku_shift": D,
        "pre_state": pre_state,
        "cooldown_until": state_after.get("cooldown_until"),
        "cross_count_20d": filters.macd_cross_count(df, date) if {"gc", "dc"} <= set(df.columns) else None,
        "earnings_date": pd.Timestamp(earnings_date) if earnings_date is not None else None,
        "earnings_within": filters.is_earnings_within(date, earnings_date) if earnings_date is not None else None,
        "units": {u: q for u, q in (state_after.get("units") or {}).items() if q},
    }
    o, pc = facts["open"], facts["prev_close"]
    facts["gap_pct"] = round((o / pc - 1) * 100, 1) if (o and pc) else None
    if stage in ("A2", "B"):
        facts["grade"] = filters.grade(row.get("macd_norm"), facts["cross_count_20d"] or 0, cfg)

    # 1차 매수일(A2·A3 손절 기준일)과 경과 거래일
    a1_date = state_after.get("a1_date")
    if a1_date is not None and pd.Timestamp(a1_date) in df.index:
        a1_date = pd.Timestamp(a1_date)
        facts["a1_date"] = a1_date
        facts["elapsed"] = idx - df.index.get_loc(a1_date) + 1  # A1 당일 = 1일째
        facts["a2_deadline"] = a2_deadline(df.index, a1_date, cfg, date, future_trading_days)
    else:
        facts["a1_date"] = None
        facts["elapsed"] = None
        facts["a2_deadline"] = None
    if stage == "A1":
        facts["a2_deadline"] = a2_deadline(df.index, date, cfg, date, future_trading_days)

    # 손절가 기준: 10일 최저가가 나온 날(기준일까지 최근 n_swing 거래일 중 저가가 가장 낮은 날)
    if stage in ("A1", "B"):
        basis_date = pd.Timestamp(date)
    else:
        basis_date = facts["a1_date"]
    facts["swing_basis_date"] = basis_date
    facts["swing_low"] = None
    facts["swing_low_date"] = None
    if basis_date is not None and basis_date in df.index and "low" in df.columns and n_swing:
        b_idx = df.index.get_loc(basis_date)
        window = df.iloc[max(0, b_idx - n_swing + 1): b_idx + 1]["low"].dropna()
        if not window.empty:
            facts["swing_low"] = _num(window.min())
            facts["swing_low_date"] = window.idxmin()
    if facts["swing_low"] is None and stage in ("A1", "B"):
        facts["swing_low"] = _num(row.get("swing_low"))
    return facts
