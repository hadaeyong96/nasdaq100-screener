"""보고서 각 줄의 "왜?" 설명(선별 이유)을 만드는 단일 모듈 (P3.7).

신호·청산·제외·경고·관찰의 사유 코드마다 흩어 쓰지 않고 이 모듈 하나에
모은다. 입력은 이미 계산된 지표 값과 config뿐이고(파일·네트워크·DB·현재
시각 접근 없음), 같은 입력이면 항상 같은 출력을 낸다 — core/ 규칙을 그대로
따른다. 전략 판정 자체는 하지 않는다(core/signals.py·core/filters.py·
core/state.py가 이미 정한 결과를 문장으로 옮길 뿐이다).

출력 구조(모든 explain_* 함수 공통):
    {"title": str, "badge": str|None,
     "checks": [{"level": "y"|"n"|"i", "text": str}, ...],
     "body": str, "next": str|None}
매수(explain_buy)는 여기에 "sections"(① 우리 규칙 ② 왜 이 종목인가 ③ 어떻게 사나
④ 다음 단계)와 "source"(근거 줄)를 더한다. ✓/✗(y/n)는 오늘 실제 값과 기준을 비교해
정하고, 값이 없으면 "i"(확인 불가)로 둔다 — 고정 "y" 금지.

호출부(engine/daily.py)가 채워 넘기는 ctx dict의 키는 각 함수 docstring에
적어 둔다. 없어도 되는 값은 None으로 두면 그 항목만 빠진다.
"""

from __future__ import annotations

from core.signals import levels

_LEVEL = ("y", "n", "i")


def _chk(level: str, text: str) -> dict:
    assert level in _LEVEL
    return {"level": level, "text": text}


def _usd(v) -> str:
    return f"${v:,.2f}" if v is not None else "확인 불가"


def _pct1(v, signed: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v:+.1f}%" if signed else f"{v:.1f}%"


def _score_breakdown(ctx: dict, cfg: dict, stage: str | None = None) -> str:
    """실제 우선순위 점수(core/filters.py priority_score) 구성 요소를 그대로 문장으로 옮긴다.

    stage를 주면 core/state.py _entry_score와 같이 등급 점수는 A2·B형에만 넣는다(A3 줄의
    ctx["grade"]는 2차 때 등급이라 점수에는 들어가지 않는다). B형 이벤트에는 등급이 없어
    facts["grade"](오늘 값으로 다시 계산한 등급)를 쓴다.
    """
    parts: list[str] = []
    grade_letter = ctx.get("grade") or None
    if stage is not None:
        grade_letter = (grade_letter or (ctx.get("facts") or {}).get("grade")) if stage in ("A2", "B") else None
    grade_pts = {"S": 40, "A": 25, "B": 10}.get(grade_letter, 0)
    parts.append(f"등급 {grade_letter}(+{grade_pts})" if grade_letter else "등급 없음(+0, 1·3차는 등급을 매기지 않음)")

    vol_ratio = ctx.get("vol_ratio")
    if vol_ratio is not None:
        if vol_ratio >= 1.5:
            parts.append(f"거래량 {vol_ratio}배(+25)")
        elif vol_ratio >= 1.2:
            parts.append(f"거래량 {vol_ratio}배(+15)")
        else:
            parts.append(f"거래량 {vol_ratio}배(+0, 기준 1.2배 미달)")

    thickness = ctx.get("cloud_thickness_pct")
    if thickness is not None:
        if thickness <= 3:
            parts.append(f"구름두께 {thickness:.1f}%(+20)")
        elif thickness <= 6:
            parts.append(f"구름두께 {thickness:.1f}%(+10)")
        else:
            parts.append(f"구름두께 {thickness:.1f}%(+0, 기준 6% 초과)")

    bb = ctx.get("bb_width_pct")
    if bb is not None:
        if bb <= 0.20:
            parts.append(f"밴드폭 백분위 {bb * 100:.0f}%(+15)")
        else:
            parts.append(f"밴드폭 백분위 {bb * 100:.0f}%(+0, 기준 20% 초과)")

    return " · ".join(parts) + f" = {ctx.get('score', 0)}점"


# ── 매수 (A1·A2·A3·B) ───────────────────────────────────────────────────

_BUY_TITLE = {"A1": "1차 정찰 매수", "A2": "2차 확인 매수", "A3": "3차 확정 매수", "B": "재진입 매수"}
_BUY_BADGE = {"A1": "슬롯의 1/9", "A2": "슬롯의 2/9", "A3": "슬롯의 6/9", "B": "슬롯 전체 · 1회"}
# 실전(live)은 계좌 슬롯이 아니라 종목별 계획금액으로 수량을 정한다 (docs/design/live_advisor.md).
_BUY_BADGE_LIVE = {"A1": "계획금액의 1/9", "A2": "계획금액의 2/9", "A3": "계획금액의 6/9", "B": "계획금액 전체 · 1회"}
_BUY_FRACTION = {"A1": "1/9", "A2": "2/9", "A3": "6/9", "B": "전체(9/9)"}
# 근거 줄: 전략 문서 장·절과 13장 근거 대본 매핑을 그대로 옮긴다.
_BUY_SOURCE = {
    "A1": "근거: 전략 v3 4장 A1 · 대본 M11, M12",
    "A2": "근거: 전략 v3 4장 A2 · 7장 등급·매매 금지 구간 · 대본 M10, M11, M12",
    "A3": "근거: 전략 v3 4장 A3 · 7장 매매 금지 구간 · 대본 M11, 일목 01·04·05·09",
    "B": "근거: 전략 v3 4장 B형 · 7장 등급·매매 금지 구간 · 대본 M06, M09, M10, M12, 일목 01·04·08",
}
_STATE_HELD = {
    "정찰": "1차 보유 중", "확인": "2차 보유 중", "확정": "3차 보유 중",
    "추세보유": "재진입 보유 중", "청산중": "청산 중", "주문대기": "주문 대기 중",
}


def _verdict(ok, text: str) -> dict:
    """값 비교 결과(True/False/None) -> 체크 한 줄. None이면 "확인 불가"."""
    if ok is None:
        return _chk("i", f"{text} · 확인 불가")
    return _chk("y" if ok else "n", text)


def _v(x, fmt: str = "{:.2f}") -> str:
    return fmt.format(x) if x is not None else "?"


def _md(d) -> str | None:
    if d is None:
        return None
    try:
        import pandas as pd  # 지역 import: 이 모듈의 다른 함수는 pandas 없이도 쓰인다

        ts = pd.Timestamp(d)
    except (TypeError, ValueError):
        return str(d)
    return f"{ts.month}/{ts.day}"


def _fact(ctx: dict, key: str):
    """facts(core.screening_view.buy_facts) 값을 먼저, 없으면 ctx 값을 쓴다."""
    facts = ctx.get("facts") or {}
    if facts.get(key) is not None:
        return facts[key]
    return ctx.get(key)


def _gc_item(ctx: dict) -> dict:
    mp, sp, m, s = _fact(ctx, "macd_prev"), _fact(ctx, "signal_prev"), _fact(ctx, "macd"), _fact(ctx, "signal")
    if None not in (mp, sp, m, s):
        ok = mp <= sp and m > s
        return _verdict(ok, f"MACD {mp:+.3f}→{m:+.3f} vs 시그널 {sp:+.3f}→{s:+.3f} · 기준: 어제 MACD ≤ 시그널, 오늘 MACD > 시그널(골든크로스)")
    return _verdict(_fact(ctx, "gc"), "MACD 골든크로스(어제 MACD ≤ 시그널, 오늘 MACD > 시그널)")


def _earnings_item(ctx: dict) -> dict:
    facts = ctx.get("facts") or {}
    n = levels(ctx.get("_cfg"))["earnings_filter_trading_days"]
    if "earnings_within" not in facts:
        return _chk("i", f"실적 발표 {n}거래일 이내 아님 · 확인 불가")
    ed = facts.get("earnings_date")
    if ed is None:
        return _chk("i", "실적 발표일 확인 불가 · 모르면 막지 않음")
    within = facts.get("earnings_within")
    return _verdict(None if within is None else not within, f"실적 발표 {_md(ed)} · 기준: {n}거래일 이내면 금지")


def _whipsaw_item(ctx: dict, cfg: dict) -> dict:
    n = _fact(ctx, "cross_count_20d")
    if n is None:
        n = ctx.get("gc_count_20d")
    w = cfg["assumptions"]["whipsaw_max_crosses_20d"]
    days = levels(cfg)["cross_window_days"]
    return _verdict(None if n is None else n < w, f"최근 {days}거래일 MACD 교차 {_v(n, '{}')}회 · 기준: {w}회 미만(휩소 아님)")


def _cloud_items(ctx: dict, cfg: dict) -> list[dict]:
    close, top = _fact(ctx, "close"), _fact(ctx, "cloud_top")
    sa, sb = _fact(ctx, "span_a"), _fact(ctx, "span_b")
    hd = _fact(ctx, "high_d_ago")
    D = cfg["assumptions"]["ichimoku_shift"]
    items = []
    if close is not None and top is not None:
        items.append(_verdict(close > top, f"종가 ${close:,.2f} vs 구름 상단 ${top:,.2f} · 기준: 종가 > 구름 상단"))
    else:
        items.append(_verdict(ctx.get("cloud_ok") if "cloud_ok" in ctx else None, "종가 > 구름 상단"))
    if sa is not None and sb is not None:
        items.append(_verdict(sa > sb, f"선행A ${sa:,.2f} vs 선행B ${sb:,.2f} · 기준: 선행A > 선행B(앞구름 양운)"))
    else:
        yang = _fact(ctx, "future_yang")
        items.append(_verdict(yang if yang is not None else ctx.get("future_yang_ok"), "앞구름 양운(선행A > 선행B)"))
    if close is not None and hd is not None:
        items.append(_verdict(close > hd, f"종가 ${close:,.2f} vs {D}일 전 고가 ${hd:,.2f} · 기준: 종가 > {D}일 전 고가(후행스팬 돌파)"))
    else:
        ck = _fact(ctx, "chikou_ok")
        items.append(_verdict(ck if ck is not None else ctx.get("chikou_ok"), f"후행스팬 돌파(종가 > {D}일 전 고가)"))
    return items


def _rule_items(stage: str, ctx: dict, cfg: dict) -> list[dict]:
    """① 우리 규칙: 차수 조건마다 오늘 실제 값과 기준. ✓/✗는 값 비교로만 정한다."""
    a = cfg["assumptions"]
    lv = levels(cfg)
    lo, mid, hi = lv["rsi_oversold"], lv["rsi_mid"], lv["rsi_overbought"]
    facts = ctx.get("facts") or {}
    rsi_now, rsi_prev = _fact(ctx, "rsi_now"), _fact(ctx, "rsi_prev")
    items: list[dict] = []
    if stage == "A1":
        items.append(_verdict(
            None if rsi_prev is None or rsi_now is None else (rsi_prev < lo <= rsi_now),
            f"RSI 어제 {_v(rsi_prev, '{:.1f}')} → 오늘 {_v(rsi_now, '{:.1f}')} · 기준: 어제 {lo:g} 미만 → 오늘 {lo:g} 이상",
        ))
        if "pre_state" in facts:
            pre = facts["pre_state"]
            items.append(_verdict(None if pre is None else pre == "대기",
                                  "보유 없음(대기 상태)" if pre == "대기" else f"{_STATE_HELD.get(pre, pre or '상태')} · 기준: 보유 중이 아닐 것"))
            cu = facts.get("cooldown_until")
            today = facts.get("date")
            if cu is None:
                items.append(_chk("y", f"재진입 대기(쿨다운) 아님 · 기준: 청산 후 {a['reentry_cooldown_days']}거래일 지남"))
            elif today is not None:
                import pandas as pd

                ok = pd.Timestamp(today) >= pd.Timestamp(cu)
                items.append(_verdict(ok, f"재진입 대기 종료일 {_md(cu)} · 기준: 청산 후 {a['reentry_cooldown_days']}거래일 지남"))
        items.append(_earnings_item(ctx))
    elif stage == "A2":
        a1_date, elapsed = _fact(ctx, "a1_date"), _fact(ctx, "elapsed")
        exp = a["a1_to_a2_expiry_days"]
        items.append(_verdict(
            None if elapsed is None else elapsed <= exp,
            f"1차 매수일 {_md(a1_date) or '?'} · 오늘 {_v(elapsed, '{}')}/{exp}거래일째 · 기준: 1차 체결이 확정된 다음 거래일부터, 1차 당일 포함 {exp}거래일째까지",
        ))
        items.append(_gc_item(ctx))
        items.append(_verdict(None if rsi_now is None else lo <= rsi_now < hi, f"RSI {_v(rsi_now, '{:.1f}')} · 기준: {lo:g} 이상 {hi:g} 미만"))
        items.append(_verdict(None if rsi_now is None else rsi_now < hi, f"금지 구간: 골든크로스 날 RSI {_v(rsi_now, '{:.1f}')} · 기준: {hi:g} 미만"))
        items.append(_whipsaw_item(ctx, cfg))
        items.append(_earnings_item(ctx))
        mn = _fact(ctx, "macd_norm")
        g = _fact(ctx, "grade")
        items.append(_chk("i", (
            f"등급 {g or '없음'} · 정규화 MACD {_v(mn, '{:+.2f}')}% · 기준: S {a['s_grade_macd_norm_min_pct']}% 이상, "
            f"A {a['b_grade_macd_norm_max_pct']}%~{a['s_grade_macd_norm_min_pct']}% & {lv['cross_window_days']}일 교차(골든+데드) "
            f"{lv['grade_a_max_crosses']}회 이하, B 그 밖"
        )))
    elif stage == "A3":
        units = facts.get("units") or {}
        pre = facts.get("pre_state")
        if "pre_state" in facts:
            items.append(_verdict(pre == "확인", f"2차 보유 {units.get('2', '?')}주 ({_STATE_HELD.get(pre, pre or '?')}) · 기준: 2차 보유 중"))
        items.extend(_cloud_items(ctx, cfg))
        m, s = _fact(ctx, "macd"), _fact(ctx, "signal")
        if None not in (m, s, rsi_now):
            items.append(_verdict(m > s and rsi_now >= mid, f"MACD {m:+.3f} vs 시그널 {s:+.3f}, RSI {rsi_now:.1f} · 기준: MACD > 시그널 & RSI {mid:g} 이상"))
        else:
            items.append(_verdict(ctx.get("momentum_ok"), f"MACD > 시그널 & RSI {mid:g} 이상"))
        gap, th = _fact(ctx, "gap_pct"), a["gap_filter_pct"]
        items.append(_verdict(None if gap is None else gap < th, f"오늘 시가 갭 {_v(gap, '{:+.1f}')}% · 기준: {th:g}% 미만"))
        items.append(_earnings_item(ctx))
        mode = cfg.get("a3", {}).get("mode", a.get("a3_entry_mode", "breakout"))
        items.append(_chk("i", "진입 방식: 돌파형 — 4가지가 모두 맞은 날 산다" if mode != "pullback"
                          else "진입 방식: 되돌림형 — 4가지 충족 + 구름 상단·기준선 근처까지 내려왔다 양봉 마감"))
    else:  # B
        items.extend(_cloud_items(ctx, cfg))
        items.append(_gc_item(ctx))
        mn, s_min = _fact(ctx, "macd_norm"), a["s_grade_macd_norm_min_pct"]
        items.append(_verdict(None if mn is None else mn >= s_min, f"정규화 MACD {_v(mn, '{:+.2f}')}% · 기준: {s_min}% 이상(0선 근처)"))
        items.append(_verdict(None if rsi_now is None else mid <= rsi_now < hi, f"RSI {_v(rsi_now, '{:.1f}')} · 기준: {mid:g} 이상 {hi:g} 미만"))
        items.append(_whipsaw_item(ctx, cfg))
        items.append(_earnings_item(ctx))
        b_pct = cfg.get("risk", {}).get("b_budget_pct")
        if b_pct is not None:
            items.append(_chk("i", f"위험 예산: 계좌의 {b_pct:g}% (A형 2%의 절반)"))
    return items


def _funnel_sentence(ctx: dict) -> str | None:
    """② 깔때기 문장 — 보고서 "오늘의 깔때기" 구역과 같은 숫자(core.screening_view)."""
    f = ctx.get("funnel")
    if not f or not f.get("steps"):
        return None
    parts = []
    for i, step in enumerate(f["steps"]):
        if i == 0 and step["label"] == "스캔":
            parts.append(f"오늘 {step['count']}종목")
        elif i == 0:
            parts.append(f"{step['label']} {step['count']}종목")
        elif step["label"] == "오늘 추천":
            continue  # 마지막 "점수 n위"가 대신한다
        else:
            parts.append(f"{step['label']} {step['count']}")
    if f.get("rank"):
        parts.append(f"점수 {f['rank']}위")
    return " → ".join(parts)


def _how_items(stage: str, ctx: dict, cfg: dict) -> list[dict]:
    """③ 어떻게 사나: 지정가·갭 보류·수량 계산식·손절가 기준."""
    a = cfg["assumptions"]
    items: list[dict] = []
    close, limit = _fact(ctx, "close"), ctx.get("limit")
    markup = cfg.get("entry", {}).get("limit_markup")
    if close is not None and limit is not None and markup:
        items.append(_chk("i", f"지정가 = 오늘 종가 ${close:,.2f} × {markup:g} = ${limit:,.2f} · 다음 거래일에 이 가격 이하로만 사요"))
    elif limit is not None:
        items.append(_chk("i", f"지정가 ${limit:,.2f}(종가 × {markup or '?'}) · 다음 거래일에 이 가격 이하로만 사요"))
    if stage == "A3":
        items.append(_chk("i", f"보류: 내일 시가가 오늘 종가보다 {a['gap_filter_pct']:g}% 이상 높게 시작하면 주문하지 않아요(갭)"))

    sz = ctx.get("sizing") or {}
    qty = ctx.get("qty") or 0
    frac = _BUY_FRACTION[stage]
    if sz.get("mode") == "live":
        budget, src, fx = sz.get("budget_krw"), sz.get("budget_source"), sz.get("fx_rate")
        if budget and fx and sz.get("ref_price"):
            items.append(_chk("i", (
                f"수량: {src} {budget:,.0f}원 ÷ (기준가 ${sz['ref_price']:,.2f} × 환율 {fx:,.0f}원) = 계획 총 {sz.get('total_shares')}주 "
                f"→ {frac} 몫 = {sz.get('raw_qty')}주" + (f" → 남은 주수로 줄여 {qty}주" if sz.get("raw_qty") != qty else "")
            )))
        elif budget and fx and limit:
            tranche = sz.get("tranche_krw") or 0
            raw = tranche / (limit * fx) if limit and fx else 0
            items.append(_chk("i", f"수량: {src} {budget:,.0f}원 × {frac} = {tranche:,.0f}원 ÷ (지정가 ${limit:,.2f} × 환율 {fx:,.0f}원) = {raw:.2f} → {qty}주(내림)"))
        else:
            items.append(_chk("n" if not qty else "i", "수량: 계획금액이 없어 계산하지 않아요(시트 '계획' 탭 또는 투자현황 기본 금액 필요)"))
    elif sz.get("mode") == "paper":
        tq, rq, fx = sz.get("target_qty"), sz.get("risk_cap_qty"), sz.get("fx_rate")
        slot = sz.get("slot_krw")
        risk_pct = sz.get("risk_pct")
        if fx and slot and limit:
            items.append(_chk("i", (
                f"수량: 슬롯 {slot:,.0f}원 × {frac} ÷ (지정가 ${limit:,.2f} × 환율 {fx:,.0f}원) = 목표 {tq}주, "
                f"위험 상한(계좌의 {risk_pct:.2f}% ÷ 주당 위험) = {rq}주 → 작은 값 {min(tq or 0, rq or 0)}주"
                + (f" → 최종 {qty}주" if qty != min(tq or 0, rq or 0) else "")
            )))
    for reason in sz.get("reduced", []):
        items.append(_chk("n", f"줄어든 이유: {reason}"))

    stop, stop_pct = ctx.get("stop"), ctx.get("stop_pct")
    n = a["swing_low_period"]
    low, low_date = _fact(ctx, "swing_low"), _fact(ctx, "swing_low_date")
    low_txt = f"{_md(low_date)} 저가 ${low:,.2f}" if (low is not None and low_date is not None) else "최저가 날짜 확인 불가"
    if stop is not None:
        if stage == "A1":
            items.append(_chk("i", f"손절가 ${stop:,.2f} = 오늘까지 최근 {n}거래일 최저가({low_txt})"))
        elif stage == "A2":
            items.append(_chk("i", f"손절가 ${stop:,.2f} = 1차 매수일({_md(_fact(ctx, 'a1_date')) or '?'})까지 최근 {n}거래일 최저가({low_txt}) — 1차와 같은 값"))
        elif stage == "A3":
            bot = _fact(ctx, "cloud_bot")
            items.append(_chk("i", f"손절가 ${stop:,.2f} = 1차 기준 {n}일 최저가({low_txt})와 오늘 구름 하단 {('$' + format(bot, ',.2f')) if bot is not None else '?'} 중 높은 값 — 1·2차분에도 같이 적용"))
        else:
            items.append(_chk("i", f"손절가 ${stop:,.2f} = 진입일(오늘)까지 최근 {n}거래일 최저가({low_txt})"))
        if stop_pct is not None and limit is not None:
            items.append(_chk("i", f"손절폭 = (손절가 − 지정가) ÷ 지정가 = {stop_pct:+.1f}% · 종가가 손절가 아래로 마감하면 전량 매도"))
    else:
        items.append(_chk("n", "손절가 계산 불가 — 매수 보류"))
    return items


def explain_buy(stage: str, ctx: dict, cfg: dict) -> dict:
    """매수 신호(A1·A2·A3·B) 한 건의 설명을 만든다.

    ctx 키: kr, score, grade, vol_ratio, cloud_thickness_pct, bb_width_pct, limit,
        stop, stop_pct, qty, risk_capped(bool), limited(bool),
        (A1) rsi_prev, rsi_now, a2_expiry_date
        (A2) rsi_now, macd_norm, a1_date
        (A3) cloud_ok, future_yang_ok, chikou_ok, momentum_ok, gap_pct
        (B)  rsi_now, macd_norm
        facts(있으면) — core.screening_view.buy_facts 결과(오늘 실제 지표 값). ✓/✗를 값 비교로 정한다.
        funnel(있으면) — {"steps", "rank"}: 보고서 "오늘의 깔때기"와 같은 숫자
        sizing(있으면) — {"mode": "live"|"paper", 수량 계산 재료, "reduced": [줄어든 이유]}
    출력: {"title","badge","checks","body","next","sections","source"}
        sections: [{"key","title","items":[체크...],"text":str|None}, ...] ① 규칙 ② 왜 ③ 어떻게 ④ 다음
    """
    kr = ctx.get("kr", "")
    assumptions = cfg["assumptions"]
    ctx = {**ctx, "_cfg": cfg}
    lv = levels(cfg)
    title = f"{kr} · {_BUY_TITLE[stage]}"
    rule_items = _rule_items(stage, ctx, cfg)
    deadline = _md(_fact(ctx, "a2_deadline")) or ctx.get("a2_expiry_date")

    if stage == "A1":
        rsi_prev, rsi_now = _fact(ctx, "rsi_prev"), _fact(ctx, "rsi_now")
        body = (
            f"RSI는 주가가 최근 얼마나 많이 올랐고 내렸는지를 0~100으로 나타낸 값이에요. "
            f"<b>{lv['rsi_oversold']:g} 아래는 \"너무 많이 떨어진 상태(과매도)\"</b>인데, {kr}는 {rsi_prev}까지 내려갔다가 "
            f"오늘 {rsi_now}로 올라왔어요. 떨어지던 힘이 멈추고 반등을 시작할 수 있다는 <b>첫 신호</b>예요. "
            f"아직 확실하지 않아서 가장 적은 금액(1/9)만 먼저 사 봐요."
        )
        expiry_txt = deadline if deadline else f"{assumptions['a1_to_a2_expiry_days']}거래일 안"
        next_ = (
            f"{expiry_txt}까지 MACD 골든크로스가 나오면 2차 매수(2/9). 안 나오면 반등 실패로 보고 1차분을 팔아요(만료). "
            f"종가가 손절가 {_usd(ctx.get('stop'))} 아래로 내려가면 바로 전량 매도. 산 뒤에는 체결 탭에 적어야 내일부터 2차 후보가 돼요."
        )
    elif stage == "A2":
        a1_date = _md(_fact(ctx, "a1_date")) or ctx.get("a1_date")
        body = (
            "MACD선이 신호선을 아래에서 위로 뚫는 순간이 <b>골든크로스</b>예요. 방향이 진짜로 바뀌었다는 뜻이라 "
            f"2차(2/9)를 더 사요. {kr}는 1차 매수" + (f"({a1_date})" if a1_date else "") + " 후 골든크로스가 나왔어요. "
            f"등급은 {ctx.get('grade') or '미확정'}등급이에요."
        )
        next_ = "일목 구름 4조건(추세 확정)이 맞으면 3차 매수(6/9). 안 맞으면 계속 지켜봐요."
    elif stage == "A3":
        met = sum(1 for c in rule_items[1:5] if c["level"] == "y") if (ctx.get("facts") or {}).get("pre_state") else None
        body = (
            "3차(가장 큰 6/9)는 <b>상승 추세가 자리 잡았다는 4가지 조건</b>이 모두 맞아야 사요: 구름 위 · 앞구름 양운 · "
            f"후행스팬 돌파 · MACD 골든크로스+RSI {lv['rsi_mid']:g} 이상. {kr}는 오늘 "
            + (f"4가지 중 {met}가지를 충족했어요." if met is not None and met != 4 else "4가지를 모두 충족했어요.")
        )
        next_ = "3차까지 다 샀어요. 이제 매도 신호(E1→E2→E3)를 기다려요. 손절가는 10일 최저가와 구름 하단 중 더 높은 값이에요."
    else:  # B
        b_pct = cfg["risk"]["b_budget_pct"]
        body = (
            "이미 <b>상승 추세(주가가 일목 구름 위)</b>에 있는 종목이 잠깐 쉬었다가 다시 오르기 시작했어요. MACD가 0선 "
            "근처에서 골든크로스를 내면 \"쉬는 구간이 끝나고 다시 달린다\"는 뜻이에요. 추세가 이미 확인된 종목이라 "
            f"1:2:6으로 나누지 않고 한 번에 사지만, <b>잃을 수 있는 금액은 총자금의 {b_pct:g}%</b>로 잡아요."
        )
        next_ = (
            f"손절가 {_usd(ctx.get('stop'))}(진입일 기준 {assumptions['swing_low_period']}일 최저가). "
            f"이후 MACD 데드크로스→RSI {lv['rsi_mid']:g} 이탈→구름 이탈 순서로 나눠 팔아요."
        )

    # 기존 checks(요약 칩): ① 규칙 확인과 같은 값 비교 결과를 그대로 쓴다(고정 "y" 없음).
    checks: list[dict] = list(rule_items)
    vol_ratio = ctx.get("vol_ratio")
    if stage == "A1" and vol_ratio is not None:
        if vol_ratio >= 1.5:
            checks.append(_chk("y", f"거래량 평균의 {vol_ratio}배 · 기준(1.5배) 충족"))
        elif vol_ratio >= 1.2:
            checks.append(_chk("i", f"거래량 평균의 {vol_ratio}배 · 소폭 가산 기준(1.2배)만 충족"))
        else:
            checks.append(_chk("n", f"거래량 평균의 {vol_ratio}배 · 기준(1.2배) 미달"))
    if ctx.get("risk_capped") and (ctx.get("qty") or 0) > 0:
        checks.append(_chk("n", "손절폭이 커서 목표 수량보다 줄었어요(위험 한도)"))
    if ctx.get("limited"):
        checks.append(_chk("n", "오늘 남은 자금 한도가 부족해 수량이 " + ("줄었어요" if (ctx.get("qty") or 0) > 0 else "0으로 보류됐어요") + "(자금 한도)"))
    score_txt = "점수 " + _score_breakdown(ctx, cfg, stage)
    checks.append(_chk("i", score_txt))

    funnel_txt = _funnel_sentence(ctx)
    # 쉬운 설명(body)은 ① 규칙 바로 아래, ②에는 깔때기 문장과 점수 내역만 둔다.
    sections = [
        {"key": "rule", "title": "① 우리 규칙", "items": rule_items, "text": body},
        {"key": "why", "title": "② 왜 이 종목인가", "items": [_chk("i", score_txt)], "text": f"<b>{funnel_txt}</b>" if funnel_txt else None},
        {"key": "how", "title": "③ 어떻게 사나", "items": _how_items(stage, ctx, cfg), "text": None},
        {"key": "next", "title": "④ 다음 단계", "items": [], "text": next_},
    ]

    badge = (_BUY_BADGE_LIVE if (ctx.get("sizing") or {}).get("mode") == "live" else _BUY_BADGE)[stage]
    return {
        "title": title, "badge": badge, "checks": checks, "body": body, "next": next_,
        "sections": sections, "source": _BUY_SOURCE[stage],
    }


# ── 매도 (STOP·E1·E2·E3·A1_EXPIRE) ───────────────────────────────────────

_SELL_TITLE = {
    "STOP": "손절", "E1": "E1 모멘텀 약화", "E2": "E2 추세 약화",
    "E3": "E3 구조 붕괴", "A1_EXPIRE": "1차 만료",
}
_SELL_BADGE = {
    "STOP": "전량 · 최우선", "E1": "1차분 매도", "E2": "2차분 매도",
    "E3": "남은 전량 매도", "A1_EXPIRE": "1차분 전량 매도",
}


def explain_sell(kind: str, ctx: dict, cfg: dict) -> dict:
    """매도·손절 신호(STOP·E1·E2·E3·A1_EXPIRE) 한 건의 설명을 만든다.

    ctx 키: kr, close, stop, (E1) macd, signal, (E2) rsi_prev, rsi_now,
        (E3) cloud_bot, chikou_broken(bool)
    """
    kr = ctx.get("kr", "")
    assumptions = cfg["assumptions"]
    title = f"{kr} · {_SELL_TITLE[kind]}"
    cooldown_days = assumptions["reentry_cooldown_days"]
    cooldown_note = f"{kr}는 앞으로 {cooldown_days}거래일 동안 다시 사지 않아요(쿨다운)."

    if kind == "STOP":
        checks = [
            _chk("n", f"종가 {_usd(ctx.get('close'))} < 손절가 {_usd(ctx.get('stop'))}"),
            _chk("i", f"손절가 기준: 최근 {assumptions['swing_low_period']}거래일 최저가(3차까지 매수했으면 구름 하단이 더 높으면 그 값)"),
        ]
        body = (
            "손절가는 <b>\"여기까지 내려오면 내 예상이 틀렸다\"</b>고 미리 정해 둔 가격이에요. 그 가격을 종가가 깨면 "
            "반등이 실패한 것으로 보고, 손실이 더 커지기 전에 전량 팔아요. 다른 어떤 신호보다 먼저 지켜요."
        )
        next_ = f"{cooldown_note} 손실은 처음에 정한 한도(계좌 위험 예산) 안에서 끝나요."
    elif kind == "A1_EXPIRE":
        checks = [_chk("n", f"1차 매수 후 {assumptions['a1_to_a2_expiry_days']}거래일 안에 2차 조건(MACD 골든크로스)이 안 나옴")]
        body = (
            "1차(정찰)만 산 상태에서 정해진 기간 안에 방향 전환(MACD 골든크로스)이 확인되지 않았어요. 반등이 이어지지 "
            "않는다고 보고 1차분을 정리해요."
        )
        next_ = cooldown_note
    elif kind == "E1":
        macd, signal = ctx.get("macd"), ctx.get("signal")
        checks = [_chk("n", "MACD 데드크로스 발생" + (f" · MACD {macd:+.2f} < 신호선 {signal:+.2f}" if macd is not None and signal is not None else ""))]
        body = (
            "MACD선이 신호선을 위에서 아래로 뚫으면 <b>데드크로스</b>예요 — 오르는 힘이 약해지기 시작했다는 뜻이에요. "
            "아직 추세가 완전히 무너진 건 아니라서 가장 먼저 산 1차분(11%)만 팔아요."
        )
        next_ = "남은 수량(2·3차분)은 다음 매도 신호(E2·E3)를 기다려요."
    else:  # E3
        checks = [
            _chk("n", f"종가 {_usd(ctx.get('close'))} < 구름 하단 {_usd(ctx.get('cloud_bot'))}"),
        ]
        if ctx.get("chikou_broken"):
            checks.append(_chk("n", "후행스팬이 선행스팬 B 아래"))
        body = (
            "일목 구름은 \"추세의 바닥\"과 같은 역할을 해요. 주가가 <b>구름 아래로 떨어지면 상승 추세 자체가 무너졌다</b>는 "
            "뜻이라, 손실 중이더라도 더 기다리지 않고 남은 주식을 모두 팔아요. 매도 신호 중 가장 강한 단계예요."
        )
        next_ = cooldown_note

    return {"title": title, "badge": _SELL_BADGE[kind], "checks": checks, "body": body, "next": next_}


# ── 제외 (매매 금지·한도) ─────────────────────────────────────────────────


def _classify_ban_reason(reason: str) -> str:
    if "골든크로스 당일 RSI" in reason or "RSI 70" in reason:
        return "OVERHEAT"
    if "휩소" in reason:
        return "WHIPSAW"
    if "구름 안" in reason:
        return "IN_CLOUD"
    if "앞구름 음운" in reason:
        return "FUTURE_YIN"
    if "시가 갭" in reason:
        return "GAP"
    if "실적" in reason:
        return "EARNINGS"
    return "OTHER"


_FILTER_TITLE = {
    "OVERHEAT": "과열로 제외", "WHIPSAW": "횡보장으로 제외", "IN_CLOUD": "추세 미확정으로 제외",
    "FUTURE_YIN": "추세 미확정으로 제외", "GAP": "시가 갭으로 제외", "EARNINGS": "실적 발표가 가까워 제외",
    "LIMIT": "동시 보유 한도로 제외", "OTHER": "매매 금지 조건으로 제외",
}
# 표시 우선순위 — 기술적 사유를 실적보다 앞세운다(둘 다 있으면 기술적 사유가 더 눈에 띄는 원인이라).
_FILTER_ORDER = ("OVERHEAT", "WHIPSAW", "GAP", "IN_CLOUD", "FUTURE_YIN", "OTHER", "EARNINGS")


def explain_filtered(reasons: list[str], ctx: dict, cfg: dict) -> dict:
    """걸러진 신호(BLOCKED) 한 건의 설명을 만든다.

    ctx 키: kr, score, stage_signal_ok(이번 신호 자체는 충족했는지, 보통 True),
        rsi_now, gc_count_20d, gap_pct, earnings_date, max_concurrent
    reasons: core/filters.py ban_reasons()가 낸 원문 목록, 또는 동시 보유 한도면 ["동시 보유 종목 수 한도 초과"]
    """
    kr = ctx.get("kr", "")
    assumptions = cfg["assumptions"]
    lv = levels(cfg)
    hi, days, earn = lv["rsi_overbought"], lv["cross_window_days"], lv["earnings_filter_trading_days"]
    codes = sorted({_classify_ban_reason(r) for r in reasons}, key=lambda c: _FILTER_ORDER.index(c) if c in _FILTER_ORDER else 99)
    if any("한도 초과" in r for r in reasons):
        codes = ["LIMIT"]
    primary = codes[0] if codes else "OTHER"
    title = f"{kr} · {_FILTER_TITLE[primary]}"

    checks: list[dict] = []
    stage_ok = ctx.get("stage_signal_ok", True)
    if stage_ok and primary != "LIMIT":
        checks.append(_chk("y", "이번 신호 조건 자체는 충족"))

    bodies: list[str] = []
    for code in codes:
        if code == "OVERHEAT":
            rsi_now = ctx.get("rsi_now")
            checks.append(_chk("n", f"RSI {rsi_now} · {hi:g} 이상" if rsi_now is not None else f"RSI {hi:g} 이상"))
            bodies.append(
                f"RSI {hi:g} 이상은 <b>\"이미 짧은 기간에 많이 오른 상태(과매수)\"</b>예요. 신호가 나왔어도 지금 사면 비싸게 "
                f"따라 사는 셈이라, 잠깐만 쉬어도 손절에 걸리기 쉬워요. RSI가 {hi:g} 아래로 내려온 뒤 다시 신호가 나면 매수해요."
            )
        elif code == "WHIPSAW":
            gc_count = ctx.get("gc_count_20d")
            limit = assumptions["whipsaw_max_crosses_20d"]
            checks.append(_chk("n", f"최근 {days}거래일 MACD 교차 {gc_count}회 · 기준({limit}회) 이상" if gc_count is not None else f"최근 {days}거래일 MACD 교차 기준({limit}회) 이상"))
            bodies.append(
                "주가가 한 방향으로 가지 않고 옆으로 오르내리면(횡보), MACD가 골든크로스와 데드크로스를 번갈아 자주 "
                "내요. 이런 때 나오는 골든크로스는 <b>가짜 신호일 가능성이 높아서</b> 매수하지 않아요."
            )
        elif code == "GAP":
            gap_pct = ctx.get("gap_pct")
            threshold = assumptions["gap_filter_pct"]
            checks.append(_chk("n", f"시가 갭 {gap_pct:+.1f}% · 기준({threshold}%) 이상" if gap_pct is not None else f"시가 갭 기준({threshold}%) 이상"))
            bodies.append(
                f"시가가 전날 종가보다 {threshold}% 이상 뛰어(갭) 열렸어요. 이미 크게 오른 가격에 진입하면 되돌림(눌림)이 "
                "와도 손절가까지 거리가 왜곡될 수 있어 오늘은 매수를 보류해요."
            )
        elif code in ("IN_CLOUD", "FUTURE_YIN"):
            checks.append(_chk("n", "종가가 구름 안에 있음" if code == "IN_CLOUD" else "앞구름이 음운(하락 구름)"))
            bodies.append("추세가 아직 위로 확실히 자리 잡지 않았어요(일목 구름 조건 미충족). 추세가 분명해질 때까지 기다려요.")
        elif code == "LIMIT":
            max_concurrent = ctx.get("max_concurrent")
            score = ctx.get("score")
            checks.append(_chk("i", f"오늘 점수 {score}점" + (f" · 동시 보유 한도 {max_concurrent}종목" if max_concurrent is not None else "")))
            bodies.append(
                "동시에 보유할 수 있는 종목 수가 이미 다 찼어요. 오늘 신호가 여러 개일 때는 <b>점수가 높은 종목부터</b> "
                "사고, 한도를 넘는 나머지는 오늘 신호를 넘겨요(다음 신호를 기다리는 건 아니에요)."
            )
        elif code == "EARNINGS":
            earnings_date = ctx.get("earnings_date")
            checks.append(_chk("n", f"실적 발표 {earnings_date}" if earnings_date else f"실적 발표 {earn}거래일 이내"))
            bodies.append(
                f"<b>실적 발표가 {earn}거래일 안</b>에 있어요. 실적 발표 다음 날에는 주가가 하루에 크게 뛰거나 빠지는 일이 "
                "흔해서, 미리 정한 손절가가 소용없을 수 있어요. 발표가 지난 뒤 조건이 다시 맞으면 그때 신호가 나요."
            )

    body = " ".join(bodies)
    next_ = None if primary == "LIMIT" else "조건이 다시 맞으면(제외 사유가 없어지면) 신호가 다시 나요."
    return {"title": title, "badge": None, "checks": checks, "body": body, "next": next_}


# ── 경고 (매도 아님) ───────────────────────────────────────────────────────


def explain_warn(kind: str, ctx: dict, cfg: dict) -> dict:
    """경고(매도 신호 아님) 한 건의 설명을 만든다.

    kind: "KIJUN_BREACH" | "RSI_RELIEF" | "TARGET_REACHED" | "STOP_NEAR" | "STOP_CHANGED" | "STOP_NEEDED"
    ctx 키: kr, close, kijun, rsi_prev, rsi_now, avg_entry, stop, stop_near_pct, old_stop, new_stop
    """
    kr = ctx.get("kr", "")
    if kind == "KIJUN_BREACH":
        title = f"{kr} · 기준선 이탈 (조기 경보)"
        checks = [_chk("n", f"종가 {_usd(ctx.get('close'))} < 기준선 {_usd(ctx.get('kijun'))}")]
        body = (
            "일목의 <b>기준선(최근 26일 중간값)</b> 아래로 종가가 내려왔어요. 아직 매도 신호는 아니지만, 이대로 구름 "
            "아래까지 떨어지면 전량 매도(E3)가 나와요. 손절 예약 주문이 제대로 걸려 있는지 확인해 두세요."
        )
        next_ = None
    elif kind == "RSI_RELIEF":
        title = f"{kr} · RSI 과열 해소 (참고용)"
        hi = levels(cfg)["rsi_overbought"]
        checks = [_chk("i", f"RSI {ctx.get('rsi_prev')} → {ctx.get('rsi_now')} · {hi:g} 아래로 내려옴")]
        body = f"RSI가 {hi:g} 이상(과매수)이었다가 다시 {hi:g} 아래로 내려왔어요. 오르는 힘이 잠시 식었다는 뜻으로, 매도 신호는 아니에요."
        next_ = None
    elif kind == "TARGET_REACHED":
        title = f"{kr} · 목표 도달 (참고용)"
        avg_entry, stop, close = ctx.get("avg_entry"), ctx.get("stop"), ctx.get("close")
        risk = (avg_entry - stop) if (avg_entry is not None and stop is not None) else None
        checks = [_chk("y", f"평균단가 {_usd(avg_entry)} · 손절폭의 2배(2R) 이상 수익")]
        body = (
            "처음에 감수하기로 한 손실폭(손절가까지 거리, 1R)의 <b>2배만큼 수익</b>이 났어요. 매도 신호는 아니고, "
            "전략은 추세가 약해지는 신호(MACD 데드크로스 등)가 나올 때까지 들고 가요. 수익 일부를 먼저 챙기고 싶다면 "
            "직접 판단하세요."
        )
        next_ = None
    elif kind == "STOP_NEAR":
        title = f"{kr} · 손절 근접"
        checks = [_chk("n", f"종가 {_usd(ctx.get('close'))} · 손절가 {_usd(ctx.get('stop'))}까지 {ctx.get('stop_near_pct')}% 이내")]
        body = "종가가 손절가에 가까워졌어요. 증권사에 건 손절 예약 주문이 지금 손절가와 같은지 확인하세요."
        next_ = None
    elif kind == "STOP_CHANGED":
        title = f"{kr} · 손절 예약 변경"
        checks = [_chk("i", f"손절가 {_usd(ctx.get('old_stop'))} → {_usd(ctx.get('new_stop'))}")]
        body = "3차 매수로 손절가가 구름 하단 기준으로 올라갔어요(원래보다 손실이 줄어드는 방향). 증권사 손절 예약 가격을 새 값으로 바꿔 두세요."
        next_ = None
    else:  # STOP_NEEDED
        title = f"{kr} · 손절 예약 필요"
        checks = [_chk("i", f"오늘 체결로 손절가 {_usd(ctx.get('stop'))}이 새로 생김")]
        body = "오늘 새로 체결된 수량이 있어요. 증권사에 손절 예약 주문을 아직 걸지 않았다면 지금 걸어 두세요."
        next_ = None

    return {"title": title, "badge": None, "checks": checks, "body": body, "next": next_}


# ── 관찰 (다음 신호를 기다리는 중) ─────────────────────────────────────────


def explain_watch(kind: str, ctx: dict, cfg: dict) -> dict:
    """관찰 목록(다음 신호를 기다리는 중) 한 건의 설명을 만든다.

    kind: "WAIT_A2" | "WAIT_A3"
    ctx 키: kr, (WAIT_A2) macd_diff, remaining_days, expiry_date
        (WAIT_A3) cloud_ok, future_yang_ok, chikou_ok, rsi_now
    """
    kr = ctx.get("kr", "")
    assumptions = cfg["assumptions"]
    if kind == "WAIT_A2":
        title = f"{kr} · 2차 신호 기다리는 중"
        diff = ctx.get("macd_diff")
        checks = [_chk("n", f"MACD선 − 신호선 = {diff:+.2f} · 아직 신호선 아래" if diff is not None else "MACD 골든크로스 전")]
        if diff is not None and diff > -0.3:
            checks.append(_chk("i", "간격이 좁혀지는 중"))
        expiry_txt = ctx.get("expiry_date") or f"{assumptions['a1_to_a2_expiry_days']}거래일 안"
        body = (
            "MACD선이 신호선을 아래에서 위로 뚫는 순간이 <b>골든크로스</b>예요. " + (f"두 선의 차이가 {diff:+.2f}로 0에 가까워지고 있어서, " if diff is not None else "") +
            f"0을 넘으면 2차 매수 신호가 나요. {expiry_txt}까지 안 넘으면 반등 힘이 부족한 것으로 보고 1차분을 팔아요."
        )
        next_ = None
    else:  # WAIT_A3
        title = f"{kr} · 3차 신호 기다리는 중"
        checks = [
            _chk("y" if ctx.get("cloud_ok") else "n", "종가가 구름 위"),
            _chk("y" if ctx.get("future_yang_ok") else "n", "앞구름이 양운(상승 구름)"),
            _chk("y" if ctx.get("chikou_ok") else "n", f"후행스팬이 {assumptions['ichimoku_shift']}일 전 가격 위"),
        ]
        rsi_now = ctx.get("rsi_now")
        mid = levels(cfg)["rsi_mid"]
        checks.append(_chk("y" if (rsi_now is not None and rsi_now >= mid) else "n", f"RSI {rsi_now} · {mid:g} 미만" if (rsi_now is not None and rsi_now < mid) else f"RSI {rsi_now} · {mid:g} 이상"))
        body = (
            "3차(가장 큰 6/9)는 <b>상승 추세가 자리 잡았다는 4가지 조건</b>이 모두 맞아야 사요. 지금 몇 가지는 맞았고 "
            "나머지가 남았어요. 모두 맞으면 3차 매수 신호가 나요."
        )
        next_ = None

    return {"title": title, "badge": None, "checks": checks, "body": body, "next": next_}
