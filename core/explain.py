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

호출부(engine/daily.py)가 채워 넘기는 ctx dict의 키는 각 함수 docstring에
적어 둔다. 없어도 되는 값은 None으로 두면 그 항목만 빠진다.
"""

from __future__ import annotations

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


def _score_breakdown(ctx: dict, cfg: dict) -> str:
    """실제 우선순위 점수(core/filters.py priority_score) 구성 요소를 그대로 문장으로 옮긴다."""
    parts: list[str] = []
    grade_letter = ctx.get("grade")
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


def explain_buy(stage: str, ctx: dict, cfg: dict) -> dict:
    """매수 신호(A1·A2·A3·B) 한 건의 설명을 만든다.

    ctx 키: kr, score, grade, vol_ratio, cloud_thickness_pct, bb_width_pct,
        stop, stop_pct, qty, risk_capped(bool), limited(bool),
        (A1) rsi_prev, rsi_now, a2_expiry_date
        (A2) rsi_now, macd_norm, a1_date
        (A3) cloud_ok, future_yang_ok, chikou_ok, momentum_ok, gap_pct
        (B)  rsi_now, macd_norm
    """
    kr = ctx.get("kr", "")
    assumptions = cfg["assumptions"]
    checks: list[dict] = []
    title = f"{kr} · {_BUY_TITLE[stage]}"

    if stage == "A1":
        rsi_prev, rsi_now = ctx.get("rsi_prev"), ctx.get("rsi_now")
        checks.append(_chk("y", f"RSI {rsi_prev} → {rsi_now} · 30을 아래에서 위로 넘음"))
        vol_ratio = ctx.get("vol_ratio")
        if vol_ratio is not None:
            if vol_ratio >= 1.5:
                checks.append(_chk("y", f"거래량 평균의 {vol_ratio}배 · 기준(1.5배) 충족"))
            elif vol_ratio >= 1.2:
                checks.append(_chk("i", f"거래량 평균의 {vol_ratio}배 · 소폭 가산 기준(1.2배)만 충족"))
            else:
                checks.append(_chk("n", f"거래량 평균의 {vol_ratio}배 · 기준(1.2배) 미달"))
        body = (
            f"RSI는 주가가 최근 얼마나 많이 올랐고 내렸는지를 0~100으로 나타낸 값이에요. "
            f"<b>30 아래는 \"너무 많이 떨어진 상태(과매도)\"</b>인데, {kr}는 {rsi_prev}까지 내려갔다가 "
            f"오늘 {rsi_now}로 올라왔어요. 떨어지던 힘이 멈추고 반등을 시작할 수 있다는 <b>첫 신호</b>예요. "
            f"아직 확실하지 않아서 가장 적은 금액(1/9)만 먼저 사 봐요."
        )
        expiry = ctx.get("a2_expiry_date")
        expiry_txt = expiry if expiry else f"{assumptions['a1_to_a2_expiry_days']}거래일 안"
        next_ = (
            f"{expiry_txt}까지 MACD 골든크로스가 나오면 2차 매수(2/9). 안 나오면 반등 실패로 보고 1차분을 팔아요(만료). "
            f"종가가 손절가 {_usd(ctx.get('stop'))} 아래로 내려가면 바로 전량 매도."
        )
    elif stage == "A2":
        rsi_now, macd_norm = ctx.get("rsi_now"), ctx.get("macd_norm")
        checks.append(_chk("y", "MACD 골든크로스 발생"))
        s_min = assumptions["s_grade_macd_norm_min_pct"]
        b_max = assumptions["b_grade_macd_norm_max_pct"]
        if macd_norm is not None:
            if macd_norm >= s_min:
                checks.append(_chk("y", f"MACD 정규화 {macd_norm:+.2f}% · S등급 기준({s_min}%) 이상"))
            elif macd_norm >= b_max:
                checks.append(_chk("i", f"MACD 정규화 {macd_norm:+.2f}% · A등급 구간"))
            else:
                checks.append(_chk("n", f"MACD 정규화 {macd_norm:+.2f}% · B등급 구간(신뢰 낮음)"))
        if rsi_now is not None:
            checks.append(_chk("y", f"RSI {rsi_now} · 30~70 사이"))
        a1_date = ctx.get("a1_date")
        body = (
            "MACD선이 신호선을 아래에서 위로 뚫는 순간이 <b>골든크로스</b>예요. 방향이 진짜로 바뀌었다는 뜻이라 "
            f"2차(2/9)를 더 사요. {kr}는 1차 매수" + (f"({a1_date})" if a1_date else "") + " 후 골든크로스가 나왔어요. "
            f"등급은 {ctx.get('grade') or '미확정'}등급이에요."
        )
        next_ = "일목 구름 4조건(추세 확정)이 맞으면 3차 매수(6/9). 안 맞으면 계속 지켜봐요."
    elif stage == "A3":
        checks.append(_chk("y" if ctx.get("cloud_ok") else "n", f"종가 {_usd(ctx.get('limit'))} " + (">" if ctx.get("cloud_ok") else "≤") + " 구름 상단 · 상승 추세 안"))
        checks.append(_chk("y" if ctx.get("future_yang_ok") else "n", "앞구름 양운(선행스팬 A > B)"))
        checks.append(_chk("y" if ctx.get("chikou_ok") else "n", f"후행스팬이 {assumptions['ichimoku_shift']}일 전 고가 위(돌파)"))
        checks.append(_chk("y" if ctx.get("momentum_ok") else "n", "MACD 골든크로스 상태 + RSI 50 이상"))
        gap_pct = ctx.get("gap_pct")
        if gap_pct is not None and abs(gap_pct) >= 1:
            checks.append(_chk("i", f"시가 갭 {gap_pct:+.1f}%"))
        body = (
            "3차(가장 큰 6/9)는 <b>상승 추세가 자리 잡았다는 4가지 조건</b>이 모두 맞아야 사요: 구름 위 · 앞구름 양운 · "
            f"후행스팬 돌파 · MACD 골든크로스+RSI 50 이상. {kr}는 오늘 4가지를 모두 충족했어요."
        )
        next_ = "3차까지 다 샀어요. 이제 매도 신호(E1→E2→E3)를 기다려요. 손절가는 10일 최저가와 구름 하단 중 더 높은 값이에요."
    else:  # B
        rsi_now, macd_norm = ctx.get("rsi_now"), ctx.get("macd_norm")
        s_min = assumptions["s_grade_macd_norm_min_pct"]
        checks.append(_chk("y", f"종가 {_usd(ctx.get('limit'))} > 구름 상단 · 상승 추세 안"))
        checks.append(_chk("y", "MACD 골든크로스 · 0선 근처에서 발생" + (f"({macd_norm:+.2f}%)" if macd_norm is not None else "")))
        checks.append(_chk("y", f"RSI {rsi_now} · 50~70 사이 (과열 아님)" if rsi_now is not None else "RSI 50~70 사이"))
        b_pct = cfg["risk"]["b_budget_pct"]
        body = (
            "이미 <b>상승 추세(주가가 일목 구름 위)</b>에 있는 종목이 잠깐 쉬었다가 다시 오르기 시작했어요. MACD가 0선 "
            "근처에서 골든크로스를 내면 \"쉬는 구간이 끝나고 다시 달린다\"는 뜻이에요. 추세가 이미 확인된 종목이라 "
            f"1:2:6으로 나누지 않고 한 번에 사지만, <b>잃을 수 있는 금액은 총자금의 {b_pct:g}%</b>로 잡아요."
        )
        next_ = (
            f"손절가 {_usd(ctx.get('stop'))}(진입일 기준 {assumptions['swing_low_period']}일 최저가). "
            "이후 MACD 데드크로스→RSI 50 이탈→구름 이탈 순서로 나눠 팔아요."
        )

    if ctx.get("risk_capped") and (ctx.get("qty") or 0) > 0:
        checks.append(_chk("n", "손절폭이 커서 목표 수량보다 줄었어요(위험 한도)"))
    if ctx.get("limited"):
        checks.append(_chk("n", "오늘 남은 자금 한도가 부족해 수량이 " + ("줄었어요" if (ctx.get("qty") or 0) > 0 else "0으로 보류됐어요") + "(자금 한도)"))

    checks.append(_chk("i", "점수 " + _score_breakdown(ctx, cfg)))

    return {"title": title, "badge": _BUY_BADGE[stage], "checks": checks, "body": body, "next": next_}


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
    if "RSI 70" in reason:
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
            checks.append(_chk("n", f"RSI {rsi_now} · 70 초과" if rsi_now is not None else "RSI 70 초과"))
            bodies.append(
                "RSI 70 이상은 <b>\"이미 짧은 기간에 많이 오른 상태(과매수)\"</b>예요. 신호가 나왔어도 지금 사면 비싸게 "
                "따라 사는 셈이라, 잠깐만 쉬어도 손절에 걸리기 쉬워요. RSI가 70 아래로 내려온 뒤 다시 신호가 나면 매수해요."
            )
        elif code == "WHIPSAW":
            gc_count = ctx.get("gc_count_20d")
            limit = assumptions["whipsaw_max_crosses_20d"]
            checks.append(_chk("n", f"최근 20거래일 MACD 교차 {gc_count}회 · 기준({limit}회) 이상" if gc_count is not None else f"최근 20거래일 MACD 교차 기준({limit}회) 이상"))
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
            checks.append(_chk("n", f"실적 발표 {earnings_date}" if earnings_date else "실적 발표 3거래일 이내"))
            bodies.append(
                "<b>실적 발표가 3거래일 안</b>에 있어요. 실적 발표 다음 날에는 주가가 하루에 크게 뛰거나 빠지는 일이 "
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
        checks = [_chk("i", f"RSI {ctx.get('rsi_prev')} → {ctx.get('rsi_now')} · 70 아래로 내려옴")]
        body = "RSI가 70 이상(과매수)이었다가 다시 70 아래로 내려왔어요. 오르는 힘이 잠시 식었다는 뜻으로, 매도 신호는 아니에요."
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
        checks.append(_chk("y" if (rsi_now is not None and rsi_now >= 50) else "n", f"RSI {rsi_now} · 50 미만" if (rsi_now is not None and rsi_now < 50) else f"RSI {rsi_now} · 50 이상"))
        body = (
            "3차(가장 큰 6/9)는 <b>상승 추세가 자리 잡았다는 4가지 조건</b>이 모두 맞아야 사요. 지금 몇 가지는 맞았고 "
            "나머지가 남았어요. 모두 맞으면 3차 매수 신호가 나요."
        )
        next_ = None

    return {"title": title, "badge": None, "checks": checks, "body": body, "next": next_}
