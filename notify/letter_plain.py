"""아침 편지 쉬운 문장판 (2026-10-23, feat/letter-plain) — 텔레그램 첨부 설명과 HTML 맨 위가 같은 문장을 쓴다.

표시 전용 순수 함수: 이미 계산된 summary를 읽어 문장만 만든다(계산·판단 로직은 notify.letter·engine 그대로).
문장 규칙: "~요" 체, 한 줄에 한 가지, 짧게. 약어 대신 쉬운 말("200일선" → "장기 평균선(200일)").
금액은 만 원 단위 반올림. 이모지 없음. 단체방판(edition="group")은 [내 돈]·종목명·개인 금액 없음.

모양: 제목 → (문제 있을 때만) "자료 문제: …" → [오늘 할 일] → [내 돈](개인판) → [시장] → [가상 실험 — 실제 돈 아님] → 맨 아래 "기준: …".
"""

from __future__ import annotations

import pandas as pd

from notify import letter

PERSONAL, GROUP = letter.PERSONAL, letter.GROUP
H_TODO, H_MONEY, H_MARKET, H_LAB = "[오늘 할 일]", "[내 돈]", "[시장]", "[가상 실험 — 실제 돈 아님]"
TODO_NONE = "없어요. 적립은 그대로 두면 돼요."
SAVINGS = ["오늘은 저축일이에요.", "들어온 돈으로 ISA에서 TIGER 미국나스닥100을 사세요.", "한도를 넘는 돈은 일반 계좌 QQQM으로요."]
CRASH = ["시장이 크게 빠진 구간이에요.", "우리 실험의 손해는 대부분 이럴 때 팔았다가 반등을 놓쳐서 생겼어요.", "팔지 말고 적립만 그대로 하세요."]
MONEY_EMPTY = "시트 노란 칸에 보유 수와 평균단가를 넣으면 여기에 보여 드려요."
CAPTION_LIMIT = 1024  # 텔레그램 첨부 설명 최대 글자 수
CAPTION_MORE = "나머지는 첨부 파일에서 볼 수 있어요."


# ── 숫자 표현 ────────────────────────────────────────────────────────────────


def josa(word: str, pair: tuple[str, str]) -> str:
    """word + 조사. pair = (받침 있을 때, 없을 때) 예: ("을", "를"), ("이", "가"), ("은", "는").
    영문 티커는 마지막 글자 읽는 소리로(L 엘·M 엠·N 엔·R 알만 받침), 한글은 받침 유무, 숫자·기타는 받침 없음으로 본다."""
    if not word:
        return word
    ch = word[-1]
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
    else:
        has = ch.upper() in "LMNR"
    return word + (pair[0] if has else pair[1])



def manwon(krw: float) -> str:
    """원 → "1,235만 원"(만 원 단위 반올림). 1만 원 미만은 "0만 원"."""
    return f"{round(krw / 10_000):,.0f}만 원"


def updown(pct: float) -> str:
    """수익률(%) → "1.1% 올랐어요" / "0.4% 내렸어요" / "그대로예요"."""
    v = round(pct, 1)
    if v == 0:
        return "그대로예요"
    return f"{abs(v):.1f}% {'올랐어요' if v > 0 else '내렸어요'}"


def _updown_stem(pct: float) -> str:
    v = round(pct, 1)
    return "그대로였고" if v == 0 else f"{abs(v):.1f}% {'올랐고' if v > 0 else '내렸고'}"


def vs_qqq(pp: float) -> str:
    """%p 차이 → "QQQ보다 0.4%p 앞서요" / "QQQ보다 0.4%p 뒤처져요" / "QQQ와 같아요"."""
    v = round(pp, 1)
    if v == 0:
        return "QQQ와 같아요"
    return f"QQQ보다 {abs(v):.1f}%p {'앞서요' if v > 0 else '뒤처져요'}"


def _md(ts) -> str:
    return letter._mdw(ts)


# ── 섹션 ─────────────────────────────────────────────────────────────────────


def todo_lines(summary: dict, cfg: dict, edition: str) -> list[str]:
    """[오늘 할 일] — 할 일이 없으면 "없어요…", 개인판은 저축일·12월 공제, 두 판 모두 급락장 경고."""
    lines: list[str] = []
    vb = summary.get("vbooks") or {}
    if vb.get("switch_today"):
        lines.append("가상 추세 전략에 전환 신호가 났어요.")
    if edition == PERSONAL:
        if summary.get("savings_day"):
            lines = SAVINGS + lines
        dt = summary.get("dec_tax")
        if dt:
            left, dl, status = dt.get("remaining_krw"), dt.get("deadline"), dt.get("status")
            if status == "ok" and left is not None:
                lines.append(f"올해 세금 없이 팔 수 있는 이익이 {manwon(left)} 남았어요.")
                if dl is not None:
                    lines.append(f"{_md(dl)}에 정리하세요.")
            elif status == "loss":
                lines.append("QQQM이 지금 손실이라 올해 세금 정리는 필요 없어요.")
            elif status == "no_room":
                lines.append("올해 세금 없이 팔 수 있는 한도는 이미 다 썼어요.")
            elif status == "no_record":
                lines.append("올해 세금 정리를 계산하려면 시트에 QQQM 보유를 넣어 주세요.")
        sc = summary.get("scorecard")
        if sc:
            lines.append(f"{int(str(sc['month'])[-2:])}월 가상 성적표가 나왔어요.")
    if not lines:
        lines.append(TODO_NONE)
    if letter.crash_warning(summary.get("qqq_close")):
        lines += CRASH
    return lines


def money_lines(summary: dict, cfg: dict) -> list[str]:
    """[내 돈](개인판만) — QQQM·ISA 평가와 손익, 합계, 남은 세금 한도."""
    w = summary.get("wallet") or {}
    q = summary.get("my_money")
    q_state = w.get("qqqm_state") or ("ok" if q else "empty")
    isa, i_state = w.get("isa"), w.get("isa_state") or "empty"
    if q_state == "empty" and i_state == "empty":
        return [MONEY_EMPTY]
    lines, ok = [], []
    for name, m, state, topic in (("QQQM", q, q_state, "QQQM은"), ("ISA", isa, i_state, "ISA는")):
        if state == "ok" and m:
            pct = m.get("pnl_pct")
            lines.append(f"{name} {manwon(m['value_krw'])}" + (f", {updown(pct)}." if pct is not None else "이에요."))
            ok.append(m)
        elif state == "no_price":
            lines.append(f"{topic} 오늘 가격을 못 받았어요.")
        else:
            lines.append(f"{topic} 시트 노란 칸을 채우면 보여 드려요.")
    if len(ok) == 2:
        lines.append(f"합계 {manwon(sum(m['value_krw'] for m in ok))}이에요.")
    room = []
    if q_state == "ok" and q:
        room.append(f"세금 없이 팔 수 있는 이익은 {manwon(q['remaining_krw'])}")
    if i_state == "ok" and isa:
        room.append(f"ISA 비과세 한도는 {manwon(isa['tax_free_left_krw'])}")
    if room:
        lines.append(", ".join(room) + " 남았어요.")
    return lines


_ZONE_WORD = {"극단적 공포": "'아주 겁먹은' 쪽", "공포": "'공포' 쪽", "중립": "'중립'", "탐욕": "'욕심' 쪽", "극단적 탐욕": "'아주 들뜬' 쪽"}


def market_lines(summary: dict) -> list[str]:
    """[시장] — QQQ 하루·일주일, 장기 평균선(200일) 대비와 큰 흐름, 투자 심리, 금리·환율, 위험 신호."""
    from factory import regimes

    lines = []
    qc = summary.get("qqq_close")
    if qc is not None and len(qc.dropna()) >= 6:
        q = qc.dropna().astype(float)
        day, week = (q.iloc[-1] / q.iloc[-2] - 1) * 100, (q.iloc[-1] / q.iloc[-6] - 1) * 100
        lines.append(f"나스닥100(QQQ)은 어제 {_updown_stem(day)}, 일주일 동안 {updown(week)}.")
        regime = regimes.classify(q).iloc[-1]
        if len(q) >= 200:
            gap = (q.iloc[-1] / q.iloc[-200:].mean() - 1) * 100
            side = f"{abs(gap):.0f}% {'위' if gap >= 0 else '아래'}"
            if regime == regimes.UP:
                lines.append(f"장기 평균선(200일)보다 {side}라 큰 흐름은 아직 상승장이에요.")
            elif regime == regimes.DOWN:
                lines.append(f"장기 평균선(200일)보다 {side}라 큰 흐름은 하락장이에요.")
            elif regime == regimes.CRASH:
                lines.append(f"장기 평균선(200일)보다 {side}이고, 최근 크게 빠져서 폭락장이에요.")
            else:
                lines.append(f"장기 평균선(200일)보다 {side}예요.")
    by = {r.get("slug"): r for r in summary.get("macro_rows") or []}
    fg = by.get("fear-greed")
    if fg:
        zone = (fg.get("badge") or {}).get("zone")
        word = _ZONE_WORD.get(zone)
        lines.append(f"투자 심리는 {word}이에요({fg['value']})." if word else f"투자 심리 점수는 {fg['value']}예요.")
    elif by.get("vix"):
        lines.append(f"시장 불안 지수(VIX)는 {by['vix']['value']:.1f}예요.")
    rest = []
    if by.get("dgs10"):
        rest.append(f"미국 10년 금리 {by['dgs10']['value']:.1f}%")
    if by.get("fx"):
        rest.append(f"환율 {by['fx']['value']:,.0f}원")
    if rest:
        lines.append(", ".join(rest) + "이에요.")
    danger = [letter._danger_text(r) for r in summary.get("macro_rows") or [] if (r.get("badge") or {}).get("label") == "위험"]
    if danger:
        lines.append("조심할 신호도 있어요: " + ", ".join(danger) + ".")
    return lines


def research_lines(cfg: dict, today_kst) -> list[str]:
    """연구 소식(월요일·판정 다음 날만, notify.letter.research_lines와 같은 날) — 쉬운 문장."""
    if not letter.research_lines(cfg, today_kst):
        return []
    news = cfg.get("research_news") or {}
    cur = news.get("current") or {}
    lines = []
    if cur.get("name"):
        if cur.get("verdict"):
            lines.append(f"연구 소식: 「{cur['name']}」은 {cur['verdict']}이에요.")
            if cur.get("plain_result") or cur.get("result"):
                lines.append(str(cur.get("plain_result") or cur["result"]))
        else:
            lines.append(f"연구 소식: 「{cur['name']}」을 시험하고 있어요(지금 단계: {cur.get('stage') or '사전 등록'}).")
    for x in news.get("in_progress") or []:
        lines.append(f"진행 중인 실험: {x}.")
    return lines


def lab_lines(summary: dict, cfg: dict, edition: str) -> list[str]:
    """[가상 실험 — 실제 돈 아님] — 웅덩이 v3.1 줄(engine.puddle_v31.letter_lines) + 연구 소식."""
    pv = summary.get("puddle_v31") or {}
    lines = list(pv.get(edition) or [])
    lines += research_lines(cfg, summary.get("letter_today_kst"))
    return lines


def problems(summary: dict, edition: str) -> list[str]:
    """자료 문제(있을 때만 맨 위 한 줄로): 자료 지연·실적 확인 실패·장부 오류·지표 누락·시트 입력 오류."""
    out = []
    info = summary.get("letter_check") or {}
    if info.get("lag", 0) > 0:
        exp = f"(기대 {_md(info['expected'])})" if info.get("expected") is not None else ""
        out.append(f"미국 자료가 {info['lag']}거래일 늦어요{exp}")
    pv = summary.get("puddle_v31") or {}
    fail = pv.get("check_personal" if edition == PERSONAL else "check_group")
    if fail:
        out.append(fail.replace("실적 확인 실패", "실적 발표 확인을 못 한 종목이 있어요"))
    if pv.get("error"):
        out.append("웅덩이 v3.1 장부를 계산하지 못했어요")
    for note in info.get("notes") or []:
        if note.startswith("실적 확인 실패") or note == "웅덩이 v3.1 장부 오류":
            continue
        out.append(note)
    if summary.get("fred_missing"):
        out.append(f"시장 지표 {len(summary['fred_missing'])}개를 못 받았어요")
    if edition == PERSONAL and summary.get("input_errors"):
        out.append(f"시트 입력 오류가 {len(summary['input_errors'])}건 있어요")
    return out


def footer(summary: dict, edition: str) -> str:
    """맨 아래 작은 줄: "기준: 10/23 미국 종가 · 자료 정상"(문제가 있으면 "자료 문제 있음")."""
    as_of = summary.get("as_of")
    base = f"{pd.Timestamp(as_of).month}/{pd.Timestamp(as_of).day} 미국 종가" if as_of is not None else "기준일 알 수 없음"
    status = "자료 문제 있음" if problems(summary, edition) else "자료 정상"
    return f"기준: {base} · {status}"


def build(summary: dict, cfg: dict, edition: str = PERSONAL) -> dict:
    """편지 구조 {title, problem, sections[{header, lines}], footer} — 캡션과 HTML이 같이 쓴다."""
    from notify import briefing

    sections = [{"header": H_TODO, "lines": todo_lines(summary, cfg, edition)}]
    if edition == PERSONAL:
        sections.append({"header": H_MONEY, "lines": money_lines(summary, cfg)})
    m = market_lines(summary)
    if m:
        sections.append({"header": H_MARKET, "lines": m})
    lab = lab_lines(summary, cfg, edition)
    if lab:
        sections.append({"header": H_LAB, "lines": lab})
    probs = problems(summary, edition)
    extra = []
    if edition == PERSONAL and summary.get("rebuilt_from"):
        extra.append(f"체결 기록이 바뀌어 {summary['rebuilt_from']}부터 다시 계산했어요.")
    return {"title": briefing.title_line(summary, cfg), "problem": ("자료 문제: " + " · ".join(probs) + ".") if probs else None,
            "sections": sections, "footer": footer(summary, edition), "extra": extra}


def to_text(doc: dict, limit: int = CAPTION_LIMIT) -> str:
    """편지 구조 → 텔레그램 글. 한도를 넘으면 뒤쪽 섹션부터 줄이고 "나머지는 첨부 파일에서…"를 붙인다."""
    def render(sections, more=False):
        parts = [doc["title"]] + ([doc["problem"]] if doc["problem"] else [])
        body = ["\n".join(parts)]
        for s in sections:
            body.append("\n".join([s["header"]] + s["lines"]))
        tail = doc.get("extra", []) + ([CAPTION_MORE] if more else []) + [doc["footer"]]
        body.append("\n".join(tail))
        return "\n\n".join(body)

    secs = list(doc["sections"])
    text = render(secs)
    while len(text) > limit and len(secs) > 1:
        secs = secs[:-1]
        text = render(secs, more=True)
    return text[:limit]
