"""아침 편지 v3 (2026-10-24) — 찌라시형 뉴스 브리핑 + 경어체. 텔레그램 첨부 설명과 HTML 맨 위가 같은 문장을 쓴다.

표시 전용 순수 함수: 이미 계산된 summary를 읽어 문장만 만든다(계산·판단 로직은 notify.letter·engine 그대로).
문장은 config/letter_phrases.yaml(사용자 수정용, 경어체). 같은 상황 문장이 여럿이면 날짜로 돌려 쓴다(전날과 다른 문장).
머리줄 "[오빠의 아침 편지] M/D(요일)"(편지 받는 한국 날짜), 섹션 "■ 이름", 항목 "▶ 한 줄". 이모지·들여쓰기 없음.
단체방판(edition="group")은 ■ 내 돈·종목명·개인 금액 없음(뉴스는 같다).

순서: 머리줄 → (문제 있을 때만) "▶ 자료 문제: …" → ■ 오늘의 핵심 → 뉴스 주제 칸(config/news_topics.yaml 순서)
→ ■ 미국 증시 마감(첫 줄 = 오늘의 한 줄) → ■ 오늘 할 일 → ■ 내 돈(개인판) → ■ 웅덩이 후보(2026-10-24) → ■ 가상 실험 (실제 돈 아님)
→ 기준 줄 → ※ 유의 줄(웅덩이 후보가 있으면 한 줄 더).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from notify import letter

PERSONAL, GROUP = letter.PERSONAL, letter.GROUP
ROOT = Path(__file__).resolve().parents[1]
PHRASES_PATH = ROOT / "config" / "letter_phrases.yaml"
CAPTION_LIMIT = 1024  # 텔레그램 첨부 설명 최대 글자 수
MESSAGE_LIMIT = 4096  # 텔레그램 문자 메시지 최대 글자 수(편지 본문은 문자 메시지로 먼저 보낸다)
BULLET, SECTION = "▶ ", "■ "
NEWS_SECTIONS = ("국제정세", "전쟁·지정학", "미국 경제·금리")  # 기본 순서(news_topics.yaml 순서가 우선)


# ── 문구 ─────────────────────────────────────────────────────────────────────


def load_phrases(path: Path | None = None) -> dict:
    """config/letter_phrases.yaml → dict."""
    with open(path or PHRASES_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def pick(entry, day) -> str:
    """문구 하나 고르기: 문자열이면 그대로, 목록이면 날짜(서수) % 개수 — 하루 지나면 다음 문장(전날과 다름)."""
    if isinstance(entry, (list, tuple)):
        if not entry:
            return ""
        n = pd.Timestamp(day).toordinal() if day is not None else 0
        return str(entry[n % len(entry)])
    return str(entry or "")


def josa(word: str, pair: tuple[str, str]) -> str:
    """word + 조사. pair = (받침 있을 때, 없을 때) 예: ("을", "를"), ("이", "가"), ("은", "는"), ("과", "와").
    영문 티커는 마지막 글자 읽는 소리로(L 엘·M 엠·N 엔·R 알만 받침), 한글은 받침 유무, 숫자·기타는 받침 없음으로 본다."""
    if not word:
        return word
    base = re.sub(r"\s*\([^()]*\)$", "", word) or word  # 끝 괄호 설명은 읽지 않는다: "지수(VIX)" → 지수
    ch = base[-1]
    if "가" <= ch <= "힣":
        has = (ord(ch) - 0xAC00) % 28 != 0
    else:
        has = ch.upper() in "LMNR"
    return word + (pair[0] if has else pair[1])


_PAIRS = {"을": ("을", "를"), "를": ("을", "를"), "이": ("이", "가"), "가": ("이", "가"), "은": ("은", "는"), "는": ("은", "는")}


def fill(template: str, **vals) -> str:
    """"{값}" 채우기. {종목}·{이름}·{항목} 바로 뒤 조사(을/를·이/가·은/는)는 값의 읽는 소리에 맞춘다."""
    def fix(m):
        key, j = m.group(1), m.group(2)
        return josa(str(vals[key]), _PAIRS[j]) if key in vals else m.group(0)

    out = re.sub(r"\{(종목|이름|항목)\}(을|를|이|가|은|는)(?![가-힣])", fix, template)
    for k, v in vals.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def phrase(ph: dict, path: str, day=None, **vals) -> str:
    """ph에서 "a.b" 경로 문구를 골라(pick) 채운다(fill)."""
    node = ph
    for part in path.split("."):
        node = (node or {}).get(part)
    return fill(pick(node, day), **vals)


# ── 숫자 표현 ────────────────────────────────────────────────────────────────


def man(krw: float) -> str:
    """원 → 만 원 단위 숫자 문자열 "1,235"(반올림)."""
    return f"{round(krw / 10_000):,.0f}"


def manwon(krw: float) -> str:
    """원 → "1,235만 원"."""
    return f"{man(krw)}만 원"


def signed(pct: float) -> str:
    """수익률(%) → "+1.1" / "-0.4" / "0.0" (소수 첫째 자리)."""
    v = round(float(pct), 1)
    return "0.0" if v == 0 else f"{v:+.1f}"


def vs_qqq_parts(pp: float) -> tuple[str, str]:
    """%p 차이 → (문구 키, 절댓값 문자열): ("same"|"ahead"|"behind", "0.4")."""
    v = round(pp, 1)
    return ("same" if v == 0 else "ahead" if v > 0 else "behind"), f"{abs(v):.1f}"


def _md(ts) -> str:
    return letter._mdw(ts)


def _day(summary: dict):
    """문구 돌려 쓰기 기준 날짜 = 편지 받는 한국 날짜(없으면 기준일)."""
    return summary.get("letter_today_kst") if summary.get("letter_today_kst") is not None else summary.get("as_of")


# ── 시장 숫자 ────────────────────────────────────────────────────────────────


def _closes(s) -> pd.Series | None:
    if s is None:
        return None
    s = s.dropna().astype(float)
    return s if len(s) >= 2 else None


def market_state(summary: dict) -> dict | None:
    """QQQ 종가 → {day(%), gap(200일 평균 대비 %), regime, drawdown(252일 고점 대비 %)}. 자료가 없으면 None."""
    from factory import regimes

    q = _closes(summary.get("qqq_close"))
    if q is None:
        return None
    out = {"day": (q.iloc[-1] / q.iloc[-2] - 1) * 100, "gap": None, "regime": regimes.UNKNOWN,
           "drawdown": (q.iloc[-1] / q.iloc[-252:].max() - 1) * 100}
    if len(q) >= 200:
        out["gap"] = (q.iloc[-1] / q.iloc[-200:].mean() - 1) * 100
        out["regime"] = regimes.classify(q).iloc[-1]
    return out


def _is_holiday_after(summary: dict) -> bool:
    """휴장 다음 날: 편지 전날(미국 날짜)이 평일인데 기준일이 그보다 앞서고, 자료 지연은 아닐 때."""
    today, as_of = summary.get("letter_today_kst"), summary.get("as_of")
    if today is None or as_of is None or (summary.get("letter_check") or {}).get("lag", 0) > 0:
        return False
    us_day = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    return us_day.dayofweek < 5 and pd.Timestamp(as_of).normalize() < us_day


def headline(summary: dict, cfg: dict, ph: dict, edition: str) -> tuple[str | None, str | None]:
    """오늘의 한 줄 → (문장, 쓴 상황). 위에서부터 먼저 맞는 것 하나: 휴장 다음 날 → 저축일(개인판) → 12월 공제(개인판)
    → 폭락장 → 하락장(장기 평균선 근처면 갈림길) → 크게 오름 → 크게 내림(상승장) → 갈림길 → 평온."""
    from factory import regimes

    day = _day(summary)
    th = ph.get("thresholds") or {}
    big, cross = float(th.get("big_move_pct", 1.5)), float(th.get("crossroads_pct", 2.0))
    if _is_holiday_after(summary):
        return phrase(ph, "headline.holiday", day), "holiday"
    if edition == PERSONAL and summary.get("savings_day"):
        return phrase(ph, "headline.savings", day), "savings"
    dt = summary.get("dec_tax") or {}
    if edition == PERSONAL and dt.get("status") == "ok" and dt.get("deadline") is not None and day is not None:
        n = (pd.Timestamp(dt["deadline"]).normalize() - pd.Timestamp(day).normalize()).days
        if n >= 0:
            return phrase(ph, "headline.dec_tax", day, n=n), "dec_tax"
    m = market_state(summary)
    if m is None:
        return None, None
    near = m["gap"] is not None and abs(m["gap"]) < cross
    if m["regime"] == regimes.CRASH:
        return phrase(ph, "headline.crash", day, dd=f"{abs(m['drawdown']):.1f}"), "crash"
    if m["regime"] == regimes.DOWN:
        if near:
            return phrase(ph, "headline.crossroads", day, z=f"{abs(m['gap']):.1f}"), "crossroads"
        return phrase(ph, "headline.down", day, z=f"{abs(m['gap']):.1f}"), "down"
    if m["day"] >= big:
        return phrase(ph, "headline.big_up", day, x=f"{abs(m['day']):.1f}"), "big_up"
    if m["day"] <= -big:
        return phrase(ph, "headline.big_down", day, x=f"{abs(m['day']):.1f}"), "big_down"
    if near:
        return phrase(ph, "headline.crossroads", day, z=f"{abs(m['gap']):.1f}"), "crossroads"
    return phrase(ph, "headline.calm", day, x=signed(m["day"])), "calm"


def _index_change(s, as_of) -> float | None:
    s = _closes(s)
    if s is None:
        return None
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    return (s.iloc[-1] / s.iloc[-2] - 1) * 100 if len(s) >= 2 else None


def market_lines(summary: dict, cfg: dict, ph: dict, edition: str) -> list[str]:
    """■ 미국 증시 마감 — 오늘의 한 줄, 지수 변화, 금리·환율, 큰 흐름(오늘의 한 줄과 겹치면 생략), 투자 심리, 주의."""
    from factory import regimes

    day = _day(summary)
    lines = []
    head, kind = headline(summary, cfg, ph, edition)
    if head:
        lines.append(head)
    m = market_state(summary)
    parts = {}
    if m is not None:
        parts["q"] = signed(m["day"])
    ix = summary.get("index_close") or {}
    for key, name in (("s", "spx"), ("c", "soxx")):
        v = _index_change(ix.get(name), summary.get("as_of"))
        if v is not None:
            parts[key] = signed(v)
    if "q" in parts:
        tpl = phrase(ph, "market.index", day)
        segs = [seg for seg in tpl.split(" · ") if all(f"{{{k}}}" not in seg or k in parts for k in ("q", "s", "c"))]
        lines.append(fill(" · ".join(segs), **parts))
    by = {r.get("slug"): r for r in summary.get("macro_rows") or []}
    rate, fx = by.get("dgs10"), by.get("fx")
    if rate or fx:
        tpl = phrase(ph, "market.rates", day)
        segs = [seg for seg in tpl.split(" · ") if ("{r}" not in seg or rate) and ("{fx}" not in seg or fx)]
        lines.append(fill(" · ".join(segs), r=f"{rate['value']:.1f}" if rate else "", fx=f"{fx['value']:,.0f}" if fx else ""))
    if m is not None and m["gap"] is not None and kind not in ("crash", "down", "crossroads"):
        cross = float((ph.get("thresholds") or {}).get("crossroads_pct", 2.0))
        z = f"{abs(m['gap']):.1f}"
        if m["regime"] == regimes.CRASH:
            lines.append(phrase(ph, "market.regime.crash", day, z=z))
        elif abs(m["gap"]) < cross:
            lines.append(phrase(ph, "market.regime.crossroads", day, z=z))
        elif m["regime"] == regimes.DOWN:
            lines.append(phrase(ph, "market.regime.down", day, z=z))
        else:
            lines.append(phrase(ph, "market.regime.up", day, z=z))
    fg = by.get("fear-greed")
    if fg and fg.get("value") is not None:
        v = float(fg["value"])
        for b in (ph.get("market") or {}).get("mood") or []:
            if v <= float(b["max"]):
                lines.append(fill(pick(b["text"], day), v=f"{v:.0f}"))
                break
    w = warn_line(summary, ph, day)
    if w:
        lines.append(w)
    t = _thermometer_line(summary)
    if t:
        lines.append(t)
    return lines


def _thermometer_line(summary: dict) -> str | None:
    """시장 온도계 한 줄(그 기준일 data/thermometer/history.csv에 값이 있을 때만) — 어떤 오류도 편지를 멈추지 않는다."""
    try:
        from thermometer.history import letter_line

        return letter_line(summary.get("as_of"))
    except Exception:
        return None


def warn_line(summary: dict, ph: dict, day=None) -> str | None:
    """주의 — 위험 단계 지표를 하루 한 줄로: "높은 10년 금리와 기준금리 인상은 조심해서 봅니다." 없으면 None.
    순서는 market.warn.items 순서, 셋 이상이면 "A, B와 C"."""
    warn = (ph.get("market") or {}).get("warn") or {}
    items = warn.get("items") or {}
    risky = {r.get("slug") for r in summary.get("macro_rows") or [] if (r.get("badge") or {}).get("label") == "위험"}
    names = [pick(v, day) for k, v in items.items() if k in risky]
    if not names:
        return None
    joined = names[0] if len(names) == 1 else ", ".join(names[:-2] + [josa(names[-2], ("과", "와"))]) + " " + names[-1]
    return fill(pick(warn.get("line") or "{항목}은 조심해서 봅니다.", day), 항목=joined)


def ledger_line(ph: dict, key: str, parts: list[str], day=None) -> str:
    """■ 가상 실험 장부 한 줄: "{이름}: 문장 문장 …" (이름 = lab.names.<key>)."""
    return fill(pick((ph.get("lab") or {}).get("line") or "{이름}: {내용}", day),
                이름=phrase(ph, f"lab.names.{key}", day), 내용=" ".join(p for p in parts if p))


# ── 섹션 ─────────────────────────────────────────────────────────────────────


def todo_lines(summary: dict, cfg: dict, ph: dict, edition: str) -> list[str]:
    """■ 오늘 할 일 — 개인판은 저축일(밀린 날 안내)·12월 공제·성적표, 두 판 모두 급락장. 없으면 "투자 포지션 변경 없습니다."."""
    day = _day(summary)
    lines: list[str] = []
    if edition == PERSONAL:
        if summary.get("savings_day"):
            lines.append(phrase(ph, "todo.savings", day))
            sd, today = cfg.get("savings_day"), summary.get("letter_today_kst")
            if sd and today is not None:
                t = pd.Timestamp(today)
                last = (t + pd.offsets.MonthEnd(0)).day
                orig = pd.Timestamp(year=t.year, month=t.month, day=min(int(sd), last))
                if orig.normalize() != t.normalize():
                    lines.append(phrase(ph, "todo.savings_moved", day, 날짜=_md(orig)))
        dl_line = _dec_tax_line(summary, ph, day)
        if dl_line:
            lines.append(dl_line)
        sc = summary.get("scorecard")
        if sc:
            lines.append(phrase(ph, "todo.scorecard", day, 월=int(str(sc["month"])[-2:])))
    if letter.crash_warning(summary.get("qqq_close")):
        lines.append(phrase(ph, "todo.crash", day))
    if not lines:
        lines.append(phrase(ph, "todo.none", day))
    return lines


def _dec_tax_line(summary: dict, ph: dict, day=None) -> str | None:
    """12월 공제 할 일 한 줄(개인판) — 없으면 None."""
    dt = summary.get("dec_tax")
    if not dt:
        return None
    status = dt.get("status")
    if status == "ok" and dt.get("remaining_krw") is not None:
        dl = _md(dt["deadline"]) if dt.get("deadline") is not None else "연말"
        return phrase(ph, "todo.dec_tax", day, 남은=man(dt["remaining_krw"]), 마감일=dl)
    if status in ("loss", "no_room", "no_record"):
        return phrase(ph, f"todo.dec_tax_{status}", day)
    return None


def money_lines(summary: dict, cfg: dict, ph: dict) -> list[str]:
    """■ 내 돈(개인판만) — QQQM·ISA 평가와 손익, 합계, 남은 세금 한도."""
    day = _day(summary)
    w = summary.get("wallet") or {}
    q = summary.get("my_money")
    q_state = w.get("qqqm_state") or ("ok" if q else "empty")
    isa, i_state = w.get("isa"), w.get("isa_state") or "empty"
    if q_state == "empty" and i_state == "empty":
        return [phrase(ph, "money.empty", day)]
    lines, ok = [], []
    for name, m, state in (("QQQM", q, q_state), ("ISA", isa, i_state)):
        if state == "ok" and m:
            pct = m.get("pnl_pct")
            if pct is None or round(pct, 1) == 0:
                lines.append(phrase(ph, "money.item_flat", day, 이름=name, 금액=man(m["value_krw"])))
            else:
                lines.append(phrase(ph, "money.item", day, 이름=name, 금액=man(m["value_krw"]), 률=f"{abs(pct):.1f}",
                                    방향="올랐습니다" if pct > 0 else "내렸습니다"))
            ok.append(m)
        elif state == "no_price":
            lines.append(phrase(ph, "money.no_price", day, 이름=name))
        else:
            lines.append(f"{name}: " + phrase(ph, "money.empty", day))
    if len(ok) == 2:
        lines.append(phrase(ph, "money.total", day, 합계=man(sum(m["value_krw"] for m in ok))))
    has_q, has_i = q_state == "ok" and bool(q), i_state == "ok" and bool(isa)
    if has_q and has_i:
        lines.append(phrase(ph, "money.room", day, 남은=man(q["remaining_krw"]), isa남은=man(isa["tax_free_left_krw"])))
    elif has_q:
        lines.append(phrase(ph, "money.room_qqqm", day, 남은=man(q["remaining_krw"])))
    elif has_i:
        lines.append(phrase(ph, "money.room_isa", day, isa남은=man(isa["tax_free_left_krw"])))
    return lines


def research_lines(cfg: dict, ph: dict, today_kst) -> list[str]:
    """연구 소식(월요일·판정 다음 날만, notify.letter.research_lines와 같은 날)."""
    if not letter.research_lines(cfg, today_kst):
        return []
    news = cfg.get("research_news") or {}
    cur = news.get("current") or {}
    lines = []
    if cur.get("name"):
        if cur.get("verdict"):
            lines.append(phrase(ph, "lab.research_result", today_kst, 이름=cur["name"], 판정=cur["verdict"]))
        else:
            lines.append(phrase(ph, "lab.research_testing", today_kst, 이름=cur["name"]))
    if news.get("in_progress"):
        lines.append(phrase(ph, "lab.in_progress", today_kst))
    return lines


def lab_lines(summary: dict, cfg: dict, ph: dict, edition: str) -> list[str]:
    """■ 가상 실험 (실제 돈 아님) — 웅덩이 장부 줄(summary["puddle_v31"] 칸, 2026-10-27부터 v3.2) + (월요일만) 필터 비교
    + 반등 확인 장부 줄(summary["puddle_rebound"] 칸, 지금 v2) + 연구 소식."""
    pv = summary.get("puddle_v31") or {}
    lines = list(pv.get(edition) or [])
    today = summary.get("letter_today_kst")
    gap = pv.get("filter_gap_pp")  # 필터 없는 판과의 차이(%p) — 장부가 이 값을 줄 때만(월요일)
    if gap is not None and today is not None and pd.Timestamp(today).dayofweek == 0:
        lines.append(phrase(ph, "lab.filter_compare", today, e=f"{abs(gap):.1f}", 방향="앞섭니다" if gap >= 0 else "뒤처집니다"))
    rb = summary.get("puddle_rebound") or {}
    lines += list(rb.get(edition) or [])  # 웅덩이 반등 확인 장부(2026-10-24, 2026-10-27부터 v2) — 웅덩이 장부 줄 아래
    vs = rb.get("vs_v31_pp")  # 비교 B(v2는 웅덩이 v3.2) 대비
    if rb.get(edition) and vs is not None and today is not None and pd.Timestamp(today).dayofweek == 0:
        lines.append(phrase(ph, "rebound.vs_v31", today, e=f"{abs(vs):.1f}", 방향="앞섭니다" if vs >= 0 else "뒤처집니다",
                            b=rb.get("b_label") or "v3.2"))
    lines += research_lines(cfg, ph, today)
    return lines


def candidate_lines(summary: dict, edition: str) -> list[str]:
    """■ 웅덩이 후보 — engine.puddle_rebound.candidate_lines가 만든 줄(개인판 단계별, 단체방판 개수만). 장부가 없으면 빈 목록."""
    rb = summary.get("puddle_rebound") or {}
    return list(rb.get("cand_personal" if edition == PERSONAL else "cand_group") or [])


def news_sections(summary: dict, ph: dict) -> list[dict]:
    """■ 오늘의 핵심 + 주제 칸. 줄 = {"text": "제목 (언론사)", "title", "source", "url"}. 못 받으면 핵심 칸에 실패 한 줄."""
    day = _day(summary)
    nw = summary.get("news")
    top_h = (ph.get("headers") or {}).get("top", "■ 오늘의 핵심")
    if not nw or nw.get("status") == "fail" or not nw.get("top"):
        return [{"header": top_h, "lines": [phrase(ph, "news.failed", day)]}] if nw is not None else []

    def line(a):
        t = _shorten(a["title"])
        return {"text": phrase(ph, "news.line", day, 제목=t, 언론사=a["source"]), "title": t, "source": a["source"], "url": a.get("url", "")}

    out = [{"header": top_h, "lines": [line(a) for a in nw["top"]]}]
    for name, arts in (nw.get("topics") or {}).items():
        if arts:
            out.append({"header": SECTION + name, "lines": [line(a) for a in arts], "news_topic": True})
    return out


def _shorten(title: str) -> str:
    from core import news as cnews

    return cnews.shorten(title)


PUDDLE_ERROR_NOTES = ("웅덩이 v3.1 장부 오류", "웅덩이 반등 확인 장부 오류", "웅덩이 v3.2 장부 오류", "웅덩이 반등 확인 v2 장부 오류")


def problems(summary: dict, ph: dict, edition: str) -> list[str]:
    """자료 문제(있을 때만 머리줄 아래 한 줄): 자료 지연·뉴스 일부 실패·실적 확인 실패·장부 오류·지표 누락·시트 입력 오류."""
    day = _day(summary)
    out = []
    info = summary.get("letter_check") or {}
    if info.get("lag", 0) > 0:
        out.append(phrase(ph, "problem.lag", day, n=info["lag"]))
    if (summary.get("news") or {}).get("status") == "partial":
        out.append(phrase(ph, "problem.news_partial", day))
    pv = summary.get("puddle_v31") or {}
    rb = summary.get("puddle_rebound") or {}
    key = "check_personal" if edition == PERSONAL else "check_group"
    fails = [f for f in (pv.get(key), rb.get(key)) if f]
    if fails:
        if edition == PERSONAL:
            names = sorted({t.strip() for f in fails for t in f[f.find("(") + 1:f.rfind(")")].split(",") if "(" in f})
            detail = f"({', '.join(names)})" if names else ""
        else:
            detail = max((f[f.find("("):] for f in fails if "(" in f), default="")
        out.append(phrase(ph, "problem.earnings_check", day, 상세=detail))
    skey = "shares_personal" if edition == PERSONAL else "shares_group"
    miss = [f for f in (pv.get(skey), rb.get(skey)) if f]  # 시가총액 순위에서 뺀 종목(주식 수 없음, 2026-10-27) — 두 장부 같은 목록이면 한 번
    if miss:
        out.append(phrase(ph, "problem.shares_missing", day, 상세=max(miss, key=len)))
    if pv.get("error"):
        out.append(phrase(ph, "problem.puddle_error", day))
    if rb.get("error"):
        out.append(phrase(ph, "problem.rebound_error", day))
    if (summary.get("curve_chart") or {}).get("error"):
        out.append(phrase(ph, "problem.chart_error", day))
    for note in info.get("notes") or []:
        if note.startswith("실적 확인 실패") or note in PUDDLE_ERROR_NOTES:
            continue
        out.append(note.rstrip(".") + ".")
    if summary.get("fred_missing"):
        out.append(phrase(ph, "problem.fred_missing", day, n=len(summary["fred_missing"])))
    if edition == PERSONAL and summary.get("input_errors"):
        out.append(phrase(ph, "problem.input_errors", day, n=len(summary["input_errors"])))
    return out


def title(summary: dict, cfg: dict, ph: dict) -> str:
    """머리줄 "[오빠의 아침 편지] 10/24(토)" — 편지 받는 한국 날짜(없으면 기준일). 모의 보고서는 " · 모의"."""
    d = _day(summary)
    t = fill(str(ph.get("title") or "[오빠의 아침 편지] {날짜}"), 날짜=_md(d) if d is not None else "")
    return t + " · 모의" if summary.get("mode") == "paper" else t


def build(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> dict:
    """편지 구조 {title, problem, sections[{header, lines}], footer, caution} — 캡션과 HTML이 같이 쓴다.
    줄은 글(str) 또는 뉴스 줄(dict: text·url). 머리표 "▶ "는 그릴 때 붙인다."""
    ph = phrases if phrases is not None else load_phrases()
    hd = ph.get("headers") or {}
    day = _day(summary)
    sections = news_sections(summary, ph)
    m = market_lines(summary, cfg, ph, edition)
    if m:
        sections.append({"header": hd.get("market", "■ 미국 증시 마감"), "lines": m})
    sections.append({"header": hd.get("todo", "■ 오늘 할 일"), "lines": todo_lines(summary, cfg, ph, edition)})
    if edition == PERSONAL:
        sections.append({"header": hd.get("money", "■ 내 돈"), "lines": money_lines(summary, cfg, ph)})
    cand = candidate_lines(summary, edition)
    if cand:
        sections.append({"header": hd.get("candidates", "■ 웅덩이 후보 (가상 실험 신호, 실제 매수 권유 아님)"), "lines": cand,
                         "candidates": True})
    lab = lab_lines(summary, cfg, ph, edition)
    if lab:
        sections.append({"header": hd.get("lab", "■ 가상 실험 (실제 돈 아님)"), "lines": lab, "lab": True})
    probs = problems(summary, ph, edition)
    as_of = summary.get("as_of")
    base = f"{pd.Timestamp(as_of).month}/{pd.Timestamp(as_of).day}" if as_of is not None else "?"
    return {"title": title(summary, cfg, ph),
            "problem": (BULLET + phrase(ph, "problem.prefix", day) + " ".join(probs)) if probs else None,
            "sections": sections, "footer": phrase(ph, "footer.base", day, 날짜=base),
            "caution": phrase(ph, "footer.caution", day), "more": phrase(ph, "footer.more", day),
            "cautions": [phrase(ph, "footer.caution", day)] + ([phrase(ph, "candidates.caution", day)] if cand else [])}


def line_text(line) -> str:
    """줄(str 또는 뉴스 dict) → "▶ …"."""
    return BULLET + (line["text"] if isinstance(line, dict) else str(line))


def to_text(doc: dict, limit: int = MESSAGE_LIMIT) -> str:
    """편지 구조 → 텔레그램 문자 메시지(들여쓰기 없음). 한도를 넘으면 뉴스 주제 칸(뒤에서부터) → 가상 실험 → 나머지 뒤쪽 칸 순으로 빼고
    "▶ 나머지는 첨부 파일에 있습니다."를 붙인다."""
    def render(sections, more=False):
        head = [doc["title"]] + ([doc["problem"]] if doc.get("problem") else [])
        body = ["\n".join(head)]
        for s in sections:
            body.append("\n".join([s["header"]] + [line_text(x) for x in s["lines"]]))
        tail = ([BULLET + doc["more"]] if more else []) + [doc["footer"]] + doc.get("cautions", [doc["caution"]])
        body.append("\n".join(tail))
        return "\n\n".join(body)

    secs = list(doc["sections"])
    text = render(secs)
    drop_order = [i for i in reversed(range(len(secs))) if secs[i].get("news_topic")]
    drop_order += [i for i in range(len(secs)) if secs[i].get("lab")]
    drop_order += [i for i in range(len(secs)) if secs[i].get("candidates")]
    drop_order += [i for i in reversed(range(len(secs))) if i not in drop_order]
    dropped: set[int] = set()
    for i in drop_order:
        if len(text) <= limit or len(dropped) >= len(secs) - 1:
            break
        dropped.add(i)
        text = render([s for j, s in enumerate(secs) if j not in dropped], more=True)
    return text[:limit]


# ── 요약 쪽지(텔레그램 문자) · 첨부 보고서 칸 (2026-10-28) ─────────────────────
# 문자 = 요약 쪽지(최대 summary.max_lines줄·max_chars자), 자세한 내용은 첨부 HTML. 표시 전용 — 계산은 그대로 읽기만.

REPORT_ORDER = ("summary", "news", "market", "thermometer", "todo", "money", "candidates", "lab")


def first_sentence(text: str) -> str:
    """문장 첫 개("…니다." 까지). 마침표가 없으면 그대로."""
    m = re.match(r"(.+?[.!?])(\s|$)", str(text).strip())
    return m.group(1) if m else str(text).strip()


def summary_todo(summary: dict, cfg: dict, ph: dict, edition: str) -> str:
    """요약 쪽지 할 일 — 급락장 → 저축일(개인판) → 12월 공제(개인판) → 그 밖에는 ■ 오늘 할 일 첫 줄, 모두 첫 문장만."""
    day = _day(summary)
    if letter.crash_warning(summary.get("qqq_close")):
        return first_sentence(phrase(ph, "todo.crash", day))
    if edition == PERSONAL:
        if summary.get("savings_day"):
            return first_sentence(phrase(ph, "todo.savings", day))
        dl = _dec_tax_line(summary, ph, day)
        if dl:
            return first_sentence(dl)
    return first_sentence(todo_lines(summary, cfg, ph, edition)[0])


def money_summary(summary: dict, ph: dict) -> str | None:
    """요약 쪽지 내 돈(개인판) — 가격이 있는 QQQM·ISA 합계와 합계 손익률. 빈칸이면 None."""
    day = _day(summary)
    w = summary.get("wallet") or {}
    q = summary.get("my_money")
    q_state = w.get("qqqm_state") or ("ok" if q else "empty")
    ok = [m for m, st in ((q, q_state), (w.get("isa"), w.get("isa_state") or "empty")) if st == "ok" and m]
    if not ok:
        return None
    total = sum(m["value_krw"] for m in ok)
    # 원가 = 편지가 종목별로 보여 주는 손익률(pnl_pct)에서 되짚은 값 — 그 값이 없으면 cost_krw
    costs = [m["value_krw"] / (1 + m["pnl_pct"] / 100) if m.get("pnl_pct") is not None else m.get("cost_krw") for m in ok]
    pct = (total / sum(costs) - 1) * 100 if all(c and c > 0 for c in costs) else None
    if pct is None or round(pct, 1) == 0:
        return phrase(ph, "summary.money_flat", day, 합계=man(total))
    return phrase(ph, "summary.money", day, 합계=man(total), 률=f"{abs(pct):.1f}", 방향="올랐습니다" if pct > 0 else "내렸습니다")


def _as_of_day(summary: dict) -> str | None:
    as_of = summary.get("as_of")
    return str(getattr(as_of, "date", lambda: as_of)())[:10] if as_of is not None else None


def thermometer_row(summary: dict) -> dict | None:
    """그 기준일 시장 온도계 기록(data/thermometer/history.csv) — 없거나 오류면 None(편지는 그대로)."""
    try:
        from thermometer import history as th

        day = _as_of_day(summary)
        return th.find(day) if day else None
    except Exception:
        return None


def thermometer_detail(summary: dict) -> dict | None:
    """보고서 「시장 온도계」 칸 자료(thermometer.history.detail) — 없거나 오류면 None."""
    try:
        from thermometer import history as th

        return th.detail(summary.get("as_of"))
    except Exception:
        return None


def market_summary(summary: dict, ph: dict) -> str | None:
    """요약 쪽지 시장 — "시장: QQQ +0.4% · 시장 온도 62도(따뜻함)"(온도계 기록이 없으면 뒷부분 생략). QQQ가 없으면 None."""
    day = _day(summary)
    m = market_state(summary)
    if m is None:
        return None
    out = phrase(ph, "summary.market", day, q=signed(m["day"]))
    t = thermometer_row(summary)
    if t and t.get("temp"):
        out += " · " + phrase(ph, "summary.temp", day, T=int(t["temp"]), 구간=t.get("zone", ""))
    return out


def lab_summary(summary: dict, ph: dict) -> str | None:
    """요약 쪽지 가상 실험 — "가상 실험: 웅덩이 v3.2 QQQ보다 0.4%p 앞섭니다 · 오늘 가상 매수 1건." 장부 값이 없으면 None."""
    day = _day(summary)
    pv = summary.get("puddle_v31") or {}
    if pv.get("vs_qqq_pp") is None:
        return None
    key, d = vs_qqq_parts(pv["vs_qqq_pp"])
    parts = [phrase(ph, "summary.lab", day, 비교=phrase(ph, f"lab.{key}", day, d=d).rstrip("."))]
    if pv.get("buys_today"):
        parts.append(phrase(ph, "summary.lab_buys", day, n=pv["buys_today"]))
    if pv.get("sells_today"):
        parts.append(phrase(ph, "summary.lab_sells", day, n=pv["sells_today"]))
    return " · ".join(parts) + "."


def news_count(summary: dict) -> int:
    """보고서 뉴스 전체 건수(핵심 + 주제 칸, 서로 겹치지 않음)."""
    nw = summary.get("news") or {}
    if nw.get("status") == "fail":
        return 0
    return len(nw.get("top") or []) + sum(len(v) for v in (nw.get("topics") or {}).values())


def candidate_count(summary: dict) -> int | None:
    """웅덩이 후보 종목 수(반등 확인 장부가 없으면 None)."""
    rb = summary.get("puddle_rebound") or {}
    if not (rb.get("cand_personal") or rb.get("cand_group")):
        return None
    return int(rb.get("cand_count") or 0)


def has_chart(summary: dict, edition: str) -> bool:
    """그 판 보고서에 그림이 있는지 — 성적 그래프(판마다), 웅덩이 후보 차트(개인판만)."""
    cc = summary.get("curve_chart") or {}
    if cc.get("group" if edition == GROUP else "personal"):
        return True
    return edition == PERSONAL and bool(summary.get("candidate_charts"))


def _more_line(summary: dict, ph: dict, edition: str) -> str:
    day = _day(summary)
    seg = []
    n = news_count(summary)
    if n:
        seg.append(phrase(ph, "summary.more_news", day, n=n))
    m = candidate_count(summary)
    if m is not None:
        seg.append(phrase(ph, "summary.more_candidates", day, m=m))
    if has_chart(summary, edition):
        seg.append(phrase(ph, "summary.more_chart", day))
    return phrase(ph, "summary.more", day, 상세=" · ".join(seg)).replace("()", "")


def summary_items(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None,
                  news_max: int | None = None) -> dict:
    """요약 쪽지 재료 → {title, problem(▶ 포함|None), lines(머리표 없는 글, 순서대로), more}. news_max = 뉴스 제목 최대 글자."""
    ph = phrases if phrases is not None else load_phrases()
    day = _day(summary)
    probs = problems(summary, ph, edition)
    lines = []
    mk = market_summary(summary, ph)
    if mk:
        lines.append(mk)
    lines.append(phrase(ph, "summary.todo", day, 할일=summary_todo(summary, cfg, ph, edition)))
    if edition == PERSONAL:
        mo = money_summary(summary, ph)
        if mo:
            lines.append(mo)
    nw = summary.get("news") or {}
    if nw.get("status") != "fail":
        for a in (nw.get("top") or [])[:3]:
            t = _shorten(a["title"])
            if news_max is not None and len(t) > news_max:
                t = t[:max(1, news_max - 1)].rstrip() + "…"
            lines.append(phrase(ph, "summary.news", day, 제목=t, 언론사=a["source"]))
    lb = lab_summary(summary, ph)
    if lb:
        lines.append(lb)
    return {"title": title(summary, cfg, ph),
            "problem": (BULLET + phrase(ph, "problem.prefix", day) + " ".join(probs)) if probs else None,
            "lines": lines, "more": _more_line(summary, ph, edition)}


def _render_summary(items: dict) -> str:
    out = [items["title"]] + ([items["problem"]] if items.get("problem") else [])
    out += [BULLET + x for x in items["lines"]] + [items["more"]]
    return "\n".join(out)


def summary_text(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> str:
    """텔레그램 문자 = 요약 쪽지(최대 max_lines줄·max_chars자). 넘치면 뉴스 제목 → 자료 문제 줄 순으로 줄인다."""
    ph = phrases if phrases is not None else load_phrases()
    sm = ph.get("summary") or {}
    max_chars, max_lines = int(sm.get("max_chars", 700)), int(sm.get("max_lines", 12))
    items = summary_items(summary, cfg, edition, ph)
    text = _render_summary(items)
    n = 60
    while len(text) > max_chars and n > 12:
        n -= 4
        items = summary_items(summary, cfg, edition, ph, news_max=n)
        text = _render_summary(items)
    if len(text) > max_chars and items.get("problem"):
        p = items["problem"]
        items["problem"] = p[:max(len(BULLET) + 8, len(p) - (len(text) - max_chars) - 1)].rstrip() + "…"
        text = _render_summary(items)
    lines = text.split("\n")
    if len(lines) > max_lines:  # 정상 구성은 최대 10줄 — 혹시 넘으면 끝 안내 줄은 남기고 앞에서 자른다
        lines = lines[:max_lines - 1] + [lines[-1]]
    return "\n".join(lines)[:max_chars]


def index_rows(summary: dict) -> list[dict]:
    """보고서 「미국 증시 마감」 지수 표 — [{name, close, day}] (QQQ·S&P500·반도체, 자료 있는 것만)."""
    ix = summary.get("index_close") or {}
    out = []
    for name, s in (("나스닥100(QQQ)", summary.get("qqq_close")), ("S&P500", ix.get("spx")), ("반도체(SOXX)", ix.get("soxx"))):
        c = _closes(s)
        if c is None:
            continue
        if summary.get("as_of") is not None:
            c = c[c.index <= pd.Timestamp(summary["as_of"])]
        if len(c) < 2:
            continue
        out.append({"name": name, "close": f"{c.iloc[-1]:,.2f}", "day": signed((c.iloc[-1] / c.iloc[-2] - 1) * 100) + "%"})
    return out


def report_sections(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> list[dict]:
    """첨부 보고서 칸(REPORT_ORDER 순서) — [{id, key, toc, title, sub(제목 옆 한 줄 요약), lines, …}]. 내 돈은 개인판만,
    웅덩이 후보는 장부가 있을 때만. 단체방판은 문자와 같은 규칙(금액·종목명 없음)."""
    ph = phrases if phrases is not None else load_phrases()
    rp = ph.get("report") or {}
    day = _day(summary)
    items = summary_items(summary, cfg, edition, ph)
    secs = {"summary": {"title": rp.get("summary", "■ 오늘 요약"), "sub": rp.get("summary_sub", "문자와 같은 내용"),
                        "lines": ([items["problem"][len(BULLET):]] if items.get("problem") else []) + items["lines"]}}
    n = news_count(summary)
    if summary.get("news") is not None:  # 뉴스를 받으려 한 날만(못 받았으면 실패 한 줄)
        secs["news"] = {"title": rp.get("news", "■ 뉴스 전체"), "sub": f"{n}건" if n else phrase(ph, "news.failed", day),
                        "groups": news_sections(summary, ph), "lines": []}
    therm_line = _thermometer_line(summary)
    m = market_state(summary)
    secs["market"] = {"title": rp.get("market", "■ 미국 증시 마감"), "sub": f"QQQ {signed(m['day'])}%" if m is not None else "",
                      "rows": index_rows(summary),
                      "lines": [x for x in market_lines(summary, cfg, ph, edition) if x != therm_line]}
    td = thermometer_detail(summary)
    none_text = rp.get("thermometer_none", "")
    secs["thermometer"] = {"title": rp.get("thermometer", "■ 시장 온도계"),
                           "sub": f"{td['temp']}도 · {td['zone']}" if td else none_text,
                           "detail": td, "lines": [] if td else [none_text]}
    secs["todo"] = {"title": rp.get("todo", "■ 오늘 할 일"), "sub": summary_todo(summary, cfg, ph, edition),
                    "lines": todo_lines(summary, cfg, ph, edition)}
    if edition == PERSONAL:
        mo = money_summary(summary, ph)
        secs["money"] = {"title": rp.get("money", "■ 내 돈"), "sub": (mo or "").removeprefix("내 돈: "),
                         "lines": money_lines(summary, cfg, ph)}
    cand = candidate_lines(summary, edition)
    if cand or (edition == PERSONAL and summary.get("candidate_charts")):
        c = candidate_count(summary) or (len(summary.get("candidate_charts") or []) if edition == PERSONAL else 0)
        secs["candidates"] = {"title": rp.get("candidates", "■ 웅덩이 후보"), "sub": f"{c}종목" if c else "없음",
                              "lines": cand, "caution": phrase(ph, "candidates.caution", day)}
    lb = lab_summary(summary, ph)
    lab = lab_lines(summary, cfg, ph, edition)
    if lab or (summary.get("curve_chart") or {}).get("group" if edition == GROUP else "personal"):
        secs["lab"] = {"title": rp.get("lab", "■ 가상 실험 (실제 돈 아님)"),
                       "sub": (lb or "").removeprefix("가상 실험: ").rstrip("."), "lines": lab}
    return [{"id": f"sec-{k}", "key": k, "toc": secs[k]["title"].removeprefix(SECTION).split(" (")[0], **secs[k]}
            for k in REPORT_ORDER if k in secs]
