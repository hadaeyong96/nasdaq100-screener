"""아침 편지 v3 (2026-10-24) — 찌라시형 뉴스 브리핑 + 경어체. 2026-10-29부터 텔레그램은 HTML 보고서 파일 하나와
설명 한 줄(caption_line)만 보내고, 내용은 모두 보고서(report_sections)에 있다.

표시 전용 순수 함수: 이미 계산된 summary를 읽어 문장만 만든다(계산·판단 로직은 notify.letter·engine 그대로).
문장은 config/letter_phrases.yaml(사용자 수정용, 경어체). 같은 상황 문장이 여럿이면 날짜로 돌려 쓴다(전날과 다른 문장).
머리줄 "[오빠의 아침 편지] M/D(요일)"(편지 받는 한국 날짜), 섹션 "■ 이름", 항목 "▶ 한 줄". 이모지·들여쓰기 없음.
단체방판(edition="group")은 ■ 내 돈·종목명·개인 금액 없음(뉴스는 같다).

보고서 칸 순서(REPORT_ORDER): ① 오늘 요약 → ② 시장 온도(+ 지수·금리·환율) → ③ 뉴스 → ④ 오늘 할 일 → ⑤ 내 돈(개인판)
→ ⑥ 웅덩이 후보 → ⑦ 가상 실험 → 기준 줄 → ※ 유의 줄. 시장 상태 문장(장기 평균선·투자 심리·오늘의 한 줄)은 없고
시장 온도 칸이 대신한다. 폭락장·급락장 경고는 ④ 오늘 할 일에만(2026-10-29).
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
    """QQQ 종가 → {day(하루 변화 %)}. 자료가 없으면 None. (장기 평균선·국면 문장은 2026-10-29 시장 온도 칸으로 대체)"""
    q = _closes(summary.get("qqq_close"))
    if q is None:
        return None
    return {"day": (q.iloc[-1] / q.iloc[-2] - 1) * 100}


def _is_holiday_after(summary: dict) -> bool:
    """휴장 다음 날: 편지 전날(미국 날짜)이 평일인데 기준일이 그보다 앞서고, 자료 지연은 아닐 때."""
    today, as_of = summary.get("letter_today_kst"), summary.get("as_of")
    if today is None or as_of is None or (summary.get("letter_check") or {}).get("lag", 0) > 0:
        return False
    us_day = pd.Timestamp(today).normalize() - pd.Timedelta(days=1)
    return us_day.dayofweek < 5 and pd.Timestamp(as_of).normalize() < us_day


def market_lines(summary: dict, cfg: dict, ph: dict, edition: str) -> list[str]:
    """② 시장 온도 칸 아래 「지수·금리·환율」 글 줄 — (휴장 다음 날만) 휴장 안내, 미국 10년 금리·환율, (위험 단계 지표가 있으면)
    주의 한 줄. 지수 등락은 표(index_rows)에만 있다. 시장 상태 문장(오늘의 한 줄·장기 평균선·투자 심리)은 없다(2026-10-29)."""
    day = _day(summary)
    lines = []
    if _is_holiday_after(summary):
        lines.append(phrase(ph, "market.holiday", day))
    by = {r.get("slug"): r for r in summary.get("macro_rows") or []}
    rate, fx = by.get("dgs10"), by.get("fx")
    if rate or fx:
        tpl = phrase(ph, "market.rates", day)
        segs = [seg for seg in tpl.split(" · ") if ("{r}" not in seg or rate) and ("{fx}" not in seg or fx)]
        lines.append(fill(" · ".join(segs), r=f"{rate['value']:.1f}" if rate else "", fx=f"{fx['value']:,.0f}" if fx else ""))
    w = warn_line(summary, ph, day)
    if w:
        lines.append(w)
    return lines


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
    """편지 구조 {title, problem, sections[{header, lines}], footer, caution} — HTML 머리줄·기준 줄·유의 줄이 쓴다.
    줄은 글(str) 또는 뉴스 줄(dict: text·url). 머리표 "▶ "는 그릴 때 붙인다."""
    ph = phrases if phrases is not None else load_phrases()
    hd = ph.get("headers") or {}
    day = _day(summary)
    sections = news_sections(summary, ph)
    m = market_lines(summary, cfg, ph, edition)
    if m:
        sections.append({"header": hd.get("market", "■ 지수·금리·환율"), "lines": m})
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
            "caution": phrase(ph, "footer.caution", day),
            "cautions": [phrase(ph, "footer.caution", day)]}  # 웅덩이 후보 유의 줄은 그 칸 안에만(2026-10-29 중복 정리)


def line_text(line) -> str:
    """줄(str 또는 뉴스 dict) → "▶ …"."""
    return BULLET + (line["text"] if isinstance(line, dict) else str(line))


# ── 텔레그램 설명 한 줄 · 보고서 칸 (2026-10-29) ───────────────────────────────
# 텔레그램 = 보고서 파일 하나 + 설명 한 줄(caption_line). 보고서 맨 위 「오늘 요약」(최대 5줄)부터. 표시 전용 — 계산은 읽기만.

REPORT_ORDER = ("summary", "thermometer", "news", "todo", "money", "candidates", "lab")
SUMMARY_MAX_LINES = 5  # ① 오늘 요약 최대 줄 수(자료 문제 줄 포함)


def has_todo(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> bool:
    """오늘 할 일이 기본 문장(todo.none "투자 포지션 변경 없습니다.")이 아니면 True — 판마다 다르다(저축일·12월 공제는 개인판만)."""
    ph = phrases if phrases is not None else load_phrases()
    return todo_lines(summary, cfg, ph, edition) != [phrase(ph, "todo.none", _day(summary))]


def caption_line(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> str:
    """텔레그램 파일 설명 한 줄: "[오빠의 아침 편지] 10/29(목) · 시장 온도 60도"(온도가 없으면 뒷부분 생략)
    + 할 일이 기본 문장이 아니면 " · 오늘 할 일 있음" + 자료 문제가 있으면 " · 자료 문제 있음"
    + 공포 매수판이 공포 구간이면 맨 끝 " · 공포 매수판 {n}/5"(2026-10-31)."""
    ph = phrases if phrases is not None else load_phrases()
    cp = ph.get("caption") or {}
    out = [title(summary, cfg, ph)]
    t = thermometer_result(summary)
    if t:
        out.append(fill(str(cp.get("temp", "시장 온도 {T}도")), T=t["temp"]))
    if has_todo(summary, cfg, edition, ph):
        out.append(str(cp.get("todo", "오늘 할 일 있음")))
    if problems(summary, ph, edition):
        out.append(str(cp.get("problem", "자료 문제 있음")))
    fb = fear_board_suffix(summary, ph)
    if fb:
        out.append(fb)
    return " · ".join(out)


def first_sentence(text: str) -> str:
    """문장 첫 개("…니다." 까지). 마침표가 없으면 그대로."""
    m = re.match(r"(.+?[.!?])(\s|$)", str(text).strip())
    return m.group(1) if m else str(text).strip()


def summary_todo(summary: dict, cfg: dict, ph: dict, edition: str) -> str:
    """① 오늘 요약 할 일 — ④ 오늘 할 일 첫 줄의 첫 문장(저축일 → 12월 공제 → 성적표 순, 개인판만). 급락장 경고 문장은
    ④에만 두므로(2026-10-29) 급락장 줄만 있는 날은 summary.todo_crash("오늘 할 일 칸을 꼭 봐 주십시오.")."""
    day = _day(summary)
    crash = phrase(ph, "todo.crash", day)
    lines = todo_lines(summary, cfg, ph, edition)
    rest = [x for x in lines if x != crash]
    if rest:
        return first_sentence(rest[0])
    return phrase(ph, "summary.todo_crash", day)


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


def thermometer_result(summary: dict) -> dict | None:
    """그 기준일 시장 온도(engine.daily._attach_thermometer가 thermometer.core.get_temperature(기준일)로 붙인 값,
    온도계 메시지와 같은 함수). 없거나·기준일과 자료 날짜가 다르거나·모양이 이상하면 None(줄만 생략)."""
    t = summary.get("thermometer")
    try:
        if not t or t.get("temp") is None or t.get("data_date") != _as_of_day(summary):
            return None
        return t
    except Exception:
        return None


def fear_board_result(summary: dict) -> dict | None:
    """그 기준일 공포 매수판(engine.daily._attach_fear_board가 thermometer.fear_board.get_fear_board(기준일)로 붙인 값).
    없거나·기준일과 자료 날짜가 다르거나·모양이 이상하면 None(표·덧붙임만 생략)."""
    b = summary.get("fear_board")
    try:
        if not b or b.get("data_date") != _as_of_day(summary) or len(b["items"]) != 5:
            return None
        return b
    except Exception:
        return None


def fear_board_suffix(summary: dict, ph: dict) -> str | None:
    """설명 한 줄 끝 " · 공포 매수판 {n}/5" — 공포 구간인 날만(개인판·단체방판 같음), 아니면 None."""
    try:
        from thermometer.core import load_config
        from thermometer.fear_board import caption_suffix

        tpl = str((ph.get("caption") or {}).get("fear_board", "공포 매수판 {n}/5"))
        return caption_suffix(fear_board_result(summary), load_config(), tpl)
    except Exception:
        return None


def thermometer_detail(summary: dict) -> dict | None:
    """보고서 ② 시장 온도 칸 자료(thermometer.message.report_detail — 온도·구간·어제 대비·항목 표·해석) — 없거나 오류면 None."""
    try:
        from thermometer.core import load_config
        from thermometer.message import report_detail

        return report_detail(thermometer_result(summary), load_config())
    except Exception:
        return None


def market_summary(summary: dict, ph: dict) -> str | None:
    """① 오늘 요약 첫 줄 — "시장 온도 60도(미지근함) · QQQ +0.4%"(둘 중 없는 쪽은 생략). 둘 다 없으면 None."""
    day = _day(summary)
    parts = []
    t = thermometer_result(summary)
    if t:
        parts.append(phrase(ph, "summary.temp", day, T=t["temp"], 구간=t["zone"]))
    m = market_state(summary)
    if m is not None:
        parts.append(phrase(ph, "summary.market", day, q=signed(m["day"])))
    return " · ".join(parts) or None


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


def summary_items(summary: dict, cfg: dict, edition: str = PERSONAL, phrases: dict | None = None) -> dict:
    """① 오늘 요약 재료 → {title, problem(▶ 포함|None), lines(머리표 없는 글, 순서대로)}.
    줄 = 시장 온도·QQQ / 할 일 / 내 돈 합계(개인판) / 가상 실험 — 자료 문제 줄까지 최대 SUMMARY_MAX_LINES줄."""
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
    lb = lab_summary(summary, ph)
    if lb:
        lines.append(lb)
    return {"title": title(summary, cfg, ph),
            "problem": (BULLET + phrase(ph, "problem.prefix", day) + " ".join(probs)) if probs else None,
            "lines": lines[:SUMMARY_MAX_LINES - (1 if probs else 0)]}


def index_rows(summary: dict) -> list[dict]:
    """보고서 ② 「지수·금리·환율」 지수 표 — [{name, close, day}] (QQQ·S&P500·반도체, 자료 있는 것만)."""
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
    """보고서 칸(REPORT_ORDER 순서) — [{id, key, toc, title, sub(제목 옆 짧은 표시: 건수·종목 수만), lines, …}].
    ② 시장 온도 = 온도계 자료(detail) + 「지수·금리·환율」(rows 표 + market 줄). 내 돈은 개인판만, 웅덩이 후보는 장부가 있을 때만.
    단체방판은 금액·종목명 없음. 같은 문장·숫자가 두 칸에 나오지 않게 제목 옆 요약은 건수만(① 오늘 요약은 요약이라 예외)."""
    ph = phrases if phrases is not None else load_phrases()
    rp = ph.get("report") or {}
    day = _day(summary)
    items = summary_items(summary, cfg, edition, ph)
    secs = {"summary": {"title": rp.get("summary", "■ 오늘 요약"), "sub": "",
                        "lines": ([items["problem"][len(BULLET):]] if items.get("problem") else []) + items["lines"]}}
    td = thermometer_detail(summary)
    none_text = rp.get("thermometer_none", "")
    secs["thermometer"] = {"title": rp.get("thermometer", "■ 시장 온도"), "sub": "", "detail": td,
                           "lines": [] if td else [none_text], "fear_board": fear_board_result(summary),
                           "market_title": rp.get("market", "지수·금리·환율"), "rows": index_rows(summary),
                           "market_lines": market_lines(summary, cfg, ph, edition)}
    n = news_count(summary)
    if summary.get("news") is not None:  # 뉴스를 받으려 한 날만(못 받았으면 실패 한 줄)
        secs["news"] = {"title": rp.get("news", "■ 뉴스"), "sub": f"{n}건" if n else "",
                        "groups": news_sections(summary, ph), "lines": []}
    secs["todo"] = {"title": rp.get("todo", "■ 오늘 할 일"), "sub": "", "lines": todo_lines(summary, cfg, ph, edition)}
    if edition == PERSONAL:
        secs["money"] = {"title": rp.get("money", "■ 내 돈"), "sub": "", "lines": money_lines(summary, cfg, ph)}
    cand = candidate_lines(summary, edition)
    if cand or (edition == PERSONAL and summary.get("candidate_charts")):
        c = candidate_count(summary) or (len(summary.get("candidate_charts") or []) if edition == PERSONAL else 0)
        secs["candidates"] = {"title": rp.get("candidates", "■ 웅덩이 후보"), "sub": f"{c}종목" if c else "없음",
                              "lines": cand, "caution": phrase(ph, "candidates.caution", day)}
    lab = lab_lines(summary, cfg, ph, edition)
    if lab or (summary.get("curve_chart") or {}).get("group" if edition == GROUP else "personal"):
        secs["lab"] = {"title": rp.get("lab", "■ 가상 실험 (실제 돈 아님)"), "sub": "", "lines": lab}
    return [{"id": f"sec-{k}", "key": k, "toc": secs[k]["title"].removeprefix(SECTION).split(" (")[0], **secs[k]}
            for k in REPORT_ORDER if k in secs]
