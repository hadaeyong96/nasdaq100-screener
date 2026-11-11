"""실적급등 추천 장부 → 아침 편지 문장 (표시 전용 순수 함수, docs/prereg/pead_watch.md 9장).

입력은 engine.pead_watch.run의 letter 칸(summary["pead_watch"]). 문장은 짧은 단문 경어체.
개인판 v2(2026-11-11, 그래프 중심): 오늘 할 일 → 새 추천(있을 때만) → 한눈에 비교(그래프 ①) → 종목 카드(그래프 ②)
→ 실적 뉴스. 성적표·끝난 종목·실적 발표 예정은 접기 「자세히 보기」(fold_sections). 머리 한 줄은 없음(사용자 지시).
단체방판: 종목 이름·그래프·뉴스 없이 개수·할 일 유무·성적표 한 줄.
"""

from __future__ import annotations

import pandas as pd

WEEKDAY = "월화수목금토일"
LEDGER_NAMES = {"main": "90일", "compare": "60일"}


def md(d) -> str:
    """'2026-11-12' → '11/12(목)'."""
    t = pd.Timestamp(d)
    return f"{t.month}/{t.day}({WEEKDAY[t.dayofweek]})"


def pct(x, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:+.{d}f}%"


def pp(x, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:+.{d}f}%p"


def who(r: dict) -> str:
    """'엔비디아(NVDA)' — 한글 이름이 없으면 티커만."""
    n = r.get("name") or r["ticker"]
    return r["ticker"] if n == r["ticker"] else f"{n}({r['ticker']})"


def _when(next_day: str | None, today_kst) -> str:
    """매수·매도 시점 말머리: 다음 거래일이 편지 받는 한국 날짜와 같으면 '오늘 밤', 아니면 'M/D(요일) 밤'."""
    if next_day and today_kst is not None and pd.Timestamp(next_day).normalize() != pd.Timestamp(today_kst).normalize():
        return f"{md(next_day)} 밤"
    return "오늘 밤"


def todo_lines(L: dict, today_kst=None) -> list[str]:
    """오늘 할 일 — 매수(다음 거래일 시가)·매도(다음 거래일 종가, 90일). 없으면 '오늘 할 일 없습니다.'"""
    w = _when(L.get("next_day"), today_kst)
    out = []
    buys, sells = L["actions"]["buy"], L["actions"]["sell"]
    if buys:
        out.append(f"{w} 장 시작 매수: {', '.join(who(r) for r in buys)}")
    if sells:
        out.append(f"{w} 장 마감 매도: {', '.join(who(r) for r in sells)}")
    return out or ["오늘 할 일 없습니다."]


def new_lines(L: dict) -> list[str]:
    """새 추천 — 종목당 한 줄 "회사명(티커) · 반응일 +31.0% · 거래량 5.1배 · 매도 예정 3/19". 없으면 빈 목록(칸 생략)."""
    out = []
    for r in L["new"]:
        if r.get("skip") == "보유 중":
            tail = "이미 보유 중이라 새로 담지 않습니다"
        else:
            tail = f"매도 예정 {md(r['sell_day'])}"
        late = " · 늦게 확인" if r.get("late") else ""
        out.append(f"{who(r)} · 반응일 {pct(r['ret'])} · 거래량 {r['vol_ratio']:.1f}배 · {tail}{late}")
    return out


def watching_lines(L: dict) -> list[str]:
    """③ 관찰 중 — 보유 n/90일, 매수가 대비·QQQ 대비, 매도 예정일. 매수 대기는 매수일만."""
    if not L["watching"]:
        return ["관찰 중인 종목이 없습니다."]
    out = []
    for r in L["watching"]:
        if r["status"] == "pending":
            out.append(f"{who(r)} · 매수 대기({md(r['buy_day'])} 시가) · 매도 예정 {md(r['plan_day'])}")
        else:
            out.append(f"{who(r)} · 보유 {r['held']}/{r['hold']}일 · 매수가 대비 {pct(r['ret'])} · QQQ 대비 {pp(r['vs_qqq'])}"
                       f" · 매도 예정 {md(r['plan_day'])}")
    return out


def earnings_lines(L: dict) -> list[str]:
    """④ 실적 발표 예정(이번 주·다음 주) — 날짜별 종목. 확인 실패 수가 있으면 한 줄 덧붙인다."""
    e = L.get("earnings") or {}
    rows = e.get("rows") or []
    out = []
    if rows:
        by: dict[str, list[str]] = {}
        for r in rows:
            by.setdefault(r["date"], []).append(who(r))
        out += [f"{md(d)}: {', '.join(v)}" for d, v in by.items()]
    else:
        out.append("이번 주·다음 주 발표 예정 종목이 없습니다.")
    if e.get("failed"):
        out.append(f"예정일 확인 실패 {e['failed']}개")
    return out


def news_lines(news: dict | None) -> list:
    """실적 발표 뉴스(data.news.earnings_news) — 줄 = {text, title, source, url}(제목 링크). 없으면 한 줄 안내."""
    from core.news import shorten

    if not news or news.get("status") == "fail":
        return ["실적 발표 뉴스를 받지 못했습니다."]
    if not news.get("top"):
        return ["최근 24시간 실적 발표 뉴스가 없습니다."]
    out = []
    for a in news["top"]:
        t = shorten(a["title"])
        out.append({"text": f"{t} ({a['source']})", "title": t, "source": a["source"], "url": a.get("url", "")})
    return out


def _score_line(name: str, sc: dict) -> str:
    if not sc or not sc.get("n"):
        return f"성적표 {name}: 아직 끝난 매매가 없습니다."
    basis = "" if sc.get("basis") == "cost" else " (비용 전)"
    return f"성적표 {name}: {sc['n']}건 · 평균 초과 {sc['mean']:+.2f}%{basis} · 승률 {sc['win']:.0f}%"


def closed_lines(L: dict) -> list[str]:
    """⑤ 끝난 종목 최근 5개 + 성적표(90·60 따로, 비용 뒤·세금 전 QQQ 대비)."""
    out = [f"{who(r)} · {md(r['sell_day'])} 매도 · {pct(r['ret'])} · QQQ 대비 {pp(r['vs_qqq'])}" for r in L["closed"]]
    if not out:
        out.append("끝난 종목이 아직 없습니다.")
    for k in ("main", "compare"):
        out.append(_score_line(LEDGER_NAMES[k], (L.get("scorecards") or {}).get(k)))
    return out


def group_lines(L: dict) -> list[str]:
    """단체방판 — 종목 이름·티커 없이 "새 추천 n개 · 보유 n개 · 오늘 할 일 있음/없음" + 성적표(90일) 한 줄."""
    c = L.get("counts") or {}
    todo = "있음" if (L["actions"]["buy"] or L["actions"]["sell"]) else "없음"
    return [f"새 추천 {c.get('new', 0)}개 · 보유 {c.get('watching', 0)}개 · 오늘 할 일 {todo}",
            _score_line(LEDGER_NAMES["main"], (L.get("scorecards") or {}).get("main"))]


def cards_title(L: dict) -> str:
    n = len((L.get("chart") or {}).get("cards") or [])
    return f"■ 종목 카드 (보유 {n}개)"


def sections(pwr: dict | None, edition: str, today_kst=None, news: dict | None = None) -> list[dict]:
    """보고서 본문 칸 [{key, title, lines}]. pwr = summary["pead_watch"] ({status, letter}), news = summary["earnings_news"].
    그래프 칸(pead_compare·pead_cards)은 lines가 비어 있고 그림은 보고서가 붙인다(그림이 없으면 한 줄 안내). 오류면 한 줄."""
    if not pwr:
        return []
    if pwr.get("status") == "error":
        return [{"key": "pead_error", "title": "■ 실적급등 추천", "lines": ["실적급등 장부 계산에 실패했습니다. 오늘은 표시하지 않습니다."]}]
    L = pwr["letter"]
    if edition == "group":
        return [{"key": "pead_group", "title": "■ 실적급등 추천", "lines": group_lines(L)}]
    secs = [{"key": "pead_todo", "title": "■ 오늘 할 일", "lines": todo_lines(L, today_kst)}]
    if L["new"]:
        secs.append({"key": "pead_new", "title": "■ 새 추천", "lines": new_lines(L)})
    has = bool((L.get("chart") or {}).get("lines"))
    secs.append({"key": "pead_compare", "title": "■ 한눈에 비교", "lines": [] if has else ["보유 중인 종목이 없습니다."]})
    if (L.get("chart") or {}).get("cards"):
        secs.append({"key": "pead_cards", "title": cards_title(L), "lines": []})
    secs.append({"key": "pead_news", "title": "■ 실적 뉴스", "lines": news_lines(news)})
    return secs


def fold_sections(pwr: dict | None, edition: str) -> list[dict]:
    """접기 「자세히 보기」 맨 위 칸(개인판): 성적표(90·60 따로) → 끝난 종목 최근 5개 → 실적 발표 예정(이번 주·다음 주)."""
    if not pwr or pwr.get("status") == "error" or edition == "group":
        return []
    L = pwr["letter"]
    closed = closed_lines(L)
    return [
        {"key": "pead_score", "title": "성적표", "lines": closed[-2:]},
        {"key": "pead_closed", "title": "끝난 종목 (최근 5개)", "lines": closed[:-2]},
        {"key": "pead_earnings", "title": "실적 발표 예정 (이번 주·다음 주)", "lines": earnings_lines(L)},
    ]
