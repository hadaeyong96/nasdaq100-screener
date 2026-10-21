"""아침 편지 짧은 판 (2026-10-17) — 할 일·시장 한눈에·실험 중인 전략 문구 (순수 함수, 네트워크·파일 없음).

보여 주는 순서·문구만 만든다. 장부·신호·추천 계산은 하지 않고 이미 계산된 결과(summary, 가상 장부 결과)를 읽기만 한다.
판(edition): "personal"(개인 채팅) | "group"(투자클럽 단체방 — 개인 항목·티커 없음).
이모지는 쓰지 않는다.
"""

from __future__ import annotations

import pandas as pd

from notify import virtual_books as vbtext

PERSONAL, GROUP = "personal", "group"
TODO_NONE = "없음 — QQQM 적립만 그대로"
STRATEGY_HEADER = "[실험 중인 전략] 가상 1억 · {start} 시작 · 매수 추천 아님"
GLANCE_HEADER = "[시장 한눈에]"
FAMILIES = ("hold", "trend", "picks", "rebalance")


def _md(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}"


# ── ② 오늘 할 일 ────────────────────────────────────────────────────────────


DEC_TAX_TODO = "올해 세금 없는 매도 한도 확인"


def december_items(summary: dict, cfg: dict) -> list[str]:
    """12월 공제 안내(개인판만): 표시일(12월 첫 거래일·15일 이후 첫 거래일)에 engine.daily가 summary["dec_tax"]를 둔 날만."""
    return [DEC_TAX_TODO] if summary.get("dec_tax") else []


def todo_items(summary: dict, cfg: dict, edition: str = PERSONAL) -> list[str]:
    """오늘 할 일 항목(해당 있는 것만, 짧게). 단체방판은 개인 항목(월간 성적표·12월 공제)을 뺀다."""
    items = []
    vb = summary.get("vbooks") or {}
    if vb.get("switch_today"):
        items.append(f"추세 전략 전환 신호(가상) — {_md(vb['switch_next_day'])} 체결" if vb.get("switch_next_day")
                     else "추세 전략 전환 신호(가상)")
    if edition == PERSONAL:
        sc = summary.get("scorecard")
        if sc:
            items.append(f"월간 성적표 보기({int(str(sc['month'])[-2:])}월)")
        items.extend(december_items(summary, cfg))
    return items


def todo_line(summary: dict, cfg: dict, edition: str = PERSONAL) -> str:
    """"오늘 할 일: …" 한 줄. 없으면 "오늘 할 일: 없음 — QQQM 적립만 그대로"."""
    items = todo_items(summary, cfg, edition)
    return "오늘 할 일: " + (" · ".join(items) if items else TODO_NONE)


# ── ③ 시장 한눈에 ───────────────────────────────────────────────────────────


def glance_lines(qqq_close: pd.Series | None, macro_rows: list[dict]) -> list[str]:
    """3줄 이내. 1줄 QQQ 하루·1주·200일선 대비·국면, 2줄 공포·탐욕·10년 금리·원/달러, 3줄 주의·위험 지표 이름(없으면 생략).

    입력: qqq_close(QQQ 확정 종가, 날짜 오름차순 — 없으면 1줄 생략), macro_rows(engine.daily._build_macro_rows 행)
    """
    lines = []
    if qqq_close is not None and len(qqq_close.dropna()) >= 6:
        from factory import regimes

        q = qqq_close.dropna().astype(float)
        parts = [f"QQQ 하루 {(q.iloc[-1] / q.iloc[-2] - 1) * 100:+.1f}%", f"1주 {(q.iloc[-1] / q.iloc[-6] - 1) * 100:+.1f}%"]
        if len(q) >= 200:
            sma = q.iloc[-200:].mean()
            parts.append(f"200일선 대비 {(q.iloc[-1] / sma - 1) * 100:+.1f}%")
        regime = regimes.classify(q).iloc[-1]
        if regime != regimes.UNKNOWN:
            parts.append(regime)
        lines.append(" · ".join(parts))
    by = {r.get("slug"): r for r in macro_rows or []}
    second = []
    fg = by.get("fear-greed")
    if fg:
        zone = (fg.get("badge") or {}).get("zone")
        second.append(f"공포·탐욕 {fg['value']}" + (f"({zone})" if zone else ""))
    elif by.get("vix"):
        second.append(f"VIX {by['vix']['value']:.1f}")
    if by.get("dgs10"):
        second.append(f"미 10년 금리 {by['dgs10']['value']:.2f}%")
    if by.get("fx"):
        second.append(f"원/달러 {by['fx']['value']:,.0f}원")
    if second:
        lines.append(" · ".join(second))
    danger = [_danger_text(r) for r in macro_rows or [] if (r.get("badge") or {}).get("label") == "위험"]
    if danger:  # 위험(빨강)만 짧게. 주의(노랑)는 그래프 접기에서 본다(2026-10-17)
        lines.append("위험: " + " · ".join(danger))
    return lines


def _danger_text(r: dict) -> str:
    """위험 단계 지표 하나 → 짧은 이름 + 값. 예: "공포 극단", "10년 금리 5.1%", "기준금리 인상"."""
    slug, v = r.get("slug"), r.get("value")
    zone = (r.get("badge") or {}).get("zone") or ""
    if slug == "fear-greed":
        return {"극단적 공포": "공포 극단", "극단적 탐욕": "탐욕 극단"}.get(zone, f"공포·탐욕 {v}")
    if slug == "vix":
        return f"VIX {v:.1f}"
    if slug == "dgs10":
        return f"10년 금리 {v:.1f}%"
    if slug == "t10y2y":
        return f"금리차 {v:+.2f}%p"
    if slug == "hy":
        return f"하이일드 스프레드 {v:.2f}%"
    if slug == "fed":
        return "기준금리 인상"
    if slug == "fx":
        return f"원/달러 {v:,.0f}원"
    return str(r.get("name", slug))


# ── ④ 실험 중인 전략 ─────────────────────────────────────────────────────────


def _family_of(cfg: dict) -> dict[str, str]:
    vb = cfg.get("virtual_books") or {}
    fam = {}
    for b in vb.get("books", []):
        fam[b["name"]] = "trend" if b.get("kind") == "sma_switch" else "hold"
    for b in (vb.get("picks") or {}).get("books", []):
        fam[b["name"]] = "picks"
    for key in ("momentum", "value_score", "accel"):  # 8번 자리: 2026-10-18부터 가치평가(6개월 강세는 꺼 둠)
        c = vb.get(key) or {}
        if c.get("name") and c.get("enabled", True):
            fam[c["name"]] = "rebalance"
    pn = vb.get("penny") or {}
    if pn.get("name") and pn.get("enabled", True):  # 10번 동전주 + 판정 비교용 그림자(2026-10-21)
        fam[pn["name"]] = "penny"
        if pn.get("shadow_name"):
            fam[pn["shadow_name"]] = "shadow"
    return fam


def letter_books(cfg: dict, base_date, summaries: list[dict], book_trades: dict[str, list[dict]],
                 extra: list[dict], bench_daily: pd.DataFrame | None, waiting: dict[str, pd.Timestamp] | None = None,
                 enabled: list[str] | None = None) -> list[dict]:
    """가상 장부 결과 → 편지·표용 장부 목록(config 순서). 계산은 하지 않고 이미 나온 값만 옮긴다.

    입력: summaries(1~5번 core.virtual_books.summarize_book 결과), book_trades({1~5번 이름: 매매 dict 목록}),
         extra(6~9번 engine.picks_books extra), bench_daily(QQQ 전략 일별 — 늦게 시작한 장부의 QQQ 대비용),
         waiting({이름: 시작 예정일} — 아직 시작 전 장부), enabled(편지에 낼 장부 이름 순서, 없으면 config 순서 전체)
    출력: [{name, label, family, status(ok|waiting|missing), cum_return_pct, vs_qqq_pp, holding, n_positions,
            trades_today[(구분, 티커)], start, value, mdd_pct, switches}]
    """
    base = pd.Timestamp(base_date).normalize()
    fam = _family_of(cfg)
    order = enabled or list(fam)
    vb = cfg.get("virtual_books") or {}
    bench_name = vb.get("benchmark_book")
    by_sum = {s["book"]: s for s in summaries or []}
    by_extra = {x["name"]: x for x in extra or []}
    bench_cum = by_sum.get(bench_name, {}).get("cum_return_pct")
    out = []
    for name in order:
        row = {"name": name, "label": vbtext.display_name(cfg, name), "family": fam.get(name, "hold"),
               "status": "missing", "cum_return_pct": None, "vs_qqq_pp": None, "holding": "", "n_positions": None,
               "trades_today": [], "start": None, "value": None, "mdd_pct": None, "switches": None}
        if name in by_sum:
            s = by_sum[name]
            row.update(status="ok", cum_return_pct=s["cum_return_pct"], holding=s.get("holding", ""),
                       start=pd.Timestamp(s["start"]) if s.get("start") is not None else None, value=s.get("value"),
                       mdd_pct=s.get("mdd_pct"), switches=s.get("switches"))
            if name != bench_name and bench_cum is not None:
                row["vs_qqq_pp"] = s["cum_return_pct"] - bench_cum
            trades = book_trades.get(name, [])
        elif name in by_extra:
            x = by_extra[name]
            s = x["summary"]
            row.update(status="ok", cum_return_pct=s["cum_return_pct"], start=pd.Timestamp(x["start"]), value=s["value"],
                       mdd_pct=s["mdd_pct"], n_positions=len(x.get("positions") or {}),
                       holding=", ".join(sorted(x.get("positions") or {})))
            if bench_daily is not None and not bench_daily.empty:
                bv = bench_daily["value"]
                at = bv[bv.index <= row["start"]]
                if len(at):
                    row["vs_qqq_pp"] = s["cum_return_pct"] - (float(bv.iloc[-1]) / float(at.iloc[-1]) - 1) * 100
            trades = x.get("trades") or []
        elif waiting and name in waiting:
            row.update(status="waiting", start=pd.Timestamp(waiting[name]))
            trades = []
        else:
            trades = []
        row["trades_today"] = [(t["side"], t["ticker"]) for t in trades
                               if t.get("fill_date") is not None and pd.Timestamp(t["fill_date"]).normalize() == base
                               and t.get("side") in ("매수", "매도")]
        out.append(row)
    return out


def _status(b: dict, cfg: dict) -> tuple[str, str]:
    """(묶는 데 쓰는 상태 키, 상태 문구). 같은 키끼리 한 줄로 묶는다(2026-10-17 다듬기)."""
    if b["status"] == "waiting":
        return f"wait:{b['start'].date()}", f"{_md(b['start'])} 시작"
    if b["status"] == "missing":
        return "missing", "오늘 계산 못함"
    if b["family"] == "trend":
        defensive = (cfg.get("virtual_books") or {}).get("defensive_ticker", "SGOV")
        if b["holding"] == defensive:
            return "down", "하락 추세라 단기 국채로 피신"
        return "up", "상승 추세라 {holdings} 보유"
    if b["family"] in ("picks", "rebalance", "penny", "shadow"):
        n = b["n_positions"] or 0
        return ("cash", "살 종목이 없어 현금 보관") if n == 0 else (f"n:{n}", f"{n}종목 보유")
    return "hold", "그냥 보유"


def _ret(b: dict) -> str:
    """누적 수익률 한 칸: +1.2% / −0.1% / 0.0% (음수는 긴 빼기표)."""
    v = round(b["cum_return_pct"], 1) + 0.0
    return "0.0%" if v == 0 else (f"+{v:.1f}%" if v > 0 else f"−{abs(v):.1f}%")


def join_names(labels: list[str]) -> str:
    """보여 주는 이름 묶기: "QLD 추세 전략"+"TQQQ 추세 전략" → "QLD·TQQQ 추세 전략",
    "기술지표 전략"+"볼린저 전략"+"가치평가 전략" → "기술지표·볼린저·가치평가 전략"."""
    if len(labels) == 1:
        return labels[0]
    words = [x[:-3].split() if x.endswith(" 전략") else x.split() for x in labels]
    common = []
    while all(len(w) > len(common) + 1 for w in words) and len({w[-1 - len(common)] for w in words}) == 1:
        common.insert(0, words[0][-1 - len(common)])
    heads = [" ".join(w[: len(w) - len(common)]) for w in words]
    return "·".join(heads) + (" " + " ".join(common) if common else "") + " 전략"


def _trades_line(books: list[dict], edition: str) -> str:
    """칸 맨 끝 한 줄: 개인판 "오늘 가상 매수: 전략명 티커 · …", 단체방판 "오늘 가상 거래 n건". 없으면 "오늘 거래 없음"."""
    if edition == GROUP:
        n = sum(len(b["trades_today"]) for b in books)
        return f"오늘 가상 거래 {n}건" if n else "오늘 거래 없음"
    parts = []
    for side in ("매수", "매도"):
        per = [(b["label"], list(dict.fromkeys(t for s, t in b["trades_today"] if s == side))) for b in books]
        per = [f"{label} {', '.join(ts)}" for label, ts in per if ts]
        if per:
            parts.append(f"오늘 가상 {side}: " + " · ".join(per))
    return " / ".join(parts) if parts else "오늘 거래 없음"


def strategy_lines(books: list[dict], cfg: dict, edition: str = PERSONAL) -> list[str]:
    """같은 상태인 전략을 한 줄로: "QLD·TQQQ 추세 전략: 상승 추세라 QLD·TQQQ 보유 (−5.3% · −8.0%)".
    QQQ 대비는 넣지 않는다(표로 보기 접기에만). 마지막 줄은 그날 가상 매매 한 줄."""
    groups: list[dict] = []
    for b in books:
        key, text = _status(b, cfg)
        g = next((x for x in groups if x["key"] == key), None)
        if g is None:
            groups.append({"key": key, "text": text, "books": [b]})
        else:
            g["books"].append(b)
    lines = []
    for g in groups:
        bs = g["books"]
        text = g["text"].format(holdings="·".join(dict.fromkeys(b["holding"] for b in bs)))
        line = f"{join_names([b['label'] for b in bs])}: {text}"
        if all(b["status"] == "ok" and b["cum_return_pct"] is not None for b in bs):
            line += " (" + " · ".join(_ret(b) for b in bs) + ")"
        lines.append(line)
    lines.append(_trades_line(books, edition))
    return lines


def strategy_header(books: list[dict], start=None) -> str:
    """"[실험 중인 전략] 가상 1억 · 10/14 시작 · 매수 추천 아님"."""
    first = start or min((b["start"] for b in books if b.get("start") is not None and b["status"] == "ok"), default=None)
    return STRATEGY_HEADER.format(start=_md(first) if first is not None else "-")


def _holding_cell(b: dict, cfg: dict, edition: str) -> str:
    """보유 칸. 단체방판은 종목 티커 대신 "n종목"·"단기 국채"(전략 이름에 있는 QQQ·QLD·TQQQ만 그대로)."""
    ok = b["status"] == "ok"
    defensive = (cfg.get("virtual_books") or {}).get("defensive_ticker", "SGOV")
    if b["family"] in ("picks", "rebalance", "penny", "shadow"):
        n = b["n_positions"] or 0
        if not ok:
            return ""
        if n == 0:
            return "현금" if edition == GROUP else f"현금({defensive})"
        # 10번 동전주·그림자는 개인판도 종목 수만(20종목 명단은 시트에)
        return f"{n}종목" if edition == GROUP or b["family"] in ("penny", "shadow") else b["holding"]
    if edition == GROUP and b["holding"] == defensive:
        return "단기 국채"
    return b["holding"]


def table_rows(books: list[dict], cfg: dict | None = None, edition: str = PERSONAL) -> tuple[list[str], list[list[str]]]:
    """접은 표(9줄): (머리글, 행). 모든 값이 0(또는 비어 있음)인 열은 뺀다(0만 있는 칸 숨김)."""
    cfg = cfg or {}
    head = ["전략", "총자산($)", "누적 수익", "QQQ 대비", "최대 낙폭", "전환 횟수", "보유"]
    rows = []
    for b in books:
        ok = b["status"] == "ok"
        rows.append([
            b["label"], f"{b['value']:,.0f}" if ok and b["value"] is not None else "",
            f"{b['cum_return_pct']:+.2f}%" if ok else ("" if b["status"] == "missing" else f"{_md(b['start'])} 시작"),
            f"{b['vs_qqq_pp']:+.2f}%p" if ok and b["vs_qqq_pp"] is not None else "",
            f"{b['mdd_pct']:.1f}%" if ok and b["mdd_pct"] is not None else "",
            str(b["switches"]) if ok and isinstance(b["switches"], int) else "",
            _holding_cell(b, cfg, edition),
        ])
    keep = [i for i in range(len(head)) if i == 0 or any(_nonzero(r[i]) for r in rows)]
    return [head[i] for i in keep], [[r[i] for i in keep] for r in rows]


def _nonzero(cell: str) -> bool:
    s = cell.replace("%p", "").replace("%", "").replace(",", "").replace("+", "").replace("-", "").strip()
    if s == "":
        return False
    try:
        return float(s) != 0.0
    except ValueError:
        return True


# ── 아침 편지 v2.1 (2026-10-21) — 점검·내 돈·가상 장부 3줄 묶음·연구 소식 ─────────────────────────

WEEKDAY_KR = "월화수목금토일"
MONEY_HEADER = "[1 내 돈 · QQQM]"
GLANCE_HEADER_V2 = "[2 시장 한눈에]"
BOOKS_HEADER_V2 = "[3 가상 장부] 가상 1억 · {start} 시작 · QQQ 대비 누적 · 매수 추천 아님"
RESEARCH_HEADER = "[4 연구 소식]"
RESEARCH_HEADER_GROUP = "[연구 소식]"
MONEY_EMPTY = "체결 탭에 QQQM 평균단가를 넣으면 여기에 표시됩니다"
GROUPS = (("base", "기준선"), ("lev", "레버리지(1~5)"), ("picks", "종목 고르기(6~10)"))
SHADOW_LINE = "비교: {cell} (10번 판정용, QQQ 대비)"


def _mdw(ts) -> str:
    """"10/16(금)"."""
    ts = pd.Timestamp(ts)
    return f"{ts.month}/{ts.day}({WEEKDAY_KR[ts.dayofweek]})"


def _pp(v: float) -> str:
    """%p 한 칸: +1.2%p / −0.3%p / 0.0%p (음수는 긴 빼기표)."""
    v = round(v, 1) + 0.0
    return "0.0%p" if v == 0 else (f"+{v:.1f}%p" if v > 0 else f"−{abs(v):.1f}%p")


def check_info(books: list[dict], base_date, expected_date, lag_days: int, skipped: dict[str, str] | None = None,
               notes: list[str] | None = None) -> dict:
    """점검 줄 재료 (표시 전용). books: letter_books 결과, skipped: {보여 주는 이름: 사유}(건너뛴 장부),
    notes: 끝에 붙일 말(예: "동전주 선정 못함(목록 못 받음)")."""
    reasons = dict(skipped or {})
    books = [b for b in books if b.get("family") != "shadow"]  # 판정 비교용 그림자는 장부 수에 넣지 않는다
    recorded = [b["label"] for b in books if b["status"] == "ok"]
    waiting = [b["label"] for b in books if b["status"] == "waiting"]
    # 건너뜀은 그날 계산 못 한 장부만(계산은 됐는데 경고만 있는 장부는 빼고), 사유를 모르면 "오류"
    skipped = {b["label"]: reasons.get(b["label"], "오류") for b in books if b["status"] == "missing"}
    return {"base": pd.Timestamp(base_date) if base_date is not None else None,
            "expected": pd.Timestamp(expected_date) if expected_date is not None else None,
            "lag": int(lag_days or 0), "recorded": recorded, "waiting": waiting, "skipped": skipped,
            "notes": list(dict.fromkeys(notes or []))}


def check_line(info: dict) -> str:
    """"점검: 기준일 10/16(금) · 장부 8개 기록, 1개 시작 대기(가치평가 전략) · 데이터 지연 없음".
    기준일이 직전 거래일보다 늦으면 "데이터 지연 k거래일(기대 10/19(월))", 건너뛴 장부는 " · 건너뜀(이름·사유)"를 붙인다."""
    parts = [f"기준일 {_mdw(info['base'])}" if info.get("base") is not None else "기준일 알수없음"]
    rec = f"장부 {len(info['recorded'])}개 기록"
    if info["waiting"]:
        rec += f", {len(info['waiting'])}개 시작 대기({', '.join(info['waiting'])})"
    parts.append(rec)
    if info["lag"] > 0:
        exp = f"(기대 {_mdw(info['expected'])})" if info.get("expected") is not None else ""
        parts.append(f"데이터 지연 {info['lag']}거래일{exp}")
    else:
        parts.append("데이터 지연 없음")
    if info["skipped"]:
        parts.append("건너뜀(" + ", ".join(f"{k}·{v}" for k, v in info["skipped"].items()) + ")")
    parts.extend(info.get("notes") or [])
    return "점검: " + " · ".join(parts)


def skip_reasons(cfg: dict, messages: list[str]) -> dict[str, str]:
    """가상 장부 경고·오류 문구 → {보여 주는 이름: 짧은 사유}. 문구가 "장부 id: …" 또는 "장부 id 오류: …"로 시작하는 것만."""
    out = {}
    names = list(vbtext.display_map(cfg))
    for msg in messages or []:
        for n in names:
            if msg.startswith(n + ":") or msg.startswith(n + " 오류"):
                if "SEC_USER_AGENT" in msg:
                    reason = "SEC 키 없음"
                elif "시작일" in msg and "전" in msg:
                    continue  # 시작 전은 '시작 대기'로 센다
                elif "오류" in msg:
                    reason = "오류"
                else:
                    reason = msg.split(":", 1)[1].strip()[:20]
                out[vbtext.display_name(cfg, n)] = reason
    return out


def money_summary(pos, close_usd: float | None, fx: float | None, year: int) -> dict | None:
    """「체결」 탭 QQQM 이동평균 보유(core.tax_harvest.Position) → 내 돈 숫자. 보유 기록이 없으면 None."""
    if pos is None or pos.rows == 0 or pos.qty <= 0 or not close_usd or not fx:
        return None
    from core import tax_harvest as th

    value = pos.qty * close_usd * fx
    plan = th.harvest_plan(pos, year, close_usd, fx)
    return {"qty": pos.qty, "close_usd": close_usd, "value_krw": value, "cost_krw": pos.cost_krw,
            "pnl_krw": value - pos.cost_krw, "pnl_pct": (value / pos.cost_krw - 1) * 100 if pos.cost_krw > 0 else None,
            "realized_krw": plan.realized_this_year_krw, "remaining_krw": plan.remaining_krw}


def money_lines(m: dict | None) -> list[str]:
    """[1 내 돈 · QQQM] 2줄. 기록이 없으면 안내 한 줄."""
    if not m:
        return [MONEY_EMPTY]
    pct = f"({m['pnl_pct']:+.1f}%)" if m.get("pnl_pct") is not None else ""
    return [f"QQQM {m['qty']:g}주 · 평가 {m['value_krw']:,.0f}원 · 손익 {m['pnl_krw']:+,.0f}원{pct}",
            f"올해 실현 이익 {m['realized_krw']:,.0f}원 · 12월 공제 남은 금액 {m['remaining_krw']:,.0f}원"]


def _group_of(b: dict, cfg: dict) -> str:
    bench = (cfg.get("virtual_books") or {}).get("benchmark_book")
    if b["name"] == bench:
        return "base"
    if b["family"] == "shadow":
        return "shadow"
    return "picks" if b["family"] in ("picks", "rebalance", "penny") else "lev"


def _book_cell(b: dict) -> str:
    if b["status"] == "waiting":
        return f"{b['label']} 시작 대기"
    if b["status"] == "missing" or b.get("vs_qqq_pp") is None:
        return f"{b['label']} 계산 못함" if b["status"] == "missing" else f"{b['label']} -"
    return f"{b['label']} {_pp(b['vs_qqq_pp'])}"


def book_lines(books: list[dict], cfg: dict, edition: str = PERSONAL) -> list[str]:
    """[3 가상 장부] 3줄 묶음(기준선 / 레버리지 1~5 / 종목 고르기 6~10) + 10번 그림자 %p 한 줄 + 오늘 거래 건수 한 줄.
    QQQ 대비 누적 %p를 먼저 보인다. 기준선은 개인판만 누적 수익률을 붙인다. 종목명은 쓰지 않는다(상세는 접기)."""
    lines = []
    for key, title in GROUPS:
        bs = [b for b in books if _group_of(b, cfg) == key]
        if not bs:
            continue
        if key == "base":
            b = bs[0]
            if b["status"] == "ok" and b["cum_return_pct"] is not None and edition == PERSONAL:
                lines.append(f"{title}: {b['label']} {_ret(b)} (아래 %p의 기준)")
            else:
                lines.append(f"{title}: {b['label']} (아래 %p의 기준)")
            continue
        lines.append(f"{title}: " + " · ".join(_book_cell(b) for b in bs))
    for b in (b for b in books if _group_of(b, cfg) == "shadow"):  # 그림자는 %p 한 줄만
        lines.append(SHADOW_LINE.format(cell=_book_cell(b)))
    n = sum(len(b["trades_today"]) for b in books if b["family"] != "shadow")
    if edition == GROUP:
        lines.append(f"오늘 가상 거래 {n}건" if n else "오늘 거래 없음")
    else:
        lines.append(f"오늘 가상 거래 {n}건 (상세는 아래 접기)" if n else "오늘 거래 없음")
    return lines


def trade_detail_line(books: list[dict]) -> str:
    """개인판 접기 안 오늘 거래 상세(전략명 + 티커). 판정 비교용 그림자 거래는 넣지 않는다."""
    return _trades_line([b for b in books if b.get("family") != "shadow"], PERSONAL)


def research_news(cfg: dict, today_kst, shown=()) -> tuple[list[str], list[str]]:
    """[4 연구 소식] 줄들 (2026-10-21, config research_news) — 개인판·단체방판 같은 글. 날짜는 한국 날짜.

    보이는 날: ① 판정일 다음 첫 편지(판정일 < 오늘, 아직 보인 적 없음 — shown에 이름이 없음) ② 매주 월요일(최근 7일 안에
    판정 난 것 + 연구 중 목록). 그 밖의 날은 빈 목록. 보이는 날에는 연구 중 줄도 붙인다.
    입력: cfg, today_kst(오늘 한국 날짜), shown(이미 편지에 보인 판정 이름들)
    출력: (줄 목록, 이번에 처음 보인 판정 이름 목록 — 호출부가 기록)
    """
    news = cfg.get("research_news") or {}
    if today_kst is None:
        return [], []
    today = pd.Timestamp(today_kst).normalize()
    monday = today.dayofweek == 0
    seen = set(shown or ())
    show = []
    for item in news.get("finished") or []:
        days = (today - pd.Timestamp(item["date"]).normalize()).days
        if days <= 0:
            continue
        if item["name"] not in seen or (monday and days <= 7):
            show.append(item)
    lines = []
    for it in show:
        lines += [f"시험 끝: 「{it['name']}」({it['desc']})", f"{it['period']} 시험 → {it['result']} → {it['verdict']}",
                  f"이유: {it['reason']}"]
    if not show and not monday:
        return [], []
    prog = [str(x) for x in news.get("in_progress") or []]
    if prog:
        lines.append("연구 중: " + " · ".join(prog))
    return lines, [it["name"] for it in show if it["name"] not in seen]
