"""실적급등 추천 편지 v2 그래프 (개인판 보고서 전용, 2026-11-11 사용자 요청 — 그래프 중심).

render       ① 한눈에 비교: 종목마다 매수일부터 "QQQ보다 얼마나 앞섰나"(QQQ 대비 초과 %p) 선. 가운데 0선 = QQQ.
             보유 중 종목은 종목마다 다른 색(LINE_COLORS, 빨강·파랑 없음 — 배경이 앞섬/뒤짐), 이름표는 각 선 끝점 바로 오른쪽.
render_card  ② 종목 카드: QQQ 캔들(왼쪽 축, 달러) + 종목 종가 선(오른쪽 축, 달러). 두 축은 매수일 시가에서 같은 높이.
입력은 engine.pead_watch.chart_data 결과(lines·cards·qqq_pct). 계산 없음 — 그리기만. 단체방판에는 넣지 않는다(종목 이름).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WIDTH, HEIGHT, DPI = 9, 5.6, 110
CARD_W, CARD_H = 9, 4.3
INK, SUB, QQQ_C, GRID = "#1F2328", "#6B7280", "#8C95A1", "#EEF0F3"
UP, DOWN, DONE, PURPLE = "#E5484D", "#3B82F6", "#B8BEC6", "#7C3AED"
BG_UP, BG_DOWN = "#FDF1F1", "#EEF3FD"
# 그림 ① 보유 중 종목 선 색(순서대로, 8개 넘으면 반복)·끝난 종목 회색 (2026-11-12)
LINE_COLORS = ["#7C3AED", "#EA580C", "#0D9488", "#DB2777", "#65A30D", "#0284C7", "#A16207", "#4F46E5"]
CLOSED_C = "#A9AFB7"
# 카드 점(2026-11-12 v2 보완): 모양으로 구별 — 반응일 검정 ★, 매수일 보라 속 빈 ○, 매도 예정일 주황 ◆
STAR_C, SELL_C = "#111111", "#F59E0B"
NAME_MAX = 8  # 이름표 글자 수 — 넘으면 줄인다
LABEL_PT = 15.0  # 이름표 한 줄 높이(포인트) — 겹침 판정·세로로 미는 간격
LABEL_SPACE = 0.92  # 한 묶음 이름표가 쓸 수 있는 세로 폭(축 높이 비율) — 넘치면 간격·글자를 줄인다
LABEL_DAYS = 35  # 이름표 자리: 가로축 오른쪽을 늘리는 날 수(약 5주) = 이름표 가로 폭으로 보는 값
TOP, BOTTOM = 0.80, 0.08
AXES_PT = HEIGHT * 72 * (TOP - BOTTOM)  # 그림 ①의 축 높이(포인트)


def short_name(name: str, limit: int = NAME_MAX) -> str:
    """긴 이름 줄이기: 띄어쓰기가 있으면 첫 낱말("어플라이드 머티어리얼즈" → "어플라이드"), 없으면 앞 limit−1글자 + "…"."""
    if len(name) <= limit:
        return name
    head = name.split(" ")[0]
    return head if " " in name and len(head) >= 2 else name[:limit - 1] + "…"


def spread_labels(ys: list[float], gap: float, lo: float, hi: float) -> list[float]:
    """세로 이름표 위치: 원래 값 순서를 지키며 서로 gap 이상 떨어지게 위로 민다. 위 끝을 넘으면 전체를 아래로. 출력 순서 = 입력 순서."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [max(ys[i], lo) for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    over = pos[-1] - hi if pos else 0
    if over > 0:
        pos = [p - over for p in pos]
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - gap)
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


def label_groups(xs: list[float], width: float) -> list[list[int]]:
    """가로로 겹칠 수 있는 이름표 묶음: 끝점 x(날짜 숫자)가 width 안에 이어지는 것끼리. 출력 = 입력 번호 묶음(x 순)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    groups: list[list[int]] = []
    for i in order:
        if groups and xs[i] - xs[groups[-1][-1]] < width:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def _setup():
    import matplotlib

    from notify.curve_chart import korean_font

    font = korean_font()
    matplotlib.rcParams["axes.unicode_minus"] = False
    if font:
        matplotlib.rcParams["font.family"] = [font, "DejaVu Sans"]


def _month_ticks(ax, dates: pd.DatetimeIndex, xs=None) -> None:
    """x축 달마다 "7월"(같은 달 한 번). xs가 있으면 정수 위치(카드), 없으면 날짜."""
    seen, ticks, labels = set(), [], []
    for k, d in enumerate(dates):
        if (d.year, d.month) in seen:
            continue
        seen.add((d.year, d.month))
        if k == 0 and d.day > 7:  # 달 중간에 시작하면 첫 눈금은 다음 달부터
            continue
        ticks.append(xs[k] if xs is not None else d)
        labels.append(f"{d.month}월")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _week_ticks(ax, dates: pd.DatetimeIndex) -> None:
    """x축 주마다 "11/2"(그 주 첫 거래일)."""
    ticks, labels, last = [], [], None
    for k, d in enumerate(dates):
        wk = d.isocalendar()[:2]
        if wk != last:
            last = wk
            ticks.append(k)
            labels.append(f"{d.month}/{d.day}")
    if len(ticks) > 1 and ticks[1] - ticks[0] < 3:  # 짧은 첫 주는 글자가 겹친다
        ticks, labels = ticks[1:], labels[1:]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _save(fig, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="white")
    return out_path


def render(chart: dict | None, out_path: Path, as_of: str | None = None) -> Path | None:
    """① 한눈에 비교 PNG. 선이 없으면 None."""
    if not chart or not chart.get("lines"):
        return None
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    _setup()
    fig = Figure(figsize=(WIDTH, HEIGHT), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.97, top=TOP, bottom=BOTTOM)
    ax = fig.add_subplot()
    fig.text(0.02, 0.955, "QQQ보다 얼마나 앞섰나", fontsize=17, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.885, "가운데 선 = QQQ · 위 = 앞섬 · 아래 = 뒤짐  ○ 매수 ◇ 60일 점검 ● 오늘", fontsize=11, color=SUB, va="top")
    if chart.get("qqq_pct") is not None:
        fig.text(0.98, 0.955, f"같은 기간 QQQ {chart['qqq_pct']:+.1f}%", fontsize=11, color=QQQ_C, fontweight="bold",
                 ha="right", va="top")
    k_open = 0
    colors = []
    for ln in chart["lines"]:  # 보유 중 종목 색은 입력 순서대로
        if ln["status"] == "open":
            colors.append(LINE_COLORS[k_open % len(LINE_COLORS)])
            k_open += 1
        else:
            colors.append(CLOSED_C)
    lines = sorted(zip(chart["lines"], colors), key=lambda t: t[0]["status"] == "open")  # 끝난 종목을 먼저(아래에) 그린다
    allv, alld = [0.0], []
    for ln, _ in lines:
        allv += [v for v in ln["excess"] if v is not None and v == v]
        alld += list(pd.to_datetime(ln["dates"]))
    lo, hi = min(allv), max(allv)
    pad = max((hi - lo) * 0.12, 1.0)
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    ax.axhspan(0, hi, color=BG_UP, zorder=0, lw=0)
    ax.axhspan(lo, 0, color=BG_DOWN, zorder=0, lw=0)
    ax.axhline(0, color=QQQ_C, lw=2.4, zorder=2)
    ends = []
    for ln, col in lines:
        open_ = ln["status"] == "open"
        last = ln["excess"][-1]
        x = pd.to_datetime(ln["dates"])
        y = ln["excess"]
        z = 4 if open_ else 3
        ax.plot(x, y, color=col, lw=2.3 if open_ else 1.3, zorder=z, solid_capstyle="round")
        ax.scatter([x[0]], [y[0]], s=40, facecolor="white", edgecolor=col, linewidth=1.6, zorder=z + 1)
        if ln.get("check_at") is not None and ln["check_at"] < len(x):
            k = ln["check_at"]
            ax.scatter([x[k]], [y[k]], s=46, marker="D", facecolor="white", edgecolor=col, linewidth=1.5, zorder=z + 1)
        ax.scatter([x[-1]], [y[-1]], s=46, color=col, zorder=z + 1)
        ends.append({"x": x[-1], "y": last, "col": col, "open": open_,
                     "text": f"{short_name(ln['name'])} {last:+.1f}%p" + ("" if open_ else " (끝)")})
    # 이름표: 각 선 끝점(●) 바로 오른쪽. 가로로 겹칠 수 있는 묶음 안에서 세로로만 밀고, 밀렸으면 같은 색 가는 선으로 잇는다
    span = hi - lo
    ys = [e["y"] for e in ends]
    fs_scale = [1.0] * len(ends)
    for grp in label_groups([e["x"].value / 864e11 for e in ends], LABEL_DAYS):
        gap = LABEL_PT / AXES_PT
        if len(grp) * gap > LABEL_SPACE:  # 너무 많으면 간격·글자를 줄인다(보유 20~30개도 잘리지 않게)
            gap = LABEL_SPACE / len(grp)
            for i in grp:
                fs_scale[i] = max(0.6, gap * AXES_PT / LABEL_PT)
        pos = spread_labels([ends[i]["y"] for i in grp], span * gap, lo + span * 0.03, hi - span * 0.03)
        for i, yy in zip(grp, pos):
            ys[i] = yy
    box = {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85}
    dx = pd.Timedelta(days=2)
    for e, yy, sc_ in zip(ends, ys, fs_scale):
        moved = abs(yy - e["y"]) > span * 0.004
        if moved:
            ax.plot([e["x"], e["x"] + dx], [e["y"], yy], color=e["col"], lw=0.8, zorder=5, solid_capstyle="round")
        ax.annotate(e["text"], xy=(e["x"] + dx, yy), xytext=(3, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=(11 if e["open"] else 9.5) * sc_, color=e["col"] if e["open"] else SUB,
                    fontweight="bold" if e["open"] else "normal", bbox=box, zorder=6, annotation_clip=False)
    ticks = [t for t in MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(lo, hi) if lo <= t <= hi]
    if 0 not in ticks:
        ticks.append(0.0)
    ticks = sorted(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["QQQ" if t == 0 else f"{t:+g}%p" for t in ticks])
    for lab in ax.get_yticklabels():
        if lab.get_text() == "QQQ":
            lab.set_fontweight("bold")
            lab.set_color(QQQ_C)
        else:
            lab.set_color(SUB)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=10, colors=SUB)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=1)
    ax.set_axisbelow(True)
    dmin, dmax = min(alld), max(alld)
    ax.set_xlim(dmin - pd.Timedelta(days=2), dmax + pd.Timedelta(days=LABEL_DAYS))  # 오른쪽 약 5주 = 이름표 자리
    _month_ticks(ax, pd.bdate_range(dmin, dmax))
    return _save(fig, out_path)


def sell_note(card: dict, next_day: str | None = None, today_kst=None) -> str:
    """카드 오른쪽 위 둘째 줄: "n/90일 · 매도 예정 YYYY-MM-DD". 매도일이 다음 거래일이면 "오늘 밤 장 마감 때 매도"
    (편지 받는 날 ≠ 다음 거래일이면 "M/D(요일) 밤 …"), 그다음 거래일이면 "내일 밤 장 마감 때 매도"."""
    from notify.pead_letter import md

    head = f"{card['held']}/{card['hold']}일"
    same = next_day is None or today_kst is None or pd.Timestamp(next_day).normalize() == pd.Timestamp(today_kst).normalize()
    if card["sell_in"] == 1:
        return f"{head} · {'오늘 밤' if same else md(next_day) + ' 밤'} 장 마감 때 매도"
    if card["sell_in"] == 2 and same:
        return f"{head} · 내일 밤 장 마감 때 매도"
    return f"{head} · 매도 예정 {card['plan_day']}"


def card_tag(card: dict) -> str:
    """카드 제목 아래 태그: 새 추천이면 반응일 숫자, 아니면 "보유 중 · 소속"."""
    if card["kind"] == "new" and card.get("sig_ret") is not None:
        return (f"새 추천 · 반응일 {card['sig_ret'] * 100:+.1f}% · QQQ 대비 {card['sig_excess'] * 100:+.1f}%p"
                f" · 거래량 {card['sig_vol']:.1f}배")
    if card["kind"] == "new":
        return f"새 추천 · {card.get('label') or ''}".rstrip(" ·")
    return f"보유 중 · {card.get('label') or ''}".rstrip(" ·")


def render_card(card: dict, out_path: Path, next_day: str | None = None, today_kst=None) -> Path:
    """② 종목 카드 PNG 한 장. 점(2026-11-12 v2 보완, 모양으로 구별): 검정 ★ = 실적 반응일(d0 종가), 보라 속 빈 ○ = 매수일 시가,
    주황 ◆ = 매도 예정일 — 가로축은 기준일까지만, 매도일이 오늘이면 그 자리에 ◆, 뒤면 오른쪽 끝 안쪽에 "→ M/D 매도 ◆".
    점 설명은 태그 줄과 축 사이 색인."""
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    _setup()
    fig = Figure(figsize=(CARD_W, CARD_H), dpi=DPI, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.90, top=0.72, bottom=0.10)
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    dates = pd.to_datetime(card["dates"])
    n = len(dates)
    xs = np.arange(n)
    fig.text(0.02, 0.96, f"{card['name']} ({card['ticker']})", fontsize=16, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.865, card_tag(card), fontsize=10.5, color=SUB, va="top")
    vq = card.get("vs_qqq")
    if vq is not None:
        fig.text(0.98, 0.96, f"QQQ 대비 {vq:+.1f}%p", fontsize=14, fontweight="bold", color=PURPLE if vq >= 0 else SUB,
                 ha="right", va="top")
    fig.text(0.98, 0.865, sell_note(card, next_day, today_kst), fontsize=10.5, color=SUB, ha="right", va="top")

    q = card["qqq"]
    k = card["buy_px"] / card["q_buy"]  # 종목 달러 = QQQ 달러 × k (매수일 시가에서 같은 높이)
    vals = [v for v in q["low"] + q["high"] if v is not None]
    vals += [c / k for c in card["close"] if c is not None]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or hi * 0.02
    lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)
    ax2.set_ylim(lo * k, hi * k)
    wbar = 0.6
    for i in range(n):
        o, h, l_, c = q["open"][i], q["high"][i], q["low"][i], q["close"][i]
        if None in (o, h, l_, c):
            continue
        col = UP if c >= o else DOWN
        ax.vlines(i, l_, h, color=col, lw=0.8, zorder=2)
        ax.add_patch(Rectangle((i - wbar / 2, min(o, c)), wbar, max(abs(c - o), (hi - lo) * 0.002), facecolor=col,
                               edgecolor=col, lw=0.5, zorder=2))
    qc = np.array([np.nan if v is None else v for v in q["close"]], dtype=float)
    ax.plot(xs, qc, color=SUB, lw=1.0, zorder=3)
    sc = np.array([np.nan if v is None else v for v in card["close"]], dtype=float)
    ax2.plot(xs, sc, color=PURPLE, lw=2.4, zorder=4, solid_capstyle="round")
    b = card["buy_at"]
    d0 = pd.Timestamp(card["d0"]) if card.get("d0") else None
    if d0 is not None and d0 in dates:
        i0 = int(dates.get_loc(d0))
        if not np.isnan(sc[i0]):
            ax2.scatter([i0], [sc[i0]], s=230, marker="*", color=STAR_C, edgecolor="white", linewidth=0.8, zorder=9)
    ax.axvline(b, color=SUB, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax2.scatter([b], [card["buy_px"]], s=90, facecolor="white", edgecolor=PURPLE, linewidth=2.2, zorder=9)
    bd = pd.Timestamp(card["buy_day"])
    right = b < n / 2
    ax2.annotate(f"매수 {bd.month}/{bd.day} ${card['buy_px']:,.0f}", (b, card["buy_px"]), xytext=(8 if right else -8, 10),
                 textcoords="offset points", ha="left" if right else "right", fontsize=10, color=PURPLE,
                 bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none"}, zorder=7)
    pdy = pd.Timestamp(card["plan_day"])
    if int(card.get("sell_in") or 0) <= 0:  # 매도일이 오늘 — 그 자리(오늘 종가)에 ◆
        if not np.isnan(sc[n - 1]):
            ax2.scatter([n - 1], [sc[n - 1]], s=90, marker="D", color=SELL_C, edgecolor="white", linewidth=1.0, zorder=9)
    else:
        ax.text(0.995, 0.03, f"→ {pdy.month}/{pdy.day} 매도 ◆", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
                color=SELL_C, fontweight="bold", bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.9},
                zorder=8)
    # 점 색인 — 그래프 바깥(태그 줄과 축 사이)
    keys = [("실적 반응일", dict(marker="*", ms=13, mfc=STAR_C, mec=STAR_C)),
            ("매수일", dict(marker="o", ms=8, mfc="white", mec=PURPLE, mew=2.0)),
            ("매도 예정일", dict(marker="D", ms=7, mfc=SELL_C, mec=SELL_C))]
    fig.legend(handles=[Line2D([], [], ls="none", label=t, **kw) for t, kw in keys],
               loc="lower left", bbox_to_anchor=(0.075, 0.745), ncol=3, fontsize=10, frameon=False, handletextpad=0.3,
               columnspacing=1.2, borderaxespad=0)
    # 끝점 이름표 — 흰 상자, 더 높은 쪽이 위
    ys = card["last_close"] / k, card["last_qqq"]
    gap = (hi - lo) * 0.13
    y_s, y_q = ys
    if abs(y_s - y_q) < gap:
        mid = (y_s + y_q) / 2
        y_s, y_q = (mid + gap / 2, mid - gap / 2) if ys[0] >= ys[1] else (mid - gap / 2, mid + gap / 2)
    box = {"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.95}
    ax.annotate(f"${card['last_close']:,.0f} {card['ret']:+.1f}%", (n - 1, ys[0]), xytext=(n - 1 + n * 0.02, y_s),
                textcoords="data", va="center", fontsize=11, fontweight="bold", color=PURPLE, bbox=box, zorder=8,
                annotation_clip=False)
    ax.annotate(f"QQQ ${card['last_qqq']:,.0f} {card['qqq_ret']:+.1f}%", (n - 1, ys[1]), xytext=(n - 1 + n * 0.02, y_q),
                textcoords="data", va="center", fontsize=10, fontweight="bold", color=SUB, bbox=box, zorder=8,
                annotation_clip=False)
    ax.set_xlim(-1, n - 1 + n * 0.30)
    (_month_ticks(ax, dates, xs) if n > 40 else _week_ticks(ax, dates))
    ax.tick_params(axis="y", colors=SUB, labelsize=9.5, length=0)
    ax2.tick_params(axis="y", colors=PURPLE, labelsize=9.5, length=0)
    ax.tick_params(axis="x", colors=SUB, labelsize=9.5, length=0)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax2.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    for a in (ax, ax2):
        a.spines["top"].set_visible(False)
        for side in ("left", "right", "bottom"):
            a.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    return _save(fig, out_path)
