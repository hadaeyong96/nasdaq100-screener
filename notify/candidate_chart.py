"""웅덩이 후보 종목 차트 (2026-10-26) — 개인판 HTML 「▶ 웅덩이 후보 차트」 전용(단체방판·텔레그램 그림에는 넣지 않음).

그리기만 한다: 자료는 engine/puddle_rebound.candidate_data(장부 World 지표·pr.macd_cross·pr.rsi_cross_up 그대로)가 만든다.
종목마다 3단 PNG(1200×900, 최근 120거래일):
  위   종가 + 60·120·200일선, 웅덩이 열린 날 ~ 대기 끝(또는 오늘) 옅은 색칠, 이 장부 가상 매수 ▲(2차는 테두리)·매도 ▼
  가운데 RSI(14), 40 가로선, 40 아래였다가 다시 넘은 날 점
  아래 MACD(12,26,9) 선·시그널선·막대, 골든크로스 날 점
"""

from __future__ import annotations

from pathlib import Path

from notify import curve_chart as cc

WIDTH, HEIGHT, DPI = 12, 9, 100  # 1200×900
STAGES = {"wait": "반등 확인 대기", "buy": "반등 확인 매수"}


def title(item: dict) -> str:
    """"{종목} — 웅덩이 {n}일째 · RSI {값} · {단계}" (2차면 단계 뒤 "(2차)")."""
    rsi = item.get("rsi")
    stage = STAGES[item["stage"]] + (" (2차)" if item.get("tranche") == 2 else "")
    return f"{item['tk']} — 웅덩이 {item['day']}일째 · RSI {rsi:.1f} · {stage}" if rsi is not None else \
        f"{item['tk']} — 웅덩이 {item['day']}일째 · RSI ? · {stage}"


def render(item: dict, path: Path) -> dict:
    """후보 한 종목 3단 차트 → {path, title}."""
    import matplotlib.dates as mdates

    fig = cc._figure()
    fig.set_size_inches(WIDTH, HEIGHT)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.3, 1.5], hspace=0.08, left=0.07, right=0.97, top=0.92, bottom=0.07)
    a1 = fig.add_subplot(gs[0])
    a2 = fig.add_subplot(gs[1], sharex=a1)
    a3 = fig.add_subplot(gs[2], sharex=a1)
    t = title(item)
    fig.suptitle(t, fontsize=17, y=0.97)
    x = item["close"].index
    p0, p1 = item["puddle"]
    for ax in (a1, a2, a3):
        ax.axvspan(p0, p1, color="#1f6fd1", alpha=0.08, linewidth=0)
        ax.grid(True, color="#eeeeee")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    a1.plot(x, item["close"], color="#222222", linewidth=1.8, label="종가")
    for col, color, name in (("sma60", "#f28c28", "60일선"), ("sma120", "#2e9d4f", "120일선"), ("sma200", "#8a3fd1", "200일선")):
        a1.plot(x, item[col], color=color, linewidth=1.2, label=name)
    for tr in item.get("trades") or []:
        y = item["close"].get(tr["date"])
        if y is None or y != y:
            continue
        marker = "v" if tr["kind"] == "sell" else "^"
        a1.scatter([tr["date"]], [y], marker=marker, s=110 if tr["kind"] == "buy2" else 80,
                   color="#d64545" if tr["kind"] == "sell" else "#1f6fd1",
                   edgecolors="black" if tr["kind"] == "buy2" else "none", linewidths=1.2, zorder=5)
    a1.legend(loc="upper left", frameon=False, fontsize=10, ncol=4)
    a2.plot(x, item["rsi_line"], color="#1f6fd1", linewidth=1.4)
    a2.axhline(item.get("level", 40), color="#d64545", linewidth=1, linestyle="--")
    rc = [d for d in item.get("recross") or [] if d in item["rsi_line"].index]
    a2.scatter(rc, [item["rsi_line"][d] for d in rc], color="#d64545", s=36, zorder=5)
    a2.set_ylim(0, 100)
    a2.set_ylabel("RSI(14)")
    hist = item["macd"] - item["signal"]
    a3.bar(x, hist, color=["#2e9d4f" if v >= 0 else "#d64545" for v in hist.fillna(0)], width=0.8, alpha=0.5)
    a3.plot(x, item["macd"], color="#222222", linewidth=1.3, label="MACD")
    a3.plot(x, item["signal"], color="#f28c28", linewidth=1.3, label="시그널")
    gd = [d for d in item.get("golden") or [] if d in item["macd"].index]
    a3.scatter(gd, [item["macd"][d] for d in gd], color="#2e9d4f", s=40, zorder=5, edgecolors="black", linewidths=0.6)
    a3.set_ylabel("MACD(12,26,9)")
    a3.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
    for ax in (a1, a2):
        ax.tick_params(labelbottom=False)
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%#m/%#d" if cc._windows() else "%-m/%-d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return {"path": path, "title": t}


def render_all(items: list[dict], out_dir: Path, stem: str = "candidate") -> list[dict]:
    """후보 자료 목록(최대 4개, engine이 이미 고름) → 그림 목록 [{path, title, tk}]. 없으면 빈 목록."""
    out = []
    for i, item in enumerate(items[:4]):
        r = render(item, out_dir / f"{stem}_{i + 1}_{item['tk']}.png")
        out.append({**r, "tk": item["tk"]})
    return out

