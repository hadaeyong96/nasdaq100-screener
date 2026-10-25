"""가상 실험 수익곡선 겹침 그래프 (2026-10-26) — QQQ 기준(시작일 = 100), 웅덩이 v3.1·웅덩이 반등 확인.

그리기만 한다: 가상 장부 DB(data/virtual_books.db)의 일별 기록(puddle_v31_daily·puddle_rebound_daily)을 읽기만 하고
장부 계산·기록은 건드리지 않는다. 월요일(chart.weekday) 편지에는 그림으로, 매일 HTML 보고서에는 「▶ 그래프로 보기」 맨 위에.
단체방판 그림에는 종목명을 쓰지 않는다(매매 표시만).

흐름: read_rows(DB) → curve_frame(100 맞춤) · trade_marks(매매 표시) → render(PNG 1200×700, 그린 글씨 목록 반환)
기록이 chart.min_days 거래일 미만이면 그래프 대신 "기록이 쌓이는 중입니다(○일째)." 그림.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

WIDTH, HEIGHT, DPI = 12, 7, 100  # 1200×700 픽셀
TITLE = "가상 실험 성적 — QQQ 기준 (시작 {md} = 100)"
FOOTER = "가상 1억 · 실제 돈 아님 · 미국 {date} 종가 기준"
WAITING = "기록이 쌓이는 중입니다({n}일째)."
PHOTO_CAPTION = "가상 실험 성적 그래프입니다."
LINES = {  # 열 이름: (표시 이름, 색, 선 모양)
    "qqq": ("QQQ", "#8a8a8a", "--"),
    "v31": ("v3.1", "#f28c28", "-"),
    "rebound": ("반등 확인", "#1f6fd1", "-"),
}
LEGEND = {"qqq": "QQQ 그냥 보유", "v31": "웅덩이 v3.1", "rebound": "웅덩이 반등 확인"}
AHEAD_COLOR, BEHIND_COLOR, BAND_COLOR = "#2e9d4f", "#d64545", "#f28c28"
WEEKDAYS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
FONT_CANDIDATES = ("NanumGothic", "NanumBarunGothic", "Malgun Gothic", "Noto Sans KR", "Noto Sans CJK KR", "AppleGothic")


# ── 데이터 (순수 함수) ───────────────────────────────────────────────────────


def curve_frame(v31_rows: list[dict], rb_rows: list[dict], start: str | None = None) -> pd.DataFrame:
    """일별 기록 → 시작일 = 100으로 맞춘 표(날짜 순).

    입력: v31_rows(puddle_v31_daily 줄: date·v31·qqq·random), rb_rows(puddle_rebound_daily 줄: date·value·qqq), start(시작일)
    출력: DataFrame(index 날짜) 열 qqq·v31·rebound·band_lo·band_hi — 각 선 = 그날 세후 평가금액 ÷ 시작일 평가금액 × 100.
         띠(band) = v3.1 무작위 경로 10·90백분위를 v3.1 시작일 값으로 나눈 것. 없는 값은 NaN."""
    v = {r["date"]: r for r in v31_rows}
    b = {r["date"]: r for r in rb_rows}
    dates = sorted(set(v) | set(b))
    if start:
        dates = [d for d in dates if d >= start]
    raw = pd.DataFrame(index=pd.to_datetime(dates), columns=["qqq", "v31", "rebound", "band_lo", "band_hi"], dtype=float)
    for d in dates:
        rv, rb = v.get(d) or {}, b.get(d) or {}
        rnd = rv.get("random") or {}
        raw.loc[pd.Timestamp(d)] = [rv.get("qqq", rb.get("qqq")), rv.get("v31"), rb.get("value"), rnd.get("10"), rnd.get("90")]
    out = pd.DataFrame(index=raw.index)
    for col in ("qqq", "v31", "rebound"):
        s = raw[col].dropna()
        out[col] = raw[col] / s.iloc[0] * 100 if len(s) and s.iloc[0] else float("nan")
    base = raw["v31"].dropna()
    for col in ("band_lo", "band_hi"):
        out[col] = raw[col] / base.iloc[0] * 100 if len(base) and base.iloc[0] else float("nan")
    return out


def ahead_segments(line: pd.Series, qqq: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """웅덩이 선과 QQQ 사이 색 구간: [(시작일, 끝일, "ahead"|"behind")]. 같으면(0.0) 칠하지 않는다.
    "ahead" = 옅은 초록, "behind" = 옅은 빨강."""
    diff = (line - qqq).dropna()
    segs: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    prev = None
    for d, x in diff.items():
        kind = "ahead" if x > 1e-9 else "behind" if x < -1e-9 else None
        if kind and segs and segs[-1][2] == kind and segs[-1][1] == prev:
            segs[-1] = (segs[-1][0], d, kind)
        elif kind:
            segs.append((d, d, kind))
        prev = d
    return segs


def trade_marks(rows: list[dict], frame: pd.DataFrame, col: str) -> list[dict]:
    """한 장부의 매매 표시: [{date, kind: buy1|buy2|sell, tk, y, line}] — y는 그날 그 장부 선 높이.
    rows의 fills(tk·tranche·date)·sells(tk·exit_date)를 쓴다."""
    out = []
    for r in rows:
        for f in r.get("fills") or []:
            out.append({"date": pd.Timestamp(f.get("date") or r["date"]), "kind": "buy2" if f.get("tranche") == 2 else "buy1",
                        "tk": f["tk"], "line": col})
        for s in r.get("sells") or []:
            out.append({"date": pd.Timestamp(s.get("exit_date") or r["date"]), "kind": "sell", "tk": s["tk"], "line": col})
    for m in out:
        m["y"] = float(frame[col].get(m["date"], float("nan"))) if col in frame else float("nan")
    return [m for m in out if m["y"] == m["y"]]


def end_labels(frame: pd.DataFrame) -> list[tuple[str, str, float]]:
    """오른쪽 끝 글씨: [(열, 글, y)] 예 ("v31", "v3.1 100.4 (+0.4%p)", 100.4). QQQ는 값만."""
    out = []
    q = frame["qqq"].dropna()
    for col in ("qqq", "v31", "rebound"):
        s = frame[col].dropna()
        if not len(s):
            continue
        last = s.iloc[-1]
        name = LINES[col][0]
        if col == "qqq" or not len(q):
            out.append((col, f"{name} {last:.1f}", last))
        else:
            pp = round(last - q.iloc[-1], 1)
            out.append((col, f"{name} {last:.1f} ({'+' if pp > 0 else ''}{pp:.1f}%p)", last))
    return out


def spread(items: list[tuple[str, str, float]], gap: float) -> list[tuple[str, str, float]]:
    """끝 글씨가 겹치지 않게 y를 gap 이상 벌린다(아래에서 위로, 값 순서 유지)."""
    out, last = [], None
    for col, text, y in sorted(items, key=lambda t: t[2]):
        y = y if last is None or y - last >= gap else last + gap
        out.append((col, text, y))
        last = y
    return out


def is_chart_day(today_kst, cfg: dict) -> bool:
    """편지에 그림을 넣는 요일(chart.weekday, 기본 월 — 한국 날짜)인지."""
    if today_kst is None:
        return False
    wd = WEEKDAYS.get(str((cfg.get("chart") or {}).get("weekday", "월")).strip()[:1], 0)
    return pd.Timestamp(today_kst).dayofweek == wd


# ── 읽기·그리기 ──────────────────────────────────────────────────────────────


def read_rows(db_path: Path) -> tuple[list[dict], list[dict]]:
    """가상 장부 DB → (v3.1 일별 줄, 반등 확인 일별 줄). 표가 없으면 빈 목록(읽기만)."""
    from store import virtual_books as vbstore

    conn = vbstore.connect(db_path)
    try:
        def rows(table):
            try:
                return [json.loads(j) for (j,) in conn.execute(f"SELECT row_json FROM {table} ORDER BY date")]
            except Exception:  # noqa: BLE001 — 표 없음(장부 첫 저장 전)
                return []

        return rows("puddle_v31_daily"), rows("puddle_rebound_daily")
    finally:
        conn.close()


def korean_font() -> str | None:
    """설치된 한글 글꼴 이름(Actions = fonts-nanum의 NanumGothic, 윈도 = Malgun Gothic). 없으면 None."""
    from matplotlib import font_manager as fm

    names = {f.name for f in fm.fontManager.ttflist}
    return next((n for n in FONT_CANDIDATES if n in names), None)


def _figure():
    from matplotlib.figure import Figure

    font = korean_font()
    import matplotlib

    matplotlib.rcParams["axes.unicode_minus"] = False
    if font:
        matplotlib.rcParams["font.family"] = [font, "DejaVu Sans"]
    return Figure(figsize=(WIDTH, HEIGHT), dpi=DPI)


def render_waiting(path: Path, n_days: int, as_of=None, start=None) -> dict:
    """기록이 모자랄 때 그림: "기록이 쌓이는 중입니다(○일째)." (+ 제목·기준 줄) → {path, labels}. labels[0] = 그 문구."""
    fig = _figure()
    labels = [WAITING.format(n=n_days)]
    fig.text(0.5, 0.55, labels[0], ha="center", va="center", fontsize=26, color="#333333")
    if start:
        labels.append(TITLE.format(md=_md(start)))
        fig.suptitle(labels[-1], fontsize=17, y=0.96)
    if as_of:
        labels.append(FOOTER.format(date=_md(as_of)))
        fig.text(0.5, 0.06, labels[-1], ha="center", fontsize=11, color="#777777")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return {"path": path, "labels": labels, "waiting": True}


def _md(d) -> str:
    t = pd.Timestamp(d)
    return f"{t.month}/{t.day}"


def render(frame: pd.DataFrame, marks: list[dict], path: Path, *, personal: bool, as_of=None) -> dict:
    """수익곡선 그림(PNG 1200×700) → {path, labels(그림에 쓴 글씨 전부), segments}.
    단체방판(personal=False)은 매매 표시만 하고 종목명은 쓰지 않는다."""
    fig = _figure()
    ax = fig.add_axes([0.07, 0.12, 0.74, 0.76])
    labels: list[str] = []
    x = frame.index
    title = TITLE.format(md=_md(x[0]))
    fig.suptitle(title, fontsize=17, y=0.96)
    labels.append(title)
    if frame["band_lo"].notna().any():
        ax.fill_between(x, frame["band_lo"], frame["band_hi"], color=BAND_COLOR, alpha=0.08, linewidth=0,
                        label="v3.1 무작위 10~90%")
        labels.append("v3.1 무작위 10~90%")
    segments = {}
    for col in ("v31", "rebound"):
        if frame[col].notna().any():
            ahead = frame[col] > frame["qqq"]
            behind = frame[col] < frame["qqq"]
            ax.fill_between(x, frame["qqq"], frame[col], where=ahead, interpolate=True, color=AHEAD_COLOR, alpha=0.15, linewidth=0)
            ax.fill_between(x, frame["qqq"], frame[col], where=behind, interpolate=True, color=BEHIND_COLOR, alpha=0.15, linewidth=0)
            segments[col] = ahead_segments(frame[col], frame["qqq"])
    for col in ("qqq", "v31", "rebound"):
        if frame[col].notna().any():
            _, color, style = LINES[col]
            ax.plot(x, frame[col], style, color=color, linewidth=2.4 if col != "qqq" else 1.8, label=LEGEND[col])
            labels.append(LEGEND[col])
    style = {"buy1": ("^", 70, 0.55), "buy2": ("^", 110, 1.0), "sell": ("v", 80, 0.9)}
    for m in marks:
        marker, size, alpha = style[m["kind"]]
        color = LINES[m["line"]][1]
        ax.scatter([m["date"]], [m["y"]], marker=marker, s=size, color=color, alpha=alpha,
                   edgecolors="black" if m["kind"] == "buy2" else "none", linewidths=1.2, zorder=5)
        if personal:
            dy = 9 if m["kind"] != "sell" else -13
            ax.annotate(m["tk"], (m["date"], m["y"]), textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8,
                        color="#333333")
            labels.append(m["tk"])
    lo, hi = ax.get_ylim()
    for col, text, y in spread(end_labels(frame), (hi - lo) * 0.05):
        ax.annotate(text, (1.01, y), xycoords=("axes fraction", "data"), fontsize=12, va="center", color=LINES[col][1],
                    annotation_clip=False)
        labels.append(text)
    ax.axhline(100, color="#bbbbbb", linewidth=0.8)
    ax.grid(True, color="#eeeeee")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=11)
    marker_note = "▲ 1차 매수 · ▲(테두리) 2차 매수 · ▼ 매도"
    footer = FOOTER.format(date=_md(as_of if as_of is not None else x[-1]))
    fig.text(0.07, 0.03, footer + "   " + marker_note, fontsize=10, color="#777777")
    labels += [footer, marker_note]
    import matplotlib.dates as mdates

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-m/%-d" if not _windows() else "%#m/%#d"))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor="white")
    return {"path": path, "labels": labels, "segments": segments, "waiting": False}


def _windows() -> bool:
    import os

    return os.name == "nt"


def build(v31_rows: list[dict], rb_rows: list[dict], out_dir: Path, *, start: str | None, as_of=None, min_days: int = 5,
          stem: str = "curve") -> dict:
    """개인판·단체방판 그림 두 장 → {"personal": {path, labels…}, "group": {…}, "days": n}.
    기록이 min_days 거래일 미만이면 두 판 모두 "기록이 쌓이는 중입니다(○일째)." 그림."""
    frame = curve_frame(v31_rows, rb_rows, start)
    if as_of is not None:
        frame = frame[frame.index <= pd.Timestamp(as_of)]
    elif len(frame):
        as_of = frame.index[-1]
    n = int(frame[["qqq", "v31", "rebound"]].notna().any(axis=1).sum())
    out = {"days": n}
    if n < min_days:
        for ed in ("personal", "group"):
            out[ed] = render_waiting(out_dir / f"{stem}_{ed}.png", n, as_of, start or (frame.index[0] if len(frame) else None))
        return out
    cut = {r["date"] for r in v31_rows + rb_rows if pd.Timestamp(r["date"]) in frame.index}
    marks = (trade_marks([r for r in v31_rows if r["date"] in cut], frame, "v31")
             + trade_marks([r for r in rb_rows if r["date"] in cut], frame, "rebound"))
    for ed in ("personal", "group"):
        out[ed] = render(frame, marks, out_dir / f"{stem}_{ed}.png", personal=ed == "personal", as_of=as_of)
    return out
