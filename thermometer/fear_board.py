"""「공포 매수판」 (2026-10-31) — 조건 5개 중 켜진 수로 평소·눌림·공포를 나눈다. 기준값은 config/thermometer.yaml fear_board.

 ① 공포·탐욕 ≤ 25 (온도계와 같은 값 — thermometer.core._fear_greed, stale_days 안의 가장 최근 값)
 ② QQQ RSI(14) ≤ 30 (core.indicators.rsi_wilder)
 ③ QQQ 종가 < 볼린저 하단(20일, 2σ) (core.bollinger_signals.add_bollinger_columns)
 ④ QQQ 종가 ≤ SMA60 그리고 SMA20 < 21거래일 전 SMA20
 ⑤ VIX 종가 ≥ 30 (온도계와 같은 값 — thermometer.core._vix, 그날 종가가 없으면 자료 없음)
판정은 board() 한 곳(라이브·과거 통계 공용). 자료 없는 조건은 ✗로 세지 않고 "자료 없음"(ok=None)으로 둔다 — count는 켜진(✓) 조건 수.

미래 자료 금지: 모든 입력을 그날까지 잘라서 계산한다(rolling·양수 shift만). prices를 주면 네트워크·파일을 건드리지 않는
순수 계산이고, 안 주면 thermometer.data.load_live(시장 폭 없이)로 오늘 자료를 받는다. prices 형식은 thermometer.core와 같다.
"""

from __future__ import annotations

import pandas as pd

from core.bollinger_signals import add_bollinger_columns
from core.indicators import rsi_wilder
from thermometer.core import _fear_greed, _vix, load_config

KEYS = ("fear_greed", "rsi", "bollinger", "trend", "vix")


def zone_of(count: int, cfg: dict) -> dict:
    """켜진 조건 수 → {"name", "line"} (fear_board.zones에서 max 이하인 첫 구간)."""
    zones = cfg["fear_board"]["zones"]
    for z in zones:
        if count <= z["max"]:
            return {"name": z["name"], "line": z["line"]}
    return {"name": zones[-1]["name"], "line": zones[-1]["line"]}


def qqq_frame(q: pd.Series, cfg: dict) -> pd.DataFrame:
    """QQQ 종가 Series → 날짜별 ②~④ 원값 DataFrame (close, rsi, bb_lower, sma_long, sma_short, sma_short_lag —
    자료가 모자란 날은 NaN). 양수 shift·rolling만 쓴다. 판정은 board()가 한다."""
    fb = cfg["fear_board"]
    q = q.dropna().astype(float)
    bb = fb["bollinger"]
    # add_bollinger_columns는 가상 장부용 열도 만든다 — 여기서는 bb_lower만 쓰므로 나머지 기준값은 계산에만 필요한 자리 채움
    bands = add_bollinger_columns(q.to_frame("close"), {
        "period": bb["period"], "std_mult": bb["std_mult"], "ddof": bb["ddof"],
        "slope_days": 1, "touch_days": 1, "width_rank_period": bb["period"], "width_rank_max": 0.0, "squeeze_bonus": 0})
    out = pd.DataFrame(index=q.index)
    out["close"] = q
    out["rsi"] = rsi_wilder(q, int(fb["rsi_days"])).where(q.expanding().count() > int(fb["rsi_days"]))
    out["bb_lower"] = bands["bb_lower"]
    out["sma_long"] = q.rolling(int(fb["sma_long"])).mean()
    out["sma_short"] = q.rolling(int(fb["sma_short"])).mean()
    out["sma_short_lag"] = out["sma_short"].shift(int(fb["sma_lag"]))
    return out


def _num(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def board(vals: dict, cfg: dict) -> dict:
    """그날 원값 → 판 (판정은 이 함수 한 곳에서만 — 라이브·과거 통계 공용).

    입력: vals{fear_greed, rsi, close, bb_lower, sma_long, sma_short, sma_short_lag, vix} — 없는 값은 None/NaN
    출력: {count(✓ 수), n_avail(자료 있는 조건 수), items[5]{key, name, ok(True|False|None=자료 없음), value(표시용|None)},
          zone, line, missing[이름]}
    """
    fb, names = cfg["fear_board"], cfg["fear_board"]["names"]
    v = {k: _num(vals.get(k)) for k in ("fear_greed", "rsi", "close", "bb_lower", "sma_long", "sma_short", "sma_short_lag", "vix")}
    fg, rsi, c, lo, vix = v["fear_greed"], v["rsi"], v["close"], v["bb_lower"], v["vix"]
    trend_have = c is not None and v["sma_long"] is not None and v["sma_short"] is not None and v["sma_short_lag"] is not None
    items = [
        {"key": "fear_greed", "ok": None if fg is None else fg <= float(fb["fear_greed_max"]),
         "value": None if fg is None else f"{fg:.0f}"},
        {"key": "rsi", "ok": None if rsi is None else rsi <= float(fb["rsi_max"]),
         "value": None if rsi is None else f"{rsi:.1f}"},
        {"key": "bollinger", "ok": None if c is None or lo is None else c < lo,
         "value": None if c is None or lo is None else f"{c:.2f} / 하단 {lo:.2f}"},
        {"key": "trend", "ok": (c <= v["sma_long"] and v["sma_short"] < v["sma_short_lag"]) if trend_have else None,
         "value": f"{c:.2f} / 60일선 {v['sma_long']:.2f} · 20일선 {v['sma_short']:.2f} (21일 전 {v['sma_short_lag']:.2f})"
         if trend_have else None},
        {"key": "vix", "ok": None if vix is None else vix >= float(fb["vix_min"]),
         "value": None if vix is None else f"{vix:.1f}"},
    ]
    for it in items:
        it["name"] = names[it["key"]]
    count = sum(1 for it in items if it["ok"])
    z = zone_of(count, cfg)
    return {"count": count, "n_avail": sum(1 for it in items if it["ok"] is not None), "items": items,
            "zone": z["name"], "line": z["line"], "missing": [it["name"] for it in items if it["ok"] is None]}


def get_fear_board(date, prices: dict | None = None, cfg: dict | None = None) -> dict:
    """그날의 공포 매수판.

    입력: date(YYYY-MM-DD·date·Timestamp — 과거 날짜도 됨), prices(thermometer.core 형식: qqq·vix·fear_greed, 없으면 오늘 자료를
         받음), cfg(config/thermometer.yaml, 없으면 읽음)
    출력: {"date", "data_date"(그날까지 QQQ 마지막 거래일|None), "count"(✓ 수 0~5), "n_avail"(자료 있는 조건 수),
          "items"[5]{key, name, ok(True|False|None=자료 없음), value(표시용 원값|None)}, "zone"(평소|눌림|공포), "line", "missing"[이름]}
    """
    cfg = cfg or load_config()
    ts = pd.Timestamp(date).normalize()
    if prices is None:
        from thermometer.data import load_live

        prices = load_live(ts.date(), cfg, breadth=False)
    q = prices["qqq"].loc[:ts].dropna()
    out = {"date": ts.date().isoformat(), "data_date": None}
    if len(q) == 0:
        out.update(board({}, cfg))
        return out
    day = q.index[-1]
    out["data_date"] = day.date().isoformat()
    vals = qqq_frame(q, cfg).iloc[-1].to_dict()
    vx = _vix(prices, day, cfg)
    vals.update(fear_greed=_fear_greed(prices, day, cfg), vix=None if vx is None else vx["v"])
    out.update(board(vals, cfg))
    return out


def caption_suffix(board: dict | None, cfg: dict, template: str = "공포 매수판 {n}/5") -> str | None:
    """텔레그램 설명 한 줄 끝 덧붙임 — 공포 구간(fear_board.caption_zone)인 날만 "공포 매수판 {n}/5", 아니면 None."""
    if not board or board.get("zone") != cfg["fear_board"]["caption_zone"]:
        return None
    return template.format(n=board["count"])
