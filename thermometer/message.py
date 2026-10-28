"""시장 온도계 텔레그램 메시지 (개인방·단체방 같은 글, 개인정보 없음). 이모지 없음, 경어체, 왼쪽 정렬."""

from __future__ import annotations

from datetime import date

WEEKDAYS = "월화수목금토일"


def _deg(x: float) -> str:
    return f"{x:.1f}"


def _side(x: float) -> str:
    return "위" if x > 0 else "아래"


def change_text(delta: int | None) -> str | None:
    """어제 대비 → "어제보다 3도 ↑" | "어제보다 2도 ↓" | "어제와 같음" | None(어제 값 없음)."""
    if delta is None:
        return None
    if delta == 0:
        return "어제와 같음"
    return f"어제보다 {abs(delta)}도 {'↑' if delta > 0 else '↓'}"


def build_message(result: dict, cfg: dict, day: date) -> str:
    """get_temperature 결과 → 메시지 글. day = 머리줄 날짜(한국 날짜, 편지와 같음)."""
    names, p = cfg["names"], result["parts"]
    lines = [f"[시장 온도계] {day.month}/{day.day}({WEEKDAYS[day.weekday()]})"]
    head = [f"■ 오늘 온도: {result['temp']}도", result["zone"]]
    ch = change_text(result.get("delta"))
    if ch:
        head.append(ch)
    lines.append(" · ".join(head))
    for key, days in (("sma_fast", cfg["sma"]["fast_days"]), ("sma_slow", cfg["sma"]["slow_days"])):
        if p[key]["deg"] is not None:
            x = p[key]["raw"]
            lines.append(f"▶ {names[key]}: {_deg(p[key]['deg'])}도 ({days}일선보다 {abs(x):.1f}% {_side(x)})")
    if p["breadth"]["deg"] is not None:
        extra = ", 강한 회복" if p["breadth"].get("strong_recovery") else ""
        lines.append(f"▶ {names['breadth']}: {_deg(p['breadth']['deg'])}도 "
                     f"({cfg['breadth']['sma_days']}일선 위 종목 {p['breadth']['raw']:.0f}%{extra})")
    if p["vix"]["deg"] is not None:
        peak = p["vix"].get("peak")
        extra = f", 지난달 {peak:.1f}에서 하락" if peak is not None else ""
        lines.append(f"▶ {names['vix']}: {_deg(p['vix']['deg'])}도 ({p['vix']['raw']:.1f}{extra})")
    if p["fear_greed"]["deg"] is not None:
        lines.append(f"▶ {names['fear_greed']}: {_deg(p['fear_greed']['deg'])}도 "
                     f"({p['fear_greed']['raw']:.0f}, {p['fear_greed']['label']})")
    if result.get("note"):
        lines.append(f"▶ {result['note']}")
    if result["missing"]:
        lines.append(f"▶ 자료 없음: {', '.join(names[k] for k in result['missing'])}. 나머지로 환산했습니다.")
    lines.append(cfg["footer"])
    return "\n".join(lines)


def report_detail(result: dict | None, cfg: dict) -> dict | None:
    """첨부 보고서 「시장 온도계」 칸 → {temp, zone, rows[{key, name, raw, deg}], note, missing[이름]}. 온도가 없으면 None."""
    if not result or result.get("temp") is None:
        return None
    fmt = {"sma_fast": lambda v: f"{v:+.1f}%", "sma_slow": lambda v: f"{v:+.1f}%", "breadth": lambda v: f"{v:.0f}%",
           "vix": lambda v: f"{v:.1f}", "fear_greed": lambda v: f"{v:.0f}"}
    rows = [{"key": k, "name": cfg["names"][k], "raw": fmt[k](p["raw"]), "deg": _deg(p["deg"])}
            for k, p in result["parts"].items() if p["deg"] is not None]
    return {"temp": result["temp"], "zone": result["zone"], "rows": rows, "note": result.get("note"),
            "missing": [cfg["names"][k] for k in result["missing"]]}
