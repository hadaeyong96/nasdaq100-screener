"""시장 온도 지표 상태 판정 + 스파크라인 좌표 (P3.8). 순수 함수 — 네트워크·현재 시각 접근 금지.

표시 전용이다. 여기서 나오는 어떤 값도 신호 판정·필터·수량 계산에 쓰이지 않는다
(docs/p3_8_instructions.md — P5-2에서 국면 필터가 성과를 나쁘게 만들었기 때문).
모든 지표를 🟢안정 · 🟡주의 · 🔴위험 3단계로 통일해 classify_macro 한 곳에서 판정한다
(기준값은 config.yaml macro.thresholds). 이모지와 글자를 함께 써서 색만으로 구분하지 않는다.
"""

from __future__ import annotations

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_BAD = "bad"
STATUS_INFO = "info"  # 비교값이 없어 판단을 못 한 경우(기준금리 6개월 전 값 없음 등)만

# 초보자용 3단계 (2026-09-30): 모든 지표를 🟢안정 · 🟡주의 · 🔴위험 하나로 통일한다.
# 색만으로 구분하지 않게 이모지 + 글자를 함께 쓴다.
_SYMBOL = {STATUS_OK: "🟢", STATUS_WARN: "🟡", STATUS_BAD: "🔴", STATUS_INFO: "⚪"}
_LABEL = {STATUS_OK: "안정", STATUS_WARN: "주의", STATUS_BAD: "위험", STATUS_INFO: "판단 보류"}
LEVELS = (STATUS_OK, STATUS_WARN, STATUS_BAD)

# 공포·탐욕 구간 이름 (0~100)
_FG_ZONES = ("극단적 공포", "공포", "중립", "탐욕", "극단적 탐욕")


def _fmt(x: float) -> str:
    """기준값을 config.yaml에 적힌 모양 그대로 표시: 1300 -> "1,300", 4.0 -> "4.0", 0.5 -> "0.5", 6 -> "6"."""
    return format(x, ",")


def _judgment(status: str, scale_ranges: dict[str, str], zone: str | None = None) -> dict:
    """판정 결과 dict. scale은 🟢→🟡→🔴 순서의 3칸 눈금이고 오늘 칸에 current=True(★).

    출력: {status, symbol, label, text, zone, scale:[{status, symbol, label, range, current}]}
    """
    scale = [
        {"status": lv, "symbol": _SYMBOL[lv], "label": _LABEL[lv], "range": scale_ranges[lv], "current": lv == status}
        for lv in LEVELS
    ]
    return {
        "status": status, "symbol": _SYMBOL[status], "label": _LABEL[status],
        "text": f"{_SYMBOL[status]} {_LABEL[status]}", "zone": zone, "scale": scale,
    }


def _higher_is_worse(value: float, caution: float, danger: float, unit: str) -> dict:
    """값이 클수록 위험한 지표: value < caution 안정 · caution <= value < danger 주의 · value >= danger 위험."""
    ranges = {
        STATUS_OK: f"{_fmt(caution)}{unit} 미만",
        STATUS_WARN: f"{_fmt(caution)}~{_fmt(danger)}{unit}",
        STATUS_BAD: f"{_fmt(danger)}{unit} 이상",
    }
    if value < caution:
        return _judgment(STATUS_OK, ranges)
    if value < danger:
        return _judgment(STATUS_WARN, ranges)
    return _judgment(STATUS_BAD, ranges)


def fear_greed_zone(value: float, th: dict) -> str:
    """공포·탐욕 구간 이름: <extreme_fear 극단적 공포 · <fear 공포 · <=neutral_high 중립 ·
    <=greed 탐욕 · 그 외 극단적 탐욕 (0~24/25~44/45~55/56~75/76~100)."""
    if value < th["extreme_fear"]:
        return _FG_ZONES[0]
    if value < th["fear"]:
        return _FG_ZONES[1]
    if value <= th["neutral_high"]:
        return _FG_ZONES[2]
    if value <= th["greed"]:
        return _FG_ZONES[3]
    return _FG_ZONES[4]


def classify_fear_greed(value: float, th: dict) -> dict:
    """CNN 공포·탐욕(0~100): 중립(45~55) 안정 · 공포(25~44)·탐욕(56~75) 주의 ·
    극단적 공포(0~24)·극단적 탐욕(76~100) 위험. 구간 이름을 zone으로 함께 돌려준다.

    입력: value, th({extreme_fear, fear, neutral_high, greed})
    """
    zone = fear_greed_zone(value, th)
    ef, f, nh, g = th["extreme_fear"], th["fear"], th["neutral_high"], th["greed"]
    ranges = {
        STATUS_OK: f"{f}~{nh} (중립)",
        STATUS_WARN: f"{ef}~{f - 1} (공포) · {nh + 1}~{g} (탐욕)",
        STATUS_BAD: f"0~{ef - 1} (극단적 공포) · {g + 1}~100 (극단적 탐욕)",
    }
    status = {"중립": STATUS_OK, "공포": STATUS_WARN, "탐욕": STATUS_WARN}.get(zone, STATUS_BAD)
    return _judgment(status, ranges, zone)


def classify_t10y2y(value: float, th: dict) -> dict:
    """장단기 금리차(%p): >= stable 안정 · danger <= v < stable 주의 · v < danger(0, 역전) 위험.

    입력: value, th({stable, danger})
    """
    stable, danger = th["stable"], th["danger"]
    ranges = {
        STATUS_OK: f"+{_fmt(stable)}%p 이상",
        STATUS_WARN: f"{_fmt(danger)}~+{_fmt(stable)}%p",
        STATUS_BAD: f"{_fmt(danger)} 미만 (역전)",
    }
    if value >= stable:
        return _judgment(STATUS_OK, ranges)
    if value >= danger:
        return _judgment(STATUS_WARN, ranges)
    return _judgment(STATUS_BAD, ranges)


def classify_fed_funds(current: float, value_before: float | None, th: dict, epsilon: float = 1e-9) -> dict:
    """기준금리(상단): lookback_months개월 전보다 내렸으면 안정(인하) · 같으면 주의(동결) ·
    올렸으면 위험(인상). 비교값이 없으면 판단 보류(info, ★ 없음).

    입력: current, value_before(lookback_months개월 전 값 또는 None), th({lookback_months})
    """
    m = th["lookback_months"]
    ranges = {
        STATUS_OK: f"최근 {m}개월 인하",
        STATUS_WARN: f"최근 {m}개월 동결",
        STATUS_BAD: f"최근 {m}개월 인상",
    }
    if value_before is None:
        return _judgment(STATUS_INFO, ranges)
    diff = current - value_before
    if diff < -epsilon:
        return _judgment(STATUS_OK, ranges, "인하")
    if diff > epsilon:
        return _judgment(STATUS_BAD, ranges, "인상")
    return _judgment(STATUS_WARN, ranges, "동결")


def value_months_ago(series: dict[str, float], latest_date_iso: str, months: int) -> float | None:
    """series({날짜: 값})에서 latest_date_iso의 months개월 전 날짜 이하 가장 최근 값 (없으면 None)."""
    import calendar
    from datetime import date

    d = date.fromisoformat(latest_date_iso)
    y, mo = divmod(d.month - 1 - months, 12)
    target_year, target_month = d.year + y, mo + 1
    day = min(d.day, calendar.monthrange(target_year, target_month)[1])  # 8/31의 6개월 전 -> 2/28(29)
    target = date(target_year, target_month, day).isoformat()
    candidates = [k for k in series if k <= target]
    return series[max(candidates)] if candidates else None


def classify_macro(code: str, value: float, thresholds: dict, *, value_before: float | None = None) -> dict:
    """시장 온도 지표 하나를 🟢안정·🟡주의·🔴위험으로 판정한다 — 판정은 이 함수 한 곳에서만 한다.
    기준값은 모두 config.yaml macro.thresholds에서 받는다(하드코딩 금지). 표시 전용.

    입력: code("FEAR_GREED"|"VIXCLS"|"DGS10"|"T10Y2Y"|"BAMLH0A0HYM2"|"DFEDTARU"|"DEXKOUS"),
         value(오늘 값), thresholds(macro.thresholds 전체),
         value_before(DFEDTARU만: lookback_months개월 전 값)
    출력: {status, symbol, label, text, zone, scale}
    """
    th = thresholds[code]
    if code == "FEAR_GREED":
        return classify_fear_greed(value, th)
    if code == "T10Y2Y":
        return classify_t10y2y(value, th)
    if code == "DFEDTARU":
        return classify_fed_funds(value, value_before, th)
    unit = {"DGS10": "%", "BAMLH0A0HYM2": "%", "DEXKOUS": "원", "VIXCLS": ""}[code]
    return _higher_is_worse(value, th["caution"], th["danger"], unit)


def scale_ranges(code: str, thresholds: dict) -> list[dict]:
    """판정과 무관하게 그 지표의 3칸 기준표만 (읽는 법 표 등). 출력: [{status, symbol, label, range}] 3개."""
    probe = {"FEAR_GREED": 50, "T10Y2Y": 1.0, "DFEDTARU": 0.0}.get(code, 0.0)
    out = classify_macro(code, probe, thresholds, value_before=0.0)["scale"]
    return [{k: v for k, v in row.items() if k != "current"} for row in out]


def is_stale(as_of_date_iso: str, report_date_iso: str, stale_days: int, trading_days_between) -> bool:
    """지표 기준일이 보고서 기준일보다 stale_days거래일 넘게 오래됐는지.

    입력: as_of_date_iso(지표 실제 기준일), report_date_iso(보고서 기준일),
         stale_days, trading_days_between(start,end)->거래일 목록을 돌려주는 호출 가능 객체
         (data.market_calendar.trading_days_between 등 — 순수성을 지키려고 주입받는다)
    출력: bool
    """
    if as_of_date_iso >= report_date_iso:
        return False
    gap = trading_days_between(as_of_date_iso, report_date_iso)
    return max(len(gap) - 1, 0) > stale_days


def value_to_y(value: float, min_v: float, max_v: float, height: float = 34.0, pad: float = 4.9) -> float:
    """값 하나를 스파크라인 y좌표로 바꾼다(SVG는 아래로 갈수록 y가 커짐).

    min_v == max_v(값이 모두 같음)면 0으로 나누지 않고 세로 가운데(height/2)를 반환한다.
    """
    if max_v == min_v:
        return height / 2
    frac = (value - min_v) / (max_v - min_v)
    frac = max(0.0, min(1.0, frac))  # 값이 범위를 벗어나도(기준선 등) 그래프 안에 고정
    return (height - pad) - frac * (height - 2 * pad)


def sparkline_points(values: list[float], width: float = 150.0, height: float = 34.0, pad_y: float = 4.9) -> str:
    """1년치 값 목록을 SVG polyline의 "x,y x,y ..." 좌표 문자열로 바꾼다 (순수 함수).

    x는 2.0 ~ (width-2.0) 사이에 값 개수만큼 균등 배치한다(템플릿 예시와 동일한 여백).
    값이 하나뿐이거나 모두 같으면 0으로 나누지 않고 y를 세로 가운데로 고정한다.
    """
    if not values:
        return ""
    min_v, max_v = min(values), max(values)
    n = len(values)
    x_pad = 2.0
    step = (width - 2 * x_pad) / (n - 1) if n > 1 else 0.0
    points = []
    for i, v in enumerate(values):
        x = x_pad + step * i
        y = value_to_y(v, min_v, max_v, height, pad_y)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)
