"""시장 온도 지표 상태 판정 + 스파크라인 좌표 (P3.8). 순수 함수 — 네트워크·현재 시각 접근 금지.

표시 전용이다. 여기서 나오는 어떤 값도 신호 판정·필터·수량 계산에 쓰이지 않는다
(docs/p3_8_instructions.md — P5-2에서 국면 필터가 성과를 나쁘게 만들었기 때문).
상태 배지는 기호(■·▲·●·▼)와 글자를 함께 써서 색만으로 구분하지 않는다.
"""

from __future__ import annotations

STATUS_OK = "ok"
STATUS_WARN = "warn"
STATUS_BAD = "bad"
STATUS_INFO = "info"

_SYMBOL = {STATUS_OK: "●", STATUS_WARN: "▲", STATUS_BAD: "■", STATUS_INFO: "▼"}


def _badge(status: str, label: str) -> dict:
    return {"status": status, "symbol": _SYMBOL[status], "label": label, "text": f"{_SYMBOL[status]} {label}"}


def classify_fear_greed(value: float, thresholds: dict) -> dict:
    """0~100 공포·탐욕 지수(또는 대체 VIX) -> 상태 배지.

    입력: value, thresholds({extreme_fear, fear, neutral_high, greed})
    출력: {status, symbol, label, text}. 경계는 표(0~24/25~44/45~55/56~75/76~100) 그대로:
         value < extreme_fear -> 극단적 공포, < fear -> 공포, <= neutral_high -> 중립,
         <= greed -> 탐욕, 그 외 -> 극단적 탐욕.
    """
    if value < thresholds["extreme_fear"]:
        return _badge(STATUS_BAD, "극단적 공포")
    if value < thresholds["fear"]:
        return _badge(STATUS_WARN, "공포")
    if value <= thresholds["neutral_high"]:
        return _badge(STATUS_OK, "중립")
    if value <= thresholds["greed"]:
        return _badge(STATUS_WARN, "탐욕")
    return _badge(STATUS_BAD, "극단적 탐욕")


def classify_vix_fallback(value: float, thresholds: dict) -> dict:
    """공포·탐욕 지수를 3일 넘게 못 받았을 때 대체하는 VIX 상태 배지 (P3.8 1번).

    입력: value(VIXCLS), thresholds({caution, alert})
    출력: {status, symbol, label, text}. value < caution -> 안정, < alert -> 주의, 그 외 -> 경계.
    """
    if value < thresholds["caution"]:
        return _badge(STATUS_OK, "안정")
    if value < thresholds["alert"]:
        return _badge(STATUS_WARN, "주의")
    return _badge(STATUS_BAD, "경계")


def classify_rise_over_window(current: float, value_n_ago: float | None, threshold_pp: float) -> dict:
    """DGS10: 일정 기간(63거래일 약 3개월) 전 대비 상승폭이 기준(pp) 이상이면 주의.

    입력: current, value_n_ago(그 기간 전 값, 없으면 비교 불가), threshold_pp
    출력: {status, symbol, label, text}. value_n_ago가 None이면 비교 불가로 "정보 없음"(info).
    """
    if value_n_ago is None:
        return _badge(STATUS_INFO, "비교값 없음")
    if current - value_n_ago >= threshold_pp:
        return _badge(STATUS_WARN, "주의")
    return _badge(STATUS_OK, "안정")


def classify_t10y2y(value: float, thresholds: dict) -> dict:
    """장단기 금리차: >=normal 정상 · 0<=v<normal 주의 · v<0 경계(역전).

    입력: value, thresholds({normal, alert})
    """
    if value >= thresholds["normal"]:
        return _badge(STATUS_OK, "정상")
    if value >= thresholds["alert"]:
        return _badge(STATUS_WARN, "주의")
    return _badge(STATUS_BAD, "경계(역전)")


def classify_hy_spread(value: float, thresholds: dict) -> dict:
    """하이일드 스프레드: <caution 안정 · caution<=v<alert 주의 · v>=alert 경계.

    입력: value, thresholds({caution, alert})
    """
    if value < thresholds["caution"]:
        return _badge(STATUS_OK, "안정")
    if value < thresholds["alert"]:
        return _badge(STATUS_WARN, "주의")
    return _badge(STATUS_BAD, "경계")


def classify_fed_funds_trend(current: float, value_3m_ago: float | None, epsilon: float = 1e-9) -> dict:
    """기준금리 상단: 3개월 전보다 높으면 인상 흐름 · 낮으면 인하 흐름 · 같으면 동결.

    입력: current, value_3m_ago(없으면 비교 불가)
    출력: {status, symbol, label, text}. 인상=warn(▲), 인하=info(▼), 동결=ok(●), 비교불가=info.
    """
    if value_3m_ago is None:
        return _badge(STATUS_INFO, "비교값 없음")
    diff = current - value_3m_ago
    if diff > epsilon:
        return _badge(STATUS_WARN, "인상 흐름")
    if diff < -epsilon:
        return _badge(STATUS_INFO, "인하 흐름")
    return _badge(STATUS_OK, "동결")


def percentile_rank(series: list[float], value: float) -> float:
    """series 안에서 value가 차지하는 백분위(0~100, <= 기준)를 구한다.

    입력: series(1년치 등 과거 값 목록, value 포함 여부 무관), value(오늘 값)
    출력: series 중 value 이하인 값의 비율(%). series가 비어 있으면 50.0(중립 취급).
    """
    if not series:
        return 50.0
    count_leq = sum(1 for v in series if v <= value)
    return count_leq / len(series) * 100


def classify_fx_percentile(value: float, series_1y: list[float], thresholds: dict) -> dict:
    """원/달러 환율: 1년 백분위 >=pct_high 달러 비쌈 · <=pct_low 달러 쌈 · 그 외 보통.

    입력: value, series_1y(최근 1년 값), thresholds({pct_high, pct_low})
    """
    pct = percentile_rank(series_1y, value)
    if pct >= thresholds["pct_high"]:
        return _badge(STATUS_WARN, "달러 비쌈")
    if pct <= thresholds["pct_low"]:
        return _badge(STATUS_INFO, "달러 쌈")
    return _badge(STATUS_OK, "보통")


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
