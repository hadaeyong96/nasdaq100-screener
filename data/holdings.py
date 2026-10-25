"""사용자 시트(「전략공장」) 「QQQM 모으기」 탭 읽기 (2026-10-22 새 출발 — 읽기 전용, 프로그램은 어떤 탭도 쓰지 않는다).

칸은 2행(머리글 줄) 이름으로 찾고 값은 3행에서 읽는다: "보유 주식 수"·"평균단가($/원)"·"평균 환율(원/$)"·"올해 실현 이익(원)"
(지금 배치: B3·C3·D3·K3). 이름·탭은 config holdings에서 읽는다. 빈칸은 None. 숫자에 쉼표·원·$가 섞여 있어도 읽는다.
머리글 이름은 문자열 하나 또는 후보 목록(예전 이름 「평균단가($)」도 읽음, 2026-10-23).

ISA 줄(2026-10-23): A열이 isa_row_name("TIGER 미국나스닥100")인 행에서 보유 주식 수·평균단가(원)를 같은 머리글로 읽는다
(지금 B4·C4). 현재가는 fetch_isa_close(yfinance 133690.KS 종가, 원화).
"""

from __future__ import annotations

import re

DEFAULTS = {"tab": "QQQM 모으기", "data_row": 3,
            "headers": {"qty": "보유 주식 수", "avg_price": ["평균단가($/원)", "평균단가($)"], "avg_fx": "평균 환율(원/$)",
                        "realized_krw": "올해 실현 이익(원)"},
            "isa_row_name": "TIGER 미국나스닥100", "isa_ticker": "133690.KS"}
ISA_KEYS = ("qty", "avg_price")  # ISA 줄에서 읽는 칸(평균단가는 원화)


def settings(cfg: dict | None) -> dict:
    """config holdings를 기본값과 합친다."""
    c = dict(DEFAULTS)
    c.update((cfg or {}).get("holdings") or {})
    c["headers"] = {**DEFAULTS["headers"], **(c.get("headers") or {})}
    return c


def _num(v) -> float | None:
    s = re.sub(r"[,\s원$₩]", "", str(v if v is not None else ""))
    if s in ("", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _cell(values: list[list], r: int, c: int):
    return values[r][c] if r < len(values) and c < len(values[r]) else ""


def _col(head: list[str], name) -> int | None:
    """머리글 이름(문자열 또는 후보 목록) → 열 번호. 없으면 None."""
    for n in [name] if isinstance(name, str) else list(name):
        if n in head:
            return head.index(n)
    return None


def _label(name) -> str:
    return name if isinstance(name, str) else name[0]


def parse_holdings(values: list[list], cfg: dict | None = None) -> dict:
    """탭 값(get_all_values 2차원 목록) → {qty, avg_price, avg_fx, realized_krw, isa: {qty, avg_price}, warnings}.

    머리글은 2행, QQQM 값은 data_row행(기본 3행), ISA 값은 A열이 isa_row_name인 행. 못 찾은 머리글·행은 None + 경고.
    """
    s = settings(cfg)
    out = {"qty": None, "avg_price": None, "avg_fx": None, "realized_krw": None,
           "isa": {k: None for k in ISA_KEYS}, "warnings": []}
    head = [str(x).strip() for x in (values[1] if len(values) > 1 else [])]
    row = int(s["data_row"]) - 1
    cols = {}
    for key, name in s["headers"].items():
        cols[key] = _col(head, name)
        if cols[key] is None:
            out["warnings"].append(f"{s['tab']} 2행에 「{_label(name)}」 머리글이 없습니다")
            continue
        out[key] = _num(_cell(values, row, cols[key]))
    isa_row = next((i for i, r in enumerate(values) if i > 1 and r and str(r[0]).strip() == s["isa_row_name"]), None)
    if isa_row is None:
        out["warnings"].append(f"{s['tab']} A열에 「{s['isa_row_name']}」 행이 없습니다")
    else:
        for key in ISA_KEYS:
            if cols.get(key) is not None:
                out["isa"][key] = _num(_cell(values, isa_row, cols[key]))
    return out


def fetch_isa_close(cfg: dict | None = None, now_kst=None, history_fn=None) -> tuple:
    """ISA 종목(yfinance 133690.KS) 마지막 확정 종가(원) → (날짜 Timestamp, 종가 float). 읽기만.

    한국장 마감(15:30) 전이면 오늘 봉은 버린다. 예외: 네트워크·빈 데이터는 그대로(호출부가 "가격 못 받음"으로)."""
    import pandas as pd

    ticker = settings(cfg)["isa_ticker"]
    if history_fn is None:
        import yfinance as yf

        history_fn = lambda: yf.Ticker(ticker).history(period="1mo", auto_adjust=False)  # noqa: E731
    raw = history_fn()
    close = raw["Close"].dropna() if raw is not None and len(raw) else pd.Series(dtype=float)
    close.index = pd.DatetimeIndex([pd.Timestamp(x).tz_localize(None).normalize() if pd.Timestamp(x).tzinfo is None
                                    else pd.Timestamp(x).tz_convert("Asia/Seoul").tz_localize(None).normalize()
                                    for x in close.index])
    if now_kst is not None:
        now = pd.Timestamp(now_kst)
        today = now.tz_localize(None).normalize() if now.tzinfo is None else now.tz_convert("Asia/Seoul").tz_localize(None).normalize()
        before_close = (now.hour, now.minute) < (15, 30)
        close = close[close.index < today] if before_close else close[close.index <= today]
    close = close[close > 0]
    if close.empty:
        raise ValueError(f"{ticker} 확정 종가 없음")
    return close.index[-1], float(close.iloc[-1])


def read_holdings(cfg: dict | None = None, client=None) -> dict:
    """사용자 시트(GOOGLE_SHEETS_ID) 「QQQM 모으기」을 읽기만 한다. 예외: 인증·네트워크 오류는 그대로(호출부가 잡는다)."""
    from data import sheets

    client = client or sheets.get_client()
    values = client.open_by_key(sheets._sheet_id()).worksheet(settings(cfg)["tab"]).get_all_values()
    return parse_holdings(values, cfg)
