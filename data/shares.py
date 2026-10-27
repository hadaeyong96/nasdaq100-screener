"""발행 주식 수 받기 (2026-10-27, 웅덩이 v3.2·반등 확인 v2 시가총액 순위 — docs/prereg/puddle_v32_forward.md 2-2-3).

무료 자료만: 먼저 yfinance `Ticker(t).info["sharesOutstanding"]`(종류별 주식 수 — GOOG·GOOGL 따로),
없거나 0 이하면 EDGAR companyfacts `dei:EntityCommonStockSharesOutstanding`의 가장 최근 보고 값 합계(회사 전체 보통주).
둘 다 실패하면 None(그 종목은 그달 순위에서 빠지고 「점검」에 표시). 실행(프로세스) 안에서 종목당 한 번만 받는다.
실패는 조용히 넘기지 않는다 — fetch_shares가 실패 티커를 돌려주고 장부가 점검 줄로 보고한다.
"""

from __future__ import annotations

import math

_MEMO: dict[str, tuple[float | None, str | None]] = {}


def edgar_total_shares(company_facts: dict) -> float | None:
    """companyfacts → 가장 최근 보고일(end)의 dei:EntityCommonStockSharesOutstanding 합(그날 값이 여러 접수에 있으면 가장 늦은 접수 하나).

    여러 종류 주식(Class A·B·C)은 같은 접수 안 여러 값 → 합 = 회사 전체 보통주. 없으면 None. (순수 함수)"""
    node = (company_facts.get("facts", {}).get("dei", {}) or {}).get("EntityCommonStockSharesOutstanding")
    if not node:
        return None
    entries = [e for es in (node.get("units") or {}).values() for e in es if e.get("val") is not None and e.get("end")]
    if not entries:
        return None
    end = max(e["end"] for e in entries)
    at_end = [e for e in entries if e["end"] == end]
    filed = max(str(e.get("filed") or "") for e in at_end)
    accns = {e.get("accn") for e in at_end if str(e.get("filed") or "") == filed}
    accn = sorted(a for a in accns if a)[-1] if any(accns) else None
    total = sum(float(e["val"]) for e in at_end if str(e.get("filed") or "") == filed and (accn is None or e.get("accn") == accn))
    return total if total > 0 else None


def _yf_shares(ticker: str) -> float | None:
    import yfinance as yf

    v = (yf.Ticker(ticker).info or {}).get("sharesOutstanding")
    return float(v) if v is not None else None


def _edgar_shares(ticker: str) -> float | None:
    from data import edgar

    cik_map = edgar.fetch_ticker_to_cik()
    cik = cik_map.get(ticker.upper().replace("-", "."), cik_map.get(ticker.upper()))
    if cik is None:
        return None
    return edgar_total_shares(edgar.fetch_company_facts(cik, force_refresh=True))


def _ok(v) -> bool:
    return v is not None and math.isfinite(v) and v > 0


def fetch_shares(tickers: list[str], *, yf_fn=None, edgar_fn=None, memo: bool = True) -> tuple[dict[str, float | None], set[str], list[str]]:
    """티커들 → (주식 수 {티커: 값 또는 None}, EDGAR 값(회사 전체)을 쓴 티커 집합, 둘 다 실패한 티커 목록).

    yf_fn·edgar_fn(티커 → 주식 수 또는 None, 예외 가능)은 테스트 주입용. memo=True면 같은 실행 안 두 번째부터 다시 받지 않는다."""
    yf_fn = yf_fn or _yf_shares
    edgar_fn = edgar_fn or _edgar_shares
    out: dict[str, float | None] = {}
    whole: set[str] = set()
    failed: list[str] = []
    for t in dict.fromkeys(tickers):
        if memo and t in _MEMO:
            v, src = _MEMO[t]
        else:
            v, src = None, None
            try:
                v = yf_fn(t)
                src = "yfinance" if _ok(v) else None
            except Exception as exc:  # noqa: BLE001 — EDGAR로 넘어간다
                print(f"[shares] yfinance 주식 수 실패 {t}: {type(exc).__name__}")
            if not _ok(v):
                try:
                    v = edgar_fn(t)
                    src = "edgar" if _ok(v) else None
                except Exception as exc:  # noqa: BLE001 — 둘 다 실패면 None
                    print(f"[shares] EDGAR 주식 수 실패 {t}: {type(exc).__name__}")
                    v = None
            if not _ok(v):
                v, src = None, None
            if memo:
                _MEMO[t] = (v, src)
        out[t] = v
        if src == "edgar":
            whole.add(t)
        if v is None:
            failed.append(t)
    if failed:
        print(f"[shares] 주식 수를 못 받은 종목 {len(failed)}개: {', '.join(failed)}")
    return out, whole, failed
