"""해자 분석 H3: 현재 나스닥100 전체의 M1~M5·등급표를 만든다
(docs/design/moat_plan.md 5장 H3, 사용자 지시 2026-10-01).

data/edgar.py로 SEC EDGAR에서 재무를 받고 core/moat.py로 지표·등급을 계산한다. 결과는
outputs/moat/에 CSV와 HTML로 저장한다(결과 파일은 커밋 안 함 — .gitignore의 outputs/).

실행(.env에 SEC_USER_AGENT 필요):
    python -u -m scripts.moat_report
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

from core import moat  # noqa: E402
from data import edgar  # noqa: E402
from data.universe import get_universe  # noqa: E402

OUT_DIR = ROOT / "outputs" / "moat"
CSV_PATH = OUT_DIR / "moat_grades.csv"
HTML_PATH = OUT_DIR / "moat_report.html"
LOG_PATH = OUT_DIR / "report.log"
SECTOR_CACHE_PATH = ROOT / "data" / "cache" / "moat_sectors.json"

_STATUS_ORDER = ["넓음", "좁음", "없음", "판단 불가"]

UNKNOWN_SECTOR = "확인 필요"


def log(msg: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def resolve_cik(ticker: str, ticker_to_cik: dict[str, int]) -> int | None:
    """티커 표기 차이(대소문자, "." <-> "-")를 흡수해 CIK를 찾는다 (순수 함수).

    입력: ticker(우리 쪽 표기, 보통 yfinance 형식), ticker_to_cik(edgar.fetch_ticker_to_cik 결과)
    출력: CIK 또는 못 찾으면 None
    """
    for candidate in (ticker, ticker.upper(), ticker.upper().replace("-", "."), ticker.upper().replace(".", "-")):
        if candidate in ticker_to_cik:
            return ticker_to_cik[candidate]
    return None


def summarize_grades(rows: list[dict]) -> dict[str, int]:
    """등급별 종목 수를 센다 (순수 함수). 집계용 — 같은 회사 중복 제거는 호출부가 먼저 한다."""
    counts = {g: 0 for g in _STATUS_ORDER}
    for row in rows:
        counts[row["grade"]] = counts.get(row["grade"], 0) + 1
    return counts


def dedupe_rows_by_cik(rows: list[dict]) -> list[dict]:
    """같은 회사의 복수 주식(예: GOOGL·GOOG는 둘 다 Alphabet, CIK 1652044)을 등급 집계에서
    한 번만 세도록 중복을 없앤다 (순수 함수, 사용자 지시 2026-10-01).

    CIK가 같은 행 중 먼저 나온 것만 남긴다(나스닥100 목록에 나온 순서). CIK를 못 찾은 행
    (cik=None)은 서로 다른 회사로 보고 전부 남긴다.
    """
    seen_ciks: set = set()
    out = []
    for row in rows:
        cik = row.get("cik")
        if cik is not None:
            if cik in seen_ciks:
                continue
            seen_ciks.add(cik)
        out.append(row)
    return out


def fetch_sector_from_yfinance(ticker: str) -> str:
    """yfinance에서 섹터 분류를 받는다 (네트워크, 사용자 지시 2026-10-01).

    GICS 자체는 MSCI·S&P의 유료 분류라 이 프로젝트가 무료로 못 쓴다. 대신 yfinance의
    Morningstar 기반 섹터 분류("Technology", "Financial Services", "Consumer Defensive" 등)를
    근사치로 쓴다 — GICS 섹터(예: "Information Technology", "Financials", "Consumer
    Staples")와 이름이 다를 수 있다는 점을 보고서에 밝혀 둔다.
    """
    import yfinance as yf

    try:
        info = yf.Ticker(ticker).info
        return info.get("sector") or UNKNOWN_SECTOR
    except Exception:
        return UNKNOWN_SECTOR


def load_sector_cache() -> dict:
    if SECTOR_CACHE_PATH.exists():
        return json.loads(SECTOR_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_sector_cache(cache: dict) -> None:
    SECTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SECTOR_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def get_sector(ticker: str, cache: dict, fetch_fn=fetch_sector_from_yfinance) -> str:
    """캐시 우선으로 섹터를 구한다 — 캐시에 없으면 fetch_fn으로 받아 캐시에 채운다(호출부가
    저장). fetch_fn을 주입할 수 있어 테스트는 네트워크 없이 돈다."""
    if ticker in cache and cache[ticker] != UNKNOWN_SECTOR:
        return cache[ticker]
    sector = fetch_fn(ticker)
    cache[ticker] = sector
    return sector


def sector_breakdown(rows: list[dict]) -> dict[str, int]:
    """종목 목록(보통 넓음 등급)의 업종별 개수를 센다 (순수 함수)."""
    counts: dict[str, int] = {}
    for r in rows:
        sector = r.get("sector") or UNKNOWN_SECTOR
        counts[sector] = counts.get(sector, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def m4_reference_grade(profile: "moat.MoatProfile") -> str | None:
    """M4를 참고 비율(설비투자 안 뺀 영업현금흐름÷순이익, reference_good_min_ratio=0.9)
    기준으로 바꿨다면 나왔을 전체 해자 등급 (순수 함수, 사용자 지시 2026-10-01 5번).

    실제 등급 판정에는 안 쓴다 — 비교용이다. M4 참고값 자체가 없으면(데이터 부족) None.
    """
    if profile.m4.reference_status is None:
        return None
    m4_alt = moat.IndicatorResult(status=profile.m4.reference_status)
    return moat.compute_moat_grade(profile.m1, profile.m2, profile.m3, m4_alt, profile.m5)


def _indicator_columns(name: str, ind: "moat.IndicatorResult") -> dict:
    return {
        f"{name}_status": ind.status,
        f"{name}_detail": ind.detail,
        f"{name}_tags": ", ".join(f"{k}={v}" for k, v in ind.tags_used.items() if v),
    }


def profile_to_row(ticker: str, name: str, profile: "moat.MoatProfile", cik: int | None = None, sector: str | None = None) -> dict:
    """MoatProfile을 CSV/HTML 한 줄짜리 dict로 편다 (순수 함수)."""
    alt_grade = m4_reference_grade(profile)
    row = {
        "ticker": ticker, "company_name": name, "cik": cik, "sector": sector or UNKNOWN_SECTOR, "grade": profile.grade,
        "data_years": len(profile.data_years),
        "latest_data_year": profile.data_years[-1] if profile.data_years else None,
        "insufficient_data_reason": profile.insufficient_data_reason or "",
        "outliers": "; ".join(profile.outliers),
        "revenue_tag_mismatch_notes": "; ".join(profile.m2.notes),  # M1·M2·M3 모두 같은 revenue_series를 쓰므로 동일함
        "m4_reference_status": profile.m4.reference_status or "",
        "m4_alternate_grade": alt_grade or "",
        "m4_grade_would_change": bool(alt_grade and alt_grade != profile.grade),
    }
    for label, ind in (("m1", profile.m1), ("m2", profile.m2), ("m3", profile.m3), ("m4", profile.m4), ("m5", profile.m5)):
        row.update(_indicator_columns(label, ind))
    return row


def render_html(rows: list[dict], generated_at: str) -> str:
    """등급표 HTML을 만든다 (순수 함수 — 파일 I/O 없음).

    표시 내용(사용자 지시): 등급별 종목 수, 판단 불가 목록, 이상치 목록, 전체 표.
    등급 집계(종목 수·넓음·판단불가 목록)는 같은 회사 중복(GOOGL·GOOG 등)을 제거한
    종목 기준으로 낸다 — "전체 표"는 티커 단위로 전부 보여준다(사용자 지시 2026-10-01).
    """
    deduped = dedupe_rows_by_cik(rows)
    counts = summarize_grades(deduped)
    wide = [r for r in deduped if r["grade"] == "넓음"]
    inconclusive = [r for r in deduped if r["grade"] == "판단 불가"]
    with_outliers = [r for r in rows if r["outliers"]]

    def esc(s) -> str:
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>나스닥100 해자 등급표</title>",
        "<style>body{font-family:sans-serif;margin:24px;color:#222}"
        "table{border-collapse:collapse;width:100%;font-size:13px}"
        "th,td{border:1px solid #ccc;padding:4px 8px;text-align:left}"
        "th{background:#f0f0f0}"
        ".wide{background:#e6f4ea}.none{background:#fbe9e7}.unknown{background:#f5f5f5}.narrow{background:#fff8e1}"
        "</style></head><body>",
        f"<h1>나스닥100 해자 등급표</h1><p>생성: {esc(generated_at)} · 티커 {len(rows)}개"
        f" · 등급 집계는 같은 회사 중복 제거 후 {len(deduped)}개 기준(사용자 지시 2026-10-01)</p>",
        "<h2>등급별 종목 수 (회사 기준, 중복 제거)</h2><ul>",
    ]
    for g in _STATUS_ORDER:
        parts.append(f"<li>{esc(g)}: {counts.get(g, 0)}개</li>")
    parts.append("</ul>")

    parts.append(f"<h2>넓음 ({len(wide)}개)</h2><p>" + (", ".join(f"{esc(r['ticker'])}({esc(r.get('sector', UNKNOWN_SECTOR))})" for r in wide) or "없음") + "</p>")

    sectors = sector_breakdown(wide)
    parts.append("<h3>넓음 종목의 업종별 개수 (yfinance 섹터 분류 — GICS 근사치)</h3><ul>")
    for sector, n in sectors.items():
        parts.append(f"<li>{esc(sector)}: {n}개</li>")
    parts.append("</ul>")

    parts.append(f"<h2>판단 불가 ({len(inconclusive)}개)</h2><ul>")
    for r in inconclusive:
        parts.append(f"<li>{esc(r['ticker'])} ({esc(r['company_name'])}) — {esc(r['insufficient_data_reason'])}</li>")
    parts.append("</ul>")

    parts.append(f"<h2>이상치 ({len(with_outliers)}개 종목 — 최근 5년 자기자본 음수만. ROIC 100% 초과는 더 이상 이상치로 안 봄)</h2><ul>")
    for r in with_outliers:
        parts.append(f"<li>{esc(r['ticker'])} — {esc(r['outliers'])}</li>")
    parts.append("</ul>")

    changed = [r for r in deduped if r.get("m4_grade_would_change")]
    parts.append(f"<h2>M4를 참고 비율(0.9 기준)로 바꿨다면 등급이 달라질 종목 ({len(changed)}개, 비교용 — 실제 등급엔 안 씀)</h2><ul>")
    for r in changed:
        parts.append(f"<li>{esc(r['ticker'])}: {esc(r['grade'])} → {esc(r['m4_alternate_grade'])} (M4 {esc(r['m4_status'])}→{esc(r['m4_reference_status'])})</li>")
    parts.append("</ul>")

    mismatches = [r for r in rows if r.get("revenue_tag_mismatch_notes")]
    if mismatches:
        parts.append(f"<h2>매출 태그 불일치 ({len(mismatches)}개 종목)</h2><ul>")
        for r in mismatches:
            parts.append(f"<li>{esc(r['ticker'])} — {esc(r['revenue_tag_mismatch_notes'])}</li>")
        parts.append("</ul>")

    parts.append("<h2>전체 표</h2><table><tr>")
    cols = ["ticker", "company_name", "sector", "grade", "m1_status", "m2_status", "m3_status", "m4_status", "m5_status", "data_years", "latest_data_year"]
    for c in cols:
        parts.append(f"<th>{esc(c)}</th>")
    parts.append("</tr>")
    _row_class = {"넓음": "wide", "좁음": "narrow", "없음": "none", "판단 불가": "unknown"}
    for r in sorted(rows, key=lambda x: (_STATUS_ORDER.index(x["grade"]), x["ticker"])):
        cls = _row_class.get(r["grade"], "")
        parts.append(f"<tr class='{cls}'>")
        for c in cols:
            parts.append(f"<td>{esc(r.get(c, ''))}</td>")
        parts.append("</tr>")
    parts.append("</table></body></html>")
    return "\n".join(parts)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    user_agent = edgar.get_user_agent()  # 없으면 여기서 EdgarConfigError로 멈춘다
    as_of = date.today()

    universe = get_universe()
    log(f"나스닥100 구성종목 {len(universe)}개 (출처: {universe.attrs.get('source', '?')})")

    ticker_to_cik = edgar.fetch_ticker_to_cik(user_agent=user_agent)
    log(f"SEC 티커->CIK 맵 {len(ticker_to_cik)}개 로드")

    sector_cache = load_sector_cache()

    rows: list[dict] = []
    for _, r in universe.iterrows():
        ticker, name = r["ticker"], r["name"]
        sector = get_sector(ticker, sector_cache)
        cik = resolve_cik(ticker, ticker_to_cik)
        if cik is None:
            log(f"  {ticker}: SEC CIK를 못 찾음 -> 판단 불가")
            profile = moat.MoatProfile(
                ticker=ticker, grade="판단 불가",
                m1=moat.IndicatorResult(status="판단 불가"), m2=moat.IndicatorResult(status="판단 불가"),
                m3=moat.IndicatorResult(status="판단 불가"), m4=moat.IndicatorResult(status="판단 불가"),
                m5=moat.IndicatorResult(status="판단 불가"),
                insufficient_data_reason="SEC EDGAR 티커 목록에서 CIK를 못 찾음",
            )
            rows.append(profile_to_row(ticker, name, profile, sector=sector))
            continue
        try:
            facts = edgar.fetch_company_facts(cik, user_agent=user_agent)
        except edgar.EdgarFetchError as exc:
            log(f"  {ticker}: companyfacts 요청 실패 -> 판단 불가 ({exc})")
            profile = moat.MoatProfile(
                ticker=ticker, grade="판단 불가",
                m1=moat.IndicatorResult(status="판단 불가"), m2=moat.IndicatorResult(status="판단 불가"),
                m3=moat.IndicatorResult(status="판단 불가"), m4=moat.IndicatorResult(status="판단 불가"),
                m5=moat.IndicatorResult(status="판단 불가"),
                insufficient_data_reason=f"companyfacts 요청 실패: {exc}",
            )
            rows.append(profile_to_row(ticker, name, profile, cik=cik, sector=sector))
            continue

        profile = moat.analyze_company(facts, ticker, cfg, as_of)
        log(f"  {ticker}: {profile.grade} (M1={profile.m1.status} M2={profile.m2.status} M3={profile.m3.status} M4={profile.m4.status} M5={profile.m5.status})")
        rows.append(profile_to_row(ticker, name, profile, cik=cik, sector=sector))

    save_sector_cache(sector_cache)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log(f"CSV 저장: {CSV_PATH}")

    generated_at = datetime.now().isoformat(timespec="seconds")
    HTML_PATH.write_text(render_html(rows, generated_at), encoding="utf-8")
    log(f"HTML 저장: {HTML_PATH}")

    deduped = dedupe_rows_by_cik(rows)
    counts = summarize_grades(deduped)
    log(f"완료 — 티커 {len(rows)}개, 회사 기준(중복 제거) 등급별 종목 수: {counts}")

    wide = [r for r in deduped if r["grade"] == "넓음"]
    log(f"넓음 종목의 업종별 개수: {sector_breakdown(wide)}")

    changed = [r for r in deduped if r.get("m4_grade_would_change")]
    log(f"M4를 참고 비율(0.9 기준)로 바꿨다면 등급이 달라질 종목 {len(changed)}개: " + ", ".join(f"{r['ticker']}({r['grade']}->{r['m4_alternate_grade']})" for r in changed))


if __name__ == "__main__":
    main()
