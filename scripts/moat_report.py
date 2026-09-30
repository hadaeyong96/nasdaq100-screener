"""해자 분석 H3: 현재 나스닥100 전체의 M1~M5·등급표를 만든다
(docs/design/moat_plan.md 5장 H3, 사용자 지시 2026-10-01).

data/edgar.py로 SEC EDGAR에서 재무를 받고 core/moat.py로 지표·등급을 계산한다. 결과는
outputs/moat/에 CSV와 HTML로 저장한다(결과 파일은 커밋 안 함 — .gitignore의 outputs/).

실행(.env에 SEC_USER_AGENT 필요):
    python -u -m scripts.moat_report
"""

from __future__ import annotations

import csv
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

_STATUS_ORDER = ["넓음", "좁음", "없음", "판단 불가"]


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
    """등급별 종목 수를 센다 (순수 함수)."""
    counts = {g: 0 for g in _STATUS_ORDER}
    for row in rows:
        counts[row["grade"]] = counts.get(row["grade"], 0) + 1
    return counts


def _indicator_columns(name: str, ind: "moat.IndicatorResult") -> dict:
    return {
        f"{name}_status": ind.status,
        f"{name}_detail": ind.detail,
        f"{name}_tags": ", ".join(f"{k}={v}" for k, v in ind.tags_used.items() if v),
    }


def profile_to_row(ticker: str, name: str, profile: "moat.MoatProfile") -> dict:
    """MoatProfile을 CSV/HTML 한 줄짜리 dict로 편다 (순수 함수)."""
    row = {
        "ticker": ticker, "company_name": name, "grade": profile.grade,
        "data_years": len(profile.data_years),
        "latest_data_year": profile.data_years[-1] if profile.data_years else None,
        "insufficient_data_reason": profile.insufficient_data_reason or "",
        "outliers": "; ".join(profile.outliers),
    }
    for label, ind in (("m1", profile.m1), ("m2", profile.m2), ("m3", profile.m3), ("m4", profile.m4), ("m5", profile.m5)):
        row.update(_indicator_columns(label, ind))
    return row


def render_html(rows: list[dict], generated_at: str) -> str:
    """등급표 HTML을 만든다 (순수 함수 — 파일 I/O 없음).

    표시 내용(사용자 지시): 등급별 종목 수, 판단 불가 목록, 이상치 목록, 전체 표.
    """
    counts = summarize_grades(rows)
    wide = [r for r in rows if r["grade"] == "넓음"]
    inconclusive = [r for r in rows if r["grade"] == "판단 불가"]
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
        f"<h1>나스닥100 해자 등급표</h1><p>생성: {esc(generated_at)} · 총 {len(rows)}개 종목</p>",
        "<h2>등급별 종목 수</h2><ul>",
    ]
    for g in _STATUS_ORDER:
        parts.append(f"<li>{esc(g)}: {counts.get(g, 0)}개</li>")
    parts.append("</ul>")

    parts.append(f"<h2>넓음 ({len(wide)}개)</h2><p>" + (", ".join(esc(r['ticker']) for r in wide) or "없음") + "</p>")

    parts.append(f"<h2>판단 불가 ({len(inconclusive)}개)</h2><ul>")
    for r in inconclusive:
        parts.append(f"<li>{esc(r['ticker'])} ({esc(r['company_name'])}) — {esc(r['insufficient_data_reason'])}</li>")
    parts.append("</ul>")

    parts.append(f"<h2>이상치 ({len(with_outliers)}개 종목)</h2><ul>")
    for r in with_outliers:
        parts.append(f"<li>{esc(r['ticker'])} — {esc(r['outliers'])}</li>")
    parts.append("</ul>")

    parts.append("<h2>전체 표</h2><table><tr>")
    cols = ["ticker", "company_name", "grade", "m1_status", "m2_status", "m3_status", "m4_status", "m5_status", "data_years", "latest_data_year"]
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

    rows: list[dict] = []
    for _, r in universe.iterrows():
        ticker, name = r["ticker"], r["name"]
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
            rows.append(profile_to_row(ticker, name, profile))
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
            rows.append(profile_to_row(ticker, name, profile))
            continue

        profile = moat.analyze_company(facts, ticker, cfg, as_of)
        log(f"  {ticker}: {profile.grade} (M1={profile.m1.status} M2={profile.m2.status} M3={profile.m3.status} M4={profile.m4.status} M5={profile.m5.status})")
        rows.append(profile_to_row(ticker, name, profile))

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

    counts = summarize_grades(rows)
    log(f"완료 — 등급별 종목 수: {counts}")


if __name__ == "__main__":
    main()
