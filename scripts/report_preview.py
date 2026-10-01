"""샘플 데이터로 개인용·공개용 보고서 미리보기를 만든다 (네트워크 없음).

사용법:
    python scripts/report_preview.py                  # outputs/report_preview.html, report_public_preview.html
    python scripts/report_preview.py --screenshot after  # + 폰 폭(375)·PC 폭(1200) 스크린샷

샘플: 오늘의 추천 4건(긴 조건 문구 포함) + 시장 온도 6칸. 실제 계좌·보유 정보는 없다.
공개용 텔레그램 본문(build_public_briefing_text)도 outputs/telegram_public_preview.txt로 남긴다.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import macro_status  # noqa: E402
from notify import briefing, report_html  # noqa: E402

OUT = ROOT / "outputs"


def _macro_row(th, slug, name, code, fcode, value, unit, series, note=None, value_before=None, **extra):
    """engine.daily._build_macro_rows와 같은 모양의 시장 온도 행 하나."""
    row = {
        "slug": slug, "name": name, "code": code, "value": value, "unit": unit, "as_of": "2026-09-30",
        "change_1w": -0.08, "series": series, "is_stale": False, "short_range": False, "note": note,
        "badge": macro_status.classify_macro(fcode, value, th, value_before=value_before), "ref_values": [],
    }
    row.update(extra)
    return row


def _buy_row(ticker, kr, limit, stop, stop_pct, condition, score):
    """engine.daily._build_buy_row + 사이징 결과와 같은 모양의 1차 정찰(b1) 행 하나."""
    return {
        "ticker": ticker, "kr": kr, "stage": "A1", "bucket": "b1", "stage_label": "1차 정찰",
        "key": f"{ticker}-A1", "limit": limit, "stop": stop, "stop_pct": stop_pct, "decision": "매수",
        "note": condition, "condition_summary": condition, "score": score, "grade": "",
        "qty": 3, "amount_krw": 800_000, "max_loss_krw": 40_000, "is_new_position": True,
    }


def sample_summary(cfg: dict) -> dict:
    """미리보기용 summary (engine.daily.build_report_summary 결과와 같은 키)."""
    th = cfg["macro"]["thresholds"]
    macro_rows = [
        _macro_row(th, "fear-greed", "공포·탐욕 지수", "CNN Fear & Greed", "FEAR_GREED", 38, "/100",
                   [55.0, 48.0, 45.0, 50.0, 42.0, 38.0], "지금 구간: 공포"),
        _macro_row(th, "dgs10", "미국 10년물 국채금리", "DGS10", "DGS10", 4.17, "%",
                   [4.0, 4.3, 4.5, 4.2, 4.17], "3개월 변화 +0.30%p (보조 설명)"),
        _macro_row(th, "t10y2y", "장단기 금리차 (10년−2년)", "T10Y2Y", "T10Y2Y", 0.52, "%p",
                   [0.1, 0.3, 0.45, 0.52], short_range=True),
        _macro_row(th, "hy", "하이일드 스프레드", "BAMLH0A0HYM2", "BAMLH0A0HYM2", 3.05, "%", [3.2, 3.1, 3.3, 3.05]),
        _macro_row(th, "fed", "미국 기준금리 (상단)", "DFEDTARU", "DFEDTARU", 4.25, "%", [4.5, 4.5, 4.25],
                   value_before=4.5),
        _macro_row(th, "fx", "원/달러 환율", "DEXKOUS · KRW=X", "DEXKOUS", 1392.0, "원",
                   [1350.0, 1380.0, 1410.0, 1392.0], is_stale=True, **{"as_of": "2026-09-26"}),
    ]
    b1 = [
        _buy_row("NVDA", "엔비디아", 118.42, 109.80, -7.3, "RSI 27.4 → 31.9 · 거래량 부족(0.62배) · 실적 임박 10/9", 31),
        _buy_row("GOOGL", "알파벳 A", 1234.56, 1180.05, -4.4, "RSI 28.9 → 30.4", 27),
        _buy_row("ODFL", "올드 도미니언 프레이트 라인", 9.87, None, None,
                 "RSI 22.1 → 30.3 · 거래량 부족(0.41배) · 실적 임박 10/14 · 스윙 저점 확인 안 됨 — 손절가 미확인", 22),
        _buy_row("CDNS", "케이던스 디자인 시스템즈", 245.10, 236.00, -3.7, "", 18),
    ]
    return {
        "mode": "live", "mode_label": "실전", "as_of": pd.Timestamp("2026-09-30"),
        "buy_groups": {"b1": b1, "b2": [], "b3": [], "b9": []}, "buy_count": len(b1),
        "filtered_rows": [], "sell_rows": [], "warn_rows": [], "data_status_rows": [],
        "unfilled_rows": [], "hold_rows": [], "watch_rows": [], "held_tickers_count": 0,
        "max_concurrent": 8, "macro_rows": macro_rows,
    }


def _screenshot(paths: dict[str, Path], tag: str) -> list[Path]:
    """paths{이름: html} -> outputs/{이름}_{tag}_{폭}.png (폰 375px·PC 1200px, 전체 페이지)."""
    from playwright.sync_api import sync_playwright

    shots = []
    with sync_playwright() as p:
        # 설치된 Playwright 버전과 브라우저 버전이 다를 때를 위해 실행 파일 경로를 환경변수로 받을 수 있다
        exe = os.environ.get("CHROMIUM_PATH")
        browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
        for name, html in paths.items():
            for width in (375, 1200):
                page = browser.new_page(viewport={"width": width, "height": 800}, device_scale_factor=2)
                page.goto(html.resolve().as_uri())
                page.wait_for_timeout(300)
                out = OUT / f"{name}_{tag}_{width}.png"
                page.screenshot(path=str(out), full_page=True)
                shots.append(out)
                page.close()
        browser.close()
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description="샘플 데이터로 보고서 미리보기 만들기")
    parser.add_argument("--screenshot", metavar="TAG", help="스크린샷 파일 이름에 붙일 표시 (예: before, after)")
    args = parser.parse_args()

    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    summary = sample_summary(cfg)
    OUT.mkdir(exist_ok=True)

    public = report_html.render_public_report(summary, cfg, OUT)
    public_preview = public.replace(OUT / "report_public_preview.html")
    private = report_html.render_report(summary, cfg, OUT)
    private_preview = private.replace(OUT / "report_preview.html")
    text = briefing.build_public_briefing_text(summary, cfg)
    (OUT / "telegram_public_preview.txt").write_text(text, encoding="utf-8")

    print(f"공개용 미리보기: {public_preview}")
    print(f"개인용 미리보기: {private_preview}")
    print("── 공개용 텔레그램 본문 ──")
    print(text)
    if args.screenshot:
        for shot in _screenshot({"report_public": public_preview, "report": private_preview}, args.screenshot):
            print(f"스크린샷: {shot}")


if __name__ == "__main__":
    main()
