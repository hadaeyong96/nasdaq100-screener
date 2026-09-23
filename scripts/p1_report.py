"""P1 점검 스크립트: 나스닥 100 일봉을 받아 지표를 계산하고,
트레이딩뷰 대조용 표와 RSI 하위 10종목 표를 뽑는다.

실행: python scripts/p1_report.py
출력: outputs/rsi_bottom10.{csv,md}, outputs/parity_check.{csv,md} (화면에도 표 출력)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

# 윈도우 콘솔은 기본 코드페이지(cp949 등)를 쓰는 경우가 많아 em dash(—) 같은
# 일부 유니코드 문자에서 print()가 UnicodeEncodeError로 죽는다. 출력 인코딩을
# UTF-8로 강제해 한국어 메시지가 항상 안전하게 출력되도록 한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.indicators import compute_indicators, last_cross_date  # noqa: E402
from data.prices import fetch_universe_prices  # noqa: E402
from data.universe import get_universe  # noqa: E402

OUTPUT_DIR = ROOT / "outputs"
PARITY_TICKERS = ["NVDA", "AAPL", "INTC", "TSLA", "GOOGL"]


def load_config() -> dict:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_rsi_bottom10(
    indicator_map: dict[str, pd.DataFrame], universe: pd.DataFrame, min_bars: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """나스닥 100 전체 중 최신 확정일 RSI 하위 10개를 뽑는다.

    RSI 계산이 가능한 종목(보유 거래일 >= min_bars)만 대상으로 하고, 그 미만인
    (상장 기간이 짧은) 종목은 별도의 "데이터 부족" 표로 분리한다 (P1.1 3번).

    입력: {ticker: compute_indicators 결과 DataFrame}, universe(ticker, name, name_kr),
         min_bars(RSI 계산에 필요한 최소 보유 거래일)
    출력: (RSI 하위 10 DataFrame, 데이터 부족 종목 DataFrame)
    """
    name_map = universe.set_index("ticker")[["name", "name_kr"]]

    def _name(ticker: str, col: str) -> str:
        return name_map.loc[ticker, col] if ticker in name_map.index else ""

    rows = []
    insufficient_rows = []
    for ticker, out in indicator_map.items():
        bars = int(out["bars"].iloc[-1])
        if bars < min_bars:
            insufficient_rows.append(
                {
                    "티커": ticker,
                    "종목명": _name(ticker, "name"),
                    "한글명": _name(ticker, "name_kr"),
                    "보유 거래일": bars,
                }
            )
            continue
        rsi = out["rsi"].dropna()
        current_rsi = rsi.iloc[-1]
        recent10 = rsi.iloc[-10:]
        min_date = recent10.idxmin()
        rows.append(
            {
                "티커": ticker,
                "종목명": _name(ticker, "name"),
                "한글명": _name(ticker, "name_kr"),
                "현재 RSI": round(current_rsi, 2),
                "최근 10거래일 RSI 최저값": round(recent10.min(), 2),
                "최저값 날짜": min_date.date().isoformat(),
            }
        )

    bottom10 = pd.DataFrame(rows).sort_values("현재 RSI", ascending=True).reset_index(drop=True).head(10)
    insufficient = (
        pd.DataFrame(insufficient_rows).sort_values("보유 거래일").reset_index(drop=True)
        if insufficient_rows
        else pd.DataFrame(columns=["티커", "종목명", "한글명", "보유 거래일"])
    )
    return bottom10, insufficient


def build_parity_check(indicator_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """트레이딩뷰 대조용 표를 만든다 (NVDA, AAPL, INTC, TSLA, GOOGL).

    입력: {ticker: compute_indicators 결과 DataFrame}
    출력: DataFrame(티커, 기준일, 종가, RSI, MACD, 시그널, 최근 골든크로스일,
                    최근 데드크로스일, 전환선, 기준선, 구름 상단, 구름 하단)
    """
    rows = []
    for ticker in PARITY_TICKERS:
        if ticker not in indicator_map:
            continue
        out = indicator_map[ticker]
        last = out.iloc[-1]
        gc_date = last_cross_date(out["gc"])
        dc_date = last_cross_date(out["dc"])
        rows.append(
            {
                "티커": ticker,
                "기준일": last.name.date().isoformat(),
                "종가": round(last["close"], 2),
                "RSI": round(last["rsi"], 2),
                "MACD": round(last["macd"], 2),
                "시그널": round(last["signal"], 2),
                "최근 골든크로스일": gc_date.date().isoformat() if gc_date is not None else "",
                "최근 데드크로스일": dc_date.date().isoformat() if dc_date is not None else "",
                "전환선": round(last["tenkan"], 2),
                "기준선": round(last["kijun"], 2),
                "구름 상단": round(last["cloud_top"], 2) if pd.notna(last["cloud_top"]) else "",
                "구름 하단": round(last["cloud_bot"], 2) if pd.notna(last["cloud_bot"]) else "",
            }
        )
    return pd.DataFrame(rows)


def _format_cell(value) -> str:
    """마크다운 표 셀 값을 소수 둘째 자리까지로 맞춘다."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str) and value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def dataframe_to_markdown(df: pd.DataFrame, header_lines: list[str] | None = None) -> str:
    """DataFrame을 붙여넣기 쉬운 마크다운 표 문자열로 바꾼다.

    입력: df, header_lines(표 위에 붙일 한 줄짜리 설명들, 예: 출처·D값)
    출력: 마크다운 문자열 (숫자는 소수 둘째 자리)
    """
    lines = list(header_lines or [])
    if lines:
        lines.append("")
    if df.empty:
        lines.append("(없음)")
        return "\n".join(lines)

    columns = list(df.columns)
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    cfg = load_config()
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("나스닥 100 구성 종목 목록을 가져오는 중...")
    universe = get_universe()
    universe_source = universe.attrs.get("source", "알 수 없음")

    print("일봉 시세를 받는 중... (캐시가 있으면 재사용)")
    price_result = fetch_universe_prices(universe["ticker"].tolist(), cfg)
    print(f"  성공 {len(price_result.prices)}종목 / 실패 {len(price_result.failed)}종목")
    print(f"  close 보완(chart API meta) 종목: {len(price_result.close_filled)}종목")
    if price_result.close_filled:
        print("  close 보완 티커:", ", ".join(sorted(price_result.close_filled)))
    if price_result.failed:
        print("  실패 티커:", ", ".join(sorted(price_result.failed)))
    if price_result.warnings:
        print(f"  경고 {len(price_result.warnings)}종목:")
        for ticker, msgs in price_result.warnings.items():
            for msg in msgs:
                print(f"    - {ticker}: {msg}")

    print("지표를 계산하는 중...")
    indicator_map = {
        ticker: compute_indicators(df, cfg) for ticker, df in price_result.prices.items()
    }

    as_of_dates = {out.index[-1] for out in indicator_map.values()}
    as_of = max(as_of_dates) if as_of_dates else None
    as_of_str = as_of.date().isoformat() if as_of is not None else "알 수 없음"

    rsi_min_bars = cfg["indicators"]["rsi"]["period"] + 1  # RSI diff() 1일 + period(14) = 15일
    rsi_bottom10, insufficient_history = build_rsi_bottom10(indicator_map, universe, rsi_min_bars)
    parity_check = build_parity_check(indicator_map)

    ichimoku_shift = cfg["indicators"]["ichimoku_shift"]
    common_header = [f"기준일 종가 데이터 출처(yfinance): {as_of_str} 확정 기준", f"일목 이동 칸수 D: {ichimoku_shift}"]

    rsi_bottom10.to_csv(OUTPUT_DIR / "rsi_bottom10.csv", index=False, encoding="utf-8-sig")
    parity_check.to_csv(OUTPUT_DIR / "parity_check.csv", index=False, encoding="utf-8-sig")

    rsi_md = dataframe_to_markdown(rsi_bottom10, common_header)
    if not insufficient_history.empty:
        rsi_md += "\n\n### 데이터 부족 종목 (RSI 계산에 필요한 " + str(rsi_min_bars) + "거래일 미만)\n\n"
        rsi_md += dataframe_to_markdown(insufficient_history)
    (OUTPUT_DIR / "rsi_bottom10.md").write_text(rsi_md, encoding="utf-8")

    parity_md = dataframe_to_markdown(parity_check, common_header)
    (OUTPUT_DIR / "parity_check.md").write_text(parity_md, encoding="utf-8")

    print(f"\n구성 종목 출처: {universe_source} ({len(universe)}종목)")
    print(f"기준일: {as_of_str} / close 보완 종목 수: {len(price_result.close_filled)}")
    print("\n[RSI 하위 10]")
    print(rsi_bottom10.to_string(index=False))
    if not insufficient_history.empty:
        print(f"\n[데이터 부족 종목] (RSI 계산에 필요한 {rsi_min_bars}거래일 미만)")
        print(insufficient_history.to_string(index=False))
    print("\n[트레이딩뷰 대조 표]")
    print(parity_check.to_string(index=False))

    print(
        f"\n저장 완료: {OUTPUT_DIR / 'rsi_bottom10.csv'}, {OUTPUT_DIR / 'rsi_bottom10.md'}, "
        f"{OUTPUT_DIR / 'parity_check.csv'}, {OUTPUT_DIR / 'parity_check.md'}"
    )


if __name__ == "__main__":
    main()
