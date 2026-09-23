# P1 작업 지시문 — 데이터 수집 + 지표 계산

아래 전체를 클로드 코드에 붙여넣는다.

---

CLAUDE.md와 docs/strategy_v3.md를 먼저 읽고, P1 범위만 구현해줘.
P1 목표는 "나스닥 100 일봉을 받아 정리본 9장의 지표를 정확히 계산하고, 트레이딩뷰와 대조할 수 있는 표를 뽑는 것"이야. 신호·상태·텔레그램은 만들지 마.

## 1. 저장소 기본 구성

- CLAUDE.md의 폴더 구조대로 빈 패키지를 만든다 (`__init__.py` 포함).
- `requirements.txt`: pandas, numpy, yfinance, pyyaml, lxml, pyarrow, pytest
- `.gitignore`: CLAUDE.md 보안 항목 반영
- `config.yaml` 초안: 아래 항목을 모두 넣는다.
  - 지표: macd(12, 26, 9), rsi 14, ichimoku(9, 26, 52), ichimoku_shift 26, volume_ma 20, swing_low 10, bollinger(20, 2, 백분위 기간 120)
  - 데이터: history_days 300
  - 정리본 12장 가정값 10개 전부 (값은 문서의 기본값, 주석에 검증 범위)

## 2. data/universe.py

- 위키백과 "Nasdaq-100" 문서의 구성 종목 표를 `pandas.read_html`로 읽어 티커 목록을 만든다.
- 실패하면 `data/universe_fallback.csv`를 쓴다. 이 파일도 이번에 현재 목록으로 만들어 둔다.
- `data/name_kr.csv` (ticker, name_kr)를 만들고, 알고 있는 한글 이름은 채우고 모르는 건 빈칸으로 둔다.
- 티커의 점(.)은 yfinance 형식(-)으로 바꾼다.
- 반환: DataFrame(ticker, name, name_kr)

## 3. data/prices.py

- yfinance로 종목별 일봉을 받는다. `auto_adjust=False`, `Close` 사용 (Adj Close 금지).
- 최소 300거래일. 컬럼은 open, high, low, close, volume 소문자, 날짜 오름차순.
- 미국 동부 시각 기준으로 장이 아직 안 끝난 날의 봉은 제거한다.
- `data/cache/{ticker}.parquet`에 캐시하고, 캐시가 오늘 확정 봉까지 있으면 다시 받지 않는다.
- 실패 종목은 모아서 반환하고 전체 실행은 계속한다.

## 4. core/indicators.py

- `compute_indicators(df, cfg) -> DataFrame` 하나로 정리본 9장의 모든 열을 추가한다:
  macd, signal, hist, gc, dc, macd_norm, rsi, tenkan, kijun, span_a, span_b, cloud_top, cloud_bot, future_yang, chikou_ok, chikou_broken, vol_ratio, swing_low, bb_width_pct
- 정리본 9장 코드와 수식을 그대로 따른다. D는 cfg에서 읽는다.
- RSI에서 loss가 0이면 100으로 처리한다.
- 보조 함수: `last_cross_date(series_bool) -> 날짜 또는 None`

## 5. scripts/p1_report.py

실행하면 `outputs/` 폴더에 CSV 두 개를 만들고 화면에도 표로 출력한다.

(1) `rsi_bottom10.csv` — 나스닥 100 전체 중 최신 확정일 RSI 하위 10개, RSI 오름차순
   티커 | 종목명 | 한글명 | 현재 RSI | 최근 10거래일 RSI 최저값 | 최저값 날짜

(2) `parity_check.csv` — 트레이딩뷰 대조용, 종목: NVDA, AAPL, INTC, TSLA, GOOGL
   티커 | 기준일 | 종가 | RSI | MACD | 시그널 | 최근 골든크로스일 | 최근 데드크로스일 | 전환선 | 기준선 | 구름 상단 | 구름 하단

숫자는 소수 둘째 자리까지.

## 6. tests/

- `test_no_lookahead.py`: 합성 또는 고정 CSV 데이터로, 여러 절단 시점 t에서 "t까지 자른 데이터의 마지막 행" == "전체 데이터의 t행"인지 모든 지표 열에 대해 확인.
- `test_rsi.py`: Wilder RSI를 반복문으로 직접 계산한 참조 구현과 비교 (오차 1e-6).
- `test_macd_cross.py`: 교차가 알려진 합성 시계열에서 gc/dc가 정확한 날짜에만 True인지 확인.
- `test_ichimoku_shift.py`: cloud_top이 D일 전의 max(span_a, span_b)와 같은지 확인.
- 테스트는 네트워크 없이 돌아가야 한다.

## 7. 완료 조건

- `pytest` 전부 통과
- `python scripts/p1_report.py` 실행 성공, CSV 2개 생성
- 수집 실패 종목 수 확인

## 8. 완료 보고 형식

마지막에 아래 형식으로 출력해줘. 이 내용을 채팅에 붙여넣어 검토할 거야.

```
[P1 완료 보고]
테스트: 통과 N / 실패 N
수집: 성공 N종목 / 실패 N종목 (실패 티커: ...)
기준일: YYYY-MM-DD
RSI 하위 10 표: (표 그대로)
대조 표: (표 그대로)
정리본과 다르게 구현했거나 판단이 필요한 부분: (없으면 "없음")
```
