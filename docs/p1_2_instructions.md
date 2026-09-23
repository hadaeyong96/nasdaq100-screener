# P1.2 지시문 — 최신 종가 누락 원인 진단과 보완

---

집 네트워크에서 p1_report.py를 실행했는데, 학교에서 실행하고 몇 시간이 지난 뒤에도 101종목 전부 2026-09-22 봉의 종가가 NaN이라 기준일이 여전히 2026-09-21이야. 전 종목이 이렇게 오래 비어 있는 건 야후 반영 지연만으로 보기 어려워. 원인을 진단하고 보완해줘. 신호·상태·텔레그램은 여전히 만들지 마.

## 1. 진단 (코드 수정 전에 먼저, 결과를 표로 보여줘)

NVDA, AAPL 두 종목에 대해 캐시 없이 아래를 각각 실행하고 마지막 3행(날짜, open, high, low, close, volume)을 비교해줘.

- (a) 현재 코드 경로 그대로
- (b) `yf.Ticker(t).history(period="10d", interval="1d", auto_adjust=False)`
- (c) `yf.download(t, period="10d", interval="1d", auto_adjust=False)`
- (d) (b)에 `repair=True` 추가
- (e) 야후 chart API를 requests로 직접 호출한 원본 JSON의 마지막 3개 값과 `meta.regularMarketPrice`, `meta.regularMarketTime`
- 설치된 yfinance 버전, 그리고 최신 버전으로 올렸을 때(`pip install -U yfinance`) (b) 결과

참고값: 구글 파이낸스 기준 2026-09-22 종가는 NVDA $228.87, AAPL $339.75.

## 2. 보완 (진단 결과에 따라)

- yfinance 업그레이드로 해결되면 requirements.txt에 최소 버전을 적어줘.
- 업그레이드로도 안 되면, 마지막 봉의 close만 비어 있고 open/high/low가 있으며 chart API의 `meta.regularMarketTime`이 그날 정규장 마감(16:00 ET) 이후라면, `meta.regularMarketPrice`로 close를 채우는 보완 로직을 `data/prices.py`에 추가해줘.
  - 채운 값은 같은 날 high와 low 사이여야 한다. 범위를 벗어나면 채우지 말고 그 봉을 버린다.
  - 채운 봉에는 `close_source="meta"` 표시를 남기고, 실행 로그에 채운 종목 수를 출력한다.
- 어떤 방법으로도 안 되면 수정하지 말고 원인과 대안(다른 데이터 소스 후보)을 보고해줘.

## 3. 작은 정리

- Invesco 출처가 406 오류로 실패했어. 브라우저와 같은 User-Agent·Accept 헤더를 넣어 한 번 더 시도하게 해줘. 그래도 실패하면 지금처럼 다음 출처로 넘어가면 돼.
- nasdaq_official 출처의 종목명에 붙는 " Common Stock", " Ordinary Shares", " Class A Common Stock" 같은 꼬리를 떼서 짧게 표시해줘.

## 4. 테스트와 실행

- close 보완 로직 테스트 추가 (범위 안이면 채움, 범위 밖이면 버림, open/high/low도 NaN이면 버림). 네트워크 없이.
- `pytest` 전부 통과, `python scripts/p1_report.py` 실행해서 기준일 확인.

## 5. 완료 보고 형식

```
[P1.2 완료 보고]
진단 표: (a)~(e) 비교
원인: ...
조치: ...
테스트: 통과 N / 실패 N
기준일: YYYY-MM-DD / close 보완 종목 수: N
구성 종목 출처: ...
판단이 필요한 부분: (없으면 "없음")
```

마지막으로 변경 사항을 "P1.2 fix latest close" 메시지로 커밋해줘. 푸시는 하지 마 (내가 직접 할게).
