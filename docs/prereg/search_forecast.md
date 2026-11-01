# 검색량 실적 예고 — 규칙 사전 등록 (앞으로 기록)

> 등록 2026-11-01. 사용자 지시문 1~5장 원문을 그대로 옮긴 것이다(빠진 부분은 지어내지 않았다).
> 과거 백테스트가 아니다. "지금부터 수집 → 예측을 먼저 잠금 → 실적 뒤 채점"하는 앞으로 기록 방식이며 과거 수익률은 계산하지 않는다.
> 이 문서를 코드보다 먼저 커밋하고 태그 **`search-forecast-prereg`**를 push한다. 시작 뒤 대상·검색어는 바꾸지 않는다.

## 1. 대상과 검색어 (시작 후 변경 금지, 지역 US)
현재 나스닥100 구성 종목만. 구성에서 빠진 종목은 제외하고 보고해.
PEP: Doritos, Cheetos, Lay's, Gatorade, Pepsi
MDLZ: Oreo, Ritz crackers, Cadbury, Sour Patch Kids, Toblerone
KDP: Dr Pepper, Keurig, Canada Dry, Snapple
MNST: Monster Energy, Reign Energy
COST: Costco, Kirkland Signature
SBUX: Starbucks
NFLX: Netflix
AAPL: iPhone, AirPods, Apple Watch, MacBook
ABNB: Airbnb
BKNG: Booking.com, Priceline, Kayak
DASH: DoorDash
ROST: Ross Dress for Less
MAR: Marriott Bonvoy
ORLY: O'Reilly Auto Parts
TSLA: Tesla Model Y, Tesla Model 3
CMCSA: Peacock, Xfinity
PYPL: PayPal, Venmo
CHTR: Spectrum Internet
config/search_forecast.yaml 에 저장.

## 2. 수집 (무료)
- 구글 트렌드: 검색어별 최근 5년 주간 값(US).
  라이브러리는 pytrends·trendspy 등 후보의 유지보수 상태를 먼저 확인하고 하나를 골라 버전 고정해.
  고른 이유와 확인한 내용을 보고에 적어. 검증 안 된 코드는 실행하지 마.
  차단(429) 대비: 요청 사이 대기, 재시도 3회, 실패 검색어는 기록만 하고 다음 주에 다시.
  한 요청에 검색어 1개씩만 넣어(서로 다른 요청의 0~100 값은 비교하지 않음. YoY는 같은 검색어 안에서만 계산).
- 위키백과 조회수(예비): Wikimedia REST pageviews API, 회사·대표 제품 문서, 월간.
  User-Agent는 저장소 이름과 연락 문구만 (개인 이메일 넣지 마).
- 저장: virtual_books.db 에 search_trends, wiki_views 테이블 (수집 시각·원값 그대로).
- 실행: GitHub Actions 주 1회(일요일 KST 오전, 기존 daily-screener 와 다른 concurrency 그룹).
  실패해도 아침 편지 실행에는 영향 없게 분리.

## 3. 점수와 예측 잠금
- 회사 점수 = (회계 분기 평균 검색량의 전년 동기 대비 변화율, 검색어 평균)
             − (그 회사 직전 8개 분기 YoY 중앙값)
- 회계 분기 끝 날짜는 EDGAR 과거 공시로 회사별 추정. 분기 끝 + 10일에 점수 계산.
- 같은 시즌(달력 분기) 안에서 점수 상위 1/3 = "매출 가속", 하위 1/3 = "매출 감속", 가운데 = 예측 없음.
- 잠금: 예측 행(회사·분기·점수·예측·계산 시각)을 DB search_predictions 에 저장하고,
  그 행들의 SHA-256 해시와 시각만 docs/forecast_locks/YYYY-Qn.txt 로 커밋·push.
  예측 내용(종목·방향)은 공개 저장소에 올리지 마.
- 잠금 시각보다 실적 발표가 먼저였으면 그 예측은 무효로 표시.

## 4. 채점 (무료)
- 매출 가속 정답 = 이번 분기 매출 YoY 성장률 > 직전 분기 YoY 성장률 (EDGAR XBRL 매출)
- 주가 반응: 실적 발표일(첫 8-K 2.02 또는 10-Q/10-K 공시일) 기준
  −1~+1일, 1개월, 6개월 QQQ 대비 초과수익 (기간이 차면 자동 채점)
- DB search_scores 에 저장

## 5. 판정 (4개 분기 채점이 모인 뒤, 2027-11 무렵. 지금은 계산 안 함)
1. 상위 1/3 매출 가속 비율 − 하위 1/3 ≥ 15%p
2. 상위 1/3 1개월 초과수익 평균 > 0 이고 > 하위 1/3
3. 1번 차이가 4개 분기 중 3개 이상 같은 방향
셋 다 충족 → PASS (이후 6개월 초과수익 확인), 아니면 FAIL. trial_ledger 에 "앞으로 기록 1건" 등록.

---

구현 해석(원문에 숫자로 없는 부분)은 첫 잠금 전에 `docs/prereg/search_forecast_impl.md`로 따로 커밋한다.
