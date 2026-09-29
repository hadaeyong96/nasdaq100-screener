# 라이브 모드를 "보유 종목 관리 조언"으로 전환 — 설계안 (v1+v2 최종)

> 2026-09-29 논의 결과를 합친 최종 설계안. 구현 전 문서. 구현은 1단계부터 작은
> 단위로 나눠 진행하고, 단계마다 pytest를 돌린다(CLAUDE.md 단계 작업 원칙).

## 0. 핵심 전환

지금 live는 "계좌 전체(`account.total_krw`)에서 이 신호에 얼마를 넣을지" 계산해
추천한다. 새 방식은 "사용자가 구글 스프레드시트에 미리 적어 둔 종목들에 대해,
오늘 전략 규칙상 뭘 해야 하는지" 조언하는 쪽으로 바뀐다. 총금액·QQQM 배분·
계좌 전체 자금 계획 로직은 live에서 완전히 빠진다 — 종목별 계획금액이 서로
독립이라 "남은 계좌 한도를 여러 종목에 나눠 배분"하는 지금의 `allocate_remaining_limit`
류 로직 자체가 필요 없어진다.

**core/state.py·core/signals.py는 하나도 안 건드린다.** 오늘 A2 조건이 됐는지,
손절인지, E1/E2인지는 이미 이 파일들이 정확히 판정한다. 바뀌는 건 ① 그 판정
결과를 "달러 슬롯 배분" 대신 "종목별 계획금액 기반 수량"으로 사이징하는 부분과
② 그 결과를 보여주는 리포트 레이어뿐이다.

**paper 모드는 이번 변경과 무관하게 지금 방식(전체 101종목 스캔 + 계좌-기반
가상 체결) 그대로 유지한다** — live(사용자 주도)와 비교할 기준선 역할을 계속
한다. `engine/daily.py`의 `run()`은 mode로 명확히 분기한다. 이건 "live/paper가
항상 같은 수량 공식을 써야 한다"는 기존 P5-1 0번 원칙을 이번에 의도적으로
깨는 것 — 모듈 docstring에 이 예외를 명시해 둔다.

## 1. 입력: 구글 스프레드시트 (계획·체결)

`data/fills.xlsx`를 없애고 구글 스프레드시트 하나(탭 "계획"·"체결")로 옮긴다.

- **계획** 시트: 티커, 계획금액(원), 메모. 계획금액 = 1차+2차+3차 합계 예산(원).
- **체결** 시트: 지금 fills.xlsx의 "체결기록" 시트와 같은 열(날짜, 종목, 차수,
  매수매도, 체결가, 수량).
- `data/sheets.py`(신규): 서비스 계정으로 읽는다. 인증 정보는 절대 커밋 안 함 —
  로컬은 `.env`, GitHub Actions는 Secret(4번)에서 읽는다.
- 시트를 읽은 시각(KST)을 캡처해 보유 종목 섹션 상단에 표시(3번).

## 2. 1:2:6 차수 계산 (계좌 총액과 완전히 분리)

기존 `core.sizing.STAGE_SLOT_FRACTION = {"A1": 1/9, "A2": 2/9, "A3": 6/9}`를
그대로 재사용하되, `slot_krw(cfg)`(계좌 총액 기반) 대신 **이 종목의 계획금액**을
기준으로 삼는 새 순수 함수를 `core/sizing.py`에 추가한다(기존 계좌-기반 함수는
그대로 둔다 — paper·P5·KJB가 계속 쓴다):

```
tranche_krw = 계획금액 × {A1: 1/9, A2: 2/9, A3: 6/9, B: 1.0}[stage]
tranche_qty = floor(tranche_krw ÷ 환율 ÷ 지정가)
```

위험 예산(2% 룰) 기반 상한은 "자금 배분 로직"의 일부로 보고 live에서는 뺀다 —
계획금액 자체가 사용자가 이미 정한 위험 한도라고 본다.

## 3. 판정 — 기존 이벤트를 4가지 말로 매핑 (새 신호 로직 없음)

`core.state.process_day`가 내는 이벤트를 그대로 재분류한다:

| 오늘 이벤트 | 새 판정 | 표시 |
| --- | --- | --- |
| 없음 | **보유** | — |
| A1·A2·A3·B(신규 진입) | **추가매수** | 차수, 손절가, 2·3차 조건. 계획 있으면 +금액(원/달러)·수량 |
| E1·E2 | **일부매도 검토** | 해당 조건 한 줄 |
| E3·STOP·A1_EXPIRE | **매도** | 손절가 또는 사유 |

## 4. 실행 환경: GitHub Actions (매일 KST 07:00)

- `.github/workflows/daily.yml`(신규): cron `0 22 * * *`(UTC, 한국 DST 없어
  고정). live → paper(`--no-send`) 순서(지금 `scripts/run_daily.ps1`과 동일한
  순서, 로직만 Actions step으로).
- Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GOOGLE_SERVICE_ACCOUNT_JSON`,
  `GOOGLE_SHEETS_ID`, `GOOGLE_DRIVE_STATE_FILE_ID_LIVE`, `GOOGLE_DRIVE_STATE_FILE_ID_PAPER`.
- 실패 알림: `if: failure()` 스텝에서 기존 `scripts/send_ops_alert.py`(윈도우
  작업 스케줄러용으로 이미 만든 것) 그대로 재사용 — 새로 안 만든다.

## 5. 상태 저장: 구글 드라이브, "빈 파일 덮어쓰기" 방식

GitHub Actions 러너는 실행마다 새 가상머신이라 `state.db`·`paper_state.db`가
실행 사이에 안 남는다. 구글 드라이브에 저장하기로 했는데, **서비스 계정은
자체 저장 용량이 없어 새 파일을 만들 수 없다** — 대신:

1. **사용자가 미리** 구글 드라이브에 빈 파일 2개(`state.db`용, `paper_state.db`용)를
   만들어 서비스 계정에 **편집 권한으로 공유**한다(체크리스트 참고).
2. 자동화는 그 파일의 **ID**(URL에 있는 고정 문자열)로 항상 "내용 덮어쓰기"
   (Drive API의 update media, 새로 만들기 아님)만 한다 — 소유자(사용자)의
   저장 용량을 쓰므로 서비스 계정 용량 문제가 없다.
3. 실행 시작: 해당 파일 ID로 다운로드 → 로컬 `data/state.db`/`data/paper_state.db`로
   저장 → 엔진 실행(지금과 동일) → 실행 끝: 같은 파일 ID로 업로드(덮어쓰기).
4. `store/db.py`에 `download_from_drive(file_id, local_path)`·
   `upload_to_drive(file_id, local_path)` 추가(같은 서비스 계정 재사용).

## 6. "오늘의 추천" 섹션 — 가격·조건만, 금액은 계획 있을 때만

오늘 진입형 이벤트(A1·A2·A3·B)가 뜬 종목은 계획 유무와 무관하게 전부 여기
나온다: 1차 진입가·손절가·2·3차 조건은 항상, 원화·달러 금액·정수 주식 수는
**계획 시트에 있는 종목만** 추가로.

## 7. 보유 종목 섹션 — 표 + 시각 먼저, 그다음 판정

1. 보유 현황 표: 종목·수량·평단·현재가·원화 손익.
2. "시트를 읽은 시각: HH:MM:SS KST"
3. 판정이 바뀐 종목 — 맨 위에 요약 한 번 더 + 개별 카드에도 "🔄 바뀜" 표시.
   (`store/db.py` positions 테이블에 `last_judgment` 필드 추가.)
4. 종목별 판정 카드(3번의 4분류).

계획 경고: 계획이 있는 종목마다 1차(1/9) 금액으로 1주도 못 사면 — 신호 여부와
무관하게 상시 — 리포트 상단에 "⚠️ [티커] 계획금액이 너무 작아 1차 매수가 0주"
경고(최소 필요 금액 같이 표시).

## 8. 라이브 수익률 계산 — 총자산 추정 대신 "매수금액 대비 손익 + 그림자 QQQM"

처음 안(일별 NAV 추정)은 기각 — "현금이 얼마 남았는지"를 알려면 결국 총자금을
알아야 해서, 이번에 뺀 계좌-전체 개념이 다시 들어온다. 대신:

- **라이브 수익률** = 실제 매수(원가 기준) 대비 손익의 비용가중 합.
  `sum(현재가치) / sum(매수원가) - 1` — 지금 갖고 있지 않은 "총자금"은 전혀
  필요 없고, 실제로 산 달러(원화)만 쓴다.
- **그림자(shadow) QQQM**: 각 실제 매수 건마다 "그날 같은 금액을 QQQM에 넣었다면"을
  나란히 기록한다 — `shadow_qty = 매수금액 ÷ 그날 QQQM가 ÷ 그날 환율`. 그림자
  QQQM 수익률도 같은 비용가중 방식으로 계산 — 실제 매매와 정확히 같은 날짜·
  금액을 쓰므로 "내가 이 돈으로 그냥 QQQM을 샀다면"과 직접 비교된다.
- paper "전체"(2단계)도 같은 방식(비용가중)으로 계산해 세 줄을 동일한 정의로
  비교한다.

## 9. 기록: 1단계부터 쌓는 두 개의 append-only 로그

월간 비교(10번, 2단계)는 미루지만, 그때 쓸 데이터는 1단계부터 쌓는다 — 2단계
시작 시점에 과거 데이터가 없어서 처음부터 다시 시작하는 일이 없게.

- **`fill_ledger`**(체결/판정 이벤트마다 한 줄): date, mode(live/paper), ticker,
  side, qty, price_usd, fx_rate, amount_krw, 그날 QQQM 종가(그림자 계산용).
  실제 체결(live)과 가상 체결(paper) 둘 다 여기 쌓는다.
- **`daily_mark`**(보유 중인 종목마다 매일 한 줄): date, mode, ticker, qty,
  price_usd, fx_rate. 새 체결이 없는 날에도 매일 찍어 8번 수익률 곡선을
  거래일 사이에도 매끈하게 그릴 수 있게 한다.

두 테이블 다 `store/db.py`에 추가(신규 스키마, 기존 `positions`·`events`는
안 건드림). 2단계에서 이 두 테이블만 읽어 월간 비교·"안 산 추천의 이후 성과"를
계산한다("안 산 추천"은 새로 기록할 필요 없이 기존 `events` 테이블의 A1·B
발생일 중 그 시점에 계획·체결에 없던 종목만 골라 캐시된 시세로 나중에 계산).

## 10. 월간 비교 (2단계, 설계만 — 이번엔 구현 안 함)

| | 수익률(%) |
| --- | --- |
| paper 전체 | `fill_ledger`(mode=paper) 비용가중 |
| 내 라이브 | `fill_ledger`(mode=live) 비용가중 |
| 그림자 QQQM | `fill_ledger`의 그날 QQQM가 컬럼으로 재구성 |

"내가 사지 않은 추천 종목의 이후 성과": `events`에서 A1·B 이벤트 날짜·종목을
뽑아 그 시점에 계획·체결에 없던 것만 걸러 이후 N일 가격을 캐시에서 계산.

## 11. 1단계 범위 (이번에 구현) / 2단계 범위 (미룸)

**1단계**: 1(시트 입력) · 2(계획 기반 사이징) · 3(판정 매핑) · 4(GitHub Actions) ·
5(드라이브 상태 저장) · 6(오늘의 추천) · 7(보유 종목 섹션) · 8(수익률 계산 방식,
계산 로직만 — 리포트에 표시는 2단계) · 9(일별 로그 기록 시작).
**2단계**: 10(월간 비교 리포트 UI).

## 12. 1단계 구현 순서 (작은 단위 — 단계마다 pytest)

| 순서 | 내용 | 새/수정 파일 | 테스트 |
| --- | --- | --- | --- |
| 1a | `data/sheets.py`: 서비스 계정 인증, 계획·체결 시트 읽기(네트워크는 몽키패치로 테스트) | `data/sheets.py`(신규) | `tests/test_sheets.py` |
| 1b | `core/sizing.py`: 계획금액 기반 차수·수량 함수(순수 함수) | `core/sizing.py` | `tests/test_sizing.py`에 추가 |
| 1c | `store/db.py`: `last_judgment` 필드, `fill_ledger`·`daily_mark` 테이블 | `store/db.py` | `tests/test_db.py`(있으면 추가, 없으면 신규) |
| 1d | `store/db.py`(또는 신규 `store/drive.py`): 드라이브 다운로드/업로드(네트워크는 몽키패치) | `store/drive.py`(신규) | `tests/test_drive.py` |
| 1e | `engine/daily.py`: live 판정 매핑(이벤트→4분류), 대상 종목을 계획∪체결로 좁히기, 1주 미만 경고, 바뀐 판정 감지 — paper 경로는 무변경 확인 | `engine/daily.py` | 기존 스위트 + 신규 회귀 테스트 |
| 1f | `notify/`: 오늘의 추천·보유 종목 섹션 리포트/텔레그램 레이아웃 | `notify/report_html.py`·`briefing.py`·`telegram.py`·템플릿 | 기존 스위트 + 스냅샷류 테스트 |
| 1g | `.github/workflows/daily.yml` + `scripts/`의 로컬 실행 경로 정리 | `.github/workflows/daily.yml` | 로컬에서 `act` 또는 수동 1회 실행 확인 |

각 항목 끝날 때마다 `pytest` 전체를 돌리고, 실패 없이 통과한 뒤에만 다음
항목으로 넘어간다.

## 13. 체크리스트 — 사용자가 직접 할 일

- [ ] 구글 클라우드 프로젝트에서 Sheets API·Drive API 활성화
- [ ] 서비스 계정 만들고 JSON 키 발급
- [ ] 계획·체결 두 탭이 있는 구글 스프레드시트 생성, **서비스 계정 이메일에
      편집 권한 공유**
- [ ] 구글 드라이브에 **빈 파일 2개**(`state.db`용, `paper_state.db`용 — 아무
      내용이나 상관없음, 자동화가 덮어씀) 만들고 **서비스 계정에 편집 권한
      공유**, 각 파일 ID 확인(URL의 `/d/<이 부분>/`)
- [ ] GitHub 저장소 Secrets 등록: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
      `GOOGLE_SERVICE_ACCOUNT_JSON`, `GOOGLE_SHEETS_ID`,
      `GOOGLE_DRIVE_STATE_FILE_ID_LIVE`, `GOOGLE_DRIVE_STATE_FILE_ID_PAPER`
- [ ] 저장소 Settings에서 Actions 활성화 확인
- [ ] 기존 윈도우 작업 스케줄러 "Nasdaq100 Screener Daily" 비활성화(이제
      GitHub Actions가 실행 — 전에 안내한 `Disable-ScheduledTask` 명령)
- [ ] 새 구글 시트에 실제 계획·체결 데이터 옮겨 적기(기존 `data/fills.xlsx` 내용)
