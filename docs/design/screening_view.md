# 오늘의 스크리닝 화면 + 매수 "왜?" 4단 설명 (feat/why-explain, 2026-10-06)

화면 견본: `docs/design/screening_mockup.html` (숫자는 예시). 실제 견본: `python -m tests.screening_sample` → `outputs/screening_sample.html`.

## 범위와 원칙

- **표시 전용**이다. 신호·필터·상태 전이·수량·손절 계산(core/signals.py, filters.py, state.py, sizing.py)과 텔레그램 브리핑(notify/briefing.py)은 바꾸지 않았다.
  `tests/test_equivalence_main.py`가 main(94bea29)에서 만든 고정값(`tests/fixtures/equivalence_main.json`)과 이벤트·가상 체결 수량·손절가·보고서 매수/매도 줄을 비교한다.
- 계산은 `core/screening_view.py`(순수 함수). DB·거래소 달력 읽기는 `engine/daily.py`의 `_screening_inputs`가 한다.
- "추천/제외"는 core/state.py가 실제로 낸 이벤트(A1·A2·A3·B·BLOCKED·STOP)를 따른다. 조건 열(✓/✗, RSI 등)은 오늘 지표 값이다.

## 구성

| 파일 | 역할 |
| --- | --- |
| `core/screening_view.py` | `build_screening_view`(레인 4개), `buy_facts`(매수 한 건의 오늘 값), `entry_state`, `a2_deadline`, `display_score` |
| `core/explain.py` | `explain_buy`에 `sections`(① 우리 규칙 ② 왜 이 종목인가 ③ 어떻게 사나 ④ 다음 단계)와 `source` 추가. 기존 키 유지 |
| `engine/daily.py` | `build_report_summary(..., recent_events, future_trading_days)` → `summary["screening"]`, 매수 줄에 `facts`·`funnel`·`sizing` 첨부. `_screening_inputs`(DB 최근 이벤트, NYSE 달력) |
| `notify/report_html.py` | `_strategy_card`(고정 문구 + config 숫자), `_screening_ctx` |
| `notify/templates/report.html.j2` | 헤더 바로 아래 "오늘의 스크리닝" 구역, "왜?" 4단 표시 |

## 차수별 후보 정의 (core/state.py process_day ⑤와 같음)

판정 시점 상태(`entry_state`): 오늘 STOP이면 없음(①에서 끝남). 오늘 신호로 주문대기가 됐으면 `pending.prev_state`. 그 밖에는 오늘 처리 후 상태.

| 레인 | 후보 | 단계 숫자 |
| --- | --- | --- |
| 1차 | 스캔 종목 중 RSI 30 상향 돌파 | 스캔 N → RSI 30 탈출 m(= funnel 1차) → 검사 통과(추천 + 보유한도로 막힘) → 오늘 추천(A1 이벤트) |
| 2차 | 판정 시점 "정찰"(1차 체결 확정, a1_date) | 1차 보유 n → 오늘 골든크로스 → 금지 구간 통과(A2 이벤트) |
| 3차 | 판정 시점 "확인"(2차 보유) | 2차 보유 n → 4조건 충족(= funnel 3차, check_a3) → 갭·금지 구간 통과(A3 이벤트) |
| 재진입 | 판정 시점 "대기" + 추세 확인(종가>구름 상단, 양운, 후행스팬). 쿨다운 무관 | 추세 종목 n → 오늘 골든크로스 → 조건·금지 구간 통과(B 이벤트) |

- 1차 제외 사유는 실제 검사만: 보유 중(단계명), 재진입 대기(cooldown_until), BLOCKED(실적·보유 한도), 오늘 손절.
- 2차 기한 = A1 당일 포함 `a1_to_a2_expiry_days`번째 거래일(그날까지 A2 판정, 다음 날 ②에서 만료). 지표 표 밖이면 NYSE 달력(`future_trading_days`).
- 2차 "참고: 2차 대상 아님": 최근 `a1_to_a2_expiry_days` 거래일 A1 이벤트 중 지금 정찰이 아니고 그 1차로 진행 중도 아닌 종목(사유: 체결 기록 없음 / 손절 / 만료).
- 2차 "대기" 줄이 10개를 넘으면 접는다. 점수는 모든 줄에 `priority_score`(등급은 A2·B형 골든크로스 날만), 순위는 추천 줄에만.

## 기존 funnel과의 대조

| `_compute_funnel` | 화면 |
| --- | --- |
| 1차 RSI 30 돌파 | 1차 레인 "RSI 30 탈출" (같음) |
| 2차 MACD 골든크로스 (전체 종목) | 레인별로 나뉨: 2차 레인 골든크로스 + 재진입 레인 골든크로스 + 그 밖(보유 중·구름 아래 등). 구역 아래 한 줄로 합계 표시 |
| 3차 일목구름 4요소 | 3차 레인 "4조건 충족" (같음) |
| 4차 매매금지 / 5차 보유한도 | 레인 제외 사유의 합 (구역 아래 한 줄) |

텔레그램 브리핑에는 현재 "스크리닝 현황" 줄이 없다(문서 10장에는 있음) — 브리핑은 이번 범위 밖이라 그대로 둠.

## "왜?" 4단 (매수)

- ① 차수 조건마다 오늘 값과 기준, ✓/✗는 값 비교(`_verdict`), 값 없으면 "확인 불가". 기존 `checks`도 같은 결과를 쓴다(고정 "y" 제거, A3 종가 자리에 지정가를 쓰던 오류 수정).
- ② 깔때기 문장은 레인과 같은 숫자 + 점수 내역(등급 점수는 A2·B만 — A3에 2차 등급을, B에 "등급 없음"을 표시하던 오류 수정).
- ③ 지정가 = 종가 × `entry.limit_markup`, A3 갭 보류, 수량식(live: 계획금액/기본 금액 × 차수 비중 ÷ (지정가 × 환율), 기준가 방식; paper: 슬롯 목표·위험 상한), 줄어든 이유, 손절가 기준과 최저가 날짜, 손절폭.
- ④ 다음 단계 + "근거: 전략 v3 4장 … · 대본 …"(13장 매핑).

## 결정 사항 (2026-10-06, 질문 목록 정리)

규칙은 바꾸지 않고, 문서를 코드 동작에 맞췄다(`docs/strategy_v3.md` v3.0.1 문서 정정). 동등성 테스트 그대로 통과.

| # | 질문 | 결정 |
| --- | --- | --- |
| 1 | 2차를 1차 당일에도 판정하나? | 아니오. 1차 체결이 확정된 **다음 거래일부터** 판정(코드 그대로). 문서 4장 정정 ① |
| 2 | A등급 "첫 골든크로스"의 뜻 | 최근 20거래일 MACD 교차(골든+데드) **1회 이하**(코드 그대로). 문서 7장 정정 ②. 값은 `strategy_levels.cross_window_days`·`grade_a_max_crosses` |
| 3 | 갭 필터 기준일 | 프로그램은 **신호일** 시가 갭으로 거름, **매수일** 시가 4% 이상이면 사람이 보류(수동 규칙). 문서 4·7·10장 정정 ③. "왜?" ③에 두 가지 모두 안내 |
| 4 | 재진입(B) 쿨다운 | 적용하지 않음(코드 그대로). 향후 백테스트 시험 후보. 문서 4·8장 정정 ④. 전략 카드에 "(쿨다운 없음)" 표시 |
| 5 | 고정값을 config로? | **옮김.** `config.yaml` `strategy_levels`: `rsi_oversold`(30), `rsi_mid`(50), `rsi_overbought`(70), `cross_window_days`(20), `grade_a_max_crosses`(1), `earnings_filter_trading_days`(3). `core.signals.levels(cfg)`로 읽는다. 전략 카드·"왜?"·제외 사유 문구도 이 값을 쓴다 |
| 6 | 텔레그램 스크리닝 현황 줄 | 넣지 않음. HTML 보고서 "오늘의 스크리닝"에서만 보여 줌. 문서 10장 정정 ⑥ |
| 7 | 실적 필터 "전체"의 범위 | **신규 매수**(1차·2차·3차·재진입)에만 적용. 보유 종목 자동 매도 없음. 문서 7·11장 정정 ⑦ |

참고: `_LEVEL_DEFAULTS`(core/signals.py)는 config에 `strategy_levels`가 없는 옛 테스트 cfg 호환용 기본값이며,
`tests/test_screening_view.py`가 config 값과 같은지 확인한다. 우선순위 점수 배점(S40·거래량 25 등)은 이번 범위에서 옮기지 않았다.
