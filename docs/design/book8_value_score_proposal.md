# 제안서 — 가상 장부 8번 「6개월 강세 전략」 → 「가치평가 전략」 교체 (구현하지 않음)

> 작성 2026-10-18 (research/value-score). **승인 전에는 아무것도 바꾸지 않는다.** main 병합·daily 코드 수정은 사용자 승인 뒤 별도 작업이다.
> 근거: `docs/prereg/value_score.md`(태그 `value-score-prereg`), 판정 결과 `docs/results/value_score_zone3.md`, `factory/catalog.yaml`(8번 자리).
> 실제 주문 없음. 로컬에서는 텔레그램을 보내지 않는다(Actions만 보냄) — 시험은 dry-run.

## 0. 지금 상태

- 8번 장부 「6개월 강세 따라하기」(config `virtual_books.momentum`)는 `start_date: null`이라 **시작 기록이 없다**(첫 월말 신호 10/30 → 다음 날 체결로 고정할 계획이었음).
  이 계획을 취소하고 그 자리를 「가치평가 전략」으로 바꾸는 것이 목표다. catalog에서는 이미 6개월 강세 장부 계획을 탈락 보관, 8번 자리를 가치평가 전략(대기)으로 적었다.
- 「가치평가 전략」의 구역 ③ 판정은 **FAIL**(세후 11.32% vs QQQ 19.79%, 무작위 36.2백분위 — `docs/results/value_score_zone3.md`). 판정과 무관하게 "앞으로 기록"으로 관찰하는 것이 사용자 지시다(채택 아님).

## 1. 결정이 먼저 필요한 것 (사용자)

1. **납입**: 판정 장부는 1억 원 + 매달 100만 원이다. 다른 가상 장부는 "처음 돈" 한 번만 넣는다. 8번도 처음 돈만 넣을지(다른 장부와 비교 쉬움, 추천),
   매달 납입까지 흉내 낼지.
2. **세금 방식**: 가치평가 전략의 12월 갈아타기 수량은 **원화 실현 이익과 250만 원 공제**로 정해진다. 지금 가상 장부 세금은 "달러·공제 없음·연말"이라
   그대로는 규칙을 재현할 수 없다. 공용 규칙 1의 "가상 장부 세금 통일(2026-12-31 전 별도 작업)"을 **먼저** 하거나, 8번만 원화·공제 장부로 둘지.
   첫 12월 실행일(2026-12-29)이 다가오므로 순서를 정해야 한다.
3. **시작일**: 병합 뒤 첫 월말 점수 → 다음 거래일 종가(공용 규칙 3). 시작 뒤 `start_date`·`start_fx`를 시트 "config 고정값"대로 고정(1~7·9번과 같은 방식).

## 2. 필요한 변경 목록

### 데이터 (새로 필요)

| 무엇 | 어디 | 내용 |
| --- | --- | --- |
| 지금 명단 | `data/` 새 모듈 | 오늘 기준 S&P 500 − (나스닥 100 ∪ S&P 100). S&P 500·S&P 100 현재 명단 출처(위키 현재 판 등)와 실패 시 멈춤·보고 |
| 재무 | `data/edgar.py` 재사용 | 대상 약 340종목 companyfacts. 매달 전부 다시 받지 말고 submissions로 새 10-K 접수가 있을 때만 갱신(캐시) |
| SIC | `data/edgar.py` | submissions의 SIC (판정과 같은 방식) |
| 그날 실제 종가·분할 | `data/prices.py` | 오늘 종가는 원래 실제 가격. EPS·주식 수 분할 조정(E1 방식)용 분할 기록 — yfinance `splits`에서 흔한 비율만 |

### 계산 (core는 판정과 같은 함수)

| 무엇 | 어디 | 내용 |
| --- | --- | --- |
| 점수 | `core/value_score.py` 그대로 | `annual_table`·`metrics`·`score_day`·`rank_order`·`pick`·`swap_choice`·`swap_qty` — 라이브와 백테스트가 같은 core를 쓴다(CLAUDE.md) |
| 장부 하루 처리 | `core/value_book.py` 새로 | `scripts/value_score_zone3_run.py`의 Book·run_value 규칙을 순수 함수 `process_day`로 옮김(분기 말 판정, 1년 조건, 12월 갈아타기·빈자리·공제 소진, 상장폐지). 판정 장부와 같은 결과가 나오는지 테스트 |
| 8번 실행 | `engine/picks_books.py` | `run_momentum` 자리에 `run_value_score` (DB 상태 장부, 따라잡기, 시작일 고정 경고 `_start_mismatch_warning` 같은 방식) |
| 호출 | `engine/virtual_books.py` 233행 근처 | 8번 호출을 바꿈(오류가 나도 다른 장부·브리핑은 그대로) |
| 시작 전 안내 | `engine/daily.py` 2417행 근처 | "아직 시작 전인 장부" 목록에서 momentum → value_score |

### 설정·표시

| 무엇 | 어디 | 내용 |
| --- | --- | --- |
| 설정 | `config.yaml` `virtual_books` | `momentum.enabled: false`(코드·기록은 남김) + 새 `value_score:` 절(name "가치평가 따라하기", start_date/start_fx null, 문턱 49·35·56·10, 업종 상한 3, 슬롯 10, 공제 250만 원 — 지시문 [가정] 값은 config에서 읽음) |
| 보여 주는 이름 | `config.yaml` `display_names` | "6개월 강세 따라하기" 줄 → "가치평가 따라하기": "가치평가 전략" |
| 장부 순서 | `factory/scorecard.py` `strategies_from_cfg`, `notify/letter.py` 130행 | `("momentum", "accel")` → `("value_score", "accel")` (8번 자리 유지) |
| 시트 탭 글 | `notify/virtual_books.py` `REBALANCE_KINDS` | `"value_score": ("가치평가", "첫 월말 점수 다음 거래일")` |
| 시트 「전략목록」 **B9** | 구글 시트(코드에 없는 탭) | 이름을 「가치평가 전략」으로 손으로 바꾸거나, 승인 시 Sheets 연결로 한 칸만 수정 |
| 시트 「전략요약」 | 자동 | "config 고정값(momentum)" 줄 → "config 고정값(value_score)" |

### 테스트·문서

- 새 테스트: `tests/test_value_book.py`(process_day가 판정 장부와 같은 매매를 내는지 합성 시장으로), 시작일 고정·DB 우선 경고(`tests/test_pin_books_679.py` 방식).
- 고칠 테스트: `tests/test_factory_catalog.py`(관찰 중 수 그대로 9), scorecard·letter·virtual_books 테스트의 8번 이름.
- 문서: `docs/design/virtual_books.md` 8번 절, `docs/factory/check_items.md`.
- DB: 6개월 강세 장부는 시작 기록이 없어 지울 행이 없는지 `paper_state.db`에서 먼저 확인(있으면 백업 뒤 정리).

## 3. 순서 (승인 뒤)

1. 1장 결정 3개 → 2. 데이터 모듈 + `core/value_book.py` + 테스트(네트워크 없음) → 3. dry-run으로 오늘 점수·첫 선정 미리 보기(텔레그램 안 보냄)
→ 4. main 병합 → 5. Actions 첫 실행에서 시작일 기록 → 6. 시트 B9 이름·config 고정값.
