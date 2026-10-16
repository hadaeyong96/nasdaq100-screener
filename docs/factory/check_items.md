# F4 전 확인 항목 (전략공장, 2026-10-16)

코드를 읽어 지금 상태만 적었다. **코드는 고치지 않았다.** F4(공장 공용 장부·비교 엔진) 전에 하나로 맞출지 정한다.

## 1. 세금

**한 줄 요약: 하나로 통일돼 있지 않다.**
- 옛 엔진(P5·P6·L1·E1·D1·해자)은 "과세연도 원화 실현손익 통산 − 기본공제 250만 원 → 22%, 다음 해 5월 납부"를 쓴다.
- 최근 탐색·검증 코드(SW1·L1b·RS26/ACCEL·탐색 B)와 가상 장부는 "그해 달러 실현 순이익 × 22%, 공제·이월결손 없음, 연말 납부"를 쓴다.

| 계열 | 방식 | 위치 |
| --- | --- | --- |
| P5 백테스트 엔진·KJB | 그해 원화 실현손익 합(종목+QQQM 통산) − 250만 원 공제 → 22%, 다음 해 5월에 QQQM을 팔아 냄(그 매도 손익은 다음 해) | `core/tax.py:16` `capital_gains_tax`, `engine/backtest.py:163` `pay_capital_gains_tax`, `engine/kjb_satellite.py:143` |
| P6·L1 (`engine/portfolio.py`) | 같음 (원화, 250만 원 공제, 5월 납부) | `engine/portfolio.py:181` `settle_may_tax` |
| E1·D1 적립 | 같음 (원화, 250만 원 공제) | `core/e1.py:290`, `core/dca.py:213` |
| 해자 H4 | 같음 (원화, 공제, 연 1회 전량 교체라 매 기간 전부 실현) | `core/moat_backtest.py:171` |
| L1b·L1d | 그해 순이익 × 22% (공제 없음), 그해 마지막 행에 납부, 납부용 매도 손익은 다음 해 | `core/l1b.py:278`, `:293` |
| SW1 (실적급등) | 매도 연도별 순손익이 양수면 × 22%, 이익 난 매매에 나눠 붙임. 공제 없음 | `core/sw1.py:139` `ledger_tax` |
| 가상 장부·RS26·ACCEL (`core/momentum_book.py`) | 그해 달러 실현 순이익 × 22%, 공제·이월결손 없음, 연말 장부 현금에서 차감(모자라면 보유 매도) | `core/virtual_books.py:19`, `core/momentum_book.py` |
| 탐색 B | 같음 (달러, 공제·이월 없음, 연말 현금 → QQQ 순서로 납부) | research/sector-pullback `core/sector_pullback.py` `run_book` |

**QQQ 그냥 보유와 매매 전략의 세금 시점 (공정성)**
- 옛 엔진(`engine/backtest.py`): 기준선(QQQM 보유)도 끝에 `_liquidate_qqqm_and_pay_tax`(`:900`)로 전량 매도·세금을 매긴다(청산 시나리오 (a), `compute_liquidated_cagr` `:977`). 끝 값은 양쪽 다 세후다. 다만 보유 쪽은 중간에 세금이 없어 과세 이연 이득을 그대로 갖는다. 이건 실제로도 그렇다.
- RS26·ACCEL·탐색 B: 전략은 해마다 세금을 내고, QQQ 기준선은 끝에 한 번만 낸다. **끝 값(세후 연복리) 비교는 공정**하다(둘 다 청산·세금 뒤).
- 반면 **구간별·연도별 비교는 전략에 불리**하다. 전략 곡선은 그해 세금을 이미 뺐고, QQQ 곡선은 세전이다. 결과 문서에 이렇게 적어 뒀다(`docs/results/rs26_zone3.md`, `docs/factory/explore/sector_pullback.md`).
- 공제 250만 원 유무: 원화 계열은 공제를 넣고, 달러 계열은 뺐다(차이 크기는 계산하지 않았다). 같은 비교 안에서는 전략·기준선이 같은 방식을 쓴다.
- 손실 이월: 어느 계열도 하지 않는다(한국 해외주식 양도세도 이월결손 없음, 그해 통산만).

## 2. 재무 데이터 사용 시점

| 코드 | 규칙 | 판단 |
| --- | --- | --- |
| E1 (`core/e1.py:4`, `:43`) | `filed ≤ 판단일`인 사실만 쓰고, 체결은 판단일 **다음 거래일** 종가 | 제출일 당일 장 마감 뒤 접수분도 다음 날 체결이라 미래 정보는 아니다. 다만 "접수 시각" 대신 "제출일(날짜)" 기준 |
| 해자 (`feature/moat` `core/moat.py:7`, `:151`) | `filed ≤ as_of`인 가장 최근 제출본 | 날짜 기준. 체결이 as_of 다음 날인지는 `core/moat_backtest.py`에서 따로 확인 필요 |
| SW1 (`scripts/sw1_data_check.py:97` `session_of`, `:107` `reaction_day`) | EDGAR **접수 시각(동부시간)**: 장전이면 그날, 장중·장후면 다음 거래일이 반응일 | 접수 시각 기준(가장 엄격) |
| `data/edgar.py:185` | `as_of`보다 늦은 `filed` 제외 | 공용 도구. 날짜 단위 |

→ **"접수 시각 다음 거래일부터"를 시각까지 지키는 것은 SW1뿐**이다. 나머지는 "제출일 ≤ 판단일, 다음 날 체결"이다. 장 마감 뒤 접수분을 같은 날 판단에 쓸 수는 있지만, 체결이 다음 날이라 수익에는 미래 정보가 들어가지 않는다. 주식 수 감소(share_count_down)·싼 우량(cheap_quality_family)은 SW1 방식(접수 시각)으로 고정할지 사전 등록 때 정한다.

## 3. 분할·합병·상장폐지·티커 변경 처리 위치

| 사건 | 위치 | 방식 |
| --- | --- | --- |
| 분할 (가격) | `data/prices.py`, `data/backtest_prices.py` | yfinance `auto_adjust=False` `Close`(분할 반영, 배당 미반영) — CLAUDE.md 규칙 3 |
| 분할 (Tiingo 원가격) | `scripts/sw1_data_check.py:226` `split_adjust`, research/sector-pullback `scripts/sector_pullback_data.py` `read_guarded` | raw × 누적 분할 비율 (기간 끝 기준) |
| 분할 (장부 수량) | `core/momentum_book.py:222`, `core/picks_book.py:153`, `data/virtual_books_data.py:133` | 그날 분할 비율만큼 보유 수량 × 비율 |
| 분할 (재무 사실) | `core/e1.py:33` `_split_factor` | filed 다음 날 ~ 판단일 분할만 반영 |
| 분사 | 처리 없음 | 분할 반영 종가라 분사일에 가격이 떨어진 것으로 보인다(예: 탐색 B 점검 6건) |
| 상장폐지·편입 제외 | `data/universe_history.py:123` `reconstruct_membership`, `:220` `point_in_time_universe` | 위키 변경 표로 시점별 나스닥100 명단 재구성 |
| 상장폐지 (가격 끝) | research/rs26-zone3 `scripts/rs26_zone3_run.py` `simulate`, research/sector-pullback `core/sector_pullback.py` `run_book` | 마지막 종가에 판 것으로 본다(파산은 낙관적) |
| 가격 공백 | `data/backtest_prices.py:116` | 복구 못 한 날 `data_gap` 처리 |
| 티커 변경 | `scripts/sw1_data_check.py:42` `EXTRA_RENAMES`, `:521` `renames()`(S&P 500 점검 `data/cache/sp500/renames.csv`), `data/universe.py:64` `_to_yfinance_ticker`(BRK.B → BRK-B 등) | 옛 티커 → 지금 티커 |
| 합병·지주회사 개편·같은 회사 확인 | `scripts/sw1_data_check.py` `step_events`(CIK·이름 유사도·공시 2건), research/sector-pullback `scripts/sector_pullback_data.py` `choose_source`(같은 CIK·지주회사 개편 규칙) | CIK 기준으로 같은 회사 확인 |
| 같은 회사 여러 주식 | research/sector-pullback `core/sector_pullback.py` `combine_shares` | 회사당 한 종목 (탐색 B 결정 3) |

## 4. F4 전에 정할 것 (제안, 결정은 사용자)

1. 공장 공용 세금 규칙 하나: 원화·250만 원 공제·5월 납부(옛 엔진) 또는 달러·공제 없음·연말(최근 코드).
2. 구간·연도별 비교 때 QQQ 기준선도 해마다 "팔았다고 치고" 세금을 매긴 곡선을 같이 보여 줄지.
3. 재무 데이터: 접수 시각 기준(SW1 방식)을 공장 기본값으로 할지.
4. 분사 처리: 그대로 둘지(보수적), 분사 목록을 만들어 보정할지.
