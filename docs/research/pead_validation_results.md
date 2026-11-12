# 실적급등 장부 검증 보강 — 과거 구간 결과 (참고)

> 정의: `docs/prereg/pead_watch_reporting.md`, `docs/prereg/pead_watch_tags.md`. 원 사전 등록(`pead-watch-prereg`)·판정 규칙은 그대로다.
> 과거 계산은 각 1회, 결과만 보고한다. 신호 선택·규칙 수정에 쓰지 않는다.

<!-- survivorship -->
## 1. 생존자 편향 점검 — 2012~2021 명단 중 가격이 없던 티커

- 명단(2011-12 ~ 2021-12 매달 말 나스닥100 ∪ S&P100) 301개 중 가격 없음 **22개** (타이밍 탐색 '279/301'과 같음).
- 이 티커들의 명단 기간 8-K 2.02(직전 발표 60일 규칙으로 센 것) 합 **294건** — SW1 실적 발표 표 6,622건 대비 4.4%. 이 발표들은 가격이 없어 신호 판정 자체를 못 했다(신호 비율이 같다고 보면 신호 636건 중 약 28건 규모).
- 기간 = 첫 달 말 ~ 마지막 달 말 + 1달(전 달 말 명단 규칙), 2012~2021로 자름. 8-K 2.02 = 정정 제외 8-K 중 Item 2.02.
- 사유는 위키 나스닥100 변경표(빠질 때 적힌 문구)와 EDGAR 마지막 공시·상장 폐지 관련 서식(25·15·DEFM14A·S-4 등). S&P100만 있던 종목은 위키 사유 칸이 없다.

| 티커 | EDGAR 이름 | 나스닥100 | S&P100 | 8-K 2.02 (센 것/전체) | 상태 | 마지막 10-Q·10-K | 근거 (위키 사유 · EDGAR 서식) | 무료 가격 확인 (SW1 때 Tiingo API) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALTR | ALTERA CORP | 2011-12-31~2015-09-30 | — | 16/16 | 인수 합병(인텔), 상장 폐지 2015-12 | 2015-10-22 | October 7, 2015 · Altera merged with Intel.<br>15-12G 2016-01-11<br>25-NSE 2015-12-28<br>DEFM14A 2015-08-24 | yfinance ALTR: 없음<br>Tiingo 목록 ALTR: 1988-04-04~2017-10-31 (NASDAQ)<br>Tiingo 목록 ALTR: 2017-11-01~2025-03-26 (NASDAQ)<br>(SW1 Tiingo: 판정·다른 회사 의심) |
| APC | ANADARKO PETROLEUM CORP | — | 2012-03-31~2016-02-29 | 16/16 | 인수 합병(옥시덴탈), 상장 폐지 2019-08 | 2019-07-30 | 15-12B 2019-08-19<br>25-NSE 2019-08-09<br>DEFM14A 2019-07-11<br>25-NSE 2018-06-07 | yfinance APC: 없음<br>Tiingo 목록 APC: 1986-09-09~2019-08-08 (NYSE)<br>Tiingo 목록 APC: 2026-02-12~2026-11-11 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| APOL | APOLLO EDUCATION GROUP INC | 2011-12-31~2012-11-30 | — | 4/4 | 지수 탈락 2012-12 → 비상장 전환 2017-02 | 2017-01-09 | December 24, 2012 · Annual index reconstitution.<br>15-12G 2017-02-13<br>25-NSE 2017-02-01<br>DEFM14A 2016-03-23 | yfinance APOL: 없음<br>Tiingo 목록 APOL: 2016-01-04~2017-02-09 (NASDAQ)<br>(SW1 Tiingo: 판정·다른 회사 의심) |
| BBBY | 20230930-DK-Butterfly-1, Inc. | 2011-12-31~2016-11-30 | — | 20/22 | 지수 탈락 2016-12 → 파산·상장 폐지 2023 | 2023-06-14 | December 19, 2016 · Annual index reconstitution.<br>15-12G 2023-09-29<br>25-NSE 2023-07-10<br>S-4 2022-10-18<br>SC TO-T 2012-05-25 | yfinance BBBY: 없음<br>yfinance BBBYQ: 없음<br>Tiingo 목록 BBBYQ: 1992-06-05~2023-09-29 (PINK)<br>(SW1 Tiingo: —) |
| BMC | BMC SOFTWARE INC | 2011-12-31~2013-06-30 | — | 6/6 | 비상장 전환 2013-09 | 2013-08-05 | July 25, 2013 · BMC Software was taken private.<br>15-12B 2013-09-20<br>25-NSE 2013-09-10<br>DEFM14A 2013-06-25 | yfinance BMC: 없음<br>(SW1 Tiingo: Tiingo 자료 없음) |
| CA | CA, INC. | 2011-12-31~2018-10-31 | — | 27/29 | 인수 합병(브로드컴), 상장 폐지 2018-11 | 2018-11-07 | November 5, 2018 · NXP Semiconductors rejoined the index following its failed acquisition by Qualcomm. CA Inc. merged with Broadcom.<br>15-12B 2018-11-15<br>25-NSE 2018-11-05<br>DEFM14A 2018-08-10<br>SC TO-T 2015-06-08 | yfinance CA: 없음<br>Tiingo 목록 CA: 1984-09-07~2018-11-06 (NASDAQ)<br>Tiingo 목록 CA: 2023-12-14~2026-05-07 (NASDAQ)<br>(SW1 Tiingo: —) |
| CTRX | Catamaran Corp | 2012-12-31~2015-06-30 | — | 10/10 | 인수(유나이티드헬스), 상장 폐지 2015-07 | 2015-05-04 | July 24, 2015 · Catamaran Corporation went private.<br>15-12G 2015-08-03<br>25-NSE 2015-07-23<br>DEFM14A 2015-06-08 | yfinance CTRX: 없음<br>Tiingo 목록 CTRX: 2006-06-13~2015-07-23 (NASDAQ)<br>(SW1 Tiingo: —) |
| DISCA | Warner Bros. Discovery, Inc. | 2012-12-31~2017-11-30 | — | 20/20 | 지수 탈락 2017-12, 상장 유지 → 2022 WBD로 합병·티커 변경 | 2026-08-06 | December 18, 2017 · Annual index reconstitution.<br>S-4 2021-11-18<br>S-4 2021-02-23<br>S-4 2019-03-05<br>S-4 2017-09-14 | yfinance DISCA: 없음<br>Tiingo 목록 DISCA: 2022-04-06~2022-09-16 (NASDAQ)<br>yfinance WBD: 없음<br>Tiingo 목록 WBD: 2005-07-08~2026-10-07 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| DTV | DIRECTV | 2011-12-31~2015-06-30 | — | 14/14 | 인수 합병(AT&T), 상장 폐지 2015-07 | 2015-08-07 | July 24, 2015 · DirecTV was acquired by AT&T.<br>DEFM14A 2014-08-21<br>15-12B 2012-09-06<br>25-NSE 2012-08-27<br>S-4 2012-03-26 | yfinance DTV: 없음<br>Tiingo 목록 DTV: 2016-09-30~2019-09-30 (NYSE)<br>(SW1 Tiingo: 판정·다른 회사 의심) |
| EMC | EMC CORP | — | 2011-12-31~2016-10-31 | 19/21 | 인수 합병(델), 상장 폐지 2016-09 | 2016-08-08 | S-4 2021-04-15<br>15-12B 2016-09-20<br>25-NSE 2016-09-07<br>DEFM14A 2016-06-06 | yfinance EMC: 없음<br>Tiingo 목록 EMC: 1988-12-16~2016-09-15 (NYSE)<br>Tiingo 목록 EMC: 2023-05-15~2026-11-11 (NYSE)<br>(SW1 Tiingo: —) |
| ENDP | Endo International plc | 2015-12-31~2016-06-30 | — | 2/2 | 지수 탈락 2016-07 → 파산·상장 폐지 2022 | 2024-03-06 | July 18, 2016 · Endo International did not meet the minimum monthly weight requirements.<br>15-12G 2023-01-31<br>25-NSE 2022-09-14<br>SC TO-T 2020-11-02 | yfinance ENDP: 없음<br>yfinance ENDPQ: 없음<br>Tiingo 목록 ENDPQ: 2000-07-18~2024-05-01 (EXPM)<br>yfinance NDOI: 없음<br>Tiingo 목록 NDOI: 2024-06-28~2025-07-31 (PINK)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| LIFE | Life Technologies Corp | 2011-12-31~2013-07-31 | — | 7/7 | 인수 합병(써모피셔), 상장 폐지 2014-02 | 2013-11-05 | August 22, 2013 · Life Technologies was in the process of merging with Thermo Fisher Scientific.<br>15-12G 2014-02-14<br>15-15D 2014-02-04<br>25-NSE 2014-02-04<br>DEFM14A 2013-07-22 | yfinance LIFE: 없음<br>Tiingo 목록 LIFE: 2026-01-29~2026-11-11 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| LMCK | Liberty Media Corp | 2011-12-31~2016-05-31 | — | 14/15 | 지수 탈락 2016-06, 2016-04 트래킹 주식으로 재편(기호 사라짐) | 2026-08-06 | June 20, 2016 · Liberty Media did not meet the minimum monthly weight requirements.<br>DEFM14A 2016-12-09<br>25-NSE 2016-06-17<br>25-NSE 2016-04-18<br>S-4 2015-12-22 | yfinance FWONK: 2014-07-08~2016-08-08 527행<br>Tiingo 목록 FWONK: 2014-07-08~2026-11-11 (NASDAQ)<br>yfinance LMCK: 없음<br>yfinance LSXMK: 없음<br>Tiingo 목록 LSXMK: 2016-04-18~2024-09-09 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| MON | MONSANTO CO /NEW/ | — | 2011-12-31~2018-05-31 | 25/28 | 인수 합병(바이엘), 상장 폐지 2018-06 | 2018-04-05 | 15-12B 2018-06-18<br>25-NSE 2018-06-07<br>DEFM14A 2016-11-10 | yfinance MON: 없음<br>Tiingo 목록 MON: 2000-10-18~2018-06-08 (NYSE)<br>Tiingo 목록 MON: 2021-03-16~2022-12-23 (NASDAQ)<br>(SW1 Tiingo: 판정·다른 회사 의심) |
| NYX | NYSE Euronext | — | 2011-12-31~2012-08-31 | 3/3 | 인수 합병(ICE), 상장 폐지 2013-11 | 2013-11-05 | 15-12B 2013-11-25<br>25-NSE 2013-11-13<br>DEFM14A 2013-04-30 | yfinance NYX: 없음<br>Tiingo 목록 NYX: 2007-06-04~2013-11-12 (NYSE)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| QRTEA | Old QVC Group, Inc. | 2011-12-31~2018-11-30 | — | 27/30 | 지수 탈락 2018-12, 상장 유지(기호 변경 QVCGA 2025) | 2026-08-04 | December 24, 2018 · Annual index reconstitution.<br>SC TO-T 2015-09-01<br>S-4 2015-09-01<br>25-NSE 2012-10-09<br>S-4 2012-04-03 | yfinance QRTEA: 없음<br>Tiingo 목록 QRTEA: 2026-02-06~2026-11-11 (NASDAQ)<br>yfinance QVCGA: 없음<br>Tiingo 목록 QVCGA: 2026-05-28~2026-11-11 (PINK)<br>(SW1 Tiingo: —) |
| RIMM | BLACKBERRY Ltd | 2011-12-31~2012-11-30 | — | 0/0 | 지수 탈락 2012-12, 상장 유지(블랙베리 BB, 외국 발행인 6-K) | 2026-09-24 | December 24, 2012 · Annual index reconstitution.<br>25 2017-10-13 | yfinance BB: 2011-11-21~2013-02-08 305행<br>Tiingo 목록 BB: 1999-02-04~2026-11-11 (NYSE)<br>yfinance BBRY: 없음<br>yfinance RIMM: 없음<br>(SW1 Tiingo: —) |
| SHLD | SEARS HOLDINGS CORP | 2011-12-31~2013-11-30 | — | 8/9 | 지수 탈락 2013-12 → 파산·상장 폐지 2018~19 | 2018-12-13 | December 23, 2013 · Annual index reconstitution.<br>15-12B 2014-12-10<br>25-NSE 2014-11-13<br>25-NSE 2014-11-04<br>15-12G 2013-01-25 | yfinance SHLD: 없음<br>Tiingo 목록 SHLD: 2023-09-13~2026-11-11 (NYSE)<br>Tiingo 목록 SHLD: 2020-11-06~2021-10-12 (NASDAQ)<br>yfinance SHLDQ: 없음<br>Tiingo 목록 SHLDQ: 2003-05-01~2022-10-31 (EXPM)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| SNDK | SANDISK CORP | 2011-12-31~2016-02-29 | — | 17/19 | 인수 합병(웨스턴디지털), 상장 폐지 2016-05 | 2016-05-02 | March 16, 2016 · SanDisk was acquired by Western Digital.<br>15-12G 2016-05-23<br>25-NSE 2016-05-12<br>DEFM14A 2016-02-05<br>SC TO-T 2014-06-24 | yfinance SNDK: 없음<br>Tiingo 목록 SNDK: 2025-02-24~2026-11-11 (NASDAQ)<br>Tiingo 목록 SNDK: 1995-11-08~2016-05-20 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| SPLS | STAPLES INC | 2011-12-31~2015-11-30 | — | 16/16 | 지수 탈락 2015-12 → 비상장 전환 2017-09 | 2017-08-24 | December 21, 2015 · Annual index reconstitution.<br>15-12G 2017-09-22<br>25-NSE 2017-09-12<br>DEFM14A 2017-08-03<br>S-4 2015-03-20 | yfinance SPLS: 없음<br>Tiingo 목록 SPLS: 2026-01-16~2026-11-11 (BATS)<br>Tiingo 목록 SPLS: 1990-03-26~2026-04-10 (NASDAQ)<br>(SW1 Tiingo: Tiingo 자료 없음) |
| STRZA | Starz, LLC | 2013-01-31~2013-02-28 | — | 1/1 | 지수 탈락 2013-03 → 인수(라이온스게이트) 2016-12 | 2016-11-08 | March 18, 2013 · Starz did not meet the minimum monthly weight requirements.<br>15-15D 2017-01-09<br>S-4 2013-04-16 | yfinance STRZA: 없음<br>Tiingo 목록 STRZA: 2016-01-04~2016-12-16 (NASDAQ)<br>(SW1 Tiingo: 판정·다른 회사 의심) |
| YHOO | ALTABA INC. | 2011-12-31~2017-05-31 | — | 22/22 | 지수 탈락 2017-06(버라이즌 사업 매각 → 알타바 펀드, 2019 청산) | 2017-05-09 | June 19, 2017 · Yahoo! Inc. was acquired by Verizon and subsequently converted to a closed-end fund.<br>25-NSE 2019-10-07<br>DEFM14A 2017-04-24 | yfinance AABA: 없음<br>Tiingo 목록 AABA: 1996-04-12~2019-11-06 (NASDAQ)<br>yfinance YHOO: 없음<br>Tiingo 목록 YHOO: 1996-04-12~2017-06-26 (NASDAQ)<br>(SW1 Tiingo: —) |

- 가격 확인은 무료 출처(yfinance, Tiingo 공개 지원 목록)만 봤고, 받아 두거나 계산에 넣지 않았다. 결제 없음.
<!-- /survivorship -->

<!-- history -->
## 2. 포트폴리오 성적 (가상 1억, 신호마다 남은 현금 1/10, 남는 현금 = QQQ) — 참고

- 주 장부 규칙 B1·90 매매(2012~2021 549건 +0.67%, 2022~2026-09 336건 −0.69% 재현 확인)를 한 계좌에. 세금 전, 수수료 0.07%·시작 환전 0.1%.
- 원화 = 달러 × 마지막 날 환율(나갈 때 환전 비용 없음). QQQ 100%도 같은 시작 달러·같은 환율. 가격은 분할만 반영(배당 제외 — 종목·QQQ 같음), 2026은 9월 말까지.

| 구간 | 최종 평가액 | QQQ 100% | QQQ 대비 | 연 환산 (QQQ) | 최대 낙폭 (QQQ) | 매수 / 건너뜀 | 평균 보유 종목 · 주식 비중 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2012~2021 | 7.53억 원 | 7.21억 원 | +4.50% | +22.02% (+21.48%) | -32.4% (-28.6%) | 541 / 8 | 19.3개 · 63% |
| 2022~2026-09 | 1.96억 원 | 2.11억 원 | -6.98% | +12.03% (+13.75%) | -32.7% (-35.2%) | 321 / 15 | 24.3개 · 66% |

2012~2021 연도별 (포트폴리오 / QQQ 100%): 2012 +14.3% / +14.5% · 2013 +33.9% / +35.1% · 2014 +20.0% / +17.4% · 2015 +12.5% / +8.3% · 2016 +10.9% / +5.9% · 2017 +32.5% / +31.5% · 2018 +1.5% / -1.0% · 2019 +33.4% / +37.8% · 2020 +35.3% / +47.6% · 2021 +31.4% / +26.8%
- 건너뜀 사유: {'최소 금액 미만': 8}

2022~2026-09 연도별 (포트폴리오 / QQQ 100%): 2022 -26.5% / -33.7% · 2023 +34.9% / +53.8% · 2024 +24.5% / +24.8% · 2025 +17.0% / +20.2% · 2026 +18.5% / +20.4%
- 건너뜀 사유: {'최소 금액 미만': 14, '1주 값보다 예산이 작음': 1}

## 3. 다음 실적 발표 전·후 (B1·90 매매, QQQ 대비 초과, 달러·비용 전) — 참고

- A = 매수일 시가 → 다음 발표 반응일 전날 종가 · B = 그 종가 → 매도일 종가(다음 발표 반응일 포함) · 반응일 하루 = B 중 그날만.
- 다음 발표 = 각 구간 데이터 조사의 실적 발표 반응일(매수일 < 반응일 ≤ 실제 매도일 중 첫째).

| 구간 | 다음 발표 있음 / 없음 | A 평균 · 중앙값 · 승률 (평균 길이) | B 평균 · 중앙값 · 승률 (평균 길이) | 반응일 하루 평균 · 중앙값 · 승률 |
| --- | --- | --- | --- | --- |
| 2012~2021 | 543 / 6 | +0.65% · -0.25% · 49% (62일) | -0.10% · -0.22% · 48% (29일) | -0.04% · -0.31% · 48% |
| 2022~2026-09 | 329 / 7 | -0.04% · -1.05% · 47% (62일) | -0.77% · -2.02% · 45% (29일) | -0.30% · -0.09% · 50% |

## 5. 무작위 비교 1,000회 vs 10,000회 · 95% 구간 (B1·90, 세금 뒤) — 참고

- 1,000회 = 타이밍 탐색과 같은 시드 20261111(재현), 10,000회 = 장부 시드 20261112. 무작위 = 같은 매수일·예정 매도일, 전 달 말 명단 1종목.
- 신호 95% 구간 = 매매 10,000번 복원 추출 평균의 2.5~97.5 백분위수.

| 구간 | 매매 | 신호 평균 (95% 구간) | 1,000회 백분위 · 무작위 중앙값 · 95% 구간 | 10,000회 백분위 · 무작위 중앙값 · 95% 구간 |
| --- | --- | --- | --- | --- |
| 2012~2021 | 549 | +0.67% (-0.64% ~ +2.03%) | 100.0 · -1.32% · -2.45% ~ -0.25% | 100.0 · -1.32% · -2.38% ~ -0.25% |
| 2022~2026-09 | 336 | -0.69% (-2.73% ~ +1.40%) | 94.8 · -2.22% · -3.99% ~ -0.47% | 94.8 · -2.18% · -3.89% ~ -0.40% |
<!-- /history -->
