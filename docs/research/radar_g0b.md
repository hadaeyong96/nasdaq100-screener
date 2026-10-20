# 신기술 레이더 G0-보완 — 결과 (수익 계산 없음)

> 2026-10-20, 브랜치 `research/tech-radar`. 규칙은 결과 전에 `docs/research/radar_g0b_rules.md`로 고정(커밋 `15bd52a`)했다.
> 코드 `core/radar/g0b.py`, `data/radar/g0b_build.py`(단계 register → papers → filter → link). G0 말뭉치·단어 표(`08296c5`)를 그대로 썼다.
> 가격·수익 캐시는 읽지 않았다(`data/radar/guard.py`). 판정 예산 사용 없음. 결과를 본 뒤 규칙은 바꾸지 않았다.

## RESULT: 데이터 기록 — 감지율 21/40(규칙 합집합), 회사 연결 1,167건(티커 394개). 연결 정확도는 새 표본 50건 검수 전이라 미정

## 1. 등록 (거르기 전 → 뒤)

| 규칙 | 거르기 전 | ① 분야 | ②가 조각 | ②나 일반어 | ③ 사건 | 통과 |
| --- | --- | --- | --- | --- | --- | --- |
| 급증 (5배·50편) | 601 | 237 | 88 | 85 | 26 | **249** |
| 꾸준한 성장 (연 30%·200편) | 723 | 149 | 51 | 169 | 2 | **397** |
| 합집합(단어) | 1,179 | | | | | **565** |

- 거르기 칸은 그 거르기에 걸린 수(중복 가능). 통과 단어의 산업 분야 비율 중앙값 **91.6%**.
- 통과 등록 연도(첫 등록, 합집합): 2012 8 · 2013 11 · 2014 5 · 2015 13 · 2016 30 · 2017 44 · 2018 99 · 2019 115 · 2020 132 · 2021 103 (2011년 이전은 급증 5개뿐).
  규칙별: 급증 2012~2021 = 4·9·3·10·20·25·49·41·45·38, 꾸준한 성장 = 4·3·2·6·16·26·63·92·102·83.
- 단어 표본 50(합집합, 시드 20261019, 분류 칸 비움): `docs/research/radar_g0b_terms_sample50.csv`. 전체 표: `exports/radar/급증_단어목록.xlsx`(커밋 안 함).

## 2. 감지율 (평가 전용 — 신호에 쓰지 않음)

| 목록 | 급증 | 꾸준한 성장 | 합집합 |
| --- | --- | --- | --- |
| 처음 20 (G0, 한 단어 4개 구조상 불가) | 6 | 8 | **8** (G0 7) |
| 보류 20 (규칙 문서에 고정) | 4 | 11 | **13** |

- 처음 20 — 잡음: deep learning, convolutional neural network, generative adversarial network, federated learning, edge computing, autonomous driving(둘 다),
  reinforcement learning(성장 2016-09), internet of things(성장 2017-03). **large language model은 급증 2021-09 등록 → ②나 일반어(끝 단어 model)로 제외.**
  못 잡음: cloud computing, software defined networking, quantum computing, augmented reality, solid state battery, perovskite solar cell, additive manufacturing (+한 단어 4개).
- 보류 20 — 성장으로만: machine translation 2017-04, speech recognition 2017-08, object detection 2017-06, face recognition 2019-09, recommender system 2018-08,
  knowledge graph 2019-07, unmanned aerial vehicle 2018-07, point cloud 2018-08, natural language processing 2017-03. 급증·성장 둘 다: smart contract, semantic segmentation.
  급증으로만: digital twin 2020-08, massive mimo 2014-08. **millimeter wave는 성장 2017-09 등록 → ① 분야(산업 비율 54%)로 제외.**
  못 잡음: wireless power transfer, visible light communication, quantum key distribution, quantum error correction, neuromorphic computing, metal organic framework.
- 성장 규칙이 잡은 것은 대부분 1990년대부터 있던 분야라 처음 등장 → 등록 시차가 20년 이상이다(예: machine translation 1994-05 → 2017-04). "새 기술을 일찍"이 아니라 "오래된 분야가 커질 때"를 잡는다.
- 표: `data/cache/radar/g0b/detection.csv`.

## 3. 회사 연결 (사업 설명 부분 2회 이상)

- 대상: 통과 단어 중 등록 달 2001-01 ~ 2021-12 **565개**(G0 검색 캐시 재사용 + 새 단어 검색). 공시 결과 **408,860건**, 티커 매칭 69,620건 = **17.0%**.
- 문서: 단어 × 연도 최대 5건 → **11,139건** 선택(새로 받은 문서 3,257개, 실패 22개 — 2001년 옛 경로 HTTP 404, 본문 없음 28줄).
- 부분 찾기: 찾음 8,677 · 못 찾음 2,434(연결 아님). 양식별 찾은 비율 10-K 81% · 10-Q 76% · 10-K/A 43% · 10-Q/A 31%.
- **연결 1,167건 · 티커 394개 · 단어 151개.** 연결 행의 단어 규칙: 꾸준한 성장 1,029 · 급증 252(둘 다인 단어는 두 번 셈).
- 연도별 연결 행: 2001~2010 18~38건, 2011~2021 56~88건. G0와 달리 2020~2021 쏠림이 없다(사건 거르기).

**구역별 매칭률** (분모 = 그 기간 공시 결과 전체, "그 구역" = 접수일에 그 구역 명단에 있던 회사)

| 구역 | 기간 | 공시 결과 | 그 구역 매칭 | 연결(행 / 티커) |
| --- | --- | --- | --- | --- |
| 전체 | 2001~2021 | 408,860 | 69,620 (17.0%, 아무 티커) | 1,167 / 394 |
| 탐색(대형) 나스닥100 ∪ S&P 100 | 2011-12 ~ 2021 | 241,175 | 11,786 (**4.9%**) | 170 / 52 |
| 탐색 ② 중소형 S&P 400 ∪ 600 | 2012~2016 | 103,149 | 14,750 (**14.3%**) | 239 / 118 |
| 검증 ① 중소형 S&P 400 ∪ 600 | 2017~2021 | 137,455 | 19,867 (**14.5%**) | 269 / 131 |
| (참고) 검증 ③ S&P 500 나머지 | 2012~2021 | 240,604 | 11,795 (**4.9%**) | 122 / 59 |

- 연결 행의 구역(전 기간): 중소형 544 · S&P 500 나머지 453 · 탐색 대형 170.

**상투어가 연결에 섞인 비율** (사후 참고 지표 — 등록·연결 규칙에는 쓰지 않음)

- 고정 표지 목록(extensive, significantly, outperform, demonstrate, effectiveness, existing work, prior work, results show, art result, code is available,
  perform well, time consuming, terms of accuracy, increasing attention, significant challenge, experiment(al), proposed, superior, promising result 등 25개, 연결 결과 보기 전에 고정):
  연결 **29/1,167 = 2.5%**, 단어 9/151, 상투어 연결만 있는 티커 13개. 통과 단어 중 30/565.
- 이 목록은 논문 상투어만 잡는다. 연결 상위 단어를 보고 일반 영어 표현 39개(near term, key component, across multiple, decision maker, success rate,
  improve the quality, across various, widely adopted, sub optimal, challenging due 등 — 목록은 결과 본 뒤 Claude 판단)로 넓히면
  **356/1,167 = 30.5%**, 이것만으로 연결된 티커 148개다. 연결 1위 "near term"(58건·40티커), 2위 "key component"(48건·26티커).
- 사건성 단어(health authority, health crisis, rt pcr) 29건도 남았다(③ 목록 밖 표현).

- 연결 표본 50(시드 20261019, 발췌는 사업 설명 부분, 분류 칸 비움): `docs/research/radar_g0b_link_sample50.csv`.
  전체 표: `data/cache/radar/g0b/ticker_term_evidence_g0b.parquet`(검수 상태 = 미검수).

## 4. 한계

1. 분야 비율은 대표 분야(primary) 하나만 본다. (나)의 일반어 목록·70% 조각 기준은 고정값이라 기술 용어도 걸린다.
2. 문서 상한(단어 × 연도 5건)이라 연결 건수는 표본 수준이다. 10-K/A·10-Q/A는 부분을 절반 넘게 못 찾는다.
3. 여전히 뒤 시점 정보가 섞인 거르기가 있다(G0 한계 1: 전체 합 20, 상투어 36개).

**다음 버전 후보** (결과를 본 뒤라 이번 규칙은 바꾸지 않음):

- ① **꾸준한 성장 규칙이 CS 상투어를 등록함** — "extensive experiment", "significantly outperform", "consistently outperform", "verify the effectiveness",
  "time consuming", "perform well" 등. 논문 수가 큰 cs 분야의 문장 표현이 연 30%로 같이 커져 성장 규칙에 걸리고, 분야 필터(cs = 산업)도 통과한다.
  위처럼 "near term"·"key component" 같은 일반 영어도 연결의 30%를 차지한다.
- ② **large language model이 일반어 규칙(끝 단어 model)으로 빠짐** — 급증 2021-09 등록, 산업 비율 99%. 끝 단어 일반어 규칙이 3단어 기술 용어를 자른다.
- ③ **millimeter wave가 분야 비율 54%로 빠짐** — 성장 2017-09 등록. 물리(천문 전파 관측)와 통신이 섞인 용어라 60% 기준에 걸렸다.

## 판정

판정(사용자·Claude 2026-10-20): 연결 표본 50 정확도 22/50 = 44% < 50% (검수 전 정한 기준) → 투자 신호 연구(G1~G3) 중단, 관심도 대시보드로만 유지.
검수 기록: `docs/research/radar_g0b_review_2026-10-20.md`.
