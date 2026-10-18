# 신기술 레이더 G0 표본 검수 (Claude 분류, 2026-10-18 — 사용자 제공)

대상: `docs/research/radar_terms_sample50.csv`, `docs/research/radar_link_sample50.csv` (G0 시드 20261018 무작위 50개씩). 아래 분류를 두 CSV의 분류 칸에 줄 순서대로 넣었다.

## 판정

- 단어 표본 50: **산업 기술 18 · 조각 11 · 순수 학문 14 · 상투어 3 · 사건 4** (산업 기술 36%)
- 회사 연결 표본 50: **맞음(개발자·공급자·고객) 9 · 피해 1 · 무관 37 · 판정불가 3 → 정확도 10/50 = 20% (기준 70%) FAIL**
  - 처음 받은 합계는 무관 38·불가 2였다. EFX "end manner"(언급 0건)를 불가로 다시 본 차이이고 판정은 같다.
- 보완: `docs/research/radar_g0b_rules.md` (G0-보완 규칙, 결과 보기 전 고정)

## 단어 표본 50 — 분류 (기술 = 산업 기술 용어, 조각 = 기술 용어 조각, 학문 = 순수 학문, 상투어, 사건)

| 단어 | 분류 | 단어 | 분류 |
| --- | --- | --- | --- |
| absorption line | 학문 | adversarial example | 기술 |
| adversarial training | 기술 | age of information | 기술 |
| agent reinforcement | 조각 | attention mechanism | 기술 |
| automatic and human | 조각 | bose einstein condensate | 학문 |
| brane world scenario | 학문 | cnn based | 조각 |
| conditional gan | 기술 | confirmed case | 사건 |
| consistency loss | 기술 | convolution neural | 조각 |
| convolutional neural network | 기술 | dice score | 학문 |
| fifth generation | 기술 | gan based | 조각 |
| graph convolutional network | 기술 | head attention | 조각 |
| higgs like | 학문 | hubble tension | 학문 |
| ii weyl | 학문 | language inference | 기술 |
| large language | 조각 | magic angle | 학문 |
| mobile ad | 조각 | model achieve | 상투어 |
| near term quantum | 기술 | node representation | 기술 |
| non orthogonal multiple | 조각 | novel coronavirus | 사건 |
| novel loss | 상투어 | order topological insulator | 학문 |
| parker solar probe | 학문 | pnictide superconductor | 학문 |
| pseudo label | 기술 | reading comprehension | 기술 |
| recurrent unit | 조각 | respiratory syndrome coronavirus | 사건 |
| second wave | 사건 | seiberg witten | 학문 |
| self supervised learning | 기술 | sentence embedding | 기술 |
| superconducting property | 학문 | supervised contrastive learning | 기술 |
| trained end | 조각 | visual input | 상투어 |
| wave background | 학문 | white box attack | 기술 |

## 회사 연결 표본 50 — 분류 (줄 순서)

| # | 티커 | 단어 | 분류 | # | 티커 | 단어 | 분류 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | IBM | acute respiratory | 무관 | 26 | CB | infection rate | 무관 |
| 2 | BA | acute respiratory | 무관 | 27 | SPGI | inference algorithm | 판정불가 |
| 3 | BAC | acute respiratory syndrome | 무관 | 28 | IDCC | iot device | 공급자 |
| 4 | TTMI | acute respiratory syndrome | 무관 | 29 | PARR | long short | 무관 |
| 5 | AAL | acute respiratory syndrome | 무관 | 30 | RMBS | mobile edge | 공급자 |
| 6 | PBH | attention based | 무관 | 31 | KEYS | network function | 공급자 |
| 7 | CEVA | autonomous driving | 공급자 | 32 | ACIW | neural model | 고객 |
| 8 | NVDA | big data | 공급자 | 33~35 | MCK | non pharmaceutical | 무관 |
| 9 | ORCL | big data | 공급자 | 36 | LNC | number of death | 무관 |
| 10 | FB | blockchain technology | 개발자 | 37 | MET | number of death | 무관 |
| 11 | ATVI | character level | 무관 | 38 | INTC | related to covid | 무관 |
| 12 | AN | codes are available | 무관 | 39 | HRL | respiratory syndrome | 무관 |
| 13 | CE | codes are available | 무관 | 40 | TPR | respiratory syndrome | 무관 |
| 14 | MMM | confirmed case | 무관 | 41 | LZB | second wave | 무관 |
| 15 | PACW | continual learning | 무관 | 42 | NYSEDEI | self driving car | 피해 |
| 16 | CEVA | deep neural network | 공급자 | 43~46 | TTMI·AAL·TTMI·ALGT | (사건 단어) | 무관 |
| 17 | EFX | end manner | 판정불가 | 47 | EL | social distancing | 무관 |
| 18 | KMB | face mask | 판정불가 | 48 | MS | stay at home | 무관 |
| 19 | AAPL | fifth generation | 무관 | 49 | HPQ | white box | 무관 |
| 20 | ATK | fifth generation | 무관 | 50 | PM | world health | 무관 |
| 21 | M | global health | 무관 | | | | |
| 22 | AGN | global health | 무관 | | | | |
| 23 | AGN | global health | 무관 | | | | |
| 24 | ENDP | health authority | 무관 | | | | |
| 25 | VVI | health crisis | 무관 | | | | |
