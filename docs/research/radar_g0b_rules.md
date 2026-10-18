# 신기술 레이더 G0-보완 — 규칙 고정 (결과 보기 전)

> 고정 2026-10-18, `research/tech-radar`. 계기: G0 표본 검수(`docs/research/radar_sample_review_2026-10-18.md`) — 단어 산업 기술 18/50,
> 회사 연결 정확도 20%(기준 70%) FAIL. 이 문서와 코드(`core/radar/g0b.py`, `data/radar/g0b_build.py`)를 커밋한 뒤에 계산한다.
> 수익 계산 없음, 판정 예산 사용 없음, 가격·수익 캐시 읽기 금지(`data/radar/guard.py`). G0 말뭉치·단어 표(코드 `08296c5`)를 그대로 쓴다.

## 1. 등록 규칙 (두 가지, 규칙 종류를 표시)

- **급증**(G0 그대로): 최근 12개월 월별 비율 평균 ≥ 직전 36개월 평균 × 5, 최근 12개월 언급 ≥ 50편.
- **꾸준한 성장**(④ 새로): 달 m에서 최근 12개월 월별 비율 평균 R1, 3년 전 같은 12개월(m−47 ~ m−36) 평균 R0.
  R0 > 0 이고 (R1 ÷ R0)^(1/3) − 1 ≥ **30%**, 그리고 최근 12개월 언급 ≥ **200편** → 등록. 처음 만족한 달.
- 둘 다 G0와 같은 후보(전체 합 20 이상, 상투어 36개 제외) 640,497개 표현에 적용한다. 한 표현이 둘 다 만족하면 두 등록 달을 모두 적는다.

## 2. 거르기 (①~③, 두 규칙 모두에 적용 — 하나라도 걸리면 제외)

- **① 분야**: 그 표현이 나온 논문(처음 등장 ~ 등록 달, 그 시점까지만) 중 **대표 분야(primary)** 가 산업 기술 분야인 비율 ≥ **60%**.
  산업 기술 분야 = `cs.*`, `eess.*`, `cond-mat.mtrl-sci`, `physics.app-ph`, `physics.optics`, `quant-ph`, `q-bio.QM` (옛 `cmp-lg`는 cs로 본다).
  두 규칙의 등록 달이 다르면 각 규칙의 등록 달까지로 따로 잰다.
- **② 조각**:
  - (가) 2단어 표현 B를 품은(앞 두 단어 또는 뒤 두 단어) **등록된** 3단어 표현 T가 있고, B가 나온 논문 중 T도 나온 논문이 **70% 넘게**면 B 제외.
    잴 기간 = 처음 ~ 두 등록 달 중 늦은 달. 같은 B를 품은 T가 여럿이면 하나라도 넘으면 제외.
  - (나) 첫 단어나 끝 단어가 일반어면 제외. 일반어(고정): based, achieve, novel, trained, end, input, model, method, proposed, propose,
    approach, framework, task, performance. (규칙 글에서 바로 나오는 결과: "large language model"은 끝 단어 model로 제외된다 — 감지율 표에 그 이유로 적는다.)
- **③ 사건**: 아래 말이 표현 안에(단어 경계) 있으면 제외(고정): coronavirus, covid, pandemic, sars, mers, respiratory syndrome, social distancing,
  lockdown, confirmed case, second wave, stay at home, outbreak, epidemic, vaccine rollout, quarantine, contact tracing, wuhan, ebola, zika, h1n1, disease spread.

## 3. 회사 연결 (⑤)

- 대상: ①~③을 통과하고 등록 달이 2001-01 ~ 2021-12인 단어(규칙 종류 무관). 검색은 G0와 같다(EDGAR 전문 검색, `"단수" OR "복수"`, 10-K·10-Q, 2001~2021).
  G0에서 이미 검색한 단어는 그 결과 캐시를 쓰고, 새로 등록된 단어만 새로 검색한다.
- 문서: G0와 같은 상한 — 티커 매칭 공시 중 단어 × 접수 연도 최대 5건(시드 20261018, G0 캐시 재사용, 없는 것만 받음).
- **부분만 본다**: 10-K는 **Item 1(Business)** ~ Item 1A(Risk Factors)/1B/Item 2 앞, 10-Q는 **Part I Item 2(MD&A)** ~ Item 3/4 앞.
  Item 1A(Risk Factors)는 넣지 않는다. 시작 표지가 여러 번 나오면(목차) 2,000자 이상인 구간 중 가장 긴 것. 못 찾으면 "부분 못 찾음"(연결 아님, 따로 집계).
- **연결 = 그 부분에서 2회 이상 언급**. 발췌는 그 부분의 첫 언급 앞뒤 200자.
- 티커·구역 매칭은 G0와 같다(접수일 기준 명단).

## 4. 감지율 (⑥, 평가 전용 — 신호에 쓰지 않음)

- 처음 20개(G0) + **보류 목록 20개(고정)**: machine translation, speech recognition, object detection, face recognition, recommender system, knowledge graph,
  smart contract, digital twin, massive mimo, millimeter wave, wireless power transfer, visible light communication, quantum key distribution,
  quantum error correction, neuromorphic computing, metal organic framework, unmanned aerial vehicle, point cloud, semantic segmentation, natural language processing.
- 규칙별(급증 / 꾸준한 성장)로 따로: 거르기 전 등록 여부·달, 거르기 뒤 남았는지, 걸렸다면 어느 거르기(①·②가·②나·③)인지, 처음 등장 달·시차.
- 정규화는 G0와 같다(소문자, 하이픈 → 띄어쓰기, 끝 단어 단수). 한 단어 4개는 G0처럼 구조상 불가.

## 5. 새 표본 (⑦)

- 등록 단어 50개(거르기 뒤, 두 규칙 합집합), 회사 연결 50건(연결된 것 중, 발췌는 사업 설명 부분), **시드 20261019**, 분류 칸 비움.
  파일: `docs/research/radar_g0b_terms_sample50.csv`, `docs/research/radar_g0b_link_sample50.csv`.

## 6. 미리 적는 한계

- ① 분야 비율은 논문의 대표 분야 하나만 본다(교차 등재 분야 무시). ② (가)의 70% 기준·(나)의 일반어 목록은 고정값이라 기술 용어도 걸릴 수 있다.
- ⑤ 문서 상한(단어 × 연도 5건)이라 "연결 건수"는 표본 수준이다. 목차·서식이 특이한 공시는 부분을 못 찾을 수 있다.
