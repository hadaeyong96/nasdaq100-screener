# 신기술 레이더 G0 — 데이터 관문 결과 (수익 계산 없음)

> 2026-10-18, 브랜치 `research/tech-radar`. 규칙은 결과 전에 `docs/research/radar_g0_rules.md`로 고정(커밋 `12b300b`·`08296c5`)했다.
> 코드 버전 `08296c5`(수집·말뭉치·단어 표·등록 실행 기록 `data/cache/arxiv/runs.jsonl`, `data/cache/radar/runs.jsonl`).
> 가격·수익 캐시는 읽지 않았다 — 모든 단계가 `data/radar/guard.py` 감사 훅을 켜고 돌았다. 판정 예산 사용 없음.

## RESULT: 관문 통과 (데이터 확보·재현성 확인) — 단, 감지율 7/20과 회사 연결의 COVID 쏠림을 한계로 적는다

## 1. arXiv 수집

| 항목 | 값 |
| --- | --- |
| 출처 | 공식 OAI-PMH `metadataPrefix=arXiv` (export.arxiv.org/oai2 → oaipmh.arxiv.org/oai 301 이동) |
| 받은 쪽 | 2,474쪽(쪽당 1,300건), 요청 간격 3초 이상, 이어받기 5구간(시간 제한·세션 종료로 끊겨도 빠진 쪽 없음) |
| 실행 시간 | 약 6.4시간(구간 합) |
| 저장한 레코드(v1 ≤ 2021-12-31) | 1,934,717 |
| 말뭉치(중복 id·1991 전 v1 뺀 뒤) | **1,934,686편**, v1 1991-01 ~ 2021-12 |
| 결측 | 제목 0 · 요약 0 · 분야 0 (0.0%) · 날짜 형식 오류 0 |
| 날짜 기준 | `<created>` = v1 제출일 (수정 버전 날짜·OAI datestamp 쓰지 않음) |

연도별 논문 수: 1991 339 · 1995 12,564 · 2000 29,926 · 2005 45,351 · 2010 67,368 · 2015 101,431 · 2018 134,614 · 2019 149,265 · 2020 172,168 · 2021 181,094.
분야별 달 표는 `exports/radar/월별_논문수.xlsx`(커밋 안 함).

## 2. 재현성

- 2015-03(v1 달)을 근사 세기 없이 처음부터 다시 세어 단어 표와 비교: **완전히 같음**(표현 146,544개, 차이 0). run `07a307071afd`.
- 해시(crc32·adler32)·시드(20261018)는 실행마다 같다. 수집은 OAI 응답 원문을 `raw/page_NNNNN.xml.gz`로 남겼다.

## 3. 달별 단어 표·급증 등록

- 2~3단어 표현(전체 합 20 이상) **640,497개**, 학술 상투어 **36개** 제외(`data/cache/radar/boilerplate.csv`).
  상투어 규칙(달의 50% 넘게 비율 1% 초과)은 "in this paper"류뿐 아니라 "black hole", "dark matter", "magnetic field" 같은 **분야 기본 용어**도 걸렀다.
- 급증 등록 **601개** (2001~2021 등록 563개).

| 등록 연도 | 수 | 상위 예시(전체 언급 수 순, 최대 10개) |
| --- | --- | --- |
| 1995 | 8 | density functional, high redshift, natural language, line of sight, globular cluster, luminosity function, absorption line, clusters of galaxy |
| 1996 | 7 | space telescope, type ia, hubble space, hubble space telescope, type iib, type iia, seiberg witten |
| 1997 | 2 | brane configuration, gauge mediated |
| 1998 | 11 | einstein condensate, bose einstein condensate, ads cft, cft correspondence, ads cft correspondence, near horizon, d3 brane, ray timing, timing explorer, ray timing explorer |
| 1999 | 4 | extra dimension, infrared galaxy, large extra, large extra dimension |
| 2000 | 6 | xmm newton, brane world, randall sundrum, non bps, noncommutative gauge, noncommutative gauge theory |
| 2001 | 13 | dark energy, control system, chandra observation, world scenario, string field theory, noncommutative space, babar detector, muon anomalous, muon anomalous magnetic, sundrum model |
| 2002 | 7 | newton observation, xmm newton observation, wave background, superconducting property, pp wave, pp wave background, penrose limit |
| 2003 | 8 | local ring, generalized parton, wmap data, microwave anisotropy, anisotropy probe, microwave anisotropy probe, wilkinson microwave, wilkinson microwave anisotropy |
| 2004 | 3 | spitzer space, spitzer space telescope, exotic baryon |
| 2005 | 1 | spin hall |
| 2006 | 2 | international linear, international linear collider |
| 2007 | 4 | graphene sheet, graphene layer, arxiv administrator, removed by arxiv |
| 2008 | 8 | bilayer graphene, based superconductor, iron based, graphene nanoribbon, layer graphene, fe based, m2 brane, unparticle physics |
| 2009 | 20 | topological insulator, fermi lat, compressed sensing, area telescope, large area telescope, massive gravity, iron based superconductor, iron pnictide, parent compound, fermi gamma |
| 2010 | 11 | fermi large, fermi large area, routing protocol, herschel space, herschel space observatory, mobile ad, mobile ad hoc, lifshitz gravity, horava lifshitz, ho ava |
| 2011 | 6 | pp collision, quantum discord, sdss dr7, herschel pac, superluminal neutrino, inverse picobarn |
| 2012 | 9 | social media, semi supervised, bayesian network, atlas and cms, inference algorithm, gev higg, field infrared, daya bay, recent lhc |
| 2013 | 13 | big data, energy harvesting, symmetry protected, inverse femtobarn, probabilistic inference, approximate inference, higgs like, belief network, gev higgs boson, expert system |
| 2014 | 12 | deep learning, convolutional neural, convolutional neural network, deep neural, metal dichalcogenide, transition metal dichalcogenide, massive mimo, black phosphorus, mode polarization, bicep2 data |
| 2015 | 16 | deep neural network, recurrent neural, recurrent neural network, convolutional network, deep convolutional, deep network, deep convolutional neural, short term memory, long short, long short term |
| 2016 | 38 | pre trained, generative adversarial, adversarial network, deep learning based, attention mechanism, representation learning, deep reinforcement, deep reinforcement learning, semantic segmentation, question answering |
| 2017 | 47 | generative adversarial network, deep learning model, autonomous driving, neural architecture, adversarial example, adversarial training, variational autoencoder, learning architecture, neural model, cnn based |
| 2018 | 71 | trained model, self supervised, graph neural, graph neural network, adversarial attack, deep learning method, self attention, meta learning, graph convolutional, low resource |
| 2019 | 73 | federated learning, transformer based, architecture search, knowledge distillation, attention module, neural architecture search, trained language, twisted bilayer, pre trained language, ml model |
| 2020 | 118 | downstream task, self supervised learning, sars cov, contrastive learning, pseudo label, reflecting surface, intelligent surface, social distancing, reconfigurable intelligent, reconfigurable intelligent surface |
| 2021 | 83 | early data, contrastive loss, compartmental model, disease spread, vision transformer, gaia edr3, non iid, early data release, transformer encoder, distillation method |

읽을 점: 2015년까지는 천체·입자 물리 관측 장비·이론이 대부분이고, 2014년 deep learning 이후 머신러닝 표현이 크게 늘어난다(2018~2021 등록의 다수).
"arxiv administrator", "removed by arxiv"(철회 논문 안내문), "ho ava"(Hořava 철자 깨짐) 같은 잡음도 등록됐다. 무작위 50개 수동 분류 칸: `docs/research/radar_terms_sample50.csv`.

## 4. 감지율 (평가 전용 — 신호에 쓰지 않음)

**20개 중 7개 등록** (2~3단어 16개 중 7개). 시차 = 처음 등장 달 → 등록 달(개월). "기준" 칸은 최근 12개월 언급 50편 이상이 된 달들의 최대 배율 / 배율 5배 이상이 된 달들의 최대 언급 수.

| 기술 | 등록 | 등록 달 | 처음 등장 | 시차 | 놓친 이유 (기준) |
| --- | --- | --- | --- | --- | --- |
| deep learning | 예 | 2014-07 | 2008-09 | 70 | — |
| convolutional neural network | 예 | 2014-10 | 2011-02 | 44 | — |
| generative adversarial network | 예 | 2017-01 | 2014-06 | 31 | — |
| reinforcement learning | 아니오 | — | 1995-01 | — | 기준 미달(배율): 오래된 분야라 50편 넘을 때 배율 최대 3.7배. 변형 "deep reinforcement learning"은 2016 등록(표현 차이) |
| large language model | 예 | 2021-09 | 2012-10 | 107 | — (등록 시점 언급 63편 — 2022 이후 폭증 전) |
| federated learning | 예 | 2019-06 | 2016-11 | 31 | — |
| internet of things | 아니오 | — | 2010-01 | — | 기준 미달(배율 3.5배 / 5배일 때 28편). 약어 "IoT"를 주로 씀 — "iot device"는 등록(표현 차이) |
| cloud computing | 아니오 | — | 2008-08 | — | 기준 미달(배율 2.7배 / 5배일 때 43편) |
| edge computing | 예 | 2017-10 | 2010-10 | 84 | — |
| software defined networking | 아니오 | — | 2012-12 | — | 기준 미달(최대 72편, 배율 3.6배 / 5배일 때 17편). 약어 "SDN"·"software defined network" 표현 차이 |
| blockchain | 아니오(한 단어) | — | 2013-11 | — | 구조상 등록 불가. 참고 규칙 적용 시 2017-06 등록(시차 43). "blockchain based" 등 2~3단어는 등록 |
| bitcoin | 아니오(한 단어) | — | 2011-12 | — | 구조상 등록 불가. 참고 규칙으로도 미등록 |
| quantum computing | 아니오 | — | 1994-12 | — | 기준 미달(배율): 1990년대부터 꾸준해 50편 넘을 때 최대 3.0배 |
| augmented reality | 아니오 | — | 1995-05 | — | 기준 미달(배율 3.3배 / 5배일 때 2편) |
| autonomous driving | 예 | 2017-10 | 2012-06 | 64 | — |
| lidar | 아니오(한 단어) | — | 1996-05 | — | 구조상 등록 불가. 참고 규칙으로도 미등록 |
| solid state battery | 아니오 | — | 2017-02 | — | 기준 미달(논문 수): 전체 40편, 12개월 최대 18편 — arXiv에 전지 연구가 적다 |
| perovskite solar cell | 아니오 | — | 2013-09 | — | 기준 미달(논문 수): 12개월 최대 48편(50 미만) |
| graphene | 아니오(한 단어) | — | 1998-04 | — | 구조상 등록 불가. 참고 규칙 적용 시 2006-08 등록(시차 100). "graphene sheet"·"bilayer graphene" 등은 2007~2008 등록 |
| additive manufacturing | 아니오 | — | 2011-06 | — | 기준 미달(배율 2.3배 / 5배일 때 36편) |

정리: 놓친 9개(2~3단어) 중 **배율 미달 7**(오래된 분야·천천히 큰 분야: 강화학습·양자컴퓨팅·클라우드·IoT·SDN·AR·적층 제조), **논문 수 미달 2**(전지·태양전지 — arXiv 밖 분야).
표현 차이(약어·변형)가 겹친 것 3(IoT, SDN, 강화학습). 감지된 7개의 처음 등장 → 등록 시차는 31~107개월(중앙값 64개월)이다 — 레이더가 "처음 나온 때"가 아니라 "몇 년 뒤 급증할 때"를 잡는다.

## 5. 10-K·10-Q 회사 연결

- 대상: 등록 달 2001-01 ~ 2021-12 단어 **563개**. EDGAR 전문 검색 요청 2,899회(초당 10회 이하, HTTP 500은 재시도), 6,634초.
- **검색 결과가 있는 단어 236개**(나머지 327개는 10-K·10-Q에 한 번도 없음 — 대부분 물리·천문 용어). 1만 건 상한에 걸린 단어 11개.
- 공시 결과 **254,853건**(공시 문서 100,366개). 티커 매칭 **55,204건 = 21.7%** (실패 78.3%), 매칭 티커 1,741개.
- 문서(언급 수·앞뒤 200자): 티커 매칭 공시 중 단어 × 연도 최대 5건 → 4,180건(문서 3,519개) 선택, 본문 확보 3,512개(HTTP 404 7개 — 2001년 옛 .txt 경로).
  언급 수가 있는 줄 4,173 중 0회 230(검색 색인과 본문 추출 차이 — 예: 표·첨부 문서). 언급 수 중앙값 1회.

**구역별 매칭** (분모 = 그 기간 접수 공시 전체 결과. "그 구역" = 접수일에 그 구역 명단에 있던 회사, "아무 티커" = 우리 명단 어디든)

| 구역 | 기간 | 공시 결과 | 그 구역 매칭 | 아무 티커 매칭 |
| --- | --- | --- | --- | --- |
| 전체 | 2001~2021 | 254,853 | — | 55,204 (21.7%) |
| 탐색(대형) 나스닥100 ∪ S&P 100 | 2011-12 ~ 2021 (명단 시작이 2011-12) | 211,711 | 6,649 (3.1%) | 51,747 (24.4%) |
| 탐색 ② 중소형 S&P 400 ∪ 600 | 2012~2016 | 23,825 | 2,657 (11.2%) | 4,730 (19.9%) |
| 검증 ① 중소형 S&P 400 ∪ 600 | 2017~2021 | 187,728 | 30,638 (16.3%) | 47,006 (25.0%) |
| (참고) 검증 ③ S&P 500 나머지 | 2012~2021 | 211,553 | 11,792 (5.6%) | 51,736 (24.5%) |

- 연도별 매칭률: 2001~2010 4~10%(명단이 S&P 500 소속 기간만), 2012~2021 19~28%.
- **쏠림**: 공시 결과의 68%(2020 88,579 + 2021 84,198)가 2020~2021년이고, 대부분 COVID 표현이다. 매칭 티커 수 상위 단어도
  "related to covid"(877)·"social distancing"(855)·"impact of covid"(800)… 순이다. 기술 단어만 보려면 COVID 단어를 따로 빼야 한다.
- 검수: 무작위 50건 `docs/research/radar_link_sample50.csv`(발췌 포함, 분류 칸 비움 — 사용자 검수). 전체 표 `data/cache/radar/ticker_term_evidence.parquet`(검수 상태 = 미검수).

## 6. 한계

1. **뒤 시점 정보가 섞인 거르기**: "전체 기간 합 20 미만 버림"과 "상투어 목록"은 1991~2021 전체 통계다. 신호로 쓰려면 그 시점까지의 자료로 다시 계산해야 한다.
2. **감지율 7/20**: 급증 규칙(5배·50편)은 갑자기 커진 ML 용어에 맞고, 오래 꾸준한 분야·arXiv 밖 분야(전지·재료·제조)는 놓친다. 한 단어 기술(블록체인·그래핀 등)은 구조상 못 잡는다.
3. **잡음 등록**: 철회 안내문("removed by arxiv"), 철자 깨짐("ho ava"), 관측 장비 이름 — 수동 분류(표본 50)가 필요하다.
4. **회사 연결**: 매칭은 우리가 가진 명단(S&P 500 1996~, 나스닥100 ∪ S&P 100·S&P 400/600 2011~)만이라 2011년 전·소형·비상장 회사는 대부분 실패한다.
   검색 결과는 "언급"일 뿐 개발자·고객·피해 구분이 없다(표본 검수 칸). EDGAR 1만 건 상한 단어 11개는 일부만 받았다.
5. 2020~2021 COVID 단어가 회사 연결을 지배한다.
6. 표현 정규화는 끝 단어 단수화만 한다(약어·동의어 통합 없음).
