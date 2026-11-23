# WikIR 검색 엔진 — 정보검색 기말 프로젝트

직접 구현한 **역색인(positional inverted index) 기반 검색 엔진**입니다.
`wikir/en1k` 코퍼스(위키백과 369,721개 문서)를 색인하고, BM25 / TF-IDF / Query Likelihood 랭킹과
근접도(proximity), RM3 질의 확장, 적합성 피드백, 유사 문서 탐색을 제공합니다.
웹 UI와 터미널 UI가 있고, TREC Eval(pytrec_eval)로 성능을 평가합니다.

> 색인·검색 알고리즘(토큰화 → 역색인 구축 → 점수 계산 → top-k → 확장/필터)은 모두 이 저장소의
> 코드로 구현했습니다. Elasticsearch / Lucene / Solr / Indri / Pyserini 같은 검색 라이브러리와
> 사전학습 모델은 사용하지 않습니다. ([외부 라이브러리 사용 내역](#외부-라이브러리-사용-내역) 참고)

---

## 1. 빠른 시작

```bash
cd ir_search_engine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_index.py          # 데이터 다운로드(최초 1회, 165MB) + 색인 (~1분, 메모리 ~3GB)
python web/app.py                      # 웹 UI → http://127.0.0.1:5000
python scripts/search_cli.py           # 터미널 UI
python -m pytest -q tests              # 정확성 테스트
```

* 데이터는 `ir_datasets`가 `~/.ir_datasets/`에 내려받습니다(zenodo.org 접속 필요).
* 색인은 `index/wikir_en1k/`(약 820MB)에 생성되며 제출물에서는 제외합니다(`.gitignore`).
* 발표 데모 전에 색인을 미리 만들어 두면 웹 서버는 1초 이내에 뜹니다.

실험 재현:

```bash
python scripts/tune.py                 # training split에서 파라미터 격자 탐색 → results/tuning/
python scripts/tune.py --what rm3df    # RM3 확장 용어 df 필터 2단계 탐색
python scripts/build_index.py --no-stem    # 스테밍 어블레이션용 색인
python scripts/run_experiments.py      # validation/test 평가 → results/experiments.md, runs/, metrics/
```

---

## 2. 기능 요약 (과제 요구사항 대응)

| 구분 | 요구사항 | 구현 | 위치 |
|---|---|---|---|
| 인덱싱 | Inverted Index 구축 | term → (doc, tf, positions) CSR 구조, 병렬 구축 | `irsearch/index_builder.py` |
| 인덱싱 (필수) | TF-IDF 또는 BM25 가중치 | BM25, TF-IDF(ltc 코사인), QL-Dirichlet 모두 구현 | `irsearch/searcher.py` |
| 인덱싱 (선택) | positional index | 모든 토큰 위치 저장 → 구문 검색, 근접도 점수 | `index_builder.py`, `searcher.phrase_matches` |
| 검색 | 질의 처리 | 토큰화·불용어·스테밍, 질의 문법(`"구문"`, `+필수`, `-제외`) | `irsearch/query_parser.py` |
| 검색 | 랭킹 알고리즘 | BM25 / TF-IDF cosine / Query Likelihood | `searcher.score_*` |
| 검색 (선택) | Query expansion | RM3 의사 적합성 피드백 | `searcher.feedback_model` |
| 검색 (선택) | Relevance feedback | 사용자가 체크한 관련 문서로 질의 확장 후 재검색 | 웹 UI `관련` 체크박스 |
| 검색 (선택) | 유사 문서 탐색 | More-Like-This(문서의 상위 tf·idf 용어로 검색) | `searcher.more_like_this` |
| 검색 (추가) | 근접도 랭킹 | 인접 질의어 쌍의 정확한 위치 일치를 BM25로 점수화 | `searcher.score_proximity` |
| UI | 질의 입력, 결과 표시 | 문서 ID·제목·스니펫·점수, 인덱싱 문서 수·검색된 문서 수·수행 시간 | `web/`, `scripts/search_cli.py` |
| UI (선택) | 하이라이팅 / 페이지네이션 / 필터링 | 질의어·확장어 하이라이트, 10개씩 페이지, 구문/필수/제외 필터 | `irsearch/snippets.py`, `web/templates/` |
| 평가 | TREC Eval | pytrec_eval, MAP·P@10·Recall·nDCG·MRR, 유의성 검정 | `irsearch/evaluation.py`, `scripts/` |

---

## 3. 시스템 구조

```
 ir_datasets (wikir/en1k)
        │ docs_iter()
        ▼
 ┌─────────────────┐   병렬(멀티프로세스)    ┌──────────────────────────────┐
 │ Analyzer        │ ─────────────────────▶ │ IndexBuilder                 │
 │ 소문자화·토큰화 │   배치별 (term, doc,     │ 배치 어휘 → 전역 term id      │
 │ Snowball 스테밍 │   tf, positions)        │ term 기준 안정 정렬(CSR)      │
 └─────────────────┘                         │ df·cf·문서길이·ltc 노름 계산  │
                                             └──────────────┬───────────────┘
                                                            ▼  .npy / .json / docs.bin
                                             ┌──────────────────────────────┐
  질의 ──▶ QueryParser ──▶ Searcher ◀─────── │ InvertedIndex (memory-map)   │
           ("구문", +, -)   │ BM25/TF-IDF/QL  └──────────────────────────────┘
                            │ 근접도, RM3, 피드백, 필터, top-k
                            ▼
              ┌─────────────┴──────────────┐
              ▼                            ▼
     웹 UI (Flask) / CLI           Runner → TREC run → pytrec_eval
     스니펫·하이라이트·페이지        MAP, nDCG, P@10, Recall, 유의성 검정
```

### 3.1 역색인 파일 구조 (`index/wikir_en1k/`)

| 파일 | 내용 |
|---|---|
| `vocab.json` | 용어 목록 (인덱스 = term id) |
| `offsets.npy` | term t의 포스팅 범위 `[offsets[t], offsets[t+1])` |
| `post_docs.npy`, `post_tf.npy` | 포스팅: 문서 번호(오름차순), 용어 빈도 |
| `pos_offsets.npy`, `positions.npy` | 포스팅별 토큰 위치(positional index) |
| `cf.npy`, `doclen.npy`, `doc_norm_ltc.npy` | 컬렉션 빈도, 문서 길이, TF-IDF 문서 벡터 노름 |
| `doc_ids.json`, `docs.bin`, `doc_offsets.npy` | 외부 문서 ID, 스니펫용 문서 저장소 |

색인 통계: 문서 369,721 · 용어 687,355 · 포스팅 42,181,032 · 위치 73,093,729 · 평균 문서 길이 197.7 토큰.

* 불용어도 위치와 함께 **모두 색인**하고, 불용어 제거는 질의 시점에 합니다.
  그래서 `"president of chile"` 같은 정확한 구문 검색이 가능하고, 불용어 어블레이션을 재색인 없이 할 수 있습니다.
* 질의가 불용어로만 이뤄진 경우(`the who`)에는 불용어를 그대로 사용합니다.

### 3.2 랭킹 모델

* **BM25**: `Σ_t qtf · idf(t) · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`, `idf = ln(1 + (N − df + 0.5)/(df + 0.5))`
* **TF-IDF (ltc.ltc)**: `w = (1 + ln tf) · ln(N/df)`, 문서·질의 벡터 L2 정규화 후 코사인 유사도
* **Query Likelihood (Dirichlet)**: `Σ_t qtf · ln(1 + tf/(μ·P(t|C))) + |q| · ln(μ/(dl + μ))`
* **근접도(positional)**: 인접한 질의어 쌍 (a, b)과 질의 내 간격 g에 대해 "b가 a 뒤 정확히 g번째에 등장"을 하나의
  의사 용어로 보고 BM25로 점수화(자체 df 사용)한 뒤 가중치 w로 더합니다.
* **RM3**: 1차 검색 상위 k개 문서로 `P(w|R) = Σ_D P(D|Q) · tf(w,D)/|D|`를 추정합니다.
  상위 m개 용어를 원 질의와 `λ·P(w|Q) + (1−λ)·P(w|R)`로 섞어 2차 검색합니다.
  df 비율이 큰 범용어("also", "year" 등)는 확장 후보에서 제외할 수 있습니다(`fb_max_df`).

---

## 4. 사용 방법

### 웹 UI (`python web/app.py`)

* 상단: 질의 입력창, 랭킹 모델 선택(BM25 / BM25+근접도 / BM25+RM3 / BM25+근접도+RM3 / TF-IDF / QL)
* 우측: **인덱싱된 문서 수, 검색된 문서 수, 검색 수행 시간, 랭킹 모델**
* 결과: 한 페이지에 10개씩, 문서 ID · 문서 제목 · 문서 내용(질의어 하이라이트 스니펫) · 점수
  * wikir 문서에는 제목 필드가 없어서 본문 앞 9단어를 제목으로 표시합니다.
* `전체 보기`: 문서 전문 + 하이라이트, `유사 문서 찾기`: More-Like-This
* `관련` 체크 후 **선택한 관련 문서로 재검색**: 적합성 피드백(확장 용어는 초록색으로 표시)
* 질의 문법: `"irish sea"`(구문), `+ireland`(필수), `-scotland`(제외), `-"isle of man"`(구문 제외)
* JSON API: `/api/search?q=irish+sea&model=bm25&k=10`

### 터미널 UI (`python scripts/search_cli.py`)

`:model bm25_rm3`, `:next`, `:prev`, `:doc <id>`, `:similar <id>`, `:rf <id> ...`, `:quit`

---

## 5. 평가 방법

* 데이터: training 1,444 / validation 100 / test 100 질의. qrels는 rel=2(질의가 그 문서의 제목) 질의당 1개,
  나머지는 rel=1(첫 문장에서 해당 문서로 링크)입니다. rel=0 판정은 없습니다.
* **튜닝은 training split에서만** 했습니다(목표 지표 MAP). validation은 일반화 확인용이고, test는 최종 보고에만 썼습니다.
* 지표: MAP, MRR, P@10, nDCG@10, nDCG, Recall@100, Recall@1000 (pytrec_eval = trec_eval C 코드).
  * 결과가 없는 질의도 0으로 포함해 평균을 냅니다(`trec_eval -c`와 동일).
  * 테스트에서 pytrec_eval 값과 직접 구현한 지표 계산을 교차 검증합니다.
* 유의성: 기준(BM25 tuned) 대비 paired randomization test(양측, 20,000회).
* 결과 파일: `results/runs/<split>/<config>.run.gz`(TREC run, 질의당 상위 1000),
  `results/metrics/<split>/<config>.trec_eval.txt`(trec_eval `-q` 형식), `results/qrels/<split>.qrels`.
  표준 trec_eval 바이너리로도 확인할 수 있습니다: `trec_eval -m all_trec results/qrels/test.qrels <(gunzip -c results/runs/test/bm25.run.gz)`

### 데이터셋 사용 시 주의한 점

* **질의 ID = rel=2 정답 문서 ID** (100%). 검색기는 질의 텍스트만 받으며 질의 ID를 절대 사용하지 않습니다.
* ir_datasets의 `scoreddocs`(제공된 BM25 결과)는 사용하지 않습니다.
* test split은 학습·튜닝에 쓰지 않았습니다.

---

## 6. 실험 결과

`results/experiments.md` 참고 (scripts/run_experiments.py가 생성).

---

## 외부 라이브러리 사용 내역

| 라이브러리 | 용도 | 검색 알고리즘 관련 여부 |
|---|---|---|
| `ir_datasets` | 데이터셋 다운로드/읽기 (과제에서 지정) | 아님 |
| `numpy` | 배열 연산 (색인·점수 계산 코드는 직접 작성) | 아님 (범용 수치 연산) |
| `PyStemmer` | Snowball 영어 어간 추출 (텍스트 정규화) | 아님 |
| `Flask` | 웹 UI | 아님 |
| `pytrec-eval-terrier` | trec_eval 지표 계산 (과제에서 권장한 pytrec_eval과 동일 API) | 평가 도구 |
| `pytest` | 테스트 | 아님 |

---

## 디렉터리 구조

```
ir_search_engine/
├── irsearch/            # 검색 엔진 패키지
│   ├── analyzer.py      #   토큰화·불용어·스테밍
│   ├── index_builder.py #   역색인 구축
│   ├── index.py         #   색인 로딩(memory-map), 포스팅/위치 접근
│   ├── query_parser.py  #   질의 문법
│   ├── searcher.py      #   랭킹 모델, 근접도, RM3, 피드백, 필터, top-k
│   ├── snippets.py      #   스니펫·하이라이트
│   ├── presets.py       #   튜닝된 파라미터 프리셋
│   ├── runner.py        #   질의 집합 일괄 실행
│   └── evaluation.py    #   TREC run/qrels, pytrec_eval, 유의성 검정
├── scripts/             # build_index, tune, run_experiments, search_cli
├── web/                 # Flask 앱, 템플릿, CSS
├── tests/               # 브루트포스 대조 테스트
└── results/             # 튜닝/실험 결과, TREC run, 지표 파일
```
