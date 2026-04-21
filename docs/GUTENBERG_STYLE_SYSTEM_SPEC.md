# Gutenberg Style System Spec

Status:
- Draft, implementation-target spec
- Last updated: 2026-04-16

This document specifies the full standalone system for building a Gutenberg-based style-mimicry benchmark from scratch.

It covers:
- corpus acquisition
- corpus cleaning and indexing
- author/book eligibility filtering
- scorer-training and scorer-calibration splits
- style-scorer dataset preparation
- style-scorer training and calibration
- benchmark manifest construction
- benchmark execution and reporting
- Modal infrastructure and canonical commands

This is the upstream system spec.

The downstream benchmark behavior, metrics, schemas, and reporting rules live in [GUTENBERG_STYLE_BENCHMARK_SPEC.md](./GUTENBERG_STYLE_BENCHMARK_SPEC.md).


## Purpose

The full system exists to produce one trustworthy artifact chain:

`raw Gutenberg corpus -> cleaned indexed corpus -> trained frozen style scorer -> held-out benchmark`

That separation is mandatory.

The system MUST NOT treat:
- a tiny checked-in fixture set
- an ad hoc hand-picked author list
- or a scorer trained on benchmark authors

as equivalent to the full benchmark stack.


## Core Principle

The benchmark needs a large corpus, even if the final released benchmark split is relatively small.

Why:
- the learned style scorer needs broad exposure to many authors and books
- style is easy to confuse with topic, genre, and book identity
- hard distractors depend on a broad corpus
- calibration is much more stable when built from large held-out author/book pools

So the system MUST maintain five separate layers:

1. `corpus_all`
   The large cleaned Gutenberg corpus.
2. `scorer_train`
   The large subset used to train the style scorer.
3. `scorer_calibration`
   Held-out scorer-side authors or author/book pairs used only for calibration and scorer diagnostics.
4. `benchmark_dev` and `benchmark_test`
   Held-out benchmark authors/books used for case construction and evaluation.
5. `smoke_fixtures`
   Tiny local books and manifests for tests only.


## Relationship To The Current Repo

The current `writing7` repo already contains pieces of this system:
- Gutenberg cleaning in `standardize_training.py`
- dataset preparation in `prepare_data.py`
- contrastive training in `train_contrastive.py`
- inference in `inference_contrastive.py`
- benchmark scaffolding in `eval/`
- Modal entrypoints in `modal_app.py`

But these pieces are not yet packaged as one standalone, fully specified system.

This document defines how a new repo SHOULD be structured if it starts from zero and owns the whole pipeline.


## What A Complete Standalone Repo Must Deliver

A compliant implementation MUST provide:

1. Canonical Modal storage for the large Gutenberg corpus.
2. A deterministic corpus-cleaning and indexing pipeline.
3. Stable manifests for books, authors, passages, and splits.
4. A scorer-training pipeline that uses only `scorer_train`.
5. A scorer-calibration pipeline separate from benchmark evaluation.
6. A benchmark-manifest builder that uses only held-out benchmark targets.
7. A benchmark runner and aggregator that comply with the benchmark spec.
8. A smoke-test path that works without the full corpus or paid model APIs.


## Recommended Repository Layout

A clean standalone repo SHOULD look like this:

```text
stylebench/
├── README.md
├── modal_app.py
├── requirements.txt
├── corpus/
│   ├── __init__.py
│   ├── fetch_gutenberg.py
│   ├── clean_books.py
│   ├── metadata.py
│   ├── manifests.py
│   └── passage_pools.py
├── training/
│   ├── __init__.py
│   ├── build_scorer_dataset.py
│   ├── train_style_scorer.py
│   ├── calibrate_style_scorer.py
│   ├── scorer_schema.py
│   └── diagnostics.py
├── benchmark/
│   ├── __init__.py
│   ├── benchmark_v2.py
│   ├── build_benchmark_manifests.py
│   ├── aggregate_benchmark_results.py
│   ├── benchmark_schema.py
│   ├── style_scoring.py
│   ├── originality.py
│   ├── fluency.py
│   └── ...
├── configs/
│   ├── corpus_v1.json
│   ├── scorer_train_v1.json
│   ├── benchmark_v1.json
│   └── prompts_v1.json
├── docs/
│   ├── GUTENBERG_STYLE_SYSTEM_SPEC.md
│   ├── GUTENBERG_STYLE_BENCHMARK_SPEC.md
│   └── SETUP_STANDALONE.md
├── scripts/
│   ├── push_scorer_to_hf.py
│   └── download_scorer_from_hf.py
└── tests/
    ├── fixtures/
    └── ...
```

The key design rule is:
- `corpus/` owns the source of truth for texts and metadata
- `training/` owns scorer preparation, training, and calibration
- `benchmark/` owns only held-out benchmark logic


## Canonical Modal Infrastructure

The standalone repo SHOULD use two Modal volumes.

### 1. Corpus Volume

Name:
- `stylebench-corpus`

Mount path:
- `/corpus`

Purpose:
- raw Gutenberg fetches or mirrors
- cleaned normalized texts
- metadata indexes
- split manifests

### 2. Artifacts Volume

Name:
- `stylebench-artifacts`

Mount path:
- `/artifacts`

Purpose:
- scorer datasets
- scorer checkpoints
- calibration artifacts
- benchmark manifests
- benchmark results


## Canonical Volume Layout

Inside `/corpus`:

```text
/corpus/
├── raw/
│   └── gutenberg/
│       ├── http/
│       └── rsync/
├── clean/
│   └── gutenberg/
├── meta/
│   ├── source_index.jsonl
│   ├── books_manifest_v1.jsonl
│   ├── authors_manifest_v1.jsonl
│   ├── eligibility_v1.jsonl
│   └── passage_pools_v1/
└── splits/
    ├── scorer_train_v1.json
    ├── scorer_calibration_v1.json
    ├── benchmark_dev_v1.json
    └── benchmark_test_v1.json
```

Inside `/artifacts`:

```text
/artifacts/
├── scorer/
│   ├── datasets/
│   ├── runs/
│   ├── final/
│   ├── style_calibration_v1.json
│   └── diagnostics_v1.json
├── benchmark/
│   ├── manifests/
│   └── reference_distributions/
└── results/
    ├── benchmark_runs/
    └── summaries/
```


## Canonical Data Layers

The system MUST keep the following layers separate.

### `corpus_all`

Definition:
- all eligible cleaned Gutenberg books

Purpose:
- source pool for every later split

### `scorer_train`

Definition:
- author-disjoint subset used to train the style scorer

Purpose:
- learn authorial style signals at scale

### `scorer_calibration`

Definition:
- held-out scorer-side subset used only for:
  - calibration
  - scorer diagnostics
  - threshold selection

Purpose:
- convert raw similarity into stable interpretation

### `benchmark_dev`

Definition:
- benchmark-side authors/books used for:
  - metric selection
  - threshold tuning
  - human validation

### `benchmark_test`

Definition:
- final held-out benchmark-side authors/books used for public results

### `smoke_fixtures`

Definition:
- tiny checked-in texts and manifests

Purpose:
- local tests
- CI
- API-free smoke runs

Rule:
- `smoke_fixtures` MUST NOT be treated as canonical benchmark data


## Corpus Acquisition

The standalone repo SHOULD support two acquisition modes.

### Mode A: HTTP Fetch

Recommended for straightforward acquisition and incremental updates.

Canonical command family:

```bash
modal volume create stylebench-corpus
modal volume create stylebench-artifacts

modal run modal_app.py::corpus_fetch_gutenberg_http \
  --start-id 1 \
  --end-id 80000
```

For parallel bulk fetch:

```bash
modal run modal_app.py::corpus_fetch_all_gutenberg_http \
  --start-id 1 \
  --end-id 80000 \
  --chunk-size 500 \
  --containers 64 \
  --per-container-concurrency 24
```

Expected output:
- cleaned or raw-plus-cleaned Gutenberg texts under `/corpus/raw/gutenberg/http` or `/corpus/clean/gutenberg`
- source metadata rows appended to `/corpus/meta/source_index.jsonl`

### Mode B: RSYNC Mirror

Recommended when a fuller historical mirror is desired.

Canonical command family:

```bash
modal run modal_app.py::corpus_ingest_gutenberg_rsync \
  --max-files 50000
```

Expected output:
- mirrored raw text files under `/corpus/raw/gutenberg/rsync`
- processed cleaned books under `/corpus/clean/gutenberg`
- source metadata rows in `/corpus/meta/source_index.jsonl`

### Acquisition Policy

The system SHOULD implement both modes, but SHOULD designate one as canonical in config.

Recommended default:
- HTTP fetch as the default operational path
- RSYNC as an optional bulk-ingest path


## Corpus Cleaning And Normalization

Every acquired text MUST be normalized before it enters `corpus_all`.

Cleaning MUST:
- remove Gutenberg header/footer boilerplate
- normalize line endings
- preserve paragraph breaks
- preserve punctuation and capitalization in the body text
- reject obviously malformed or empty texts

Cleaning SHOULD:
- normalize Unicode punctuation where helpful
- strip repeated blank lines
- record language signals and source metadata

The cleaned corpus is the source of truth.

The system MUST NOT:
- build scorer datasets directly from raw fetched files
- build benchmark manifests directly from raw fetched files


## Metadata And Book Manifests

The system MUST emit a books manifest for the cleaned corpus.

Each record SHOULD include:

```json
{
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "title": "Emma",
  "author": "Jane Austen",
  "source": "gutenberg_http",
  "gutenberg_id": "158",
  "raw_path": "raw/gutenberg/http/pg158.txt",
  "clean_path": "clean/gutenberg/jane_austen/emma.txt",
  "language": "en",
  "publication_year": 1815,
  "period_bucket": "1800_1849",
  "genre": "novel",
  "is_translation": false,
  "clean_word_count": 158231,
  "clean_char_count": 913442,
  "clean_sentence_count": 7345
}
```

The repo SHOULD also emit an authors manifest with:
- `author_id`
- display name
- number of eligible books
- total clean word count
- eligibility flags


## Eligibility And Filtering

The builder MUST filter the cleaned corpus before any splits are frozen.

### Include

- English prose fiction and narrative nonfiction
- books with enough continuous text to support passage extraction
- authors with stable metadata

### Exclude

- poetry
- plays
- fragments
- severe OCR corruption
- anonymous or uncertain authors for canonical Author Track
- duplicate editions of the same work
- translations unless intentionally included in an auxiliary slice

### Recommended Thresholds

- minimum clean words per book: `40,000`
- minimum clean sentences per book: `1,500`
- minimum eligible books per canonical Author Track author: `3`


## Split Strategy

The standalone repo MUST freeze author-level splits before training or benchmarking.

Recommended split families:

- `scorer_train`
- `scorer_calibration`
- `benchmark_dev`
- `benchmark_test`

### Hard Rules

1. Benchmark authors MUST be disjoint from scorer-train authors.
2. Benchmark authors SHOULD be disjoint from scorer-calibration authors.
3. Split manifests MUST be materialized and versioned.
4. Split membership MUST be derived deterministically from a fixed seed and the cleaned eligible author list.

Recommended proportions:
- scorer_train: `70%`
- scorer_calibration: `10%`
- benchmark_dev: `10%`
- benchmark_test: `10%`


## Passage Pools

The system MUST precompute reusable passage pools from cleaned books.

Why:
- both scorer prep and benchmark manifests need stable text windows
- sentence-level non-overlap must be enforceable
- repeated on-the-fly sampling makes benchmark versions unstable

Passage pools SHOULD be stored under:
- `/corpus/meta/passage_pools_v1/`

Each passage record SHOULD include:
- `passage_id`
- `book_id`
- `author_id`
- `text`
- `start_sentence`
- `end_sentence`
- `word_count`
- `region_bucket`

The sentence splitter and passage-selection rules SHOULD match the benchmark spec.


## Scorer Dataset Preparation

The style scorer is trained on the large corpus, not the benchmark split.

The standalone repo MUST build scorer datasets from `scorer_train` only.

### Training Objective

The recommended default is a contrastive author-style embedder.

The model SHOULD:
- pull together same-author passages
- push apart different-author passages
- reduce topic leakage where practical

The current `writing7` contrastive encoder is a valid starting point, but the standalone repo SHOULD expose the scorer as a clean `training/` subsystem rather than mixing it with benchmark logic.

### Training Dataset Requirements

The scorer prep pipeline SHOULD create:
- non-overlapping same-author positives
- different-author negatives
- metadata-matched negatives where possible
- optional hard negatives from scorer-side embeddings

It SHOULD also emit:
- training manifests
- validation manifests
- diagnostic metadata for each pair or grouped sample


## Scorer Training

Canonical training artifacts MUST be written under:
- `/artifacts/scorer/runs/...`
- `/artifacts/scorer/final/`

The training pipeline SHOULD expose:

```bash
modal run modal_app.py::build_scorer_dataset \
  --corpus-manifest /corpus/meta/books_manifest_v1.jsonl \
  --split /corpus/splits/scorer_train_v1.json

modal run modal_app.py::train_style_scorer \
  --config configs/scorer_train_v1.json
```

Recommended training outputs:
- model checkpoint directory
- tokenizer files
- scorer config JSON
- training metrics JSON
- diagnostic plots or summary JSON


## Scorer Calibration

Calibration is distinct from training and distinct from benchmark evaluation.

The repo MUST support:

```bash
modal run modal_app.py::calibrate_style_scorer \
  --model-dir /artifacts/scorer/final \
  --split /corpus/splits/scorer_calibration_v1.json
```

Calibration artifacts SHOULD include:
- `style_calibration_v1.json`
- scorer-side reference distributions
- diagnostics on held-out scorer-calibration authors

The benchmark MUST consume a frozen calibrated scorer artifact, not a training checkpoint in flux.


## Benchmark Build Integration

Once the scorer is frozen, the repo MUST build benchmark manifests from `benchmark_dev` and `benchmark_test`.

Canonical command family:

```bash
modal run modal_app.py::build_benchmark_manifests \
  --benchmark-config configs/benchmark_v1.json \
  --corpus-manifest /corpus/meta/books_manifest_v1.jsonl \
  --split-root /corpus/splits \
  --out-root /artifacts/benchmark/manifests
```

This step MUST:
- select benchmark targets only from benchmark splits
- select conditioning passages
- select held-out evaluation passages
- select distractors
- emit versioned case manifests

The exact benchmark behavior is defined by [GUTENBERG_STYLE_BENCHMARK_SPEC.md](./GUTENBERG_STYLE_BENCHMARK_SPEC.md).


## Benchmark Execution

Canonical command family:

```bash
modal run modal_app.py::run_benchmark \
  --track author \
  --split test \
  --cases /artifacts/benchmark/manifests/cases_author_test_v1.jsonl \
  --model openai:gpt-4o-mini \
  --scorer-dir /artifacts/scorer/final
```

Aggregation:

```bash
modal run modal_app.py::aggregate_benchmark \
  --input /artifacts/results/benchmark_runs/author_test_model_x.jsonl \
  --out /artifacts/results/summaries/author_test_model_x.summary.json
```


## Canonical Build Order

The full repo SHOULD be operated in this order:

1. Create Modal volumes.
2. Acquire Gutenberg corpus.
3. Clean and normalize books.
4. Build books and authors manifests.
5. Filter for eligibility.
6. Freeze author-level splits.
7. Build reusable passage pools.
8. Build scorer datasets from `scorer_train`.
9. Train scorer.
10. Calibrate scorer on `scorer_calibration`.
11. Build benchmark manifests from `benchmark_dev` and `benchmark_test`.
12. Run benchmark.
13. Aggregate benchmark results.
14. Validate metrics against human dev annotations.


## Canonical Config Files

The standalone repo SHOULD version three config layers.

### `configs/corpus_v1.json`

Should contain:
- acquisition mode
- cleaning rules
- language filters
- eligibility thresholds
- split seed

### `configs/scorer_train_v1.json`

Should contain:
- base encoder
- loss/objective
- batch size
- epochs
- learning rate
- pooling
- adversary settings if used
- scorer dataset paths

### `configs/benchmark_v1.json`

Should contain:
- benchmark version
- track list
- prompt bank path
- generation profile
- originality thresholds
- prompt-adherence thresholds
- fluency thresholds


## Release Artifacts

A compliant standalone system SHOULD be able to publish:

1. `books_manifest_v1.jsonl`
2. split manifests
3. scorer final checkpoint
4. scorer calibration JSON
5. benchmark case manifests
6. benchmark reference distributions
7. benchmark result JSONL
8. benchmark summary JSON

The released benchmark SHOULD NOT require access to the entire raw corpus for normal evaluation runs, but the upstream training system does require the large corpus.


## Smoke-Test Policy

The standalone repo MUST also support a tiny local mode.

That mode SHOULD use:
- a few checked-in texts
- stub or tiny scorer fixtures
- 1 or 2 prompts
- 1 sample per case

This allows:
- CI
- schema validation
- quick regression tests

It MUST be clearly labeled non-canonical.


## Implementation-Complete Contract Appendix

This appendix is normative. A future implementation MUST follow these file contracts and deterministic algorithms unless the system version is explicitly changed.


## Normative IDs, Slugs, And Relative Paths

The standalone repo MUST use stable relative paths inside the mounted Modal volumes. All manifest paths MUST be stored relative to the mount root, not as absolute machine-specific paths.

### Slug Function

The canonical slug function MUST match the current `writing7` Gutenberg ingest logic:

1. Unicode-normalize with `NFKD`.
2. ASCII-fold by dropping non-ASCII codepoints.
3. Lowercase.
4. Replace apostrophes and hyphens with spaces.
5. Remove non-alphanumeric characters other than spaces.
6. Collapse whitespace.
7. Replace spaces with underscores.
8. Fall back to `"untitled"` if the result is empty.

Examples:
- `Jane Austen` -> `jane_austen`
- `L'Assommoir` -> `l_assommoir`
- `The Well-Beloved` -> `the_well_beloved`

### ID Rules

- `author_id = "author:" + author_slug`
- `book_id = "book:" + author_slug + ":" + title_slug`
- `work_id = "work:" + author_slug + ":" + title_core_slug`
- `source_id = "source:" + acquisition_mode + ":" + source_key`
- `passage_id = "passage:" + book_id.replace(":", "_") + ":" + start_sentence + ":" + end_sentence`

### Safe Component Clipping

Filesystem components MUST be clipped deterministically when long:

1. If a slug length is `<= max_len`, keep it unchanged.
2. Otherwise append `"__" + sha1(slug + salt)[:8]` after truncation.
3. Canonical limits:
   - `author_slug`: `80`
   - `title_slug`: `120`


## Canonical Corpus Record Schemas

The following files are authoritative and MUST be versioned.

### `/corpus/meta/source_index.jsonl`

One JSON object per acquired source file:

```json
{
  "source_id": "source:http:158",
  "source": "gutenberg_http",
  "acquisition_mode": "http",
  "gutenberg_id": "158",
  "url": "https://www.gutenberg.org/cache/epub/158/pg158.txt",
  "raw_relpath": "raw/gutenberg/http/pg158.txt",
  "clean_relpath": "clean/gutenberg/jane_austen/emma.txt",
  "header_title": "Emma",
  "header_author": "Jane Austen",
  "languages": ["english"],
  "raw_sha1": "f4b2d7f14f35f3f3d5e0d9f7136d1efaa7d4ab8d",
  "clean_sha1": "24c8f1fb5791f1948ef89f3998dcf8c4db718ef1",
  "raw_bytes": 912345,
  "clean_bytes": 845210,
  "fetched_at_utc": "2026-04-16T12:33:10Z"
}
```

Required fields:
- `source_id`
- `source`
- `acquisition_mode`
- `gutenberg_id` or `url`
- `raw_relpath`
- `clean_relpath`
- `header_title`
- `header_author`
- `raw_sha1`
- `clean_sha1`

### `/corpus/meta/books_manifest_v1.jsonl`

One JSON object per canonical cleaned book:

```json
{
  "book_id": "book:jane_austen:emma",
  "work_id": "work:jane_austen:emma",
  "author_id": "author:jane_austen",
  "author_slug": "jane_austen",
  "title_slug": "emma",
  "title": "Emma",
  "author": "Jane Austen",
  "source_ids": ["source:http:158"],
  "primary_source_id": "source:http:158",
  "clean_relpath": "clean/gutenberg/jane_austen/emma.txt",
  "language": "en",
  "languages_header": ["english"],
  "publication_year": null,
  "period_bucket": "unknown",
  "genre": "unknown",
  "is_translation": false,
  "duplicate_group_id": "work:jane_austen:emma",
  "duplicate_rank": 0,
  "clean_word_count": 158231,
  "clean_char_count": 913442,
  "clean_sentence_count": 7345,
  "alpha_char_ratio": 0.91
}
```

Required fields:
- all fields shown above except `publication_year`, which MAY be `null`

Rules:
- `source_ids` MUST contain at least one `source_id`
- `duplicate_rank = 0` denotes the canonical retained edition for that `work_id`
- `period_bucket` and `genre` MAY be `"unknown"`, but the field MUST exist

### `/corpus/meta/authors_manifest_v1.jsonl`

One JSON object per author:

```json
{
  "author_id": "author:jane_austen",
  "author_slug": "jane_austen",
  "display_name": "Jane Austen",
  "source_names": ["Jane Austen", "Austen, Jane"],
  "candidate_book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:mansfield_park",
    "book:jane_austen:persuasion"
  ],
  "eligible_book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:mansfield_park",
    "book:jane_austen:persuasion"
  ],
  "n_candidate_books": 3,
  "n_eligible_books": 3,
  "total_clean_words": 421113,
  "eligible_author_track": true,
  "eligible_book_track": true,
  "exclusion_reasons": []
}
```

### `/corpus/meta/eligibility_v1.jsonl`

One JSON object per canonical book, capturing every filter outcome:

```json
{
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "passes_language": true,
  "passes_cleaning": true,
  "passes_length": true,
  "passes_prose_heuristic": true,
  "passes_duplicate_filter": true,
  "passes_author_stability": true,
  "eligible_corpus_all": true,
  "eligible_author_track": true,
  "eligible_book_track": true,
  "duplicate_of_book_id": null,
  "exclusion_reasons": []
}
```


## Canonicalization And Duplicate-Edition Resolution

The standalone repo MUST canonicalize authors and collapse duplicate editions deterministically.

### Author Canonicalization

1. Parse header title and author from the first `400` lines when possible.
2. Build `author_slug` with the canonical slug function.
3. Group records by `author_slug`.
4. Choose `display_name` as:
   - the most frequent non-empty header author string within the slug group
   - tie-break by longest string
   - final tie-break lexicographically

This simple rule is sufficient for a first compliant implementation and is reproducible from the corpus alone.

### Title Core Normalization

`title_core_slug` MUST be derived from `title_slug` after stripping trailing edition-like markers.

Tokens removed only when they appear at the end:
- `illustrated`
- `complete`
- `abridged`
- `unabridged`
- `volume`
- `vol`
- `part`
- roman numerals
- decimal numerals

Example:
- `emma_illustrated` -> `emma`
- `ivanhoe_volume_1` -> `ivanhoe`

### Duplicate-Edition Resolution

1. Group books by `(author_slug, title_core_slug)`.
2. Rank candidate editions within each group by:
   - English language present in header before missing language
   - source filename suffix score:
     - `-0.txt`: `5`
     - `.txt.utf-8`: `4`
     - `-utf8.txt`: `4`
     - `-8.txt`: `3`
     - `.txt`: `2`
     - everything else: `1`
   - larger `clean_word_count`
   - larger `clean_char_count`
   - lexicographically smaller `clean_relpath`
3. Keep the top-ranked edition as the canonical book.
4. Mark every lower-ranked edition with:
   - `passes_duplicate_filter = false`
   - `duplicate_of_book_id = canonical_book_id`
   - `eligible_corpus_all = false`


## Eligibility Algorithm

The canonical corpus filter MUST be deterministic and MUST use the following book-level rules.

### Required Book-Level Rules

A book passes `eligible_corpus_all` only if all are true:
- language header is missing or contains `"english"`
- `clean_word_count >= 40000`
- `clean_sentence_count >= 1500`
- `alpha_char_ratio >= 0.70`
- it is not filtered as a duplicate edition
- it passes prose heuristics

### Prose Heuristics

The builder MUST compute the following from the cleaned text:

- `short_line_rate = fraction of non-empty lines with <= 8 tokens`
- `speaker_line_rate = fraction of non-empty lines matching ^[A-Z][A-Z .'-]{1,30}[.:]$`

Then:
- mark `passes_prose_heuristic = false` if `short_line_rate > 0.45`
- mark `passes_prose_heuristic = false` if `speaker_line_rate > 0.20`

This rule intentionally rejects obvious poetry and dramatic dialogue layouts without requiring external metadata.

### Track Eligibility

- `eligible_author_track = eligible_corpus_all and author has >= 3 eligible books`
- `eligible_book_track = eligible_corpus_all`


## Split Manifest Contract

Each split file under `/corpus/splits/` MUST be a single JSON object.

Example:

```json
{
  "artifact_type": "corpus_split",
  "split_version": "splits_v1",
  "split_name": "benchmark_test",
  "builder_seed": 42,
  "author_ids": ["author:jane_austen", "author:george_eliot"],
  "book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:persuasion",
    "book:george_eliot:middlemarch"
  ],
  "author_track_book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:persuasion",
    "book:george_eliot:middlemarch"
  ],
  "book_track_book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:persuasion",
    "book:george_eliot:middlemarch"
  ],
  "counts": {
    "authors": 2,
    "books": 3
  }
}
```

### Deterministic Assignment Algorithm

The split builder MUST:

1. Read `authors_manifest_v1.jsonl`.
2. Keep only authors with at least one `eligible_book_id`.
3. Sort authors by:
   - `sha1(f"{builder_seed}|{author_id}")[:16]` interpreted as an unsigned integer
   - then `author_id`
4. Assign authors by contiguous slices in this sorted order.

Default proportions:
- `scorer_train`: `0.70`
- `scorer_calibration`: `0.10`
- `benchmark_dev`: `0.10`
- `benchmark_test`: remainder

Small-corpus adjustment:
- if the corpus has at least 4 eligible authors, each non-training split MUST contain at least 1 author
- the adjustment MUST steal authors from `scorer_train` only

The split builder MUST write separate files:
- `scorer_train_v1.json`
- `scorer_calibration_v1.json`
- `benchmark_dev_v1.json`
- `benchmark_test_v1.json`


## Passage Pool Contract

Passage pools are authoritative corpus-side artifacts and MUST be written under:

- `/corpus/meta/passage_pools_v1/{author_slug}/{book_slug}.jsonl`

Each record MUST include:

```json
{
  "passage_id": "passage:book_jane_austen_emma:120:133",
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "text": "Mr. Knightley was one of the few people...",
  "start_sentence": 120,
  "end_sentence": 133,
  "start_char": 15244,
  "end_char": 16251,
  "word_count": 214,
  "char_count": 1007,
  "region_bucket": 1,
  "text_sha1": "95f2f7b275e049e55b2dd5f956126f84fd2edfde"
}
```

### Passage Pool Extraction Algorithm

The standalone repo MUST use the exact sentence splitter defined in the benchmark spec.

For each cleaned book:

1. Split into ordered sentences.
2. For every `start_sentence` from `0` to `n_sentences - 1`:
   - grow `end_sentence` until either:
     - `word_count >= 150` and `sentence_count >= 6`, or
     - `word_count > 300`, or
     - `sentence_count > 18`
3. Accept the window only if:
   - `150 <= word_count <= 300`
   - `6 <= sentence_count <= 18`
   - alphabetic characters are at least `60%` of non-whitespace characters
4. Compute:
   - `start_char` and `end_char` relative to the cleaned text
   - `region_bucket = min(4, floor(start_sentence * 5 / max(1, total_sentences)))`
5. Write every accepted window to the per-book JSONL file.

The passage pool is intentionally dense. Non-overlap is enforced later by the scorer dataset builder and benchmark case builder.


## Scorer Dataset Contract

The canonical scorer dataset MUST be built from passage pools, not from raw books.

### Authoritative Dataset Location

The standalone repo MUST write:

```text
/artifacts/scorer/datasets/
├── train_pairs_v1.jsonl
├── validation_pairs_v1.jsonl
├── test_pairs_v1.jsonl
└── scorer_dataset_meta_v1.json
```

Implementations MAY additionally materialize a Hugging Face `DatasetDict` cache, but the JSONL pair manifests above are authoritative.

### Pair Row Schema

```json
{
  "pair_id": "scorerpair:train:00000001",
  "split": "train",
  "passage1_id": "passage:book_jane_austen_emma:120:133",
  "passage2_id": "passage:book_jane_austen_persuasion:710:724",
  "text1": "Mr. Knightley was one of the few people...",
  "text2": "Sir Walter Elliot, of Kellynch Hall...",
  "label": 1,
  "pair_role": "positive_same_author_cross_book",
  "neg_type": null,
  "book1": "book:jane_austen:emma",
  "book2": "book:jane_austen:persuasion",
  "author1": "author:jane_austen",
  "author2": "author:jane_austen",
  "same_author": true,
  "same_book": false
}
```

Required fields:
- all fields shown above

### Canonical Pair Construction Algorithm

The canonical scorer objective is author-style affinity.

Therefore:
- same-author pairs are positive
- different-author pairs are negative
- same-author different-book pairs MUST NOT be labeled negative in the canonical scorer dataset

For each book in `scorer_train`:

1. Load its passage pool and deterministically rank passages by:
   - `sha1(f"{builder_seed}|{passage_id}")`
2. Take up to `20` anchor passages.
3. For each anchor:
   - create one positive pair:
     - prefer a passage from a different book by the same author
     - otherwise use a non-overlapping passage from the same book
   - create two negative pairs:
     - one `negative_matched` from a different author sharing `period_bucket` when possible, else any different author
     - one `negative_hard` from the nearest different-author book centroid among scorer-train books when available, else `negative_random`

### Internal Train/Validation/Test Assignment

After all pair rows are created:

1. Compute `row_hash = sha1(f"{builder_seed}|{pair_id}")[:16]`.
2. Sort rows by `row_hash`, then `pair_id`.
3. Assign splits by proportions:
   - `train`: `0.85`
   - `validation`: `0.10`
   - `test`: remainder

This internal split is only for scorer training diagnostics. It does not replace `scorer_calibration`.


## Scorer Training Contract

The canonical released scorer MUST use the following architecture unless the system version changes:

- base encoder: `roberta-large`
- pooling: `attn`
- projection head: enabled
- contrastive objective: `supcon`
- maximum tokens per side: `384`
- topic/content adversary: enabled
- semantic adversary model: `sentence-transformers/all-MiniLM-L6-v2`
- semantic adversary loss: `cosine`
- `adv_lambda = 0.7`
- `supcon_temperature = 0.07`
- early stopping patience: `3`

Canonical training defaults:
- epochs: `6`
- batch size per device: `32` on 1 GPU, `24` on 4 GPUs
- learning rate: `2e-5`
- warmup steps: `1000`
- gradient accumulation: `1` on 1 GPU, `2` on 4 GPUs

If a smaller model is used for smoke mode or local iteration, it MUST NOT be described as the canonical released scorer.


## Scorer Artifact Layout

The final scorer artifact MUST live at `/artifacts/scorer/final/` and MUST contain:

```text
/artifacts/scorer/final/
├── config.json
├── model.safetensors or pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.json
├── merges.txt
├── scorer_manifest.json
├── train_config.json
└── test_metrics.json
```

### `scorer_manifest.json`

```json
{
  "artifact_type": "style_scorer",
  "artifact_version": "style_scorer_v1",
  "model_name": "roberta-large",
  "pooling": "attn",
  "use_projection": true,
  "contrastive_mode": "supcon",
  "use_topic_adversary": true,
  "semantic_adversary_model": "sentence-transformers/all-MiniLM-L6-v2",
  "max_length": 384,
  "primary_score": "calibrated_or_score_0_1",
  "train_pairs_relpath": "../datasets/train_pairs_v1.jsonl",
  "validation_pairs_relpath": "../datasets/validation_pairs_v1.jsonl",
  "test_pairs_relpath": "../datasets/test_pairs_v1.jsonl",
  "system_spec_version": "2026-04-16",
  "benchmark_spec_version": "2026-04-16"
}
```


## Calibration Contract

Calibration MUST be built from `scorer_calibration`, not from benchmark targets.

### Calibration Pair File

The canonical pair file MUST be:

- `/artifacts/scorer/datasets/scorer_calibration_pairs_v1.csv`

Required CSV columns:
- `text1`
- `text2`
- `label`
- `group`
- `book1`
- `book2`
- `author1`
- `author2`
- `pair_role`
- `neg_type`

Rules:
- `label = 1` means same author
- `label = 0` means different author
- `group` MUST be `author1` for same-author rows and `book1` for different-author rows

### Calibration Artifact

The canonical artifact MUST be:

- `/artifacts/scorer/style_calibration_v1.json`

Schema:

```json
{
  "artifact_type": "style_calibration",
  "artifact_version": "style_calibration_v1",
  "style_calibration": {
    "method": "logistic",
    "coef": 5.11,
    "intercept": -2.04
  },
  "meta": {
    "n_samples": 10000,
    "num_chunks": "auto",
    "chunk_size": 14,
    "overlap": 4,
    "aggregate": "mean",
    "topk": 5,
    "max_length": 512,
    "selection_metric": "brier",
    "n_splits": 5,
    "method_requested": "auto"
  },
  "selection": {
    "scores": {
      "logistic": 0.121,
      "isotonic": 0.129
    },
    "chosen": "logistic"
  }
}
```

The repo MUST also write:
- `/artifacts/scorer/calibration_report_v1.json`
- `/artifacts/scorer/style_calibration_reliability.png`


## Modal Entrypoint Contract

A compliant standalone repo MUST expose these local entrypoints in `modal_app.py`.

### Corpus

```bash
modal run modal_app.py::corpus_fetch_gutenberg_http --start-id 1 --end-id 2000
modal run modal_app.py::corpus_fetch_all_gutenberg_http --start-id 1 --end-id 80000 --chunk-size 500 --containers 96
modal run modal_app.py::corpus_ingest_gutenberg_rsync --max-files 50000 --rsync-shards 10
modal run modal_app.py::build_corpus_manifests --corpus-config configs/corpus_v1.json
modal run modal_app.py::freeze_corpus_splits --corpus-config configs/corpus_v1.json
modal run modal_app.py::build_passage_pools --corpus-config configs/corpus_v1.json
```

### Scorer

```bash
modal run modal_app.py::build_scorer_dataset --config configs/scorer_train_v1.json
modal run modal_app.py::train_style_scorer --config configs/scorer_train_v1.json
modal run modal_app.py::calibrate_style_scorer --config configs/scorer_train_v1.json
```

### Benchmark

```bash
modal run modal_app.py::build_benchmark_manifests --benchmark-config configs/benchmark_v1.json
modal run modal_app.py::run_benchmark --track author --split test --model openai:gpt-4o-mini
modal run modal_app.py::aggregate_benchmark --input /artifacts/results/benchmark_runs/author_test_model_x.jsonl
```

### Current-Repo Crosswalk

The closest `writing7` equivalents are:
- `corpus_fetch_gutenberg_http` -> `gutenberg_fetch_http`
- `corpus_fetch_all_gutenberg_http` -> `gutenberg_fetch_all_http`
- `corpus_ingest_gutenberg_rsync` -> `gutenberg_ingest`
- `build_scorer_dataset` -> `prepare_remote_gpu`
- `train_style_scorer` -> `train_contrastive_remote_gpu` or `train_contrastive_remote_multi_gpu`
- `calibrate_style_scorer` -> `calibrate_style_similarity_remote_gpu`


## Dependency Freeze Contract

A released benchmark version is not complete until dependency versions are frozen.

The standalone repo MUST check in one of:
- `uv.lock`
- `requirements.lock.txt`

It MUST also record:
- Python version
- lockfile digest
- scorer manifest digest
- benchmark config digest

The implementation MAY choose the exact locked versions at repo-creation time, but once a benchmark version is released, the lockfile becomes part of the release contract.


## Mapping To The Current `writing7` Repo

If a future Codex is implementing the standalone repo by borrowing from `writing7`, the closest current equivalents are:

- corpus fetch over HTTP:
  - `modal_app.py::gutenberg_fetch_http`
  - `modal_app.py::gutenberg_fetch_all_http`
- corpus ingest from rsync:
  - `modal_app.py::gutenberg_ingest`
- prepare cleaned training data:
  - `modal_app.py::prepare_remote_gpu`
- train scorer:
  - `modal_app.py::train_contrastive_remote_gpu`
  - `modal_app.py::train_contrastive_remote_multi_gpu`
- benchmark layer:
  - `eval/`

Those current entrypoints are a useful source of implementation logic, but the standalone repo SHOULD adopt the cleaner separation defined by this spec.


## Acceptance Standard

The standalone repo SHOULD claim to implement the Gutenberg style system only when:

1. it can acquire and clean a large Gutenberg corpus on Modal
2. it can freeze author-disjoint scorer and benchmark splits
3. it can train and calibrate a scorer from `scorer_train`
4. it can build benchmark manifests from held-out targets
5. it can run and aggregate the benchmark using the benchmark spec
6. it can exercise the full stack in smoke mode without the large corpus
