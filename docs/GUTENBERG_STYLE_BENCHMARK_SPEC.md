# Gutenberg Style Mimicry Benchmark Spec

Status:
- Draft, implementation-target spec
- Last updated: 2026-04-16

This document is intentionally more detailed than a normal design note. It is meant to be sufficient for a future Codex model, or a human engineer, to build the benchmark system from scratch with minimal additional product decisions.

This document specifies the benchmark layer only.

For the full standalone system from raw Gutenberg acquisition through scorer training and calibration, see [GUTENBERG_STYLE_SYSTEM_SPEC.md](./GUTENBERG_STYLE_SYSTEM_SPEC.md).

The scope of this spec is the benchmark system itself:
- corpus preparation for benchmark use
- manifest building
- case construction
- generation protocol
- automatic scoring
- result aggregation
- human-validation hooks
- repository structure and public interfaces

It does not require this document to fully specify how to train the neural style scorer from first principles. It does require the benchmark repo to define a stable scorer interface and to consume a frozen scorer artifact produced by the upstream system spec.


## Purpose

The benchmark exists to answer one question:

`How good is a model at imitating a target literary style on new content without copying?`

For this benchmark, "good" means all of the following at once:
- stylistically faithful
- not copied from the references
- responsive to the content prompt
- readable enough to evaluate
- comparable across models and runs

This benchmark is focused on Gutenberg books because:
- they are abundant
- many authors have distinctive styles
- they are long enough to support held-out conditioning and evaluation splits
- they are legally practical for benchmarking work


## Normative Language

The keywords `MUST`, `MUST NOT`, `SHOULD`, `SHOULD NOT`, and `MAY` are used in the RFC sense:
- `MUST`: required for a compliant implementation
- `SHOULD`: recommended default; deviations should be justified
- `MAY`: optional

If future code disagrees with this document, this document is the source of truth unless it is explicitly revised.


## What A Complete Implementation Must Deliver

A compliant implementation of this spec MUST provide:

1. A deterministic manifest-building pipeline from cleaned Gutenberg texts to benchmark cases.
2. Two benchmark tracks:
   - `author`
   - `book`
3. Explicit scorer-train vs benchmark-target disjointness.
4. Explicit separation of:
   - conditioning passages
   - target evaluation passages
   - distractor passages
5. A canonical runner that:
   - calls a text-generation model
   - scores style against held-out targets
   - scores against distractors
   - runs originality and prompt-adherence checks
   - emits per-sample JSONL results
6. An aggregate report generator with confidence intervals.
7. A small human-eval validation path for checking whether automatic metrics align with human judgments.
8. A smoke-test path that works without external APIs by using stub generators and tiny local fixtures.


## Benchmark Definition

The benchmark construct is:

`style mimicry quality = style fidelity on novel content under originality and task-compliance constraints`

This decomposes into four measured components:

1. `Style fidelity`
   The output should resemble the target author or target book more than matched distractors.
2. `Originality`
   The output should not be a near-copy, stitched paraphrase, or named-entity transplant from the references.
3. `Prompt adherence`
   The output should actually address the requested scenario.
4. `Basic fluency`
   The output should be long enough and readable enough to evaluate.

The benchmark MUST report these components separately. A single scalar MAY be published later, but only as a derived view, never as the only score.


## Non-Goals

This benchmark is not designed to measure:
- abstract literary quality
- factual accuracy
- plot continuation quality
- safety policy behavior
- historical authenticity
- "reads like old literature" as a generic effect


## Current Repo Audit

This spec is written against the current `writing7` repo, which already contains useful building blocks:
- Gutenberg cleaning in [standardize_training.py](../standardize_training.py)
- chunking and pair construction in [prepare_data.py](../prepare_data.py)
- the contrastive style encoder in [train_contrastive.py](../train_contrastive.py)
- inference in [inference_contrastive.py](../inference_contrastive.py)
- a current proxy benchmark in [eval/benchmark_style.py](../eval/benchmark_style.py)
- reference-distribution tooling in [eval/build_reference_distributions.py](../eval/build_reference_distributions.py)

The current benchmark remains a proxy because:
- it scores generations against the same excerpt used for prompting
- it does not enforce held-out evaluation passages
- it does not compute a real anti-copy metric
- it does not compute a canonical prompt-adherence metric
- it mixes same-book discrimination and author-style affinity in ways that make score interpretation unstable

This document defines the replacement target.


## Design Principles

The benchmark MUST follow these principles:

1. `Held-out evaluation`
   The output MUST be scored against passages not shown in the prompt.
2. `Disjointness`
   Benchmark authors MUST be disjoint from style-scorer training authors for the canonical benchmark.
3. `Multi-axis measurement`
   Style, originality, prompt adherence, and fluency MUST be measured separately.
4. `Matched distractors`
   The target MUST compete against plausible distractors, not only random negatives.
5. `Determinism`
   Case manifests, prompt banks, seeds, and thresholds MUST be versioned and reproducible.
6. `Local reproducibility`
   Canonical metrics MUST NOT depend on a changing closed-model LLM judge.


## Official Tracks

The benchmark MUST implement two official tracks.

### Author Track

Question:
- Can a model imitate an author's style across works?

Canonical target definition:
- conditioning passages come from one or more books by the author
- target evaluation passages come from a different held-out book by the same author

Why this matters:
- it reduces the chance that the model is rewarded for book-specific plot or setting residue
- it is closer to the intuitive notion of "authorial style"

Eligibility:
- an author SHOULD have at least 3 eligible books for the canonical Author Track
- authors with only 2 eligible books MAY be used only in auxiliary or non-canonical evaluations

### Book Track

Question:
- Can a model imitate the voice of a specific book?

Canonical target definition:
- conditioning passages and target evaluation passages come from the same book
- they MUST be non-overlapping at the sentence level

Why this matters:
- it measures book-level narration, rhythm, diction, and local stylistic texture

Headline policy:
- the `author` track SHOULD be the main headline benchmark
- the `book` track SHOULD be reported alongside it


## Implementation Scope And Repository Layout

An implementation inside the current repo SHOULD add or standardize the following structure:

```text
writing7/
├── docs/
│   └── GUTENBERG_STYLE_BENCHMARK_SPEC.md
├── eval/
│   ├── __init__.py
│   ├── benchmark_style.py                  # Existing smoke/proxy benchmark; keep but de-emphasize
│   ├── benchmark_v2.py                     # Canonical benchmark runner
│   ├── aggregate_benchmark_results.py      # Aggregate metrics + confidence intervals
│   ├── build_benchmark_manifests.py        # Build targets, passages, cases, splits
│   ├── benchmark_schema.py                 # Dataclasses / validators / schema helpers
│   ├── benchmark_io.py                     # JSON/JSONL loading + saving helpers
│   ├── passage_sampling.py                 # Sentence splitting + passage extraction
│   ├── distractors.py                      # Distractor selection logic
│   ├── style_scoring.py                    # Target-vs-distractor scoring primitives
│   ├── originality.py                      # Copy detection metrics
│   ├── prompt_adherence.py                 # Deterministic prompt adherence metrics
│   ├── fluency.py                          # Readability / malformed / repetition checks
│   ├── llm_clients.py                      # Existing generator adapters
│   ├── human_eval/
│   │   ├── README.md
│   │   ├── build_human_eval_packets.py
│   │   └── validate_auto_metrics.py
│   └── benchmark_data/
│       ├── VERSION
│       ├── config_v1.json
│       ├── prompts_v1.json
│       ├── books_manifest.jsonl
│       ├── passages_author_v1.jsonl
│       ├── passages_book_v1.jsonl
│       ├── scorer_train_authors_v1.json
│       ├── benchmark_dev_targets_author_v1.json
│       ├── benchmark_test_targets_author_v1.json
│       ├── benchmark_dev_targets_book_v1.json
│       ├── benchmark_test_targets_book_v1.json
│       ├── cases_author_dev_v1.jsonl
│       ├── cases_author_test_v1.jsonl
│       ├── cases_book_dev_v1.jsonl
│       ├── cases_book_test_v1.jsonl
│       ├── scorer_calibration_author_v1.json
│       ├── benchmark_reference_distributions_author_v1.json
│       └── benchmark_reference_distributions_book_v1.json
├── tests/
│   ├── fixtures/
│   │   ├── books/
│   │   └── manifests/
│   ├── test_benchmark_schema.py
│   ├── test_passage_sampling.py
│   ├── test_originality.py
│   ├── test_prompt_adherence.py
│   ├── test_distractors.py
│   ├── test_benchmark_runner_smoke.py
│   └── test_aggregate_benchmark_results.py
└── models/
    └── README.md                           # Documents scorer artifact expectations
```

Notes:
- If the benchmark is later extracted to a standalone repo, this structure SHOULD be preserved under a top-level package rather than rewritten into a different architecture.
- The canonical implementation SHOULD use plain Python, `argparse`, and JSON/JSONL files rather than databases or orchestration frameworks.


## Core Data Model

The benchmark MUST materialize its data model explicitly. It MUST NOT rely on ad hoc in-memory dictionaries with undocumented keys.

The simplest compliant implementation is:
- dataclasses in `eval/benchmark_schema.py`
- validation helpers that raise descriptive exceptions
- JSON/JSONL serialization helpers

All IDs MUST be stable, lowercase, ASCII slugs.

### Identifier Rules

- `author_id`: `author:{slug}`
- `book_id`: `book:{author_slug}:{title_slug}`
- `passage_id`: `passage:{book_slug}:{start_sentence}:{end_sentence}`
- `prompt_id`: `prompt:{family}:{nn}`
- `case_id`: `case:{track}:{split}:{target_slug}:{prompt_slug}`
- `run_id`: timestamp plus model slug, for example `run:2026-04-16T12-33-10Z:gpt-4o-mini`

Slug rules:
- lowercase
- ASCII only
- spaces become `_`
- collapse repeated `_`
- strip leading and trailing `_`


### Input Book Manifest Schema

The benchmark builder SHOULD accept a JSONL manifest of cleaned or cleanable source books.

Each line MUST contain:

```json
{
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "title": "Emma",
  "author": "Jane Austen",
  "source_path": "training/austen/emma.txt",
  "gutenberg_id": "158",
  "language": "en",
  "publication_year": 1815,
  "period_bucket": "1800_1849",
  "genre": "novel",
  "is_translation": false
}
```

Required fields:
- `book_id`
- `author_id`
- `title`
- `author`
- `source_path`

Recommended fields:
- `gutenberg_id`
- `language`
- `publication_year`
- `period_bucket`
- `genre`
- `is_translation`


### Cleaned Book Record Schema

After cleaning and eligibility checks, the builder MUST emit a normalized books manifest.

Each record MUST include:

```json
{
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "title": "Emma",
  "author": "Jane Austen",
  "source_path": "training/austen/emma.txt",
  "clean_path": "eval/cache/cleaned/book_jane_austen_emma.txt",
  "gutenberg_id": "158",
  "language": "en",
  "publication_year": 1815,
  "period_bucket": "1800_1849",
  "genre": "novel",
  "is_translation": false,
  "clean_word_count": 158231,
  "clean_char_count": 913442,
  "clean_sentence_count": 7345,
  "eligible_author_track": true,
  "eligible_book_track": true,
  "exclusion_reasons": []
}
```


### Passage Record Schema

The builder MUST materialize passages. A passage is a contiguous window of sentences from a cleaned book.

Each passage record MUST include:

```json
{
  "passage_id": "passage:book_jane_austen_emma:120:133",
  "book_id": "book:jane_austen:emma",
  "author_id": "author:jane_austen",
  "track": "author",
  "role_pool": "conditioning_or_eval",
  "text": "Mr. Knightley was one of the few people...",
  "start_sentence": 120,
  "end_sentence": 133,
  "start_char": 15244,
  "end_char": 16251,
  "word_count": 214,
  "char_count": 1007,
  "region_bucket": 1
}
```

Rules:
- `start_sentence` is inclusive
- `end_sentence` is exclusive
- `region_bucket` is a coarse index used to spread passages across the book, for example `0..4`
- `text` MAY be stored inline or reconstructable from offsets, but a released benchmark SHOULD store the text directly for reproducibility


### Prompt Record Schema

The prompt bank MUST be versioned and materialized.

Each prompt record MUST include:

```json
{
  "prompt_id": "prompt:interpersonal:01",
  "family": "interpersonal",
  "text": "Write a scene in which two people who know each other well speak politely while both conceal a serious grievance.",
  "required_keywords": ["two people", "conceal", "grievance"],
  "preferred_pov": "any",
  "dialogue_expected": true,
  "target_word_range": [500, 800]
}
```

Required fields:
- `prompt_id`
- `family`
- `text`

Recommended fields:
- `required_keywords`
- `preferred_pov`
- `dialogue_expected`
- `target_word_range`


### Benchmark Target Schema

A target record identifies the thing to imitate.

Author Track target example:

```json
{
  "target_id": "author:jane_austen",
  "track": "author",
  "author_id": "author:jane_austen",
  "conditioning_book_ids": [
    "book:jane_austen:emma",
    "book:jane_austen:mansfield_park"
  ],
  "evaluation_book_id": "book:jane_austen:persuasion"
}
```

Book Track target example:

```json
{
  "target_id": "book:jane_austen:emma",
  "track": "book",
  "author_id": "author:jane_austen",
  "book_id": "book:jane_austen:emma"
}
```


### Benchmark Case Schema

A benchmark case is the unit that gets executed by the runner before sampling.

Each case MUST include:

```json
{
  "case_id": "case:author:test:jane_austen:interpersonal_01",
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "split": "test",
  "target_id": "author:jane_austen",
  "prompt_id": "prompt:interpersonal:01",
  "conditioning_passage_ids": [
    "passage:book_jane_austen_emma:120:133",
    "passage:book_jane_austen_mansfield_park:410:425",
    "passage:book_jane_austen_emma:2240:2255"
  ],
  "evaluation_passage_ids": [
    "passage:book_jane_austen_persuasion:710:724",
    "passage:book_jane_austen_persuasion:1900:1914",
    "passage:book_jane_austen_persuasion:3100:3113",
    "passage:book_jane_austen_persuasion:4500:4515"
  ],
  "distractor_target_ids": [
    "author:george_eliot",
    "author:edith_wharton",
    "author:anthony_trollope",
    "author:henry_james",
    "author:william_makepeace_thackeray"
  ],
  "distractor_passage_ids_by_target": {
    "author:george_eliot": [
      "passage:book_george_eliot_middlemarch:600:614",
      "passage:book_george_eliot_middlemarch:2400:2414",
      "passage:book_george_eliot_middlemarch:3900:3914",
      "passage:book_george_eliot_middlemarch:5100:5114"
    ]
  },
  "generation_profile_id": "leaderboard_v1",
  "sample_seeds": [11, 29, 47]
}
```

Rules:
- cases MUST be fully materialized before the benchmark runs
- the runner MUST NOT choose prompts or passages on the fly for canonical runs


### Per-Sample Result Schema

Each generated sample MUST produce one JSONL record.

Required fields:
- benchmark metadata
- case metadata
- generator metadata
- raw prompt text
- raw output text
- all metric components

Example:

```json
{
  "run_id": "run:2026-04-16T12-33-10Z:gpt_4o_mini",
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "split": "test",
  "case_id": "case:author:test:jane_austen:interpersonal_01",
  "sample_index": 0,
  "sample_seed": 11,
  "generator": {
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "model_version": "2026-04-16",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 900
  },
  "scorer": {
    "name": "contrastive_style_embedder",
    "model_dir": "models/style_embedder/final",
    "reference_distribution_version": "author_v1"
  },
  "prompt_text": "Write a scene in which two people...",
  "conditioning_texts": ["...", "...", "..."],
  "output_text": "...",
  "style_metrics": {
    "target_similarity_mean": 0.7421,
    "distractor_similarity_means": {
      "author:george_eliot": 0.6118,
      "author:edith_wharton": 0.6550
    },
    "style_win_rate_case": 1.0,
    "style_margin_case": 0.1032,
    "top1_target_case": 1,
    "mrr_case": 1.0,
    "style_percentile_case": 0.78
  },
  "originality_metrics": {
    "char_8gram_overlap_max": 0.14,
    "token_lcs_ratio_max": 0.08,
    "copy_score": 0.14,
    "copy_flag": false
  },
  "prompt_metrics": {
    "semantic_similarity_0_1": 0.63,
    "keyword_coverage": 0.67,
    "prompt_score": 0.64,
    "prompt_pass": true
  },
  "fluency_metrics": {
    "word_count": 641,
    "repetition_rate_6gram": 0.03,
    "malformed_flag": false,
    "min_length_pass": true,
    "fluency_pass": true
  },
  "valid_flags": {
    "originality_pass": true,
    "prompt_pass": true,
    "fluency_pass": true,
    "valid": true
  }
}
```


### Benchmark Config Schema

The builder MUST emit one versioned config file, for example `eval/benchmark_data/config_v1.json`.

This file is the single source of truth for benchmark-wide constants. Thresholds and model IDs SHOULD live here rather than being duplicated across scripts.

Example:

```json
{
  "benchmark_version": "gutenberg_style_v1",
  "builder_seed": 42,
  "tracks": ["author", "book"],
  "generation_profiles": {
    "leaderboard_v1": {
      "temperature": 0.8,
      "top_p": 0.95,
      "max_tokens": 900,
      "n_samples_per_case": 3,
      "sample_seeds": [11, 29, 47]
    }
  },
  "passage_policy": {
    "min_words": 150,
    "max_words": 300,
    "preferred_words": 220,
    "min_sentences": 6,
    "max_sentences": 18,
    "region_buckets": 5
  },
  "originality_thresholds": {
    "char_8gram_overlap_max": 0.30,
    "token_lcs_ratio_max": 0.20,
    "joint_char_overlap_threshold": 0.20,
    "joint_lcs_threshold": 0.15
  },
  "prompt_adherence": {
    "encoder_model_id": "local_prompt_encoder_v1",
    "semantic_weight": 0.85,
    "keyword_weight": 0.15,
    "prompt_pass_threshold": 0.45
  },
  "fluency_thresholds": {
    "min_words_valid": 350,
    "max_words_valid": 1000,
    "max_repetition_rate_6gram": 0.20
  },
  "bootstrap_resamples": 1000
}
```


## Corpus Curation Rules

### Include

- English prose fiction and narrative nonfiction
- books with enough continuous text to yield stable passage windows
- authors with stable metadata
- books with sufficiently low OCR noise after cleaning

### Exclude

- poetry
- plays
- fragmentary texts
- heavily corrupted OCR
- anonymous or multi-author works for the canonical Author Track
- suspected translations unless explicitly marked and intentionally included in an auxiliary slice
- duplicate editions of the same work

### Minimum Length Rules

Recommended defaults:
- minimum clean word count per book: `40,000`
- minimum clean sentence count per book: `1,500`
- minimum books per canonical Author Track author: `3`

These values MAY be raised after corpus inspection, but SHOULD be stable within a benchmark version.


## Text Cleaning And Normalization

The builder MUST clean all source books before sentence splitting.

Cleaning SHOULD:
- remove Project Gutenberg headers and footers
- normalize Unicode quotes and dashes where helpful
- normalize line endings
- preserve paragraph breaks
- collapse excessive blank lines
- preserve punctuation and capitalization inside the body text

The simplest compliant path is:
- reuse `standardize_training.py` for boilerplate removal
- emit cleaned text files into a benchmark cache directory


## Sentence Splitting Contract

The benchmark MUST use one canonical sentence splitter for both:
- passage extraction
- any sentence-index-based non-overlap rules

It SHOULD NOT use one splitter in data prep and a different one in benchmark execution.

Recommended implementation:
- a deterministic regex-based segmenter in `eval/passage_sampling.py`
- protect a small canonical abbreviation list before splitting
- split on `.`, `?`, `!` followed by optional closing quotes/brackets and whitespace
- merge fragments shorter than 30 characters into neighboring sentences

Minimum abbreviation list:
- `mr.`
- `mrs.`
- `ms.`
- `dr.`
- `st.`
- `prof.`
- `jr.`
- `sr.`
- `vs.`
- `etc.`
- `e.g.`
- `i.e.`

If a future implementation prefers spaCy, it MAY do so only if:
- the exact model and version are pinned
- the benchmark version is bumped if outputs change materially


## Passage Extraction Algorithm

Passages MUST be contiguous sentence windows.

Recommended defaults:
- target word range per passage: `150` to `300`
- preferred target center: about `220` words
- minimum sentence count per passage: `6`
- maximum sentence count per passage: `18`

Canonical extraction algorithm:

1. Split cleaned book into ordered sentences.
2. Walk through the sentence list and greedily form candidate windows whose word counts fall within `150..300`.
3. Reject windows that:
   - contain too little alphabetic content
   - are dominated by chapter headings or front matter
   - have fewer than 6 sentences unless the book is unusually sparse
4. Assign each window a `region_bucket` by dividing the book into 5 equal-length sentence regions.
5. Store all candidate passages in a passage pool for that book.

Passage pools SHOULD be built once and reused. The runner SHOULD NOT resample passages from raw books on each run.


## Non-Overlap Rules

These rules are mandatory.

1. Conditioning passages MUST NOT overlap each other at the sentence level.
2. Evaluation passages MUST NOT overlap conditioning passages at the sentence level.
3. Distractor passages MUST NOT be drawn from the target author in the Author Track.
4. Book Track stress tests MAY include same-author distractors, but the canonical Book Track SHOULD still include different-author distractors.
5. When selecting multiple passages from the same book for one case, there SHOULD be at least a 3-sentence gap between windows.


## Deterministic Selection Rules

The benchmark builder MUST be deterministic.

Recommended policy:
- use one global `builder_seed`, default `42`
- use `random.Random(builder_seed)` for top-level author shuffling
- derive per-target or per-case seeds from a stable hash of:
  - `benchmark_version`
  - `track`
  - `target_id`
  - `prompt_id` where applicable

Recommended derived-seed rule:

`derived_seed = int(sha256(f"{benchmark_version}|{track}|{target_id}|{prompt_id}|{builder_seed}").hexdigest()[:8], 16)`

This avoids accidental changes when the order of unrelated targets changes.


## Case Enumeration

Canonical case manifests MUST be explicit Cartesian products.

For each released track and split:
- take every released target in that track and split
- pair it with every released prompt in the canonical prompt bank
- attach the fixed generation profile and sample seeds

In other words:

`released_cases = released_targets x released_prompts`

This rule is important because it makes benchmark coverage legible and prevents quiet changes to prompt sampling policy.


## Deterministic Passage Selection For Cases

When converting a target and passage pool into a concrete case, the builder SHOULD use one deterministic policy.

Recommended v1 policy:

1. Partition each book's candidate passages by `region_bucket`.
2. For conditioning passages:
   - prefer distinct region buckets
   - within each bucket, rank passages by closeness to preferred word count
   - break ties with the per-target derived seed
3. For evaluation passages:
   - prefer one passage per distinct region bucket
   - avoid any bucket already used by conditioning when the track allows it
   - break ties with the per-target derived seed
4. If a required region bucket is empty:
   - back off to the nearest non-empty bucket not already used
5. Once chosen, passage IDs MUST be written into the case manifest and never resampled for the same benchmark version


## Split Strategy

The benchmark MUST maintain three disjoint layers:
- `scorer_train`
- `benchmark_dev`
- `benchmark_test`

### Hard Rule

For canonical benchmark reporting:
- authors in `benchmark_dev` and `benchmark_test` MUST be disjoint from `scorer_train` authors

This rule applies even to the Book Track. If the scorer has seen the benchmark author during training, the benchmark partly becomes a memorization test.


## Author Track Construction

The canonical Author Track SHOULD be built as follows:

1. Collect eligible authors with at least 3 eligible books.
2. Shuffle authors with a fixed seed.
3. Split authors into:
   - scorer-train authors
   - benchmark-dev authors
   - benchmark-test authors
4. For each benchmark author:
   - choose 2 books for conditioning
   - choose 1 different book for evaluation
   - harvest 3 conditioning passages total
   - harvest 4 evaluation passages total from the evaluation book

Recommended author split proportions:
- scorer-train: `70%`
- benchmark-dev: `15%`
- benchmark-test: `15%`

Recommended conditioning layout:
- 1 passage from conditioning book A
- 1 passage from conditioning book B
- 1 additional passage from either A or B, from a different region bucket

Recommended evaluation layout:
- 4 passages from the held-out evaluation book
- one passage from each of 4 distinct region buckets when possible


## Book Track Construction

The canonical Book Track SHOULD be built as follows:

1. Use books from benchmark authors only if author disjointness from scorer training is preserved.
2. For each benchmark book:
   - draw 3 conditioning passages from distinct region buckets
   - draw 4 evaluation passages from different region buckets
3. Ensure sentence-level non-overlap.

Recommended split policy:
- split at `author_id`, not merely `book_id`

This makes the Book Track harder but cleaner.


## Distractor Selection

Distractors MUST be plausible confounders.

Each case SHOULD have exactly 5 distractor targets:
- 3 `matched` distractors
- 2 `hard` distractors

### Matched Distractors

Matched distractors SHOULD be selected by metadata closeness:
- same `period_bucket` when possible
- same `genre` when possible
- similar passage length bucket

Suggested metadata score:

`meta_score = 2*(same_period) + 2*(same_genre) + 1*(same_length_bucket)`

Rank by descending `meta_score`, then break ties with seeded randomness.

### Hard Distractors

Hard distractors SHOULD come from metadata-matched candidates that are also stylistically close under the frozen scorer.

Suggested algorithm:

1. Build one profile embedding per distractor candidate by mean-pooling scorer embeddings over its conditioning candidate passages.
2. Build the same profile embedding for the target.
3. Select the nearest candidates by cosine similarity among metadata-matched candidates not already chosen.

### Distractor Passage Layout

For each distractor target:
- sample 4 evaluation passages
- mirror the target's passage count and approximate word lengths


## Prompt Bank

The prompt bank MUST be versioned, fixed, and shared across models.

### Prompt Design Requirements

Prompts MUST:
- induce new content
- be broad enough to work across many authors
- support narrative prose rather than only exposition
- avoid bizarre modern constraints unless explicitly part of an auxiliary slice

Prompts SHOULD:
- be interpretable without specialized knowledge
- avoid pushing toward one narrow genre
- encourage scene-level writing, not only summaries

### Prompt Families

The canonical v1 prompt bank SHOULD contain at least 16 prompts across these 8 families:
- interpersonal
- travel
- memory
- mystery
- social_conflict
- setting
- domestic_tension
- ambition_failure

### Canonical v1 Prompt List

This list is sufficient for a first compliant implementation.

```json
[
  {
    "prompt_id": "prompt:interpersonal:01",
    "family": "interpersonal",
    "text": "Write a scene in which two people who know each other well speak politely while both conceal a serious grievance.",
    "required_keywords": ["two", "people", "conceal", "grievance"]
  },
  {
    "prompt_id": "prompt:interpersonal:02",
    "family": "interpersonal",
    "text": "Write a scene in which a trusted companion asks for help but withholds the most important fact.",
    "required_keywords": ["companion", "help", "withholds", "fact"]
  },
  {
    "prompt_id": "prompt:travel:01",
    "family": "travel",
    "text": "Write a scene of arrival in an unfamiliar place where the traveler first notices something that does not fit the expected order of things.",
    "required_keywords": ["arrival", "unfamiliar", "traveler", "notice"]
  },
  {
    "prompt_id": "prompt:travel:02",
    "family": "travel",
    "text": "Write a journey scene in which a delay forces the characters to observe one another more closely than they intended.",
    "required_keywords": ["journey", "delay", "observe", "characters"]
  },
  {
    "prompt_id": "prompt:memory:01",
    "family": "memory",
    "text": "Write a reflective scene in which an ordinary object brings back a memory the narrator would rather leave untouched.",
    "required_keywords": ["object", "memory", "narrator", "untouched"]
  },
  {
    "prompt_id": "prompt:memory:02",
    "family": "memory",
    "text": "Write a scene in which a character recalls an earlier promise and begins to understand it differently.",
    "required_keywords": ["character", "recalls", "promise", "differently"]
  },
  {
    "prompt_id": "prompt:mystery:01",
    "family": "mystery",
    "text": "Write a scene in which a small inconsistency leads a character to suspect that an important truth has been hidden.",
    "required_keywords": ["inconsistency", "suspect", "truth", "hidden"]
  },
  {
    "prompt_id": "prompt:mystery:02",
    "family": "mystery",
    "text": "Write a scene in which a letter or message is received and its tone alarms the recipient before its meaning is fully clear.",
    "required_keywords": ["letter", "message", "alarms", "recipient"]
  },
  {
    "prompt_id": "prompt:social_conflict:01",
    "family": "social_conflict",
    "text": "Write a public social scene in which embarrassment spreads because one person fails to follow an expected custom.",
    "required_keywords": ["public", "social", "embarrassment", "custom"]
  },
  {
    "prompt_id": "prompt:social_conflict:02",
    "family": "social_conflict",
    "text": "Write a scene in which a conversation about manners or duty becomes a disguised argument about power.",
    "required_keywords": ["conversation", "duty", "argument", "power"]
  },
  {
    "prompt_id": "prompt:setting:01",
    "family": "setting",
    "text": "Write a scene in which the physical setting gradually reveals the emotional state of the person moving through it.",
    "required_keywords": ["setting", "reveals", "emotional", "person"]
  },
  {
    "prompt_id": "prompt:setting:02",
    "family": "setting",
    "text": "Write a descriptive scene in which weather or landscape changes the course of a human decision.",
    "required_keywords": ["descriptive", "weather", "landscape", "decision"]
  },
  {
    "prompt_id": "prompt:domestic_tension:01",
    "family": "domestic_tension",
    "text": "Write a domestic scene in which a routine task is interrupted by news that no one is ready to discuss plainly.",
    "required_keywords": ["domestic", "routine", "news", "discuss"]
  },
  {
    "prompt_id": "prompt:domestic_tension:02",
    "family": "domestic_tension",
    "text": "Write a household scene in which a minor disagreement exposes a much larger unease.",
    "required_keywords": ["household", "disagreement", "larger", "unease"]
  },
  {
    "prompt_id": "prompt:ambition_failure:01",
    "family": "ambition_failure",
    "text": "Write a scene in which a character must decide whether to persist in a plan that is clearly beginning to fail.",
    "required_keywords": ["character", "decide", "plan", "fail"]
  },
  {
    "prompt_id": "prompt:ambition_failure:02",
    "family": "ambition_failure",
    "text": "Write a scene in which a person receives an opportunity that appears favorable but carries an unspoken cost.",
    "required_keywords": ["opportunity", "favorable", "cost", "unspoken"]
  }
]
```

This list MAY be expanded in future benchmark versions, but the exact list for a released version MUST remain fixed.


## Generation Protocol

The benchmark runner MUST use a fixed generation profile for canonical reporting.

### Canonical Generation Profile

Recommended `leaderboard_v1` profile:
- `temperature = 0.8`
- `top_p = 0.95`
- `max_tokens = 900`
- `n_samples_per_case = 3`
- `sample_seeds = [11, 29, 47]`

If a provider does not support seeding:
- record that fact explicitly
- still run the benchmark
- do not label the run as strictly deterministic

### Prompt Template

The canonical runner SHOULD use a system prompt plus a user prompt.

System prompt:

```text
You are a careful creative writer. Imitate stylistic features of the reference passages, including voice, rhythm, syntax, diction, and paragraph movement, while writing entirely new content. Do not reuse names, places, quotations, or distinctive phrases from the references. Do not mention the references or the benchmark. Output only the new prose.
```

User prompt template:

```text
You will receive style references and a content prompt.

Write an original passage in prose.

Requirements:
- Follow the content prompt.
- Match the style of the references.
- Do not copy phrases, names, places, or plot specifics from the references.
- Keep the response between 500 and 800 words.

STYLE REFERENCES
[Reference 1]
{conditioning_text_1}

[Reference 2]
{conditioning_text_2}

[Reference 3]
{conditioning_text_3}

CONTENT PROMPT
{prompt_text}
```

Rules:
- conditioning passages MUST appear in a stable order
- the prompt MUST be stored in each result record
- the raw output text MUST be stored in each result record


## Style Scorer Contract

The benchmark depends on a frozen style scorer but SHOULD NOT hard-code one architecture into the benchmark logic.

`eval/style_scoring.py` MUST expose a scorer adapter with, at minimum, the following conceptual interface:

```python
class StyleScorer:
    def score_pair(self, text1: str, text2: str) -> dict: ...
    def score_many(self, hypothesis: str, references: list[str]) -> list[dict]: ...
    def embed(self, text: str) -> list[float] | object: ...
```

Required `score_pair` fields:
- `cosine`
- `score_0_1`
- `calibrated` if available
- `pairs`
- `aggregate`

### Canonical Scorer Source

For this repo, the default adapter SHOULD wrap `ContrastiveBookMatcherInference` from [inference_contrastive.py](../inference_contrastive.py).

### Scorer Eligibility Rules

For canonical runs, the scorer MUST:
- be trained only on `scorer_train` authors
- be frozen before benchmark evaluation
- use a versioned calibration artifact

### Group Similarity Definition

For a hypothesis `h` and a reference group `R = {r1, r2, ..., rn}`:

`group_similarity(h, R) = mean(score_pair(h, ri).primary_score for ri in R)`

Where `primary_score` is:
- `calibrated` if available
- otherwise `score_0_1`

The benchmark MUST use the same `primary_score` consistently throughout a run.


## Style Metrics

Let:
- `T` be the target evaluation passage group
- `D1..Dk` be the distractor passage groups
- `sT = group_similarity(output, T)`
- `sDj = group_similarity(output, Dj)`

The benchmark MUST compute the following per sample.

### Target-vs-Distractor Win Rate

Case-level win rate:

`style_win_rate_case = mean(1 if sT > sDj else 0.5 if sT == sDj else 0 for each distractor j)`

### Style Margin

`style_margin_case = sT - mean(sDj for all distractors j)`

### Target Rank

Let `scores = [sT, sD1, ..., sDk]`.

Define:
- `top1_target_case = 1` if `sT` is the highest score else `0`
- `rank_target_case = 1 + number of distractors with score > sT`
- `mrr_case = 1 / rank_target_case`

### Style Percentile

The benchmark SHOULD compute:

`style_percentile_case = percentile_of(sT, same-track reference distribution)`

This is an interpretation metric, not the only style metric.


## Originality Metrics

The canonical benchmark MUST detect copying locally and deterministically.

It MUST NOT rely only on instructions such as "do not copy."

### Normalization For Copy Detection

Before originality scoring:
- lowercase
- replace all runs of whitespace with a single space
- strip leading and trailing whitespace
- replace punctuation with spaces for token-based overlap calculations

### Required Originality Metrics

#### 1. Character 8-gram Overlap

For each conditioning passage `c`:

`char_8gram_overlap(h, c) = |G8(h) ∩ G8(c)| / max(1, |G8(h)|)`

Where `G8(x)` is the set of character 8-grams of normalized text `x`.

Store:
- `char_8gram_overlap_max = max over conditioning passages`

#### 2. Token LCS Ratio

For each conditioning passage `c`:

`token_lcs_ratio(h, c) = LCS(tokens(h), tokens(c)) / max(1, len(tokens(h)))`

Store:
- `token_lcs_ratio_max = max over conditioning passages`

The implementation MAY use a dynamic-programming LCS or an efficient approximate version for long texts, but the exact choice MUST be documented and stable within a benchmark version.

### Optional Originality Metrics

These are useful but not required for a first compliant implementation:
- named-entity reuse count
- minhash similarity
- repeated rare-phrase overlap

### Copy Score And Copy Flag

Recommended v1 copy score:

`copy_score = max(char_8gram_overlap_max, token_lcs_ratio_max)`

Recommended v1 copy flag:
- `copy_flag = True` if `char_8gram_overlap_max >= 0.30`
- or if `token_lcs_ratio_max >= 0.20`
- or if both `char_8gram_overlap_max >= 0.20` and `token_lcs_ratio_max >= 0.15`

Recommended v1 originality pass:
- `originality_pass = not copy_flag`

Thresholds SHOULD be validated on the human dev set and stored in benchmark config, not scattered through the codebase.


## Prompt Adherence Metrics

The canonical benchmark MUST use a deterministic prompt-adherence metric.

It MAY additionally use an LLM judge for audits, but the official score MUST come from a frozen local metric.

### Required Inputs

Each prompt SHOULD provide:
- prompt text
- optional `required_keywords`

### Canonical v1 Prompt Metric

The benchmark SHOULD compute:

1. `semantic_similarity_0_1`
   - cosine similarity between frozen prompt-encoder embeddings for prompt text and output text
   - mapped to `0..1`

2. `keyword_coverage`
   - fraction of `required_keywords` whose lowercase forms appear in the normalized output
   - if the prompt has no `required_keywords`, set this to `1.0`

3. `prompt_score`

Recommended formula:

`prompt_score = 0.85 * semantic_similarity_0_1 + 0.15 * keyword_coverage`

4. `prompt_pass`

Recommended default:

`prompt_pass = prompt_score >= 0.45`

The prompt encoder model ID MUST be versioned in benchmark config.


## Fluency And Readability Metrics

The benchmark SHOULD keep these lightweight and deterministic.

Required v1 checks:

1. `word_count`
2. `min_length_pass`
   - recommended valid range: `350..1000` words
3. `repetition_rate_6gram`
   - fraction of repeated 6-grams among all 6-grams
4. `malformed_flag`
   - true if the output is empty, obviously truncated, contains tool/API error markers, or has too little alphabetic content
5. `fluency_pass`
   - true if `min_length_pass` and not `malformed_flag` and `repetition_rate_6gram < 0.20`


## Valid Output Definition

A sample is `valid` only if all of the following are true:
- `originality_pass`
- `prompt_pass`
- `fluency_pass`

The benchmark MUST report style metrics:
- over all outputs
- over valid outputs only


## Headline Metrics

The recommended headline metric tuple for public reporting is:
- `style_win_rate_valid`
- `mean_style_margin_valid`
- `top1_target_accuracy_valid`
- `originality_pass_rate`
- `prompt_pass_rate`
- `valid_rate`

The benchmark SHOULD also report:
- `mrr_valid`
- `style_percentile_valid_mean`
- `sample_count`
- `valid_sample_count`

The benchmark SHOULD NOT optimize development to one opaque composite score.


## Reference Distributions And Calibration

The benchmark SHOULD maintain two different artifacts:

1. `scorer_calibration`
   - maps raw scorer outputs to a consistent `0..1` style-affinity scale
   - built from held-out same-author vs different-author pairs from scorer-train authors only

2. `benchmark_reference_distributions`
   - used to interpret target similarity scores as percentiles
   - built from benchmark-like held-out comparisons

The existing [eval/build_reference_distributions.py](../eval/build_reference_distributions.py) is a useful starting point, but the canonical artifact SHOULD be benchmark-track-specific and versioned.


## Benchmark Runner Algorithm

`eval/benchmark_v2.py` MUST implement the following execution flow.

### Inputs

Required inputs:
- case manifest path
- prompt bank path
- style scorer config
- generator model ID
- output JSONL path

Optional inputs:
- concurrency
- limit on number of cases
- resume mode
- stream print mode

### Canonical Execution Steps

1. Load prompt bank, case manifest, and benchmark config.
2. Load the scorer adapter and reference distributions.
3. Resolve conditioning, evaluation, and distractor passage texts.
4. For each case:
   - build the canonical prompt
   - generate exactly `n_samples_per_case`
   - store raw prompts and raw outputs
5. For each sample output:
   - score style against target evaluation passages
   - score style against each distractor target's evaluation passages
   - compute style metrics
   - run originality metrics against conditioning passages
   - run prompt-adherence metrics against the prompt
   - run fluency checks
   - compute final valid flags
6. Write one JSON object per sample to the results JSONL.
7. After all cases complete, optionally run aggregation into a summary JSON.

### Resume Behavior

The runner SHOULD support resume mode:
- if a result for `(case_id, sample_index)` already exists in the output JSONL, it SHOULD be skipped unless `--overwrite` is passed

### Error Handling

If generation fails:
- store the failure in the result record
- mark `malformed_flag = true`
- mark the sample invalid
- continue the run unless `--fail-fast` is enabled


## CLI Contracts

The benchmark system SHOULD expose the following command-line entrypoints.

### 1. Build Manifests

```bash
python -m eval.build_benchmark_manifests \
  --books-manifest eval/benchmark_data/books_input.jsonl \
  --out-dir eval/benchmark_data \
  --benchmark-version gutenberg_style_v1 \
  --seed 42
```

Required behavior:
- clean books
- filter for eligibility
- build passage pools
- split authors
- build target manifests
- build case manifests
- write a config JSON

### 2. Run Canonical Benchmark

```bash
python -m eval.benchmark_v2 \
  --track author \
  --split test \
  --cases eval/benchmark_data/cases_author_test_v1.jsonl \
  --prompts eval/benchmark_data/prompts_v1.json \
  --model openai:gpt-4o-mini \
  --model-dir models/style_embedder/final \
  --out results/author_test_gpt4omini.jsonl
```

Required arguments:
- `--track`
- `--split`
- `--cases`
- `--prompts`
- `--model`
- `--model-dir`
- `--out`

Recommended optional arguments:
- `--temperature`
- `--top-p`
- `--max-tokens`
- `--concurrency`
- `--limit`
- `--resume`
- `--stream-print`

### 3. Aggregate Results

```bash
python -m eval.aggregate_benchmark_results \
  --input results/author_test_gpt4omini.jsonl \
  --out results/author_test_gpt4omini.summary.json
```

Required behavior:
- compute global means
- compute medians
- compute per-prompt breakdowns
- compute per-target breakdowns
- compute bootstrap confidence intervals

### 4. Build Human-Eval Packets

```bash
python -m eval.human_eval.build_human_eval_packets \
  --input results/author_dev_multiple_models.jsonl \
  --out eval/human_eval/dev_packets_v1.jsonl
```

### 5. Validate Automatic Metrics

```bash
python -m eval.human_eval.validate_auto_metrics \
  --human-annotations eval/human_eval/annotations_v1.jsonl \
  --benchmark-results results/author_dev_multiple_models.jsonl
```


## Public Module Contracts

The following module responsibilities SHOULD be explicit.

### `eval/benchmark_schema.py`

Must define:
- dataclasses or typed objects for books, passages, prompts, targets, cases, and result records
- validation helpers

### `eval/passage_sampling.py`

Must define:
- canonical sentence splitter
- passage window extraction
- non-overlap validation

### `eval/distractors.py`

Must define:
- metadata matching
- hard distractor selection

### `eval/style_scoring.py`

Must define:
- scorer adapter
- group scoring
- target-vs-distractor metrics

### `eval/originality.py`

Must define:
- normalization for copy detection
- char n-gram overlap
- token LCS ratio
- copy flag logic

### `eval/prompt_adherence.py`

Must define:
- frozen local prompt encoder adapter
- keyword coverage
- prompt score formula

### `eval/fluency.py`

Must define:
- word count
- repetition rate
- malformed detection
- final fluency pass logic

### `eval/benchmark_v2.py`

Must define:
- benchmark runner
- CLI entrypoint

### `eval/aggregate_benchmark_results.py`

Must define:
- aggregation functions
- bootstrap confidence intervals
- summary JSON writer


## Aggregate Report Schema

The aggregate summary JSON SHOULD include:

```json
{
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "split": "test",
  "generator": {
    "provider": "openai",
    "model_name": "gpt-4o-mini"
  },
  "sample_count": 480,
  "valid_sample_count": 401,
  "metrics_all": {
    "style_win_rate_mean": 0.71,
    "style_margin_mean": 0.082,
    "top1_target_accuracy_mean": 0.56,
    "mrr_mean": 0.73,
    "originality_pass_rate": 0.91,
    "prompt_pass_rate": 0.88,
    "valid_rate": 0.84
  },
  "metrics_valid": {
    "style_win_rate_mean": 0.79,
    "style_margin_mean": 0.101,
    "top1_target_accuracy_mean": 0.63,
    "mrr_mean": 0.80
  },
  "bootstrap_ci_95": {
    "style_win_rate_valid": [0.75, 0.83],
    "style_margin_valid": [0.091, 0.111]
  },
  "by_prompt_family": {
    "interpersonal": {
      "style_win_rate_valid": 0.82,
      "valid_rate": 0.87
    }
  },
  "by_target": {
    "author:jane_austen": {
      "style_win_rate_valid": 0.91,
      "valid_rate": 0.89
    }
  }
}
```

Required aggregation policy:
- aggregate over sample records
- also compute target-level and prompt-level summaries
- bootstrap over benchmark cases or `(target, prompt)` units, not raw tokens

Recommended default:
- `1000` bootstrap resamples


## Versioning Policy

The benchmark MUST be versioned.

`benchmark_version` MUST change when any of the following changes:
- prompt bank contents
- case manifests
- target population
- validity thresholds
- metric formulas
- canonical scorer

Suggested format:
- `gutenberg_style_v1`
- `gutenberg_style_v2`

Documentation-only edits SHOULD NOT change the benchmark version.


## Human Evaluation

Human evaluation is required for validating the automatic metric stack, even if it is not required for every model run.

### Human Dev Set

The implementation SHOULD build a human-eval packet from benchmark-dev outputs.

Recommended protocol:
- show annotators the conditioning passages
- show the content prompt
- show two model outputs in blinded order
- ask which output better matches the target style
- ask whether either output appears copied
- ask whether each output addresses the prompt

### Human Annotation Schema

Each annotation row SHOULD include:
- `pair_id`
- `case_id`
- `output_a_model`
- `output_b_model`
- `preferred_style_output`
- `copy_suspected_a`
- `copy_suspected_b`
- `prompt_pass_a`
- `prompt_pass_b`

### Validation Goal

Before the benchmark is described as mature, automatic style ranking SHOULD show a reasonably strong correlation with human pairwise style judgments on benchmark-dev.

At minimum, the repo SHOULD be able to report:
- pairwise agreement rate between automatic rank and human preference
- rank correlation where applicable
- false-positive rate of copy detection on human-judged non-copies


## Testing And Acceptance

The benchmark repo MUST include tests.

### Required Unit Tests

1. `test_benchmark_schema.py`
   - schema validation accepts good records and rejects malformed ones
2. `test_passage_sampling.py`
   - sentence splitting is deterministic
   - passage windows satisfy length and non-overlap rules
3. `test_originality.py`
   - exact copies are flagged
   - clearly novel text is not flagged
4. `test_prompt_adherence.py`
   - obviously off-prompt output scores below clearly on-prompt output
5. `test_distractors.py`
   - distractor selection excludes target
   - distractor counts and metadata constraints hold
6. `test_benchmark_runner_smoke.py`
   - runner can execute a tiny fixture set with a stub generator and stub scorer
7. `test_aggregate_benchmark_results.py`
   - aggregation computes stable means and CI output structure

### Required Smoke Test

The repo SHOULD support a local smoke run with:
- 2 authors
- 2 prompts
- 1 sample per case
- a stub generator that returns fixed text
- a stub scorer or tiny fixture scorer

This smoke path is required so benchmark code can be exercised in CI without GPUs or paid APIs.

### Benchmark Readiness Checklist

The benchmark SHOULD be described as "ready" only if all are true:

1. scorer-train authors are disjoint from benchmark authors
2. conditioning and evaluation passages are disjoint
3. originality checks are implemented and enforced
4. prompt adherence is implemented and enforced
5. aggregate reports include confidence intervals
6. automatic metrics have been checked against a human dev set
7. the current proxy benchmark is no longer presented as the canonical benchmark


## Migration Plan From The Current Repo

The current [eval/benchmark_style.py](../eval/benchmark_style.py) SHOULD remain in the repo as:
- a smoke test
- a legacy proxy
- a quick integration harness

It SHOULD NOT remain the canonical benchmark once `benchmark_v2.py` exists.

Recommended migration order:

1. Add explicit schemas and benchmark manifests.
2. Build passage pools and case manifests.
3. Add the new runner with held-out target-vs-distractor scoring.
4. Add originality and prompt-adherence modules.
5. Add aggregation and CI-friendly smoke tests.
6. Add human-dev validation tooling.


## Recommended Defaults For v1

Use these defaults unless there is a documented reason not to.

- primary track: `author`
- conditioning passages per case: `3`
- conditioning words per passage: `150..300`
- target evaluation passages per case: `4`
- distractor targets per case: `5`
- samples per case: `3`
- prompt bank size: `16`
- style score basis: calibrated scorer output if available, else `score_0_1`
- prompt score formula: `0.85 * semantic_similarity + 0.15 * keyword_coverage`
- copy thresholds:
  - char 8-gram overlap `>= 0.30`
  - token LCS ratio `>= 0.20`
- valid length range: `350..1000` words
- repetition threshold: `< 0.20`
- bootstrap resamples: `1000`


## Open But Bounded Choices Beyond Canonical v1

For canonical `gutenberg_style_v1`, several previously open choices are fixed by the implementation-complete appendix below.

The only remaining bounded choices that MAY vary without changing the benchmark design are:
- whether to emit optional parquet mirrors of JSONL result files
- whether to emit optional HTML or markdown audit reports alongside JSON summaries
- whether to store additional debug-only per-case prompt renderings beyond the required JSONL fields

These choices do not change benchmark behavior and therefore do not require a benchmark-version bump.


## Implementation-Complete Contract Appendix

This appendix is normative. A future implementation MUST follow these contracts unless `benchmark_version` is changed.


## Canonical Benchmark Data File Contract

The benchmark builder MUST materialize these files exactly.

### `eval/benchmark_data/VERSION`

Plain text file containing exactly:

```text
gutenberg_style_v1
```

### `eval/benchmark_data/prompts_v1.json`

Must be a JSON array of prompt objects sorted by `prompt_id` ascending.

### `eval/benchmark_data/benchmark_{split}_targets_{track}_v1.json`

Each target manifest MUST be one JSON object:

```json
{
  "artifact_type": "benchmark_targets",
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "split": "test",
  "targets": [
    {
      "target_id": "author:jane_austen",
      "track": "author",
      "author_id": "author:jane_austen",
      "conditioning_book_ids": [
        "book:jane_austen:emma",
        "book:jane_austen:mansfield_park"
      ],
      "evaluation_book_id": "book:jane_austen:persuasion"
    }
  ]
}
```

### `eval/benchmark_data/cases_{track}_{split}_v1.jsonl`

This file MUST be JSONL, one case per line, sorted by:
- `target_id`
- then `prompt_id`

The builder MUST NOT emit cases in arbitrary hash-map order.


## Exact Sentence Splitter Contract

The canonical sentence splitter for `gutenberg_style_v1` MUST be implemented as follows:

1. Normalize newlines to `\n`.
2. Protect the following abbreviations by temporarily replacing the final period with `<DOT>`:
   - `mr.`
   - `mrs.`
   - `ms.`
   - `dr.`
   - `st.`
   - `prof.`
   - `jr.`
   - `sr.`
   - `vs.`
   - `etc.`
   - `e.g.`
   - `i.e.`
3. Split on the regex:

```text
[.!?]+["')\\]]*\\s+(?=[A-Z])
```

4. Restore protected periods by replacing `<DOT>` back to `.`.
5. Trim whitespace from each segment.
6. Drop empty segments.
7. Merge any segment shorter than `30` characters into the previous segment when possible; otherwise into the next segment.

This exact splitter MUST be used for:
- passage pools
- non-overlap checks
- benchmark case construction


## Exact Target Construction Algorithm

### Author Track

For each eligible benchmark author:

1. Collect all `eligible_author_track` books and sort by:
   - `sha1(f"{benchmark_version}|author|{target_id}|{book_id}|{builder_seed}")`
   - then `book_id`
2. Use:
   - first two books as `conditioning_book_ids`
   - third book as `evaluation_book_id`
3. Ignore any remaining books for canonical `v1`.

This exact rule is why canonical Author Track eligibility requires at least 3 books.

### Book Track

For each eligible benchmark book:
- `book_id` itself is the target
- conditioning and evaluation passages are selected from that one book with sentence-level non-overlap


## Exact Case Construction Algorithm

Given one target and one prompt:

1. Derive `case_seed = int(sha256(f"{benchmark_version}|{track}|{target_id}|{prompt_id}|{builder_seed}").hexdigest()[:8], 16)`.
2. Use `random.Random(case_seed)`.
3. For every candidate passage under the relevant target books, rank by:
   - absolute distance from preferred word count `220`
   - then `sha1(f"{case_seed}|{passage_id}")`
4. Select conditioning passages:
   - Author Track: `3` passages total across the two conditioning books
   - Book Track: `3` passages from the target book
5. Select evaluation passages:
   - `4` passages
   - non-overlapping with conditioning passages
   - prefer distinct `region_bucket` values
6. If not enough passages satisfy the preferred region-bucket pattern:
   - back off to the next-best ranked non-overlapping passage
7. Materialize the chosen `conditioning_passage_ids`, `evaluation_passage_ids`, and `sample_seeds` into the case manifest.

The runner MUST consume the materialized case file exactly as written.


## Exact Distractor Selection Algorithm

Each case MUST have exactly `5` distractor targets:
- `3` matched distractors
- `2` hard distractors

### Candidate Pool

The distractor candidate pool is:
- all other targets in the same track and split
- excluding the target itself
- excluding same-author targets in Author Track

### Metadata Match Score

For each candidate, compute:

`meta_score = 2*(same_period_known_and_equal) + 2*(same_genre_known_and_equal) + 1*(same_length_bucket)`

Definitions:
- `same_period_known_and_equal = 1` only if neither side is `"unknown"` and the values match
- `same_genre_known_and_equal = 1` only if neither side is `"unknown"` and the values match
- `same_length_bucket = 1` if the median evaluation passage word count differs by at most `40`

Sort candidates by:
- `meta_score` descending
- then `sha1(f"{case_seed}|{candidate_target_id}")`

Take the top `20` as the metadata-matched pool.

### Matched Distractors

Take the first `3` candidates from the sorted metadata-matched pool.

### Hard Distractors

For the remaining metadata-matched pool:

1. Compute one profile embedding per candidate target by mean-pooling the scorer embeddings of its conditioning passages.
2. Compute the target profile embedding the same way.
3. Rank remaining candidates by cosine similarity to the target profile descending.
4. Take the top `2` not already selected as matched distractors.

### Distractor Passages

For each chosen distractor target:
- take exactly `4` evaluation passages
- use the same deterministic passage-ranking logic as for the target
- preserve approximate passage-length parity with the target evaluation group when possible


## Exact Generator Adapter Contract

The canonical runner MUST use a provider-prefixed model ID:

- `openai:<model>`
- `anthropic:<model>`
- `gemini:<model>`
- `kimi:<model>`
- `stub:<name>` for smoke tests

The generator adapter MUST conceptually expose:

```python
class GenerationRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    seed: int | None

class GenerationResponse:
    ok: bool
    output_text: str
    provider: str
    model_name: str
    model_version: str | None
    finish_reason: str | None
    seed_supported: bool
    latency_ms: float | None
    error_type: str | None
    error_message: str | None
```

Rules:
- on success, `ok = true`, `output_text` MUST be `.strip()`'d, and `error_* = null`
- on failure, `ok = false`, `output_text = ""`, and `error_type` and `error_message` MUST be populated
- the runner MUST still write a result row for failed generations

### Smoke-Test Stub Provider

`stub:<name>` MUST be implemented for CI.

Required behaviors:
- `stub:echo_prompt` returns the prompt text truncated to the requested word range
- `stub:fixed_prose` returns a deterministic fixed paragraph


## Exact Prompt-Adherence Contract

For canonical `gutenberg_style_v1`, the prompt encoder is no longer implementation-defined.

### Frozen Encoder

The benchmark MUST use:

- `sentence-transformers/all-MiniLM-L6-v2`

Embedding rule:
- mean-pool the final token embeddings with attention-mask weighting
- L2-normalize
- compute cosine similarity
- map to `[0, 1]` with `(cosine + 1) / 2`

### Keyword Coverage

Keyword matching MUST:
- lowercase both prompt keywords and output text
- collapse whitespace to single spaces
- treat a keyword as covered if its normalized substring appears in the normalized output

### Prompt Score

The canonical formula is:

`prompt_score = 0.85 * semantic_similarity_0_1 + 0.15 * keyword_coverage`

`prompt_pass = prompt_score >= 0.45`


## Exact Originality Contract

For canonical `gutenberg_style_v1`, the LCS implementation is no longer open.

### Tokenization For LCS

1. Lowercase.
2. Replace punctuation with spaces.
3. Collapse whitespace.
4. Split on spaces.

### Exact Token LCS

The implementation MUST use exact dynamic programming over the token sequences, not an approximate substitute.

Because:
- generated outputs are capped by the fluency policy
- conditioning passages are short enough for exact DP

The metric is:

`token_lcs_ratio(h, c) = exact_LCS(tokens(h), tokens(c)) / max(1, len(tokens(h)))`


## Exact Fluency Contract

`malformed_flag` MUST be set to `true` if any are true:
- output text is empty after stripping
- output contains `[LLM ERROR]`
- output contains `api key`
- output contains `rate limit`
- alphabetic characters are less than `40%` of non-whitespace characters

`repetition_rate_6gram` MUST be:
- number of repeated 6-gram occurrences after the first occurrence
- divided by total 6-gram count


## Exact Benchmark Reference Distribution Contract

The benchmark MUST write:

- `eval/benchmark_data/benchmark_reference_distributions_author_v1.json`
- `eval/benchmark_data/benchmark_reference_distributions_book_v1.json`

Each file MUST have this schema:

```json
{
  "artifact_type": "benchmark_reference_distributions",
  "artifact_version": "benchmark_reference_distributions_author_v1",
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "score_basis": "calibrated_or_score_0_1",
  "source_split": "dev",
  "global": {
    "target_similarity": {
      "p1": 0.41,
      "p5": 0.49,
      "p10": 0.52,
      "p25": 0.60,
      "p50": 0.69,
      "p75": 0.78,
      "p90": 0.84,
      "p95": 0.87,
      "p99": 0.91,
      "mean": 0.68,
      "median": 0.69,
      "count": 640
    },
    "distractor_similarity": {
      "p1": 0.18,
      "p5": 0.22,
      "p10": 0.25,
      "p25": 0.31,
      "p50": 0.40,
      "p75": 0.49,
      "p90": 0.57,
      "p95": 0.61,
      "p99": 0.69,
      "mean": 0.40,
      "median": 0.40,
      "count": 3200
    },
    "style_margin": {
      "p1": -0.10,
      "p5": -0.04,
      "p10": -0.01,
      "p25": 0.05,
      "p50": 0.14,
      "p75": 0.24,
      "p90": 0.31,
      "p95": 0.35,
      "p99": 0.41,
      "mean": 0.15,
      "median": 0.14,
      "count": 640
    }
  }
}
```

### Build Algorithm

The artifact MUST be built from `benchmark_dev` targets only:

1. For each target, take its evaluation passages `p1..p4`.
2. For each `pi`:
   - set hypothesis = text of `pi`
   - set target references = texts of the other evaluation passages of that same target
   - compute `sT`
3. For that same `pi`, compute distractor group similarities against every distractor target selected by the canonical distractor builder.
4. Store:
   - `sT` in `target_similarity`
   - every distractor score in `distractor_similarity`
   - `sT - mean(distractors)` in `style_margin`

`style_percentile_case` MUST be computed against `global.target_similarity`.


## Exact Human-Eval Packet Contract

The human-eval builder MUST create packets only from `benchmark_dev` results.

### Representative Sample Selection

For each `(case_id, model)`:

1. Prefer valid outputs.
2. Rank outputs by:
   - `style_margin_case` descending
   - then `prompt_score` descending
   - then `sample_index` ascending
3. Take the first ranked output as the representative sample.

### Packet Enumeration

For each `case_id`:

1. Collect representative samples from all models.
2. Enumerate all unordered model pairs.
3. Rank model pairs by absolute difference in automatic `style_margin_case` descending.
4. Keep at most `3` model pairs per case.
5. For each selected pair, derive:

`packet_seed = int(sha256(f"{benchmark_version}|{case_id}|{model_a}|{model_b}").hexdigest()[:8], 16)`

6. If `packet_seed` is even, keep model ordering as `(A, B)`. If odd, swap to `(B, A)`.

### Packet Schema

```json
{
  "packet_id": "packet:gutenberg_style_v1:author:case123:modelA:modelB",
  "benchmark_version": "gutenberg_style_v1",
  "track": "author",
  "case_id": "case:author:dev:jane_austen:interpersonal_01",
  "prompt_id": "prompt:interpersonal:01",
  "prompt_text": "Write a scene in which two people...",
  "conditioning_texts": ["...", "...", "..."],
  "output_a": {
    "model_name": "openai:gpt-4o-mini",
    "sample_id": "run:...:case123:0",
    "text": "..."
  },
  "output_b": {
    "model_name": "anthropic:claude-3-5-sonnet",
    "sample_id": "run:...:case123:0",
    "text": "..."
  }
}
```

### Annotation Schema

```json
{
  "annotation_id": "annotation:packet123:annotator01",
  "packet_id": "packet:gutenberg_style_v1:author:case123:modelA:modelB",
  "annotator_id": "annotator01",
  "preferred_style_output": "A",
  "copy_suspected_a": false,
  "copy_suspected_b": false,
  "prompt_pass_a": true,
  "prompt_pass_b": true,
  "notes": ""
}
```

### Validation Aggregation Rules

The validator MUST:

1. Group annotations by `packet_id`.
2. Compute majority vote separately for:
   - `preferred_style_output`
   - `copy_suspected_a`
   - `copy_suspected_b`
   - `prompt_pass_a`
   - `prompt_pass_b`
3. Treat ties as unresolved and exclude them from agreement metrics.
4. Define the automatic winner as the representative sample with larger `style_margin_case`.
5. Report:
   - `pairwise_style_agreement = mean(automatic_winner == human_majority_winner)`
   - `copy_false_positive_rate = fraction of outputs with auto copy_flag = true among human-majority non-copies`
   - `prompt_false_negative_rate = fraction of outputs with auto prompt_pass = false among human-majority prompt passes`


## Exact Aggregation Contract

The aggregate report MUST bootstrap over `case_id`, not individual samples.

Algorithm:

1. Group result rows by `case_id`.
2. For each bootstrap resample:
   - sample cases with replacement
   - include all sample rows belonging to each sampled case
   - recompute aggregate means
3. Compute `95%` confidence intervals by the percentile method:
   - lower = `2.5th` percentile
   - upper = `97.5th` percentile


## Canonical v1 Choices That Are No Longer Open

For `gutenberg_style_v1`, these choices are fixed:
- passage text MUST be stored inline in passage JSONL records
- prompt encoder MUST be `sentence-transformers/all-MiniLM-L6-v2`
- hard distractor profiles MUST use mean-pooled conditioning-passage embeddings
- token LCS MUST be exact dynamic programming


## Immediate Next Implementation Tasks

If building from this spec in the current repo, the next concrete work SHOULD be:

1. Create `eval/benchmark_schema.py`.
2. Create `eval/passage_sampling.py`.
3. Create `eval/build_benchmark_manifests.py`.
4. Materialize `eval/benchmark_data/prompts_v1.json`.
5. Create `eval/style_scoring.py`, `eval/originality.py`, `eval/prompt_adherence.py`, and `eval/fluency.py`.
6. Create `eval/benchmark_v2.py`.
7. Create `eval/aggregate_benchmark_results.py`.
8. Add the smoke tests under `tests/`.


## Acceptance Standard

`writing7` SHOULD claim to have a quantitative style mimicry benchmark only when the canonical benchmark path follows this spec, or a documented successor spec, end to end.

Until then:
- [eval/benchmark_style.py](../eval/benchmark_style.py) SHOULD be described as a proxy or smoke benchmark
- not as the final benchmark for quantifying style mimicry quality
