# Style Mimicry Implementation Handoff

This document is the implementation brief I would hand to another Codex model if the goal is to make this repo into a genuinely strong benchmark for:

`How good are different LLMs at mimicking a target literary style while writing new content?`

It is intentionally concrete. Treat it as a build spec plus execution plan.


## Goal

Build the strongest version of the style-mimicry system that is still consistent with the repo's current architecture:

- `corpus/` owns texts, metadata, and splits
- `training/` owns the learned style scorer
- `eval/` owns benchmark manifests, runner logic, and reporting

The end state should be:

- a style scorer that is harder to fool with topic, plot residue, names, or setting
- a benchmark runner that better detects copying and task evasion
- a leaderboard output that can compare multiple model runs cleanly
- a stress suite that proves the scorer is learning style rather than just content similarity


## Current State

Important current truths:

- The benchmark shape is basically right for style mimicry.
- The largest remaining weakness is scorer leakage from topic/content/book identity into the style signal.
- The repo now already contains:
  - hidden-target originality checking
  - basic leaderboard row generation
  - a `style_masked_v1` text view that masks likely entities/numbers
  - offline smoke fallback for scorer training

What is still missing:

- real topic/content-adversarial training
- chunked multi-window style scoring
- blended raw-vs-masked scoring
- stronger anti-copy checks against entire target works
- explicit scorer stress tests for topic leakage
- better leaderboard packaging and benchmark diagnostics


## Non-Goals

Do not turn this into:

- a general authorship-attribution product
- a generic literary-quality benchmark
- a rewrite/content-preservation benchmark
- an LLM-judge-dependent benchmark

This system is specifically about:

- `style-conditioned original generation`
- `held-out style scoring`
- `copy resistance`
- `cross-model comparability`


## Build Order

Implement these phases in order. Do not skip ahead unless blocked.

### Phase 1: Finish The Style Scorer

This is the highest-leverage phase.

#### 1. Add Real Topic/Content-Adversarial Training

Implement the spec's missing adversary path in `training/`.

What to build:

- a semantic/content teacher model, default:
  - `sentence-transformers/all-MiniLM-L6-v2`
- a topic/content adversary head on top of the style embedding
- gradient reversal so:
  - adversary gets better at predicting content/semantic clusters
  - style encoder gets better at removing content information

How to structure it:

- add gradient-reversal and adversary modules in `training/transformer_style_model.py`
- keep the main encoder architecture as:
  - `roberta-large`
  - attention pooling
  - projection head
- adversary should consume the final normalized embedding or the projected pre-normalization vector

Training losses:

- `pair_loss`: BCE over same-author vs different-author pairs
- `contrastive_loss`: supervised contrastive author loss
- `adv_loss`: content/topic adversary loss

Optimize:

- `total_loss = pair_loss + contrastive_loss + adv_lambda * adv_loss_with_gradient_reversal`

Config:

- honor:
  - `use_topic_adversary`
  - `semantic_adversary_model`
  - `adv_lambda`
- update defaults in `configs/scorer_train_v1.json` to the canonical intended settings once the path is real

Important implementation note:

- do not make smoke tests depend on network downloads
- the adversary path must degrade gracefully in local smoke mode
- use the bag-of-words path or disable adversary in smoke fixtures if heavyweight deps are unavailable

#### 2. Create Content Targets For The Adversary

The adversary needs targets that represent content, not author.

Recommended approach:

- in `training/build_scorer_dataset.py`, compute a content embedding for each passage using the semantic teacher
- cluster those embeddings into content/topic buckets
- store:
  - passage-level content cluster id
  - pair-level cluster metadata

Fallback if clustering is too heavy for smoke mode:

- derive a lightweight proxy label from:
  - semantic hashing
  - metadata bucket
  - nearest semantic centroid index

Canonical training should use semantic clusters, not just book ids.

#### 3. Add Raw-vs-Masked Dual-View Scoring

Do not rely on masking alone.

Implement two parallel views:

- `raw`
- `style_masked_v1`

Best target design:

- either train two scorer artifacts and ensemble them
- or train one scorer with mixed exposure to both views and score both views at inference time

Preferred inference output:

- `raw_similarity`
- `masked_similarity`
- `blended_similarity`

Recommended default:

- `blended_similarity = 0.35 * raw + 0.65 * masked`

Make the blend configurable and record it in the scorer manifest.

#### 4. Add Chunked Multi-Window Scoring

Single whole-passage embeddings are too blunt.

Implement chunked scoring in `eval/style_scoring.py`.

Target behavior:

- split each text into overlapping chunks
- embed chunks separately
- compare hypothesis chunks to reference chunks
- aggregate with a stable summary, e.g.:
  - top-k mean
  - trimmed mean
  - or median of best-match scores

Recommended default:

- chunk size: around 120 to 180 words
- overlap: around 40 to 60 words
- aggregation:
  - compute best match per hypothesis chunk
  - aggregate with mean of top `k`

Expose chunk settings in scorer config and calibration metadata.


### Phase 2: Prove The Scorer Is Actually Learning Style

This phase is mandatory. Do not trust the scorer without it.

#### 5. Add Topic-Leakage Stress Tests

Create tests and diagnostics specifically designed to catch false style wins from content overlap.

Add fixtures and tests for:

- same topic, different author
- same author, different topic/book
- entity-swapped versions of the same passage
- passages with numbers/dates/places changed but syntax preserved
- adversarial near-topic matches

Add tests under `tests/` for:

- scorer invariance to entity swaps
- masked scoring improving over raw scoring in topic-confusable cases
- adversarial training reducing content-cluster predictability

#### 6. Add Scorer Diagnostics Reports

Extend `training/diagnostics.py` or add it if needed.

Produce reports such as:

- same-author accuracy
- same-book vs cross-book comparisons
- topic-confusable negative accuracy
- raw-vs-masked ablation
- adversary accuracy:
  - this should go down if the encoder is successfully hiding content

Write JSON diagnostics artifacts and, if easy, simple plots.


### Phase 3: Harden The Benchmark Itself

#### 7. Extend Originality Checks Beyond Visible References

The benchmark must check copying against more than the 3 conditioning passages.

Current system already checks hidden evaluation passages. Extend further.

Target behavior:

- compare generation against:
  - conditioning passages
  - hidden evaluation passages
  - the full target evaluation book when available
  - optionally all conditioning books in the author track

Implementation:

- add a full-book comparison path in `eval/originality.py`
- cache or precompute lightweight reference indexes if needed
- keep smoke mode small and deterministic

#### 8. Add Named-Entity Transplant Detection

The benchmark prompt explicitly forbids reusing names/places.

Implement a deterministic heuristic that flags:

- shared title-case entity sequences
- repeated place/name spans
- suspicious overlap of rare capitalized tokens

Use it as an additional originality submetric, not the only metric.

#### 9. Add A Benchmark Stress Split

Create a mini manifest specifically for benchmark sanity checks:

- same-era, same-genre confusable authors
- highly distinctive authors with obvious names/places removed
- target/distractor pairs designed to expose topic leakage

This should not replace the main benchmark.
It should exist to answer:

`Does the scorer still work when content overlap is actively trying to fool it?`


### Phase 4: Improve Reporting And Leaderboard Outputs

#### 11. Publish Better Summary Outputs

Keep the current separate-axis philosophy.

Official reported metrics should include:

- `style_win_rate_valid`
- `style_margin_valid`
- `top1_target_accuracy_valid`
- `style_percentile_valid`
- `valid_rate`
- `originality_pass_rate`
- `conditioning_copy_free_rate`
- `target_evaluation_copy_free_rate`
- `full_target_book_copy_free_rate` if implemented
- `fluency_pass_rate`

Derived scoreboard metric:

- `style_mimicry_score = style_win_rate_valid * valid_rate`

This is fine as a leaderboard sort key, but never publish it alone.

#### 12. Build A Real Multi-Run Leaderboard Tool

Extend the current leaderboard builder so it can:

- ingest multiple summary JSON files or result JSONL files
- output:
  - sorted JSON leaderboard
  - markdown table
  - CSV
- optionally group by:
  - track
  - split
  - prompt family

Add tie-break rules:

- `style_mimicry_score`
- then `style_win_rate_valid`
- then `valid_rate`
- then `style_margin_valid`

## Files To Change

This is the expected write scope.

### Must Change

- `training/transformer_style_model.py`
- `training/train_style_scorer.py`
- `training/build_scorer_dataset.py`
- `eval/style_scoring.py`
- `eval/originality.py`
- `eval/aggregate_benchmark_results.py`
- `configs/scorer_train_v1.json`
- `docs/SETUP_STANDALONE.md`

### Likely New Files

- `training/style_text.py`
- `training/topic_adversary.py` or equivalent helper module
- `training/semantic_content_labels.py` or equivalent
- `eval/build_benchmark_leaderboard.py` improvements
- new tests:
  - `tests/test_style_text.py`
  - `tests/test_build_scorer_dataset.py`
  - adversary-specific tests
  - topic leakage tests

### Maybe Change

- `modal_app.py`
- `models/README.md`
- `docs/GUTENBERG_STYLE_SYSTEM_SPEC.md`
- `docs/GUTENBERG_STYLE_BENCHMARK_SPEC.md`


## Acceptance Criteria

Do not call the work done unless these are true.

### Scorer

- adversarial training path exists and runs in production mode
- smoke mode still passes offline
- scorer artifacts record:
  - text view
  - chunking settings
  - whether topic adversary was enabled
  - adversary model id
  - blend weights if using multi-view scoring

### Tests

- all existing tests pass
- new tests cover:
  - entity masking behavior
  - dataset content-label generation
  - adversary path smoke coverage
  - masked-vs-raw scorer behavior
  - chunked scoring behavior
  - hidden-target/full-book copy detection

### Benchmark

- benchmark summary includes separate style/copy/fluency axes
- leaderboard builder supports multiple summaries
- benchmark can still run in local smoke mode with no network

### Diagnostics

- scorer diagnostics show whether topic leakage improved


## Recommended Execution Prompt For Another Codex Model

Give the next Codex model something close to this:

> Implement the full style-mimicry hardening plan in `docs/STYLE_MIMICRY_IMPLEMENTATION_HANDOFF.md`. Work in phases. Prioritize the scorer first: topic/content-adversarial training, semantic content labels, raw-vs-masked blended scoring, and chunked multi-window scoring. Then harden originality against full target books, improve leaderboard/reporting, and add diagnostics and tests. Preserve local smoke mode and keep the benchmark deterministic. Do not stop at analysis; make the code changes, update tests/docs/configs, and run the relevant test slices.


## Suggested Commands

Use these while iterating:

```bash
.venv/bin/python -m unittest tests.test_style_text tests.test_style_scoring tests.test_build_scorer_dataset tests.test_originality tests.test_aggregate_benchmark_results tests.test_build_benchmark_leaderboard tests.test_production_hardening tests.test_benchmark_runner_smoke
```

For wider regression:

```bash
.venv/bin/python -m unittest discover -s tests
```


## Final Guidance

If tradeoffs are required, prioritize in this order:

1. make the scorer less topic-sensitive
2. prove that with diagnostics/tests
3. harden anti-copy
4. improve reporting
5. expand human validation

If the work gets too large, it is acceptable to land it in multiple passes, but the scorer-adversary work and the topic-leakage stress tests should be treated as one unit.
