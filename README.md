# Gutenberg Style System

This repo implements the Gutenberg style system and benchmark described in:

- `docs/GUTENBERG_STYLE_SYSTEM_SPEC.md`
- `docs/GUTENBERG_STYLE_BENCHMARK_SPEC.md`
- `docs/STYLE_MIMICRY_IMPLEMENTATION_HANDOFF.md`

The code is organized into:

- `corpus/` for acquisition, cleaning, manifests, eligibility, and splits
- `training/` for scorer dataset construction, transformer training, and calibration
- `eval/` for benchmark manifest building, running, and aggregation
- `modal_app.py` for production Modal entrypoints

The repo now supports two modes:

- production Modal mode with named Modal volumes, Gutenberg catalog plus HTTP/rsync acquisition, self-contained root-relative corpus manifests, Hugging Face-backed transformer training, and benchmark runs against `stub:`, `openai:`, or `anthropic:` providers
- local smoke mode that keeps the test suite lightweight and API-free

Start with `docs/SETUP_STANDALONE.md` for the production workflow.
