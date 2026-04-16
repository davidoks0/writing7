# Repository Guidelines

## Project Structure & Module Organization
- Core training scripts live at the repo root: `prepare_data.py` (dataset prep), `train_contrastive.py` (contrastive style embedder), `calibrate_style_similarity.py`, and `inference_contrastive.py`.
- `data/processed/` stores generated datasets; keep raw books in an untracked `training/` folder.
- `models/` holds local artifacts (`style_embedder`, `book_matcher_contrastive`); avoid committing checkpoints.
- `eval/` contains the style benchmark (`benchmark_style.py`) and book fixtures; `scripts/` has utilities like `push_to_hf.py` and `download_hf_model.py`.
- `modal_app.py` packages the workflow for Modal GPU/CPU runs, mounting volumes `writing7-training` (input) and `writing7-artifacts` (outputs).

## Build, Test, and Development Commands
- Environment: `source venv/bin/activate` (or `.venv`) then `pip install -r requirements.txt`.
- Prepare data locally: `python prepare_data.py --training-dir training --output data/processed`.
- Train: `python train_contrastive.py --model roberta-large --contrastive-only --output models/style_embedder --epochs 6`.
- Calibrate without retraining: `python calibrate_style_similarity.py --model models/style_embedder/final --pairs path/to/pairs.csv`.
- Modal GPU flow: `modal run modal_app.py::prepare_remote_gpu` then `modal run modal_app.py::train_contrastive_gpu -- --model roberta-base`.
- Quick smoke inference: `python inference_contrastive.py --model models/book_matcher_contrastive/final --text1 "...first..." --text2 "...second..."`.

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with snake_case modules and functions. Use docstrings and light type hints (see `train_contrastive.py` patterns).
- Prefer small, composable functions; avoid hidden globals. Keep CLI args mirrored between local scripts and Modal entrypoints.
- Logging via `print` is fine for CLIs; gate expensive GPU ops behind flags.

## Testing Guidelines
- No formal unit suite yet; run smoke checks after changes:
  - Data → train → inference on a small sample.
  - Style benchmark (requires LLM API keys and GPU): `python -m eval.benchmark_style --model openai:gpt-4o-mini --book_path eval/books/gatsby.txt --n_excerpts 1 --n_samples 1 --model_dir models/book_matcher_contrastive/final`.
- Record metrics (F1/accuracy/ROC-AUC) after training; rerun calibration if tokenization or style features change.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative (e.g., “Update ANN configurations”).
- In PRs, include: scope/intent, commands run, datasets/volumes touched, and before/after metrics or calibration changes. Note Modal commands when used.
- Do not commit raw books, large artifacts, or secrets; publish models via `scripts/push_to_hf.py` or keep them on Modal volumes.

## Security & Configuration Tips
- Keep API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACE_HUB_TOKEN`, Modal token) in env vars; never commit them.
- Validate volume names before Modal runs to avoid overwriting artifacts. When sharing, document the model dir (e.g., `models/book_matcher_contrastive/final`) and calibration JSONs alongside it.
