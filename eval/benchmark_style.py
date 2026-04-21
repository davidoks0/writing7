from __future__ import annotations

import argparse

from eval.benchmark_v2 import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Legacy smoke/proxy wrapper around the canonical benchmark runner.")
    parser.add_argument("--track", default="author")
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--model", default="stub:fixed_prose")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    run_benchmark(
        track=args.track,
        split=args.split,
        cases_path=args.cases,
        prompts_path=args.prompts,
        model=args.model,
        model_dir=args.model_dir,
        out_path=args.out,
        config_path=args.config,
        limit=1,
    )


if __name__ == "__main__":
    main()

