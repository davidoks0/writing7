# Scorer Artifacts

The benchmark expects a scorer artifact directory containing:

- `config.json`
- `scorer_manifest.json`
- `style_calibration_v1.json`
- tokenizer metadata files
- `test_metrics.json`

The local smoke build writes a lightweight bag-of-words scorer artifact under
`build/artifacts/scorer/final/`.
