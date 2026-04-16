# Setting Up Standalone Writing7 Style Benchmark

## Step 1: Create Standalone Repository

```bash
# From your Projects directory (outside of writing7/)
cd /Users/davidoks/Projects
cp -r writing7/writing7-style-benchmark .
cd writing7-style-benchmark

# Initialize as git repository
git init
git add .
git commit -m "Initial commit - Writing7 style benchmark"

# Create GitHub repo and push
# (Create repo on GitHub first, then:)
git remote add origin https://github.com/your-username/writing7-style-benchmark.git
git push -u origin main
```

## Step 2: Download Models from Modal (Fix for Directory Error)

The "is a directory" error occurs because Modal is trying to overwrite existing directories. Here's how to fix it:

```bash
# Method 1: Download with force flag
modal volume get writing7-artifacts /models/book_matcher_contrastive ./models/book_matcher_contrastive --force

# Method 2: Download to temporary location first
modal volume get writing7-artifacts /models ./temp_models
mv temp_models/book_matcher_contrastive ./models/
rm -rf temp_models

# Method 3: Download individual files if directories fail
mkdir -p models/book_matcher_contrastive/final
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/pytorch_model.bin ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/config.json ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/tokenizer.json ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/tokenizer_config.json ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/vocab.json ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/final/merges.txt ./models/book_matcher_contrastive/final/
modal volume get writing7-artifacts /models/book_matcher_contrastive/calibration.json ./models/book_matcher_contrastive/
modal volume get writing7-artifacts /models/book_matcher_contrastive/style_calibration.json ./models/book_matcher_contrastive/
```

## Step 3: Update Model Integration

Once you have the actual models downloaded, update the scorer to use real inference:

1. Copy the actual `ContrastiveBookMatcher` class from `train_contrastive.py`
2. Update `scorer.py` to load real model weights
3. Replace the mock style feature extraction with the real implementation

## Step 4: Upload to HuggingFace

```bash
# From the writing7/ directory (where upload_to_hf.py is)
cd ../writing7
python upload_to_hf.py --model-dir models/book_matcher_contrastive --repo-id your-org/writing7-contrastive-v1

# Update the benchmark to use your actual HF model
cd ../writing7-style-benchmark
# Edit download_model.py and README.md to use your real model ID
```

## Step 5: Test End-to-End

```bash
cd /Users/davidoks/Projects/writing7-style-benchmark

# Install dependencies
pip install -e .

# Download your uploaded model
python download_model.py --model-id your-org/writing7-contrastive-v1

# Run benchmark
python run_benchmark.py --model openai:gpt-4o-mini --book books/gatsby.txt --model-path ./models/your-model/final
```

## Directory Structure After Setup

```
/Users/davidoks/Projects/
├── writing7/                           # Original training repo
│   ├── train_contrastive.py
│   ├── upload_to_hf.py
│   └── models/book_matcher_contrastive/ # Downloaded from Modal
└── writing7-style-benchmark/           # Standalone benchmark repo
    ├── README.md
    ├── setup.py
    ├── requirements.txt
    ├── run_benchmark.py
    ├── scorer.py
    ├── download_model.py
    ├── books/
    ├── examples/
    └── models/                         # Downloaded from HuggingFace
```

## Troubleshooting Modal Downloads

If you continue to have issues with Modal downloads:

1. **Check Modal version**: `modal --version`
2. **Try with verbose output**: `modal volume get writing7-artifacts /models/book_matcher_contrastive ./models/book_matcher_contrastive -v`
3. **Download in chunks**: Download subdirectories individually
4. **Use Modal web interface**: Go to Modal dashboard and download manually if needed

The key issue you're seeing is that Modal won't overwrite existing directories. Always ensure the target directory doesn't exist before downloading.