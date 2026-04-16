**Overview**
- Focus: train with style-hard negatives instead of metadata heuristics.
- Sources: same-author books, embedding-neighbor books, and ANN-mined chunk pairs.
- Removed: metadata-based negatives (e.g., period/topic/era), and any period heuristics.

**Negative Types**
- author_same: different book by the same author; strongest stylistic confounder. Skipped for anonymous/unknown/various authors.
- embed_neighbor: different-author book selected from top-K centroid neighbors by sentence-embedding similarity.
- random: uniform random book from the split (different from anchor book); acts as regularization.
- ann_mined: chunk-level hard negatives found by the two-stage ANN miner using the current contrastive model’s encoder and head.
- model_mined: optional re-scored pairs from a trained model head; kept for continuity but off the main path.

**Sampling Weights**
- Distribution per book: use a small random fraction and split the rest between the two hard sources.
  - If both author_same and embed_neighbor exist: random=r, author_same=(1−r)/2, embed_neighbor=(1−r)/2.
  - If only one hard source exists: random=r, hard=(1−r).
  - If neither exists: random=1.0.
- Parameter: random_neg_frac=r (default ≈ 0.10). This keeps ~10% easy negatives to stabilize training.
- Each negative example records neg_type in the dataset.

**Book-Level Neighbors (Stage 1)**
- Purpose: restrict hard-negative search to stylistically similar books.
- Method: compute a centroid per book using `sentence-transformers/all-MiniLM-L6-v2` over up to 80 chunks per book; L2-normalize; take top-K cosine neighbors.
- Default K: `num_hard_negative_books = 50` (centroid neighbors) for all paths.
- GPU behavior: encodes centroids in large microbatches; CPU path uses smaller batches.

**Two-Stage ANN Miner (Stage 2)**
- Status: enabled by default in the GPU prepare path; off by default on CPU.
- Flow per anchor book:
  - Anchors: sample up to `ann_anchors_per_book` chunks from the anchor book.
  - Candidate pool: take up to `ann_k_neighbors` neighbor books (from Stage 1) and sample `ann_pool_samples_per_book` chunks from each.
  - Embed: encode anchors and pool with the trained contrastive encoder; compute cosine sims on-device (GPU when available).
  - Filter: keep the top-1 neighbor per anchor above `ann_sim_threshold` (cosine), then rescore with the model head; keep only those with probability ≤ `ann_prob_max`.
  - Caps: per-book cap `ann_max_negatives_per_book` and optional global `ann_max_total_negatives`.
- Tags: mined pairs are labeled with `neg_type="ann_mined"` and include author/topic annotations for analysis.

**Default Parameters (GPU path)**
- Centroid neighbors: `num_hard_negative_books=50`.
- ANN miner: `use_ann_chunk_negatives=True`, `ann_k_neighbors=20`, `ann_sim_threshold=0.55`, `ann_prob_max=0.20`, `ann_anchors_per_book=40`, `ann_pool_samples_per_book=20`, `ann_batch_size=64`, `ann_max_negatives_per_book=40`.
- Notes:
  - `ann_k_neighbors` limits neighbor books for the chunk-level pool. It is separate from `num_hard_negative_books` (centroid neighbors).
  - Increase `ann_k_neighbors` if you want broader pools at the cost of throughput.

**What’s Removed**
- No metadata-driven negatives: we do not use topic/era/period heuristics to pick “similar” books.
- No period heuristics: negatives focus on author identity and embedding proximity only, plus a small random slice.

**Outputs and Inspection**
- Fields per pair: `pair_type`, `label`, `neg_type` (for negatives), `book1`, `book2`, `author1`, `author2`, `same_author`, and lightweight topic tags (`topic1/2`, `same_topic`) for analysis.
- Splits: returned as HuggingFace `DatasetDict` with `train/validation/test` splits.

**How To Run (GPU default)**
- Modal entrypoint: `prepare_remote_gpu` enables ANN mining by default and saves to `/vol/data/processed`.
- Key args to tune:
  - `num_hard_negative_books` (centroid neighbors; default 50)
  - `ann_k_neighbors`, `ann_anchors_per_book`, `ann_pool_samples_per_book`, `ann_prob_max`, `ann_sim_threshold`
  - `n_positive_per_book`, `n_negative_per_book`, `random_neg_frac` (random regularization)

**Scaling Note (59k books on H200)**
- The two-stage plan avoids NxN chunk comparisons by restricting to centroid neighbors first, then mining within a limited pool per book on GPU.
- With `num_hard_negative_books=50`, `ann_k_neighbors≈20`, `ann_anchors_per_book≈40`, and `ann_pool_samples_per_book≈20`, the miner keeps device matmuls small and batched; expect minutes to tens of minutes depending on text size and I/O.
- For larger runs or multi-GPU, shard by book IDs and aggregate mined JSONL outputs before final dataset assembly.

**Rationale**
- Same-author and embedding-neighbor negatives directly target stylistic confusion.
- ANN-mined chunk pairs expose near-duplicate style collisions missed at the book level while the head rescoring guards against false negatives.
- A small fraction of random negatives regularizes the decision boundary and reduces overfitting to hard cases.

