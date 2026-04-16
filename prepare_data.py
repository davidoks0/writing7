"""
Prepare training data for book-text matching classifier.

Creates positive pairs (text chunks from same book) and negative pairs (text chunks from different books).
Defaults emphasize higher-quality chunks, cleaning boilerplate, hard negatives, and simple cross-book deduplication.
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import unicodedata
import random
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import re
import json

import numpy as np
from datasets import Dataset, DatasetDict

# Use PG/boilerplate cleaning by default
try:
    from standardize_training import clean_text as _clean_text
except Exception:
    _clean_text = None

try:
    import spacy
    # Use lightweight rule-based sentencizer for speed
    _nlp = spacy.blank("en")
    if "sentencizer" not in _nlp.pipe_names:
        _nlp.add_pipe("sentencizer")
    # Allow very long texts (we don't run parser/NER). 50M is a safe upper bound.
    _nlp.max_length = 50_000_000
    USE_SPACY = True
except Exception:
    USE_SPACY = False
    _nlp = None
    print("Warning: spaCy not available, falling back to simple sentence splitting")


def split_sentences_spacy(text: str) -> List[str]:
    """Fast, rule-based sentence splitting using spaCy sentencizer."""
    doc = _nlp(text)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]


def split_sentences_simple(text: str) -> List[str]:
    """Simple sentence splitting with basic improvements."""
    # Split on sentence-ending punctuation, but handle common abbreviations
    # This is still imperfect but better than before
    sentences = []
    
    # Pattern: end of sentence followed by space and capital letter
    # Improved to handle quotes, parentheticals, etc.
    pattern = r'[.!?]+["\'\)]*\s+(?=[A-Z])'
    
    # Split on the pattern
    parts = re.split(pattern, text)
    
    for part in parts:
        part = part.strip()
        if len(part) > 20:  # Filter very short sentences
            sentences.append(part)
    
    # If no splits found, return whole text
    if not sentences and len(text.strip()) > 20:
        sentences.append(text.strip())
    
    return sentences


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using best available method.

    Falls back to a fast regex splitter if spaCy raises a max_length error or
    if the input exceeds the configured limit. For sentencizer-only, increasing
    max_length is safe; we bump it on the fly when needed.
    """
    if USE_SPACY:
        try:
            # Bump spaCy limit dynamically if needed (no parser/NER used)
            try:
                if len(text) + 1024 > getattr(_nlp, "max_length", 10_000_000):
                    _nlp.max_length = len(text) + 1024
            except Exception:
                pass
            return split_sentences_spacy(text)
        except Exception:
            # Safety net: fall back to simple
            return split_sentences_simple(text)
    else:
        return split_sentences_simple(text)


def _process_book_worker(job):
    """Worker to process a single book file into chunks + lightweight metadata.

    Args (packed tuple): (book_path_str, chunk_size, overlap, max_chunks_per_book, use_hard_negatives)
    Returns a tuple (status, book_id, chunks, metadata_or_error).
    """
    try:
        book_path_str, chunk_size, overlap, max_chunks_per_book, use_hard_negatives = job
        path = Path(book_path_str)
        # Use author/title for uniqueness with the new layout
        author_id = path.parent.name
        book_id = f"{author_id}/{path.stem}"
        sentences = load_book(path)
        chunks = create_text_chunks(sentences, chunk_size, overlap)
        if chunks:
            if len(chunks) > max_chunks_per_book:
                chunks = random.sample(chunks, max_chunks_per_book)
            # Metadata-based negatives removed; return empty metadata
            return ("ok", book_id, chunks, {})
        return ("empty", book_id, [], {})
    except Exception as e:
        try:
            bid = f"{path.parent.name}/{path.stem}"  # type: ignore[name-defined]
        except Exception:
            bid = ""
        return ("error", bid, [], str(e))


def load_book(book_path: Path) -> List[str]:
    """Load a book and split into sentences in a single pass."""
    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Unicode normalization and whitespace cleanup
    try:
        text = unicodedata.normalize('NFC', text)
        text = text.replace('\u00A0', ' ').replace('\u200B', '')
        text = re.sub(r"[\t\r\f]+", " ", text)
        text = re.sub(r"\s+", " ", text)
    except Exception:
        pass
    # Clean boilerplate if available
    if _clean_text is not None:
        try:
            text = _clean_text(text)
        except Exception:
            # Fall back silently if cleaning fails
            pass
    # Single-pass sentence splitting is much faster than per-line processing
    return split_sentences(text)


_EN_STOP = set("the of and to in a that is it for on as with by be this from at are was were have has had or an not but which you he she they we his her their our its can could would should into about than then them no so such more any may one other who what when where how why all those these some if do does did been being very over out up down again ever always never upon under between".split())

def _is_likely_english(text: str) -> bool:
    """Lightweight check to filter obviously non-English chunks without heavy deps.

    Heuristics: ratio of [a-z] letters and presence of common stopwords.
    """
    if not text or len(text) < 40:
        return True
    tl = text.lower()
    letters = sum(1 for c in tl if 'a' <= c <= 'z')
    total_letters = sum(1 for c in tl if c.isalpha())
    if total_letters == 0:
        return False
    ratio = letters / float(total_letters)
    # stopword presence
    words = set(tl.split())
    sw = len(words & _EN_STOP)
    return (ratio >= 0.6) and (sw >= 2)


def create_text_chunks(sentences: List[str], chunk_size: int = 20, overlap: int = 5, min_chars: int = 200) -> List[str]:
    """Create overlapping chunks of sentences."""
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = ' '.join(sentences[i:i + chunk_size])
        if len(chunk.strip()) >= int(min_chars):  # Higher minimum chunk length by default
            if _is_likely_english(chunk):
                chunks.append(chunk.strip())

    return chunks


def sample_non_overlapping_pair(chunks: List[str], chunk_size: int = 20, overlap: int = 5, rng: random.Random = None) -> Tuple[str, str]:
    """Sample two chunks that don't share any sentences.

    Given overlapping chunks, we need to ensure the selected pair doesn't share content.
    Adjacent chunks share `overlap` sentences, so we need at least `ceil(overlap / step)`
    chunks between selected indices to avoid overlap.

    Args:
        chunks: List of text chunks
        chunk_size: Number of sentences per chunk
        overlap: Number of overlapping sentences between adjacent chunks
        rng: Random number generator

    Returns:
        Tuple of (chunk1, chunk2) that don't share sentences
    """
    if rng is None:
        rng = random.Random()

    if len(chunks) < 2:
        raise ValueError("Need at least 2 chunks")

    # Calculate minimum index gap to avoid overlap
    # step = chunk_size - overlap (how far each chunk advances)
    # Two chunks at indices i and j share content if |i - j| * step < chunk_size
    # So we need |i - j| >= ceil(chunk_size / step) = ceil(chunk_size / (chunk_size - overlap))
    step = max(1, chunk_size - overlap)
    min_gap = max(1, (chunk_size + step - 1) // step)  # ceil division

    # If we don't have enough chunks for non-overlap, fall back to maximizing distance
    if len(chunks) <= min_gap:
        # Just pick first and last
        return chunks[0], chunks[-1]

    # Sample first index, then sample second ensuring minimum gap
    idx1 = rng.randrange(len(chunks))

    # Valid indices for second chunk
    valid_indices = [j for j in range(len(chunks)) if abs(j - idx1) >= min_gap]

    if not valid_indices:
        # Fallback: pick the furthest chunk
        idx2 = 0 if idx1 >= len(chunks) // 2 else len(chunks) - 1
    else:
        idx2 = rng.choice(valid_indices)

    return chunks[idx1], chunks[idx2]


def _is_anonymous_author(author_slug: str) -> bool:
    """Return True if the author slug denotes anonymous/unknown/various.

    We avoid using such buckets for "same-author" hard negatives, since they
    don't represent a consistent author style.
    """
    a = (author_slug or "").strip().lower()
    if not a:
        return True
    # Common slugs from our ingest/slugify: 'anonymous', 'unknown_author'
    anon_exact = {"anonymous", "unknown", "unknown_author", "various", "misc", "multiple_authors"}
    if a in anon_exact:
        return True
    # Substring heuristics to catch variants
    for token in ("anonymous", "unknown", "various"):
        if token in a:
            return True
    return False


# Removed: metadata inference for negatives


def prepare_datasets(
    training_dir: Path = Path('training'),
    chunk_size: int = 14,
    overlap: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    exclude_titles: Optional[List[str]] = None,
    # Embedding-based hard negatives
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    # Fraction of easy random negatives to keep for regularization
    random_neg_frac: float = 0.10,
    # Optional: model-mined negatives (expensive on CPU; use conservatively)
    use_model_mined_negatives: bool = True,
    miner_model: str = 'contrastive',
    miner_model_dir: Optional[str] = None,
    n_mined_trials: int = 200,
    n_mined_keep: int = 20,
    # Optional: ANN chunk-level hard negatives via contrastive embeddings
    # GPU-friendly two-stage miner (restricted to neighbor books)
    use_ann_chunk_negatives: bool = False,
    ann_miner_model_dir: Optional[str] = None,
    ann_k_neighbors: int = 20,
    ann_sim_threshold: float = 0.55,
    ann_prob_max: float = 0.20,
    ann_anchors_per_book: int = 120,
    ann_pool_samples_per_book: int = 200,
    ann_batch_size: int = 32,
    ann_max_negatives_per_book: int = 100,
    ann_max_total_negatives: Optional[int] = None,
    # Cross-book chunk deduplication (fingerprint-based)
    dedup_across_books: bool = True,
    dedup_threshold: int = 3,
    # Parallelism for per-book cleaning/splitting/chunking (None → auto)
    workers: Optional[int] = None,
    # Cross-book embedding batching controls (GPU only)
    embed_microbatch: int = 512,
    embed_books_per_pass: int = 2000,
) -> DatasetDict:
    """
    Prepare train/val/test datasets with positive and negative pairs.
    
    Args:
        training_dir: Directory containing book text files
        chunk_size: Number of sentences per chunk
        overlap: Overlap between chunks
        train_ratio: Ratio of books for training
        val_ratio: Ratio of books for validation
        max_chunks_per_book: Maximum chunks to sample per book
        use_hard_negatives: If True, sample negatives from similar books
        n_positive_per_book: Number of positive pairs per book
        n_negative_per_book: Number of negative pairs per book
        random_neg_frac: Share of easy random negatives to include (0–0.9)
    
    Returns:
        DatasetDict with train/val/test splits
    """
    # Load all books (search recursively for .txt files)
    def _norm_title(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    excl_norm: Set[str] = set()
    if exclude_titles:
        try:
            excl_norm = {_norm_title(t) for t in exclude_titles if t and t.strip()}
        except Exception:
            excl_norm = set()
    book_files = list(training_dir.rglob('*.txt'))
    if excl_norm:
        book_files = [p for p in book_files if _norm_title(p.stem) not in excl_norm]
    print(f"Found {len(book_files)} books")
    
    # Process books into chunks (parallel by default)
    book_chunks: Dict[str, List[str]] = {}
    # Resolve worker count (auto: up to 32 or CPU count)
    try:
        auto_workers = min(os.cpu_count() or 8, 32)
    except Exception:
        auto_workers = 8
    eff_workers = workers if (workers and workers > 0) else auto_workers
    eff_workers = min(max(1, eff_workers), len(book_files) or 1)
    
    if eff_workers > 1 and len(book_files) > 1:
        print(f"Using {eff_workers} CPU workers for chunking...")
        jobs = [
            (str(p), chunk_size, overlap, max_chunks_per_book, use_hard_negatives)
            for p in book_files
        ]
        done = 0
        report_every = max(100, len(jobs) // 20)
        with ProcessPoolExecutor(max_workers=eff_workers) as ex:
            futures = [ex.submit(_process_book_worker, j) for j in jobs]
            for fut in as_completed(futures):
                status, bid, chunks, md = fut.result()
                done += 1
                if status == "ok":
                    book_chunks[bid] = chunks
                    # no metadata collection
                    print(f"  {bid}: {len(chunks)} chunks")
                elif status == "error":
                    print(f"Error processing {bid}: {md}")
                # progress
                if done % report_every == 0:
                    print(f"Processed {done}/{len(jobs)} books...")
    else:
        for book_file in book_files:
            author_id = book_file.parent.name
            book_id = f"{author_id}/{book_file.stem}"
            try:
                sentences = load_book(book_file)
                chunks = create_text_chunks(sentences, chunk_size, overlap)
                if chunks:
                    if len(chunks) > max_chunks_per_book:
                        chunks = random.sample(chunks, max_chunks_per_book)
                    book_chunks[book_id] = chunks
                    # no metadata collection
                    print(f"  {book_id}: {len(chunks)} chunks")
            except Exception as e:
                print(f"Error processing {book_id}: {e}")

    # Simple cross-book deduplication by fingerprinting chunks
    if dedup_across_books and book_chunks:
        import hashlib
        def _fingerprint(s: str) -> str:
            # Normalize: lowercase, remove punctuation, collapse whitespace
            s_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", s.lower())).strip()
            return hashlib.sha1(s_norm.encode("utf-8", errors="ignore")).hexdigest()

        fp_counts: Dict[str, int] = defaultdict(int)
        per_book_fps: Dict[str, List[str]] = {}
        for bid, chunks in book_chunks.items():
            fps = [_fingerprint(c) for c in chunks]
            per_book_fps[bid] = fps
            for fp in set(fps):
                fp_counts[fp] += 1
        removed_total = 0
        for bid, chunks in list(book_chunks.items()):
            fps = per_book_fps.get(bid, [])
            keep = [c for c, fp in zip(chunks, fps) if fp_counts.get(fp, 0) <= dedup_threshold]
            removed = len(chunks) - len(keep)
            if removed > 0:
                removed_total += removed
                book_chunks[bid] = keep
        if removed_total > 0:
            print(f"Dedup: removed {removed_total} repeated chunks across books (threshold>{dedup_threshold}).")
    
    # Optional: embedding-based hard negatives (compute book-level embeddings)
    book_neighbors: Dict[str, List[str]] = {}
    if use_embedding_hard_negatives and len(book_chunks) > 1:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(embedding_model, device=device)
            model.max_seq_length = 256
            book_ids_all = list(book_chunks.keys())
            # Cross-book batching on GPU (supports single- and multi-GPU)
            if device == 'cuda':
                micro = max(64, int(embed_microbatch))
                bpp = max(256, int(embed_books_per_pass))
                n_gpus = int(torch.cuda.device_count())
                use_multi = n_gpus > 1
                if use_multi:
                    targets = [f"cuda:{i}" for i in range(n_gpus)]
                    print(f"Encoding book centroids on {n_gpus} GPUs with microbatch={micro}, books_per_pass={bpp} ...")
                    pool = model.start_multi_process_pool(target_devices=targets)
                else:
                    print(f"Encoding book centroids on GPU with microbatch={micro}, books_per_pass={bpp} ...")
                    pool = None

                centroids: List[np.ndarray] = []
                try:
                    for start in range(0, len(book_ids_all), bpp):
                        end = min(len(book_ids_all), start + bpp)
                        ids_pass = book_ids_all[start:end]
                        texts: List[str] = []
                        counts: List[int] = []
                        for bid in ids_pass:
                            sample = book_chunks[bid][:num_chunks_for_embed]
                            texts.extend(sample)
                            counts.append(len(sample))
                        if not texts:
                            continue
                        if use_multi and pool is not None:
                            embs_pass = model.encode_multi_process(
                                texts,
                                pool,
                                batch_size=micro,
                                normalize_embeddings=True,
                                show_progress_bar=False,
                            )
                        else:
                            embs_pass = model.encode(
                                texts,
                                convert_to_numpy=True,
                                batch_size=micro,
                                normalize_embeddings=True,
                                show_progress_bar=False,
                            )
                        if embs_pass.dtype != np.float32:
                            embs_pass = embs_pass.astype(np.float32)
                        idx = 0
                        for c in counts:
                            if c <= 0:
                                cent = np.zeros((embs_pass.shape[1],), dtype=np.float32)
                            else:
                                cent = np.mean(embs_pass[idx:idx + c], axis=0, dtype=np.float32)
                            centroids.append(cent)
                            idx += c
                        print(f"  encoded {end}/{len(book_ids_all)} books...")
                finally:
                    if use_multi and pool is not None:
                        try:
                            model.stop_multi_process_pool(pool)
                        except Exception:
                            pass
                book_embs = np.vstack(centroids).astype(np.float32, copy=False)
            else:
                # CPU path: keep per-book encode with modest batch size
                book_embs = []
                bs = 64
                print(f"Encoding book centroids on CPU (batch_size={bs}) ...")
                for i_bid, bid in enumerate(book_ids_all):
                    sample = book_chunks[bid][:num_chunks_for_embed]
                    emb = model.encode(sample, convert_to_numpy=True, batch_size=bs, normalize_embeddings=True)
                    if emb.dtype != np.float32:
                        emb = emb.astype(np.float32)
                    centroid = np.mean(emb, axis=0, dtype=np.float32)
                    book_embs.append(centroid)
                    if (i_bid + 1) % 500 == 0:
                        print(f"  encoded {i_bid + 1}/{len(book_ids_all)} books...")
                book_embs = np.vstack(book_embs).astype(np.float32, copy=False)
            # Top-K neighbors without materializing NxN matrix
            neigh = _topk_neighbors(book_embs, int(num_hard_negative_books))
            for i, bid in enumerate(book_ids_all):
                book_neighbors[bid] = [book_ids_all[j] for j in neigh.get(i, []) if j != i][:int(num_hard_negative_books)]
            print("Computed embedding-based neighbors for hard negatives (top-K).")
        except Exception as e:
            print(f"Embedding-based hard negatives disabled due to error: {e}")
            book_neighbors = {}

    # Split books into train/val/test
    book_ids = list(book_chunks.keys())
    random.shuffle(book_ids)
    
    n_train = int(len(book_ids) * train_ratio)
    n_val = int(len(book_ids) * val_ratio)
    
    train_books = set(book_ids[:n_train])
    val_books = set(book_ids[n_train:n_train + n_val])
    test_books = set(book_ids[n_train + n_val:])
    
    print(f"\nSplit: {len(train_books)} train, {len(val_books)} val, {len(test_books)} test books")
    
    # ---------------------- Topic heuristics for metadata ----------------------
    _TOPIC_VOCAB = [
        'religious', 'historical', 'adventure', 'romance', 'general'
    ]
    _TOPIC_KEYWORDS = {
        'religious': set(['god', 'church', 'prayer', 'scripture', 'lord', 'faith', 'holy', 'sacred', 'divine']),
        'historical': set(['prince', 'king', 'castle', 'knight', 'lord', 'duke', 'empire', 'queen', 'court']),
        'adventure': set(['captain', 'ship', 'sea', 'voyage', 'island', 'expedition', 'jungle', 'desert', 'treasure']),
        'romance': set(['love', 'heart', 'soul', 'passion', 'beloved', 'darling', 'kiss', 'romance', 'marriage']),
    }

    def _label_topic(text: str) -> int:
        tl = text.lower()
        scores = {k: 0 for k in _TOPIC_KEYWORDS.keys()}
        for topic, keys in _TOPIC_KEYWORDS.items():
            scores[topic] = sum(tl.count(w) for w in keys)
        if scores and max(scores.values()) > 0:
            best = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            best = 'general'
        return _TOPIC_VOCAB.index(best)

    # Cache for heavy miners to avoid reloading per-book
    _miner_cache: Dict[str, object] = {}

    # Create pairs for each split
    def create_pairs(book_set: set):
        """Create positive and negative pairs."""
        pairs = []
        rng = random.Random(42)
        total_books = len(book_set)
        done = 0
        report_every = max(250, total_books // 20) if total_books else 1
        
        for book_id in book_set:
            chunks = book_chunks[book_id]
            author_id = book_id.split('/', 1)[0] if '/' in book_id else ''
            # Disable author-based negatives for anonymous/unknown-style buckets
            if _is_anonymous_author(author_id):
                author_id = ''
            if len(chunks) < 2:
                continue
            
            # Positive pairs (same book) - use non-overlapping chunks to prevent data leakage
            for _ in range(n_positive_per_book):
                if len(chunks) >= 2:
                    try:
                        chunk1, chunk2 = sample_non_overlapping_pair(
                            chunks, chunk_size=chunk_size, overlap=overlap, rng=rng
                        )
                    except ValueError:
                        # Fallback if not enough chunks
                        chunk1, chunk2 = random.sample(chunks, 2)
                    t1 = _label_topic(chunk1)
                    t2 = _label_topic(chunk2)
                    pairs.append({
                        'text1': chunk1,
                        'text2': chunk2,
                        'label': 1,
                        'book1': book_id,
                        'book2': book_id,
                        'pair_type': 'positive',
                        'author1': author_id,
                        'author2': author_id,
                        'same_author': True,
                        'topic1': int(t1),
                        'topic2': int(t2),
                        'same_topic': bool(t1 == t2),
                    })
            
            # Negative pairs (tiered hardness):
            # - author_same: same author, different book (very hard)
            # - embed_neighbor: nearest by embeddings (hard)
            # - random: random different book (easy)
            easy_books = [bid for bid in book_set if bid != book_id]
            author_id = book_id.split('/', 1)[0] if '/' in book_id else ''
            author_books: List[str] = []
            if author_id:
                author_books = [
                    bid for bid in easy_books
                    if ('/' in bid and bid.split('/', 1)[0] == author_id)
                ]
            medium_books: List[str] = []
            hard_books: List[str] = []
            neg_types = ['random', 'author_same', 'embed_neighbor']
            if use_embedding_hard_negatives and book_neighbors.get(book_id):
                hard_books = [b for b in book_neighbors.get(book_id, []) if b in book_set and b != book_id]

            # Random regularization + split remaining mass across available hard sources
            rf = float(max(0.0, min(0.9, random_neg_frac)))
            w_map = {k: 0.0 for k in neg_types}
            if author_books and hard_books:
                rem = 1.0 - rf
                w_map['random'] = rf
                w_map['author_same'] = rem / 2.0
                w_map['embed_neighbor'] = rem / 2.0
            elif author_books:
                w_map['random'] = rf
                w_map['author_same'] = 1.0 - rf
            elif hard_books:
                w_map['random'] = rf
                w_map['embed_neighbor'] = 1.0 - rf
            else:
                w_map['random'] = 1.0
            weights = [w_map[t] for t in neg_types]

            # Create negative pairs
            for _ in range(n_negative_per_book):
                choice = rng.choices(neg_types, weights=weights, k=1)[0]
                if choice == 'embed_neighbor' and hard_books:
                    other_book_id = rng.choice(hard_books)
                elif choice == 'author_same' and author_books:
                    other_book_id = rng.choice(author_books)
                else:
                    other_book_id = rng.choice(easy_books)
                    choice = 'random'
                chunk1 = rng.choice(chunks)
                chunk2 = rng.choice(book_chunks[other_book_id])
                t1 = _label_topic(chunk1)
                t2 = _label_topic(chunk2)
                pairs.append({
                    'text1': chunk1,
                    'text2': chunk2,
                    'label': 0,
                    'book1': book_id,
                    'book2': other_book_id,
                    'pair_type': 'negative',
                    'neg_type': choice,
                    'author1': author_id,
                    'author2': other_book_id.split('/', 1)[0] if '/' in other_book_id else '',
                    'same_author': bool(author_id and author_id == (other_book_id.split('/', 1)[0] if '/' in other_book_id else '')),
                    'topic1': int(t1),
                    'topic2': int(t2),
                    'same_topic': bool(t1 == t2),
                })

            # Optional: add model-mined hard negatives (choose high-scoring false pairs)
            # Build a candidate set favoring hard/medium books, with random as fallback
            negative_candidates = []
            try:
                # Prioritize harder candidates when available
                seen = set()
                for lst in (hard_books, medium_books, easy_books):
                    for b in lst:
                        if b not in seen:
                            negative_candidates.append(b)
                            seen.add(b)
            except Exception:
                negative_candidates = [bid for bid in book_set if bid != book_id]

            if use_model_mined_negatives and negative_candidates:
                try:
                    # Lazy import to keep base prepare fast
                    scored = []  # (chunk1, chunk2, other_book_id)
                    # Select a small pool of candidates uniformly
                    trials = min(n_mined_trials, len(negative_candidates) * 5)
                    for _ in range(trials):
                        ob = rng.choice(negative_candidates)
                        c1 = rng.choice(chunks)
                        c2 = rng.choice(book_chunks[ob])
                        scored.append((c1, c2, ob))

                    if scored:
                        # Score pairs with contrastive embedder (cosine similarity)
                        from hard_negative_mining import ContrastiveChunkEmbedder
                        def _pick_dir(default_local: str, default_vol: str) -> str:
                            import os
                            if miner_model_dir:
                                return miner_model_dir
                            try:
                                if os.path.exists(os.path.join(default_vol, 'pytorch_model.bin')) or os.path.exists(os.path.join(default_vol, 'model.safetensors')):
                                    return default_vol
                            except Exception:
                                pass
                            return default_local
                        mdir = _pick_dir('models/book_matcher_contrastive/final', '/vol/models/book_matcher_contrastive/final')
                        embedder = _miner_cache.get('contrastive_embedder')  # type: ignore[assignment]
                        if embedder is None:
                            try:
                                from transformers.utils import logging as _hf_logging
                                _hf_logging.set_verbosity_error()
                            except Exception:
                                pass
                            embedder = ContrastiveChunkEmbedder(mdir)
                            _miner_cache['contrastive_embedder'] = embedder
                        pairs_infer = [(s1, s2) for (s1, s2, _ob) in scored]
                        probs = embedder.pair_probs(pairs_infer, batch_size=64)

                        # Take top-k by similarity (hardest negatives)
                        order = np.argsort(-np.array(probs))
                        k = min(n_mined_keep, len(order))
                        for idx in order[:k]:
                            s1, s2, ob = scored[idx]
                            t1 = _label_topic(s1)
                            t2 = _label_topic(s2)
                            pairs.append({
                                'text1': s1,
                                'text2': s2,
                                'label': 0,
                                'book1': book_id,
                                'book2': ob,
                                'pair_type': 'negative',
                                'neg_type': 'model_mined',
                                'topic1': int(t1),
                                'topic2': int(t2),
                                'same_topic': bool(t1 == t2),
                            })
                except Exception as e:
                    print(f"Model-mined negatives skipped due to error: {e}")

            # Per-book progress
            done += 1
            if done % report_every == 0:
                print(f"Pairing progress: {done}/{total_books} books, pairs={len(pairs)}")
        
        # GPU-friendly ANN per-book miner restricted to neighbor books
        if use_ann_chunk_negatives and book_set and ann_miner_model_dir:
            try:
                import torch  # type: ignore
                from hard_negative_mining import ContrastiveChunkEmbedder
            except Exception as e:
                print(f"ANN mining unavailable (deps): {e}")
            else:
                try:
                    embedder = ContrastiveChunkEmbedder(model_dir=ann_miner_model_dir)
                except Exception as e:
                    print(f"ANN embedder init failed: {e}")
                    embedder = None
                if embedder is not None:
                    import math as _math
                    import threading as _threading
                    from contextlib import nullcontext as _nullcontext
                    n_gpus = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
                    ann_bs = max(8, int(ann_batch_size))

                    # Cache pool embeddings per neighbor book to avoid re-embedding across targets
                    pool_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
                    pool_lock = _threading.Lock()

                    def _get_pool_cached(nb: str, emb: ContrastiveChunkEmbedder, per_nb: int) -> Tuple[np.ndarray, List[str]]:
                        try:
                            with pool_lock:
                                cached = pool_cache.get(nb)
                            if cached is not None:
                                return cached
                        except Exception:
                            pass
                        # Deterministic sampling per neighbor book
                        chs = book_chunks.get(nb, [])
                        if not chs:
                            return np.zeros((0, 1), dtype=np.float32), []
                        if len(chs) > per_nb:
                            seed = (abs(hash(nb)) % (2**32))
                            rng_local = np.random.default_rng(seed)
                            idx = rng_local.choice(len(chs), size=per_nb, replace=False)
                            pts = [chs[i] for i in idx]
                        else:
                            pts = list(chs)
                        E = emb.embed_texts(pts, batch_size=ann_bs)
                        try:
                            with pool_lock:
                                pool_cache[nb] = (E, pts)
                        except Exception:
                            pass
                        return E, pts

                    def _process_one(book_id: str, emb) -> List[Dict[str, object]]:
                        out: List[Dict[str, object]] = []
                        anchors_src = book_chunks.get(book_id, [])
                        if not anchors_src:
                            return out
                        # Sample anchors
                        A = max(1, int(ann_anchors_per_book))
                        if len(anchors_src) > A:
                            idx = np.random.choice(len(anchors_src), size=A, replace=False)
                            anchors = [anchors_src[i] for i in idx]
                        else:
                            anchors = list(anchors_src)
                        # Gather pool from neighbor books (restricted)
                        neigh_books = [b for b in (book_neighbors.get(book_id, []) or []) if b in book_set and b != book_id]
                        if ann_k_neighbors > 0:
                            neigh_books = neigh_books[: int(ann_k_neighbors)]
                        pool_texts: List[str] = []
                        pool_meta: List[str] = []  # neighbor book id per chunk
                        per_nb = max(1, int(ann_pool_samples_per_book))
                        for nb in neigh_books:
                            chs = book_chunks.get(nb, [])
                            if not chs:
                                continue
                            if len(chs) > per_nb:
                                sel = np.random.choice(len(chs), size=per_nb, replace=False)
                                pts = [chs[i] for i in sel]
                            else:
                                pts = list(chs)
                            pool_texts.extend(pts)
                            pool_meta.extend([nb] * len(pts))
                        if not pool_texts:
                            return out
                        # Embed anchors and pool (pool may be cached per neighbor book)
                        E_a = emb.embed_texts(anchors, batch_size=ann_bs)
                        # Assemble pool embeddings from cached per-neighbor chunks
                        E_list: List[np.ndarray] = []
                        pool_texts = []
                        pool_meta = []
                        for nb in neigh_books:
                            E_nb, T_nb = _get_pool_cached(nb, emb, per_nb)
                            if E_nb.size == 0:
                                continue
                            E_list.append(E_nb)
                            pool_texts.extend(T_nb)
                            pool_meta.extend([nb] * len(T_nb))
                        E_p = np.vstack(E_list) if E_list else np.zeros((0, 1), dtype=np.float32)
                        if E_a.size == 0 or E_p.size == 0:
                            return out
                        # Compute sims on device if available
                        try:
                            device = emb.device if hasattr(emb, 'device') else torch.device('cpu')
                            Ea = torch.from_numpy(E_a).to(device=device)
                            Ep = torch.from_numpy(E_p).to(device=device)
                            # bf16 autocast for faster matmuls on CUDA
                            _ctx = (
                                torch.autocast(device_type='cuda', dtype=torch.bfloat16)  # type: ignore[attr-defined]
                                if device.type == 'cuda' else _nullcontext()
                            )
                            with _ctx:
                                sims = Ea @ Ep.T
                            sims_cpu = sims.detach().float().cpu().numpy()
                            del sims
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                        except Exception:
                            sims_cpu = E_a @ E_p.T
                        # Select candidates above threshold (top-1 per anchor)
                        cand_pairs: List[Tuple[str, str, str]] = []  # (t1, t2, nb_book)
                        for i in range(sims_cpu.shape[0]):
                            row = sims_cpu[i]
                            j = int(np.argmax(row))
                            if float(row[j]) < float(ann_sim_threshold):
                                continue
                            t1 = anchors[i]
                            t2 = pool_texts[j]
                            cand_pairs.append((t1, t2, pool_meta[j]))
                        if not cand_pairs:
                            return out
                        # Rescore with model head and filter by prob_max
                        pairs_infer = [(a, b) for (a, b, _nb) in cand_pairs]
                        probs = emb.pair_probs(pairs_infer, batch_size=ann_bs)
                        kept = 0
                        for (t1, t2, nb), p in zip(cand_pairs, probs):
                            if float(p) <= float(ann_prob_max):
                                t1t = _label_topic(t1)
                                t2t = _label_topic(t2)
                                out.append({
                                    'text1': t1,
                                    'text2': t2,
                                    'label': 0,
                                    'book1': book_id,
                                    'book2': nb,
                                    'pair_type': 'negative',
                                    'neg_type': 'ann_mined',
                                    'author1': (book_id.split('/',1)[0] if '/' in book_id else ''),
                                    'author2': (nb.split('/',1)[0] if '/' in nb else ''),
                                    'same_author': bool(('/' in book_id) and ('/' in nb) and (book_id.split('/',1)[0] == nb.split('/',1)[0])),
                                    'topic1': int(t1t),
                                    'topic2': int(t2t),
                                    'same_topic': bool(t1t == t2t),
                                })
                                kept += 1
                                if kept >= int(ann_max_negatives_per_book):
                                    break
                        return out

                    total_ann = 0
                    books_list = list(book_set)
                    if n_gpus <= 1:
                        # Single device path with periodic progress
                        progress_every = max(500, len(books_list) // 20)
                        done = 0
                        for bid in books_list:
                            try:
                                out_pairs = _process_one(bid, embedder)
                                if out_pairs:
                                    pairs.extend(out_pairs)
                                    total_ann += len(out_pairs)
                            except Exception:
                                pass
                            done += 1
                            if done % progress_every == 0:
                                print(f"ANN mining progress: {done}/{len(books_list)} books, ann_pairs={total_ann}")
                    else:
                        # Multi-GPU with thread workers, one embedder per device
                        try:
                            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
                        except Exception:
                            ThreadPoolExecutor = None  # type: ignore
                        embedders = {}
                        for i in range(n_gpus):
                            try:
                                embedders[i] = ContrastiveChunkEmbedder(model_dir=ann_miner_model_dir, device=f"cuda:{i}")
                            except Exception:
                                embedders[i] = embedder  # fallback to default
                        progress_every = max(500, len(books_list) // 20)
                        done = 0
                        if ThreadPoolExecutor is None:
                            # Fallback to round-robin single-thread loop
                            for idx, bid in enumerate(books_list):
                                dev = idx % n_gpus
                                try:
                                    out_pairs = _process_one(bid, embedders.get(dev, embedder))
                                    if out_pairs:
                                        pairs.extend(out_pairs)
                                        total_ann += len(out_pairs)
                                except Exception:
                                    pass
                                done += 1
                                if done % progress_every == 0:
                                    print(f"ANN mining progress: {done}/{len(books_list)} books, ann_pairs={total_ann}")
                        else:
                            with ThreadPoolExecutor(max_workers=n_gpus) as ex:
                                futs = []
                                for idx, bid in enumerate(books_list):
                                    dev = idx % n_gpus
                                    emb_i = embedders.get(dev, embedder)
                                    futs.append(ex.submit(_process_one, bid, emb_i))
                                for fut in _as_completed(futs):
                                    try:
                                        out_pairs = fut.result()
                                        if out_pairs:
                                            pairs.extend(out_pairs)
                                            total_ann += len(out_pairs)
                                    except Exception:
                                        pass
                                    done += 1
                                    if done % progress_every == 0:
                                        print(f"ANN mining progress: {done}/{len(books_list)} books, ann_pairs={total_ann}")
                    if total_ann > 0:
                        print(f"ANN (GPU) mining added {total_ann} negatives.")

        return pairs
    
    train_pairs = create_pairs(train_books)
    val_pairs = create_pairs(val_books)
    test_pairs = create_pairs(test_books)
    
    print(f"\nCreated pairs:")
    print(f"  Train: {len(train_pairs)} ({sum(p['label'] for p in train_pairs)} pos, {sum(1-p['label'] for p in train_pairs)} neg)")
    print(f"  Val: {len(val_pairs)} ({sum(p['label'] for p in val_pairs)} pos, {sum(1-p['label'] for p in val_pairs)} neg)")
    print(f"  Test: {len(test_pairs)} ({sum(p['label'] for p in test_pairs)} pos, {sum(1-p['label'] for p in test_pairs)} neg)")
    
    # Guard against empty datasets to provide a clearer error message
    if len(train_pairs) == 0 and len(val_pairs) == 0 and len(test_pairs) == 0:
        raise ValueError(
            "No training examples were created. Ensure your training_dir contains .txt files. "
            "If you are using a Modal Volume, verify files exist at /input/training (e.g., `modal volume ls writing7-training /training`)."
        )

    # Create HuggingFace datasets
    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)
    test_dataset = Dataset.from_list(test_pairs)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })


# ------------------------------ Sharding helpers ------------------------------
def _save_shard_jsonl(book_chunks: Dict[str, List[str]], book_metadata: Dict[str, Dict[str, str]], out_path: Path) -> None:
    """Save pre-chunked results to a JSONL shard.

    Each line: {"book_id": str, "chunks": List[str], "metadata": Dict[str,str]}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for bid, chunks in book_chunks.items():
            md = book_metadata.get(bid, {})
            rec = {"book_id": bid, "chunks": chunks, "metadata": md}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # Atomic move avoids partial reads by other processes
    try:
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        os.rename(tmp_path, out_path)


def chunk_books_to_jsonl(
    base_dir: Path,
    rel_paths: List[str],
    out_jsonl: Path,
    chunk_size: int = 14,
    overlap: int = 4,
    max_chunks_per_book: int = 800,
    use_hard_negatives: bool = True,
    workers: Optional[int] = None,
    exclude_titles: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Process a subset of book files and write a JSONL shard with chunks and metadata.

    Returns a small dict of counts for diagnostics.
    """
    def _norm_title(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())

    excl_norm: Set[str] = set()
    if exclude_titles:
        try:
            excl_norm = {_norm_title(t) for t in exclude_titles if t and t.strip()}
        except Exception:
            excl_norm = set()

    files = [base_dir / p for p in rel_paths]
    if excl_norm:
        files = [p for p in files if _norm_title(p.stem) not in excl_norm]
    book_chunks: Dict[str, List[str]] = {}
    # no metadata collection for shards

    # Resolve worker count
    try:
        auto_workers = min(os.cpu_count() or 8, 32)
    except Exception:
        auto_workers = 8
    eff_workers = workers if (workers and workers > 0) else auto_workers
    eff_workers = min(max(1, eff_workers), len(files) or 1)

    if eff_workers > 1 and len(files) > 1:
        jobs = [
            (str(p), chunk_size, overlap, max_chunks_per_book, use_hard_negatives)
            for p in files
        ]
        with ProcessPoolExecutor(max_workers=eff_workers) as ex:
            futures = [ex.submit(_process_book_worker, j) for j in jobs]
            for fut in as_completed(futures):
                status, bid, chunks, md = fut.result()
                if status == "ok":
                    book_chunks[bid] = chunks
                    # no metadata collection
                elif status == "error":
                    # Keep going; shard write should proceed
                    pass
    else:
        for p in files:
            author_id = p.parent.name
            bid = f"{author_id}/{p.stem}"
            try:
                sentences = load_book(p)
                chunks = create_text_chunks(sentences, chunk_size, overlap)
                if chunks:
                    if len(chunks) > max_chunks_per_book:
                        chunks = random.sample(chunks, max_chunks_per_book)
                    book_chunks[bid] = chunks
                    # no metadata collection
            except Exception:
                pass

    _save_shard_jsonl(book_chunks, {}, out_jsonl)
    return {"books": len(book_chunks), "chunks": sum(len(v) for v in book_chunks.values())}


def load_shards_jsonl(shard_paths: List[Path]) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """Load multiple JSONL shard files back into in-memory dicts."""
    all_chunks: Dict[str, List[str]] = {}
    all_md: Dict[str, Dict[str, str]] = {}
    for sp in shard_paths:
        try:
            with sp.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        bid = rec.get("book_id")
                        if not bid:
                            continue
                        ch = rec.get("chunks") or []
                        md = rec.get("metadata") or {}
                        # Last writer wins on duplicates; should be unique by design
                        all_chunks[bid] = list(ch)
                        if md:
                            all_md[bid] = dict(md)
                    except Exception:
                        continue
        except Exception:
            continue
    return all_chunks, all_md


def prepare_datasets_from_prechunked(
    book_chunks: Dict[str, List[str]],
    book_metadata: Dict[str, Dict[str, str]] | None = None,
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    use_hard_negatives: bool = True,
    exclude_titles: Optional[List[str]] = None,
    # Embedding-based hard negatives
    use_embedding_hard_negatives: bool = True,
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
    num_chunks_for_embed: int = 80,
    num_hard_negative_books: int = 50,
    n_positive_per_book: int = 20,
    n_negative_per_book: int = 40,
    # Optional: model-mined negatives (expensive on CPU; use conservatively)
    use_model_mined_negatives: bool = True,
    miner_model: str = 'contrastive',
    miner_model_dir: Optional[str] = None,
    n_mined_trials: int = 200,
    n_mined_keep: int = 20,
    # Optional: ANN chunk-level hard negatives via contrastive embeddings
    use_ann_chunk_negatives: bool = False,
    ann_miner_model_dir: Optional[str] = None,
    ann_k_neighbors: int = 20,
    ann_sim_threshold: float = 0.55,
    ann_prob_max: float = 0.20,
    ann_anchors_per_book: int = 120,
    ann_pool_samples_per_book: int = 200,
    ann_batch_size: int = 32,
    ann_max_negatives_per_book: int = 100,
    ann_max_total_negatives: Optional[int] = None,
    # Cross-book chunk deduplication (fingerprint-based)
    dedup_across_books: bool = True,
    dedup_threshold: int = 3,
    # Cross-book embedding batching controls (GPU only)
    embed_microbatch: int = 512,
    embed_books_per_pass: int = 2000,
) -> DatasetDict:
    """Finish dataset build from pre-chunked books.

    Mirrors the latter half of prepare_datasets, including dedup, neighbors and pair creation.
    """
    book_metadata = book_metadata or {}

    # Optionally drop excluded titles (match on normalized title component from book_id 'author/title')
    if exclude_titles:
        def _norm_title(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", s.lower())
        excl = {_norm_title(t) for t in exclude_titles if t and t.strip()}
        if excl:
            drop = []
            for bid in list(book_chunks.keys()):
                title = bid.split('/', 1)[1] if '/' in bid else bid
                if _norm_title(title) in excl:
                    drop.append(bid)
            for bid in drop:
                book_chunks.pop(bid, None)
                if bid in book_metadata:
                    book_metadata.pop(bid, None)

    # Simple cross-book deduplication by fingerprinting chunks
    if dedup_across_books and book_chunks:
        import hashlib
        def _fingerprint(s: str) -> str:
            # Normalize: lowercase, remove punctuation, collapse whitespace
            s_norm = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", s.lower())).strip()
            return hashlib.sha1(s_norm.encode("utf-8", errors="ignore")).hexdigest()

        fp_counts: Dict[str, int] = defaultdict(int)
        per_book_fps: Dict[str, List[str]] = {}
        for bid, chunks in book_chunks.items():
            fps = [_fingerprint(c) for c in chunks]
            per_book_fps[bid] = fps
            for fp in set(fps):
                fp_counts[fp] += 1
        removed_total = 0
        for bid, chunks in list(book_chunks.items()):
            fps = per_book_fps.get(bid, [])
            keep = [c for c, fp in zip(chunks, fps) if fp_counts.get(fp, 0) <= dedup_threshold]
            removed = len(chunks) - len(keep)
            if removed > 0:
                removed_total += removed
                book_chunks[bid] = keep
        if removed_total > 0:
            print(f"Dedup: removed {removed_total} repeated chunks across books (threshold>{dedup_threshold}).")

    # Optional: embedding-based hard negatives (compute book-level embeddings)
    book_neighbors: Dict[str, List[str]] = {}
    if use_embedding_hard_negatives and len(book_chunks) > 1:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = SentenceTransformer(embedding_model, device=device)
            model.max_seq_length = 256
            book_ids_all = list(book_chunks.keys())
            if device == 'cuda':
                micro = max(64, int(embed_microbatch))
                bpp = max(256, int(embed_books_per_pass))
                print(f"Encoding book centroids on GPU with microbatch={micro}, books_per_pass={bpp} ...")
                centroids: List[np.ndarray] = []
                for start in range(0, len(book_ids_all), bpp):
                    end = min(len(book_ids_all), start + bpp)
                    ids_pass = book_ids_all[start:end]
                    texts: List[str] = []
                    counts: List[int] = []
                    for bid in ids_pass:
                        sample = book_chunks[bid][:num_chunks_for_embed]
                        texts.extend(sample)
                        counts.append(len(sample))
                    if not texts:
                        continue
                    embs_pass = model.encode(texts, convert_to_numpy=True, batch_size=micro, normalize_embeddings=True)
                    if embs_pass.dtype != np.float32:
                        embs_pass = embs_pass.astype(np.float32)
                    idx = 0
                    for c in counts:
                        if c <= 0:
                            cent = np.zeros((embs_pass.shape[1],), dtype=np.float32)
                        else:
                            cent = np.mean(embs_pass[idx:idx + c], axis=0, dtype=np.float32)
                        centroids.append(cent)
                        idx += c
                    print(f"  encoded {end}/{len(book_ids_all)} books...")
                book_embs = np.vstack(centroids).astype(np.float32, copy=False)
            else:
                book_embs = []
                bs = 64
                print(f"Encoding book centroids on CPU (batch_size={bs}) ...")
                for i_bid, bid in enumerate(book_ids_all):
                    sample = book_chunks[bid][:num_chunks_for_embed]
                    emb = model.encode(sample, convert_to_numpy=True, batch_size=bs, normalize_embeddings=True)
                    if emb.dtype != np.float32:
                        emb = emb.astype(np.float32)
                    centroid = np.mean(emb, axis=0, dtype=np.float32)
                    book_embs.append(centroid)
                    if (i_bid + 1) % 500 == 0:
                        print(f"  encoded {i_bid + 1}/{len(book_ids_all)} books...")
                book_embs = np.vstack(book_embs).astype(np.float32, copy=False)
            neigh = _topk_neighbors(book_embs, int(num_hard_negative_books))
            for i, bid in enumerate(book_ids_all):
                book_neighbors[bid] = [book_ids_all[j] for j in neigh.get(i, []) if j != i][:int(num_hard_negative_books)]
            print("Computed embedding-based neighbors for hard negatives (top-K).")
        except Exception as e:
            print(f"Embedding-based hard negatives disabled due to error: {e}")
            book_neighbors = {}

    # Split books into train/val/test
    book_ids = list(book_chunks.keys())
    random.shuffle(book_ids)

    n_train = int(len(book_ids) * train_ratio)
    n_val = int(len(book_ids) * val_ratio)

    train_books = set(book_ids[:n_train])
    val_books = set(book_ids[n_train:n_train + n_val])
    test_books = set(book_ids[n_train + n_val:])

    print(f"\nSplit: {len(train_books)} train, {len(val_books)} val, {len(test_books)} test books")

    # ---------------------- Topic heuristics for metadata ----------------------
    _TOPIC_VOCAB = [
        'religious', 'historical', 'adventure', 'romance', 'general'
    ]
    _TOPIC_KEYWORDS = {
        'religious': set(['god', 'church', 'prayer', 'scripture', 'lord', 'faith', 'holy', 'sacred', 'divine']),
        'historical': set(['prince', 'king', 'castle', 'knight', 'lord', 'duke', 'empire', 'queen', 'court']),
        'adventure': set(['captain', 'ship', 'sea', 'voyage', 'island', 'expedition', 'jungle', 'desert', 'treasure']),
        'romance': set(['love', 'heart', 'soul', 'passion', 'beloved', 'darling', 'kiss', 'romance', 'marriage']),
    }

    def _label_topic(text: str) -> int:
        tl = text.lower()
        scores = {k: 0 for k in _TOPIC_KEYWORDS.keys()}
        for topic, keys in _TOPIC_KEYWORDS.items():
            scores[topic] = sum(tl.count(w) for w in keys)
        if scores and max(scores.values()) > 0:
            best = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            best = 'general'
        return _TOPIC_VOCAB.index(best)

    # Create pairs for each split (copy of logic in prepare_datasets)
    def create_pairs(book_set: set):
        pairs = []
        rng = random.Random(42)
        total_books = len(book_set)
        done = 0
        report_every = max(250, total_books // 20) if total_books else 1

        for book_id in book_set:
            chunks = book_chunks[book_id]
            author_id = book_id.split('/', 1)[0] if '/' in book_id else ''
            if _is_anonymous_author(author_id):
                author_id = ''
            if len(chunks) < 2:
                continue

            for _ in range(n_positive_per_book):
                if len(chunks) >= 2:
                    chunk1, chunk2 = random.sample(chunks, 2)
                    t1 = _label_topic(chunk1)
                    t2 = _label_topic(chunk2)
                    pairs.append({
                        'text1': chunk1,
                        'text2': chunk2,
                        'label': 1,
                        'book1': book_id,
                        'book2': book_id,
                        'pair_type': 'positive',
                        'author1': author_id,
                        'author2': author_id,
                        'same_author': True,
                        'topic1': int(t1),
                        'topic2': int(t2),
                        'same_topic': bool(t1 == t2),
                    })

            easy_books = [bid for bid in book_set if bid != book_id]
            author_id = book_id.split('/', 1)[0] if '/' in book_id else ''
            author_books: List[str] = []
            if author_id:
                author_books = [
                    bid for bid in easy_books
                    if ('/' in bid and bid.split('/', 1)[0] == author_id)
                ]
            medium_books: List[str] = []
            hard_books: List[str] = []
            neg_types = ['random', 'author_same', 'embed_neighbor']
            if use_embedding_hard_negatives and (book_id in book_neighbors):
                hard_books = [b for b in book_neighbors.get(book_id, []) if b in book_set and b != book_id]

            # Random regularization + split remaining mass across available hard sources
            rf = float(max(0.0, min(0.9, random_neg_frac)))
            w_map = {k: 0.0 for k in neg_types}
            if author_books and hard_books:
                rem = 1.0 - rf
                w_map['random'] = rf
                w_map['author_same'] = rem / 2.0
                w_map['embed_neighbor'] = rem / 2.0
            elif author_books:
                w_map['random'] = rf
                w_map['author_same'] = 1.0 - rf
            elif hard_books:
                w_map['random'] = rf
                w_map['embed_neighbor'] = 1.0 - rf
            else:
                w_map['random'] = 1.0
            weights = [w_map[t] for t in neg_types]

            for _ in range(n_negative_per_book):
                choice = rng.choices(neg_types, weights=weights, k=1)[0]
                if choice == 'embed_neighbor' and hard_books:
                    other_book_id = rng.choice(hard_books)
                elif choice == 'author_same' and author_books:
                    other_book_id = rng.choice(author_books)
                else:
                    other_book_id = rng.choice(easy_books)
                    choice = 'random'
                chunk1 = rng.choice(chunks)
                chunk2 = rng.choice(book_chunks[other_book_id])
                t1 = _label_topic(chunk1)
                t2 = _label_topic(chunk2)
                pairs.append({
                    'text1': chunk1,
                    'text2': chunk2,
                    'label': 0,
                    'book1': book_id,
                    'book2': other_book_id,
                    'pair_type': 'negative',
                    'neg_type': choice,
                    'author1': author_id,
                    'author2': other_book_id.split('/', 1)[0] if '/' in other_book_id else '',
                    'same_author': bool(author_id and author_id == (other_book_id.split('/', 1)[0] if '/' in other_book_id else '')),
                    'topic1': int(t1),
                    'topic2': int(t2),
                    'same_topic': bool(t1 == t2),
                })

            negative_candidates = []
            try:
                seen = set()
                for lst in (hard_books, medium_books, easy_books):
                    for b in lst:
                        if b not in seen:
                            negative_candidates.append(b)
                            seen.add(b)
            except Exception:
                negative_candidates = [bid for bid in book_set if bid != book_id]

            if use_model_mined_negatives and negative_candidates:
                try:
                    scored = []
                    trials = min(n_mined_trials, len(negative_candidates) * 5)
                    for _ in range(trials):
                        ob = rng.choice(negative_candidates)
                        c1 = rng.choice(chunks)
                        c2 = rng.choice(book_chunks[ob])
                        scored.append((c1, c2, ob))

                    if scored:
                        probs = []
                        # Score pairs with contrastive embedder (cosine similarity)
                        from hard_negative_mining import ContrastiveChunkEmbedder
                        def _pick_dir(default_local: str, default_vol: str) -> str:
                            import os
                            if miner_model_dir:
                                return miner_model_dir
                            try:
                                if os.path.exists(os.path.join(default_vol, 'pytorch_model.bin')) or os.path.exists(os.path.join(default_vol, 'model.safetensors')):
                                    return default_vol
                            except Exception:
                                pass
                            return default_local
                        mdir = _pick_dir('models/book_matcher_contrastive/final', '/vol/models/book_matcher_contrastive/final')
                        embedder = _miner_cache.get('contrastive_embedder')  # type: ignore[assignment]
                        if embedder is None:
                            try:
                                from transformers.utils import logging as _hf_logging
                                _hf_logging.set_verbosity_error()
                            except Exception:
                                pass
                            embedder = ContrastiveChunkEmbedder(mdir)
                            _miner_cache['contrastive_embedder'] = embedder
                        pairs_infer = [(s1, s2) for (s1, s2, _ob) in scored]
                        probs = embedder.pair_probs(pairs_infer, batch_size=64)

                        order = np.argsort(-np.array(probs))
                        k = min(n_mined_keep, len(order))
                        for idx in order[:k]:
                            s1, s2, ob = scored[idx]
                            t1 = _label_topic(s1)
                            t2 = _label_topic(s2)
                            pairs.append({
                                'text1': s1,
                                'text2': s2,
                                'label': 0,
                                'book1': book_id,
                                'book2': ob,
                                'pair_type': 'negative',
                                'neg_type': 'model_mined',
                                'topic1': int(t1),
                                'topic2': int(t2),
                                'same_topic': bool(t1 == t2),
                            })
                except Exception as e:
                    print(f"Model-mined negatives skipped due to error: {e}")

            done += 1
            if done % report_every == 0:
                print(f"Pairing progress: {done}/{total_books} books, pairs={len(pairs)}")

        # ANN experimental miner can be integrated here if desired (omitted to keep sharded path concise)
        return pairs

    train_pairs = create_pairs(train_books)
    val_pairs = create_pairs(val_books)
    test_pairs = create_pairs(test_books)

    print(f"\nCreated pairs:")
    print(f"  Train: {len(train_pairs)} ({sum(p['label'] for p in train_pairs)} pos, {sum(1-p['label'] for p in train_pairs)} neg)")
    print(f"  Val: {len(val_pairs)} ({sum(p['label'] for p in val_pairs)} pos, {sum(1-p['label'] for p in val_pairs)} neg)")
    print(f"  Test: {len(test_pairs)} ({sum(p['label'] for p in test_pairs)} pos, {sum(1-p['label'] for p in test_pairs)} neg)")

    if len(train_pairs) == 0 and len(val_pairs) == 0 and len(test_pairs) == 0:
        raise ValueError(
            "No training examples were created from prechunked data. Ensure shards are non-empty."
        )

    train_dataset = Dataset.from_list(train_pairs)
    val_dataset = Dataset.from_list(val_pairs)
    test_dataset = Dataset.from_list(test_pairs)

    return DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    
    datasets = prepare_datasets()
    
    # Save datasets
    datasets.save_to_disk('data/processed')
    print("\nSaved datasets to data/processed/")
    
    # Show some examples
    print("\nExample training samples:")
    for i in range(min(3, len(datasets['train']))):
        ex = datasets['train'][i]
        print(f"\n{i+1}. Label: {ex['label']}")
        print(f"   Text1: {ex['text1'][:100]}...")
        print(f"   Text2: {ex['text2'][:100]}...")
def _topk_neighbors(embs: np.ndarray, topk: int) -> Dict[int, List[int]]:
    """Compute top-k neighbors by cosine (dot) without materializing NxN.

    Prefers GPU with torch if available; falls back to blocked CPU with argpartition.
    Expects row-normalized embs (but will normalize defensively).
    Returns mapping from row index -> list of neighbor row indices.
    """
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            # Normalize in float32 for numerical stability, then compute sims in bf16 if available
            device = 'cuda'
            E32 = torch.from_numpy(embs).to(device=device, dtype=torch.float32)
            E32 = torch.nn.functional.normalize(E32, p=2, dim=1)
            use_bf16 = torch.cuda.is_bf16_supported()
            sim_dtype = torch.bfloat16 if use_bf16 else torch.float32
            Emat = E32.to(sim_dtype) if sim_dtype != torch.float32 else E32

            n = Emat.shape[0]
            # Larger blocks reduce kernel launches; H200 has ample memory for ~60k books
            block = max(128, min(2048, n))
            out: Dict[int, List[int]] = {}
            for start in range(0, n, block):
                end = min(n, start + block)
                sims = torch.matmul(Emat[start:end], Emat.T)
                # Mask self-similarity to very negative
                idx = torch.arange(start, end, device=Emat.device)
                neg_val = torch.finfo(sims.dtype).min
                sims[torch.arange(end - start, device=Emat.device), idx] = neg_val
                k = min(topk, n - 1)
                _, inds = torch.topk(sims, k=k, dim=1, largest=True, sorted=True)
                inds_cpu = inds.detach().cpu().numpy()
                for row, i in enumerate(range(start, end)):
                    out[i] = inds_cpu[row].tolist()
                del sims, inds
                torch.cuda.synchronize()
            return out
    except Exception:
        pass

    # CPU fallback
    n = embs.shape[0]
    out: Dict[int, List[int]] = {}
    block = max(32, min(256, n))
    for start in range(0, n, block):
        end = min(n, start + block)
        sims = embs[start:end] @ embs.T
        for row, i in enumerate(range(start, end)):
            sims_row = sims[row]
            sims_row[i] = -1e9
            k = min(topk, n - 1)
            idx_part = np.argpartition(-sims_row, k)[:k]
            order = idx_part[np.argsort(-sims_row[idx_part])]
            out[i] = order.tolist()
    return out
