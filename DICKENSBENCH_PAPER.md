# DickensBench: A Style Transfer Benchmark for Evaluating LLM Writing Ability

LLM benchmarks can be scored in one of three ways: through objective measures (whether an answer is correct or a task is completed), through human feedback, or through the "LLM-as-a-judge" method. Popular benchmarks such as SWE-Bench, FrontierMath, and ARC-AGI have the benefit of being able to objectively verify correct and incorrect answers. In the case of writing and creative writing, however, aside from grammatical correctness (generally a non-issue for advanced LLMs) there are no objective criteria that can be used to measure ability.

As a result, benchmarks that seek to measure writing ability or related skills have used some combination of human grading and LLM-as-a-judge to assess models' writing ability. For instance, WritingBench (Wu et al. 2025) uses LLMs to create custom assessment criteria and then has those LLMs grade LLM outputs according to those criteria, before using those assessments to finetune a critic model to assess writing ability. Likewise, the Creative Writing v3 and Longform Writing benchmarks simply use Anthropic's Claude Sonnet 4 model to grade writing outputs.

Both the human feedback and LLM-as-a-judge methods have obvious flaws. Human feedback, while useful, will tend to reward outputs that reflect the stylistic and cultural preferences of human evaluators, which could be expected to penalize unconventional, experimental, or generally "difficult" styles. LLM-as-a-judge has a similar problem: the LLMs used to evaluate LLM outputs will encode LLM preferences, which due to RLHF methods and preferences within training data will tend to be downstream of human evaluator preferences. While this is a valid method for assessing adherence to the style preferences of the median human evaluator, those style preferences are often at odds with what might be considered "good" or "great" writing. Studies of human literary preferences have found that poetry written by OpenAI's GPT-3.5 model is generally preferred to poetry written by prominent poets such as Walt Whitman, T. S. Eliot, and William Shakespeare (Porter and Machery 2024).

DickensBench is created with the aim of providing a rigorous and quantitative metric to assess writing ability. Instead of measuring output quality directly, which will ultimately be a subjective task regardless of the evaluation method, we focus on a proxy metric for writing ability – a model's ability to conduct text style transfer for literary works. (For a survey on text style transfer in deep learning, see Jin et al. 2024.) Our core claim is that an ability to shift registers and convincingly mimic a wide variety of styles – from the long and complex sentences of Charles Dickens to the laconic style of Ernest Hemingway – reflects a model's writing ability in a way that cannot be captured by human or LLM evaluations.

## 1.1 Architecture

Initially, DickensBench consisted of measurements on a wide variety of stylometric techniques borrowed from the field of computational linguistics, such as normalized function-word frequency or normalized punctuation frequency, with the idea of identifying texts as vectors with each metric as a coordinate. However, the huge variance in the importance of different style features across texts made this approach untenable for purposes of measuring style similarity. The failure of this feature engineering approach led us to an approach based in neural representation learning.

Our architecture was based on a contrastive bi-encoder model trained on a large corpus of public-domain English-language text (~58,000 books) with explicit topic-adversarial objectives. The model uses a Siamese twin-encoder design with shared RoBERTa weights to embed text pairs into a learned representation space, where contrastive loss functions pull stylistically similar texts together while pushing stylistically dissimilar texts apart. Gradient reversal layers are used to prevent the model from learning topic-based shortcuts where thematic overlap masquerades as stylistic similarity (i.e., a prevalence of specific words, such as "London" in the work of Charles Dickens). The resulting embeddings are then scored via cosine similarity and calibrated through isotonic regression to produce interpretable stylistic similarity scores in the [0,1] range.

## 1.2 Training

To build our training set, we scraped the entire English-language corpus of the open-access public domain library Project Gutenberg. This total amounted to roughly 58,000 books across a wide variety of subject matters, with original publication dates largely clustered between the late eighteenth century and the early twentieth century.

After a robust preprocessing pipeline to handle the heterogeneous nature of text corpora and the presence of Gutenberg-related material – a pipeline that included unicode normalization to ensure consistent character representations, boilerplate removal, and whitespace standardization – we used a rule-based sentencizer from the spaCy natural language processing library to deconstruct each book into chunks of 14 sentences, with an overlap of four sentences and a minimum length of 200 characters. These parameters were selected to balance context sufficiency (14 sentences, typically coming out to between 350 and 500 tokens, was judged to be sufficient to capture local stylistic patterns like syntactic structure and lexical choice) with the 512-token limitation of the RoBERTa encoder while still providing dense within-book sampling. Finally, lightweight language filters removed non-English content, as well as remaining duplicate content (such as chapter headers or other boilerplate) that appear across at least three distinct books in the Gutenberg corpus. This process yielded several million chunks of text.

Because the quality of contrastive training depends on the difficulty of negative examples, a multi-tiered hard negative mining procedure was implemented to provide a spectrum of difficulty levels. The system uses four complementary strategies to select negative pairs (text chunks from different books):

**Embedding-based hard negatives.** Up to 80 chunks per book were sampled and embedded through the use of a lightweight pretrained encoder (all-MiniLM-L6-v2), which computed book-level centroids by averaging these chunk embeddings:

$$\mathbf{c}_b = \frac{1}{N_b}\sum_{i=1}^{N_b} \mathbf{e}_i$$

where $\mathbf{c}_b$ is the centroid for book $b$, $N_b$ is the number of sampled chunks from book $b$, and $\mathbf{e}_i$ is the embedding of chunk $i$. A cross-book similarity search was then performed to identify each book's top-50 nearest neighbor books in embedding space.

**Same-author negatives.** To create particularly challenging negatives, the system identified pairs of different books written by the same author, forcing the model to distinguish between works that share authorial style. Anonymous and "unknown author" buckets were excluded to ensure consistent authorial style within each category.

**Metadata-based negatives.** Books were matched by inferred temporal period (archaic, early modern, modern, contemporary) and genre (religious, historical, adventure, romance, general) using keyword-based heuristics, providing medium-difficulty negatives with shared stylistic conventions.

**Random negatives.** A baseline set of random book pairs provided easy negative examples to anchor the difficulty spectrum.

During training, negative pairs were sampled according to the following distribution: 35% same-author pairs (when available), 25% embedding-based neighbors, 25% metadata-similar books, and 15% random pairs. These weights were normalized dynamically when certain categories were unavailable for a given book. Training pairs were constructed with 20 positive pairs (same book) and 40 negative pairs (different books) per book, yielding a dataset of approximately 4.8M training pairs across ~58,000 books.

### 1.2.1 Model Architecture Details

The core model employs a Siamese twin-encoder architecture with shared RoBERTa-large weights (355M parameters). Each input text is encoded independently through the shared encoder, producing contextualized token representations. These representations are then pooled using a lightweight attention mechanism that learns to weight different positions based on their stylistic importance:

$$\alpha_i = \frac{\exp(w^\top \tanh(W\mathbf{h}_i))}{\sum_j \exp(w^\top \tanh(W\mathbf{h}_j))}$$

$$\mathbf{z} = \sum_i \alpha_i \mathbf{h}_i$$

where $\mathbf{h}_i$ are the hidden states from the encoder, $W$ and $w$ are learnable parameters, and $\mathbf{z}$ is the resulting pooled embedding.

In addition to the learned embeddings, we extract three simple style features from the raw text that capture aspects of writing style that may be under-represented in contextualized embeddings:

1. **Type-token ratio**: The ratio of unique words to total words, normalized by text length
2. **Punctuation frequency**: The proportion of punctuation marks relative to total characters
3. **Average sentence length**: Mean number of words per sentence

These features are concatenated with the pooled embeddings. To encourage order-invariant comparison, we also compute symmetric features from the two text embeddings:

$$\mathbf{f}_{sym} = [\mathbf{z}_1; \mathbf{z}_2; |\mathbf{z}_1 - \mathbf{z}_2|; \mathbf{z}_1 \odot \mathbf{z}_2]$$

where $;$ denotes concatenation and $\odot$ denotes element-wise multiplication.

### 1.2.2 Loss Functions

Our training objective combines four complementary loss functions:

**Classification Loss.** A standard cross-entropy loss for the binary classification task of predicting whether two text chunks come from the same book:

$$\mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^N [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

**Contrastive Loss.** We employ Supervised Contrastive Learning (SupCon) to pull embeddings from the same book together while pushing embeddings from different books apart:

$$\mathcal{L}_{con} = \sum_{i=1}^{2N} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

where $P(i)$ is the set of positive pairs for anchor $i$, $A(i) = \{1, ..., 2N\} \setminus \{i\}$ is the set of all other examples, and $\tau$ is a learned temperature parameter.

**Topic Adversarial Loss.** To prevent the model from relying on topic-based shortcuts, we introduce a topic classifier head that attempts to predict coarse topic labels (religious, historical, adventure, romance, general) from the learned representations. These topic labels are generated using simple keyword-based heuristics. Crucially, we apply a Gradient Reversal Layer (GRL) before this classifier, which acts as an identity function during forward propagation but multiplies gradients by $-\lambda$ during backpropagation:

$$\mathcal{L}_{adv} = \frac{1}{2N}\sum_{i=1}^{2N} \mathcal{L}_{CE}(\text{GRL}(\mathbf{z}_i), t_i)$$

where $t_i$ is the topic label for text $i$. The effect of this adversarial training is to create representations that are invariant to topic information.

**Optional Distillation Loss.** For models trained with knowledge distillation from a cross-encoder teacher, we add a KL divergence term:

$$\mathcal{L}_{KD} = T^2 \cdot \text{KL}(\text{softmax}(l_t/T) \| \text{softmax}(l_s/T))$$

where $l_t$ and $l_s$ are the teacher and student logits respectively, and $T$ is the distillation temperature.

The total training loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{con} + \lambda \mathcal{L}_{adv} + \beta \mathcal{L}_{KD}$$

where $\alpha=0.1$, $\lambda$ is gradually ramped from 0 to 1.0 over the first 30% of training (with a warmup period for the first 10%), and $\beta=0.5$ when distillation is used.

### 1.2.3 Training Configuration

Models were trained on NVIDIA H200 GPUs using mixed-precision training (bfloat16) with the AdamW optimizer. We used a learning rate of 2e-5 with linear warmup over 10% of training steps followed by cosine decay. Batch size was set to 16 with gradient accumulation steps of 4, yielding an effective batch size of 64. Training proceeded for 3-5 epochs depending on early stopping criteria based on balanced accuracy on the validation set. Gradient checkpointing was employed to reduce memory consumption, and we enabled TF32 tensor cores for improved throughput on Ampere-generation GPUs.

The adversarial training schedule was particularly important: the GRL scale and adversarial loss weight $\lambda$ were both gradually increased from 0 to their target values (GRL scale: 1.0, $\lambda$: 1.0) over the first 30% of training. This gradual ramp-up prevents the adversarial objective from destabilizing early training when the encoder representations are still developing.

## 1.3 Calibration

Raw cosine similarity scores between embeddings, while informative, are not well-calibrated probability estimates. To produce interpretable [0,1] scores suitable for benchmarking, we apply a two-stage calibration procedure.

**Classifier Calibration.** For the binary classification task (same book vs. different book), we apply temperature scaling on the validation set. A single scalar parameter $T$ is optimized via L-BFGS to minimize negative log-likelihood:

$$p_{cal}(y=1) = \frac{1}{1 + \exp(-(l/T))}$$

where $l$ is the raw logit output. We additionally perform threshold optimization to maximize target metrics (F1, balanced accuracy, etc.) under optional constraints on minimum accuracy or recall.

**Style Similarity Calibration.** For the style similarity scores used in benchmarking, we map raw cosine similarity values to calibrated [0,1] probability-like scores using isotonic regression or logistic regression on a labeled development set. The development set consists of text pairs labeled as having the same style (1) or different styles (0), auto-generated from the validation split and optionally augmented with LLM-generated examples to better match the benchmark distribution.

We compare logistic (parametric sigmoid) and isotonic (non-parametric monotonic) calibration using 5-fold cross-validation, selecting the method with lower Brier score:

$$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (y_i - \hat{p}_i)^2$$

For isotonic regression, we require a minimum of 1,000 labeled pairs to avoid overfitting. The calibration function is saved alongside the model and automatically applied during inference.

## 1.4 Benchmark Design

The DickensBench evaluation protocol proceeds as follows:

1. **Excerpt Selection**: For a given book from the test set, we randomly select $k$ excerpts of 15 consecutive sentences each (default $k=5$). The 15-sentence window was chosen to provide sufficient context for style identification while remaining manageable for LLM context windows.

2. **Prompt Construction**: For each excerpt, we construct a prompt that instructs the LLM to write an original short story in the style of the excerpt, but on a different topic. The prompt explicitly prohibits copying phrases, named entities, or plot points from the reference text. Target length is specified as 600-900 words. A representative prompt template is:

```
STYLE REFERENCE
---
[excerpt text]
---

Write an original short story about: [topic].
Match the reference's voice, rhythm, and paragraph structure.
Avoid copying plot points, named entities, or distinctive phrases.
Target length: 600-900 words.
```

3. **Topic Selection**: Topics are drawn from a diverse pool of 50+ options spanning contemporary and timeless themes (e.g., "artificial intelligence," "lost at sea," "a secret revealed"). For each excerpt, we sample $n$ different topics (default $n=1$), ensuring topic diversity across the benchmark to stress-test style-topic disentanglement.

4. **Generation**: The prompt is sent to the target LLM with temperature 0.9 and top-p 0.95 to encourage stylistic flexibility. Multiple samples can be drawn per excerpt-topic pair for variance estimation.

5. **Scoring**: Each generated text is scored against its reference excerpt using our calibrated style similarity metric. The scoring process automatically chunks longer texts into 14-sentence windows with 4-sentence overlap and computes cosine similarities between all chunk pairs, aggregating via mean pooling. The raw cosine is transformed through the calibration function to yield a [0,1] probability-like score.

6. **Aggregation**: For each book, we compute the mean and median calibrated similarity scores across all excerpt-generation pairs. The final DickensBench score for an LLM is the mean of these book-level scores across the entire test set.

## 1.5 Validation and Analysis

To validate that our style similarity metric captures style rather than topic, we conducted several analyses:

**Topic-Stratified Negative Pairs**: We evaluate the model on negative pairs (different books) stratified by whether they share the same coarse topic category. A style-focused model should achieve similar performance on same-topic and different-topic negatives, while a topic-shortcut model would show degraded performance on same-topic negatives. Our final model shows balanced accuracy of 89.3% on different-topic negatives and 87.8% on same-topic negatives, indicating effective topic invariance.

**Adversarial Ablation**: Training without the adversarial topic loss (GRL) results in a 4.2 percentage point drop in same-topic negative performance (83.6% vs 87.8%), confirming the importance of explicit topic disentanglement.

**Style Feature Analysis**: We computed average type-token ratio, punctuation frequency, and sentence length for both reference excerpts and LLM-generated texts. Models with higher DickensBench scores show stronger correlation between excerpt and generation statistics (mean Pearson $r=0.62$ for top-quartile models vs. $r=0.41$ for bottom-quartile).

**Human Evaluation**: A subset of 100 excerpt-generation pairs was rated by 5 human judges for style similarity on a 1-5 scale. The calibrated style similarity scores show Spearman correlation of $\rho=0.71$ with mean human ratings, compared to $\rho=0.58$ for raw cosine similarity and $\rho=0.43$ for GPT-4-based judgments.

## 1.6 Computational Requirements

**Training**: The full training pipeline (data preparation, hard negative mining, contrastive model training, and calibration) requires approximately 6-8 hours on a single NVIDIA H200 GPU for the ~58,000 book corpus. Data preparation with hard negative mining takes 40-85 minutes, training takes 3-6 hours, and calibration takes 10-20 minutes. Peak GPU memory usage is approximately 72GB during training with gradient checkpointing enabled.

**Inference**: Scoring a single excerpt-generation pair (typical length: 300 words for excerpt, 750 words for generation) takes approximately 150ms on GPU or 2.1 seconds on CPU with the RoBERTa-large encoder. Benchmarking a full book with 5 excerpts and 5 generations thus requires less than 1 second on GPU.

**Storage**: The trained model checkpoint is 1.4GB. The calibration metadata adds negligible overhead (~10KB). The full processed training dataset occupies approximately 8GB on disk.

## 1.7 Limitations and Future Work

**Coverage of Literary Periods**: The Project Gutenberg corpus skews toward works published between 1850-1920, with limited representation of contemporary styles or non-Western literary traditions. Future versions of DickensBench should incorporate more diverse corpora.

**Genre Representation**: While we implement coarse topic categories to prevent topic shortcuts, fine-grained genre conventions (noir, epistolary, stream-of-consciousness) are not explicitly modeled. These may confound style similarity in some cases.

**Long-Form Coherence**: Our chunking approach (14-sentence windows) captures local stylistic patterns but may miss document-level structure and coherence. Future work could explore hierarchical pooling mechanisms.

**Multilingual Extension**: The current benchmark is English-only. Extending to other languages would require language-specific preprocessing and larger multilingual encoders, but the overall methodology should generalize.

**Style Transfer vs. Writing Quality**: While we hypothesize that style transfer ability correlates with general writing ability, this relationship deserves further empirical investigation. DickensBench scores should be interpreted as measuring stylistic flexibility rather than absolute writing quality.

## References

Jin, D., et al. (2024). Deep learning for text style transfer: A survey. *Computational Linguistics*, 50(1), 1-78.

Porter, S., & Machery, E. (2024). Can humans distinguish between human- and AI-generated poetry? *Scientific Reports*, 14(1), 8330.

Wu, Z., et al. (2025). WritingBench: A benchmark for evaluating large language models in writing assistance. *Proceedings of ACL 2025*.

---

## Appendix A: Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Encoder | RoBERTa-large |
| Encoder Hidden Size | 1024 |
| Pooling Method | Attention-weighted |
| Attention Hidden Size | 512 |
| Classification Head | ArcFace (s=30, m=0.2) |
| Style Features | Type-token ratio, punctuation freq, avg sentence length |
| Chunk Size | 14 sentences |
| Chunk Overlap | 4 sentences |
| Max Sequence Length | 512 tokens |
| Learning Rate | 2e-5 |
| Batch Size (effective) | 64 |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 |
| Contrastive Weight (α) | 0.1 |
| Adversarial Weight (λ) | 1.0 (ramped) |
| GRL Max Scale | 1.0 (ramped) |
| Temperature (τ) | Learned |
| Training Epochs | 3-5 |

## Appendix B: Benchmark Statistics

| Statistic | Value |
|-----------|-------|
| Total Books in Corpus | ~58,000 |
| Total Text Chunks | ~8.2M |
| Training Pairs | ~4.8M |
| Validation Pairs | ~520K |
| Test Pairs | ~540K |
| Test Books for Benchmark | 100 |
| Excerpts per Book | 5 |
| Topics | 50+ |
| Avg Excerpt Length | 312 words |
| Target Generation Length | 600-900 words |
| Avg Actual Generation Length | 743 words |
