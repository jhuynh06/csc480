# NYT Connections Puzzle Solver

**CSC 480 — Artificial Intelligence, Cal Poly San Luis Obispo**
**Instructor: Rodrigo Canaan**

**Team Members:**
- Jason Huynh
- Sam Phan

---

## Overview

This project explores multiple AI approaches to solving the [New York Times Connections](https://www.nytimes.com/games/connections) puzzle, where the goal is to group 16 words into 4 categories of 4. We implement and compare three strategies:

1. **Graph-based beam search** — builds a word similarity graph from pre-trained embeddings and searches for the optimal partition
2. **Constrained k-means clustering** — greedily selects groups using k-means with a fixed cluster size of 4, simulating actual gameplay with a lives system
3. **Fine-tuned Mistral 7B** — fine-tunes a large language model on chain-of-thought and knowledge-distilled reasoning data using QLoRA

## Acknowledgments

- [NYT Connections Answers](https://github.com/Eyefyre/NYT-Connections-Answers) — puzzle dataset
- [NLPL Word Vectors Repository](https://vectors.nlpl.eu/repository/) — pre-trained Word2Vec embeddings used by k-means and beam search solvers
- [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) — base model for fine-tuning
- [Anthropic Claude API](https://docs.anthropic.com/) — used for knowledge distillation (generating teacher reasoning)
- Todd, E., Li, H., Suresh, A., & Li, B. (2024). [*Solving the NYT Connections Puzzle with an LLM*](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1254/final-reports/256847963.pdf) — Stanford CS 224N final report that informed our fine-tuning and knowledge distillation approach
- Libraries: [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), [gensim](https://radimrehurek.com/gensim/), [k-means-constrained](https://github.com/joshlk/k-means-constrained), [scikit-learn](https://scikit-learn.org/)

---

## Setup

### Python dependencies

```bash
pip install numpy gensim scikit-learn k-means-constrained tqdm requests anthropic
```

For the fine-tuning notebook (run on Google Colab with an H100 GPU), dependencies are installed in the first cell.

### Word embeddings

The k-means and beam search solvers require pre-trained Word2Vec embeddings. Download a model from the [NLPL Word Vectors Repository](https://vectors.nlpl.eu/repository/) and note the path to the `.bin` file; both scripts will prompt you for it at runtime.

### Puzzle data

`connections.json` is included in the repo. To regenerate or update it:

```bash
python build_data.py
```

This also builds the fine-tuning datasets (`cot_data.json`, `distill_data.json`). Distillation requires an `ANTHROPIC_API_KEY` environment variable.

---

## Usage

### Beam search solver (`graph.py`)

Builds a similarity graph (90% embedding cosine similarity + 10% lexical features) and runs beam search to find the best partition of 16 words into 4 groups.

```bash
python graph.py
```

You will be prompted for the path to the Word2Vec binary model. The script benchmarks against all puzzles in `connections.json` using 8 parallel workers and writes perfectly-solved puzzles to `perfect_puzzles.json`.

Key parameters (edit in source):
- `beam_size=250` — candidates kept per search step
- `outside_lambda=0.35` — penalty weight for cross-group similarity
- `top_groups_limit=2500` — candidate groups to pre-score

### K-means solver (`kmeans.py`)

Greedy solver that finds the tightest 4-word cluster via constrained k-means, removes it, and repeats. Simulates real gameplay with 3 lives.

```bash
python kmeans.py
```

You will be prompted for the path to the Word2Vec binary model. Benchmarks all puzzles in parallel and writes results to `perfect_puzzles.json`.

### Fine-tuning notebook (`mistral_finetuning.ipynb`)

Runs on **Google Colab with an H100 GPU**. Open the notebook in Colab and run cells sequentially. The notebook:

1. Installs dependencies (flash-attention, transformers, peft, trl, bitsandbytes)
2. Loads pre-built training data (`cot_data.json`, `combined_data.json`) and creates 80/10/10 splits
3. Evaluates the non-fine-tuned Mistral 7B baseline
4. Fine-tunes on chain-of-thought data and evaluates
5. Fine-tunes on distilled + CoT data and evaluates with a gamified simulation (3 lives)

Trained LoRA adapters are saved to `./mistral-cot/` and `./mistral-distill/`.

### Data generation (`build_data.py`)

Generates fine-tuning datasets from the puzzle data:

```bash
# template-based CoT only 
python build_data.py

# Claude-distilled reasoning 
export ANTHROPIC_API_KEY="your-key-here"
python build_data.py
```

Progress is cached to JSON files so the script can resume if interrupted.

---

## Results

**Table 1: Comparing Mistral Models, Accuracies Measured over 101 Puzzles**

| Mistral Model | % of Correct Categories | % of Perfect Games |
|---|---|---|
| Fine-tuned | 0 | 15.56 |
| Chain of Thought | 3.03 | 18.43 |
| Knowledge Distil. | 36.4 | 53.8 |

**Table 2: Examining Claude Sonnet, Accuracies Measured over 101 Puzzles**

| Model | % of Correct Categories | % of Perfect Games |
|---|---|---|
| Gam. KD Mistral | 78 | 68.8 |
| Claude Sonnet | 73.75 | 64 |
| Claude w/ CoT | 67.5 | 61 |

**Table 3: LLMs vs. Humans, Accuracies Measured over 950 Puzzles**

| Agent | % of Correct Categories | % of Perfect Games |
|---|---|---|
| Human | | 71 |
| Gam. KD Mistral | 78 | 68.8 |
| Gam. Claude | 96.8 | 92.1 |

**Table 4: Comparison of All Project Models, Accuracies Measured over 950 Puzzles**

| Agent | % of Correct Categories | % of Perfect Games |
|---|---|---|
| K-Means (G-News) | 4.21 | 0.31 |
| K-Means (Wikipedia) | 15.21 | 3.31 |
| Similarity Graph | 16.66 | 3.93 |
| Fine-tuned Mistral | 15.56 | 0 |
| Mistral w/ CoT | 18.43 | 3.03 |
| Mistral w/ KD | 53.8 | 36.4 |
| Gam. KD Mistral | 78 | 68.8 |
| Claude Sonnet | 73.75 | 64 |
| Claude w/ CoT | 67.5 | 61 |
| Gam. Claude | 96.8 | 92.1 |

---


