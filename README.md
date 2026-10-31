# Selective RAG: When Does Imperfect Retrieval Hurt More Than No-RAG?

This project studies when retrieval-augmented generation helps factual question answering and when noisy retrieval can make a correct model answer worse. It compares a small instruction-tuned Qwen model under three matched conditions:

1. **No-RAG** — answer from the question alone.
2. **Clean RAG** — answer with verified, answer-bearing Wikipedia evidence.
3. **Imperfect RAG** — answer with the top BM25-retrieved passages, which may contain irrelevant or unsupported evidence.

The experiment uses a frozen PopQA sample of 200 test questions, balanced between popular (head) and long-tail (tail) entities. A score-based router is also evaluated to decide when retrieval should be used.

## Main results

| Condition | Exact match | Token F1 |
|---|---:|---:|
| No-RAG | 13.0% | 18.08% |
| Imperfect RAG | 61.5% | 71.77% |
| Clean RAG | 73.5% | 86.94% |

Key findings:

- Imperfect retrieval produced **103 rescue events** and **6 damage events** relative to No-RAG.
- All six observed damage events occurred in the head/popular group; tail questions received the largest retrieval benefit.
- Development Recall@3 was **72.5%** (29/40); test Recall@3 was **39.0%** for head questions and **85.0%** for tail questions.
- The threshold router reduced conditional damage from **23.1% to 7.7%**, but reached only **52.0% EM**, below always using Imperfect RAG at 61.5%.

The detailed report is in [`report/selective_rag_report.tex`](report/selective_rag_report.tex).

## Repository structure

```text
config.yaml          Frozen seeds, model revision, retrieval, and generation settings
requirements.txt     Python dependencies
scripts/              Data preparation, retrieval, generation, scoring, and analysis
tests/                Unit tests
report/               LaTeX report and figures
data/                 Generated datasets and retrieval artifacts (ignored by Git)
outputs/              Generated predictions, metrics, tables, and figures (ignored by Git)
```

Generated datasets and outputs are intentionally not stored in GitHub. They are reproducible from the scripts below. The natural distractor corpus can be large, so `data/corpus/distractor_pages.jsonl` remains ignored.

## Requirements

- Python 3.10 or newer
- A CUDA-capable GPU is recommended for the configured 4-bit model
- Network access for PopQA, Wikipedia, and the first Hugging Face model download
- PowerShell examples below assume Windows; the Python commands are platform-independent

Create and activate an environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`. Its pinned revision and the generation contract are recorded in `config.yaml`. Do not commit model weights, API keys, Hugging Face caches, or generated artifacts.

## Rebuild the data and retrieval pipeline

Run all commands from the repository root:

```powershell
# 1. Freeze the PopQA development/test sample
python scripts/01_prepare_popqa.py

# 2. Download the answer-bearing Wikipedia pages
python scripts/02_build_wikipedia_corpus.py

# 3. Build the natural distractor corpus
python scripts/02_add_distractors.py --count 500

# 4. Clean and freeze the corpus, then create overlapping chunks
python scripts/freeze_cleaned_corpus.py
python scripts/chunk_corpus.py

# 5. Build the BM25 token index
python scripts/build_bm25.py

# 6. Generate candidate evidence and a manual verification sheet
python scripts/gold_candidates.py
python scripts/verify_passages.py
```

Before freezing retrieval, inspect `data/manifests/passage_verification_sheet.jsonl` and verify that the selected clean passages support the subject–relation–answer fact and are non-conflicting. Then run:

```powershell
# 7. Freeze verified evidence and top-5 retrieval results
python scripts/freeze_retrieval.py
python scripts/compute_recall_at_3.py
```

The pipeline writes manifests under `data/manifests/`, corpus files under `data/corpus/`, BM25 files under `data/index/`, and retrieval files under `data/retrieval/`.

## Run generation and evaluation

The generation scripts use the frozen artifacts and the pinned model configuration:

```powershell
# Quick model/prompt smoke test
python scripts/step51_smoke.py

# Validate all three conditions on the 40-question development split
python scripts/step53_validate_dev.py

# Generate the 200-question test predictions
python scripts/step55_run_test.py

# Score predictions and measure latency
python scripts/08_score.py
python scripts/step56_measure_latency.py

# Evaluate the router and paired No-RAG vs Imperfect-RAG transitions
python scripts/evaluate_policies.py
python scripts/paired_statistics.py
python scripts/build_error_analysis.py

# Create the final tables and figures
python scripts/make_phase10_outputs.py
```

Important generated files include:

- `outputs/test_predictions.jsonl` — predictions for all three conditions
- `outputs/test_scoring_ledger.jsonl` — normalized answers and exact-match scores
- `outputs/policy_evaluation.json` — router and policy metrics
- `outputs/paired_statistics.json` — rescue, damage, and paired significance analysis
- `outputs/phase10/` — final tables, figures, and manifest

## Optional experiments

Build and evaluate adversarial retrieval impurities:

```powershell
python scripts/11_build_impurities.py
python scripts/12_evaluate_impurities.py
```

Run the optional Groq Llama comparison. Keep the API key outside the repository:

```powershell
$env:GROQ_API_KEY = "your-key"
python scripts/13_groq_llama8b_comparison.py
```

## Tests

Run the unit-test suite after installing the dependencies:

```powershell
python -m pytest
```

## Report

The report source and figures are in `report/`. With a LaTeX installation that includes `pdflatex`:

```powershell
cd report
pdflatex -interaction=nonstopmode selective_rag_report.tex
pdflatex -interaction=nonstopmode selective_rag_report.tex
cd ..
```

## Reproducibility and limitations

The experiment freezes the dataset split, corpus, BM25 index, evidence mapping, prompt format, decoding settings, and model revision. Results are specific to one small model, one English factual dataset, one Wikipedia snapshot, and a controlled single-hop task. Retrieval latency in the report measures lookup over frozen retrieval results, not a full online BM25 rebuild and search.
