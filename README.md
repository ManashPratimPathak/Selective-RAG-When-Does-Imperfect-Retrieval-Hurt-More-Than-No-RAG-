# Selective RAG

Reproducible experiments for selective retrieval-augmented generation using PopQA,
Wikipedia evidence, BM25 retrieval, a local Qwen model, and router/policy evaluation.

## What belongs in GitHub

Keep the source and experiment contract:

- `scripts/` — data preparation, retrieval, generation, scoring, routing, and analysis code
- `tests/` — unit tests
- `config.yaml` — seeds, model revisions, retrieval, and generation settings
- `requirements.txt` — Python dependencies
- `report/` — report source and figures
- `LICENSE` and this README

The pipeline-generated contents of `data/`, `outputs/`, local caches, ZIP archives,
and `presentation_phase_1_to_5/` should not be uploaded. They can be regenerated and
are ignored by `.gitignore`. Do not commit `.venv/`, Hugging Face caches, API keys, or
model weights.

Important: ignore rules do not remove files already tracked by Git. Before the first
public push, inspect `git status --ignored` and untrack any old generated artifacts
that are already in the index. If the repository history already contains large
artifacts, a new clean repository or an approved history rewrite is needed to remove
those blobs from GitHub's storage.

## Environment

Python 3.10+ is recommended. Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The default model is `Qwen/Qwen2.5-1.5B-Instruct`; the first model run needs network
access to download it. The configured 4-bit loading is intended for a compatible GPU.
Wikipedia and PopQA preparation also require network access.

## Rebuild the data and retrieval artifacts

Run from the repository root. Each command writes only ignored/generated artifacts:

```powershell
python scripts/01_prepare_popqa.py
python scripts/02_build_wikipedia_corpus.py
python scripts/02_add_distractors.py
python scripts/freeze_cleaned_corpus.py
python scripts/chunk_corpus.py
python scripts/build_bm25.py
python scripts/freeze_retrieval.py
```

The exact generation/evaluation path depends on the frozen artifacts and model
availability. Run the relevant helper scripts in `scripts/` and then the tests:

```powershell
python -m pytest
```

The numbered files `02_fetch_wikipedia.py`, `07_generate.py`, `09_route.py`, and
`10_analyze.py` are currently placeholders; use the implemented helper scripts until
those wrappers are completed.

## Optional Groq comparison

`scripts/13_groq_llama8b_comparison.py` uses the `GROQ_API_KEY` environment variable.
Never place that key in the repository or in command history. Its results are generated
under `outputs/` and are intentionally ignored.
