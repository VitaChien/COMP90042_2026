# COMP90042 2026 Project — Group 073

**Automated fact-checking for climate claims.** Given a claim, the system (1) **retrieves** the
most relevant evidence passages from a 1.2M-passage corpus and (2) **classifies** the claim as
one of `{SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED}`.

### Pipeline

```
claim ──► BM25 retrieval ──► BERT cross-encoder rerank ──► CNN + BiLSTM + multi-head attention
          (1.2M → 200)        (200 → top-4 evidence)        classifier  ──► label (4 classes)
```

---

## 1. Contents of this submission

```
Group_073_COMP90042_Project_2026.ipynb   # main notebook (run top to bottom; contains run logs)
README.md                                # this file
eval.py                                  # official scorer (called by cells 3.1 / 3.2)
environment.yml                          # conda environment for local runs
src/                                     # importable library modules
  config.py  data_loader.py  preprocessing.py  retriever_bm25.py
  retriever_cross_enc.py  hard_negatives.py  evaluator.py  utils.py  __init__.py
scripts/                                 # entry points called by the notebook
  build_bm25.py  train_cross_encoder.py  run_inference.py  __init__.py
```

**Not included**:
data files, the BM25 index/caches (`cache/`), trained checkpoints (`checkpoints/`), and all
generated predictions (`outputs/`).

---

## 2. Data

The notebook expects the four provided data files in a `data/` folder **next to the notebook**:

| Path (relative to project root) | Description |
|--------------------------------|-------------|
| `data/evidence.json`             | knowledge source, ~1.2M passages (~174 MB) |
| `data/train-claims.json`         | labelled training claims (1228) |
| `data/dev-claims.json`           | labelled dev claims (154) |
| `data/test-claims-unlabelled.json` | unlabelled test claims (153) |

These are **not** in the zip (data files must not be submitted). Download them from the course
and place them in `data/`. Cell **1.2** verifies they are present and stops with a clear error if
any are missing.

---

## 3. How to run (Google Colab)

Keep **everything in one folder** — `src/`, `scripts/`, `eval.py`, the notebook, and a `data/`
subfolder holding the four data files. Pick whichever setup you use:

- **From Google Drive (how we run it):** put the unzipped project (plus `data/`) in a Drive
  folder, e.g. `MyDrive/COMP90042_2026/`, open the notebook in Colab, and in **cell 1.1** set
  `PROJECT_ROOT = "/content/drive/MyDrive/COMP90042_2026"`. Drive is mounted automatically.
- **Uploaded into the session:** unzip the project into e.g. `/content/COMP90042_073`, add the
  data files under `/content/COMP90042_073/data/`, and set `PROJECT_ROOT = ""` (auto-detect) or
  the explicit path.

Then:

1. **Runtime → Change runtime type → GPU** (a T4 is enough; CPU is far too slow for
   cross-encoder training).
2. `Runtime → Run all`. Cell 1.1 mounts Drive (if used), installs dependencies and sets paths;
   cell 1.2 verifies the data files are present.

> If cell 1.2 reports files missing, set `PROJECT_ROOT` at the top of cell 1.1 to the folder that
> contains `src/`, `scripts/` and `data/`, then re-run from cell 1.1.

### Execution order (the cells already follow this — just run top to bottom)

| Cell | Step | Builds / writes |
|------|------|-----------------|
| 1.1  | Install deps, set paths, imports | — |
| 1.2  | Verify data files exist | — |
| 1.3  | Seed + helper functions | — |
| 1.5  | Load `evidence.json` + claims into memory | — |
| 1.4  | Build BM25 index (idempotent) | `cache/bm25_index/` |
| 2.1  | Train BERT cross-encoder (4 epochs) | `checkpoints/cross_encoder.pt`, `cache/bm25_train_top200.json` |
| 2.2  | Two-stage retrieval for dev / test / train | `outputs/{dev,test,train}-retriever-only-k4-bm25200.json` |
| 2.3–2.6, 15 | Build vocab, define + train the classifier | (best model kept in memory) |
| 3.1  | Score retriever-only on dev (evidence F) | — |
| 3.2  | Score full pipeline on dev | `outputs/dev_predictions_cnn_bilstm_multihead_final.json` |
| 3.3  | Predict on the test set | `outputs/test_predictions_cnn_bilstm_multihead_final.json` |

---

## 4. Outputs

All artefacts are written under the project root and are **regenerated on every run**:

| Path | What it is |
|------|-----------|
| `cache/bm25_index/` | saved BM25 index (multiple shard files) |
| `cache/bm25_train_top200.json` | cached BM25 top-200 over train (for hard-negative mining) |
| `checkpoints/cross_encoder.pt` | final cross-encoder weights (plus `cross_encoder_epoch{1..4}.pt`) |
| `outputs/dev-retriever-only-k4-bm25200.json` | reranked top-4 evidence for dev |
| `outputs/test-retriever-only-k4-bm25200.json` | reranked top-4 evidence for test |
| `outputs/train-retriever-only-k4-bm25200.json` | reranked top-4 evidence for train (classifier input) |
| `outputs/dev_predictions_cnn_bilstm_multihead_final.json` | **dev** predictions (label + evidence) |
| `outputs/test_predictions_cnn_bilstm_multihead_final.json` | **test** predictions (leaderboard format) |

Score any prediction file with the official scorer:

```bash
python eval.py --predictions outputs/dev_predictions_cnn_bilstm_multihead_final.json \
               --groundtruth data/dev-claims.json
```

---

## 5. Reported results (dev set)

| System | Evidence F | Claim Accuracy | Harmonic Mean |
|--------|-----------:|---------------:|--------------:|
| Retriever only (BM25 → CE rerank, top-4) | 0.2017 | — (random label) | — |
| **Full pipeline** (retriever + classifier) | **0.2017** | **0.4221** | **0.2729** |

These figures are the saved cell outputs in the notebook. Re-running reproduces them up to minor nondeterminism.
