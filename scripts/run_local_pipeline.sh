#!/usr/bin/env bash
# Run the full retriever pipeline locally (Apple M4 MPS / CUDA / CPU auto-detect).
# Mirrors Group_073_COMP90042_Project_2026.ipynb step-by-step but skips Colab,
# Drive, and GitHub setup.
#
# Usage:
#   bash scripts/run_local_pipeline.sh                    # full pipeline, EPOCHS=4
#   EPOCHS=1 bash scripts/run_local_pipeline.sh           # quick smoke test
#   RESUME_FROM=checkpoints/cross_encoder_epoch2.pt \
#     bash scripts/run_local_pipeline.sh                  # resume training
#   SKIP_TRAIN=1 bash scripts/run_local_pipeline.sh       # only re-run inference + eval

set -euo pipefail

# ─── CONFIGURE ────────────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-4}"           # mirrors notebook cell 2.1
RESUME_FROM="${RESUME_FROM:-}"
TOP_K="${TOP_K:-4}"
BM25_TOP_K="${BM25_TOP_K:-200}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Make `from src...` imports work when running scripts/*.py directly
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

PY="conda run -n comp90042 --no-capture-output python"

echo
echo "=== 1.3 · Build BM25 index (idempotent — skips if cache/bm25_index/ exists) ==="
$PY scripts/build_bm25.py

if [[ "$SKIP_TRAIN" != "1" ]]; then
    echo
    echo "=== 2.1 · Train cross-encoder (epochs=$EPOCHS) ==="
    TRAIN_ARGS=(--epochs "$EPOCHS")
    [[ -n "$RESUME_FROM" ]] && TRAIN_ARGS+=(--resume "$RESUME_FROM")
    $PY scripts/train_cross_encoder.py "${TRAIN_ARGS[@]}"
fi

echo
echo "=== 3.1 · Run dev inference (BM25 top-$BM25_TOP_K → CE rerank top-$TOP_K) ==="
$PY scripts/run_inference.py \
    --split dev \
    --mode retriever-only \
    --top-k "$TOP_K" \
    --bm25-top-k "$BM25_TOP_K"

echo
echo "=== 3.2 · Evaluate on dev ==="
PRED_FILE="outputs/dev-retriever-only-k${TOP_K}-bm25${BM25_TOP_K}.json"
$PY eval.py \
    --predictions "$PRED_FILE" \
    --groundtruth data/dev-claims.json \
    --verbose

echo
echo "=== 3.3 · Generate test predictions (for submission) ==="
$PY scripts/run_inference.py \
    --split test \
    --mode retriever-only \
    --top-k "$TOP_K" \
    --bm25-top-k "$BM25_TOP_K"

echo
echo "Done."
echo "  Final checkpoint:  checkpoints/cross_encoder.pt"
echo "  Test predictions:  outputs/test-retriever-only-k${TOP_K}-bm25${BM25_TOP_K}.json"
