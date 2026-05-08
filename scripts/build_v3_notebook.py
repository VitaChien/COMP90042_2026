"""Generate Group_073_CNN_BiLSTM_Multihead_Balanced.ipynb from inline source.

Run from repo root:  python3 scripts/build_v3_notebook.py
"""

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "Group_073_CNN_BiLSTM_Multihead_Balanced.ipynb"


def md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


def code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


CELLS = []

# ---------- title + readme ----------
CELLS.append(
    md(
        "# 2026 COMP90042 Project — Group 073 (CNN+BiLSTM+Multihead, balanced)\n"
        "*Fact-checking: BM25+cross-encoder retrieval + multi-kernel CNN + BiLSTM + multi-head attention classifier with class-balanced loss.*"
    )
)

CELLS.append(
    md(
        "# Readme\n"
        "\n"
        "**Before running:**\n"
        "\n"
        "1. From your local machine, push the working branch to GitHub:\n"
        "   `git push -u origin <branch>`\n"
        "2. Add a GitHub Personal Access Token to Colab Secrets (only needed for private repos).\n"
        "3. On Google Drive at `/content/drive/MyDrive/COMP90042_2026/`, place the data:\n"
        "   - Required: `data/evidence.json` (~1 GB), `data/train-claims.json`, `data/dev-claims.json`, `data/test-claims-unlabelled.json`\n"
        "   - Optional GPU path (saves ~2 min): `cache/bm25_index/` (pre-built BM25 index)\n"
        "   - Required GPU path: `checkpoints/cross_encoder.pt` (trained cross-encoder; see cell 1.3 note)\n"
        "   - Optional CPU path (saves ~1 hr): `cache/faiss/sentence-transformers_all-MiniLM-L6-v2.faiss` + matching `_evidence_ids.json`\n"
        "\n"
        "Cell 1.1 clones the code from GitHub to `/content/COMP90042_2026` (Colab's fast local SSD) and symlinks `data/`, `cache/`, `checkpoints/`, `outputs/` from Drive — so code is git-managed and data persists across sessions.\n"
        "\n"
        "**Pipeline (auto-selects on device):**\n"
        "- GPU: BM25 top-200 → cross-encoder rerank → CNN+BiLSTM+MHA classifier\n"
        "- CPU: FAISS dense (sentence-transformers/all-MiniLM-L6-v2) → CNN+BiLSTM+MHA classifier\n"
        "\n"
        "Trained with class-balanced cross-entropy."
    )
)

# ---------- Section 1 ----------
CELLS.append(
    md(
        "# 1.DataSet Processing\n"
        "(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)"
    )
)

CELLS.append(
    code("""# @title 1.1 · Setup — Sync code from GitHub, mount Drive, install packages

import os

# Stop JAX/TF from preallocating GPU memory.
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import shutil
import subprocess
import sys

from google.colab import drive

drive.mount("/content/drive")

# -- EDIT IF NEEDED ----------------------------------------------------------
GITHUB_USER = "VitaChien"
REPO_NAME = "COMP90042_2026"
BRANCH = "vita/cnn_bilstm_multihead_balance_refined"
DRIVE_DATA = "/content/drive/MyDrive/COMP90042_2026"
PROJECT_ROOT = "/content/COMP90042_2026"
# ----------------------------------------------------------------------------

repo_url = f"https://@github.com/{GITHUB_USER}/{REPO_NAME}.git"
if not os.path.exists(PROJECT_ROOT):
    subprocess.check_call(["git", "clone", "-b", BRANCH, repo_url, PROJECT_ROOT])
else:
    subprocess.check_call(["git", "-C", PROJECT_ROOT, "pull"])

for sub in ("data", "cache", "checkpoints", "outputs"):
    src = f"{DRIVE_DATA}/{sub}"
    dst = f"{PROJECT_ROOT}/{sub}"
    os.makedirs(src, exist_ok=True)
    if os.path.exists(dst) and not os.path.islink(dst):
        shutil.rmtree(dst)
    if not os.path.islink(dst):
        os.symlink(src, dst)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch",
        "transformers>=4.40",
        "bm25s[full]>=0.3",
        "faiss-cpu",
        "sentence-transformers>=2.7",
        "scikit-learn>=1.3",
        "tqdm>=4.65",
    ]
)

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
print(f"Working directory: {os.getcwd()}")""")
)

CELLS.append(
    code("""# @title 1.2 · Verify data files exist
from pathlib import Path

root = Path(PROJECT_ROOT)
checks = [
    ("data/evidence.json", "required (~1 GB)"),
    ("data/train-claims.json", "required"),
    ("data/dev-claims.json", "required"),
    ("data/test-claims-unlabelled.json", "required"),
    ("cache/bm25_index", "optional (GPU path) — saves ~2 min"),
    ("checkpoints/cross_encoder.pt", "optional (GPU path only) — see cell 1.3 note"),
    ("cache/faiss", "optional (CPU path) — saves ~1 hr first-run encoding"),
]
missing_required = False
for rel, note in checks:
    p = root / rel
    if p.exists():
        size = f"{p.stat().st_size/1e6:.0f} MB" if p.is_file() else "dir"
        print(f"  OK  {rel:45s} {size:10s}  ({note})")
    else:
        icon = "XX" if "required" in note else "--"
        print(f'  {icon}  {rel:45s} {"MISSING":10s}  ({note})')
        if "required" in note:
            missing_required = True

if missing_required:
    raise FileNotFoundError("Required data files are missing. Upload them to Drive first.")""")
)

CELLS.append(
    code("""# @title 1.3 · Imports, utilities, load data, build vocab, build/load BM25+CE retriever

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.config import Config
from src.retriever_bm25 import BM25Retriever, build_bm25_index
from src.retriever_cross_enc import load_cross_encoder, rerank
from src.v3_helpers import (
    BM25CERetriever,
    build_vocab_full_corpus,
    pick_evidence_ids,
    set_seed,
    simple_tokenise,
)

LABEL2ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2,
    "DISPUTED": 3,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


# ---------------- reproducibility ----------------
set_seed(42)


# ---------------- paths ----------------
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
CACHE_DIR = Path("cache")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EVIDENCE_PATH = DATA_DIR / "evidence.json"
TRAIN_PATH = DATA_DIR / "train-claims.json"
DEV_PATH = DATA_DIR / "dev-claims.json"
TEST_PATH = DATA_DIR / "test-claims-unlabelled.json"


# ---------------- utilities ----------------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def get_claim_items(claims_json: Dict) -> List[Tuple[str, Dict]]:
    return list(claims_json.items())


def concatenate_evidence(
    evidence_ids: List[str],
    evidence_corpus: Dict[str, str],
    max_evidence: int = 5,
) -> str:
    selected_ids = evidence_ids[:max_evidence]
    evidence_texts = []
    for eid in selected_ids:
        if eid in evidence_corpus:
            evidence_texts.append(evidence_corpus[eid])
    if len(evidence_texts) == 0:
        return "No relevant evidence found."
    return " ".join(evidence_texts)


# ---------------- load data ----------------
evidence_corpus = load_json(EVIDENCE_PATH)
train_claims = load_json(TRAIN_PATH)
dev_claims = load_json(DEV_PATH)

print("Number of evidence passages:", len(evidence_corpus))
print("Number of train claims:", len(train_claims))
print("Number of dev claims:", len(dev_claims))


# ---------------- vocab ----------------
vocab = build_vocab_full_corpus(train_claims, evidence_corpus, min_freq=2, max_vocab_size=50_000)
print("Vocab size:", len(vocab))


# ---------------- Retriever: GPU = BM25 + Cross-encoder, CPU = FAISS dense ----------------
# CE rerank is BERT forward over BM25 top-200 per claim — prohibitively slow
# without GPU. FAISS dense (encode-once-cache) is hundreds of times faster on CPU.
if device == "cuda":
    cfg = Config()
    bm25_cache = cfg.cache_dir / "bm25_index"
    if not bm25_cache.exists():
        print("Building BM25 index (one-time, ~2 min)...")
        build_bm25_index(evidence_corpus, bm25_cache)

    bm25 = BM25Retriever.from_cache(bm25_cache)

    ce_ckpt = cfg.ckpt_dir / "cross_encoder.pt"
    if not ce_ckpt.exists():
        raise FileNotFoundError(
            f"Cross-encoder checkpoint missing: {ce_ckpt}\\n"
            "Run scripts/train_cross_encoder.py first or copy a pretrained one."
        )
    ce_tok, ce_model = load_cross_encoder(cfg.cross_encoder_model, ce_ckpt, device=device)

    def _rerank_fn(claim_text, candidates, top_k):
        return rerank(
            ce_model, ce_tok, claim_text, candidates, evidence_corpus,
            top_k=top_k, batch_size=64, device=device, max_len=cfg.ce_max_len,
        )

    retriever = BM25CERetriever(bm25=bm25, rerank_fn=_rerank_fn, bm25_top_k=200)
    print("Retriever ready: BM25 top-200 -> CE rerank (GPU)")
else:
    print("No GPU detected — falling back to FAISS dense retriever (CPU-friendly).")
    from src.v3_helpers import FAISSDenseRetriever
    retriever = FAISSDenseRetriever(
        evidence_corpus=evidence_corpus,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_dir="cache/faiss",
        batch_size=64,
        device=device,
    )
    print("Retriever ready: FAISS dense (CPU)")""")
)

CELLS.append(
    md(
        "### Cross-encoder checkpoint required\n"
        "\n"
        "This notebook expects a trained cross-encoder at `checkpoints/cross_encoder.pt`. "
        "If absent, train one first on Colab (~30 min on T4):\n"
        "\n"
        "```\n"
        "!python scripts/train_cross_encoder.py --epochs 4\n"
        "```\n"
        "\n"
        "The script and modules came from the `vita/retriever` branch."
    )
)

# ---------- Section 2 ----------
CELLS.append(
    md(
        "# 2.Model Implementation\n"
        "(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)"
    )
)

CELLS.append(
    code("""# @title 2.1 · Encode text + CNNBiLSTMDataset + collate fn

def encode_tokens(tokens, vocab, max_len=512):
    ids = [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]
    return ids[:max_len]


def encode_claim_evidence(claim_text, evidence_text, vocab, max_len=512):
    tokens = [
        "<CLAIM>",
        *simple_tokenise(claim_text),
        "<EVIDENCE>",
        *simple_tokenise(evidence_text),
    ]
    return encode_tokens(tokens, vocab, max_len=max_len)


class CNNBiLSTMDataset(Dataset):
    def __init__(
        self,
        claims_json,
        evidence_corpus,
        vocab,
        max_len=512,
        max_evidence=5,
        use_gold_evidence=True,
        retriever=None,
        retrieval_top_k=5,
        is_test=False,
        p_retrieved_for_training: float = 0.0,
        seed: int = 42,
    ):
        self.items = list(claims_json.items())
        self.evidence_corpus = evidence_corpus
        self.vocab = vocab
        self.max_len = max_len
        self.max_evidence = max_evidence
        self.use_gold_evidence = use_gold_evidence
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.is_test = is_test
        self.p_retrieved_for_training = p_retrieved_for_training
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        claim_id, instance = self.items[idx]
        claim_text = instance["claim_text"]

        if self.is_test:
            evidence_ids = self.retriever.retrieve(claim_text, top_k=self.retrieval_top_k)
        elif self.use_gold_evidence:
            gold = instance.get("evidences", [])
            if self.p_retrieved_for_training > 0.0 and self.retriever is not None:
                retrieved = self.retriever.retrieve(claim_text, top_k=self.retrieval_top_k)
                evidence_ids = pick_evidence_ids(
                    gold=gold, retrieved=retrieved,
                    p_retrieved=self.p_retrieved_for_training, rng=self._rng,
                )
            else:
                evidence_ids = gold
        else:
            evidence_ids = self.retriever.retrieve(claim_text, top_k=self.retrieval_top_k)

        evidence_text = concatenate_evidence(
            evidence_ids=evidence_ids,
            evidence_corpus=self.evidence_corpus,
            max_evidence=self.max_evidence,
        )

        input_ids = encode_claim_evidence(
            claim_text=claim_text,
            evidence_text=evidence_text,
            vocab=self.vocab,
            max_len=self.max_len,
        )

        item = {
            "claim_id": claim_id,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "evidence_ids": evidence_ids,
        }

        if not self.is_test:
            item["label"] = torch.tensor(
                LABEL2ID[instance["claim_label"]], dtype=torch.long
            )

        return item


def cnn_bilstm_collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    input_ids = pad_sequence(
        input_ids, batch_first=True, padding_value=vocab["<PAD>"]
    )
    attention_mask = (input_ids != vocab["<PAD>"]).long()

    output = {
        "claim_ids": [item["claim_id"] for item in batch],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "evidence_ids": [item["evidence_ids"] for item in batch],
    }

    if "label" in batch[0]:
        output["labels"] = torch.stack([item["label"] for item in batch])

    return output""")
)

CELLS.append(
    code("""# @title 2.2 · Model — multi-kernel CNN + BiLSTM + Multi-head Attention + Attention Pooling

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states, attention_mask=None):
        scores = self.attention(hidden_states).squeeze(-1)
        if attention_mask is not None:
            attention_mask = attention_mask.bool()
            scores = scores.masked_fill(~attention_mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled


class CNNBiLSTMMultiheadClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        cnn_channels=64,
        kernel_sizes=(3, 5, 7),
        lstm_hidden_dim=128,
        lstm_layers=1,
        num_labels=4,
        num_heads=4,
        dropout=0.3,
        pad_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=cnn_channels,
                    kernel_size=k,
                    padding=k // 2,
                )
                for k in kernel_sizes
            ]
        )

        cnn_output_dim = cnn_channels * len(kernel_sizes)

        self.bilstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_output_dim = lstm_hidden_dim * 2

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

        self.attention_pooling = AttentionPooling(lstm_output_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, num_labels),
        )

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = self.embedding_dropout(embedded)

        conv_input = embedded.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            x = F.relu(conv(conv_input))
            conv_outputs.append(x)
        conv_output = torch.cat(conv_outputs, dim=1)

        lstm_input = conv_output.transpose(1, 2)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(
                lstm_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_output, _ = self.bilstm(packed)
            lstm_output, _ = pad_packed_sequence(
                packed_output, batch_first=True, total_length=lstm_input.size(1)
            )
        else:
            lstm_output, _ = self.bilstm(lstm_input)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        attn_output, _ = self.multihead_attn(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=key_padding_mask,
        )
        attn_output = self.layer_norm(attn_output + lstm_output)

        pooled_output = self.attention_pooling(
            attn_output, attention_mask=attention_mask
        )
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits""")
)

CELLS.append(
    code("""# @title 2.3 · Evaluate fn + class weights + train fn (class-balanced cross-entropy)

def evaluate_cnn_bilstm(model, dataloader, device="cpu", desc="eval"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).long()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    per_class_f1 = f1_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(4)),
        zero_division=0,
    )

    print("Dev classification accuracy:", round(acc, 4))
    print("Dev macro F1:", round(macro_f1, 4))
    print("Dev weighted F1:", round(weighted_f1, 4))

    print("\\nPer-class F1:")
    for i, score in enumerate(per_class_f1):
        print(f"{ID2LABEL[i]}: {score:.4f}")

    print("\\nClassification report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            labels=list(range(4)),
            target_names=[ID2LABEL[i] for i in range(4)],
            zero_division=0,
        )
    )

    print("\\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_preds, labels=list(range(4))))

    return acc, macro_f1, weighted_f1


def get_class_weights_from_dataset(dataset, num_labels, device):
    labels = []
    for item in dataset:
        label = item["label"] if "label" in item else item["labels"]
        labels.append(int(label))
    labels = np.array(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_labels),
        y=labels,
    )
    return torch.tensor(class_weights, dtype=torch.float).to(device)


def train_cnn_bilstm_multikernel_multihead_balanced(
    train_claims,
    dev_claims,
    evidence_corpus,
    vocab,
    retriever,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    max_len=256,
    max_evidence=4,
    device="cpu",
    seed: int = 42,
):
    set_seed(seed)

    train_dataset = CNNBiLSTMDataset(
        claims_json=train_claims,
        evidence_corpus=evidence_corpus,
        vocab=vocab,
        max_len=max_len,
        max_evidence=max_evidence,
        use_gold_evidence=True,
        retriever=retriever,
        retrieval_top_k=max_evidence + 4,
        p_retrieved_for_training=0.5,
        is_test=False,
        seed=seed,
    )

    dev_dataset = CNNBiLSTMDataset(
        claims_json=dev_claims,
        evidence_corpus=evidence_corpus,
        vocab=vocab,
        max_len=max_len,
        max_evidence=max_evidence,
        use_gold_evidence=True,
        is_test=False,
    )

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=cnn_bilstm_collate_fn,
        generator=g,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cnn_bilstm_collate_fn,
    )

    dev_dataset_retrieved = CNNBiLSTMDataset(
        claims_json=dev_claims,
        evidence_corpus=evidence_corpus,
        vocab=vocab,
        max_len=max_len,
        max_evidence=max_evidence,
        use_gold_evidence=False,
        retriever=retriever,
        retrieval_top_k=10,
        is_test=False,
    )
    dev_loader_retrieved = DataLoader(
        dev_dataset_retrieved,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cnn_bilstm_collate_fn,
    )

    class_weights = get_class_weights_from_dataset(
        dataset=train_dataset, num_labels=4, device=device
    )

    model = CNNBiLSTMMultiheadClassifier(
        vocab_size=len(vocab),
        embedding_dim=128,
        cnn_channels=64,
        kernel_sizes=(3, 5, 7),
        lstm_hidden_dim=128,
        lstm_layers=1,
        num_labels=4,
        num_heads=4,
        dropout=0.3,
        pad_idx=vocab["<PAD>"],
    ).to(device)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    print("Class weights:", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_macro_f1 = 0.0
    best_state_dict = None

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs} train",
            leave=False,
            position=1,
        )
        for batch in batch_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimiser.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            total_loss += loss.item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)

        dev_acc_gold, dev_macro_f1_gold, _ = evaluate_cnn_bilstm(
            model, dev_loader, device, desc="dev (gold)"
        )
        dev_acc_ret, dev_macro_f1_ret, _ = evaluate_cnn_bilstm(
            model, dev_loader_retrieved, device, desc="dev (retrieved)"
        )

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            gold_f1=f"{dev_macro_f1_gold:.4f}",
            ret_f1=f"{dev_macro_f1_ret:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}  "
            f"gold-dev acc={dev_acc_gold:.4f} F1={dev_macro_f1_gold:.4f}  "
            f"retrieved-dev acc={dev_acc_ret:.4f} F1={dev_macro_f1_ret:.4f}"
        )

        if dev_macro_f1_ret > best_macro_f1:
            best_macro_f1 = dev_macro_f1_ret
            best_state_dict = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            tqdm.write(f"  -> new best (retrieved-dev F1={best_macro_f1:.4f}) saved")

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model""")
)

CELLS.append(
    code("""# @title 2.4 · Run training (10 epochs, class-balanced loss)

cnn_bilstm_multihead_model = train_cnn_bilstm_multikernel_multihead_balanced(
    train_claims=train_claims,
    dev_claims=dev_claims,
    evidence_corpus=evidence_corpus,
    vocab=vocab,
    retriever=retriever,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    max_len=256,
    max_evidence=4,
    device=device,
)

# Save best weights
ckpt_path = Path("checkpoints") / "cnn_bilstm_multihead_balanced.pt"
ckpt_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(cnn_bilstm_multihead_model.state_dict(), ckpt_path)
print("Saved checkpoint:", ckpt_path)""")
)

CELLS.append(
    code("""# @title 2.5 · Multi-seed run (3 seeds, ~30 min x 3) — for reporting mean +/- std

from src.v3_helpers import multi_seed_run, set_seed

def _runner(seed: int):
    set_seed(seed)
    model = train_cnn_bilstm_multikernel_multihead_balanced(
        train_claims=train_claims,
        dev_claims=dev_claims,
        evidence_corpus=evidence_corpus,
        vocab=vocab,
        retriever=retriever,
        epochs=10, batch_size=32, lr=1e-3,
        max_len=256, max_evidence=4,
        device=device,
        seed=seed,
    )
    # Evaluate on retrieved-dev (the metric we ship)
    dev_dataset_ret = CNNBiLSTMDataset(
        claims_json=dev_claims, evidence_corpus=evidence_corpus, vocab=vocab,
        max_len=256, max_evidence=4, use_gold_evidence=False,
        retriever=retriever, retrieval_top_k=10, is_test=False,
    )
    dev_loader_ret = DataLoader(dev_dataset_ret, batch_size=32, shuffle=False,
                                  collate_fn=cnn_bilstm_collate_fn)
    acc, macro_f1, weighted_f1 = evaluate_cnn_bilstm(model, dev_loader_ret, device)
    return {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}

summary = multi_seed_run(_runner, seeds=[42, 43, 44])
for m, stats in summary.items():
    print(f"{m}: {stats['mean']:.4f} ± {stats['std']:.4f}  (values: {stats['values']})")""")
)

# ---------- Section 3 ----------
CELLS.append(
    md(
        "# 3.Testing and Evaluation\n"
        "(You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)"
    )
)

CELLS.append(
    code("""# @title 3.1 · Predict on dev set (BM25+CE top-10 -> use top-4)

def predict_claims_cnn_bilstm(
    claims_json,
    evidence_corpus,
    retriever,
    model,
    vocab,
    output_path,
    retrieval_top_k=10,
    max_evidence=4,
    batch_size=32,
    max_len=256,
    is_test=False,
    device="cpu",
):
    dataset = CNNBiLSTMDataset(
        claims_json=claims_json,
        evidence_corpus=evidence_corpus,
        vocab=vocab,
        max_len=max_len,
        max_evidence=max_evidence,
        use_gold_evidence=False,
        retriever=retriever,
        retrieval_top_k=retrieval_top_k,
        is_test=is_test,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cnn_bilstm_collate_fn,
    )

    model.eval()
    predictions = {}

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            for claim_id, pred_id, evidence_ids in zip(
                batch["claim_ids"], pred_ids, batch["evidence_ids"]
            ):
                predictions[claim_id] = {
                    "claim_text": claims_json[claim_id]["claim_text"],
                    "claim_label": ID2LABEL[pred_id],
                    "evidences": evidence_ids[:max_evidence],
                }

    save_json(predictions, output_path)
    print("Saved predictions to:", output_path)
    return predictions


DEV_PRED_PATH = OUTPUT_DIR / "dev_predictions_cnn_bilstm_multihead_bal_4evi.json"

dev_predictions_cnn_bilstm = predict_claims_cnn_bilstm(
    claims_json=dev_claims,
    evidence_corpus=evidence_corpus,
    retriever=retriever,
    model=cnn_bilstm_multihead_model,
    vocab=vocab,
    output_path=DEV_PRED_PATH,
    retrieval_top_k=10,
    max_evidence=4,
    batch_size=32,
    max_len=256,
    is_test=False,
    device=device,
)""")
)

CELLS.append(
    code("""# @title 3.2 · Evaluate dev predictions — classification F1 + eval.py

def evaluate_dev_predictions_classification_f1(dev_claims, predictions):
    gold_labels = []
    pred_labels = []

    for claim_id, gold_instance in dev_claims.items():
        if claim_id not in predictions:
            continue
        gold_labels.append(LABEL2ID[gold_instance["claim_label"]])
        pred_labels.append(LABEL2ID[predictions[claim_id]["claim_label"]])

    acc = accuracy_score(gold_labels, pred_labels)
    macro_f1 = f1_score(
        gold_labels, pred_labels, average="macro", zero_division=0
    )
    weighted_f1 = f1_score(
        gold_labels, pred_labels, average="weighted", zero_division=0
    )
    per_class_f1 = f1_score(
        gold_labels,
        pred_labels,
        average=None,
        labels=list(range(4)),
        zero_division=0,
    )

    print("Dev classification accuracy:", round(acc, 4))
    print("Dev classification macro F1:", round(macro_f1, 4))
    print("Dev classification weighted F1:", round(weighted_f1, 4))

    print("\\nPer-class F1:")
    for i, score in enumerate(per_class_f1):
        print(f"{ID2LABEL[i]}: {score:.4f}")

    print("\\nClassification report:")
    print(
        classification_report(
            gold_labels,
            pred_labels,
            labels=list(range(4)),
            target_names=[ID2LABEL[i] for i in range(4)],
            zero_division=0,
        )
    )

    print("\\nConfusion matrix:")
    print(confusion_matrix(gold_labels, pred_labels, labels=list(range(4))))

    return acc, macro_f1, weighted_f1


dev_cls_acc, dev_cls_macro_f1, dev_cls_weighted_f1 = (
    evaluate_dev_predictions_classification_f1(
        dev_claims=dev_claims, predictions=dev_predictions_cnn_bilstm
    )
)

# Run official eval.py for Evidence F-score + Harmonic Mean
subprocess.check_call(
    [
        sys.executable,
        f"{PROJECT_ROOT}/eval.py",
        "--predictions",
        str(DEV_PRED_PATH),
        "--groundtruth",
        str(DEV_PATH),
    ]
)""")
)

CELLS.append(
    code("""# @title 3.3 · Generate final predictions on test set (for submission)

test_claims = load_json(TEST_PATH)

TEST_PRED_PATH = OUTPUT_DIR / "test_predictions_cnn_bilstm_multihead_bal.json"

test_predictions = predict_claims_cnn_bilstm(
    claims_json=test_claims,
    evidence_corpus=evidence_corpus,
    retriever=retriever,
    model=cnn_bilstm_multihead_model,
    vocab=vocab,
    output_path=TEST_PRED_PATH,
    retrieval_top_k=10,
    max_evidence=4,
    batch_size=32,
    max_len=256,
    is_test=True,
    device=device,
)

print(f"\\nTest predictions saved to {TEST_PRED_PATH}")""")
)

CELLS.append(
    md(
        "## Object Oriented Programming codes here\n"
        "\n"
        "All OOP code is inline above:\n"
        "\n"
        "| Class | Section | Purpose |\n"
        "|-------|---------|---------|\n"
        "| `BM25CERetriever` | 1.3 | Adapter wrapping BM25 top-200 candidate retrieval + cross-encoder reranking |\n"
        "| `CNNBiLSTMDataset` | 2.1 | Pairs claim with gold (train) or retrieved (predict) evidence |\n"
        "| `AttentionPooling` | 2.2 | Learnable weighted-sum pooling over sequence |\n"
        "| `CNNBiLSTMMultiheadClassifier` | 2.2 | Multi-kernel CNN → BiLSTM → Multi-head Attention → AttentionPooling → MLP |\n"
    )
)


# ---------- write notebook ----------
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {OUT}")
