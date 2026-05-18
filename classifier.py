"""
classifier.py — NLI classification module for COMP90042 fact-checking pipeline.

Classes:  NLIClassifier, FactCheckPipeline, FactCheckDataset
Helpers:  eval_macro_f1_dev(), load_deberta_checkpoint()
Training: train_deberta()
"""

import os
import gc
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
try:
    from bitsandbytes.optim import AdamW8bit as AdamW
    print("Using 8-bit AdamW (bitsandbytes) — optimizer memory ~0.37 GB vs 1.46 GB fp32.")
except ImportError:
    from torch.optim import AdamW
    print("bitsandbytes not found — falling back to fp32 AdamW (may OOM on 15 GB GPU).")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, f1_score

LABEL2ID    = {"SUPPORTS": 0, "REFUTES": 1, "NOT_ENOUGH_INFO": 2, "DISPUTED": 3}
ID2LABEL    = {v: k for k, v in LABEL2ID.items()}
LABEL_NAMES = ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFO", "DISPUTED"]


# ── NLIClassifier ─────────────────────────────────────────────────────────────

class NLIClassifier:
    """
    Wraps a HuggingFace sequence-classification model for fact-checking.

    Zero-shot NLI model (3-class):
        label_map = {entailment->SUPPORTS, contradiction->REFUTES, neutral->NOT_ENOUGH_INFO}
    Fine-tuned 4-class model:
        label_map = {supports->SUPPORTS, refutes->REFUTES, ...}  (pass-through)
    """

    DEFAULT_LABEL_MAP = {
        "entailment":    "SUPPORTS",
        "contradiction": "REFUTES",
        "neutral":       "NOT_ENOUGH_INFO",
    }

    def __init__(self, model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", device=0):
        self.device = torch.device(
            f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu"
        )
        print(f"Loading model on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model     = self.model.half().to(self.device)
        self.model.eval()
        self.label_map = dict(self.DEFAULT_LABEL_MAP)
        print("Model ready.")

    def score_single(self, claim, premise):
        """Returns {fact_check_label: probability} after applying label_map."""
        inputs = self.tokenizer(
            premise, claim,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits.float(), dim=-1).squeeze().cpu().tolist()
        scores = {}
        for i, p in enumerate(probs):
            raw        = self.model.config.id2label[i].lower()
            fact_label = self.label_map.get(raw, "NOT_ENOUGH_INFO")
            scores[fact_label] = scores.get(fact_label, 0.0) + p
        return scores


# ── FactCheckPipeline ─────────────────────────────────────────────────────────

class FactCheckPipeline:
    """
    End-to-end fact-checking pipeline.

    predict() supports oracle/pipeline dual eval modes:
      Oracle mode (gold_evidence_ids): skips retrieval, measures classifier ceiling.
      Pipeline mode (default): runs full BM25 → CrossEncoder retrieval.

    P0: DISPUTED predicted by a learned classifier (no threshold rule).
    P3: Top-3 passages concatenated with [SEP] for joint multi-evidence scoring.
    """

    def __init__(self, retriever, classifier, top_k=10, ev_return_k=5):
        self.retriever   = retriever
        self.classifier  = classifier
        self.ev_return_k = ev_return_k
        self.top_k       = top_k
        self._ev_lookup  = None

    def _get_ev_lookup(self):
        if self._ev_lookup is None:
            self._ev_lookup = {
                eid: self.retriever.texts[i]
                for i, eid in enumerate(self.retriever.ids)
            }
        return self._ev_lookup

    def predict(self, claim, gold_evidence_ids=None):
        """
        Returns (label, evidence_ids_list).

        gold_evidence_ids: pass item["evidences"] for oracle mode (dev eval only).
        """
        if gold_evidence_ids is not None:
            ev_lookup = self._get_ev_lookup()
            passages  = [{"id": eid, "text": ev_lookup[eid]}
                         for eid in gold_evidence_ids if eid in ev_lookup]
        else:
            passages = self.retriever.retrieve(claim, top_k=self.top_k)

        if not passages:
            return "NOT_ENOUGH_INFO", []

        # P3: top-3 passages concatenated for joint NLI scoring
        selected_passages = passages[:3]
        combined = " [SEP] ".join(p["text"] for p in selected_passages)
        ev_ids   = [p["id"] for p in passages[:self.ev_return_k]]

        scores = self.classifier.score_single(claim, combined)
        return max(scores, key=scores.get), ev_ids


# ── FactCheckDataset ──────────────────────────────────────────────────────────

class FactCheckDataset(Dataset):
    """
    PyTorch Dataset for fine-tuning DeBERTa on 4-class fact-checking.

    Input:  evidence text [SEP] claim
    Output: label index (SUPPORTS=0, REFUTES=1, NOT_ENOUGH_INFO=2, DISPUTED=3)

    P1: For NOT_ENOUGH_INFO claims, gold evidence is replaced with
        BM25-retrieved passages (topically related, not entailing).
    P2: Exposes class_weights for weighted cross-entropy loss.
    """

    def __init__(self, data_dict, evidence_dict, tokenizer,
                 bm25_retriever=None, max_length=512, max_ev=3):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.items      = []

        for item in data_dict.values():
            claim = item["claim_text"]
            if item["claim_label"] not in LABEL2ID:
                continue
            label_id = LABEL2ID[item["claim_label"]]

            if item["claim_label"] == "NOT_ENOUGH_INFO" and bm25_retriever is not None:
                retrieved = bm25_retriever.retrieve(claim, top_k=max_ev)
                ev_text   = " ".join(r["text"] for r in retrieved)
            else:
                ev_text = " ".join(
                    evidence_dict.get(ev, "") for ev in item.get("evidences", [])[:max_ev]
                )
            self.items.append((claim, ev_text, label_id))

        label_counts = Counter(item[2] for item in self.items)
        total        = len(self.items)
        n            = len(LABEL2ID)
        self.class_weights = torch.tensor(
            [total / (label_counts.get(i, 1) * n) for i in range(n)],
            dtype=torch.float,
        )
        print(f"Dataset: {len(self.items):,} examples | {n} classes")
        print(f"Class weights: { {ID2LABEL[i]: f'{self.class_weights[i].item():.3f}' for i in range(n)} }")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        claim, evidence, label = self.items[idx]
        enc = self.tokenizer(
            evidence, claim,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            {k: v.squeeze(0) for k, v in enc.items()},
            torch.tensor(label, dtype=torch.long),
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def eval_macro_f1_dev(model, tokenizer, dev_data, evidence_dict, device,
                      batch_size=16, label2id=None, label_names=None,
                      bm25_retriever=None):
    """Dev macro-F1 using oracle evidence. Returns (macro_f1, y_true, y_pred).

    id2label is always derived from model.config.id2label — not a parameter.
    bm25_retriever: if provided, NEI examples use retrieved evidence (matches training).
    """
    _label2id    = label2id    or LABEL2ID
    _id2label    = {int(k): v for k, v in model.config.id2label.items()}
    _label_names = label_names or LABEL_NAMES
    model.eval()
    y_true, y_pred = [], []
    items = []
    for v in dev_data.values():
        if v["claim_label"] not in _label2id:
            continue
        if v["claim_label"] == "NOT_ENOUGH_INFO" and bm25_retriever is not None:
            ev_text = " ".join(
                r["text"] for r in bm25_retriever.retrieve(v["claim_text"], top_k=3)
            )
        else:
            ev_text = " ".join(
                evidence_dict.get(ev, "") for ev in v.get("evidences", [])[:3]
            )
        items.append((v["claim_text"], ev_text, v["claim_label"]))
    for i in range(0, len(items), batch_size):
        batch    = items[i : i + batch_size]
        claims   = [x[0] for x in batch]
        ev_texts = [x[1] for x in batch]
        y_true.extend(x[2] for x in batch)
        enc = tokenizer(ev_texts, claims, max_length=256,
                        truncation=True, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            preds = model(**enc).logits.argmax(dim=-1).cpu().tolist()
        y_pred.extend(_id2label[p] for p in preds)
    return (
        f1_score(y_true, y_pred, average="macro", labels=_label_names, zero_division=0),
        y_true,
        y_pred,
    )


def load_deberta_checkpoint(local_path, hub_repo="", num_labels=4):
    """Try local Drive path first, then HF Hub.
    Returns (model, tokenizer, source_path) or (None, None, None).
    """
    _id2label = {v: k for k, v in list(LABEL2ID.items())[:num_labels]}
    _label2id = {k: v for k, v in list(LABEL2ID.items())[:num_labels]}

    if os.path.exists(local_path) and os.path.isdir(local_path):
        has_config  = os.path.exists(os.path.join(local_path, "config.json"))
        has_weights = any(
            f.endswith(".bin") or f.endswith(".safetensors")
            for f in os.listdir(local_path)
        )
        if has_config and has_weights:
            print(f"Found local checkpoint: {os.path.abspath(local_path)}")
            try:
                m = AutoModelForSequenceClassification.from_pretrained(
                    local_path, num_labels=num_labels,
                    id2label=_id2label, label2id=_label2id,
                    ignore_mismatched_sizes=True,
                )
                t = AutoTokenizer.from_pretrained(local_path)
                return m, t, local_path
            except Exception as e:
                print(f"  Local load failed: {e}")

    if hub_repo and not hub_repo.startswith("your-hf"):
        print(f"Trying HuggingFace Hub: {hub_repo}")
        try:
            m = AutoModelForSequenceClassification.from_pretrained(
                hub_repo, num_labels=num_labels,
                id2label=_id2label, label2id=_label2id,
                ignore_mismatched_sizes=True,
            )
            t = AutoTokenizer.from_pretrained(hub_repo)
            print(f"  Loaded from Hub: {hub_repo}")
            return m, t, hub_repo
        except Exception as e:
            print(f"  Hub load failed: {e}")

    return None, None, None


# ── train_deberta ─────────────────────────────────────────────────────────────

def train_deberta(
    train_data,
    dev_data,
    evidence_dict,
    bm25_retriever,
    deberta_best_dir,
    hf_deberta_repo="",
    base_model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    epochs=5,
    lr=2e-5,
    batch_size=4,
    patience=2,
    disputed_weight_boost=3.0,
    reuse_if_found=True,
    gpu_models_to_offload=None,
):
    """Fine-tune DeBERTa for 4-class fact-checking.

    Args:
        gpu_models_to_offload: list of nn.Module objects (e.g. CrossEncoder, DenseRetriever
            embedding model) to move to CPU before training. NOT moved back — caller restores.

    Returns:
        (model, tokenizer): best checkpoint loaded from deberta_best_dir.
    """
    ft_device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ft_tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Offload competing GPU models
    if gpu_models_to_offload:
        _freed = []
        for m in gpu_models_to_offload:
            if m is not None:
                try:
                    m.cpu()
                    _freed.append(type(m).__name__)
                except Exception:
                    pass
        if _freed:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Moved to CPU: {', '.join(_freed)}")
            if torch.cuda.is_available():
                print(f"GPU free: {torch.cuda.mem_get_info()[0]/1e9:.1f} GB")

    # Checkpoint detection
    _ckpt_model, _ckpt_tok, _ckpt_src = load_deberta_checkpoint(deberta_best_dir, hf_deberta_repo, num_labels=4)

    if _ckpt_model is not None and reuse_if_found:
        print("Evaluating checkpoint on dev set (oracle mode) ...")
        _ckpt_model = _ckpt_model.half().to(ft_device)
        _ckpt_model.eval()
        _pre_f1, _pre_yt, _pre_yp = eval_macro_f1_dev(
            _ckpt_model, _ckpt_tok, dev_data, evidence_dict, ft_device
        )
        print(f"Dev macro-F1 (oracle): {_pre_f1:.4f}")
        print(classification_report(_pre_yt, _pre_yp, labels=LABEL_NAMES, zero_division=0))
        print("reuse_if_found=True → Training SKIPPED.")
        return _ckpt_model, _ckpt_tok

    # No cached model (or force-retrain) — load fresh base
    if _ckpt_model is not None:
        del _ckpt_model
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Loading base model for fine-tuning: {base_model}")
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    # Build dataset
    train_dataset = FactCheckDataset(
        train_data, evidence_dict, ft_tokenizer,
        bm25_retriever=bm25_retriever,
        max_length=256, max_ev=3,
    )
    # Boost DISPUTED weight — auto-computed weight is insufficient for the rare class
    train_dataset.class_weights[3] *= disputed_weight_boost
    print(f"DISPUTED weight boosted ×{disputed_weight_boost:.1f} → {train_dataset.class_weights.tolist()}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train: {len(train_dataset):,} examples | {len(train_loader)} batches/epoch")

    ft_model = ft_model.float().to(ft_device)
    ft_model.gradient_checkpointing_enable()
    use_amp   = ft_device.type == "cuda"

    optimizer   = AdamW(ft_model.parameters(), lr=lr, weight_decay=0.1)
    total_steps = len(train_loader) * epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = total_steps // 10,
        num_training_steps= total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    weighted_loss_fn = torch.nn.CrossEntropyLoss(
        weight=train_dataset.class_weights.to(device=ft_device, dtype=torch.float32)
    )

    best_f1    = 0.0
    no_improve = 0
    for epoch in range(epochs):
        ft_model.train()
        total_loss = 0.0
        for batch_enc, batch_labels in train_loader:
            batch_enc    = {k: v.to(ft_device) for k, v in batch_enc.items()}
            batch_labels = batch_labels.to(ft_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=ft_device.type, dtype=torch.float16, enabled=use_amp):
                logits = ft_model(**batch_enc).logits
                loss   = weighted_loss_fn(logits.float(), batch_labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()

        avg_loss     = total_loss / len(train_loader)
        dev_f1, _, _ = eval_macro_f1_dev(ft_model, ft_tokenizer, dev_data, evidence_dict, ft_device,
                                         batch_size=8)
        print(f"Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  dev macro-F1={dev_f1:.4f}")

        if dev_f1 > best_f1:
            best_f1    = dev_f1
            no_improve = 0
            ft_model.save_pretrained(deberta_best_dir)
            ft_tokenizer.save_pretrained(deberta_best_dir)
            try:
                ft_model.push_to_hub(
                    hf_deberta_repo,
                    commit_message=f"Best checkpoint epoch {epoch+1} F1={dev_f1:.4f}",
                )
                ft_tokenizer.push_to_hub(hf_deberta_repo)
                print(f"  → New best ({best_f1:.4f}). Saved locally + pushed to Hub.")
            except Exception as e:
                print(f"  → New best ({best_f1:.4f}). Saved locally; Hub push FAILED: {e}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stop at epoch {epoch+1}. Best dev macro-F1: {best_f1:.4f}")
                break

    print(f"Training complete. Best dev macro-F1: {best_f1:.4f}")

    # Free training model + optimizer before reloading best checkpoint (prevents double-load OOM)
    del ft_model, optimizer, scheduler, scaler
    gc.collect()
    torch.cuda.empty_cache()

    # Reload best checkpoint
    ft_model = AutoModelForSequenceClassification.from_pretrained(
        deberta_best_dir, id2label=ID2LABEL, label2id=LABEL2ID,
    ).to(ft_device)
    ft_tokenizer = AutoTokenizer.from_pretrained(deberta_best_dir)
    print(f"Best model reloaded from {deberta_best_dir}")
    return ft_model, ft_tokenizer


