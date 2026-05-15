"""
retriever.py — Evidence retrieval module for COMP90042 fact-checking pipeline.

Classes:  BM25Retriever, DenseRetriever, HybridRetriever
Helpers:  load_bm25_index()
Training: train_cross_encoder()
"""

import os
import re
import json
import pickle as _pkl
import random
import gc
import shutil

import numpy as np
import torch
import bm25s
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import CrossEncoder, SentenceTransformer
from huggingface_hub import snapshot_download, create_repo, upload_folder

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

_ABBREVS = {
    "US": "United States",
    "UK": "United Kingdom",
    "EU": "European Union",
    "UN": "United Nations",
}



def normalize_claim(text):
    text = re.sub(r"(\d),(?=\d)", r"\1", text)
    text = re.sub(r"\b(\d{4})-(\d{2})-(\d{2})\b", r"\3/\2/\1", text)
    for abbr, full in _ABBREVS.items():
        text = re.sub(rf"\b{abbr}\b", full, text)
    return text


# ── BM25Retriever ─────────────────────────────────────────────────────────────

class BM25Retriever:
    """BM25-based evidence retrieval over the flat evidence corpus.

    PATCH P1: _preprocess now removes stopwords and applies Porter stemming
    for better term-matching quality. Default top_k raised to 10.
    """

    def __init__(self, evidence_dict, k1=1.5, b=0.75):
        self.ids   = list(evidence_dict.keys())
        self.texts = list(evidence_dict.values())
        self.k1    = k1
        self.b     = b
        self.bm25  = None

    def _preprocess(self, text):
        text   = text.lower()
        text   = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        tokens = [STEMMER.stem(t) for t in tokens if t not in STOP_WORDS]
        return tokens

    def build_index(self):
        tokenized = [self._preprocess(t) for t in self.texts]
        self.bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self.bm25.index(tokenized)
        print(f"BM25 index built over {len(self.texts):,} passages (k1={self.k1}, b={self.b})")

    def save(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        self.bm25.save(cache_dir)
        with open(os.path.join(cache_dir, "corpus_meta.json"), "w") as f:
            json.dump({"ids": self.ids, "texts": self.texts}, f)

    @classmethod
    def load(cls, cache_dir, evidence_dict):
        obj = cls.__new__(cls)
        obj.ids   = list(evidence_dict.keys())
        obj.texts = list(evidence_dict.values())
        meta_path = os.path.join(cache_dir, "corpus_meta.json")
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path))
            obj.ids, obj.texts = meta["ids"], meta["texts"]
        obj.bm25 = bm25s.BM25.load(cache_dir)
        return obj

    def retrieve(self, claim, top_k=10):
        if self.bm25 is None:
            raise RuntimeError("Call build_index() first")
        claim   = normalize_claim(claim)
        tokens  = self._preprocess(claim)
        if not tokens:
            return []
        results, scores = self.bm25.retrieve([tokens], k=min(top_k, len(self.ids)))
        indices = results[0].tolist()
        scores  = scores[0].tolist()
        return [
            {"id": self.ids[i], "text": self.texts[i], "score": float(s)}
            for i, s in zip(indices, scores)
        ]


# ── DenseRetriever ────────────────────────────────────────────────────────────

class DenseRetriever:
    """Dense bi-encoder retrieval — first-stage candidate generation.

    Encodes the full evidence corpus once (cached to Drive).
    At query time returns top-k passages by cosine similarity.
    Complements BM25 by handling vocabulary mismatch.
    """

    EMBED_MODEL = "BAAI/bge-base-en-v1.5"  # 109M params, 768-dim; fp16 ~2.5-3h on T4

    def __init__(self, evidence_dict):
        self.ids        = list(evidence_dict.keys())
        self.texts      = list(evidence_dict.values())
        self.model      = None
        self.embeddings = None   # shape: (N, D), L2-normalised

    def build_index(self, cache_path=None, batch_size=128, use_faiss=True):
        """Encode corpus into dense vectors; load from Drive cache if available.

        A .meta.json sidecar is saved alongside the .npy file recording which
        model produced it. If the sidecar shows a different model than
        EMBED_MODEL, the stale cache is deleted and the corpus is re-encoded.
        Falls back to numpy dot-product if faiss is not installed.
        """
        meta_path = (cache_path + ".meta.json") if cache_path else None

        if cache_path and os.path.exists(cache_path):
            # Check whether the cache was built with the current model
            stale = False
            if meta_path and os.path.exists(meta_path):
                meta = json.load(open(meta_path))
                if meta.get("embed_model") != self.EMBED_MODEL:
                    print(
                        f"  Stale cache: built with '{meta['embed_model']}', "
                        f"current model is '{self.EMBED_MODEL}'. Re-encoding..."
                    )
                    os.remove(cache_path)
                    os.remove(meta_path)
                    stale = True
            else:
                # No sidecar — peek at the dimension to catch silent mismatches
                _probe = np.load(cache_path, mmap_mode="r")
                _dim   = _probe.shape[1] if _probe.ndim == 2 else -1
                del _probe
                # Known dim for common models
                _known = {"all-MiniLM-L6-v2": 384, "bge-large-en-v1.5": 1024,
                          "bge-base-en-v1.5": 768}
                _expected = next(
                    (d for k, d in _known.items() if k in self.EMBED_MODEL), None
                )
                if _expected and _dim != _expected:
                    print(
                        f"  Stale cache: dim={_dim} does not match "
                        f"'{self.EMBED_MODEL}' (expected {_expected}). Re-encoding..."
                    )
                    os.remove(cache_path)
                    stale = True

            if not stale:
                print(f"Loading dense embeddings from {cache_path}")
                self.embeddings = np.load(cache_path)
                print(f"  Loaded: {self.embeddings.shape}")
                self._build_faiss_if_available(use_faiss)
                return

        print(f"Loading encoder: {self.EMBED_MODEL}")
        encode_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(self.EMBED_MODEL, device=encode_device)
        if encode_device == "cuda":
            self.model.half()  # fp16: halves weights + activations; T4 tensor cores make it ~1.5x faster
        print(f"Encoding {len(self.texts):,} passages "
              f"(batch={batch_size}, device={encode_device}, fp16={encode_device=='cuda'}) ...")
        self.embeddings = self.model.encode(
            self.texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            np.save(cache_path, self.embeddings)
            if meta_path:
                json.dump(
                    {"embed_model": self.EMBED_MODEL,
                     "shape": list(self.embeddings.shape)},
                    open(meta_path, "w"),
                )
            print(f"  Saved to: {cache_path}  (meta: {meta_path})")
        print(f"Dense index ready: {self.embeddings.shape}")
        # Offload encoder from GPU now that corpus embeddings are saved in RAM.
        # This frees ~1.3 GB VRAM before CrossEncoder loads in the next step.
        if self.model is not None and torch.cuda.is_available():
            self.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            print("Dense encoder offloaded to CPU (GPU freed for CrossEncoder).")
        self._build_faiss_if_available(use_faiss)

    def _build_faiss_if_available(self, use_faiss):
        self._faiss_index = None
        if not use_faiss:
            return
        try:
            import faiss
            d = self.embeddings.shape[1]
            nlist = min(128, max(1, len(self.embeddings) // 39))
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(self.embeddings.astype("float32"))
            index.add(self.embeddings.astype("float32"))
            index.nprobe = 64
            self._faiss_index = index
            print(f"FAISS IVF index built (nlist={nlist}, nprobe=64).")
        except ImportError:
            print("faiss not installed — falling back to numpy dot-product retrieval.")

    def retrieve(self, claim, top_k=50):
        if self.embeddings is None:
            raise RuntimeError(
                "DenseRetriever has no corpus embeddings. "
                "Call build_index() first — if it OOM-ed previously, "
                "reduce batch_size (default is now 32)."
            )
        if self.model is None:
            self.model = SentenceTransformer(self.EMBED_MODEL)
        claim_emb = self.model.encode([claim], normalize_embeddings=True).astype("float32")
        if getattr(self, "_faiss_index", None) is not None:
            scores, idxs = self._faiss_index.search(claim_emb.astype("float32"), top_k)
            scores, idxs = scores[0], idxs[0]
            return [
                {"id": self.ids[i], "text": self.texts[i], "score": float(s)}
                for i, s in zip(idxs, scores) if i >= 0
            ]
        scores  = (claim_emb @ self.embeddings.T)[0]
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return [
            {"id": self.ids[i], "text": self.texts[i], "score": float(scores[i])}
            for i in top_idx
        ]


# ── HybridRetriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    """BM25 + Dense candidate fusion via RRF, then CrossEncoder re-ranking.

    With dense_retriever=None: BM25 top_k*6 → CrossEncoder.
    With dense_retriever set: BM25 top-50 + Dense top-50 merged via RRF → CrossEncoder.
    RRF (Reciprocal Rank Fusion) boosts passages that rank highly in both lists,
    improving recall over either source alone without needing score calibration.
    """

    CE_MODEL = "cross-encoder/nli-deberta-v3-base"  # NLI-trained; better domain alignment for fact-checking than bge-reranker (MS-MARCO)
    _RRF_K   = 60

    def __init__(self, evidence_dict, bm25_retriever, dense_retriever=None):
        self.bm25_retriever  = bm25_retriever
        self.dense_retriever = dense_retriever
        self.ids             = list(evidence_dict.keys())
        self.texts           = list(evidence_dict.values())
        self.cross_encoder   = None

    def build_dense_index(self):
        print(f"Loading CrossEncoder: {self.CE_MODEL}")
        ce_device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load on CPU first, convert to fp16, then move to GPU.
        # This avoids holding a full fp32 copy (~1.4 GB) on GPU during loading.
        self.cross_encoder = CrossEncoder(self.CE_MODEL, device="cpu", max_length=256)
        if ce_device == "cuda":
            self.cross_encoder.model.half()
            self.cross_encoder.model.to(ce_device)
            self.cross_encoder._target_device = torch.device(ce_device)
        print(f"CrossEncoder loaded on {ce_device} (fp16={ce_device == 'cuda'}).")

    @staticmethod
    def _rrf_merge(bm25_results, dense_results, k=60):
        rrf_scores = {}
        for rank, item in enumerate(bm25_results):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
        for rank, item in enumerate(dense_results):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + 1.0 / (k + rank + 1)
        id_to_item = {item["id"]: item for item in dense_results}
        id_to_item.update({item["id"]: item for item in bm25_results})
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)
        return [id_to_item[eid] for eid in sorted_ids]

    def retrieve(self, claim, top_k=10):
        if self.dense_retriever is not None:
            bm25_cands  = self.bm25_retriever.retrieve(claim, top_k=50)
            dense_cands = self.dense_retriever.retrieve(claim, top_k=50)
            candidates  = self._rrf_merge(bm25_cands, dense_cands, k=self._RRF_K)[:top_k * 5]
        else:
            candidates = self.bm25_retriever.retrieve(claim, top_k=top_k * 5)

        if self.cross_encoder is None:
            return candidates[:top_k]

        pairs  = [(claim, c["text"]) for c in candidates]
        if not pairs:
            return []
        scores = self.cross_encoder.predict(pairs, batch_size=8)
        for cand, sc in zip(candidates, scores.tolist()):
            cand["ce_score"] = sc
        ranked = sorted(candidates, key=lambda x: x["ce_score"], reverse=True)
        return ranked[:top_k]


# ── Three-tier BM25 loader ─────────────────────────────────────────────────────

def load_bm25_index(cache_dir, evidence_dict, hub_repo=""):
    """Three-tier load: local Drive → HF Hub → rebuild from evidence_dict."""
    if os.path.exists(os.path.join(cache_dir, "params.index.json")):
        print(f"Loading BM25 index from Drive: {cache_dir}")
        return BM25Retriever.load(cache_dir, evidence_dict)

    if hub_repo and not hub_repo.startswith("your-hf"):
        try:
            print(f"Trying HF Hub: {hub_repo} ...")
            local_hub = snapshot_download(repo_id=hub_repo, repo_type="model")
            r = BM25Retriever.load(local_hub, evidence_dict)
            r.save(cache_dir)
            print(f"  Downloaded from Hub and cached to Drive: {cache_dir}")
            return r
        except Exception as e:
            print(f"  Hub load failed ({e}); rebuilding from evidence_dict ...")

    print("Building BM25 index from evidence_dict (one-time, ~1-2 min) ...")
    r = BM25Retriever(evidence_dict)
    r.build_index()
    _tmp_dir = cache_dir + ".tmp"
    r.save(_tmp_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.replace(_tmp_dir, cache_dir)
    print(f"  Saved to Drive: {cache_dir}")
    if hub_repo and not hub_repo.startswith("your-hf"):
        try:
            create_repo(repo_id=hub_repo, repo_type="model", exist_ok=True)
            upload_folder(folder_path=cache_dir, repo_id=hub_repo,
                          commit_message="bm25s index built from evidence.json")
            print(f"  Pushed to HF Hub: {hub_repo}")
        except Exception as e:
            print(f"  Hub push failed (Drive cache intact): {e}")
    return r


# ── train_cross_encoder ────────────────────────────────────────────────────────

def train_cross_encoder(
    train_data,
    evidence_dict,
    bm25_retriever,
    ce_finetuned_dir,
    hf_ce_repo="",
    base_model="cross-encoder/nli-deberta-v3-base",
    epochs=3,
    lr=1e-6,
    batch_size=4,
    val_split=0.1,
    reuse_if_found=True,
    pairs_cache_path=None,
    gpu_models_to_offload=None,
):
    """Fine-tune a CrossEncoder on claim-passage pairs from train_data.

    Positive pairs: (claim, gold_passage) → label 1
    Negatives:      (claim, BM25 non-gold passage) → label 0  [1:1 ratio, B1 fix]
    Best checkpoint tracked across all epochs (B2 fix).

    Args:
        gpu_models_to_offload: list of nn.Module objects to move to CPU before
            training (e.g. [classifier.model, dense_retriever.model]).
            They are NOT moved back — caller is responsible for restoring.

    Returns:
        CrossEncoder: the best-checkpoint model loaded from ce_finetuned_dir.
    """
    from sentence_transformers import InputExample
    from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
    from torch.utils.data import DataLoader as SentDataLoader

    # Offload competing GPU models before loading CrossEncoder
    if gpu_models_to_offload:
        for m in gpu_models_to_offload:
            if m is not None:
                try:
                    m.cpu()
                except Exception:
                    pass
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"GPU memory freed. Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Three-tier CE checkpoint loader
    def _try_load_ce(path):
        if not (os.path.exists(path) and os.path.isdir(path)):
            return None
        has_model = any(f.endswith((".json", ".bin", ".safetensors")) for f in os.listdir(path))
        if not has_model:
            return None
        try:
            # Load on CPU for the sanity-check prediction — no GPU memory needed
            # for a single test pair, and avoids a ~750 MB transient VRAM spike.
            m = CrossEncoder(path, device="cpu")
            score = m.predict([["test claim", "test passage"]])[0]
            print(f"  Loaded CE from {path}  test_score={score:.4f}")
            return m
        except Exception as e:
            print(f"  CE load failed at {path}: {e}")
            return None

    m = _try_load_ce(ce_finetuned_dir)
    if m and reuse_if_found:
        print(f"reuse_if_found=True → CrossEncoder training SKIPPED. Returning cached model.")
        return m
    if not m and hf_ce_repo and not hf_ce_repo.startswith("your-hf"):
        try:
            print(f"Trying HF Hub: {hf_ce_repo} ...")
            hub_local = snapshot_download(repo_id=hf_ce_repo, repo_type="model")
            m = _try_load_ce(hub_local)
            if m:
                m.save(ce_finetuned_dir)
                print(f"  Cached to Drive: {ce_finetuned_dir}")
                if reuse_if_found:
                    return m
        except Exception as e:
            print(f"  HF Hub load failed: {e}")

    # Free cached CE from GPU — it was only needed for the reuse check above.
    # Keeping it alive wastes ~750 MB VRAM while the training model loads below.
    if m is not None:
        try:
            m.model.cpu()
        except Exception:
            pass
        del m
        gc.collect()
        torch.cuda.empty_cache()

    # Build or load training pairs
    cache_path = pairs_cache_path or os.path.join(os.path.dirname(ce_finetuned_dir), "ce_train_pairs.pkl")
    if os.path.exists(cache_path):
        print(f"Loading CE training pairs from cache: {cache_path}")
        with open(cache_path, "rb") as _f:
            ce_train_samples = _pkl.load(_f)
    else:
        print("Building CE training pairs (hard-negative mining) ...")
        ce_train_samples = []
        for item in train_data.values():
            claim    = item["claim_text"]
            gold_ids = set(item.get("evidences", []))
            if not gold_ids:
                continue
            for eid in gold_ids:
                if eid in evidence_dict:
                    ce_train_samples.append(InputExample(texts=[claim, evidence_dict[eid]], label=1.0))
            # B1 fix: one negative per positive (1:1 ratio)
            candidates = bm25_retriever.retrieve(claim, top_k=20)
            neg_pool = [c for c in candidates if c["id"] not in gold_ids]
            for c in neg_pool[:len(gold_ids)]:
                ce_train_samples.append(InputExample(texts=[claim, c["text"]], label=0.0))
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as _f:
            _pkl.dump(ce_train_samples, _f, protocol=_pkl.HIGHEST_PROTOCOL)
        print(f"  Cached {len(ce_train_samples):,} pairs to {cache_path}")

    print(f"CrossEncoder training pairs: {len(ce_train_samples):,}")

    random.seed(42)
    random.shuffle(ce_train_samples)
    split_idx      = int((1 - val_split) * len(ce_train_samples))
    ce_train_only  = ce_train_samples[:split_idx]
    ce_val_samples = ce_train_samples[split_idx:]
    ce_val_pairs   = [s.texts for s in ce_val_samples]
    ce_val_labels  = [int(s.label) for s in ce_val_samples]
    ce_evaluator   = CrossEncoderClassificationEvaluator(
        sentence_pairs=ce_val_pairs, labels=ce_val_labels, name="ce-val",
    )
    print(f"  CE train: {len(ce_train_only):,}  CE val: {len(ce_val_samples):,}")

    ce_device     = "cuda" if torch.cuda.is_available() else "cpu"
    ce_model      = CrossEncoder(base_model, num_labels=1, device=ce_device, max_length=256,
                               model_kwargs={"ignore_mismatched_sizes": True})
    ce_model.model.gradient_checkpointing_enable()
    print(f"CrossEncoder running on: {ce_device} (gradient checkpointing enabled)")

    try:
        import bitsandbytes as bnb
        _opt_cls = bnb.optim.AdamW8bit
        print("CE optimizer: 8-bit AdamW (saves ~1.3 GB VRAM vs fp32 Adam).")
    except ImportError:
        _opt_cls = torch.optim.AdamW
        print("CE optimizer: fp32 AdamW (bitsandbytes not found).")

    ce_train_loader = SentDataLoader(ce_train_only, shuffle=True, batch_size=batch_size)

    # B2 fix: track best score across all epochs; save only on improvement
    _best_score = -1.0
    for _epoch in range(epochs):
        ce_model.fit(
            train_dataloader=ce_train_loader,
            epochs=1,
            warmup_steps=100 if _epoch == 0 else 0,
            optimizer_class=_opt_cls,
            optimizer_params={"lr": lr},
            use_amp=(ce_device == "cuda"),
            show_progress_bar=True,
        )
        _result = ce_evaluator(ce_model, output_path=None)
        if isinstance(_result, dict):
            _epoch_score = next(
                (v for k, v in _result.items() if "_ap" in k.lower()),
                next(iter(_result.values()))
            )
        else:
            _epoch_score = float(_result)

        if _epoch_score > _best_score:
            _best_score = _epoch_score
            ce_model.save(ce_finetuned_dir)
            print(f"  Epoch {_epoch+1}/{epochs} — new best ({_epoch_score:.4f}) saved → {ce_finetuned_dir}")
        else:
            print(f"  Epoch {_epoch+1}/{epochs} — score {_epoch_score:.4f} ≤ best {_best_score:.4f}, not saved")

        if os.path.isdir(ce_finetuned_dir) and hf_ce_repo and not hf_ce_repo.startswith("your-hf"):
            try:
                create_repo(repo_id=hf_ce_repo, repo_type="model", exist_ok=True)
                upload_folder(
                    folder_path=ce_finetuned_dir,
                    repo_id=hf_ce_repo,
                    commit_message=f"CE fine-tuned epoch {_epoch+1}/{epochs} best_val={_best_score:.4f}",
                )
                print(f"  → Pushed to HF Hub: {hf_ce_repo}")
            except Exception as _e:
                print(f"  → Hub push failed (Drive checkpoint intact): {_e}")

    print(f"CrossEncoder fine-tuning complete. Best val={_best_score:.4f}. Checkpoint: {ce_finetuned_dir}")
    return CrossEncoder(ce_finetuned_dir, device=ce_device)
