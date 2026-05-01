import json
import re
import os
import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)


LABEL2ID = {
    "SUPPORTS": 0,
    "REFUTES": 1,
    "NOT_ENOUGH_INFO": 2,
    "DISPUTED": 3
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}


# ============================================================
# 1. Utility functions
# ============================================================

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalise_text(text: str) -> str:
    """
    Basic text cleaning.
    You can improve this later with lemmatisation, stopword removal, etc.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\,\-\%°]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def simple_tokenise(text: str) -> List[str]:
    return normalise_text(text).split()


def get_claim_items(claims_json: Dict) -> List[Tuple[str, Dict]]:
    """
    Converts:
    {
      "claim-123": {...},
      "claim-456": {...}
    }

    into:
    [
      ("claim-123", {...}),
      ("claim-456", {...})
    ]
    """
    return list(claims_json.items())


def concatenate_evidence(
    evidence_ids: List[str],
    evidence_corpus: Dict[str, str],
    max_evidence: int = 5
) -> str:
    """
    Convert evidence IDs into one text string.
    """
    selected_ids = evidence_ids[:max_evidence]

    evidence_texts = []
    for eid in selected_ids:
        if eid in evidence_corpus:
            evidence_texts.append(evidence_corpus[eid])

    if len(evidence_texts) == 0:
        return "No relevant evidence found."

    return " ".join(evidence_texts)


# ============================================================
# 2. BM25 Evidence Retriever
# ============================================================

class BM25Retriever:
    def __init__(self, evidence_corpus: Dict[str, str]):
        self.evidence_corpus = evidence_corpus
        self.evidence_ids = list(evidence_corpus.keys())
        self.evidence_texts = [evidence_corpus[eid] for eid in self.evidence_ids]

        print("Building BM25 index...")
        tokenised_corpus = [simple_tokenise(text) for text in tqdm(self.evidence_texts)]
        self.bm25 = BM25Okapi(tokenised_corpus)

    def retrieve(self, claim_text: str, top_k: int = 5) -> List[str]:
        query_tokens = simple_tokenise(claim_text)
        scores = self.bm25.get_scores(query_tokens)

        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved_ids = [self.evidence_ids[i] for i in top_indices]

        return retrieved_ids

    def evaluate_recall_at_k(
        self,
        claims_json: Dict,
        k: int = 5
    ) -> float:
        """
        Evaluates retrieval using gold evidence IDs.

        Recall@k:
        percentage of claims where at least one gold evidence appears
        in the retrieved top-k evidence list.
        """
        total = 0
        hit = 0

        for claim_id, instance in tqdm(get_claim_items(claims_json)):
            claim_text = instance["claim_text"]
            gold_evidence = set(instance.get("evidences", []))

            if len(gold_evidence) == 0:
                continue

            retrieved = set(self.retrieve(claim_text, top_k=k))

            if len(gold_evidence.intersection(retrieved)) > 0:
                hit += 1

            total += 1

        return hit / total if total > 0 else 0.0


# ============================================================
# 3. Dataset for Transformer Verifier
# ============================================================

class ClaimEvidenceDataset(Dataset):
    def __init__(
        self,
        claims_json: Dict,
        evidence_corpus: Dict[str, str],
        tokenizer,
        max_length: int = 512,
        max_evidence: int = 5,
        use_gold_evidence: bool = True,
        retriever: BM25Retriever = None,
        retrieval_top_k: int = 5,
        is_test: bool = False
    ):
        self.items = get_claim_items(claims_json)
        self.evidence_corpus = evidence_corpus
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_evidence = max_evidence
        self.use_gold_evidence = use_gold_evidence
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.is_test = is_test

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        claim_id, instance = self.items[idx]
        claim_text = instance["claim_text"]

        if self.use_gold_evidence and not self.is_test:
            evidence_ids = instance.get("evidences", [])
        else:
            if self.retriever is None:
                raise ValueError("Retriever is required when not using gold evidence.")
            evidence_ids = self.retriever.retrieve(claim_text, top_k=self.retrieval_top_k)

        evidence_text = concatenate_evidence(
            evidence_ids=evidence_ids,
            evidence_corpus=self.evidence_corpus,
            max_evidence=self.max_evidence
        )

        encoded = self.tokenizer(
            claim_text,
            evidence_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {
            "claim_id": claim_id,
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "evidence_ids": evidence_ids
        }

        if not self.is_test:
            label = instance["claim_label"]
            item["label"] = torch.tensor(LABEL2ID[label], dtype=torch.long)

        return item


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    attention_mask = torch.stack([x["attention_mask"] for x in batch])
    claim_ids = [x["claim_id"] for x in batch]
    evidence_ids = [x["evidence_ids"] for x in batch]

    output = {
        "claim_ids": claim_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "evidence_ids": evidence_ids
    }

    if "label" in batch[0]:
        labels = torch.stack([x["label"] for x in batch])
        output["labels"] = labels

    return output


# ============================================================
# 4. Train Transformer Verifier
# ============================================================

def train_verifier(
    train_claims: Dict,
    dev_claims: Dict,
    evidence_corpus: Dict[str, str],
    model_name: str = "distilroberta-base",
    output_dir: str = "outputs/verifier_model",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 512,
    max_evidence: int = 5,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    model.to(device)

    train_dataset = ClaimEvidenceDataset(
        claims_json=train_claims,
        evidence_corpus=evidence_corpus,
        tokenizer=tokenizer,
        max_length=max_length,
        max_evidence=max_evidence,
        use_gold_evidence=True,
        is_test=False
    )

    dev_dataset = ClaimEvidenceDataset(
        claims_json=dev_claims,
        evidence_corpus=evidence_corpus,
        tokenizer=tokenizer,
        max_length=max_length,
        max_evidence=max_evidence,
        use_gold_evidence=True,
        is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)

    total_steps = len(train_loader) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_macro_f1 = 0.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader):
            optimiser.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimiser.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Training loss: {avg_loss:.4f}")

        dev_acc, dev_macro_f1 = evaluate_verifier(
            model=model,
            dataloader=dev_loader,
            device=device
        )

        print(f"Dev accuracy with gold evidence: {dev_acc:.4f}")
        print(f"Dev macro F1 with gold evidence: {dev_macro_f1:.4f}")

        if dev_macro_f1 > best_macro_f1:
            best_macro_f1 = dev_macro_f1
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            print(f"Saved best model to {output_dir}")

    return model, tokenizer


def evaluate_verifier(model, dataloader, device: str):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(classification_report(
        all_labels,
        all_preds,
        target_names=[ID2LABEL[i] for i in range(4)]
    ))

    return acc, macro_f1


# ============================================================
# 5. Full Pipeline Evaluation: retrieve evidence, then classify
# ============================================================

def evaluate_full_pipeline(
    dev_claims: Dict,
    evidence_corpus: Dict[str, str],
    model_path: str,
    retrieval_top_k: int = 5,
    max_evidence: int = 5,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Building retriever...")
    retriever = BM25Retriever(evidence_corpus)

    print(f"Retrieval Recall@{retrieval_top_k}:")
    recall = retriever.evaluate_recall_at_k(dev_claims, k=retrieval_top_k)
    print(f"{recall:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)

    dev_dataset = ClaimEvidenceDataset(
        claims_json=dev_claims,
        evidence_corpus=evidence_corpus,
        tokenizer=tokenizer,
        max_length=max_length,
        max_evidence=max_evidence,
        use_gold_evidence=False,
        retriever=retriever,
        retrieval_top_k=retrieval_top_k,
        is_test=False
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    print("Evaluating full pipeline with retrieved evidence...")
    acc, macro_f1 = evaluate_verifier(
        model=model,
        dataloader=dev_loader,
        device=device
    )

    print(f"Full pipeline accuracy: {acc:.4f}")
    print(f"Full pipeline macro F1: {macro_f1:.4f}")


# ============================================================
# 6. Predict test labels and evidence IDs
# ============================================================

def predict_test(
    test_claims: Dict,
    evidence_corpus: Dict[str, str],
    model_path: str,
    output_path: str,
    retrieval_top_k: int = 5,
    max_evidence: int = 5,
    batch_size: int = 8,
    max_length: int = 512,
    device: str = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    retriever = BM25Retriever(evidence_corpus)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    test_dataset = ClaimEvidenceDataset(
        claims_json=test_claims,
        evidence_corpus=evidence_corpus,
        tokenizer=tokenizer,
        max_length=max_length,
        max_evidence=max_evidence,
        use_gold_evidence=False,
        retriever=retriever,
        retrieval_top_k=retrieval_top_k,
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    predictions = {}

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=1).cpu().numpy().tolist()

            for claim_id, pred_id, evidence_ids in zip(
                batch["claim_ids"],
                pred_ids,
                batch["evidence_ids"]
            ):
                predictions[claim_id] = {
                    "claim_label": ID2LABEL[pred_id],
                    "evidences": evidence_ids[:max_evidence]
                }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)

    print(f"Saved predictions to {output_path}")


# ============================================================
# 7. Main CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True,
                        choices=["train", "eval_pipeline", "predict"])

    parser.add_argument("--evidence_path", type=str, default="data/evidence.json")
    parser.add_argument("--train_path", type=str, default="data/train-claims.json")
    parser.add_argument("--dev_path", type=str, default="data/dev-claims.json")
    parser.add_argument("--test_path", type=str, default="data/test-claims-unlabelled.json")

    parser.add_argument("--model_name", type=str, default="distilroberta-base")
    parser.add_argument("--model_path", type=str, default="outputs/verifier_model")
    parser.add_argument("--output_path", type=str, default="outputs/test_predictions.json")

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_evidence", type=int, default=5)
    parser.add_argument("--retrieval_top_k", type=int, default=5)

    args = parser.parse_args()

    evidence_corpus = load_json(args.evidence_path)

    if args.mode == "train":
        train_claims = load_json(args.train_path)
        dev_claims = load_json(args.dev_path)

        train_verifier(
            train_claims=train_claims,
            dev_claims=dev_claims,
            evidence_corpus=evidence_corpus,
            model_name=args.model_name,
            output_dir=args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            max_length=args.max_length,
            max_evidence=args.max_evidence
        )

    elif args.mode == "eval_pipeline":
        dev_claims = load_json(args.dev_path)

        evaluate_full_pipeline(
            dev_claims=dev_claims,
            evidence_corpus=evidence_corpus,
            model_path=args.model_path,
            retrieval_top_k=args.retrieval_top_k,
            max_evidence=args.max_evidence,
            batch_size=args.batch_size,
            max_length=args.max_length
        )

    elif args.mode == "predict":
        test_claims = load_json(args.test_path)

        predict_test(
            test_claims=test_claims,
            evidence_corpus=evidence_corpus,
            model_path=args.model_path,
            output_path=args.output_path,
            retrieval_top_k=args.retrieval_top_k,
            max_evidence=args.max_evidence,
            batch_size=args.batch_size,
            max_length=args.max_length
        )


if __name__ == "__main__":
    main()