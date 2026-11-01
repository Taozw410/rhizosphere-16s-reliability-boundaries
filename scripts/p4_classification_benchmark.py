#!/usr/bin/env python3
"""SILVA-based taxonomy classifier benchmark for Paper 4."""

from __future__ import annotations

import argparse
import gzip
import itertools
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference",
        default="ref_region/silva_V4_515F_806R.fa.gz",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/classification_benchmark",
    )
    parser.add_argument("--genera", type=int, default=24)
    parser.add_argument("--sequences-per-genus", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sequence-length", type=int, default=200)
    return parser.parse_args()


def mutate_sequence(
    sequence: str,
    rng: np.random.Generator,
    substitution_rate: float,
) -> str:
    alphabet = np.array(list("ACGT"))
    sequence = list(sequence)
    mutate = rng.random(len(sequence)) < substitution_rate
    for position in np.flatnonzero(mutate):
        choices = alphabet[alphabet != sequence[position]]
        sequence[position] = str(rng.choice(choices))
    return "".join(sequence)


def one_hot(sequence: str, length: int) -> np.ndarray:
    sequence = (sequence + "N" * length)[:length]
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    output = np.zeros((4, len(sequence)), dtype=np.float32)
    for index, base in enumerate(sequence):
        if base in mapping:
            output[mapping[base], index] = 1.0
    return output


def kmer_tokens(sequence: str, length: int, k: int = 3) -> list[int]:
    mapping = {
        "".join(token): index
        for index, token in enumerate(
            itertools.product("ACGT", repeat=k)
        )
    }
    pad = len(mapping)
    sequence = (sequence + "N" * length)[:length]
    return [
        mapping.get(sequence[index : index + k], pad)
        for index in range(len(sequence) - k + 1)
    ]


class CNNClassifier(nn.Module):
    def __init__(self, n_classes: int, length: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).squeeze(-1))


class KmerTransformer(nn.Module):
    def __init__(self, n_classes: int, vocabulary: int = 65) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 32, padding_idx=64)
        layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.classifier = nn.Linear(32, n_classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.eq(64)
        output = self.encoder(
            self.embedding(tokens),
            src_key_padding_mask=mask,
        )
        mask_values = (~mask).unsqueeze(-1).float()
        pooled = (output * mask_values).sum(dim=1) / mask_values.sum(
            dim=1
        ).clamp_min(1)
        return self.classifier(pooled)


def train_neural(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_validation: torch.Tensor,
    y_validation: torch.Tensor,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=128,
        shuffle=True,
    )
    best_state = None
    best_loss = float("inf")
    for _ in range(4):
        model.train()
        for features, labels in loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), labels)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(x_validation),
                y_validation,
            ).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def load_silva(
    path: Path,
    genera: int,
    sequences_per_genus: int,
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    current = None
    chunks = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if current and len("".join(chunks)) >= 300:
                    grouped[current].append(re.sub("[^ACGT]", "", "".join(chunks)))
                parts = line[1:].split(";")
                genus = parts[5].strip() if len(parts) > 5 else ""
                current = (
                    genus
                    if genus
                    and genus.lower()
                    not in {"uncultured", "uncultured bacterium"}
                    else None
                )
                chunks = []
            elif current:
                chunks.append(line)
    if current and len("".join(chunks)) >= 300:
        grouped[current].append(re.sub("[^ACGT]", "", "".join(chunks)))
    eligible = sorted(
        (
            (genus, sequences)
            for genus, sequences in grouped.items()
            if len(sequences) >= sequences_per_genus
        ),
        key=lambda item: (-len(item[1]), item[0]),
    )[:genera]
    return {
        genus: sequences[:sequences_per_genus]
        for genus, sequences in eligible
    }


def predict_probability(model, features):
    return model.predict_proba(features)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(2)
    random.seed(42)
    np.random.seed(42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sequences = load_silva(
        Path(args.reference),
        args.genera,
        args.sequences_per_genus,
    )
    genera = sorted(sequences)
    if len(genera) < args.genera:
        raise RuntimeError("Not enough eligible genera")
    rows = []
    for seed in range(6100, 6100 + args.seeds):
        rng = np.random.default_rng(seed)
        shuffled = genera.copy()
        rng.shuffle(shuffled)
        unseen_genera = set(shuffled[:4])
        known_genera = [genus for genus in genera if genus not in unseen_genera]
        train_sequences = []
        validation_sequences = []
        test_sequences = []
        train_labels = []
        validation_labels = []
        test_labels = []
        unseen_sequences = []
        for genus in genera:
            genus_sequences = sequences[genus].copy()
            rng.shuffle(genus_sequences)
            if genus in unseen_genera:
                unseen_sequences.extend(genus_sequences)
                continue
            train_count = 6
            for sequence in genus_sequences[:train_count]:
                for _ in range(3):
                    train_sequences.append(
                        mutate_sequence(
                            sequence[: args.sequence_length],
                            rng,
                            0.003,
                        )
                    )
                    train_labels.append(genus)
            for sequence in genus_sequences[train_count : train_count + 2]:
                validation_sequences.append(
                    mutate_sequence(
                        sequence[: args.sequence_length],
                        rng,
                        0.005,
                    )
                )
                validation_labels.append(genus)
            for sequence in genus_sequences[train_count + 2 :]:
                test_sequences.append(
                    mutate_sequence(
                        sequence[: args.sequence_length],
                        rng,
                        0.005,
                    )
                )
                test_labels.append(genus)
        unseen_sequences = [
            mutate_sequence(sequence[: args.sequence_length], rng, 0.005)
            for sequence in unseen_sequences
        ]

        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(4, 5),
            max_features=2000,
            lowercase=False,
        )
        x_train_text = vectorizer.fit_transform(train_sequences)
        x_validation_text = vectorizer.transform(validation_sequences)
        x_test_text = vectorizer.transform(test_sequences)
        x_unseen_text = vectorizer.transform(unseen_sequences)

        classifiers = {
            "kmer_knn": KNeighborsClassifier(
                n_neighbors=5,
                metric="cosine",
                weights="distance",
            ),
            "multinomial_nb": MultinomialNB(alpha=0.5),
            "linear_sgd": SGDClassifier(
                loss="log_loss",
                alpha=1e-4,
                max_iter=2000,
                random_state=seed,
            ),
        }
        for method_id, classifier in classifiers.items():
            classifier.fit(x_train_text, train_labels)
            validation_probability = predict_probability(
                classifier,
                x_validation_text,
            )
            threshold = float(
                np.quantile(validation_probability.max(axis=1), 0.05)
            )
            known_retention = float(
                np.mean(validation_probability.max(axis=1) >= threshold)
            )
            for scenario, x, labels, unseen in [
                ("closed_set", x_test_text, test_labels, False),
                (
                    "unseen_genus",
                    x_unseen_text,
                    ["UNKNOWN"] * x_unseen_text.shape[0],
                    True,
                ),
            ]:
                probability = predict_probability(classifier, x)
                confidence = probability.max(axis=1)
                predictions = classifier.predict(x)
                if unseen:
                    predictions = np.where(
                        confidence >= threshold,
                        predictions,
                        "UNKNOWN",
                    )
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario,
                        "method_id": method_id,
                        "accuracy": accuracy_score(labels, predictions),
                        "macro_f1": f1_score(
                            labels,
                            predictions,
                            average="macro",
                            zero_division=0,
                        ),
                        "unknown_recall": (
                            np.mean(np.asarray(predictions) == "UNKNOWN")
                            if unseen
                            else np.nan
                        ),
                        "known_retention": known_retention if unseen else 1.0,
                        "threshold": threshold,
                    }
                )

        x_train_onehot = torch.tensor(
            np.stack(
                [
                    one_hot(sequence, args.sequence_length)
                    for sequence in train_sequences
                ]
            )
        )
        x_validation_onehot = torch.tensor(
            np.stack(
                [
                    one_hot(sequence, args.sequence_length)
                    for sequence in validation_sequences
                ]
            )
        )
        x_test_onehot = torch.tensor(
            np.stack(
                [
                    one_hot(sequence, args.sequence_length)
                    for sequence in test_sequences
                ]
            )
        )
        x_unseen_onehot = torch.tensor(
            np.stack(
                [
                    one_hot(sequence, args.sequence_length)
                    for sequence in unseen_sequences
                ]
            )
        )
        label_map = {label: index for index, label in enumerate(known_genera)}
        y_train = torch.tensor([label_map[label] for label in train_labels])
        y_validation = torch.tensor(
            [label_map[label] for label in validation_labels]
        )
        cnn = train_neural(
            CNNClassifier(len(known_genera), args.sequence_length),
            x_train_onehot,
            y_train,
            x_validation_onehot,
            y_validation,
            seed,
        )

        token_train = torch.tensor(
            [
                kmer_tokens(sequence, args.sequence_length)
                for sequence in train_sequences
            ]
        )
        token_validation = torch.tensor(
            [
                kmer_tokens(sequence, args.sequence_length)
                for sequence in validation_sequences
            ]
        )
        token_test = torch.tensor(
            [
                kmer_tokens(sequence, args.sequence_length)
                for sequence in test_sequences
            ]
        )
        token_unseen = torch.tensor(
            [
                kmer_tokens(sequence, args.sequence_length)
                for sequence in unseen_sequences
            ]
        )
        transformer = train_neural(
            KmerTransformer(len(known_genera)),
            token_train,
            y_train,
            token_validation,
            y_validation,
            seed + 1,
        )

        for method_id, model, validation_x, test_x, unseen_x in [
            (
                "cnn",
                cnn,
                x_validation_onehot,
                x_test_onehot,
                x_unseen_onehot,
            ),
            (
                "kmer_transformer",
                transformer,
                token_validation,
                token_test,
                token_unseen,
            ),
        ]:
            model.eval()
            with torch.no_grad():
                validation_probability = torch.softmax(
                    model(validation_x),
                    dim=1,
                ).numpy()
                threshold = float(
                    np.quantile(
                        validation_probability.max(axis=1),
                        0.05,
                    )
                )
                known_retention = float(
                    np.mean(
                        validation_probability.max(axis=1) >= threshold
                    )
                )
                test_probability = torch.softmax(model(test_x), dim=1).numpy()
                unseen_probability = torch.softmax(
                    model(unseen_x),
                    dim=1,
                ).numpy()
            test_predictions = [
                known_genera[index]
                for index in test_probability.argmax(axis=1)
            ]
            rows.append(
                {
                    "seed": seed,
                    "scenario": "closed_set",
                    "method_id": method_id,
                    "accuracy": accuracy_score(test_labels, test_predictions),
                    "macro_f1": f1_score(
                        test_labels,
                        test_predictions,
                        average="macro",
                        zero_division=0,
                    ),
                    "unknown_recall": np.nan,
                    "known_retention": 1.0,
                    "threshold": threshold,
                }
            )
            unknown_predictions = np.where(
                unseen_probability.max(axis=1) >= threshold,
                "PREDICTED",
                "UNKNOWN",
            )
            rows.append(
                {
                    "seed": seed,
                    "scenario": "unseen_genus",
                    "method_id": method_id,
                    "accuracy": np.mean(unknown_predictions == "UNKNOWN"),
                    "macro_f1": np.nan,
                    "unknown_recall": np.mean(
                        unknown_predictions == "UNKNOWN"
                    ),
                    "known_retention": known_retention,
                    "threshold": threshold,
                }
            )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "classification_metrics_by_seed.csv", index=False)
    summary = (
        metrics.groupby(["scenario", "method_id"])
        .agg(
            seeds=("seed", "nunique"),
            accuracy_mean=("accuracy", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            unknown_recall_mean=("unknown_recall", "mean"),
            known_retention_mean=("known_retention", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "classification_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
