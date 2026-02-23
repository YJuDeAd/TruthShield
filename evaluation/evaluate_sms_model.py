import argparse
import json
import os
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    average_precision_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ----------------------------------------------------------
# DATASET WRAPPER
# ----------------------------------------------------------

def tokenize(text: str):
    return str(text).lower().split()


class SMSDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, vocab: dict, max_len: int) -> None:
        self.vocab = vocab
        self.max_len = max_len
        self.texts = dataframe["text"].astype(str).tolist()
        self.labels = dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def _text_to_sequence(self, text: str):
        pad_idx = 0
        unk_idx = 1
        seq = [self.vocab.get(tok, unk_idx) for tok in tokenize(text)]
        if len(seq) < self.max_len:
            seq.extend([pad_idx] * (self.max_len - len(seq)))
        else:
            seq = seq[: self.max_len]
        return seq

    def __getitem__(self, idx: int):
        seq = self._text_to_sequence(self.texts[idx])
        label = self.labels[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(label, dtype=torch.long)


class HybridModel(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        self.conv = nn.Conv1d(in_channels=hidden_size * 2, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.embed_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.relu(self.conv(lstm_out))
        pooled = torch.mean(conv_out, dim=2)
        out = self.fc_dropout(pooled)
        out = self.sigmoid(self.fc(out))
        return out.squeeze(-1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------

def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray):
    # Evaluate precision/recall/F1 behavior across decision thresholds.
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls, f1_scores = [], [], []
    for threshold in thresholds:
        preds = (y_scores >= threshold).astype(int)
        precisions.append(precision_score(y_true, preds, zero_division=0))
        recalls.append(recall_score(y_true, preds, zero_division=0))
        f1_scores.append(f1_score(y_true, preds, zero_division=0))
    return thresholds, np.array(precisions), np.array(recalls), np.array(f1_scores)


def save_plots(y_true: np.ndarray, y_scores: np.ndarray, y_pred: np.ndarray, out_dir: str) -> dict:
    # Core point-metrics based on the chosen threshold.
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
    except ValueError:
        # Happens when y_true contains only one class.
        metrics["roc_auc"] = None

    try:
        metrics["average_precision"] = float(average_precision_score(y_true, y_scores))
    except ValueError:
        # Happens when PR computation is undefined for the label distribution.
        metrics["average_precision"] = None

    # 1) Confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Ham (0)", "Spam (1)"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - SMS Model")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # 2) ROC curve.
    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_val = metrics["roc_auc"] if metrics["roc_auc"] is not None else 0.0
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.legend()
    except ValueError:
        ax.text(0.2, 0.5, "ROC unavailable (single class in y_true)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - SMS Model")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "roc_curve.png", dpi=200)
    plt.close(fig)

    # 3) Precision-Recall curve.
    fig, ax = plt.subplots(figsize=(6, 5))
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_scores)
    ap_val = metrics["average_precision"] if metrics["average_precision"] is not None else 0.0
    ax.plot(recall_vals, precision_vals, label=f"AP = {ap_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - SMS Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "pr_curve.png", dpi=200)
    plt.close(fig)

    # 4) Core-metrics bar chart.
    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["Accuracy", "Precision", "Recall", "F1"]
    values = [metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]]
    bars = ax.bar(names, values)
    ax.set_ylim(0, 1)
    ax.set_title("Core Metrics - SMS Model")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "metric_bar_chart.png", dpi=200)
    plt.close(fig)

    # 5) Threshold sensitivity chart.
    thresholds, p_curve, r_curve, f1_curve = compute_threshold_metrics(y_true, y_scores)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(thresholds, p_curve, label="Precision")
    ax.plot(thresholds, r_curve, label="Recall")
    ax.plot(thresholds, f1_curve, label="F1")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold vs Precision/Recall/F1 - SMS Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "threshold_vs_metrics.png", dpi=200)
    plt.close(fig)

    # 6) Class distribution in test set.
    fig, ax = plt.subplots(figsize=(6, 4))
    classes, counts = np.unique(y_true, return_counts=True)
    class_names = ["Ham (0)" if x == 0 else "Spam (1)" for x in classes]
    bars = ax.bar(class_names, counts)
    ax.set_title("Class Distribution (Test Set) - SMS")
    for bar, value in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(int(value)), ha="center")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "class_distribution.png", dpi=200)
    plt.close(fig)

    return metrics


# ----------------------------------------------------------
# EVALUATION PIPELINE
# ----------------------------------------------------------

def run_evaluation(args: argparse.Namespace) -> None:
    # Select GPU if available, otherwise CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print(f"Loading test data from: {args.test_csv}")

    test_df = pd.read_csv(args.test_csv)
    # Ensure required columns are valid before inference.
    test_df = test_df.dropna(subset=["text", "label"]).copy()
    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    # Load vocabulary + model architecture, then load trained weights.
    with open(args.vocab_path, "rb") as fp:
        vocab = pickle.load(fp)

    vocab_size = len(vocab) + 2
    model = HybridModel(vocab_size, args.embed_size, args.hidden_size, args.dropout)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    dataset = SMSDataset(test_df, vocab, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    y_true, y_scores, y_pred = [], [], []

    # Batched inference.
    with torch.no_grad():
        for x_batch, y_batch in tqdm(loader, desc="Evaluating SMS", unit="batch"):
            x_batch = x_batch.to(device)
            probs = model(x_batch)
            # Apply user-provided decision threshold.
            preds = (probs >= args.threshold).long()

            y_true.extend(y_batch.numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)
    y_pred_np = np.array(y_pred, dtype=int)

    # Generate plots + summary report artifacts.
    metrics = save_plots(y_true_np, y_scores_np, y_pred_np, args.output_dir)
    report = classification_report(y_true_np, y_pred_np, target_names=["Ham (0)", "Spam (1)"], zero_division=0)

    result = {
        "threshold": args.threshold,
        "test_size": int(len(y_true_np)),
        "metrics": metrics,
        "classification_report": report,
    }

    with open(Path(args.output_dir) / "metrics_summary.json", "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)

    with open(Path(args.output_dir) / "classification_report.txt", "w", encoding="utf-8") as fp:
        fp.write(report)

    print("\nEvaluation complete.")
    print(json.dumps(result["metrics"], indent=2))
    print(f"Saved outputs to: {args.output_dir}")


# ----------------------------------------------------------
# CLI ARGUMENTS
# ----------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SMS model with metrics and plots.")
    parser.add_argument("--test-csv", default="data/processed/sms/test.csv")
    parser.add_argument("--model-path", default="models/sms_model/sms_model.pt")
    parser.add_argument("--vocab-path", default="models/sms_model/vocab.pkl")
    parser.add_argument("--output-dir", default="evaluation/sms")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=300)
    parser.add_argument("--embed-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())