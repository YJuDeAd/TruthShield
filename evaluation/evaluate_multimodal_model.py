import argparse
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


# ----------------------------------------------------------
# DATASET WRAPPER
# ----------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_image_path(raw_path: str, workspace_root: Path) -> Path | None:
    candidate = Path(str(raw_path).replace("file:///", ""))
    if candidate.exists():
        return candidate

    fixed_text = str(candidate).replace("\\unnamed\\", "\\TruthShield\\")
    fixed = Path(fixed_text)
    if fixed.exists():
        return fixed

    tail_match = None
    parts = list(candidate.parts)
    if "data" in parts:
        idx = parts.index("data")
        tail_match = Path(*parts[idx:])

    if tail_match is not None:
        rejoined = workspace_root / tail_match
        if rejoined.exists():
            return rejoined

    return None


class MultimodalDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_len: int, image_size: int):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = Image.open(row["resolved_image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)

        encoding = self.tokenizer(
            str(row["text"]),
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "image": image_tensor,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = models.resnet50(weights=None)
        self.image_model.fc = nn.Linear(2048, 256)

        self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc = nn.Linear(768, 256)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_features = self.image_model(image)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)
        combined = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)


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
    ConfusionMatrixDisplay(cm, display_labels=["Real (0)", "Fake (1)"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Multimodal Model")
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
    ax.set_title("ROC Curve - Multimodal Model")
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
    ax.set_title("Precision-Recall Curve - Multimodal Model")
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
    ax.set_title("Core Metrics - Multimodal Model")
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
    ax.set_title("Threshold vs Precision/Recall/F1 - Multimodal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "threshold_vs_metrics.png", dpi=200)
    plt.close(fig)

    # 6) Class distribution in test set.
    fig, ax = plt.subplots(figsize=(6, 4))
    classes, counts = np.unique(y_true, return_counts=True)
    class_names = ["Real (0)" if x == 0 else "Fake (1)" for x in classes]
    bars = ax.bar(class_names, counts)
    ax.set_title("Class Distribution (Test Set) - Multimodal")
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
    workspace_root = Path(args.workspace_root).resolve()

    print(f"Using device: {device}")
    print(f"Loading test data from: {args.test_csv}")

    test_df = pd.read_csv(args.test_csv)
    # Ensure required columns are valid before inference.
    test_df = test_df.dropna(subset=["image_path", "text", "label"]).copy()
    test_df["text"] = test_df["text"].astype(str)
    test_df["label"] = test_df["label"].astype(int)

    resolved_paths = []
    keep_rows = []

    for idx, row in test_df.iterrows():
        fixed = resolve_image_path(str(row["image_path"]), workspace_root)
        if fixed is not None:
            keep_rows.append(idx)
            resolved_paths.append(str(fixed))

    filtered_df = test_df.loc[keep_rows].copy().reset_index(drop=True)
    filtered_df["resolved_image_path"] = resolved_paths

    skipped = len(test_df) - len(filtered_df)
    if skipped > 0:
        print(f"Skipped {skipped} rows due to missing image files.")

    # Build dataset + dataloader after path resolution.
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = MultimodalDataset(filtered_df, tokenizer, args.max_len, args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load model architecture, then trained checkpoint weights.
    model = MultimodalModel()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    y_true, y_scores, y_pred = [], [], []

    # Batched inference.
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating Multimodal", unit="batch"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(image, input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1]
            # Apply user-provided decision threshold.
            preds = (probs >= args.threshold).long()

            y_true.extend(labels.cpu().numpy().tolist())
            y_scores.extend(probs.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    y_true_np = np.array(y_true, dtype=int)
    y_scores_np = np.array(y_scores, dtype=float)
    y_pred_np = np.array(y_pred, dtype=int)

    # Generate plots + summary report artifacts.
    metrics = save_plots(y_true_np, y_scores_np, y_pred_np, args.output_dir)
    report = classification_report(y_true_np, y_pred_np, target_names=["Real (0)", "Fake (1)"], zero_division=0)

    result = {
        "threshold": args.threshold,
        "test_size_after_image_filter": int(len(y_true_np)),
        "skipped_missing_images": int(skipped),
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
    parser = argparse.ArgumentParser(description="Evaluate multimodal model with metrics and plots.")
    parser.add_argument("--test-csv", default="data/processed/social_media/test.csv")
    parser.add_argument("--model-path", default="models/multimodal_model/best_model.pt")
    parser.add_argument("--output-dir", default="evaluation/multimodal")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.7)
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())