import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os
import random
import json


# ======================
# CONFIG
# ======================

TRAIN_PATH = "data/processed/news/train.csv"
TEST_PATH  = "data/processed/news/test.csv"

MODEL_PATH       = "models/news_model/roberta_news.pt"
BEST_MODEL_PATH  = "models/news_model/roberta_news_best.pt"
CHECKPOINT_DIR   = "models/news_model/checkpoints/"

# ---- T4 GPU settings (16GB VRAM) ----
MODEL_NAME       = "roberta-large"   # Switch to "roberta-base" if Out of Memory
BATCH_SIZE       = 8                 # Reduce to 4 if Out of Memory, increase GRAD_ACCUM to 8
GRAD_ACCUM_STEPS = 4                 # Effective batch size = 32
MAX_LEN          = 512
LR               = 1e-5
WEIGHT_DECAY     = 0.01
LR_LAYER_DECAY   = 0.9               # Layer-wise learning rate decay

EPOCHS                   = 10
EARLY_STOPPING_PATIENCE  = 5
LABEL_SMOOTHING          = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================
# HELPERS
# ======================

def set_seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


def save_metadata(checkpoint_dir, epoch, best_val_loss, patience_counter, train_loss, val_loss, val_acc):
    metadata = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc
    }
    meta_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(checkpoint_dir, epoch):
    meta_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


# ======================
# DATASET
# ======================

class NewsDataset(Dataset):
    """
    Dynamic padding version — no fixed max_length padding here.
    Padding is handled per-batch by DataCollatorWithPadding,
    which avoids wasting attention on padding tokens for short texts.
    """
    def __init__(self, df, tokenizer):
        self.texts     = df["content"].astype(str).tolist()
        self.labels    = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
        )
        return {
            "input_ids":      encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels":         self.labels[idx],
        }


# ======================
# LAYER-WISE LR DECAY
# ======================

def get_optimizer_grouped_parameters(model, lr, weight_decay, lr_decay=0.9):
    no_decay = ["bias", "LayerNorm.weight"]
    layers = [model.roberta.embeddings] + list(model.roberta.encoder.layer)
    layers.reverse()

    grouped_params = []
    for i, layer in enumerate(layers):
        layer_lr = lr * (lr_decay ** i)
        grouped_params += [
            {
                "params": [p for n, p in layer.named_parameters()
                           if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            },
            {
                "params": [p for n, p in layer.named_parameters()
                           if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": layer_lr,
            },
        ]

    grouped_params += [
        {
            "params": [p for n, p in model.classifier.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": lr,
        },
        {
            "params": [p for n, p in model.classifier.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": lr,
        },
    ]
    return grouped_params


# ======================
# EVALUATION
# ======================

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ======================
# MAIN
# ======================

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    set_seed(42)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. LOAD & CLEAN DATA
    # ------------------------------------------------------------------
    print("\nLoading data...")
    train_full = pd.read_csv(TRAIN_PATH)
    test_df    = pd.read_csv(TEST_PATH)

    # Ensure correct types — large CSVs can sometimes load labels as floats
    train_full["label"] = train_full["label"].astype(int)
    test_df["label"]    = test_df["label"].astype(int)
    train_full["content"] = train_full["content"].astype(str)
    test_df["content"]    = test_df["content"].astype(str)

    # Drop any remaining NaN content rows
    train_full = train_full.dropna(subset=["content", "label"]).reset_index(drop=True)
    test_df    = test_df.dropna(subset=["content", "label"]).reset_index(drop=True)

    # Drop duplicates before splitting to prevent val leakage
    before = len(train_full)
    train_full = train_full.drop_duplicates(subset=["content"]).reset_index(drop=True)
    print(f"Removed {before - len(train_full)} duplicate rows from training data.")

    train_df, val_df = train_test_split(
        train_full,
        test_size=0.1,
        random_state=42,
        stratify=train_full["label"]
    )

    print(f"\nTraining set  : {len(train_df)}")
    print(f"Validation set: {len(val_df)}")
    print(f"Test set      : {len(test_df)}")
    print(f"\nTrain label distribution:\n{train_df['label'].value_counts().to_string()}")

    # ------------------------------------------------------------------
    # 2. TOKENIZER & DATASETS
    # ------------------------------------------------------------------
    print(f"\nLoading tokenizer: {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = NewsDataset(train_df, tokenizer)
    val_dataset   = NewsDataset(val_df,   tokenizer)
    test_dataset  = NewsDataset(test_df,  tokenizer)

    # ------------------------------------------------------------------
    # 3. DATALOADERS (dynamic padding via DataCollatorWithPadding)
    # ------------------------------------------------------------------
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collator, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # 4. MODEL SETUP
    # ------------------------------------------------------------------
    print(f"\nLoading model: {MODEL_NAME}...")
    model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(DEVICE)

    # ------------------------------------------------------------------
    # CHECKPOINT RESUME LOGIC (AUTO-RESUME FOR COLAB)
    # ------------------------------------------------------------------
    START_EPOCH      = 0
    best_val_loss    = float("inf")
    patience_counter = 0

    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR)
                       if f.startswith("epoch_") and f.endswith(".pt")]

        if checkpoints:
            epochs_saved    = [int(f.split("_")[1].split(".")[0]) for f in checkpoints]
            latest_epoch    = max(epochs_saved)
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{latest_epoch}.pt")

            print(f"\n{'='*60}")
            print(f"FOUND CHECKPOINT: epoch_{latest_epoch}.pt")
            print(f"{'='*60}")

            model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
            START_EPOCH = latest_epoch
            print(f"Model weights loaded from epoch {latest_epoch}")

            metadata = load_metadata(CHECKPOINT_DIR, latest_epoch)
            if metadata:
                best_val_loss    = metadata.get("best_val_loss", float("inf"))
                patience_counter = metadata.get("patience_counter", 0)
                print(f"Restored best_val_loss   : {best_val_loss:.4f}")
                print(f"Restored patience_counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            print(f"{'='*60}\n")
            print(f"Resuming training from epoch {START_EPOCH + 1}...\n")

    # ------------------------------------------------------------------
    # OPTIMIZER & LOSS
    # ------------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"]
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=LABEL_SMOOTHING
    )

    optimizer = AdamW(
        get_optimizer_grouped_parameters(model, LR, WEIGHT_DECAY, LR_LAYER_DECAY),
        lr=LR
    )

    num_training_steps = (EPOCHS - START_EPOCH) * len(train_loader) // GRAD_ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * num_training_steps),
        num_training_steps=num_training_steps
    )

    scaler = GradScaler(device="cuda")

    # ------------------------------------------------------------------
    # 5. TRAINING LOOP
    # ------------------------------------------------------------------
    print("Starting training...")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Total epochs        : {EPOCHS} (starting from epoch {START_EPOCH + 1})")
    print(f"Steps per epoch     : {len(train_loader)}")
    print(f"Gradient accum steps: {GRAD_ACCUM_STEPS}\n")

    for epoch in range(START_EPOCH, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(loop):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            with autocast(device_type="cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, labels)
                loss    = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS
            loop.set_postfix(loss=loss.item() * GRAD_ACCUM_STEPS)

        train_loss        = total_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, DEVICE)

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss  : {val_loss:.4f}")
        print(f"  Val Acc   : {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"{'='*60}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        save_metadata(CHECKPOINT_DIR, epoch+1, best_val_loss, patience_counter,
                      train_loss, val_loss, val_acc)
        print(f"Checkpoint saved: epoch_{epoch+1}.pt")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"NEW BEST MODEL saved! Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break

        print()

    # ------------------------------------------------------------------
    # 6. FINAL EVALUATION
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)

    print("\nLoading best model...")
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    test_loss, test_acc = evaluate(model, test_loader, loss_fn, DEVICE)
    print(f"\nTest Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Generating Classification Report"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds   = torch.argmax(outputs.logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(true_labels, predictions,
                                target_names=["Real (0)", "Fake (1)"]))

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nFinal model saved to {MODEL_PATH}")
    print(f"Best model saved to  {BEST_MODEL_PATH}")
    print("\nTraining complete!")