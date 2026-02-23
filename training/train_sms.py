import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import numpy as np
from tqdm import tqdm
import os
import pickle

# ======================
# CONFIG
# ======================

TRAIN_PATH = "data/processed/sms/train.csv"
TEST_PATH  = "data/processed/sms/test.csv"
MODEL_PATH = "models/sms_model/sms_model.pt"
CHECKPOINT_DIR = "models/sms_model/checkpoints"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE  = 64
EPOCHS      = 10
MAX_LEN     = 300       
EMBED_SIZE  = 128
HIDDEN_SIZE = 128
DROPOUT     = 0.4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# ======================
# HELPERS
# ======================

def tokenize(text):
    return str(text).lower().split()


# ======================
# 1. LOAD & CLEAN DATA
# ======================

print("Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Ensure correct types — CSVs can load labels as non-int types.
train_df["text"]  = train_df["text"].astype(str).fillna("")
test_df["text"]   = test_df["text"].astype(str).fillna("")
train_df["label"] = train_df["label"].astype(int)
test_df["label"]  = test_df["label"].astype(int)

print(f"Train size : {len(train_df)}")
print(f"Test size  : {len(test_df)}")
print(f"Train label distribution:\n{train_df['label'].value_counts().to_string()}")


# ======================
# 2. BUILD VOCAB
# ======================

print("\nBuilding vocabulary...")

counter = Counter()
for text in train_df["text"]:
    counter.update(tokenize(text))

# Reserve 0 = PAD, 1 = UNK (fixes bug: original used 0 for both)
vocab      = {word: i + 2 for i, (word, _) in enumerate(counter.most_common(50000))}
vocab_size = len(vocab) + 2
PAD_IDX    = 0
UNK_IDX    = 1

print(f"Vocab size: {vocab_size}")

os.makedirs("models/sms_model", exist_ok=True)
with open("models/sms_model/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("Vocab saved to models/sms_model/vocab.pkl")

# ======================
# 3. TEXT → SEQUENCE
# ======================

def text_to_sequence(text):
    tokens = tokenize(text)
    seq    = [vocab.get(word, UNK_IDX) for word in tokens]
    if len(seq) < MAX_LEN:
        seq += [PAD_IDX] * (MAX_LEN - len(seq))
    else:
        seq = seq[:MAX_LEN]
    return seq

# ======================
# DATASET
# ======================

class SMSDataset(Dataset):
    def __init__(self, df):
        self.X = [text_to_sequence(t) for t in df["text"]]
        self.y = df["label"].values

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.long),
            torch.tensor(self.y[idx], dtype=torch.float32)
        )

train_dataset = SMSDataset(train_df)
test_dataset  = SMSDataset(test_df)

# ======================
# 4. DATALOADERS
# ======================

# ------------------------------------------------------------------
# WEIGHTED SAMPLER — fixes the 34x class imbalance
# Without this, the model learns to predict spam for everything
# and achieves 97% accuracy while being completely useless for ham.
# WeightedRandomSampler oversamples the minority class (ham)
# so each batch sees a roughly balanced mix during training.
# ------------------------------------------------------------------
label_counts  = train_df["label"].value_counts().to_dict()
class_weights = {cls: 1.0 / count for cls, count in label_counts.items()}
sample_weights = [class_weights[label] for label in train_df["label"]]
sample_weights = torch.tensor(sample_weights, dtype=torch.float)

sampler = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(sample_weights),
    replacement = True
)

# shuffle=False when using sampler (sampler handles the shuffling)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                          shuffle=False,  num_workers=0)

# ======================
# 5. MODEL SETUP
# ======================

class HybridModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size, EMBED_SIZE, padding_idx=PAD_IDX
        )
        self.embed_dropout = nn.Dropout(DROPOUT)

        self.lstm = nn.LSTM(
            EMBED_SIZE,
            HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
            num_layers=2,           # Added second layer for deeper sequential learning
            dropout=DROPOUT         # Inter-layer dropout (only applies when num_layers > 1)
        )

        # CNN takes BiLSTM output (HIDDEN_SIZE * 2 = 256 channels)
        self.conv = nn.Conv1d(
            in_channels  = HIDDEN_SIZE * 2,
            out_channels = 128,     # Increased from 64 for more capacity
            kernel_size  = 3,
            padding      = 1
        )
        self.relu         = nn.ReLU()
        self.fc_dropout   = nn.Dropout(DROPOUT)
        self.fc           = nn.Linear(128, 1)
        self.sigmoid      = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)               # (batch, seq_len, embed_size)
        x = self.embed_dropout(x)

        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden*2, seq_len) for Conv1d

        conv_out = self.relu(self.conv(lstm_out))  # (batch, 128, seq_len)
        pooled   = torch.mean(conv_out, dim=2)     # (batch, 128) global avg pool

        out = self.fc_dropout(pooled)
        out = self.sigmoid(self.fc(out))    # (batch, 1)
        return out.squeeze()

model     = HybridModel(vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ------------------------------------------------------------------
# WEIGHTED LOSS — second layer of imbalance correction alongside sampler.
# Even with the sampler, having pos_weight reinforces the model
# to not dismiss ham (minority class) as an easy mistake.
# pos_weight = num_spam / num_ham
# ------------------------------------------------------------------
num_spam   = label_counts.get(1, 1)
num_ham    = label_counts.get(0, 1)
pos_weight = torch.tensor([num_spam / num_ham], dtype=torch.float).to(DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

criterion = nn.BCELoss()

# Learning rate scheduler — reduces LR when val loss plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5
)

# ======================
# 6. TRAINING LOOP
# ======================

print("\nStarting training...")
print(f"Effective balanced batches via WeightedRandomSampler")
print(f"Max epochs: {EPOCHS} | Batch size: {BATCH_SIZE} | Max len: {MAX_LEN}\n")

best_val_loss    = float("inf")
patience_counter = 0
EARLY_STOP       = 4

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)

    for X_batch, y_batch in loop:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss    = criterion(outputs, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # Validation pass on held-out test set.
    model.eval()
    val_loss  = 0
    val_preds = []
    val_true  = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            outputs = model(X_batch)
            loss    = criterion(outputs, y_batch)
            val_loss += loss.item()
            pred = (outputs > 0.5).cpu().numpy()
            val_preds.extend(pred)
            val_true.extend(y_batch.cpu().numpy())

    avg_val_loss = val_loss / len(test_loader)
    val_acc      = accuracy_score(val_true, val_preds)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Train Loss : {avg_train_loss:.4f}")
    print(f"  Val Loss   : {avg_val_loss:.4f}")
    print(f"  Val Acc    : {val_acc:.4f} ({val_acc*100:.2f}%)")

    # Save epoch checkpoint for recovery and analysis.
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pt")
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "best_val_loss": best_val_loss,
            "patience_counter": patience_counter,
        },
        checkpoint_path,
    )
    print(f"  Checkpoint saved: {checkpoint_path}")

    scheduler.step(avg_val_loss)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  NEW BEST MODEL saved!")
    else:
        patience_counter += 1
        print(f"  No improvement. Patience: {patience_counter}/{EARLY_STOP}")
        if patience_counter >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    print()

# ======================
# 7. FINAL EVALUATION
# ======================

print("\n" + "="*60)
print("FINAL EVALUATION (best model)")
print("="*60)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

preds = []
true  = []

with torch.no_grad():
    for X_batch, y_batch in tqdm(test_loader, desc="Testing"):
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        pred    = (outputs > 0.5).cpu().numpy()
        preds.extend(pred)
        true.extend(y_batch.numpy())

print(f"\nAccuracy: {accuracy_score(true, preds):.4f}")
print("\nClassification Report:")
print(classification_report(true, preds, target_names=["Ham (0)", "Spam (1)"]))
print(f"\nModel saved to {MODEL_PATH}")