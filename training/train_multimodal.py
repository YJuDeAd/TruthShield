import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

# =====================
# CONFIG
# =====================

TRAIN_CSV = "data/processed/social_media/train.csv"
TEST_CSV = "data/processed/social_media/test.csv"

MODEL_DIR = "models/multimodal_model"
CHECKPOINT_DIR = f"{MODEL_DIR}/checkpoints"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 4               # safe for 4GB VRAM
ACCUMULATION_STEPS = 4       # effective batch size = 16
EPOCHS = 10
LR = 2e-5
MAX_LEN = 128                
IMAGE_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================
# HELPERS
# =====================

def resolve_image_path(raw_path: str, workspace_root: Path) -> str | None:
    candidate = Path(str(raw_path).replace("file:///", ""))
    if candidate.exists():
        return str(candidate)

    fixed = Path(str(candidate).replace("\\unnamed\\", "\\TruthShield\\"))
    if fixed.exists():
        return str(fixed)

    parts = list(candidate.parts)
    if "data" in parts:
        idx = parts.index("data")
        rejoined = workspace_root / Path(*parts[idx:])
        if rejoined.exists():
            return str(rejoined)

    return None


# =====================
# DATASET
# =====================


class MultimodalDataset(Dataset):

    def __init__(self, df):
        workspace_root = Path(__file__).resolve().parents[1]

        self.image_paths = []
        self.texts = []
        self.labels = []

        total_rows = len(df)
        for image_path, text, label in zip(
            df["image_path"].tolist(),
            df["text"].astype(str).tolist(),
            df["label"].tolist(),
        ):
            resolved = resolve_image_path(str(image_path), workspace_root)
            if resolved is None:
                continue

            self.image_paths.append(resolved)
            self.texts.append(text)
            self.labels.append(label)

        skipped = total_rows - len(self.labels)
        if skipped > 0:
            print(f"Skipped {skipped} rows due to missing image files.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # IMAGE
        path = self.image_paths[idx]

        image = Image.open(path).convert("RGB")
        image = image_transform(image)

        # TEXT
        encoding = tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# =====================
# MODEL
# =====================

class MultimodalModel(nn.Module):

    def __init__(self):

        super().__init__()

        # IMAGE MODEL (ResNet50)
        self.image_model = models.resnet50(weights="IMAGENET1K_V1")

        for param in self.image_model.parameters():
            param.requires_grad = False

        self.image_model.fc = nn.Linear(2048, 256)

        # TEXT MODEL (BERT)
        self.text_model = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        for param in self.text_model.parameters():
            param.requires_grad = False

        self.text_fc = nn.Linear(768, 256)

        # FUSION
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image, input_ids, attention_mask):

        image_features = self.image_model(image)

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)

        combined = torch.cat(
            (image_features, text_features),
            dim=1
        )

        output = self.classifier(combined)

        return output


# =====================
# MAIN
# =====================

def main() -> None:
    print("Using device:", DEVICE)

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # ------------------------------------------------------------------
    # 2. DATASETS
    # ------------------------------------------------------------------
    train_dataset = MultimodalDataset(train_df)
    test_dataset = MultimodalDataset(test_df)

    if len(train_dataset) == 0:
        raise RuntimeError("No valid training samples found after resolving image paths.")
    if len(test_dataset) == 0:
        raise RuntimeError("No valid test samples found after resolving image paths.")

    num_workers = 0 if os.name == "nt" else 2
    pin_memory = DEVICE.type == "cuda"

    # ------------------------------------------------------------------
    # 3. DATALOADERS
    # ------------------------------------------------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # ------------------------------------------------------------------
    # 4. MODEL, LOSS, OPTIMIZER
    # ------------------------------------------------------------------
    model = MultimodalModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")
    autocast = torch.amp.autocast

    best_accuracy = 0.0

    # ------------------------------------------------------------------
    # 5. TRAINING + EVALUATION LOOP
    # ------------------------------------------------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader)
        optimizer.zero_grad()

        for step, batch in enumerate(loop):
            image = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            with autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                outputs = model(image, input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        print("Epoch loss:", total_loss)

        model.eval()
        preds = []
        true = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                image = batch["image"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["label"].to(DEVICE)

                outputs = model(image, input_ids, attention_mask)
                predictions = torch.argmax(outputs, dim=1)

                preds.extend(predictions.cpu().numpy())
                true.extend(labels.cpu().numpy())

        acc = accuracy_score(true, preds)
        print("\nAccuracy:", acc)
        print(classification_report(true, preds))

        checkpoint_path = f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        if acc > best_accuracy:
            best_accuracy = acc
            torch.save(model.state_dict(), f"{MODEL_DIR}/best_model.pt")
            print("Best model saved")

    torch.save(model.state_dict(), f"{MODEL_DIR}/final_model.pt")
    print("Training complete")


if __name__ == "__main__":
    main()