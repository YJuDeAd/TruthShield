import pickle
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from config import (
    MULTIMODAL_IMAGE_SIZE,
    MULTIMODAL_MAX_LEN,
    MULTIMODAL_MODEL_PATH,
    NEWS_MODEL_NAME,
    NEWS_MODEL_PATH,
    SMS_DROPOUT,
    SMS_EMBED_SIZE,
    SMS_HIDDEN_SIZE,
    SMS_MAX_LEN,
    SMS_MODEL_PATH,
    SMS_VOCAB_PATH,
)


# ========== SMS Model Definition ==========
class HybridModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, SMS_EMBED_SIZE, padding_idx=0)
        self.embed_dropout = nn.Dropout(SMS_DROPOUT)
        self.lstm = nn.LSTM(
            SMS_EMBED_SIZE,
            SMS_HIDDEN_SIZE,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=SMS_DROPOUT,
        )
        self.conv = nn.Conv1d(
            in_channels=SMS_HIDDEN_SIZE * 2,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.fc_dropout = nn.Dropout(SMS_DROPOUT)
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


# ========== Multimodal Model Definition ==========
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

    def forward(
        self, image: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        image_features = self.image_model(image)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)
        combined = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)


# ========== Model Manager ==========
class ModelManager:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.news_model = None
        self.news_tokenizer = None
        self.sms_model = None
        self.sms_vocab = None
        self.multimodal_model = None
        self.multimodal_tokenizer = None
        self.multimodal_transform = None
        self.models_loaded = {"news": False, "sms": False, "multimodal": False}

    def load_news_model(self):
        if not NEWS_MODEL_PATH.exists():
            print(f"Warning: News model not found at {NEWS_MODEL_PATH}")
            return
        
        try:
            self.news_tokenizer = RobertaTokenizer.from_pretrained(NEWS_MODEL_NAME)
            self.news_model = RobertaForSequenceClassification.from_pretrained(
                NEWS_MODEL_NAME, num_labels=2
            )
            self.news_model.load_state_dict(
                torch.load(NEWS_MODEL_PATH, map_location=self.device)
            )
            self.news_model.to(self.device)
            self.news_model.eval()
            self.models_loaded["news"] = True
            print("✓ News model loaded")
        except Exception as e:
            print(f"✗ Failed to load news model: {e}")

    def load_sms_model(self):
        if not SMS_MODEL_PATH.exists() or not SMS_VOCAB_PATH.exists():
            print(f"Warning: SMS model files not found")
            return
        
        try:
            with open(SMS_VOCAB_PATH, "rb") as f:
                self.sms_vocab = pickle.load(f)
            
            vocab_size = len(self.sms_vocab) + 2
            self.sms_model = HybridModel(vocab_size)
            self.sms_model.load_state_dict(
                torch.load(SMS_MODEL_PATH, map_location=self.device)
            )
            self.sms_model.to(self.device)
            self.sms_model.eval()
            self.models_loaded["sms"] = True
            print("✓ SMS model loaded")
        except Exception as e:
            print(f"✗ Failed to load SMS model: {e}")

    def load_multimodal_model(self):
        if not MULTIMODAL_MODEL_PATH.exists():
            print(f"Warning: Multimodal model not found at {MULTIMODAL_MODEL_PATH}")
            return
        
        try:
            self.multimodal_tokenizer = BertTokenizer.from_pretrained(
                "bert-base-uncased"
            )
            self.multimodal_model = MultimodalModel()
            self.multimodal_model.load_state_dict(
                torch.load(MULTIMODAL_MODEL_PATH, map_location=self.device)
            )
            self.multimodal_model.to(self.device)
            self.multimodal_model.eval()
            
            self.multimodal_transform = transforms.Compose([
                transforms.Resize((MULTIMODAL_IMAGE_SIZE, MULTIMODAL_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            self.models_loaded["multimodal"] = True
            print("✓ Multimodal model loaded")
        except Exception as e:
            print(f"✗ Failed to load multimodal model: {e}")

    def load_all_models(self):
        print("Loading models...")
        self.load_news_model()
        self.load_sms_model()
        self.load_multimodal_model()
        print(f"Models ready: {self.models_loaded}")

    def predict_news(self, text: str, threshold: float = 0.7) -> dict:
        if not self.models_loaded["news"]:
            raise ValueError("News model not loaded")
        
        start = time.time()
        
        encoding = self.news_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            outputs = self.news_model(**encoding)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            prob_real = probs[0].item()
            prob_fake = probs[1].item()
            verdict = "Fake" if prob_fake >= threshold else "Real"
            confidence = max(prob_real, prob_fake)
        
        processing_time_ms = (time.time() - start) * 1000
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "probabilities": {"Real": prob_real, "Fake": prob_fake},
            "processing_time_ms": processing_time_ms,
        }

    def predict_sms(self, text: str, threshold: float = 0.7) -> dict:
        if not self.models_loaded["sms"]:
            raise ValueError("SMS model not loaded")
        
        start = time.time()
        
        # Preprocess text
        tokens = text.lower().split()
        seq = [self.sms_vocab.get(tok, 1) for tok in tokens]  # 1 = UNK
        if len(seq) < SMS_MAX_LEN:
            seq.extend([0] * (SMS_MAX_LEN - len(seq)))  # 0 = PAD
        else:
            seq = seq[:SMS_MAX_LEN]
        
        tensor = torch.tensor([seq], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            prob_spam = self.sms_model(tensor).item()
            prob_ham = 1.0 - prob_spam
            verdict = "Fake" if prob_spam >= threshold else "Real"
            confidence = max(prob_ham, prob_spam)
        
        processing_time_ms = (time.time() - start) * 1000
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "probabilities": {"Real": prob_ham, "Fake": prob_spam},
            "processing_time_ms": processing_time_ms,
        }

    def predict_multimodal(
        self,
        text: str,
        image: Image.Image,
        threshold: float = 0.7
    ) -> dict:
        if not self.models_loaded["multimodal"]:
            raise ValueError("Multimodal model not loaded")
        
        start = time.time()
        
        # Preprocess image
        image_tensor = self.multimodal_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        
        # Preprocess text
        encoding = self.multimodal_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MULTIMODAL_MAX_LEN,
        )
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        
        with torch.no_grad():
            logits = self.multimodal_model(
                image_tensor,
                encoding["input_ids"],
                encoding["attention_mask"]
            )
            probs = torch.softmax(logits, dim=1)[0]
            prob_real = probs[0].item()
            prob_fake = probs[1].item()
            verdict = "Fake" if prob_fake >= threshold else "Real"
            confidence = max(prob_real, prob_fake)
        
        processing_time_ms = (time.time() - start) * 1000
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "probabilities": {"Real": prob_real, "Fake": prob_fake},
            "processing_time_ms": processing_time_ms,
        }

    def auto_detect_model_type(self, text: str, has_image: bool) -> str:
        if has_image:
            return "multimodal"
        
        # Heuristic for SMS vs News
        url_keywords = ["http", "www.", ".com/", "click", "bit.ly"]
        has_url = any(kw in text.lower() for kw in url_keywords)
        word_count = len(text.split())
        
        if has_url or word_count < 20:
            return "sms"
        return "news"


# Global model manager instance
model_manager = ModelManager()


def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def load_image_from_base64(base64_str: str) -> Image.Image:
    import base64
    
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data))
