import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertModel
from torchvision import models, transforms
from PIL import Image
import pickle

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# CONFIG — must match training configs exactly
# ======================

# SMS model config (must match train_sms.py)
SMS_MAX_LEN     = 300
SMS_EMBED_SIZE  = 128
SMS_HIDDEN_SIZE = 128
SMS_DROPOUT     = 0.4

# News model config (must match train_news.py)
NEWS_MODEL_NAME = "roberta-large"


# =====================================================
# SMS MODEL DEFINITION
# =====================================================

class HybridModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, SMS_EMBED_SIZE, padding_idx=0)
        self.embed_dropout = nn.Dropout(SMS_DROPOUT)
        self.lstm = nn.LSTM(
            SMS_EMBED_SIZE, SMS_HIDDEN_SIZE,
            batch_first=True, bidirectional=True,
            num_layers=2, dropout=SMS_DROPOUT
        )
        self.conv = nn.Conv1d(
            in_channels=SMS_HIDDEN_SIZE * 2,
            out_channels=128,
            kernel_size=3, padding=1
        )
        self.relu       = nn.ReLU()
        self.fc_dropout = nn.Dropout(SMS_DROPOUT)
        self.fc         = nn.Linear(128, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.embed_dropout(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out.permute(0, 2, 1)
        conv_out = self.relu(self.conv(lstm_out))
        pooled   = torch.mean(conv_out, dim=2)
        out = self.fc_dropout(pooled)
        out = self.sigmoid(self.fc(out))
        return out.squeeze()


# =====================================================
# LOAD SMS MODEL
# =====================================================

# Load vocab saved during training
with open("models/sms_model/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

sms_vocab_size = len(vocab) + 2

sms_model = HybridModel(sms_vocab_size)
sms_model.load_state_dict(
    torch.load("models/sms_model/sms_model.pt", map_location=DEVICE)
)
sms_model.to(DEVICE)
sms_model.eval()


# =====================================================
# LOAD NEWS MODEL
# =====================================================

news_tokenizer = RobertaTokenizer.from_pretrained(NEWS_MODEL_NAME)

news_model = RobertaForSequenceClassification.from_pretrained(
    NEWS_MODEL_NAME,
    num_labels=2
)
news_model.load_state_dict(
    torch.load("models/news_model/roberta_news_best.pt", map_location=DEVICE)
)
news_model.to(DEVICE)
news_model.eval()


# =====================================================
# LOAD MULTIMODAL MODEL
# =====================================================

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_model = models.resnet50(weights=None)
        self.image_model.fc = nn.Linear(2048, 256)
        self.text_model  = BertModel.from_pretrained("bert-base-uncased")
        self.text_fc     = nn.Linear(768, 256)
        self.classifier  = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_model(image)
        text_outputs   = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features  = text_outputs.last_hidden_state[:, 0, :]
        text_features  = self.text_fc(text_features)
        combined       = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined)


mm_model = MultimodalModel()
mm_model.load_state_dict(
    torch.load("models/multimodal_model/best_model.pt", map_location=DEVICE)
)
mm_model.to(DEVICE)
mm_model.eval()

mm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =====================================================
# SMS PREPROCESSING
# =====================================================

UNK_IDX = 1
PAD_IDX = 0

def tokenize(text):
    return str(text).lower().split()

def text_to_sequence(text):
    tokens = tokenize(text)
    seq    = [vocab.get(word, UNK_IDX) for word in tokens]
    if len(seq) < SMS_MAX_LEN:
        seq += [PAD_IDX] * (SMS_MAX_LEN - len(seq))
    else:
        seq = seq[:SMS_MAX_LEN]
    return seq


# =====================================================
# PREDICTION FUNCTIONS
# =====================================================

def predict_news(text):
    encoding = news_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs    = news_model(**encoding)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return prediction


def predict_multimodal(text, image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0).to(DEVICE)

    encoding = mm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    encoding = {k: v.to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs    = mm_model(image, encoding["input_ids"], encoding["attention_mask"])
        prediction = torch.argmax(outputs, dim=1).item()

    return prediction


def predict_sms(text):
    seq    = text_to_sequence(text)
    tensor = torch.tensor([seq], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        prob       = sms_model(tensor).item()
        prediction = 1 if prob > 0.5 else 0

    return prediction


# =====================================================
# ROUTER
# =====================================================

def is_sms_or_phishing(text):
    """
    Heuristic to decide if input looks like SMS/phishing vs a news article.
    SMS/phishing signals: contains URLs, very short, informal punctuation.
    News signals: longer, no URLs, proper sentences.
    """
    url_keywords = ["http", "www.", ".com/", ".net/", ".org/",
                    "click here", "click now", "bit.ly", "tinyurl"]
    has_url      = any(kw in text.lower() for kw in url_keywords)
    word_count   = len(text.split())
    char_count   = len(text)

    # Route to SMS model if: has a URL, OR text is very short (< 20 words AND < 150 chars)
    if has_url:
        return True
    if word_count < 20 and char_count < 150:
        return True
    return False


def unified_predict(text=None, image_path=None, input_type=None):
    """
    Route input to the correct model.

    Args:
        text       : input text
        image_path : path to image (triggers multimodal model)
        input_type : optional override — "news", "sms", or "multimodal"
                     Use this when the heuristic might be wrong (e.g. short news headlines)
    """
    if not text and not image_path:
        return "Invalid input: provide at least text or an image path."

    # --- Explicit override (most reliable) ---
    if input_type == "multimodal" or (text and image_path):
        result     = predict_multimodal(text, image_path)
        model_used = "Multimodal Model"

    elif input_type == "news":
        result     = predict_news(text)
        model_used = "News Model (RoBERTa)"

    elif input_type == "sms":
        result     = predict_sms(text)
        model_used = "SMS/Phishing Model (Bi-LSTM + CNN)"

    # --- Auto-detect fallback ---
    elif text and is_sms_or_phishing(text):
        result     = predict_sms(text)
        model_used = "SMS/Phishing Model (Bi-LSTM + CNN)"

    else:
        result     = predict_news(text)
        model_used = "News Model (RoBERTa)"

    label = "FAKE / SPAM" if result == 1 else "REAL / HAM"

    return {
        "model_used": model_used,
        "prediction": label
    }


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    # Short headline — use input_type="news" to force correct routing
    sample_news_short = "Breaking news: Government confirms economic collapse."
    print("Short news headline (explicit routing):")
    print(unified_predict(text=sample_news_short, input_type="news"))

    # Long article — auto-detected correctly without input_type
    sample_news_long = (
        "The government announced today that the national budget has fallen into deficit "
        "following months of economic instability driven by rising inflation and declining "
        "exports across key industrial sectors. Economists warn this could impact public "
        "services and welfare programs over the next fiscal year."
    )
    print("\nLong news article (auto-detected):")
    print(unified_predict(text=sample_news_long))

    # SMS with URL — auto-detected correctly
    sample_sms_url = "Congratulations! You have won a $1000 gift card. Click: http://claim-now.com"
    print("\nPhishing SMS with URL (auto-detected):")
    print(unified_predict(text=sample_sms_url))

    # Short SMS, no URL — force with input_type
    sample_sms_plain = "Free entry in a weekly competition to win FA Cup tickets."
    print("\nShort SMS no URL (explicit routing):")
    print(unified_predict(text=sample_sms_plain, input_type="sms"))