import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

PROCESSED_PATH = "data/processed/news/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

all_data = []

# ----------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def filter_min_length(df, min_chars=50):
    mask = df["content"].str.len() >= min_chars
    removed = (~mask).sum()
    if removed > 0:
        print(f"  Removed {removed} rows with content < {min_chars} chars")
    return df[mask].copy()


# ======================
# 1. WELFake DATASE
# Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
# NOTE: WELFake label convention is OPPOSITE to our convention.
#       We flip it: WELFake 0 (fake) → our 1, WELFake 1 (real) → our 0
# ======================

welfake_df = pd.read_csv("data/raw/news/WELFake/WELFake_Dataset.csv")

welfake_df["label"] = welfake_df["label"].apply(lambda x: 1 if x == 0 else 0)
welfake_df["content"] = (welfake_df["title"].fillna("") + " " + welfake_df["text"].fillna("")).apply(clean_text)
welfake_df = filter_min_length(welfake_df[["content", "label"]])

all_data.append(welfake_df)
print("WELFake loaded:", len(welfake_df))
print("  WELFake label distribution:")
print(" ", welfake_df["label"].value_counts().to_dict())


# ======================
# 2. FakeNewsNet — BuzzFeed ONLY
# Download: https://github.com/KaiDMML/FakeNewsNet
# NOTE: PolitiFact is already inside WELFake — skip it to avoid duplication.
#       GossipCop is also inside WELFake AND is title-only — skip it entirely.
# ======================

buzz_fake = pd.read_csv("data/raw/news/FakeNewsNet/BuzzFeed_fake_news_content.csv")
buzz_real = pd.read_csv("data/raw/news/FakeNewsNet/BuzzFeed_real_news_content.csv")

buzz_fake["label"] = 1
buzz_real["label"] = 0

buzz_fake["content"] = (buzz_fake["title"].fillna("") + " " + buzz_fake["text"].fillna("")).apply(clean_text)
buzz_real["content"] = (buzz_real["title"].fillna("") + " " + buzz_real["text"].fillna("")).apply(clean_text)

buzz_fake = filter_min_length(buzz_fake[["content", "label"]])
buzz_real = filter_min_length(buzz_real[["content", "label"]])

all_data.extend([buzz_fake, buzz_real])
print("BuzzFeed loaded:", len(buzz_fake) + len(buzz_real))


# ======================
# COMBINE ALL DATA
# ======================

df = pd.concat(all_data, ignore_index=True)
df = df[df["content"].notna() & (df["content"].str.strip() != "")]

print("\nTOTAL SAMPLES BEFORE DEDUP:", len(df))
print("Label distribution (0=real, 1=fake):")
print(df["label"].value_counts().to_string())


# ======================
# SPLIT
# ======================

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# Remove duplicates in training data
train_df = train_df.drop_duplicates(subset=["content"])

# Remove train samples that leak into test set
test_contents = set(test_df["content"].unique())
train_df = train_df[~train_df["content"].isin(test_contents)]

train_df.to_csv(PROCESSED_PATH + "train.csv", index=False)
test_df.to_csv(PROCESSED_PATH + "test.csv", index=False)

print("\nFinal Train:", len(train_df))
print("Final Test:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts().to_string())
print("\nTest label distribution:")
print(test_df["label"].value_counts().to_string())