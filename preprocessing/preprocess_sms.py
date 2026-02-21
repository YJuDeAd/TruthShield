import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

PROCESSED_PATH = "data/processed/sms/"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# ======================
# 1. SMS Spam Collection Dataset
# Download: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# ======================

sms_path = "data/raw/sms/SMS_Spam_Collection_Dataset/spam.csv"

sms_df = pd.read_csv(sms_path, encoding='latin-1')
sms_df = sms_df[['v1', 'v2']]
sms_df.columns = ['label', 'text']

sms_df['label'] = sms_df['label'].map({
    'ham': 0,
    'spam': 1
})

sms_df = sms_df[['text', 'label']]

print("SMS samples:", len(sms_df))

# ======================
# 2. Phishing Email Dataset
# Download: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
# ======================

phishing_folder = "data/raw/sms/Phishing_Email_Dataset/"

all_files = glob.glob(phishing_folder + "*.csv")

phish_list = []

for file in all_files:
    
    try:
        df = pd.read_csv(file)
        
        # Try to detect text column automatically
        text_col = None
        
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
        
        if text_col is None:
            continue
        
        df = df[[text_col]]
        df.columns = ['text']
        
        # All emails in these datasets are phishing
        df['label'] = 1
        
        phish_list.append(df)
        
        print("Loaded:", file, "Samples:", len(df))
        
    except:
        print("Skipped:", file)

phish_df = pd.concat(phish_list, ignore_index=True)

print("Total phishing samples:", len(phish_df))

# ======================
# COMBINE ALL DATA
# ======================

combined_df = pd.concat([sms_df, phish_df], ignore_index=True)

combined_df.dropna(inplace=True)

print("Total combined samples:", len(combined_df))

# ======================
# SPLIT
# ======================

train_df, test_df = train_test_split(
    combined_df,
    test_size=0.2,
    random_state=42,
    stratify=combined_df['label']
)

train_df.to_csv(PROCESSED_PATH + "train.csv", index=False)
test_df.to_csv(PROCESSED_PATH + "test.csv", index=False)

# ======================
# TF-IDF
# ======================

vectorizer = TfidfVectorizer(
    max_features=15000,
    stop_words='english'
)

vectorizer.fit(train_df['text'])

joblib.dump(vectorizer, PROCESSED_PATH + "vectorizer.pkl")

print("\nPreprocessing complete.")
print("Train size:", len(train_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label"].value_counts().to_string())
print("\nTest label distribution:")
print(test_df["label"].value_counts().to_string())