import pandas as pd
from sklearn.model_selection import train_test_split

# Load original dataset
df = pd.read_csv("data/raw/Phishing_Email.csv")

print("Original dataset:", df.shape)

# Keep only required columns
df = df[["Email Text", "Email Type"]]

# Rename columns
df = df.rename(columns={
    "Email Text": "text",
    "Email Type": "label"
})

# Remove missing and empty emails
df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

# Remove duplicate emails
df = df.drop_duplicates(subset=["text"])

# Convert labels into numbers
df["label"] = df["label"].map({
    "Safe Email": 0,
    "Phishing Email": 1
})

# Remove any unrecognized labels
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("\nCleaned dataset:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

# Split into training (70%), validation (15%), testing (15%)
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["label"]
)

val, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42,
    stratify=temp["label"]
)

# Save processed datasets
train.to_csv("data/processed/train.csv", index=False)
val.to_csv("data/processed/val.csv", index=False)
test.to_csv("data/processed/test.csv", index=False)

print("\nDataset splitting completed!")
print("Training emails:", len(train))
print("Validation emails:", len(val))
print("Testing emails:", len(test))

print("\nFiles saved successfully!")