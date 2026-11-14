import pandas as pd

train = pd.read_csv("data/processed/train.csv")

keywords = [
    "newsletter",
    "unsubscribe",
    "offer",
    "promotion",
    "recruitment",
    "sale",
    "discount",
    "click",
    "password",
    "account"
]

for label, name in [(0, "SAFE"), (1, "PHISHING")]:

    emails = train[train["label"] == label]["text"].fillna("")

    print(f"\n{name} EMAILS")
    print("Total:", len(emails))

    for word in keywords:
        count = emails.str.contains(
            word, case=False, regex=False
        ).sum()

        percentage = count / len(emails) * 100

        print(
            f"{word}: {count} emails ({percentage:.2f}%)"
        )