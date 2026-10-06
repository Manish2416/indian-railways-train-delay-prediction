from pathlib import Path
import pandas as pd

DATA_DIR = Path(".")

train = pd.read_csv(DATA_DIR / "ir_train.csv")
test = pd.read_csv(DATA_DIR / "ir_test.csv")

print("=== SHAPES ===")
print("Train:", train.shape)
print("Test :", test.shape)

print("\n=== DATA QUALITY ===")
print("Train missing:", train.isna().sum().sum())
print("Test missing :", test.isna().sum().sum())
print("Duplicate train rows:", train.duplicated().sum())
print("Duplicate test rows :", test.duplicated().sum())

print("\n=== JOURNEY IDS ===")
print("Unique train IDs:", train["journey_id"].nunique())
print("Duplicate train IDs:", train["journey_id"].duplicated().sum())

print("\n=== CONSTANT COLUMNS ===")
print([c for c in train.columns if train[c].nunique(dropna=False) <= 1])

print("\n=== TRAIN/TEST DIFFERENCE ===")
print("Train only:", sorted(set(train.columns) - set(test.columns)))
print("Test only :", sorted(set(test.columns) - set(train.columns)))

dates = pd.to_datetime(train["departure_date"], errors="coerce")
print("\n=== DATES ===")
print("Invalid dates:", dates.isna().sum())
print("Earliest:", dates.min())
print("Latest:", dates.max())

print("\n=== TARGET ===")
print(train["is_delayed"].value_counts())
print(train["is_delayed"].value_counts(normalize=True).mul(100).round(2))

print("\n=== DELAY CONSISTENCY ===")
print("Not delayed but >15 min:",
      ((train["is_delayed"] == 0) & (train["delay_minutes"] > 15)).sum())
print("Delayed but <=15 min:",
      ((train["is_delayed"] == 1) & (train["delay_minutes"] <= 15)).sum())
