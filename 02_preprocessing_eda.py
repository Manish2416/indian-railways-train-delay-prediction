from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(".")
df = pd.read_csv(DATA_DIR / "ir_train.csv")
df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

if "is_overloaded" in df.columns and df["is_overloaded"].nunique() <= 1:
    df = df.drop(columns=["is_overloaded"])

print("Cleaned shape:", df.shape)

delay = df["is_delayed"].value_counts().rename(index={0:"On Time", 1:"Delayed"}).to_frame("Count")
delay["Percentage"] = (delay["Count"] / delay["Count"].sum() * 100).round(2)
print("\nOverall delay distribution:\n", delay)

train_type = df.groupby("train_type")["is_delayed"].agg(["count","mean"])
train_type["delay_percentage"] = (train_type["mean"] * 100).round(2)
print("\nDelay by train type:\n", train_type.sort_values("mean", ascending=False))

zone = df.groupby("zone_abbr")["is_delayed"].agg(["count","mean"])
zone["delay_percentage"] = (zone["mean"] * 100).round(2)
print("\nDelay by zone:\n", zone.sort_values("mean", ascending=False))

season = df.groupby("season")["is_delayed"].agg(["count","mean"])
season["delay_percentage"] = (season["mean"] * 100).round(2)
print("\nDelay by season:\n", season.sort_values("mean", ascending=False))

hour = df.groupby("departure_hour")["is_delayed"].mean().mul(100)
print("\nDelay by departure hour:\n", hour.round(2))

print("\nPrimary delay causes:\n", df["primary_delay_cause"].value_counts())
print("\nDelay duration:\n", df["delay_minutes"].describe())

figures = [
    ("Train Delay Status Distribution", ["On Time", "Delayed"], delay["Count"].values, "bar"),
    ("Delay Rate by Train Type", train_type.sort_values("delay_percentage").index, train_type.sort_values("delay_percentage")["delay_percentage"].values, "barh"),
    ("Delay Rate by Railway Zone", zone.index, zone["delay_percentage"].values, "bar"),
    ("Delay Rate by Season", season.sort_values("delay_percentage").index, season.sort_values("delay_percentage")["delay_percentage"].values, "bar"),
]

for title, x, y, kind in figures:
    plt.figure(figsize=(11, 6))
    if kind == "barh":
        plt.barh(x, y)
    else:
        plt.bar(x, y)
    plt.title(title)
    plt.ylabel("Delayed Journeys (%)" if "Status" not in title else "Number of Journeys")
    plt.xticks(rotation=45 if kind == "bar" else 0)
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 5))
plt.plot(hour.index, hour.values, marker="o")
plt.title("Train Delay Rate by Departure Hour")
plt.xlabel("Departure Hour")
plt.ylabel("Delay Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(11, 7))
df["primary_delay_cause"].value_counts().sort_values().plot(kind="barh")
plt.title("Distribution of Primary Delay Causes")
plt.xlabel("Number of Journeys")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(df["delay_minutes"], bins=50)
plt.title("Distribution of Train Delay Duration")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of Journeys")
plt.tight_layout()
plt.show()
