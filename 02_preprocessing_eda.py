from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(".")
OUTPUT_DIR = DATA_DIR / "results" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_DIR / "ir_train.csv")
df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

if "is_overloaded" in df.columns and df["is_overloaded"].nunique() <= 1:
    df = df.drop(columns=["is_overloaded"])

# Overall delay distribution
delay = (
    df["is_delayed"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename(index={0: "On Time", 1: "Delayed"})
    .to_frame("Count")
)
delay["Percentage"] = (delay["Count"] / delay["Count"].sum() * 100).round(2)

plt.figure(figsize=(9, 6))
plt.bar(delay.index, delay["Count"])
plt.title("Overall Train Delay Distribution")
plt.ylabel("Number of journeys")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_overall_delay_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

# Delay by train type
train_type = df.groupby("train_type")["is_delayed"].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(12, 7))
plt.barh(train_type.index[::-1], train_type.values[::-1])
plt.title("Delay Rate by Train Type")
plt.xlabel("Delayed journeys (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "06_delay_rate_by_train_type.png", dpi=200, bbox_inches="tight")
plt.close()

# Delay by railway zone
zone = df.groupby("zone_abbr")["is_delayed"].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(11, 7))
plt.barh(zone.index[::-1], zone.values[::-1])
plt.title("Delay Rate by Railway Zone")
plt.xlabel("Delayed journeys (%)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "07_delay_rate_by_zone.png", dpi=200, bbox_inches="tight")
plt.close()

# Delay by season
season = df.groupby("season")["is_delayed"].mean().mul(100).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(season.index, season.values)
plt.title("Delay Rate by Season")
plt.ylabel("Delayed journeys (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_delay_rate_by_season_from_raw_data.png", dpi=200, bbox_inches="tight")
plt.close()

# Delay by departure hour
hour = df.groupby("departure_hour")["is_delayed"].mean().mul(100)
plt.figure(figsize=(12, 6))
plt.plot(hour.index, hour.values, marker="o")
plt.title("Train Delay Rate by Departure Hour")
plt.xlabel("Departure hour")
plt.ylabel("Delay rate (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "08_delay_rate_by_departure_hour.png", dpi=200, bbox_inches="tight")
plt.close()

# Primary delay causes
causes = df["primary_delay_cause"].value_counts().sort_values()
plt.figure(figsize=(12, 8))
plt.barh(causes.index, causes.values)
plt.title("Distribution of Primary Delay Causes")
plt.xlabel("Number of journeys")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "09_primary_delay_causes.png", dpi=200, bbox_inches="tight")
plt.close()

# Delay-duration distribution
plt.figure(figsize=(10, 6))
plt.hist(df["delay_minutes"], bins=50)
plt.title("Distribution of Train Delay Duration")
plt.xlabel("Delay (minutes)")
plt.ylabel("Number of journeys")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "10_delay_duration_distribution.png", dpi=200, bbox_inches="tight")
plt.close()

print("EDA visualizations saved to:", OUTPUT_DIR)
for path in sorted(OUTPUT_DIR.glob("*.png")):
    print(path)
