from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA

DATA_DIR = Path(".")
df = pd.read_csv(DATA_DIR / "ir_train.csv")
df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

if "is_overloaded" in df.columns:
    df = df.drop(columns=["is_overloaded"])

# Same engineered features used by the project
df["average_speed_kmph"] = df["distance_km"] / df["scheduled_travel_hours"]
df["stops_per_100km"] = df["num_scheduled_stops"] / df["distance_km"] * 100
df["distance_per_stop_km"] = df["distance_km"] / df["num_scheduled_stops"]
df["average_fleet_age"] = (df["loco_age_years"] + df["coach_age_years"]) / 2
df["fleet_age_difference"] = (df["loco_age_years"] - df["coach_age_years"]).abs()
df["weather_risk_score"] = df["fog_risk_score"] + df["season_severity_score"] + df["is_monsoon_season"]
df["utilisation_ratio"] = df["seat_utilisation_pct"] / 100
df["operational_pressure_score"] = df["zone_congestion_index"] + df["utilisation_ratio"] + df["late_incoming_rake"]
df["route_complexity_score"] = df["num_scheduled_stops"] + df["psr_count"] + df["is_hdn_route"]
df["infrastructure_score"] = df["track_doubled"] + df["is_electrified"] + df["has_lhb_coaches"]
df["temporal_pressure_score"] = df["is_peak_hour"] + df["is_night_departure"] + df["is_festival_season"]

X = df.drop(columns=["is_delayed", "delay_minutes", "primary_delay_cause", "journey_id", "train_number", "departure_date", "zone"], errors="ignore")

categorical = ["train_type", "season", "zone_abbr", "source_station_category", "destination_station_category", "traction_type"]
numerical = [c for c in X.columns if c not in categorical]

sample = X.sample(n=min(100000, len(X)), random_state=42)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical)
])

processed = preprocessor.fit_transform(sample)
print("Original encoded dimensions:", processed.shape[1])

pca_full = PCA()
pca_full.fit(processed)

cumulative = np.cumsum(pca_full.explained_variance_ratio_)
n95 = np.argmax(cumulative >= 0.95) + 1

pca_95 = PCA(n_components=n95)
reduced = pca_95.fit_transform(processed)

print("Reduced PCA dimensions:", reduced.shape[1])
print("Variance retained:", pca_95.explained_variance_ratio_.sum() * 100)
