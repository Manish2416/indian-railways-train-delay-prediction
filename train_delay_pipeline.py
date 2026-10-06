from pathlib import Path
import pandas as pd

DATA_DIR = Path(".")

def load_and_clean():
    df = pd.read_csv(DATA_DIR / "ir_train.csv")
    df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")
    if "is_overloaded" in df.columns and df["is_overloaded"].nunique() <= 1:
        df = df.drop(columns=["is_overloaded"])
    return df

def engineer_features(df):
    df = df.copy()
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
    return df

if __name__ == "__main__":
    train = load_and_clean()
    features = engineer_features(train)
    print("Cleaned:", train.shape)
    print("Feature-engineered:", features.shape)
