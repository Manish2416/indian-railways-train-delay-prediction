from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import xgboost as xgb

DATA_DIR = Path(".")
df = pd.read_csv(DATA_DIR / "ir_train.csv")
df["departure_date"] = pd.to_datetime(df["departure_date"], errors="coerce")

if "is_overloaded" in df.columns:
    df = df.drop(columns=["is_overloaded"])

# Feature engineering
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

df = df.drop(columns=["departure_date", "zone"], errors="ignore")

target = "is_delayed"
X = df.drop(columns=[target, "delay_minutes", "primary_delay_cause", "journey_id", "train_number"], errors="ignore")
y = df[target]

categorical = ["train_type", "season", "zone_abbr", "source_station_category", "destination_station_category", "traction_type"]
numerical = [c for c in X.columns if c not in categorical]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Memory-conscious development sample, matching the verified project workflow
X_train_model, _, y_train_model, _ = train_test_split(
    X_train, y_train, train_size=300000, random_state=42, stratify=y_train
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True), categorical)
])

X_train_processed = preprocessor.fit_transform(X_train_model)
X_valid_processed = preprocessor.transform(X_valid)

models = {
    "Logistic Regression": LogisticRegression(max_iter=200, solver="lbfgs", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=5,
        max_features="sqrt", n_jobs=-1, random_state=42, class_weight="balanced"
    ),
    "XGBoost": xgb.XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=42
    )
}

rows = []
probabilities = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_processed, y_train_model)
    pred = model.predict(X_valid_processed)
    prob = model.predict_proba(X_valid_processed)[:, 1]

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_valid, pred),
        "Precision": precision_score(y_valid, pred, zero_division=0),
        "Recall": recall_score(y_valid, pred, zero_division=0),
        "F1-Score": f1_score(y_valid, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_valid, prob)
    }
    rows.append(row)
    probabilities[name] = prob

    print(pd.Series(row))
    print("Confusion Matrix:")
    print(confusion_matrix(y_valid, pred))

results = pd.DataFrame(rows)
print("\nMODEL COMPARISON")
print(results.to_string(index=False))
results.to_csv(DATA_DIR / "model_comparison_results.csv", index=False)

plt.figure(figsize=(9, 7))
for name, prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_valid, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_valid, prob):.4f})")
plt.plot([0, 1], [0, 1], "--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
