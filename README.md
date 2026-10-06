# Indian Railways Train Delay Prediction

Analysis and Prediction of Train Delays in Indian Railways using Python, EDA, feature engineering, PCA, and machine learning.

## Project Sections

1. Data Understanding & Schema Introspection
2. Data Cleaning & Deduplication
3. Data Preprocessing
4. Exploratory Data Analysis
5. Feature Engineering
6. Principal Component Analysis (PCA)
7. Predictive Modeling & Verification
8. Key Discoveries & Verdicts

## Dataset

- Training data: 1,500,000 rows, 45 columns
- Test data: 375,000 rows, 42 columns
- Target: `is_delayed`

Raw Kaggle CSV files are intentionally not committed to this repository.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Dummy Baseline | 71.89% | 71.89% | 100.00% | 83.65% | — |
| Logistic Regression | **86.26%** | 89.01% | **92.28%** | **90.62%** | **0.9238** |
| Random Forest | 83.51% | **92.63%** | 83.72% | 87.95% | 0.9156 |
| XGBoost | 86.14% | 89.04% | 92.06% | 90.52% | 0.9223 |

PCA reduced 88 encoded features to 37 components while retaining 95.11% variance.

## Python Files

- `check_data.py` — data validation
- `02_preprocessing_eda.py` — cleaning, preprocessing and EDA
- `03_predictive_modelling.py` — ML training and verification
- `train_delay_pipeline.py` — reusable feature-engineering pipeline
- `generate_final_predictions.py` — competition submission template
- `requirements.txt` — dependencies

## Running

Place `ir_train.csv`, `ir_test.csv`, `ir_data_dictionary.csv`, and `ir_sample_submission.csv` beside the scripts, then run:

```bash
pip install -r requirements.txt
python check_data.py
python 02_preprocessing_eda.py
python 03_predictive_modelling.py
```

## Leakage Prevention

The predictive models exclude `is_delayed`, `delay_minutes`, `primary_delay_cause`, `journey_id`, and `train_number`.
