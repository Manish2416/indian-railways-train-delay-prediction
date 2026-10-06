from pathlib import Path
import pandas as pd

DATA_DIR = Path(".")
sample = pd.read_csv(DATA_DIR / "ir_sample_submission.csv")
test = pd.read_csv(DATA_DIR / "ir_test.csv")

print("Test shape:", test.shape)
print("Sample submission shape:", sample.shape)
print("Submission columns:", sample.columns.tolist())

# After fitting the final model:
# test_processed = preprocessor.transform(test_features)
# predictions = final_model.predict(test_processed)
# submission = sample.copy()
# submission["is_delayed"] = predictions
# submission.to_csv(DATA_DIR / "final_predictions.csv", index=False)
