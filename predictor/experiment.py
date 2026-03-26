import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

sys.path.append('.')
from eda import InsuranceDataProcessor
from predictor import InsurancePremiumPredictor

CONFIGS_PATH = Path('eda/configs/filters')
MODELS_PATH = Path('models')
DATA_PATH = Path('~/MLOps/vehicle-insurance-data/motor_data11-14lats.csv')

print("Loading data...")
df = pd.read_csv(DATA_PATH)

thresh_year = 13
df_filter = df['INSR_BEGIN'].apply(lambda x: int(x.split('-')[2])) <= thresh_year

df_train = df[df_filter].copy()
df_test = df[~df_filter].copy()

print(f"Train size: {len(df_train)}")
print(f"Test size: {len(df_test)}")

predictor = InsurancePremiumPredictor(CONFIGS_PATH, MODELS_PATH)

print("\n" + "="*60)
print("Training model...")
print("="*60)
predictor.fit(df_train, True)

print("\n" + "="*60)
print("Predicting on test data...")
print("="*60)
predictions = predictor.predict(df_test)

df_test_zero, df_test_else, _ = predictor.processor.transform(df_test, train=False)
actuals = pd.concat([df_test_zero, df_test_else]).sort_index()['PREMIUM'].values

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
r2 = r2_score(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

print("\n" + "="*60)
print("Test Results")
print("="*60)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

predictor = InsurancePremiumPredictor(CONFIGS_PATH, MODELS_PATH)
predictor.fit(df)
predictor.save()
