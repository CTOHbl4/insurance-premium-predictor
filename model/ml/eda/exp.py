import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from insurance_data_processor import InsuranceDataProcessor

idp = InsuranceDataProcessor('configs/filters')

df = pd.read_csv('~/MLOps/project/generator/vehicle-insurance-data/motor_data11-14lats.csv')
idp.fit(df)
df_zero, df_else, _ = idp.transform(df)

# Statistics for zero vs else
print("=== Zero-Insured ===")
print(f"Premium mean: {df_zero['PREMIUM'].mean():.2f}")
print(f"Premium std: {df_zero['PREMIUM'].std():.2f}")
print(f"Premium quantiles: {df_zero['PREMIUM'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])}")
print(f"Sample size: {len(df_zero)}")

print("\n=== Positive-Insured ===")
print(f"Premium mean: {df_else['PREMIUM'].mean():.2f}")
print(f"Premium std: {df_else['PREMIUM'].std():.2f}")
print(f"Premium quantiles: {df_else['PREMIUM'].quantile([0.25, 0.5, 0.75, 0.9, 0.95])}")
print(f"Sample size: {len(df_else)}")

# Feature counts
print(f"\nZero unique OBJECT_ID: {df_zero['OBJECT_ID'].nunique()}")
print(f"Positive unique OBJECT_ID: {df_else['OBJECT_ID'].nunique()}")
