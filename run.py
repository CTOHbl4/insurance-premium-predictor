import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

import warnings
warnings.filterwarnings('ignore', message='.*no_silent_downcasting.*')

sys.path.append('.')
from predictor import InsurancePremiumPredictor


CONFIGS_PATH = Path('eda/configs/filters')
MODELS_PATH = Path('models')
DATA_PATH = Path('data')
DATA_PATH.mkdir(exist_ok=True)

TRAINING_DATA_FILE = DATA_PATH / 'training_data.csv'


def load_existing_training_data() -> pd.DataFrame:
    """Load existing training data if exists."""
    if TRAINING_DATA_FILE.exists():
        return pd.read_csv(TRAINING_DATA_FILE)
    return pd.DataFrame()


def save_training_data(df: pd.DataFrame) -> None:
    """Save training data to file."""
    TRAINING_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_DATA_FILE, index=False)
    print(f"Training data saved to {TRAINING_DATA_FILE}")


def train_model(df: pd.DataFrame) -> InsurancePremiumPredictor:
    """Train model on full dataset."""
    print(f"Training model on {len(df)} records...")
    predictor = InsurancePremiumPredictor(CONFIGS_PATH, MODELS_PATH)
    predictor.fit(df)
    predictor.save()
    print("Model training complete.")
    return predictor


def update_model() -> None:
    """Update model by retraining on all accumulated data."""
    print("=" * 60)
    print("UPDATE MODE: Retraining model on all accumulated data")
    print("=" * 60)

    df = load_existing_training_data()

    if df.empty:
        print("No training data found. Please run inference first to collect data.")
        return

    predictor = train_model(df)

    metrics = predictor.get_metrics()
    print("\nTraining metrics:")
    for model_name, model_metrics in metrics.items():
        if model_metrics and 'best_score' in model_metrics:
            print(f"  {model_name}: RMSE = {model_metrics['best_score']:.2f}")


def inference(file_path: str) -> None:
    """
    Run inference on new data.

    If data contains PREMIUM column: store as training data (no prediction)
    Otherwise: predict premiums and save results
    """
    print("=" * 60)
    print("INFERENCE MODE")
    print("=" * 60)

    input_path = Path(file_path)
    if not input_path.exists():
        print(f"Error: File {file_path} not found.")
        return

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} records from {file_path}")

    if 'PREMIUM' in df.columns:
        print("Data contains PREMIUM column. Appending to training data.")

        df.to_csv(TRAINING_DATA_FILE, mode='a', header=not TRAINING_DATA_FILE.exists(), index=False)
        print(f"Appended {len(df)} records to {TRAINING_DATA_FILE}")

        print("\nTo update the model, run: python run.py -mode update")

    else:
        print("Data does not contain PREMIUM. Running prediction...")

        predictor = InsurancePremiumPredictor(CONFIGS_PATH, MODELS_PATH)

        if not MODELS_PATH.exists():
            print("Error: No trained model found. Please run update mode first.")
            return

        try:
            predictor.load()
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        predictions = predictor.predict(df)

        df_result = df.copy()
        df_result['predicted_premium'] = predictions

        output_path = input_path.parent / f"{input_path.stem}_predicted{input_path.suffix}"
        df_result.to_csv(output_path, index=False)

        print(f"\nPredictions saved to: {output_path}")
        print(f"Predictions summary:")
        print(f"  Min: {predictions.min():.2f}")
        print(f"  Max: {predictions.max():.2f}")
        print(f"  Mean: {predictions.mean():.2f}")
        print(f"  Median: {np.median(predictions):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Insurance Premium Predictor')
    parser.add_argument('-mode', required=True, choices=['inference', 'update'],
                        help='Mode: inference (predict) or update (retrain)')
    parser.add_argument('-file', help='Path to input file (required for inference mode)')

    args = parser.parse_args()

    if args.mode == 'inference':
        if not args.file:
            print("Error: -file argument is required for inference mode")
            parser.print_help()
            sys.exit(1)
        inference(args.file)

    elif args.mode == 'update':
        update_model()

    else:
        print(f"Unknown mode: {args.mode}")
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
