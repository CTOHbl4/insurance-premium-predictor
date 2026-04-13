import pandas as pd
import shutil
import sys
import logging
from pathlib import Path
import time
import warnings
import json

sys.path.append('/app')

from ml import InsurancePremiumPredictor
from services.mysql import DatabaseManager
from services.config import Config

warnings.filterwarnings('ignore', message='.*no_silent_downcasting.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self):
        self.db = DatabaseManager()

        self.staging_models = Path(Config.STAGING_MODELS_PATH)
        self.active_models = Path(Config.ACTIVE_MODELS_PATH)
        self.backup_models = Path(Config.BACKUP_MODELS_PATH)

        self.staging_filters = Path(Config.STAGING_FILTERS_PATH)
        self.active_filters = Path(Config.ACTIVE_FILTERS_PATH)
        self.backup_filters = Path(Config.BACKUP_FILTERS_PATH)

        self.predictor_state_filename = Config.PREDICTOR_STATE_FILENAME

        self.new_model_flag = self.active_models / Config.FLAG_FILENAME

        for path in [self.staging_models, self.active_models, self.backup_models,
                     self.staging_filters, self.active_filters, self.backup_filters]:
            path.mkdir(parents=True, exist_ok=True)

        self.predictor = None
        self.last_seen_samples = 0

    def fetch_training_data(self) -> pd.DataFrame:
        logger.info("Fetching training data...")
        df = self.db.get_training_data()
        logger.info(f"Fetched {len(df)} records")
        return df

    def get_validation_mape(self) -> float:
        """Get validation MAPE from current predictor."""
        if self.predictor is None:
            return float('inf')
        return self.predictor.metrics.get('validation_mape', float('inf'))

    def check_drift(self, current_mape: float) -> str:
        """Check for model or data drift using validation MAPE."""
        prev_mape = self.get_validation_mape()

        if prev_mape == float('inf'):
            return 'model'

        mape_increase = current_mape - prev_mape

        if mape_increase > 10:
            logger.warning(f"MAPE increased from {prev_mape:.2f}% to {current_mape:.2f}%")
            return 'model'
        elif mape_increase > 5:
            logger.info(f"MAPE increased from {prev_mape:.2f}% to {current_mape:.2f}%")
            return 'data'

        return None

    def train_model(self, train_df: pd.DataFrame, val_df: pd.DataFrame, full_train: bool = True) -> float:
        """Train model and return validation MAPE."""
        logger.info(f"Training on {len(train_df)} records... (full_train={full_train})")

        if full_train or self.predictor is None:
            if self.predictor is None:
                self.predictor = InsurancePremiumPredictor(self.staging_filters, self.staging_models, self.predictor_state_filename)
            else:
                self.predictor.reinit(self.staging_filters, self.staging_models)
            self.predictor.fit(train_df, val_df, verbose=True)
        else:
            self.predictor.reinit(self.staging_filters, self.staging_models)
            self.predictor.retrain(train_df, val_df)

        self.predictor.save()

        logger.info(f"Training complete. Version: {self.predictor.get_version()}")

        logger.info(f"Training metrics. Version: {json.dumps(self.predictor.metrics, indent=2, default=self.predictor.numpy_converter)}")

    def swap_models(self) -> None:
        """Swap staging model to active and notify consumer."""
        logger.info("Swapping models...")

        version = self.predictor.get_version()

        shutil.rmtree(self.backup_models, ignore_errors=True)
        shutil.rmtree(self.backup_filters, ignore_errors=True)

        if self.active_models.exists():
            shutil.move(str(self.active_models), str(self.backup_models))
            shutil.move(str(self.active_filters), str(self.backup_filters))
        else:
            self.backup_models.mkdir(parents=True, exist_ok=True)
            self.backup_filters.mkdir(parents=True, exist_ok=True)

        shutil.move(str(self.staging_models), str(self.active_models))
        shutil.move(str(self.staging_filters), str(self.active_filters))

        self.predictor.change_directories(self.active_filters, self.active_models)

        self.staging_models.mkdir(parents=True, exist_ok=True)
        self.staging_filters.mkdir(parents=True, exist_ok=True)

        self.new_model_flag.touch()
        logger.info(f"Model swapped. Version: {version}")

    def run_once(self) -> None:
        logger.info("="*60)
        logger.info("Starting training cycle")
        logger.info("="*60)

        df = self.fetch_training_data()
        total_samples = len(df)
        new_samples = total_samples - self.last_seen_samples

        if self.predictor is None:
            if total_samples < Config.INITIAL_TRAINING_SAMPLES:
                logger.info(f"Need {Config.INITIAL_TRAINING_SAMPLES} total records for initial training. Have {total_samples}. Skipping.")
                return
        else:
            if new_samples < Config.MIN_TRAINING_SAMPLES:
                logger.info(f"Need {Config.MIN_TRAINING_SAMPLES} new records. Have {new_samples}. Skipping.")
                return

        train_df = df.iloc[:-Config.MIN_VALIDATION_SAMPLES]
        val_df = df.iloc[-Config.MIN_VALIDATION_SAMPLES:]

        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")

        # Check if we have an existing model
        has_model = self.predictor is not None and self.active_models.exists()

        if not has_model:
            # First time training - use fit (with gridsearch)
            logger.info("No existing model. Running full training (fit)...")
            self.train_model(train_df, val_df, full_train=True)
        else:
            current_mape = self.predictor.calculate_mape(val_df)
            drift_detected = self.check_drift(current_mape)

            if drift_detected == 'model':
                logger.warning("Model drift detected. Running full training (fit)...")
                self.train_model(train_df, val_df, full_train=True)
            elif drift_detected == 'data':
                logger.info("Data drift detected. Running incremental training (retrain)...")
                self.train_model(train_df, val_df, full_train=False)
            else:
                logger.info("No significant drift detected. Skipping training.")
                return
        self.swap_models()

        self.last_seen_samples = total_samples

    def run(self):
        logger.info("Model Trainer started.")
        logger.info(f"Min training samples: {Config.MIN_TRAINING_SAMPLES}")
        logger.info(f"Min validation samples: {Config.MIN_VALIDATION_SAMPLES}")
        logger.info(f"Active models: {self.active_models}")

        self.db.connect()

        if self.active_models.exists() and any(self.active_models.iterdir()):
            try:
                self.predictor = InsurancePremiumPredictor(self.active_filters, self.active_models, self.predictor_state_filename)
                self.predictor.load()
                logger.info(f"Loaded existing model. Version: {self.predictor.get_version()}")
            except Exception as e:
                logger.error(f"Could not load existing model: {e}")

        while True:
            try:
                self.run_once()
                logger.info("Waiting for new data...")
                time.sleep(60)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(60)

        self.db.close()


if __name__ == '__main__':
    ModelTrainer().run()
