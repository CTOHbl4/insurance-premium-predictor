import json
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from kafka import KafkaConsumer
import warnings

sys.path.append('/app')
from ml import InsurancePremiumPredictor
from services.config import Config
from services.mysql import DatabaseManager

warnings.filterwarnings('ignore', message='.*no_silent_downcasting.*')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InsuranceKafkaConsumer:
    def __init__(self):
        self.consumer = None
        self.db = DatabaseManager()
        self.predictor = None
        self.train_buffer = []
        self.validate_buffer = []
        self.active_model_path = Path(Config.ACTIVE_MODELS_PATH)
        self.active_filters_path = Path(Config.ACTIVE_FILTERS_PATH)
        self.predictor_state_filename = Config.PREDICTOR_STATE_FILENAME
        self.new_model_flag = self.active_model_path / Config.FLAG_FILENAME
        self.batch_size = Config.BATCH_SIZE

    def init_consumer(self):
        self.consumer = KafkaConsumer(
            Config.TRAIN_TOPIC,
            Config.VALIDATE_TOPIC,
            bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=Config.GROUP_ID,
            auto_offset_reset='earliest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logger.info(f"Connected to Kafka at {Config.KAFKA_BOOTSTRAP_SERVERS}")

    def load_active_model(self):
        if self.active_model_path.exists():
            try:
                predictor = InsurancePremiumPredictor(
                    self.active_filters_path,
                    self.active_model_path,
                    self.predictor_state_filename
                )
                predictor.load()
                logger.info(f"Loaded active model: {predictor.get_version()}")
                self.predictor = predictor
                return True
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                return False
        else:
            logger.warning(f"No active model found at {self.active_model_path}. Waiting...")
            return False

    def check_and_update_model(self):
        if self.new_model_flag.exists():
            logger.info("New model detected! Reloading...")
            if self.load_active_model():
                self.new_model_flag.unlink()
                logger.info("Model updated successfully")
            else:
                logger.error("Failed to load new model, keeping old one")

    def flush_validate_buffer(self):
        if not self.validate_buffer:
            return

        if self.predictor is None:
            logger.warning("Model not ready. Skipping batch prediction.")
            self.validate_buffer = []
            return

        try:
            df = pd.concat(self.validate_buffer, ignore_index=True)

            predictions = self.predictor.predict(df)

            inserted = self.db.insert_batch(
                df,
                source_topic='validate',
                model_version=self.predictor.get_version(),
                predictions=predictions
            )
            logger.info(f"Saved {inserted} predictions from batch")

        except Exception as e:
            logger.error(f"Error processing validate batch: {e}")
        finally:
            self.validate_buffer = []

    def flush_train_buffer(self):
        if not self.train_buffer:
            return
        try:
            df = pd.concat(self.train_buffer, ignore_index=True)

            inserted = self.db.insert_batch(df, source_topic='train')
            logger.info(f"Saved {inserted} training records from batch")

        except Exception as e:
            logger.error(f"Error processing train batch: {e}")
        finally:
            self.train_buffer = []

    def process_train_message(self, message):
        try:
            df = pd.DataFrame([message.value])
            self.train_buffer.append(df)

            if len(self.train_buffer) >= self.batch_size:
                self.flush_train_buffer()

        except Exception as e:
            logger.error(f"Error buffering train message: {e}")

    def process_validate_message(self, message):
        try:
            df = pd.DataFrame([message.value])
            self.validate_buffer.append(df)

            if len(self.validate_buffer) >= self.batch_size:
                self.flush_validate_buffer()

        except Exception as e:
            logger.error(f"Error buffering validate message: {e}")

    def run(self):
        logger.info("Starting Kafka consumer...")
        logger.info(f"Batch size: {self.batch_size}")
        self.init_consumer()
        self.db.connect()

        self.load_active_model()

        try:
            for message in self.consumer:
                self.check_and_update_model()

                if message.topic == Config.TRAIN_TOPIC:
                    self.process_train_message(message)
                elif message.topic == Config.VALIDATE_TOPIC:
                    self.process_validate_message(message)
                else:
                    logger.warning(f"Unknown topic: {message.topic}")

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self.flush_validate_buffer()
            self.flush_train_buffer()

            if self.consumer:
                self.consumer.close()
            self.db.close()
            logger.info("Consumer stopped.")


if __name__ == '__main__':
    InsuranceKafkaConsumer().run()
