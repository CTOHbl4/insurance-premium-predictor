import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    TRAIN_TOPIC = os.getenv('TRAIN_TOPIC', 'insurance_train')
    VALIDATE_TOPIC = os.getenv('VALIDATE_TOPIC', 'insurance_validate')
    GROUP_ID = os.getenv('GROUP_ID', 'insurance_consumer_group')

    MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'insurance_db')
    MYSQL_USER = os.getenv('MYSQL_USER', 'insurance_user')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'insurance_pass')

    STAGING_MODELS_PATH = os.getenv('STAGING_MODELS_PATH', '/app/ml/checkpoints/staging/models')
    ACTIVE_MODELS_PATH = os.getenv('ACTIVE_MODELS_PATH', '/app/ml/checkpoints/active/models')
    BACKUP_MODELS_PATH = os.getenv('BACKUP_MODELS_PATH', '/app/ml/checkpoints/backup/models')
    STAGING_FILTERS_PATH = os.getenv('STAGING_FILTERS_PATH', '/app/ml/checkpoints/staging/filters')
    ACTIVE_FILTERS_PATH = os.getenv('ACTIVE_FILTERS_PATH', '/app/ml/checkpoints/active/filters')
    BACKUP_FILTERS_PATH = os.getenv('BACKUP_FILTERS_PATH', '/app/ml/checkpoints/backup/filters')
    PREDICTOR_STATE_FILENAME = os.getenv('PREDICTOR_STATE_FILENAME', 'predictor_state.json')
    FLAG_FILENAME = os.getenv('FLAG_FILENAME', 'NEW_MODEL.flag')

    MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', 10000))
    INITIAL_TRAINING_SAMPLES = int(os.getenv('INITIAL_TRAINING_SAMPLES', 200000))
    MIN_VALIDATION_SAMPLES = int(os.getenv('MIN_VALIDATION_SAMPLES', 5000))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 100))
