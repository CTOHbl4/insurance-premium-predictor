# producer.py
import pandas as pd
import json
import time
from kafka import KafkaProducer

# Config
KAFKA_BOOTSTRAP = "192.168.1.114:9092"
CSV_PATH1 = "vehicle-insurance-data/motor_data11-14lats.csv"
CSV_PATH2 = "vehicle-insurance-data/motor_data14-2018.csv"

# Topics
TRAIN_TOPIC = "insurance_train"
VALIDATE_TOPIC = "insurance_validate"

MONTH_MAP = {
    'JAN': 0, 'FEB': 1, 'MAR': 2, 'APR': 3, 'MAY': 4, 'JUN': 5,
    'JUL': 6, 'AUG': 7, 'SEP': 8, 'OCT': 9, 'NOV': 10, 'DEC': 11
}


def _transform_date(date_str: str) -> int:
    _, month, year = date_str.split('-')
    return int(year) * 12 + MONTH_MAP[month]


def main():
    df1 = pd.read_csv(CSV_PATH1)
    df2 = pd.read_csv(CSV_PATH2)
    df = pd.concat([df1, df2])
    df['START_MNTH'] = df['INSR_BEGIN'].apply(_transform_date)
    df = df.sort_values('START_MNTH').reset_index(drop=True).drop(columns=['START_MNTH'])

    total = len(df)
    print(f"Loaded {total} records")

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8')
    )
    print(f"Connected to Kafka at {KAFKA_BOOTSTRAP}")

    def send_batch(topic, batch):
        for _, row in batch.iterrows():
            producer.send(topic, row.to_dict())
        producer.flush()
        print(f"Sent {len(batch)} to {topic}")

    # Initial for training
    n_training = 110_000  # 210_000
    send_batch(TRAIN_TOPIC, df.head(n_training))

    print("Waiting 7 minutes...")  # 15 for initial = 200000
    time.sleep(420)

    idx = n_training
    to_send = 1000
    while idx < total:
        end = min(idx + to_send, total)
        batch = df.iloc[idx:end].copy()

        send_batch(VALIDATE_TOPIC, batch.drop(columns=['PREMIUM']))
        time.sleep(5)
        send_batch(TRAIN_TOPIC, batch)

        idx = end
        time.sleep(10)

    print("\nDone!")
    producer.close()


if __name__ == "__main__":
    main()
