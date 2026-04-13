import mysql.connector
import pandas as pd
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy import create_engine
from services.config import Config


class DatabaseManager:
    def __init__(self):
        self.config = {
            'host': Config.MYSQL_HOST,
            'port': Config.MYSQL_PORT,
            'database': Config.MYSQL_DATABASE,
            'user': Config.MYSQL_USER,
            'password': Config.MYSQL_PASSWORD
        }
        self.conn = None
        self.engine = None

    def connect(self):
        self.engine = create_engine(
            f"mysql+mysqlconnector://{self.config['user']}:{self.config['password']}@"
            f"{self.config['host']}:{self.config['port']}/{self.config['database']}"
        )

        self.conn = mysql.connector.connect(**self.config)
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()

    def insert_batch(self, df: pd.DataFrame, source_topic: str,
                     model_version: str = None,
                     predictions: List[float] = None) -> int:

        df_result = df.copy()

        if predictions is not None:
            df_result['PREDICTED_PREMIUM'] = predictions
        else:
            df_result['PREDICTED_PREMIUM'] = None

        df_result['source_topic'] = source_topic
        df_result['model_version'] = model_version
        df_result['ingestion_time'] = datetime.now()

        df_result.to_sql(
            'insurance_records',
            self.engine,
            if_exists='append',
            index=False,
            method='multi'
        )

        return len(df_result)

    def get_training_data(self, limit: int = None) -> pd.DataFrame:
        """Fetch training data (records with PREMIUM not NULL)."""
        query = """
            SELECT SEX, INSR_BEGIN, INSR_END, EFFECTIVE_YR, INSR_TYPE,
                 INSURED_VALUE, OBJECT_ID, PROD_YEAR, SEATS_NUM,
                 CARRYING_CAPACITY, TYPE_VEHICLE, CCM_TON, MAKE, `USAGE`,
                 CLAIM_PAID, PREMIUM FROM insurance_records
            WHERE PREMIUM IS NOT NULL
            ORDER BY ingestion_time ASC
        """
        if limit:
            query += f" LIMIT {limit}"

        df = pd.read_sql(query, self.engine)
        df['OBJECT_ID'] = pd.to_numeric(df['OBJECT_ID'], errors='coerce').astype(int)

        return df

    def get_training_count(self) -> int:
        """Get total number of training records."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM insurance_records WHERE PREMIUM IS NOT NULL")
        count = cursor.fetchone()[0]
        cursor.close()
        return count

    def save_quality_metric(self, name: str, value: float, details: Dict = None) -> None:
        """Save quality metric for dashboard."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO quality_metrics (metric_name, metric_value, metric_details, recorded_time)
            VALUES (%s, %s, %s, NOW())
        """, (name, value, json.dumps(details) if details else None))
        self.conn.commit()
        cursor.close()
