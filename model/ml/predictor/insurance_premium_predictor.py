import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any

from ml.eda import InsuranceDataProcessor
from ml.predictor.base_models import GradientBoostingPremiumModel, NeuralNetworkPremiumModel


class InsurancePremiumPredictor:
    """
    Main predictor class that orchestrates both initial and consequent models.

    Handles:
        - Data preprocessing
        - First occurrence prediction with Gradient Boosting
        - Subsequent occurrence prediction with Neural Network with state tracking
    """

    def __init__(self, configs_path: Path, models_path: Path, predictor_state_filename: str):
        """
        Initialize predictor with configs and models path.

        Args:
            configs_path: Path to preprocessing configs
            models_path: Path to save/load models
        """
        self.initial_model = GradientBoostingPremiumModel(models_path)
        self.consequent_model = NeuralNetworkPremiumModel(models_path)
        self.reinit(configs_path, models_path)
        self.is_fitted = False
        self.metrics = None
        self.fit_num = 0
        self.retrain_num = 0
        self.predictor_state_filename = predictor_state_filename

    def numpy_converter(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError

    def reinit(self, configs_path: Path, models_path: Path):
        self.change_directories(configs_path, models_path)
        # key: (OBJECT_ID, is_zero) -> {'TOTAL_PREMIUM': float, 'TOTAL_DURATION': float}
        self.state = {}

    def change_directories(self, configs_path: Path, models_path: Path):
        self.configs_path = Path(configs_path)
        self.models_path = Path(models_path)

        self.processor = InsuranceDataProcessor(self.configs_path)
        self.initial_model.models_path = self.models_path
        self.consequent_model.models_path = self.models_path

    def _prepare_consequent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare consequent data with previous premium and total duration."""
        if len(df) == 0:
            return df
        df = df.sort_values(['OBJECT_ID', 'START_MNTH'])
        df['TOTAL_PREMIUM'] = df.groupby('OBJECT_ID')['PREMIUM'].cumsum() - df['PREMIUM']
        df['TOTAL_DURATION'] = df.groupby('OBJECT_ID')['DURATION'].cumsum()
        return df

    def calculate_mape(self, df: pd.DataFrame) -> float:
        if not self.is_fitted or len(df) == 0:
            return float('inf')

        predictions = self.predict(df)
        actuals = df['PREMIUM'].values
        mask = actuals > 0

        if mask.sum() == 0:
            return float('inf')

        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
        return mape

    def _fit(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.processor.fit(df)
        df_zero, df_else, data_metrics = self.processor.transform(df, train=True)

        df_zero_init = df_zero[~df_zero.duplicated(subset=['OBJECT_ID'], keep='first')].copy()
        df_else_init = df_else[~df_else.duplicated(subset=['OBJECT_ID'], keep='first')].copy()

        df_zero_conseq = df_zero[df_zero.duplicated(subset=['OBJECT_ID'], keep='first')].copy()
        df_else_conseq = df_else[df_else.duplicated(subset=['OBJECT_ID'], keep='first')].copy()

        df_zero_conseq = self._prepare_consequent_data(df_zero_conseq)
        df_else_conseq = self._prepare_consequent_data(df_else_conseq)

        data_metrics['train_length'] = {}
        data_metrics['train_length']['initial'] = {}
        data_metrics['train_length']['consequent'] = {}
        data_metrics['train_length']['initial']['INSR_ZERO'] = len(df_zero_init)
        data_metrics['train_length']['initial']['ELSE'] = len(df_else_init)
        data_metrics['train_length']['consequent']['INSR_ZERO'] = len(df_zero_conseq)
        data_metrics['train_length']['consequent']['ELSE'] = len(df_else_conseq)

        self._store_state(df_zero, df_else)
        return df_zero_init, df_else_init, df_zero_conseq, df_else_conseq, data_metrics

    def fit(self, df_train: pd.DataFrame, df_val: pd.DataFrame, verbose: bool = False) -> 'InsurancePremiumPredictor':
        df_zero_init, df_else_init, df_zero_conseq, df_else_conseq, data_metrics = self._fit(df_train)

        self.initial_model.fit(df_zero_init, df_else_init, verbose)
        self.consequent_model.fit(df_zero_conseq, df_else_conseq, verbose)

        if self.is_fitted:
            self.fit_num += 1
        self.is_fitted = True
        self.metrics = {
            'validation_mape': self.calculate_mape(df_val),
            'data_metrics': data_metrics,
            'initial_model': self.initial_model.get_metrics(),
            'consequent_model': self.consequent_model.get_metrics()
        }
        return self

    def retrain(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> 'InsurancePremiumPredictor':
        df_zero_init, df_else_init, df_zero_conseq, df_else_conseq, data_metrics = self._fit(df_train)

        self.initial_model.retrain(df_zero_init, df_else_init)
        self.consequent_model.retrain(df_zero_conseq, df_else_conseq)
        self.metrics['data_metrics'] = data_metrics

        if self.is_fitted:
            self.retrain_num += 1
        self.is_fitted = True
        self.metrics['validation_mape'] = self.calculate_mape(df_val)
        return self

    def _store_state(self, df_zero: pd.DataFrame, df_else: pd.DataFrame) -> None:
        for df_part, is_zero in [(df_zero, True), (df_else, False)]:
            for obj_id, group in df_part.groupby('OBJECT_ID'):
                group_sorted = group.sort_values('START_MNTH')
                self.state[(obj_id, is_zero)] = {
                    'TOTAL_PREMIUM': group_sorted['PREMIUM'].sum(),
                    'TOTAL_DURATION': group_sorted['DURATION'].sum()
                }

    def _split_by_state(self, df: pd.DataFrame, is_zero: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataframe into initial and consequent based on state dictionary.

        Args:
            df: Dataframe with 'OBJECT_ID' column
            is_zero: Whether this is zero-insured data

        Returns:
            Tuple of (df_initial, df_consequent)
        """
        df = df.copy()

        in_state = df['OBJECT_ID'].apply(lambda obj_id: (obj_id, is_zero) in self.state)

        df_initial = df[~in_state].copy()
        df_consequent = df[in_state].copy()

        if len(df_consequent) > 0:
            df_consequent['TOTAL_PREMIUM'] = df_consequent.apply(
                lambda row: self.state[(row['OBJECT_ID'], is_zero)]['TOTAL_PREMIUM'],
                axis=1
            )
            df_consequent['TOTAL_DURATION'] = df_consequent.apply(
                lambda row: self.state[(row['OBJECT_ID'], is_zero)]['TOTAL_DURATION'] + row['DURATION'],
                axis=1
            )

        return df_initial, df_consequent

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict premiums for new data (sequential in time).

        Returns:
            Array of predictions in original order
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        df_zero, df_else, _ = self.processor.transform(df, train=False)

        df_zero_initial, df_zero_consequent = self._split_by_state(df_zero, True)
        df_else_initial, df_else_consequent = self._split_by_state(df_else, False)

        all_predictions = []

        for df_zero_part, df_else_part, model in [
            (df_zero_initial, df_else_initial, self.initial_model),
            (df_zero_consequent, df_else_consequent, self.consequent_model)
        ]:
            zero_pred, else_pred = model.predict(df_zero_part, df_else_part)

            if len(df_zero_part) > 0:
                df_zero_part['prediction'] = zero_pred
                all_predictions.append(df_zero_part[['prediction']])

            if len(df_else_part) > 0:
                df_else_part['prediction'] = else_pred
                all_predictions.append(df_else_part[['prediction']])

            is_initial = (model == self.initial_model)

            for is_zero, df_part in [(True, df_zero_part), (False, df_else_part)]:
                for _, row in df_part.iterrows():
                    key = (row['OBJECT_ID'], is_zero)
                    if is_initial:
                        self.state[key] = {
                            'TOTAL_PREMIUM': row['prediction'],
                            'TOTAL_DURATION': row['DURATION']
                        }
                    else:
                        self.state[key]['TOTAL_PREMIUM'] += row['prediction']
                        self.state[key]['TOTAL_DURATION'] += row['DURATION']

        result = pd.concat(all_predictions).sort_index()
        return result['prediction'].values

    def save(self) -> None:
        self.initial_model.save()
        self.consequent_model.save()

        state_serializable = {
            f"{k[0]}_{k[1]}": {
                'TOTAL_PREMIUM': float(v['TOTAL_PREMIUM']),
                'TOTAL_DURATION': int(v['TOTAL_DURATION'])
            } for k, v in self.state.items()
        }

        metadata = {
            'state': state_serializable,
            'fit_num': self.fit_num,
            'retrain_num': self.retrain_num,
            'metrics': self.metrics
        }

        with open(self.models_path / self.predictor_state_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=self.numpy_converter)

    def load(self) -> 'InsurancePremiumPredictor':
        self.initial_model.load()
        self.consequent_model.load()

        state_path = self.models_path / self.predictor_state_filename
        if state_path.exists():
            with open(state_path, 'r') as f:
                data = json.load(f)

            state_serializable = data['state']
            self.fit_num = data.get('fit_num', 0)
            self.retrain_num = data.get('retrain_num', 0)
            self.metrics = data.get('metrics', None)

            self.state = {
                (int(k.split('_')[0]), k.split('_')[1] == 'True'): v
                for k, v in state_serializable.items()
            }

        self.is_fitted = True
        return self

    def get_version(self):
        return f"{self.fit_num}:{self.retrain_num}"
