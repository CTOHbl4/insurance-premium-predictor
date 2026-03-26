import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from eda import InsuranceDataProcessor
from predictor.base_models import GradientBoostingPremiumModel, NeuralNetworkPremiumModel


class InsurancePremiumPredictor:
    """
    Main predictor class that orchestrates both initial and consequent models.

    Handles:
        - Data preprocessing
        - First occurrence prediction with Gradient Boosting
        - Subsequent occurrence prediction with Neural Network with state tracking
    """

    def __init__(self, configs_path: Path, models_path: Path):
        """
        Initialize predictor with configs and models path.

        Args:
            configs_path: Path to preprocessing configs
            models_path: Path to save/load models
        """
        self.configs_path = Path(configs_path)
        self.models_path = Path(models_path)

        self.processor = InsuranceDataProcessor(self.configs_path)
        self.initial_model = GradientBoostingPremiumModel(self.models_path)
        self.consequent_model = NeuralNetworkPremiumModel(self.models_path)

        self.is_fitted = False
        self.state = {}  # key: (OBJECT_ID, is_zero) -> {'PREVIOUS_PREMIUM': float, 'TOTAL_DURATION': float}

    def _prepare_consequent_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare consequent data with previous premium and total duration."""
        if len(df) == 0:
            return df
        df = df.sort_values(['OBJECT_ID', 'START_MNTH'])
        df['PREVIOUS_PREMIUM'] = df.groupby('OBJECT_ID')['PREMIUM'].shift(1)
        df['TOTAL_DURATION'] = df.groupby('OBJECT_ID')['DURATION'].cumsum()
        return df.dropna(subset=['PREVIOUS_PREMIUM'])

    def fit(self, df: pd.DataFrame, verbous: bool = False) -> 'InsurancePremiumPredictor':
        """
        Fit both models on raw training data (2011-2014).

        Args:
            df: Raw input dataframe

        Returns:
            self
        """
        self.processor.fit(df)
        df_zero, df_else, _ = self.processor.transform(df, train=True)

        df_zero_init = df_zero[~df_zero.duplicated(subset=['OBJECT_ID'], keep='first')].copy()
        df_else_init = df_else[~df_else.duplicated(subset=['OBJECT_ID'], keep='first')].copy()

        df_zero_conseq = df_zero[df_zero.duplicated(subset=['OBJECT_ID'], keep='first')].copy()
        df_else_conseq = df_else[df_else.duplicated(subset=['OBJECT_ID'], keep='first')].copy()

        df_zero_conseq = self._prepare_consequent_data(df_zero_conseq)
        df_else_conseq = self._prepare_consequent_data(df_else_conseq)

        self.initial_model.fit(df_zero_init, df_else_init, verbous)
        self.consequent_model.fit(df_zero_conseq, df_else_conseq, verbous)

        self._store_state(df_zero, df_else)

        self.is_fitted = True
        return self

    def _store_state(self, df_zero: pd.DataFrame, df_else: pd.DataFrame) -> None:
        """Store last premium and total duration for each OBJECT_ID from training data."""
        for obj_id in df_zero['OBJECT_ID'].unique():
            obj_data = df_zero[df_zero['OBJECT_ID'] == obj_id].sort_values('START_MNTH')
            if len(obj_data) > 0:
                self.state[(obj_id, True)] = {
                    'PREVIOUS_PREMIUM': obj_data.iloc[-1]['PREMIUM'],
                    'TOTAL_DURATION': obj_data['DURATION'].sum()
                }
        for obj_id in df_else['OBJECT_ID'].unique():
            obj_data = df_else[df_else['OBJECT_ID'] == obj_id].sort_values('START_MNTH')
            if len(obj_data) > 0:
                self.state[(obj_id, False)] = {
                    'PREVIOUS_PREMIUM': obj_data.iloc[-1]['PREMIUM'],
                    'TOTAL_DURATION': obj_data['DURATION'].sum()
                }

    def update_state(self, df: pd.DataFrame) -> None:
        """
        Update state with new data that has PREMIUM column.

        Args:
            df: Raw input dataframe with PREMIUM column
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        df_zero, df_else, _ = self.processor.transform(df, train=False)

        for _, row in df_zero.iterrows():
            key = (row['OBJECT_ID'], True)
            self.state[key] = {
                'PREVIOUS_PREMIUM': row['PREMIUM'],
                'TOTAL_DURATION': row['DURATION']
            }

        for _, row in df_else.iterrows():
            key = (row['OBJECT_ID'], False)
            self.state[key] = {
                'PREVIOUS_PREMIUM': row['PREMIUM'],
                'TOTAL_DURATION': row['DURATION']
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

        df_initial = df[~in_state]
        df_consequent = df[in_state]

        if len(df_consequent) > 0:
            df_consequent['PREVIOUS_PREMIUM'] = df_consequent.apply(
                lambda row: self.state[(row['OBJECT_ID'], is_zero)]['PREVIOUS_PREMIUM'],
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
                            'PREVIOUS_PREMIUM': row['prediction'],
                            'TOTAL_DURATION': row['DURATION']
                        }
                    else:
                        self.state[key]['PREVIOUS_PREMIUM'] = row['prediction']
                        self.state[key]['TOTAL_DURATION'] += row['DURATION']

        result = pd.concat(all_predictions).sort_index()
        return result['prediction'].values

    def save(self) -> None:
        """Save both models and state."""
        self.initial_model.save()
        self.consequent_model.save()

        state_serializable = {
            f"{k[0]}_{k[1]}": v for k, v in self.state.items()
        }
        with open(self.models_path / 'predictor_state.json', 'w') as f:
            json.dump(state_serializable, f)

    def load(self) -> 'InsurancePremiumPredictor':
        """Load both models and state."""
        self.initial_model.load()
        self.consequent_model.load()

        state_path = self.models_path / 'predictor_state.json'
        if state_path.exists():
            with open(state_path, 'r') as f:
                state_serializable = json.load(f)
            self.state = {
                (int(k.split('_')[0]), k.split('_')[1] == 'True'): v
                for k, v in state_serializable.items()
            }

        self.is_fitted = True
        return self
