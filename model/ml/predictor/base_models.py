import pandas as pd
import numpy as np
import warnings
import joblib
import json
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from ml.eda import CategoricalEncoder

import warnings


class BasePremiumModel:
    """Base class for premium forecasting models."""

    def __init__(self, models_path: Optional[Path] = None):
        self.models_path = Path(models_path) if models_path else None
        self.model_zero = None
        self.model_else = None
        self.metrics_zero = None
        self.metrics_else = None
        self.param_grid_zero = {}
        self.param_grid_else = {}
        self.exclude_columns = ['PREMIUM', 'OBJECT_ID', 'START_MNTH', 'PROD_YEAR']

    def _create_base_pipeline(self, name: str) -> Tuple[Pipeline, Dict]:
        raise NotImplementedError

    def fit(self, df_zero: pd.DataFrame, df_else: pd.DataFrame, verbose: bool = False):
        self.model_zero, self.metrics_zero = self._train_model(df_zero, 'INSR_ZERO', verbose)
        self.model_else, self.metrics_else = self._train_model(df_else, 'ELSE', verbose)
        return self

    def _train_model(self, df: pd.DataFrame, name: str, verbose: bool = False) -> Tuple[Optional[Pipeline], Dict]:
        if df.empty:
            return None, {'name': name, 'n_samples': 0, 'message': 'No data'}

        X = df.drop(columns=[c for c in self.exclude_columns if c in df.columns])
        y = df['PREMIUM']

        pipeline, grid = self._create_base_pipeline(name)

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=KFold(3, shuffle=True, random_state=42),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            refit=True,
            verbose=verbose
        )

        warnings.filterwarnings('ignore', message='.*no_silent_downcasting.*')
        grid_search.fit(X, y)

        metrics = {
            'name': name,
            'n_samples': len(X),
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
        }

        return grid_search.best_estimator_, metrics

    def retrain(self, df_zero: pd.DataFrame, df_else: pd.DataFrame):
        if not df_zero.empty:
            X_zero = df_zero.drop(columns=[c for c in self.exclude_columns if c in df_zero.columns])
            self.model_zero.fit(X_zero, df_zero['PREMIUM'])
        if not df_else.empty:
            X_else = df_else.drop(columns=[c for c in self.exclude_columns if c in df_else.columns])
            self.model_else.fit(X_else, df_else['PREMIUM'])
        return self

    def predict(self, df_zero: pd.DataFrame, df_else: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        pred_zero = self.model_zero.predict(df_zero.drop(columns=[c for c in self.exclude_columns if c in df_zero.columns])) if len(df_zero) > 0 else np.array([])
        pred_else = self.model_else.predict(df_else.drop(columns=[c for c in self.exclude_columns if c in df_else.columns])) if len(df_else) > 0 else np.array([])
        return pred_zero, pred_else

    def save(self, path: Optional[Path] = None) -> None:
        save_path = Path(path) if path else self.models_path
        if save_path is None:
            raise ValueError("No save path provided.")

        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model_zero, save_path / f'{self.__class__.__name__.lower()}_INSR_ZERO.pkl')
        joblib.dump(self.model_else, save_path / f'{self.__class__.__name__.lower()}_ELSE.pkl')

        metadata = {
            'metrics_zero': self._serialize_metrics(self.metrics_zero),
            'metrics_else': self._serialize_metrics(self.metrics_else)
        }

        with open(save_path / f'{self.__class__.__name__.lower()}_model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: Optional[Path] = None):
        load_path = Path(path) if path else self.models_path
        if load_path is None:
            raise ValueError("No load path provided.")

        self.model_zero = joblib.load(load_path / f'{self.__class__.__name__.lower()}_INSR_ZERO.pkl')
        self.model_else = joblib.load(load_path / f'{self.__class__.__name__.lower()}_ELSE.pkl')

        metadata_path = load_path / f'{self.__class__.__name__.lower()}_model_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.metrics_zero = metadata.get('metrics_zero')
            self.metrics_else = metadata.get('metrics_else')

        return self

    def _serialize_metrics(self, metrics: Dict) -> Dict:
        if metrics is None:
            return None

        serialized = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                serialized[k] = self._serialize_metrics(v)
            elif isinstance(v, (np.float32, np.float64)):
                serialized[k] = float(v)
            elif isinstance(v, np.int64):
                serialized[k] = int(v)
            else:
                serialized[k] = v
        return serialized

    def get_metrics(self) -> Dict[str, Any]:
        return {
            'INSR_ZERO': self.metrics_zero,
            'ELSE': self.metrics_else
        }


class GradientBoostingPremiumModel(BasePremiumModel):
    def __init__(self, models_path: Optional[Path] = None):
        super().__init__(models_path)

        self.param_grid_zero = {
            'regressor__n_estimators': [200],
            'regressor__max_depth': [3, 5],
            'regressor__learning_rate': [0.05],
            'regressor__subsample': [0.8],
            'regressor__min_samples_split': [2, 5],
        }

        self.param_grid_else = {
            'regressor__n_estimators': [500, 700],
            'regressor__max_depth': [7, 9],
            'regressor__learning_rate': [0.05],
            'regressor__subsample': [0.7],
            'regressor__min_samples_split': [10, 20],
            'regressor__min_samples_leaf': [5],
            'regressor__max_features': ['log2']
        }

    def _create_base_pipeline(self, name: str) -> Pipeline:
        grid = self.param_grid_else if name == 'ELSE' else self.param_grid_zero
        return Pipeline([
            ('categorical_encoder', CategoricalEncoder()),
            ('preprocessor', StandardScaler()),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ]), grid


class NeuralNetworkPremiumModel(BasePremiumModel):
    def __init__(self, models_path: Optional[Path] = None):
        super().__init__(models_path)
        self.param_grid_zero = {
            'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25)],
            'regressor__activation': ['relu'],
            'regressor__alpha': [0.01, 0.05],
            'regressor__learning_rate_init': [0.01],
            'regressor__max_iter': [1500]
        }

        self.param_grid_else = {
            'regressor__hidden_layer_sizes': [(100,), (150, 100), (100, 50, 25)],
            'regressor__activation': ['relu'],
            'regressor__alpha': [0.01, 0.05],
            'regressor__learning_rate_init': [0.01],
            'regressor__max_iter': [1500]
        }

    def _create_base_pipeline(self, name: str) -> Pipeline:
        grid = self.param_grid_else if name == 'ELSE' else self.param_grid_zero
        return Pipeline([
            ('categorical_encoder', CategoricalEncoder()),
            ('preprocessor', StandardScaler()),
            ('regressor', MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1))
        ]), grid
