from pathlib import Path
from typing import Optional
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from eda import CategoricalEncoder
from .base_model import BasePremiumModel


class NeuralNetworkPremiumModel(BasePremiumModel):
    def __init__(self, models_path: Optional[Path] = None):
        super().__init__(models_path)
        self.param_grid = {
            'regressor__hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50)],
            'regressor__activation': ['relu', 'tanh'],
            'regressor__alpha': [0.0001, 0.001, 0.01],
            'regressor__learning_rate_init': [0.001, 0.01, 0.1],
            'regressor__max_iter': [500, 1000]
        }

    def _create_base_pipeline(self) -> Pipeline:
        return Pipeline([
            ('categorical_encoder', CategoricalEncoder()),
            ('preprocessor', StandardScaler()),
            ('regressor', MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1))
        ])
