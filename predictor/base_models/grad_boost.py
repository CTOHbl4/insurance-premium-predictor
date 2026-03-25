from pathlib import Path
from typing import Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from eda import CategoricalEncoder
from .base_model import BasePremiumModel

class GradientBoostingPremiumModel(BasePremiumModel):
    def __init__(self, models_path: Optional[Path] = None):
        super().__init__(models_path)
        self.param_grid = {
            'regressor__n_estimators': [50, 100],
            'regressor__max_depth': [5, 7],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__subsample': [0.8]
        }

    def _create_base_pipeline(self) -> Pipeline:
        return Pipeline([
            ('categorical_encoder', CategoricalEncoder()),
            ('preprocessor', StandardScaler()),
            ('regressor', GradientBoostingRegressor(random_state=42))
        ])
