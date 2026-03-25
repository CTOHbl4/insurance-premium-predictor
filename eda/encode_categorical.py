import pandas as pd
import category_encoders as ce
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Categorical encoder for insurance data following sklearn transformer protocol.

    Encodings:
        - MAKE: Frequency encoding
        - SEX: Label encoding
        - INSR_TYPE: One-hot encoding
        - TYPE_VEHICLE: Target encoding (mean with smoothing)
        - USAGE: Target encoding (mean with smoothing)
    """

    def __init__(self):
        self.make_encoder = None
        self.sex_encoder = None
        self.insr_encoder = None
        self.type_encoder = None
        self.usage_encoder = None

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self._feature_names_out = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """
        Fit encoders on training data.

        Parameters
        ----------
        X : pd.DataFrame
            Input dataframe with categorical columns
        y : pd.Series, optional
            Target variable (required for target encoding)

        Returns
        -------
        self : CategoricalEncoder
            Fitted encoder
        """
        if y is None:
            raise ValueError("y (target) is required for target encoding")

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(X.columns)

        self.make_encoder = ce.CountEncoder(cols=['MAKE'], normalize=True)
        self.make_encoder.fit(X['MAKE'])

        self.sex_encoder = ce.OrdinalEncoder(cols=['SEX'])
        self.sex_encoder.fit(X['SEX'])

        self.insr_encoder = ce.OneHotEncoder(
            cols=['INSR_TYPE'],
            use_cat_names=True,
            handle_unknown='ignore'
        )
        self.insr_encoder.fit(X[['INSR_TYPE']])

        self.type_encoder = ce.TargetEncoder(cols=['TYPE_VEHICLE'])
        self.type_encoder.fit(X[['TYPE_VEHICLE']], y)

        self.usage_encoder = ce.TargetEncoder(cols=['USAGE'])
        self.usage_encoder.fit(X[['USAGE']], y)

        self._generate_feature_names()
        return self

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Transform dataframe using fitted encoders.
        """
        if self.make_encoder is None:
            raise ValueError("Encoder not fitted. Call fit() first.")

        X_encoded = X.copy()

        X_encoded['MAKE_FREQ'] = self.make_encoder.transform(X_encoded['MAKE']).fillna(0)
        X_encoded['SEX_LABEL'] = self.sex_encoder.transform(X_encoded['SEX'])

        insr_encoded = self.insr_encoder.transform(X_encoded[['INSR_TYPE']])
        X_encoded = pd.concat([X_encoded, insr_encoded], axis=1)

        type_encoded = self.type_encoder.transform(X_encoded[['TYPE_VEHICLE']])
        X_encoded['TYPE_VEHICLE_TARGET'] = type_encoded.fillna(0)

        usage_encoded = self.usage_encoder.transform(X_encoded[['USAGE']])
        X_encoded['USAGE_TARGET'] = usage_encoded.fillna(0)

        drop_cols = ['MAKE', 'SEX', 'INSR_TYPE', 'TYPE_VEHICLE', 'USAGE']
        X_encoded = X_encoded.drop(columns=[c for c in drop_cols if c in X_encoded.columns])

        return X_encoded

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _generate_feature_names(self):
        if self.insr_encoder is not None:
            insr_cols = list(self.insr_encoder.get_feature_names_out())
            self._feature_names_out = [
                'MAKE_FREQ',
                'SEX_LABEL',
                'TYPE_VEHICLE_TARGET',
                'USAGE_TARGET'
            ] + insr_cols

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        if self._feature_names_out is None:
            raise ValueError("Encoder not fitted. Call fit() first.")
        return self._feature_names_out
