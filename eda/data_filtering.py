import pandas as pd
import json
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from scikit_posthocs import posthoc_dunn
from scipy.stats import mannwhitneyu
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np


class InsuranceDataProcessor:
    """
    Insurance data preprocessing pipeline for premium forecasting.

    Handles:
        - Date transformation and duration calculation
        - Missing value imputation
        - Outlier filtering
        - Make standardization
        - Association rule corrections
    """

    MONTH_MAP = {
        'JAN': 0, 'FEB': 1, 'MAR': 2, 'APR': 3, 'MAY': 4, 'JUN': 5,
        'JUL': 6, 'AUG': 7, 'SEP': 8, 'OCT': 9, 'NOV': 10, 'DEC': 11
    }

    CONFIG_NAMES = {
        'CCM_TON': 'ccm_ton_map.json',
        'MAKE': 'make_map.json',
        'PROD_YEAR': 'prod_year_map.json',
        'SEATS_NUM': 'seats_num_map.json',
        'INSURED_VALUE': 'insured_value_map.json',
        'INFERENCE': 'inference_nans_replacements.json',
        'OUTLIERS': 'filtering_config.json',
        'APRIORI': 'apriori_rules.json'
    }

    def __init__(
        self,
        configs_path: Path,
        filter_quantile: float = 0.99,
        posthoc_dunn: float = 0.9,
        mann_whitney_thresh: float = 0.5,
        apriori_min_support: float = 0.1,
        apriori_min_confidence: float = 0.9975
    ):
        """
        Initialize processor with configuration path and thresholds.

        Args:
            configs_path: Path to directory containing JSON config files
            filter_quantile: Quantile threshold for outlier filtering
            posthoc_dunn: Posthoc category mixing threshold
            mann_whitney_thresh: Mann-Whitney category mixing threshold
            apriori_min_support: Minimum support for Apriori algorithm
            apriori_min_confidence: Minimum confidence for association rules
            apriori_prefix_sep: Separator for prefix in one-hot encoding
        """
        self.configs_path = Path(configs_path)
        self._cache: Dict[str, Any] = {}

        self.filter_quantile = filter_quantile

        self.man_whitn_thresh = mann_whitney_thresh
        self.posthoc_thresh = posthoc_dunn

        self.apriori_min_support = apriori_min_support
        self.apriori_min_confidence = apriori_min_confidence
        self.apriori_prefix_sep = '_!_'
        self.is_trained = all(
            (self.configs_path / filename).exists()
            for filename in self.CONFIG_NAMES.values()
        )

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _transform_date(self, date_str: str) -> int:
        """Convert 'DD-MON-YYYY' to total months since year 0."""
        _, month, year = date_str.split('-')
        return int(year) * 12 + self.MONTH_MAP[month]

    def _load_config(self, filename: str) -> dict:
        """Load JSON configuration file with caching."""
        if filename not in self._cache:
            with open(self.configs_path / filename, 'r') as f:
                self._cache[filename] = json.load(f)
        return self._cache[filename]

    def _save_config(self, filename: str, data: dict) -> None:
        """Save configuration to JSON file."""
        file_path = self.configs_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _apply_make_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize MAKE column using mapping."""
        make_map = self._load_config(self.CONFIG_NAMES['MAKE'])
        df['MAKE'] = (
            df['MAKE']
            .fillna('')
            .astype(str)
            .str.upper()
            .str.replace(' ', '')
            .map(make_map)
            .fillna('UNKNOWN')
        )
        return df

    def _apply_filter(self, df: pd.DataFrame, filter_config: dict, filter_key: str) -> Tuple[pd.DataFrame, int]:
        """Apply a single filter configuration to dataframe."""
        rule = filter_config[filter_key]
        original_len = len(df)

        if filter_key.startswith('MIX_'):
            col_name = filter_key[4:]
            for old, new in rule:
                df[col_name] = df[col_name].replace(old, new)

        elif isinstance(rule, (int, float)):
            if filter_key == 'PREMIUM' and 'PREMIUM' not in df.columns:
                return df, 0
            df = df[df[filter_key] < rule]

        elif isinstance(rule, list):
            mask = False
            for threshold, vehicle_type in rule:
                mask |= (df[filter_key] < threshold) & (df['TYPE_VEHICLE'] == vehicle_type)
            df = df[mask]

        return df, original_len - len(df)

    def _find_similar_categories(self, df: pd.DataFrame, col: str) -> List[Tuple]:
        """Find statistically similar categories using Dunn's test and Mann-Whitney validation."""
        dunn_results = posthoc_dunn(df, val_col='PREMIUM', group_col=col, p_adjust='bonferroni')
        similar_pairs = []

        for i in range(len(dunn_results.index)):
            for j in range(i + 1, len(dunn_results.columns)):
                if dunn_results.iloc[i, j] > self.posthoc_thresh:
                    cat1, cat2 = dunn_results.index[i], dunn_results.columns[j]
                    group1 = df[df[col] == cat1]['PREMIUM']
                    group2 = df[df[col] == cat2]['PREMIUM']

                    if len(group1) > 0 and len(group2) > 0:
                        _, mw_p_val = mannwhitneyu(group1, group2)
                        if mw_p_val > self.man_whitn_thresh:
                            similar_pairs.append((cat1, cat2))

        return similar_pairs

    def _parse_rule_item(self, item: str) -> Dict[str, Any]:
        """Parse a single rule item into a dictionary."""
        col, val = item.split(self.apriori_prefix_sep)
        return {col: int(val) if col == 'INSR_TYPE' else val}

    def _parse_rule_items(self, items: set) -> Dict[str, Any]:
        """Parse a set of rule items into a dictionary."""
        result = {}
        for item in items:
            result.update(self._parse_rule_item(item))
        return result

    # ============================================================================
    # Processing Methods
    # ============================================================================

    def _handle_nans(self, df: pd.DataFrame, train: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """Internal method for NaN handling."""
        quality_metrics = {}
        df = df.copy()

        df = df.dropna(subset=['OBJECT_ID'])

        # Calculate duration
        df['START_MNTH'] = df['INSR_BEGIN'].apply(self._transform_date)
        df['END_MNTH'] = df['INSR_END'].apply(self._transform_date)
        df['DURATION'] = df['END_MNTH'] - df['START_MNTH']

        df = df.sort_values(['OBJECT_ID', 'START_MNTH', 'DURATION'])

        if train:
            original_len = len(df)
            df = df[df['DURATION'] > 0]
            quality_metrics['INSR_ZERO_duration_removed'] = original_len - len(df)

            original_len = len(df)
            df = df.drop_duplicates(
                subset=['OBJECT_ID', 'START_MNTH', 'INSURED_VALUE', 'SEX', 'PREMIUM', 'INSR_TYPE'],
                keep='last',
                ignore_index=True
            )
            quality_metrics['duplicates_removed'] = original_len - len(df)

        # Drop unnecessary columns
        drop_cols = ['INSR_BEGIN', 'INSR_END', 'END_MNTH', 'EFFECTIVE_YR', 'CLAIM_PAID', 'CARRYING_CAPACITY']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Handle Trade plates
        trade_mask = df['TYPE_VEHICLE'] == 'Trade plates'
        if trade_mask.any():
            for col in ['CCM_TON', 'SEATS_NUM']:
                quality_metrics[f'trade_plates_{col}_nan'] = (trade_mask & df[col].isna()).sum()
            quality_metrics['trade_plates_prod_year_nan'] = (trade_mask & df['PROD_YEAR'].isna()).sum()

            df.loc[trade_mask, ['CCM_TON', 'SEATS_NUM']] = 0.0
            df.loc[trade_mask & df['PROD_YEAR'].isna(), 'PROD_YEAR'] = 2010

        # Drop premium NaNs during training
        if train:
            quality_metrics['premium_nan_count'] = df['PREMIUM'].isna().sum()
            df = df.dropna(subset=['PREMIUM'])

        # Apply make mapping
        df = self._apply_make_mapping(df)
        quality_metrics['unknown_makes'] = (df['MAKE'] == 'UNKNOWN').sum()

        return df, quality_metrics

    def handle_nans(self, df: pd.DataFrame, train: bool = False) -> Tuple[pd.DataFrame, Dict]:
        """
        Clean missing values and engineer duration feature.

        Args:
            df: Input dataframe
            train: If True, apply training-specific operations

        Returns:
            tuple: (processed_df, quality_metrics)
        """
        df, quality_metrics = self._handle_nans(df, train)

        # Impute missing values
        impute_configs = [
            ('CCM_TON', lambda x: x == 0, 'mean'),
            ('SEATS_NUM', pd.isna, 'mean'),
            ('PROD_YEAR', pd.isna, 'median'),
            ('INSURED_VALUE', pd.isna, 'mean')
        ]

        for col, condition, _ in impute_configs:
            impute_map = self._load_config(self.CONFIG_NAMES[col])
            mask = condition(df[col])
            quality_metrics[f'{col}_imputed'] = mask.sum()

            if mask.any():
                df.loc[mask, col] = df.loc[mask].apply(
                    lambda row: impute_map.get(f"{row['TYPE_VEHICLE']} {row['MAKE']}", row[col]),
                    axis=1
                )

        quality_metrics['remaining_nans'] = df.isna().sum().to_dict()

        if train:
            df = df.dropna()
        else:
            inference_config = self._load_config(self.CONFIG_NAMES['INFERENCE'])
            df = df.fillna(inference_config)

        df['OBJECT_AGE'] = df['START_MNTH'] - (df['PROD_YEAR'] - 2000) * 12

        return df, quality_metrics

    def filter_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Remove outliers and standardize categorical values."""
        quality_metrics = {'removed_by_filter': {}}
        df = df.copy()

        df_zero = df[df['INSURED_VALUE'] == 0]
        df_else = df[df['INSURED_VALUE'] != 0]

        quality_metrics['INSR_ZERO_insured_records'] = len(df_zero)
        quality_metrics['ELSE_insured_records'] = len(df_else)

        filter_config = self._load_config(self.CONFIG_NAMES['OUTLIERS'])

        for filter_key in filter_config['ELSE'].keys():
            df_else, removed_else = self._apply_filter(df_else, filter_config['ELSE'], filter_key)
            if removed_else > 0:
                quality_metrics['removed_by_filter'][f'ELSE_{filter_key}'] = removed_else

        for filter_key in filter_config['INSR_ZERO'].keys():
            df_zero, removed_zero = self._apply_filter(df_zero, filter_config['INSR_ZERO'], filter_key)
            if removed_zero > 0:
                quality_metrics['removed_by_filter'][f'INSR_ZERO_{filter_key}'] = removed_zero

        quality_metrics['final_INSR_ZERO_records'] = len(df_zero)
        quality_metrics['final_ELSE_records'] = len(df_else)

        return df_zero.drop(columns=['INSURED_VALUE']), df_else, quality_metrics

    def apply_apriori(self, df: pd.DataFrame, config_key: str) -> Tuple[pd.DataFrame, Dict]:
        """Apply association rule corrections."""
        rule_misses = {}
        rules = self._load_config(self.CONFIG_NAMES['APRIORI'])[config_key]
        df = df.copy()

        for i, rule in enumerate(rules):
            antecedents = rule['antecedents']
            consequent = rule['consequent']
            consequent_key = list(consequent.keys())[0]
            consequent_value = consequent[consequent_key]
            confidence = rule['confidence']

            mask = pd.Series(True, index=df.index)
            for col, val in antecedents.items():
                mask &= (df[col] == val)

            misses_mask = mask & (df[consequent_key] != consequent_value)
            misses_count = misses_mask.sum()

            if misses_count > 0:
                rule_misses[i] = misses_count
                if confidence >= self.apriori_min_confidence:
                    df.loc[misses_mask, consequent_key] = consequent_value

        return df, rule_misses

    def transform(self, df: pd.DataFrame, train: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Run complete preprocessing pipeline.

        Returns:
            tuple: (df_zero, df_else, all_metrics)
        """
        if (not self.is_trained) and train:
            self.fit(df)
        elif (not self.is_trained):
            raise ValueError('Processor is not trained')
        all_metrics = {}

        df, metrics_nans = self.handle_nans(df, train=train)
        all_metrics['nans'] = metrics_nans

        df_zero, df_else, metrics_outliers = self.filter_outliers(df)
        all_metrics['outliers'] = metrics_outliers

        df_zero, metrics_apriori_zero = self.apply_apriori(df_zero, 'INSR_ZERO')
        df_else, metrics_apriori_else = self.apply_apriori(df_else, 'ELSE')
        all_metrics['apriori'] = {'INSR_ZERO': metrics_apriori_zero, 'ELSE': metrics_apriori_else}

        return df_zero, df_else, all_metrics

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit preprocessing configs based on collected data.

        Args:
            df: Input dataframe
        """
        df, _ = self._handle_nans(df, train=True)

        # ========== 1. Train NaN imputation configs ==========
        impute_configs = [
            ('CCM_TON', lambda x: x == 0, 'mean'),
            ('SEATS_NUM', pd.isna, 'mean'),
            ('PROD_YEAR', pd.isna, 'median'),
            ('INSURED_VALUE', pd.isna, 'mean')
        ]

        for col, condition, agg_func in impute_configs:
            mask = condition(df[col])
            agg_map = df[~mask].groupby(['TYPE_VEHICLE', 'MAKE'])[col].agg(agg_func).to_dict()
            self._save_config(
                self.CONFIG_NAMES[col],
                {' '.join(k): v for k, v in agg_map.items()}
            )

        # ========== 2. Train inference config ==========
        medians = df.median(numeric_only=True).to_dict()
        del medians['OBJECT_ID']

        modes = df[['TYPE_VEHICLE', 'MAKE', 'USAGE']].mode().iloc[0].to_dict()
        self._save_config(
            self.CONFIG_NAMES['INFERENCE'],
            {**medians, **modes}
        )

        # ========== 3. Train outlier filtering configs ==========
        filtering_config = {'INSR_ZERO': {}, 'ELSE': {}}
        apriori_config = {'INSR_ZERO': [], 'ELSE': []}

        for df_part, insr_key in [(df[df['INSURED_VALUE'] == 0], 'INSR_ZERO'),
                                  (df[df['INSURED_VALUE'] != 0], 'ELSE')]:
            df_part = df_part

            filtering_config[insr_key]['PREMIUM'] = df_part['PREMIUM'].quantile(self.filter_quantile)
            if insr_key == 'ELSE':
                filtering_config[insr_key]['INSURED_VALUE'] = df_part['INSURED_VALUE'].quantile(self.filter_quantile)

            for col in ('SEATS_NUM', 'CCM_TON'):
                limits = [
                    (df_part[col][df_part['TYPE_VEHICLE'] == vt].quantile(self.filter_quantile), vt)
                    for vt in df_part['TYPE_VEHICLE'].unique()
                ]
                filtering_config[insr_key][col] = limits

            for col in ['USAGE', 'TYPE_VEHICLE', 'SEX']:
                similar_pairs = self._find_similar_categories(df_part, col)
                filtering_config[insr_key][f'MIX_{col}'] = similar_pairs

            # ========== 4. Train Apriori rules ==========
            categorical_cols = ['USAGE', 'TYPE_VEHICLE', 'MAKE', 'SEX', 'INSR_TYPE']
            df_encoded = df_part[categorical_cols].copy()
            df_encoded['INSR_TYPE'] = df_encoded['INSR_TYPE'].astype(str)
            df_encoded['SEX'] = df_encoded['SEX'].astype(str)

            df_binary = pd.get_dummies(df_encoded, prefix_sep=self.apriori_prefix_sep)

            frequent_itemsets = apriori(df_binary, min_support=self.apriori_min_support, use_colnames=True)
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.apriori_min_confidence
            )

            apriori_config[insr_key] = [
                {
                    'antecedents': self._parse_rule_items(ante),
                    'consequent': self._parse_rule_items(cons),
                    'confidence': conf
                }
                for ante, cons, conf in zip(
                    rules['antecedents'],
                    rules['consequents'],
                    rules['confidence']
                )
            ]

        self._save_config(self.CONFIG_NAMES['OUTLIERS'], filtering_config)
        self._save_config(self.CONFIG_NAMES['APRIORI'], apriori_config)

        self._cache.clear()

        return self
