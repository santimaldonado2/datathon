import logging

from imblearn.base import BaseSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.constants import QUAD, NOT_TRANSFORMED_COLUMNS, LOG_TRANSFORMATION, COORDINATES, CADASTRAL_QUALITY, \
    CADASTRAL_QUALITY_ORDER, AREA, BUILDING_YEAR, B8, B4, SAVI_L, SAVI, PSSR, B2, EVI, EVI2, GEOMS
import pandas as pd
import math

logger = logging.getLogger(__name__)


class NumericTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys, transformation, add_log_value=0):
        self.keys = keys
        self.transformation = transformation
        self.add_log_value = add_log_value
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_df = pd.DataFrame()
        for key in self.keys:
            if self.transformation == QUAD:
                transformed_df['{}_{}'.format(QUAD, key)] = X[key] ** 2

            if self.transformation == LOG_TRANSFORMATION:
                transformed_df['{}_{}'.format(LOG_TRANSFORMATION, key)] = X[key].apply(
                    lambda x: math.log(x + self.add_log_value))

        self.feature_names = list(transformed_df.columns)
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class DataFrameIndexSelector(BaseEstimator, TransformerMixin):

    def __init__(self, keys):
        self.keys = keys
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_df = X[self.keys].copy()
        self.feature_names = transformed_df.columns
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class OneHotOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, keys, category_orders, drop_first=False):
        self.keys = keys
        self.category_orders = category_orders
        self.feature_names = []
        self.drop_first = drop_first

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_df = pd.DataFrame(index=X.index)
        for key, category_order in zip(self.keys, self.category_orders):
            if self.drop_first:
                category_order = category_order[1:]
            while category_order:
                transformed_df.loc[X[key].isin(category_order), '{}_{}'.format(key, category_order[0])] = 1
                category_order = category_order[1:]

            transformed_df.loc[X[key].isna(), '{}_Na'.format(key)] = 1
            transformed_df.fillna(0, inplace=True)
        self.feature_names = transformed_df.columns
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, keys, category_orders, na_value=-1):
        self.keys = keys
        self.category_orders = category_orders
        self.na_value = na_value
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def get_element_index(self, element, args):
        if element and element in args:
            return args.index(element) / len(args)
        else:
            return self.na_value

    def transform(self, X):
        transformed_df = pd.DataFrame()
        for key, category_order in zip(self.keys, self.category_orders):
            transformed_df['{}_ordinal'.format(key)] = X[key].apply(self.get_element_index, args=[category_order])
        self.feature_names = transformed_df.columns
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class IntervalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys, intervals):
        self.keys = keys
        self.intervals = intervals
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_df = pd.DataFrame(index=X.index)
        for key, interval_groups in zip(self.keys, self.intervals):
            prev_interval = None
            for interval in interval_groups:
                if not prev_interval:
                    transformed_df.loc[X[key] < interval, '{}_interval'.format(key)] = '<{}'.format(interval)
                else:
                    transformed_df.loc[
                        (X[key] >= prev_interval) & (X[key] < interval), '{}_interval'.format(key)] = str(
                        prev_interval)
                prev_interval = interval

            transformed_df.loc[X[key] >= prev_interval, '{}_interval'.format(key)] = '>={}'.format(prev_interval)

        self.feature_names = transformed_df.columns
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class IntervalCategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys, intervals, feat_name):
        self.keys = keys
        self.intervals = intervals
        self.feat_name = feat_name
        self.pipeline = None

    def fit(self, X, y=None):
        self.pipeline = Pipeline(steps=[
            ('interval', IntervalTransformer(self.keys, self.intervals)),
            ('onehot', OneHotEncoder(sparse=False))
        ])
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def get_feature_names(self):
        features = self.pipeline.named_steps['onehot'].categories_[0]
        return ['{}_{}'.format(self.feat_name, feat) for feat in features]


class AntiquityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys, current_year=2020):
        self.keys = keys
        self.current_year = current_year
        self.feature_names = []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        transformed_df = pd.DataFrame()
        for key in self.keys:
            transformed_df['{}_antiquity'.format(key)] = self.current_year - X[key]

        self.feature_names = transformed_df.columns
        return transformed_df

    def get_feature_names(self):
        return self.feature_names


class LogAntiquityTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys, add_log_value=1, current_year=2020):
        self.keys = keys
        self.transformation = LOG_TRANSFORMATION
        self.add_log_value = add_log_value
        self.current_year = current_year
        self.pipeline = None

    def fit(self, X, y=None):
        self.pipeline = Pipeline(steps=[
            ('antiquity', AntiquityTransformer(self.keys, self.current_year)),
            ('log', NumericTransformer(['{}_antiquity'.format(key) for key in self.keys], self.transformation,
                                       self.add_log_value))
        ])
        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

    def get_feature_names(self):
        return self.pipeline.named_steps['log'].get_feature_names()


class ImbalanceTransformer(BaseSampler):

    def __init__(self, min_samples, max_samples):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.under_sampler = None
        self.over_sampler = None
        self.feature_names = None
        self.dist_dict = None

    def _fit_resample(self, X, y):
        y_prov = pd.Series(y)
        dist_dict = y_prov.value_counts().to_dict()

        under_sampler_dict = {key: min(self.max_samples, dist_dict[key]) for key in dist_dict}
        over_sampler_dict = {key: max(self.min_samples, under_sampler_dict[key]) for key in under_sampler_dict}

        self.under_sampler = RandomUnderSampler(sampling_strategy=under_sampler_dict)
        X_und, y_und = self.under_sampler.fit_resample(X, y)

        self.over_sampler = SMOTE(sampling_strategy=over_sampler_dict)
        return self.over_sampler.fit_resample(X_und, y_und)

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

    def get_feature_names(self):
        return self.feature_names


class ScalerTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, keys):
        self.keys = keys
        self.pipeline = None

    def fit(self, X, y=None):
        self.pipeline = Pipeline([
            ('selector', DataFrameIndexSelector(self.keys)),
            ('scaler', StandardScaler())
        ])

        self.pipeline.fit(X, y)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_transformed = self.pipeline.transform(X_copy)
        X_copy[self.keys] = X_transformed
        return X_copy

    def get_feature_names(self):
        self.pipeline.named_steps['selector'].get_feature_names()


class SAVIITransformer(BaseEstimator, TransformerMixin):
    """Create Soil Adjusted Vegetarian Index
       https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/savi/script.js
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed[SAVI] = (X[B8] - X[B4]) / (X[B8] + X[B4] + SAVI_L) * (1.0 + SAVI_L)

        return X_transformed

    def get_feature_names(self):
        return [SAVI]


class PSSRITransformer(BaseEstimator, TransformerMixin):
    """Create Pigment Specific Simple Ratio Index
       https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/pssrb1/script.js
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed[PSSR] = (X[B8] / X[B4])

        return X_transformed

    def get_feature_names(self):
        return [PSSR]


class EVITransformer(BaseEstimator, TransformerMixin):
    """Create Enhanced Vegetation Index
       https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/evi/script.js
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed[EVI] = 2.5 * (X[B8] - X[B4]) / (X[B8] + 6. * X[B4] - 7.5 * X[B2])

        return X_transformed

    def get_feature_names(self):
        return [EVI]


class EVI2Transformer(BaseEstimator, TransformerMixin):
    """Create Enhanced Vegetation Index 2
       https://github.com/sentinel-hub/custom-scripts/blob/master/sentinel-2/evi2/script.js
    """

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_transformed = pd.DataFrame()
        X_transformed[EVI2] = 2.4 * (X[B8] - X[B4]) / (X[B8] + X[B4] + 1)

        return X_transformed

    def get_feature_names(self):
        return [EVI2]


FEATURES_BY_NAME = {
    'not_transformed': DataFrameIndexSelector(keys=NOT_TRANSFORMED_COLUMNS),
    'squared_coordinates': NumericTransformer(keys=COORDINATES, transformation=QUAD),
    'cadastral_ordinal_encoder_onehot': OneHotOrdinalEncoder(keys=[CADASTRAL_QUALITY],
                                                             category_orders=[CADASTRAL_QUALITY_ORDER]),
    # 'cadastral_ordinal_encoder': CustomOrdinalEncoder(keys=[CADASTRAL_QUALITY],
    #                                                   category_orders=[CADASTRAL_QUALITY_ORDER]),
    'log_area': NumericTransformer(keys=[AREA], transformation=LOG_TRANSFORMATION, add_log_value=1),
    'savi': SAVIITransformer(),
    'pssr': PSSRITransformer(),
    'evi': EVITransformer(),
    'evi2': EVI2Transformer(),
    'squared_geoms': NumericTransformer(keys=GEOMS, transformation=QUAD),
}


def get_features(feature_names=None):
    if not feature_names:
        feature_names = FEATURES_BY_NAME.keys()

    feature_names = [f for f in feature_names if isinstance(f, str)]
    if any(name not in FEATURES_BY_NAME for name in feature_names):
        logger.error(str([name for name in feature_names if name not in FEATURES_BY_NAME.keys()]))
        raise KeyError('Possible Keys are: {}'.format(sorted(FEATURES_BY_NAME.keys())))

    feature_transformers = [(name, FEATURES_BY_NAME[name]) for name in feature_names]
    return FeatureUnion(transformer_list=feature_transformers)
