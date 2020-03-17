from imblearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

from src.constants import NOT_TRANSFORMED_COLUMNS
from src.features.features import get_features, ImbalanceTransformer, ScalerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score)
import pickle
import json

MODELS_BY_NAME = {
    'logistic_regression': LogisticRegression(),
    'gradient_boosting': GradientBoostingClassifier()
}


class SoilClassifier:

    def __init__(self, feature_names=None, classifier='logistic_regression', min_samples=1000, max_samples=8000):
        self.feature_names = feature_names
        self.classifier = classifier
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.pipeline = None
        self.train_predictions = None
        self.metrics = {}

    def fit(self, X_train, y_train):
        steps = []

        steps.append(['scaler', ScalerTransformer(NOT_TRANSFORMED_COLUMNS)])
        steps.append(('features', get_features(self.feature_names)))
        steps.append(('sampler', ImbalanceTransformer(self.min_samples, self.max_samples)))
        steps.append(('classifier', MODELS_BY_NAME[self.classifier]))

        self.pipeline = Pipeline(steps=steps)

        self.pipeline.fit(X_train, y_train)
        self.train_predictions = self.pipeline.predict(X_train)

        self.metrics['train'] = self.calculate_metrics(y_train, self.train_predictions)

    def predict(self, X):
        return self.pipeline.predict(X)

    def calculate_metrics(self, y_true, y_pred):
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'custom_accuracy': self.custom_accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
            'recall_macro': recall_score(y_true, y_pred, average='macro'),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        }

    def evaluate(self, X_test, y_test):
        test_predictions = self.predict(X_test)
        self.metrics['test'] = self.calculate_metrics(y_test, test_predictions)

    def dump(self, file_path):
        model_json = {
            'feature_names': self.feature_names,
            'classifier': self.classifier,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples,
            'pipeline': self.pipeline
        }

        pickle.dump(model_json, open(file_path, 'wb'))
        self._dump_unicode(file_path)

    def _dump_unicode(self, file_path):
        model_json = {
            'feature_names': self.feature_names,
            'classifier': self.classifier,
            'min_samples': self.min_samples,
            'max_samples': self.max_samples,
            'metrics': self.metrics
        }

        json.dump(model_json, open(file_path.replace('pkl', 'json'), 'w'))

    def load(self, file_path):
        model_json = pickle.load(open(file_path, 'rb'))
        self.feature_names = model_json['feature_names']
        self.classifier = model_json['classifier']
        self.min_samples = model_json['min_samples']
        self.max_samples = model_json['max_samples']
        self.pipeline = model_json['pipeline']

    def _get_weights(self, target):
        if target == 'RESIDENTIAL':
            return 0.12
        if target == 'AGRICULTURE':
            return 2
        else:
            return 1

    def custom_accuracy_score(self, y_true, y_pred):
        weights = y_true.apply(self._get_weights)
        return ((y_true == y_pred) * weights).sum() / weights.sum()
