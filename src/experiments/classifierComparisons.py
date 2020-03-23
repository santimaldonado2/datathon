import logging

import click
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC

from src.constants import TARGET_FEATURE


@click.command()
def main():
    X_train = pd.read_csv('data/interim/X_train.csv')
    y_train = pd.read_csv('data/interim/y_train.csv')[TARGET_FEATURE]
    X_test = pd.read_csv('data/interim/X_test.csv')
    y_test = pd.read_csv('data/interim/y_test.csv')[TARGET_FEATURE]

    data = (X_train, y_train, X_test, y_test)

    models = [
        # ('gradient_boosting', GradientBoostingClassifier(n_iter_no_change=10), {
        #     'n_estimators': randint(low=50, high=52),
        #     'subsample': uniform(loc=0.5, scale=0.5),
        #     'min_samples_split': randint(low=2, high=6),
        #     'max_depth': randint(low=3, high=8),
        #     'max_features': ['sqrt', None]
        # }),
        ('support_vector_machine', SVC(), {
            'C': [1, 10, 100, 1000],
            'gamma': [1e-1, 1e-2, 1e-3, 1e-4],
            'kernel': ['linear', 'rbf']
        }),
        # ('ridge', RidgeClassifier(normalize=False), {
        #     'alpha': [1, 10, 100],
        #     'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        #
        # }),
        # ('lda', LinearDiscriminantAnalysis(), {
        #     'shrinkage': uniform(loc=0, scale=1),
        #     'solver': ['svd', 'lsqr', 'eigen']
        #
        # }),
        # ('random_forest', RandomForestClassifier(), {
        #     'n_estimators': randint(low=50, high=52),
        #     'min_samples_split': randint(low=2, high=6),
        #     'max_depth': randint(low=3, high=8),
        #     'max_features': ['sqrt', None]
        # }),

    ]

    for model in models:
        logging.info('#######{} Randomized Search########'.format(model[0]))
        execute_randomized_search(data, model)


def _get_weights(target):
    if target == 'RESIDENTIAL':
        return 0.12
    if target == 'AGRICULTURE':
        return 2
    else:
        return 1


def custom_accuracy_score(y_true, y_pred):
    weights = y_true.apply(_get_weights)
    return ((y_true == y_pred) * weights).sum() / weights.sum()


def calculate_metrics(y_true, y_pred):
    return dict(accuracy=accuracy_score(y_true, y_pred), custom_accuracy=custom_accuracy_score(y_true, y_pred),
                balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
                precision_macro=precision_score(y_true, y_pred, average='macro'),
                precision_weighted=precision_score(y_true, y_pred, average='weighted'),
                recall_macro=recall_score(y_true, y_pred, average='macro'),
                recall_weighted=recall_score(y_true, y_pred, average='weighted'),
                f1_macro=f1_score(y_true, y_pred, average='macro'),
                f1_weighted=f1_score(y_true, y_pred, average='weighted'))


def get_cv_results(data, model, model_name):
    X_train, y_train, X_test, y_test = data
    train_metrics = calculate_metrics(y_train, model.predict(X_train))
    test_metrics = calculate_metrics(y_test, model.predict(X_test))

    metrics = {'train_{}'.format(metric): train_metrics[metric] for metric in train_metrics.keys()}
    metrics.update({'test_{}'.format(metric): test_metrics[metric] for metric in test_metrics.keys()})

    metrics['model'] = model_name
    metrics['best_params'] = model.best_params_
    return metrics


def execute_randomized_search(data, model_configuration):
    X_train, y_train, X_test, y_test = data
    model_name, estimator, params = model_configuration

    rs = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        verbose=2,
        n_iter=12,
        cv=5,
        n_jobs=-1,
    )

    rs.fit(X_train, y_train)

    results = get_cv_results(data, rs, model_name)

    results_df = pd.read_csv('src/experiments/results/classifierComparisons/results.csv')

    results_df = results_df.append(results, ignore_index=True)

    results_df.to_csv('src/experiments/results/classifierComparisons/results.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
