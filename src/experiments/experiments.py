from pathlib import Path

import click
import logging

from dotenv import load_dotenv, find_dotenv

from src.constants import TARGET_FEATURE
from src.models.model import SoilClassifier
import pandas as pd
import os

@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info('Running Experiments')

    feature_options = [
        ('base', ['not_transformed']),
        ('1', ['not_trasnfromed', 'squared_coordinates', 'cadastral_ordinal_encoder_onehot', 'log_area',
               'building_year_decades']),
        ('2', ['not_trasnfromed', 'squared_coordinates', 'cadastral_ordinal_encoder', 'log_area',
               'building_year_decades']),
        ('3', ['not_trasnfromed', 'squared_coordinates', 'cadastral_ordinal_encoder_onehot', 'log_area',
               'building_antiquity_log']),
        ('4', ['not_trasnfromed', 'squared_coordinates', 'cadastral_ordinal_encoder', 'log_area',
               'building_antiquity_log']),
        ('5', ['not_trasnfromed', 'cadastral_ordinal_encoder_onehot', 'log_area',
               'building_year_decades']),
        ('6', ['not_trasnfromed', 'cadastral_ordinal_encoder', 'log_area',
               'building_year_decades']),
        ('7', ['not_trasnfromed', 'cadastral_ordinal_encoder_onehot', 'log_area',
               'building_antiquity_log']),
        ('8', ['not_trasnfromed', 'cadastral_ordinal_encoder', 'log_area',
               'building_antiquity_log']),
        ('all', None),
    ]

    classifier_options = ['logistic_regression', 'gradient_boosting']

    min_samples_options = [500, 1000, 2000]

    max_samples_options = [5000, 6000, 7000, 8000, 1000]

    data_type_options = ['balanced', 'imbalanced']

    for features in feature_options:
        for classifier in classifier_options:
            for min_samples in min_samples_options:
                for max_samples in max_samples_options:
                    for data_type in data_type_options:
                        run_experiment(features=features[1],
                                       features_comb_id=features[0],
                                       classifier=classifier,
                                       min_samples=min_samples,
                                       max_samples=max_samples,
                                       data_type=data_type)


def run_experiment(features, features_comb_id, classifier, min_samples, max_samples, data_type):
    logger = logging.getLogger(__name__)
    logger.info(
        'Running Experiment data_type = {data_type} features={features_comb_id} classifier={classifier} samples=[{min_samples},{max_samples}]'.format(
            data_type=data_type, features_comb_id=features_comb_id, classifier=classifier, min_samples=min_samples,
            max_samples=max_samples))
    try:
        df_train = pd.read_csv(os.path.join('data/processed', 'train_data_{}.csv'.format(data_type)))
        df_test = pd.read_csv(os.path.join('data/processed', 'test_data_{}.csv'.format(data_type)))

        model = SoilClassifier(feature_names=features,
                               classifier=classifier,
                               min_samples=min_samples,
                               max_samples=max_samples)

        model.fit(df_train, df_train[TARGET_FEATURE])
        model.evaluate(df_test, df_test[TARGET_FEATURE])

        model.dump(
            'src/experiments/results/{data_type}_{features_comb_id}_{classifier}_{min_samples}_{max_samples}.pkl'.format(
                data_type=data_type,
                features_comb_id=features_comb_id,
                classifier=classifier,
                min_samples=min_samples,
                max_samples=max_samples
            ))
    except Exception:
        logger.error(
            '''Exception Running Experiment data_type = {data_type} 
            features={features_comb_id} classifier={classifier} samples=[{min_samples},{max_samples}]'''.format(
                data_type=data_type, features_comb_id=features[0], classifier=classifier, min_samples=min_samples,
                max_samples=max_samples))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
