import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import load_dotenv, find_dotenv

from src.constants import TARGET_FEATURE
from src.models.model import SoilClassifier


@click.command()
def main():
    logger = logging.getLogger(__name__)
    logger.info('Running Experiments')

    feature_options = [
        ('all', None),
    ]

    classifier_options = ['gradient_boosting']

    min_samples_options = [500]

    max_samples_options = [9000]

    for features in feature_options:
        for classifier in classifier_options:
            for min_samples in min_samples_options:
                for max_samples in max_samples_options:
                    run_experiment(features=features[1],
                                   features_comb_id=features[0],
                                   classifier=classifier,
                                   min_samples=min_samples,
                                   max_samples=max_samples)


def run_experiment(features, features_comb_id, classifier, min_samples, max_samples):
    logger = logging.getLogger(__name__)
    logger.info(
        'Running Exp features={features_comb_id} classifier={classifier} samples=[{min_samples},{max_samples}]'.format(
            features_comb_id=features_comb_id,
            classifier=classifier,
            min_samples=min_samples,
            max_samples=max_samples))

    df_train = pd.read_csv(os.path.join('data/processed', 'train_data.csv'))
    df_test = pd.read_csv(os.path.join('data/processed', 'test_data.csv'))

    model = SoilClassifier(feature_names=features,
                           classifier=classifier,
                           min_samples=min_samples,
                           max_samples=max_samples)

    model.fit(df_train, df_train[TARGET_FEATURE])
    model.evaluate(df_test, df_test[TARGET_FEATURE])

    model.dump(
        'src/experiments/results/minmax_{features_comb_id}_{classifier}_{min_samples}_{max_samples}_scaled.pkl'.format(
            features_comb_id=features_comb_id,
            classifier=classifier,
            min_samples=min_samples,
            max_samples=max_samples
        ))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
