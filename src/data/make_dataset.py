# -*- coding: utf-8 -*-
import click
import logging
import random
import pandas as pd
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split

from src.constants import (BALANCED_CLASS_DISTRIBUTION, TARGET_FEATURE, BALANCED, ALL, MODELATE_RAW_DATA_SET,
                           DATA_FILE_NAMES_BALANCED, TRAIN, TEST, IMBALANCED,
                           DATA_FILE_NAMES_IMBALANCED, DATA_SET_TYPES)


@click.command()
@click.option('--input_filepath', type=click.Path(exists=True))
@click.option('--output_filepath', type=click.Path())
@click.option('--data_set_type', type=click.Choice(DATA_SET_TYPES), default='all')
def main(input_filepath, output_filepath, data_set_type):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(os.path.join(input_filepath, MODELATE_RAW_DATA_SET), sep='|')

    if data_set_type == ALL or BALANCED:
        logger.info('making balanced data')
        data_train_balanced, data_test_balanced = custom_split_train_test(df=df)

        data_train_balanced.to_csv(os.path.join(output_filepath, DATA_FILE_NAMES_BALANCED[TRAIN]), index=False)
        data_test_balanced.to_csv(os.path.join(output_filepath, DATA_FILE_NAMES_BALANCED[TEST]), index=False)
        logger.info(
            'Test class distribution {}'.format(data_test_balanced[TARGET_FEATURE].value_counts(normalize=True)))
        logger.info('imbalanced data train {}'.format(data_train_balanced.shape))
        logger.info('imbalanced data test{}'.format(data_test_balanced.shape))
        logger.info('Finished Imbalanced data')

    if data_set_type == ALL or IMBALANCED:
        logger.info('making imbalanced data')
        data_train, data_test = train_test_split(df, test_size=0.2)

        data_train.to_csv(os.path.join(output_filepath, DATA_FILE_NAMES_IMBALANCED[TRAIN]), index=False)
        data_test.to_csv(os.path.join(output_filepath, DATA_FILE_NAMES_IMBALANCED[TEST]), index=False)
        logger.info('Test class distribution {}'.format(data_test[TARGET_FEATURE].value_counts(normalize=True)))
        logger.info('imbalanced data train {}'.format(data_train.shape))
        logger.info('imbalanced data test {}'.format(data_test.shape))
        logger.info('Finished Imbalanced data')


def custom_split_train_test(test_size=0.05, df=None):
    train_indexes_all = []
    test_indexes_all = []
    for name, prop in BALANCED_CLASS_DISTRIBUTION.items():
        class_indexes = set(df.loc[df[TARGET_FEATURE] == name].index)
        test_indexes = random.sample(class_indexes, round(prop * len(df) * test_size))
        train_indexes = class_indexes - set(test_indexes)

        train_indexes_all += train_indexes
        test_indexes_all += test_indexes

    random.shuffle(train_indexes_all)
    random.shuffle(test_indexes_all)

    return df.iloc[train_indexes_all].copy(), df.iloc[test_indexes_all].copy(),


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
