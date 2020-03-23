# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.constants import (TARGET_FEATURE)
from src.models.model import SoilClassifier


@click.command()
@click.option('--input_filepath', type=click.Path(exists=True))
@click.option('--model_file_name', type=str)
def main(input_filepath, model_file_name):
    """Trains  a model and dumps it."""

    logger = logging.getLogger(__name__)
    logger.info('Training Model')

    df = pd.read_csv(os.path.join(input_filepath), sep='|')

    model = SoilClassifier(
        feature_names=['not_correlated', 'cadastral_ordinal_encoder_onehot', 'log_area', 'log_antiquity',
                       'squared_geoms', 'pssr', 'savi'],
        classifier='gradient_boosting',
        min_samples=1000,
        max_samples=15000)

    model.fit(df, df[TARGET_FEATURE])

    model.dump('models/{}.pkl'.format(model_file_name))
    logger.info('Training model finished, find the model into models/{}.pkl'.format(model_file_name))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
