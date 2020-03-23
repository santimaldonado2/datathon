# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from src.constants import (TARGET_FEATURE, COLUMNS_TO_DUMP)
from src.models.model import SoilClassifier


@click.command()
@click.option('--input_filepath', type=click.Path(exists=True))
@click.option('--model_file_name', type=str)
def main(input_filepath, model_file_name):
    """ Train the model with the whole "Modelar" file and put the trained model into
        models folder
    """
    logger = logging.getLogger(__name__)
    logger.info('Predicting')

    df = pd.read_csv(os.path.join(input_filepath), sep='|')

    model = SoilClassifier()
    model.load('models/{}.pkl'.format(model_file_name))

    predictions = model.predict(df)

    df[TARGET_FEATURE] = predictions

    df[COLUMNS_TO_DUMP].to_csv('data/predictions/{}.csv'.format(model_file_name), index=False)
    df[COLUMNS_TO_DUMP].to_csv('AFI_maldo.txt'.format(model_file_name),
                               index=False,
                               sep='|',
                               encoding='UTF-8')

    logger.info('Finish Predictions, find the predicitons into data/predictions/{}.csv'.format(model_file_name))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
