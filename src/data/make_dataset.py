# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # read in csv files
    reviews = pd.read_csv('%s/reviews.csv' % input_filepath)
    beers = pd.read_csv('%s/beers.csv' % input_filepath)

    # mask for rows in which review length is greater than 0
    has_text_review_mask = reviews.apply(lambda x: True if len(x['text'].strip()) > 0 else False , axis=1)

    reviews_with_text = reviews[has_text_review_mask]

    logger.info('merging dataframes')

    #merge with beer dataframe
    reviews_with_beer_info = reviews_with_text.merge(beers, left_on='beer_id', right_on='id')


    logger.info('outputting dataframe to file')
    #output beer dataframe
    reviews_with_beer_info.to_pickle('%s/beers_reviews.pkl' % output_filepath)
    reviews_with_beer_info.to_csv('%s/beers_reviews.csv' % output_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
