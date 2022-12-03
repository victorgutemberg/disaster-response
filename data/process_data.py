import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    Load data from the messages and categories csv files and merge them to a single dataframe.

    Input:
    messages_filepath path to the messages CSV file.
    categories_filepath path to the categories CSV file.

    Output:
    df merged dataframe containing messages and categories.
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    return messages.merge(categories, on='id')


def clean_data(df):
    '''
    clean_data
    Convert categories values from string to int and drop duplicates.

    Input:
    df dataframe containing messages and categories.

    Output:
    df dataframe after cleaning.
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = [col_name[:-2] for col_name in row]

    categories.columns = category_colnames

    # convert to integer
    categories = categories.apply(lambda column: column.str[-1:]).astype(int)

    # filter values different than one or zero and convert to boolean
    categories = categories[(categories <= 1).all(1)].astype(bool)

    # drop the original categories column from `df`
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat((df, categories), axis=1)

    # drop duplicates
    print('Number of duplicates dropped:', df.duplicated().sum())
    df = df.drop_duplicates()

    return df

def save_data(df, database_filepath):
    '''
    save_data
    Saves a dataframe to a Sqlite file.

    Input:
    df dataframe to be saved.
    database_filepath path to the database file.
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df.to_sql('messages_expanded', engine, index=False, if_exists='replace')


def main():
    parser = argparse.ArgumentParser('Process data')
    parser.add_argument('messages_filepath', type=Path, help='The path to the messages CSV file.')
    parser.add_argument('categories_filepath', type=Path, help='The path to the categories CSV file.')
    parser.add_argument('database_filepath', type=Path, help='The path to the database file.')
    args = parser.parse_args()

    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath

    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
            .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
    
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
    
    print('Cleaned data saved to database!')


if __name__ == '__main__':
    main()