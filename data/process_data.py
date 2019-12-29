import pandas as pd
import re
import sys
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Read original files
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    # Make categories column a list of strings of the form ⟨category⟩-⟨value⟩
    categories = pd.DataFrame(categories.iloc[:, 0].str.split(';').tolist(),
                              index=categories.index)
    # Change each column name to ⟨category⟩
    categories.columns = [re.sub('-.*', '', x) for x in categories.iloc[0]]
    # Strip ⟨category⟩ and convert to number.
    for column in categories:
        categories[column] = categories[column].map(
            lambda x: int(re.sub('.*-', '', x)))
    # Return a single dataframe with the messages and labels
    df = pd.merge(messages, categories, left_index=True, right_index=True)
    return df


def clean_data(df):
    df = df.drop_duplicates()
    # In a few places, the value 2 appears for the "related" column.
    # We change this to 1, since some classification algorithms
    # support only 2 categories.
    df.loc[:, "related"] = df["related"].map(lambda x: min(1, x))
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
