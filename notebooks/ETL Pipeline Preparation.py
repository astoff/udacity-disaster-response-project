# %%
'''
# ETL Pipeline Preparation
Follow the instructions below to help you create your ETL pipeline.
### 1. Import libraries and load datasets.
- Import Python libraries
- Load `messages.csv` into a dataframe and inspect the first few lines.
- Load `categories.csv` into a dataframe and inspect the first few lines.
'''

# %%
# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
# %%
# load messages dataset
messages = pd.read_csv('messages.csv', index_col='id')
messages.head()

# %%
# load categories dataset
categories = pd.read_csv('categories.csv', index_col='id')
categories.head()

# %%
'''
### 2. Merge datasets.
- Merge the messages and categories datasets using the common id
- Assign this combined dataset to `df`, which will be cleaned in the following steps
'''

# %%
# merge datasets
df = pd.merge(messages, categories, left_index=True, right_index=True)
df.head()

# %%
'''
### 3. Split `categories` into separate category columns.
- Split the values in the `categories` column on the `;` character so that each value becomes a separate column. You'll find [this method](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.Series.str.split.html) very helpful! Make sure to set `expand=True`.
- Use the first row of categories dataframe to create column names for the categories data.
- Rename columns of `categories` with new column names.
'''

# %%
# create a dataframe of the 36 individual category columns
categories = pd.DataFrame(categories.iloc[:,0].str.split(';').tolist(), index=categories.index)
categories.head()

# %%
# select the first row of the categories dataframe
row = categories.iloc[0]

# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
import re
category_colnames = [re.sub('-.*', '', x) for x in row]
print(category_colnames)

# %%
# rename the columns of `categories`
categories.columns = category_colnames
categories.head()

# %%
'''
### 4. Convert category values to just numbers 0 or 1.
- Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). For example, `related-0` becomes `0`, `related-1` becomes `1`. Convert the string to a numeric value.
- You can perform [normal string actions on Pandas Series](https://pandas.pydata.org/pandas-docs/stable/text.html#indexing-with-str), like indexing, by including `.str` after the Series. You may need to first convert the Series to be of type string, which you can do with `astype(str)`.
'''

# %%
for column in categories:
    # set each value to be the last character of the string
    categories[column] = categories[column].map(lambda x: int(re.sub('.*-', '', x)))
    
    # convert column from string to numeric
    #categories2[column] = 
categories.head()

# %%
'''
### 5. Replace `categories` column in `df` with new category columns.
- Drop the categories column from the df dataframe since it is no longer needed.
- Concatenate df and categories data frames.
'''

# %%
# drop the original categories column from `df`
df = df.drop(columns='categories')

df.head()

# %%
# concatenate the original dataframe with the new `categories` dataframe
df = pd.merge(df, categories, left_index=True, right_index=True)
df.head()

# %%
'''
### 6. Remove duplicates.
- Check how many duplicates are in this dataset.
- Drop the duplicates.
- Confirm duplicates were removed.
'''

# %%
# check number of duplicates
df.duplicated().sum()

# %%
# drop duplicates
df = df.drop_duplicates()

# %%
# check number of duplicate
df.duplicated().sum()


# %%
'''
### 7. Save the clean dataset into an sqlite database.
You can do this with pandas [`to_sql` method](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html) combined with the SQLAlchemy library. Remember to import SQLAlchemy's `create_engine` in the first cell of this notebook to use it below.
'''

# %%
engine = create_engine('sqlite:///InsertDatabaseName.db')
df.to_sql('InsertTableName', engine, index=False)

# %%
'''
### 8. Use this notebook to complete `etl_pipeline.py`
Use the template file attached in the Resources folder to write a script that runs the steps above to create a database based on new datasets specified by the user. Alternatively, you can complete `etl_pipeline.py` in the classroom on the `Project Workspace IDE` coming later.
'''

# %%
