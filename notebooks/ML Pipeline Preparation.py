# %%
'''
# ML Pipeline Preparation
Follow the instructions below to help you create your ML pipeline.
### 1. Import libraries and load data from database.
- Import Python libraries
- Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
- Define feature and target variables X and Y
'''

# %%
# import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

import os
import nltk
# nltk.data.path.insert(0, os.path.expanduser("~/Downloads/nltk_data"))
# nltk.download('punkt', download_dir=nltk.data.path[0])
nltk.download('punkt')

# %%
# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql('InsertTableName', engine)
X = df['message']
Y = df.drop(columns=['message', 'original', 'genre', 'child_alone'])
Y['related'] = Y['related'].map(lambda x: min(x, 1))

# %%
'''
### 2. Write a tokenization function to process your text data
'''

# %%


stem = SnowballStemmer("english").stem

def tokenize(text):
    words = word_tokenize(text.lower())
    words = [stem(w) for w in words]
    return words


# %%
'''
### 3. Build a machine learning pipeline
This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.
'''

# %%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

pipeline = Pipeline([
    ("vectorizer", TfidfVectorizer(tokenizer=tokenize)),
    ("classifier", RandomForestClassifier())
#    ("classifier", MultiOutputClassifier(GradientBoostingClassifier()))
#    ("classifier", KNeighborsClassifier())
#    ("classifier", MultiOutputClassifier(AdaBoostClassifier()))
#    ("classifier", MLPClassifier())
])

# %%
'''
### 4. Train pipeline
- Split data into train and test sets
- Train pipeline
'''

# %%

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
#X_train, Y_train = X, Y

pipeline.fit(X_train, Y_train)


# %%
'''
### 5. Test your model
Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.
'''

# %%

Y_pred = pd.DataFrame(pipeline.predict(X_test),
                      index=Y_test.index,
                      columns=Y_test.columns)


for col in Y_pred.columns:
    print(">>> %s" % col)
    print(classification_report(Y_test[col], Y_pred[col]))

print(classification_report(Y_test, Y_pred))
# %%
'''
### 6. Improve your model
Use grid search to find better parameters. 
'''

# %%
parameters = {
    "vectorizer__ngram_range": [(1,1), (1,2), (1,5)]
    "vectorizer__stop_words": [None, "english"]
    "vectorizer__use_idf": [True, False],
    "classifier__n_estimators": [10, 50, 100],
    "classifier__max_features": [0.01, 0.05, 0.1],
}

# Notes: max_features = 0.5 gives score ~0.63, "log2" gives 0.53.  But
# fitting times are too long with 0.5

cv = GridSearchCV(pipeline,
                  param_grid=parameters,
                  scoring='f1_micro',
                  n_jobs=-1)

cv.fit(X_train, Y_train)

# %%
'''
### 7. Test your model
Show the accuracy, precision, and recall of the tuned model.  

Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!
'''

# %%

Y_pred = pd.DataFrame(cv.predict(X_test),
                      index=Y_test.index,
                      columns=Y_test.columns)


for col in Y_pred.columns:
    print(">>> %s" % col)
    print(classification_report(Y_test[col], Y_pred[col]))

print(classification_report(Y_test, Y_pred))

# %%
'''
### 8. Try improving your model further. Here are a few ideas:
* try other machine learning algorithms
* add other features besides the TF-IDF
'''

# %%


# %%
'''
### 9. Export your model as a pickle file
'''

# %%


# %%
'''
### 10. Use this notebook to complete `train.py`
Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.
'''

# %%
