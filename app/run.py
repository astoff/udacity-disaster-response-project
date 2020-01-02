import json
import plotly
from plotly.graph_objects import Heatmap
import pandas as pd
import numpy as np

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from itertools import product

app = Flask(__name__)

stem = SnowballStemmer("english").stem

def tokenize(text):
    words = word_tokenize(text.lower())
    words = [stem(w) for w in words]
    return words

# load data
engine = create_engine('sqlite:///../out/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../out/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df_categories = df.drop(columns=['message', 'original', 'genre']).sum()
    category_names = list(df_categories.index)
    df_categories['total'] = df.shape[0]
    df_categories = df_categories.sort_values(ascending=False)

    df_cat_matrix = pd.DataFrame([])
    for cat1, cat2 in product(category_names, category_names):
        pos1, pos2 = category_names.index(cat1), category_names.index(cat2)
        if pos1 <= pos2:
            items = df[df[cat2] == 1][cat1].sum()
        else:
            items = None
        df_cat_matrix.loc[cat1, cat2] = items
    print(df_cat_matrix)

    # Logarithmic color scale for heatmap
    myscale = [[0.0, '#000004'],
               [1.0E-6, '#180f3d'],
               [1.0E-5, '#440f76'],
               [1.0E-4, '#721f81'],
               [1.0E-3, '#9e2f7f'],
               [1.0E-2, '#cd4071'],
               [1.0E-1, '#f1605d'],
               [0.3, '#fd9668'],
               [0.7, '#feca8d'],
               [1.0, '#fcfdbf']]
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=[s.replace("_", " ") for s in df_categories.index],
                    y=list(df_categories)
                )
            ],

            'layout': {
                'title': 'Distribution of message categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    colorscale=myscale,
                    z=df_cat_matrix.values,
                    x=list(df_cat_matrix.columns),
                    y=list(df_cat_matrix.index),
                )
            ],

            'layout': {
                'title': 'Number of messages belonging to two given categories',
                'yaxis': {
                    'title': "Category 1"
                },
                'xaxis': {
                    'title': "Category 2"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
