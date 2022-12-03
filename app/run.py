import argparse
import json
from pathlib import Path


import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # get messages grouped by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # get the top n categories most used
    n_categories = 10
    categories = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    top_categories = categories.sum().sort_values(ascending=False).head(n_categories)
    category_names = [category.replace('_', ' ').title() for category in top_categories.index]
    
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
                    x=category_names,
                    y=top_categories
                ),
            ],
            'layout': {
                'title': f'Top {n_categories} Most Frequent Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process data')
    parser.add_argument('database_filepath', type=Path, help='The path to the database file.')
    parser.add_argument('model_filepath', type=Path, help='The path to the pickled trained model.')
    args = parser.parse_args()

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath

    # load data
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages_expanded', engine)

    # load model
    model = joblib.load(model_filepath)

    app.run(host='0.0.0.0', port=3000, debug=True)