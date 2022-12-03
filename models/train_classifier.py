import argparse
from pathlib import Path

import joblib
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from constants import TABLE_NAME


def load_data(database_filepath):
    '''
    Load data
    Load the data stored in a Sqlite database.
    
    Input:
    database_filepath path to the database file.

    Output:
    X messages numpy matrix.
    y categories numpy matrix.
    categories list of category names.
    '''
    # load data from database
    print(f'sqlite:///{database_filepath}')
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(TABLE_NAME, engine)

    # break table into input data and labels.
    X = df[['message']]
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    categories = y.columns.to_numpy()
    X = X.to_numpy().flatten()
    y = y.to_numpy()

    return X, y, categories


def tokenize(text):
    '''
    Tokenize
    Tonekize text by word and lemmatize each token.

    Input:
    text a sentence to be tokenized.

    Output:
    tokens a list of tokens after breaking the input text.
    '''
    # case normalization
    text = text.lower()

    # break text in tokens
    tokens = word_tokenize(text)

    # lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def build_model():
    '''
    build_model
    Defines the model to classify the messages.

    Ouput:
    cv GridSearch with a model and parameters preset.
    '''
    estimator = RandomForestClassifier()

    pipeline = Pipeline((
        ('tokenize', TfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(estimator)),
    ))

    parameters = {
        'tokenize__stop_words': [None, 'english'],
        # 'tokenize__lowercase': [False, True],
        # 'tokenize__smooth_idf': [False, True],
        # 'classifier__estimator__n_estimators': [50, 100, 150, 200],
        # 'classifier__estimator__criterion': ['gini', 'entropy', 'log_loss'],
    }

    cv = GridSearchCV(pipeline, parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    evaluate_model
    Evaluate model
    Prints the f1-score, precision and recall for each of the categories.

    Input:
    model the model to be tested.
    X_test the test data input.
    y_test the test data labels.
    category_names the list of possible category names.
    '''
    y_pred = model.predict(X_test)

    for column_number, column_name in enumerate(category_names):
        avg_scores = classification_report(y_test[:,column_number], y_pred[:,column_number], output_dict=True, zero_division=0)['weighted avg']

        print(f'------------ {column_name} ({column_number + 1})------------')
        print(f'f1-score: {avg_scores["f1-score"]}, precision: {avg_scores["precision"]}, recall: {avg_scores["recall"]}')


def save_model(model, model_filepath):
    '''
    save_model
    Save the model to a pickled file.

    Input:
    model the model to be saved.
    model_filepath the path to the pickled trained model.
    '''
    joblib.dump(model, model_filepath)


def main():
    parser = argparse.ArgumentParser('Process data')
    parser.add_argument('database_filepath', type=Path, help='The path to the database file.')
    parser.add_argument('model_filepath', type=Path, help='The path to the pickled trained model.')
    args = parser.parse_args()

    database_filepath = args.database_filepath
    model_filepath = args.model_filepath

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print('Building model...')
    model = build_model()
    
    print('Training model...')
    model.fit(X_train, Y_train)
    
    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()