from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine
import nltk
import pandas as pd
import pickle
import sys

nltk.download('punkt')


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('DisasterResponse', engine)
    X = df['message']
    Y = df.drop(columns=['message', 'original', 'genre'])
    return X, Y, Y.columns


stem = SnowballStemmer("english").stem


def tokenize(text):
    words = word_tokenize(text.lower())
    words = [stem(w) for w in words]
    return words


stop_words = [stem(w) for w in
              TfidfVectorizer(stop_words="english").get_stop_words()]

def build_model():
    pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=tokenize,
                                       stop_words=stop_words)),
        ("classifier", RandomForestClassifier(n_estimators=100,
                                              max_features=0.1,
                                              n_jobs=-1))])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = pd.DataFrame(model.predict(X_test),
                          index=Y_test.index,
                          columns=Y_test.columns)

    for col in Y_pred.columns:
        print("Report for category “%s”:" % col)
        print(classification_report(Y_test[col], Y_pred[col]))

    print("Summary report:")
    print(classification_report(Y_test, Y_pred))


def save_model(model, model_filepath):
    s = pickle.dumps(model)
    with open(model_filepath, "wb") as f:
        f.write(s)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
