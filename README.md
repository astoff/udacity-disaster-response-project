# Disaster Response Pipeline Project

This is my “disaster response pipeline” project for the data scientist
nanodegree.

## Contents

- The folder `notebooks` contains some exploratory scripts.  See in
  particular `notebooks/ML Pipeline Preparation.py`, where a grid
  search is used to find good parameter for the model.

- The folder `data` contains the original message dataset (message
  text and categories), as two separate CSV files.  The script
  `process_data.py` transforms and saves this to a database file.

- The folder `models` contains the script `train_classifier.py` to
  train the model.

- The folder `app` contains the Flask webapp that categorizes new
  messages.  It assumes the database file and model generated by the
  scripts mentioned above are in the `out` folder at the root of this
  repository.

## Instructions

1. Run the following commands in the project's root directory to set
   up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv
        data/disaster_categories.csv out/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves `python
        models/train_classifier.py data/DisasterResponse.db
        out/classifier.pkl`

2. Run the following command in the app's directory to run your web
    app.  `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgments

The code in this repository is based on a template provided by
Udacity.  In particular, the web app is almost unmodified from their
example.
