# Disaster Response Pipeline Project

## Introduction
During the event of a disaster, a quick response is crutial for lives to be saved. Social media platforms can be a good tool in this situation since they can give almost direct access to the problems people are facing. The problem with this news source is that there is a high number of messages that might be unrelated to the current disaster. Even given that the message is related to the event, they might still be part of different categories, and routing them to the appropriate emergency service is important for an effective response.

This project offers a solution to classify text messages according to a set of categories. The first category `related` specifies if the message is related to a disaster.

|Category Name|
|-|
|Related|
|Request|
|Offer|
|Aid Related|
|Medical Help|
|Medical Products|
|Search And Rescue|
|Security|
|Military|
|Child Alone|
|Water|
|Food|
|Shelter|
|Clothing|
|Money|
|Missing People|
|Refugees|
|Death|
|Other Aid|
|Infrastructure Related|
|Transport|
|Buildings|
|Electricity|
|Tools|
|Hospitals|
|Shops|
|Aid Centers|
|Other Infrastructure|
|Weather Related|
|Floods|
|Storm|
|Fire|
|Earthquake|
|Cold|
|Other Weather|
|Direct Report|

## Project structure

* app
  - template
    - master.html: main page of the web app.
    - go.html: classification result page of the web app.
  - run.py: Flask file that runs the app.
* data
  - disaster_categories.csv: data to process.
  - disaster_messages.csv: data to process.
  - process_data.py: Run the Extraction, Transformation and Loading (ETL) of the data.
  - InsertDatabaseName.db: database to save clean data to.
* models
  - train_classifier.py: Trains a classifier and save a pickled model.
  - classifier.pkl: saved model.
* README.md

## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores it in a database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage


-----

This code is a part of an exercise from the [Data Science Nanodegree](https://github.com/udacity/DSND_Term2) by Udacity