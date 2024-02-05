# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Instructions](#instructions)
3. [License](#license)
4. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset contains pre-labelled tweet and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

This project is divided in the following key sections:

1. Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
2. Build a machine learning pipeline to train the which can classify text message in various categories
3. Run a web app which can show model results in real time


<a name="instructions"></a>

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/


<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) 
* [Figure Eight](https://www.figure-eight.com/) 

![Train Classifier without Category Level Precision Recall](screenshots/train_classifier.png)

6. Sample run of train_classifier.py with precision, recall etc. for each category

![Train Classifier with Category Level Precision Recall](screenshots/train_classifier_category_precision_recall.png)

