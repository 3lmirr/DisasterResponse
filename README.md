# Disaster Response Pipeline Project

## Table of Contents
1. [Description](#description)
2. [Instructions](#instructions)
3. [License](#license)
4. [Acknowledgement](#acknowledgement)

<a name="descripton"></a>
## Description

This project, a collaboration between Udacity's Data Science Nanodegree Program and Figure Eight, revolves around analyzing a dataset comprising pre-labeled tweets and messages associated with real-life disaster events. The primary objective is to develop an efficient Natural Language Processing (NLP) model capable of swiftly categorizing incoming messages in real-time.

The project encompasses several key phases:

1. Data Processing: This phase entails constructing an ETL (Extract, Transform, Load) pipeline to collect data from the source, cleanse it thoroughly, and then store it in a SQLite database.

2. Machine Learning Pipeline: Here, the focus is on building a robust machine learning pipeline. This pipeline will be trained to classify text messages into diverse categories, leveraging the power of NLP techniques.

3. Web Application Deployment: In the final stage, a user-friendly web application is deployed. This application serves as an interface for showcasing the model's results in real-time, providing users with a seamless experience to interact with the classification system.


<a name="instructions"></a>

## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to [http://0.0.0.0:3001/](http://127.0.0.1:3000)


<a name="license"></a>
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
## Acknowledgements

* [Udacity](https://www.udacity.com/) 
* [Figure Eight](https://www.figure-eight.com/) 



