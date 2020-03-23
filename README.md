Datathon Univeristyhack 2020 - Minsait Land Classification
==============================

Repo for the Minsait Land Classification 2020.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── predictions    <- Predictions
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── experiments    <- Scripts to execute some experiments
    │   │   └── resutls    <- Results of the differents experiments runned
    │   │   └── classifierComparisons.py <- Run grid searchs over differents models           
    │   │   └── experiments.py <- Some experiments
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model.py   <- Whole Pipeline
    │   │   ├── predict_model.py <- Make predictions
    │   │   └── train_model.py   <- Train and dump the models   
    │   └── constants.py   <- all the constants used in the project
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

Requirements
------------
- Python3
- Pip3
- Virtualenv

You must create a virtualenv and activate it

Makefile Commands
------------
- `requirements`: Will install all the requirements into the virtualenv
- `train_model`: Will use the file data/raw/Modelar_UH2020.txt, train the model
                 and dump it into models folder
- `predict`: Will create the predictions for data/raw/Estimar_UH2020.txt, and save
             them into data/predicitions
- `train_and_predict`: Will execute the train_model and then the predict
- `full_train_and_predict`: Will install the requirements then train, and finally predict



<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
