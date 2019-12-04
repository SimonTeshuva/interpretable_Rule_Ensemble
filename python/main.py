import os
import sys
import graphviz

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# from functions import *
from dataset_preparation_scripts import *
from xgboost_functions import *
from model_independent_functions import *

# needs to be run before anything else to move working directory up to appropriate level
def change_cwd_to_project_root():
    os.chdir('../')
    os.chdir('../')
    cwd = os.getcwd()
    return cwd


if __name__ == "__main__":
    change_cwd_to_project_root()
    
    dataset_name = "titanic2"
    dataset_name = "metacritic_games"

    if dataset_name == "titanic2":
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
        data_args = ["titanic2", "test.csv", "train.csv", "Survived", cols_to_drop]
        data = prepare_dataset_test_train_separate(*data_args)
        ts = str(get_timestamp())
        error, nodes = sweep_param("estimators", data, "titanic2", ts, save_results = True, min_val = 1, max_val = 20)
#    C:\Users\simon\Documents\Python Workspace\InterpretableRulesets\Experiments\results\titanic2\2019_12_04_16_14_01.085663\estimators\models
        model = load_xgb_model("titanic2", "2019_12_04_16_14_01.085663", "estimators", "1estimators")
        print(count_nodes(model))
    elif dataset_name == "metacritic_games" :
#        cols_to_drop = ['GameID','name', 'platform', 'developer'	, 'publisher'	, 'genres', 'players', 'rating', 'attribute', 'release_date', 'link', 'user_score']
        cols_to_drop = ['GameID','name', 'publisher'	, 'genres', 'players', 'rating', 'attribute', 'release_date', 'link', 'user_score']
        split_size = 0.33
        param_name = "estimators"
        file_name = "metacritic_games.csv"
        dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)
        Y_name = "fan_favorite"
        experiment_timestamp = str(get_timestamp())

        data = prepare_dataset_one_file(dataset_name, file_name, Y_name, split_size, cols_to_drop)
        [X_train, Y_train, X_test, Y_test] = data
        error, nodes = sweep_param(param_name, data, dataset_name, experiment_timestamp, save_results = True, min_val = 1, max_val = 20)
        