# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 16:47:21 2019

@author: simon
"""

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


# from functions import *
from dataset_preparation_scripts import *
from xgboost_functions import *
from model_independent_functions import *

if __name__ == "__main__":
    change_cwd_to_project_root()
    cols_to_drop = ['name', 'platform', 'developer'	, 'publisher'	, 'genre(s)', 'players', 'rating', 'attribute', 'release_date', 'link']
    split_size = 0.33
    param_name = "estimators"
    dataset_name = "metacritic_games"
    file_name = "metacritic_games.csv"
    dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)
    Y_name = "user_score"
    experiment_timestamp = str(get_timestamp())

    data = prepare_dataset_one_file(dataset_name, file_name, Y_name, split_size, cols_to_drop)
    [X_train, Y_train, X_test, Y_test] = data
    
    model = generate_xgb_model(3, 10, X_train, Y_train, X_test, Y_test)

       
#    error, nodes = sweep_param(param_name, data, dataset_name, experiment_timestamp, save_results = True, min_val = 1, max_val = 20)