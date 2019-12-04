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

    cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
    data_args = ["titanic2", "test.csv", "train.csv", "Survived", cols_to_drop]
    data = prepare_dataset_test_train_separate(*data_args)
    ts = str(get_timestamp())
#    error, nodes = sweep_param("estimators", data, "titanic2", ts, save_results = True, min_val = 1, max_val = 20)
#    C:\Users\simon\Documents\Python Workspace\InterpretableRulesets\Experiments\results\titanic2\2019_12_04_16_14_01.085663\estimators\models
    model = load_xgb_model("titanic2", "2019_12_04_16_14_01.085663", "estimators", "1estimators")
    print(count_nodes(model))