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
from statistics import median

# from functions import *
from dataset_preparation_scripts import *
from orb.xgboost import *
from orb.branchbound import *
from orb.experimentation import *
from orb.rules import *

# needs to be run before anything else to move working directory up to appropriate level
def change_cwd_to_project_root():
    os.chdir('../')
    os.chdir('../')
    cwd = os.getcwd()
    return cwd


if __name__ == "__main__":
#    change_cwd_to_project_root()
    
    choice = 4

    if choice == 1:
        dataset_name = "titanic2"
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
        data_args = ["titanic2", "test.csv", "train.csv", "Survived", cols_to_drop]
        data = prepare_dataset_test_train_separate(*data_args)
        ts = str(get_timestamp())
        error, nodes = sweep_param("estimators", data, "titanic2", ts, save_results = True, min_val = 1, max_val = 20)
#    C:\Users\simon\Documents\Python Workspace\InterpretableRulesets\Experiments\results\titanic2\2019_12_04_16_14_01.085663\estimators\models
        model = load_xgb_model("titanic2", "2019_12_04_16_14_01.085663", "estimators", "1estimators")
        print(count_nodes(model))
    elif choice == 2:
        dataset_name = "metacritic_games"
        cols_to_drop = ['GameID','name', 'platform', 'developer'	, 'publisher'	, 'genre(s)', 'players', 'rating', 'attribute', 'release_date', 'link', 'user_score']
#        cols_to_drop = ['GameID','name', 'publisher'	, 'genre(s)', 'players', 'rating', 'attribute', 'release_date', 'link']
        split_size = 0.33
        param_name = "estimators"
        file_name = "metacritic_games.csv"
        experiment_timestamp = str(get_timestamp())
        
        dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)
        frame = pd.read_csv(os.path.join(dataset_dir, file_name))        
        frame = frame.drop(cols_to_drop, axis='columns')
        frame = frame.dropna()

        target_col = "top_rated"
        
            
        props = develop_proposition_set(frame, target_col, median)
                            
        rules_as_propositions = branch_and_bound(props)
        
        for rule_as_propositions in rules_as_propositions:
            print(rule_as_propositions)
        rules = Conjunction_of_ruleset(rules_as_propositions)       
        a = 2
        if a == 1:
            model_type = "Classifier"
            Y_name = "top_rated"
            cols_to_drop.append('user_score')
        else:
            model_type = "Regressor"    
            Y_name = "user_score"
            cols_to_drop.append('top_rated')

#        error, nodes = sweep_param(param_name, data, dataset_name, model_type, experiment_timestamp, save_results = True, min_val = 1, max_val = 3)
        generate_xgb_model(model_type, 4, 10, X_train, Y_train, X_test, Y_test)

    elif choice == 3:
        data = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
            "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
            "area": [8.516, 17.10, 3.286, 9.597, 1.221],
            "population": [200.4, 143.5, 1252, 1357, 52.98] }
        frame = pd.DataFrame(data)
        target_col = 'country'
        
        props = develop_proposition_set(frame, target_col, median)

        rules_as_propositions = branch_and_bound(props)
        
        rules = Conjunction_of_ruleset(rules_as_propositions) 
    else:
        dataset_name = "titanic2"
        cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin']
        data_args = ["titanic2", "test.csv", "train.csv", "Survived", cols_to_drop]
        data = prepare_dataset_test_train_separate(*data_args)
        target_col = 'Survived'
        
        dataset_dir = os.path.join(os.getcwd(), "Datasets", dataset_name)

        dataset = pd.read_csv(os.path.join(dataset_dir, "train.csv"))

        frame = pd.DataFrame(dataset)
        frame = frame.drop(cols_to_drop, axis='columns')
        frame = frame.dropna()
        frame = pd.get_dummies(frame)
        
        props = develop_proposition_set(frame, target_col, [median])
    
        rules_as_propositions = branch_and_bound(props)
        
#        rules = Conjunction_of_ruleset(rules_as_propositions) 
#        for rule in rules:
#            print(rule)

        
# add rules of the same type which are not contradictory (population < 100 and population > 50)
# xgbRegresssor not working yet
# I may be creating duplicate rule
# boolean columns need to be fixed. at the moment <,<=, >=,> are used as though it is numbers
# implement doctest