import os
import sys

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from additional_scripts import *
from datetime import datetime
import graphviz

def prepare_dataset(dataset_name, test_name, train_name, results = []):
    working_dir = os.getcwd()
    dataset_dir = "\\".join(working_dir.split("\\")[:-1]) + "\\" + "datasets" +  "\\" + dataset_name

    test_dir = dataset_dir + "\\" + test_name
    train_dir = dataset_dir + "\\" + train_name

    test = pd.read_csv(test_dir)
    train = pd.read_csv(train_dir)

    train_df = pd.DataFrame(train)
    train_df = train_df.dropna()

    test_df = pd.DataFrame(test)
    test_df = test_df.dropna()

    return test_df, train_df
'''
def prepare_dataset2(dataset_name):
    working_dir = os.getcwd()
    dataset_dir = "\\".join(working_dir.split("\\")[:-1]) + "\\" + "datasets" + "\\" + dataset_name

    test_dir = dataset_dir  + "\\" + "test.csv"
    train_dir = dataset_dir + "\\" + "train.csv"

    test = pd.read_csv(test_dir)
    train = pd.read_csv(train_dir)

    train_df = pd.DataFrame(train)
    train_df = train_df.dropna()

    test_df = pd.DataFrame(test)
    test_df = test_df.dropna()

    return test_df, train_df

def X_Y_split(df, Y_name):
    Y = df[Y_name]
    X = df.drop[Y_name]
    return X,Y

'''


def generate_model(depth, estimators, X_train, Y_train, X_test, Y_test):
    model = xgb.XGBClassifier(max_depth=depth, learning_rate=0.1, n_estimators=estimators)
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
              eval_set=[(X_test, Y_test)])
    return model

def sweep_depth(data, result_dir, min_depth_limit = 1, max_depth_limit = 10):
    rmses = []
    no_nodes = []
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    for depth in range(min_depth_limit, max_depth_limit+1):
        model = xgb.XGBClassifier(max_depth=depth, learning_rate=0.1, n_estimators=10)
        model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
            eval_set=[(X_test, Y_test)])
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        nodes = count_nodes(model)
        rmses.append(rmse)
        no_nodes.append(nodes)
        model.save_model(result_dir+"\\"+"models")
    return rmses, no_nodes

def sweep_estimators(data, result_dir, min_estimators = 1, max_estimators = 50):
    rmses = []
    no_nodes = []
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    models_dir = os.makedirs(result_dir+"\\"+"models")
    for estimators in range(min_estimators,max_estimators+1):
        model = xgb.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=estimators)
        model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
            eval_set=[(X_test, Y_test)])
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        nodes = count_nodes(model)
        rmses.append(rmse)
        no_nodes.append(nodes)
        
        model.save_model(result_dir+"\\"+"models" +"\\" + str(estimators) + "estimators")
               	
        for i in range(estimators):
#            figure(1)
#            tree_dir = result_dir+"\\"+"trees"+"\\" + str(estimators) + "estimators" + str(i)
            xgb.plot_tree(model, num_trees=i)
#            plt.save_fig(tree_dir)
            
            
    return rmses, no_nodes

def count_nodes(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len(tree) for tree in trees])
    return nodes

def generate_interpretability_curve(data, parameter_sweep, result_dir, min_val, max_val, x_name, y_name, title_name):
    rmses, nodes = parameter_sweep(data, min_val, max_val)
    plt.plot(nodes, rmses)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    
def run_experiment(dataset_name, test_file_name, train_file_name, experiment, experiment_params = []):
#     test_df, train_df = prepare_dataset(dataset_name, test_name, train_name, results=[])
    titanic_exp = 1
    if titanic_exp == 1:
        [X_train, Y_train, X_test, Y_test] = prepare_titanic()
        data = [X_train, Y_train, X_test, Y_test]
    else:
        [X_train, Y_train, X_test, Y_test] = prepare_dataset(dataset_name, test_file_name, train_file_name)
        data = [X_train, Y_train, X_test, Y_test]
        
    experiment_name_str = experiment.__name__
    experiment_timestamp = str(get_timestamp())
    
    working_dir = os.getcwd()
    result_dir = "\\".join(working_dir.split("\\")[:-2]) + "\\" + "Experiments"+ "\\"+ "results" +"\\" + dataset_name + "\\" + experiment_timestamp + "\\" + experiment_name_str
    print(result_dir)
    os.makedirs(result_dir)
    
    results = experiment(data, result_dir, *experiment_params)

#    save_results(results, result_dir)

def get_timestamp():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = str(datetime.fromtimestamp(timestamp))    

    dt_object = dt_object.replace(" ", "_")
    dt_object= dt_object.replace(":", "_")
    dt_object = dt_object.replace("-", "_")
    return dt_object

def save_tree():
    return

def save_results(experiment_results, results_dir):
    return
#todo
'''
git_ignore
command line git
xgb import fix
test functions
save and load models
clean up experiment functions
investigate open source xgboost library 
- to find a way to get structured representation of trees
- can we wrap realKD (java library)

tree visualisation xgboost

download and familiarise with example mario uploaded in trello

- model complexity/interpretabilty, test data, prediction results, estimators (given ideal depth)
'''