from datetime import datetime
import os
import sys


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
# import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

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

def count_nodes(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len(tree) for tree in trees])
    return nodes

def sweep_depth(data, min_depth_limit = 1, max_depth_limit = 10):
    rmses = []
    no_nodes = []
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    for depth in range(min_depth_limit, max_depth_limit+1):
        model = xgb.XGBClassifier(max_depth=depth, learning_rate=0.1, n_estimators=100)
        model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
            eval_set=[(X_test, Y_test)])
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        nodes = count_nodes(model)
        rmses.append(rmse)
        no_nodes.append(nodes)
    return rmses, no_nodes

def sweep_estimators(data, min_estimators = 1, max_estimators = 50):
    rmses = []
    no_nodes = []
    X_train, Y_train, X_test, Y_test = data[0], data[1], data[2], data[3]
    for estimators in range(min_estimators,max_estimators+1):
        model = xgb.XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=estimators)
        model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
            eval_set=[(X_test, Y_test)])
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        nodes = count_nodes(model)
        rmses.append(rmse)
        no_nodes.append(nodes)
    return rmses, no_nodes

def get_timestamp():
    timestamp = 1545730073
    dt_object = str(datetime.fromtimestamp(timestamp))
    dt_object = dt_object.replace(" ", "_")
    dt_object= dt_object.replace(":", "_")
    dt_object = dt_object.replace("-", "_")
    return dt_object


def generate_interpretability_curve(parameter_sweep, min_val, max_val, x_name, y_name, title_name):
    rmses, nodes = parameter_sweep(min_val, max_val)
    plt.plot(nodes, rmses)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)


def generate_model(depth, estimators, X_train, Y_train, X_test, Y_test):
    model = xgb.XGBClassifier(max_depth=depth, learning_rate=0.1, n_estimators=estimators)
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
              eval_set=[(X_test, Y_test)])
    return model


def save_results(experiment_results, results_dir):
    return

def X_Y_split(df, Y_name):
    Y = df[Y_name]
    X = df.drop[Y_name]
    return X,Y

def run_experiment(dataset_name, test_name, train_name, experiment, experiment_params = []):
    test_df, train_df = prepare_dataset(dataset_name, test_name, train_name, results=[])


    experiment_name_str = experiment.__name__
    print(get_timestamp())
    experiment_tag = experiment_name_str +"_"+ str(get_timestamp())

    results = experiment(experiment_params, test_df, train_df)

    working_dir = os.getcwd()
    result_dir = "\\".join(working_dir.split("\\")[:-1]) + "\\" + "results" +"\\" + dataset_name + "\\" + experiment_tag
    print(result_dir)
    os.makedirs(result_dir)

    save_results(results, result_dir)

run_experiment("titanic", "test.csv", "train.csv", count_nodes)