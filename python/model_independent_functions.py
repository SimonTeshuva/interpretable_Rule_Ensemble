# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:09:36 2019

@author: simon
"""
import os

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import xgboost as xgb

from datetime import datetime
from sklearn.model_selection import train_test_split

from xgboost_functions import *

def generate_interpretability_curve(error, interpretability, x_name, y_name, title_name, dataset_name, experiment_timestamp, param_name):
    result_directory_curves = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp, param_name, "plots")
    try:
        os.makedirs(result_directory_curves)
    except FileExistsError:
        pass
    tree_dir = os.path.join(result_directory_curves , "interpretability_curve.pdf")
        
    plt.figure(0)
    
    plt.plot(interpretability, error)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    
    plt.savefig(tree_dir, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)
    
    print("\n\n\n\n curve plotted and saved \n\n\n\n\n")
    for i in range(len(error)):
        print(error[i], interpretability[i])

    
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