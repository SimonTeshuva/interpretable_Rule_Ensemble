# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:06:02 2019

@author: simon
"""

import os

import numpy as np 
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.metrics import mean_squared_error

from model_independent_functions import *

def generate_xgb_model(depth, estimators, X_train, Y_train, X_test, Y_test):
    model = xgb.XGBClassifier(max_depth=depth, learning_rate=0.1, n_estimators=estimators)
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc",
              eval_set=[(X_test, Y_test)])
    return model

def sweep_param(param_name, data, dataset_name, experiment_timestamp, save_results = False, min_val = 1, max_val = 20):
    rmses = []
    no_nodes = []
    X_train, Y_train, X_test, Y_test = data

        
    for param_val in range(min_val,max_val+1):
        if param_name == "depth":
            model = generate_xgb_model(param_val, 10, X_train, Y_train, X_test, Y_test)
        elif param_name == "estimators":
            model = generate_xgb_model(5, param_val, X_train, Y_train, X_test, Y_test)
        else:
            return "invalid parameter choice"
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        nodes = count_nodes(model)
        rmses.append(rmse)
        no_nodes.append(nodes)
        
        if save_results:
            save_xgb_model(model, dataset_name, param_name, param_val, experiment_timestamp)
            save_xgb_trees(model, dataset_name, param_name, param_val, experiment_timestamp)
            
    if save_results:
        generate_interpretability_curve(rmses, no_nodes, "nodes by " + param_name, "error", "Model complexity vs Error", dataset_name, experiment_timestamp, param_name)
        
    return rmses, no_nodes

def save_xgb_model(model, dataset_name, param_name, iteration_number, experiment_timestamp):
    result_directory_models = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp, param_name, "models")
    try:
        os.makedirs(result_directory_models)    
    except FileExistsError:
        pass
    model.save_model(os.path.join(result_directory_models, str(iteration_number) + param_name))

def save_xgb_trees(model, dataset_name, param_name, iteration_number, experiment_timestamp):
    result_directory_trees = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp, param_name, "trees")
    try:
        os.makedirs(result_directory_trees) 
    except FileExistsError:
        pass
    estimators = count_trees(model)
    for i in range(count_trees(model)):
        tree_dir = os.path.join(result_directory_trees, str(estimators) + "estimators" + str(i)+".pdf")
        tree = xgb.plot_tree(model, num_trees=i)
        plt.savefig(tree_dir, dpi=300, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)


def count_trees(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    return len(trees)    

def count_nodes(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len(tree) for tree in trees])
    return nodes

'''
# directory of the form 
# AttributeError: module 'xgboost' has no attribute 'load_model'
def load_xgb_model(dataset_name, timestamp, parameter_swept, model_name):
    model_dir = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, timestamp, parameter_swept, "models", model_name)
    model = xgb.load_model(model_dir)
    return model
# \Experiments\results\titanic2\2019_12_04_16_14_01.085663\estimators\models"
   
'''

'''
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
    os.makedirs(result_dir+"\\"+"models")
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
        try:
            os.makedirs(result_dir+"\\"+"trees"+"\\")
        except FileExistsError:
            pass
        
        for i in range(estimators):
            tree_dir = result_dir+"\\"+"trees"+"\\" + str(estimators) + "estimators" + str(i)+".pdf"
            tree = xgb.plot_tree(model, num_trees=i)
            plt.savefig(tree_dir, dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, bbox_inches=None, pad_inches=0.1,
                        frameon=None, metadata=None)
    return rmses, no_nodes
'''