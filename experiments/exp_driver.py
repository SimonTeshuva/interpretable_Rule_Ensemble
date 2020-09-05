import re
import os
import random
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from orb.rules import AdditiveRuleEnsemble, ExponentialObjective, SquaredLossObjective, Rule
import rulefit
from xgb_scripts.xgb_functions import count_trees
import xgboost as xgb
from realkd.rules import GradientBoostingRuleEnsemble, LogisticLoss, SquaredLoss

from experiments.data_preprocessing import *
from experiments.exp_architecture import *

def many_exp(dataset_name, fixed_exp_params = {}, params_vectors = {}):
    res_df = pd.DataFrame()

    runs = param_options(fixed_exp_params, params_vectors) # rename to variable_params
    for run in runs:
        print(run)
        exp = Experiment(dataset_name, **run)
        res, model = exp.run()
        res_df = res_df.append(res, ignore_index=True)

    all_res = pd.read_csv('all_results.csv')
    all_res = all_res.append(res_df, ignore_index=True)
    all_res.to_csv('all_results.csv', mode='w')

    return res_df

def run_and_plot(dataset_names, reqs_vect, repeats):
    res_df = pd.DataFrame()

    for idx, dataset_name in enumerate(dataset_names):
        plt.figure(num=idx+1)
        for fixed, variable, x, y in reqs_vect:
            X, Y_test, Y_train = [], [], []
            for i in range(repeats):
                print("Repeat: ", str(i))
                results = many_exp(dataset_name=dataset_name, fixed_exp_params=fixed, params_vectors=variable)
                res_df = res_df.append(results, ignore_index=True)
                X.append(list(results[x]))
                Y_test.append(list(results[y + '_test']))
                Y_train.append(list(results[y + '_train']))
            X = np.array(X).mean(axis=0)
            Y_test = np.array(Y_test).mean(axis=0)
            Y_train = np.array(Y_train).mean(axis=0)

            X, Y_test, Y_train = zip(*sorted(zip(X, Y_test, Y_train)))

            plt.plot(X, Y_test, label=' '.join([fixed['model_class'],'test']))
            plt.plot(X, Y_train, label=' '.join([fixed['model_class'],'train']))
        plt.legend()
        plt.title(dataset_name)
        plt.xlabel('model complexity')
        plt.ylabel('rmse')
        plt.show()

    res_df.to_csv('out.csv')

    return res_df


def get_best(res_df, dataset_name, max_interpretable):
    res_df = res_df.loc[res_df['dataset'] == dataset_name]

    orb_df = res_df.loc[res_df['model_class'] == 'orb']
    rulefit_df = res_df.loc[res_df['model_class'] == 'rulefit']
    xgboost_df = res_df.loc[res_df['model_class'] == "xgb.XGBRegressor"]
    xgboost_df = xgboost_df.append(res_df.loc[res_df['model_class'] == "xgb.XGBClassifier"], ignore_index=True)

    dde_metric = ('rmse' if orb_df.iloc[0]["objective_function"] == "SquaredLossObjective" else 'accuracy') + '_test'

    best_orb_innacuracy = None
    best_orb = None
    for idx, orb_model in orb_df.iterrows():
        complexity = orb_model['orb_complexity']
        innacuracy = orb_model[dde_metric]

        if best_orb is None or (best_orb_innacuracy > innacuracy and complexity < max_interpretable):
            best_orb_innacuracy = innacuracy
            best_orb = orb_model

    closest_xgb = None
    closest_xgb_complexity = None
    for idx, xgb_model in xgboost_df.iterrows():
        complexity = xgb_model['forest_complexity']

        if closest_xgb is None or abs(best_orb["orb_complexity"] - complexity) < abs(best_orb["orb_complexity"] - closest_xgb_complexity):
            closest_xgb_complexity = complexity
            closest_xgb = xgb_model


    closest_rulefit = None
    closest_rulefit_complexity = None
    for idx, rulefit_model in rulefit_df.iterrows():
        complexity = rulefit_model['RuleFit_complexity']

        if closest_rulefit is None or abs(best_orb["orb_complexity"] - complexity) < abs(best_orb["orb_complexity"] - closest_rulefit_complexity):
            closest_rulefit_complexity = complexity
            closest_rulefit = rulefit_model

    return best_orb, closest_xgb, closest_rulefit


def make_comparison(dataset_name, best_orb, closest_xgboost, closest_rulefit, downsample_size):
    problem_type = 'r' if best_orb["objective_function"] == "SquaredLossObjective" else 'c'

    print('best_orb')
    print(best_orb)
    exp = Experiment(dataset_name=dataset_name, model_class=best_orb['model_class'], objective=best_orb['objective_function'],
                     params=best_orb['model_params'], dde=[rmse if problem_type == 'r' else accuracy], die=[orb_complexity], downsample_size=downsample_size)
    orb_res, orb_model = exp.run()
    print('orb_complexity', orb_res['orb_complexity'])
    print('rmse_test', orb_res['rmse_test'])

    print('closest_rulefit')
    print(closest_rulefit)
    exp = Experiment(dataset_name=dataset_name, model_class=closest_rulefit['model_class'], objective=closest_rulefit['objective_function'],
                     params=closest_rulefit['model_params'], dde=[rmse if problem_type == 'r' else accuracy], die=[RuleFit_complexity], downsample_size=downsample_size)
    rulefit_res, rulefit_model = exp.run()
    rules = rulefit_model.get_rules()
    rules = rules[rules.coef != 0].sort_values("support", ascending=False)
    coef_rule = [(rule[1]['coef'], rule[1]['rule']) for rule in rules.iterrows()]
    for coef, rule in coef_rule:
        print(coef, 'if', rule)
    print('rulefit_complexity', rulefit_res['RuleFit_complexity'])
    print('rmse_test', rulefit_res['rmse_test'])

    print("closest_xgboost")
    print(closest_xgboost)
    exp = Experiment(dataset_name=dataset_name, model_class=closest_xgboost['model_class'], objective=closest_xgboost['objective_function'],
                     params=closest_xgboost['model_params'], dde=[rmse if problem_type == 'r' else accuracy], die=[forest_complexity], downsample_size=downsample_size)
    xgboost_res, xgb_model = exp.run()
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    trees = [[node for node in tree] for tree in trees]
    for tree in trees:
        for node in tree:
            print(node)

    for i in range(count_trees(xgb_model)):
        plt.figure(num=i+1)
        tree = xgb.plot_tree(xgb_model, num_trees=i)
        plt.title('tree' + str(i))
    print('xgb_complexity', xgboost_res['forest_complexity'])
    print('rmse_test', xgboost_res['rmse_test'])
    plt.show()

def full_comparison(reqs_vect, dataset_names = ['halloween_candy_ranking', 'boston'], repeats = 5, max_interpretable = 20):

    for dataset_name in dataset_names:
        print(dataset_name)
        res_df = run_and_plot([dataset_name], reqs_vect, repeats)

        # best_orb, closest_xgboost, closest_rulefit = get_best(res_df, dataset_name, max_interpretable)
        # make_comparison(dataset_name, best_orb, closest_xgboost, closest_rulefit, downsample_size)

    return res_df


def param_options(fixed_exp_params, params_vectors):
    from itertools import product
    from copy import deepcopy

    experiments = []

    params_list = [params_vectors[key] for key in params_vectors.keys()]
    params_keys = list(params_vectors.keys())

    every_param_option = list(product(*[params for params in params_list]))

    for i in range(len(every_param_option)):
        exp_params_dict = dict()
        params = every_param_option[i]
        for j in range(len(params_keys)):
            param_key = params_keys[j]
            param = params[j]
            exp_params_dict[param_key] = param
        exp = deepcopy(fixed_exp_params)
        exp['params'] = exp_params_dict
        experiments.append(exp)

    return experiments

if __name__ == "__main__":

    dataset_names = ['halloween_candy_ranking']# , 'boston'] #, 'avocado_prices']
    downsample_size = None
    repeats = 5
    max_interpretable = 30

    orb_fixed = {"model_class": 'orb',
             "objective": 'SquaredLossObjective',
             "dde": [rmse],
             "die": [orb_complexity],
             "downsample_size": downsample_size}
    k_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    orb_variable = {'k': k_lst,'reg': [50], 'max_col_attr': [10]}
    orb_x = 'orb_complexity'
    orb_y = 'rmse'

    orb_experiments = (orb_fixed, orb_variable, orb_x, orb_y)

    rulefit_fixed = {"model_class": 'rulefit',
             "objective": 'regress',
             "dde": [rmse],
             "die": [RuleFit_complexity],
             "downsample_size": downsample_size}

    Cs_lst = [[0.1], [0.2], [0.5], [0.75], [1], [2], [5], [10], [20], [50], [100], [200], [500], [1000]]
    Cs = [np.array(C) for C in Cs_lst]
    rulefit_variable = {'Cs': Cs}
    rulefit_x = 'RuleFit_complexity'
    rulefit_y = 'rmse'

    rulefit_experiments = (rulefit_fixed, rulefit_variable, rulefit_x, rulefit_y)

    xgboost_fixed = {"model_class": 'xgb.XGBRegressor',
             "objective": 'reg:squarederror',
             "dde": [rmse],
             "die": [forest_complexity],
             "downsample_size": downsample_size}
    n_estimators_lst = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    xgboost_variable = {'max_depth': [10],'reg_lambda': [50], 'n_estimators': n_estimators_lst}
    xgboost_x = 'forest_complexity'
    xgboost_y = 'rmse'

    xgboost_experiments = (xgboost_fixed, xgboost_variable, xgboost_x, xgboost_y)

    realkd_fixed = {"model_class": 'realkd',
             "objective": 'squared_loss',
             "dde": [rmse],
             "die": [orb_complexity],
             "downsample_size": downsample_size}
    max_rules = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    realkd_variable = {'max_rules': max_rules,'reg': [50.0], 'max_col_attr': [10]}
    realkd_x = 'orb_complexity'
    realkd_y = 'rmse'

    realkd_experiments = (realkd_fixed, realkd_variable, realkd_x, realkd_y)

    reqs_vect = [realkd_experiments, orb_experiments, rulefit_experiments, xgboost_experiments]
#    reqs_vect = [xgboost_experiments]

    full_comparison(reqs_vect, dataset_names=['halloween_candy_ranking'], repeats=repeats, max_interpretable=max_interpretable)


    #ToDO
    # saving models
    # saving results to csv in a readable way
    # empty rules still being created
    # rulefit did not work on categorical data, introduced dummies.
    # re-introduce alpha into orb
    # only testing on very small datasets (300 points) -- potentially improvements with more features and rows. This could happen
    # because with small datasets it is very likely that rulefit will stumble into the best rules. <-- goto numpy implementation.
    # increase test data size. remove downsampling .
    # rulefit using Cs as [] or other
    # create permanent table...
    # still using old version of realkd
    # the write to all_results is busted
    # separate experimental structure, data_processing and expreriment driver into different files. This is too bulky