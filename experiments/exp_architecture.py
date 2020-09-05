import re
import os
import random
import datetime

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

from orb.rules import AdditiveRuleEnsemble, ExponentialObjective, SquaredLossObjective
import rulefit
import xgboost as xgb
from realkd.rules import GradientBoostingRuleEnsemble, LogisticLoss, SquaredLoss

from experiments.data_preprocessing import *

def generate_random_seed(max_seed=2**32 - 1):
    return random.randrange(max_seed)

def get_timestamp():
    now = datetime.datetime.now()
    timestamp = datetime.datetime.timestamp(now)
    dt_object = str(datetime.datetime.fromtimestamp(timestamp))

    dt_object = dt_object.replace(" ", "_")
    dt_object = dt_object.replace(":", "_")
    dt_object = dt_object.replace("-", "_")
    return dt_object

class DataIndependentEvaluator:

    def __init__(self, string, f):
        self.string = string
        self.f = f

    def __call__(self, model):
        return self.f(model)

    def __str__(self):
        return self.string


class DataDependentEvaluator:

    def __init__(self, string, f):
        self.string = string
        self.f = f

    def __call__(self, model, X=None, y=None):
        return self.f(model, X, y)

    def __str__(self):
        return self.string


def orb_complexity():
    def f(model):
        return sum([len(str(rule.__repr__()).split('&')) for rule in model.members])+len(model.members)

    return DataIndependentEvaluator('orb_complexity', f)

def RuleFit_complexity(): # add getting the most important rules
    def f(model):
#        no_rules = 10
        # model_no_ruls = 10
        rules = model.get_rules()
        rules = rules[rules.coef != 0].sort_values("support", ascending=False)#[:no_rules]
        return sum([(rule[1]['coef'] != 0) + len(rule[1]['rule'].split("&")) for rule in rules.iterrows()])
    return DataIndependentEvaluator('RuleFit_complexity', f)

def forest_complexity():
    def f(model):
        trees = model.get_booster().get_dump()
        trees = [tree.split('\n') for tree in trees]
        nodes = sum([len([node for node in tree if node.count(':') > 0]) for tree in trees])
        return nodes

    return DataIndependentEvaluator('forest_complexity', f)

def rmse():
    def f(model, X, y):
        return (sum((model.predict(X) - y) ** 2)/len(y)) ** 0.5

    return DataDependentEvaluator('rmse', f)

def accuracy():
    def f(model, X, y):
        return sum((model.predict(X)==y))/len(y)

    return DataDependentEvaluator('accuracy', f)

def rulefit_wrapper(objective= 'regress', max_rules = 2000, Cs=1):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode=objective, model_type='r', Cs=Cs)
    return rf

def rulefit_regression_wrapper(max_rules = 2000, Cs=1):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode="regress", model_type='r', Cs=Cs)
    return rf

def rulefit_classification_wrapper(max_rules = 2000, Cs=1):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode="classify", model_type='r', Cs=Cs)
    return rf

def rulefit_regression_wrapper_gb(objective = "ls", n_estimators = 10, max_depth = 10, alpha = 0.9, warmstart = False, learning_rate = 0.1, max_rules = 2000, Cs=1):
    gb = GradientBoostingRegressor(loss=objective, n_estimators=n_estimators, max_depth=max_depth, alpha=alpha, warm_start=warmstart, learning_rate=learning_rate) # switch to default
    rf = rulefit.RuleFit(tree_generator=gb, max_rules=max_rules, rfmode="regress", model_type='r', Cs=Cs)
    return rf

def rulefit_classification_wrapper_gb(objective = "exponential", n_estimators = 10, max_depth = 10, alpha = 0.9, warmstart = False, learning_rate = 0.1, max_rules = 2000, Cs=1):
    gb = GradientBoostingClassifier(loss=objective, n_estimators=n_estimators, max_depth=max_depth, alpha=alpha, warm_start=warmstart, learning_rate=learning_rate)
    rf = rulefit.RuleFit(tree_generator=gb, max_rules=max_rules, rfmode="classify", model_type='r', Cs=Cs)
    return rf

MODEL_MAP = {"orb": AdditiveRuleEnsemble,
             "xgb.XGBRegressor": xgb.XGBRegressor,
             "xgb.XGBClassifier": xgb.XGBClassifier,
             "rulefit_classification": rulefit_classification_wrapper,
             "rulefit_regression": rulefit_regression_wrapper,
             "rulefit": rulefit_wrapper,
             "realkd": GradientBoostingRuleEnsemble
             }

PROBLEM_MAP = {"SquaredLossObjective": SquaredLossObjective,
               "ExponentialObjective": ExponentialObjective,
               "binary:logistic": "binary:logistic",
               "reg:squarederror": "reg:squarederror",
               "logistic": "logistic",
               "ls": "ls",
               'regress': 'regress',
               'classify': 'classify',
               'logistic_loss': LogisticLoss,
               'squared_loss': SquaredLoss
               }

class Experiment:
    def __init__(self, dataset_name, model_class="orb", objective="ExponentialObjective", params={'k':2, 'reg': 50},
                 dde = [rmse], die = [orb_complexity], downsample_size = None):
        self.dataset = dataset_name
        self.model_class_key = model_class
        self.model_class = MODEL_MAP[model_class]
        self.params = params
        self.dde = dde
        self.die = die
        self.seed = generate_random_seed()
        self.downsample_size = downsample_size
        self.objective = PROBLEM_MAP[objective]
        self.norm_mean = True if model_class == "xgboost" else False
        self.results = dict()

    def run(self):
        target_name, without = dataset_signature(self.dataset)
        data, target, seed = exp_data_prep(dataset_name=self.dataset,
                                           model_class=self.model_class_key, random_seed=self.seed,
                                           downsample_size=self.downsample_size, norm_mean=self.norm_mean)
        [train, train_target, test, test_target] = data

        if self.model_class_key == 'realkd':
            model = self.model_class(**self.params, loss=self.objective)
        else:
            model = self.model_class(**self.params, objective=self.objective)

        if self.model_class_key == 'rulefit': # this clearly needs to be improved
            features = train.columns
            train = train.values
            test = test.values
            model.fit(train, train_target, feature_names=features)
        elif self.model_class_key == 'realkd':
            model.fit(train, train_target, verbose=True)
        else:
            model.fit(train, train_target)

        for evaluator in self.die:
            eval_metric = evaluator()
            self.results[evaluator.__name__] = eval_metric(model)

        for evaluator in self.dde: # failing here
            eval_metric = evaluator()
            self.results[evaluator.__name__+"_train"] = eval_metric(model, train, train_target)
            self.results[evaluator.__name__+"_test"] = eval_metric(model, test, test_target)

        res_dict = {"dataset": self.dataset,
                    "target": target_name,
                    "no_feat": len(features) if self.model_class_key == 'rulefit' else len(train.columns),
                    "no_rows": len(train_target)+len(test_target),
                    "model_params": {**self.params},
                    "objective_function": self.objective if type(self.objective) == str else self.objective.__name__,
                    "model_class": self.model_class_key,
                    "timestamp": get_timestamp(),
                    "random_seed": generate_random_seed()}

        for key, value in self.results.items():
            res_dict[key] = value

        return res_dict, model