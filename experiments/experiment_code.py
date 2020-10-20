import os
import random
import datetime

import numpy as np
import pandas as pd

from time import time
from statistics import median, mean
from csv import DictWriter

from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2

import rulefit
from realkd.rules import GradientBoostingRuleEnsemble, LogisticLoss, SquaredLoss
from realkd import evaluation

from dataset_prep import *

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

def rulefit_complexity():  # add getting the most important rules
    def f(model):
        rules = model.get_rules()
        rules = rules[rules.coef != 0].sort_values("support", ascending=False)
        return sum([(rule[1]['coef'] != 0) + len(rule[1]['rule'].split("&")) for rule in rules.iterrows()])
    return DataIndependentEvaluator('rulefit_complexity', f)


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
        return sum((model.predict(X) == y))/len(y)

    return DataDependentEvaluator('accuracy', f)


def rulefit_wrapper(objective= 'regress', max_rules=2000, Cs=1, random_state=None):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode=objective, model_type='r', Cs=Cs, random_state=random_state)
    return rf


def rulefit_regression_wrapper(max_rules = 2000, Cs=1):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode="regress", model_type='r', Cs=Cs)
    return rf


def rulefit_classification_wrapper(max_rules = 2000, Cs=1):
    rf = rulefit.RuleFit(max_rules=max_rules, rfmode="classify", model_type='r', Cs=Cs)
    return rf


def rulefit_regression_wrapper_gb(objective = "ls", n_estimators = 10, max_depth = 10, warmstart = False, learning_rate = 0.1, max_rules = 2000, Cs=1):
    gb = GradientBoostingRegressor(loss=objective, n_estimators=n_estimators, max_depth=max_depth, warm_start=warmstart, learning_rate=learning_rate) # switch to default
    rf = rulefit.RuleFit(tree_generator=gb, max_rules=max_rules, rfmode="regress", model_type='r', Cs=Cs)
    return rf


def rulefit_classification_wrapper_gb(objective = "exponential", n_estimators = 10, max_depth = 10, warmstart = False, learning_rate = 0.1, max_rules = 2000, Cs=1):
    gb = GradientBoostingClassifier(loss=objective, n_estimators=n_estimators, max_depth=max_depth, warm_start=warmstart, learning_rate=learning_rate)
    rf = rulefit.RuleFit(tree_generator=gb, max_rules=max_rules, rfmode="classify", model_type='r', Cs=Cs)
    return rf


MODEL_MAP = {"rulefit_classification": rulefit_classification_wrapper,
             "rulefit_regression": rulefit_regression_wrapper,
             "rulefit": rulefit_wrapper,
             "realkd": GradientBoostingRuleEnsemble
             }

PROBLEM_MAP = {"binary:logistic": "binary:logistic",
               "reg:squarederror": "reg:squarederror",
               "logistic": "logistic",
               "ls": "ls",
               'regress': 'regress',
               'classify': 'classify',
               'logistic': LogisticLoss.__str__(),
               'squared': SquaredLoss.__str__()
               }

EVAL_MAP = {'orb_complexity': orb_complexity,
            'rulefit_complexity': rulefit_complexity,
            'forest_complexity': forest_complexity,
            'rmse': rmse,
            'accuracy': accuracy}

def rmse_(data, target):

    def metric(model):
        return (sum((model.predict(data) - target) ** 2)/len(target)) ** 0.5

    return metric

METRIC_MAP = {'accuracy': evaluation.accuracy,
              'roc_auc': evaluation.roc_auc,
              'log_loss': evaluation.log_loss,
              'rmse': rmse_,
              'r2': evaluation.r2}

DISCRETISATION_MAP = {'qcut': pd.qcut,
                      'cut': pd.cut}

PREMADE_DATASET_LIST = ['tic-tac-toe', 'breast_cancer', 'iris', 'kr - vs - kp',
                        'make_friedman1', 'make_friedman2', 'make_friedman3',
                        'make_classification2', 'make_classification3', 'load_wine', 'make_hastie_10_2',
                        'noisy_pairity_', 'noisy_pairity_1', 'noisy_pairity_2', 'noisy_pairity_3',
                        'noisy_pairity_4', 'noisy_pairity_5', 'noisy_pairity_6', 'digits5', 'load_diabetes']

class Experiment:
    def __init__(self, dataset_name, model_class="realkd", objective="logistic", params={'k':3, 'reg': 50},
                 metrics = ['roc_auc'], downsample_size = None, seed = None, test_size = 0.2, pos_class=None):
        self.dataset = dataset_name
        self.model_class_key = model_class
        self.model_class = MODEL_MAP[model_class]
        self.params = params
        self.metrics = metrics
        self.seed = generate_random_seed() if seed is None else seed
        self.downsample_size = downsample_size
        self.objective = PROBLEM_MAP[objective]
        self.norm_mean = True if model_class == "xgboost" else False
        self.results = dict()
        self.test_size = test_size
        self.pos_class = pos_class


    def run(self):
        target_name, without = dataset_signature(self.dataset)

        if self.dataset in PREMADE_DATASET_LIST:
            data_prepper = prep_data_sklearn
        else:
            data_prepper = exp_data_prep

        data, target, seed = data_prepper(dataset_name=self.dataset,
                                            model_class=self.model_class_key, random_seed=self.seed,
                                            downsample_size=self.downsample_size, norm_mean=self.norm_mean,
                                            test_size = self.test_size, pos_class=self.pos_class)

        [train, train_target, test, test_target] = data

        if self.model_class_key == 'realkd':
            model = self.model_class(**self.params, loss=self.objective)
        else:
            model = self.model_class(**self.params, objective=self.objective)

        if self.model_class_key == 'rulefit': # this clearly needs to be improved
            features = train.columns
            train = train.values
            test = test.values
            start = time()
            model.fit(train, train_target, feature_names=features)
        elif self.model_class_key == 'realkd':
            start = time()
            real_max = model.max_rules
            model.max_rules = 1
            while model.max_rules <= real_max:
                model.fit(train, train_target, verbose=3)
                model.max_rules += 1
#                print(model.members)
        else:
            start = time()
            model.fit(train, train_target)

        self.fit_time = time() - start

        for metric_name in self.metrics:
            rule_lengths = []
            if self.model_class_key == 'rulefit':
                rules = model.get_rules()
                useful_rules = rules[rules.coef != 0].sort_values("support", ascending=False)
                for idx, rule in useful_rules.iterrows():
                    rule_lengths.append((len(rule['rule'].split('&'))))
                no_rules = len(useful_rules)

                if metric_name == 'r2':
                    test_score = skl_r2(test_target, model.predict(test))
                    train_score = skl_r2(train_target, model.predict(train))
                elif metric_name == 'roc_auc':
                    test_score = skl_auc(test_target, model.predict(test))
                    train_score = skl_auc(train_target, model.predict(train))
                else:
                    test_score = None
                    train_score = None

                self.results[metric_name+'_test'] = (no_rules, test_score)
                self.results[metric_name+'_train'] = (no_rules, train_score)


            else:
                eval_metric = METRIC_MAP[metric_name](test, test_target)
                res = evaluation.ensemble_length_vs_perf(model, eval_metric)
                self.results[metric_name+'_length_vs_perf'+'_test'] = res

                eval_metric = METRIC_MAP[metric_name](train, train_target)
                res = evaluation.ensemble_length_vs_perf(model, eval_metric)
                self.results[metric_name+'_length_vs_perf'+'_train'] = res

                rule_lengths = [len(str(rule.q).split('&')) for rule in model.members]

            self.results['rule_lengths'] = rule_lengths


        res_dict = {"dataset": self.dataset,
                    "target": target_name,
                    "no_feat": len(features) if self.model_class_key == 'rulefit' else len(train.columns),
                    "no_rows": len(train_target)+len(test_target),
                    "model_params": {**self.params},
                    "objective_function": self.objective if type(self.objective) == str else self.objective.__name__,
                    "model_class": self.model_class_key,
                    "random_seed": seed}

        for key, value in self.results.items():
            res_dict[key] = value

        return res_dict, model


def rulefit_sweep(fixed_params, variable_lst, max_rules):
    test_res_vect = []
    train_res_vect = []
    best_rulefit = None
    best_rulefit_score = None

    fit_times = []
    max_fitted_rules = 0
    rule_lengths = []
    for params in variable_lst:
        print(params)
        fixed_params['params'] = params
        metric = fixed_params['metrics'][0]
        exp = Experiment(**fixed_params)
        res, model = exp.run()
        rule_lengths += exp.results['rule_lengths']
        if (best_rulefit == None) or ((exp.results[metric+'_test'][1] > best_rulefit_score) and (exp.results[metric+'_test'][0] <= max_rules)):
            best_rulefit = model
            best_rulefit_score = exp.results[metric+'_test'][1]
            no_rules = exp.results[metric + '_test'][0]
            if no_rules > max_fitted_rules:
                max_fitted_rules = no_rules
            fit_times.append((no_rules, exp.fit_time))

        print(exp.results[metric+'_test'], exp.results[metric+'_train'])
        test_res_vect.append(exp.results[metric+'_test'])
        train_res_vect.append(exp.results[metric+'_train'])

    average_rule_length = sum(rule_lengths) / len(rule_lengths)

    fit_times.sort()
    filtered_fit_times = list(filter(lambda item: item[0] == max_fitted_rules, fit_times))
    fit_times = [item[1] for item in filtered_fit_times]
    average_fit_time = sum(fit_times)/len(fit_times)

    test_res_vect = list(filter(lambda item: item[0] <= max_rules, test_res_vect))
    train_res_vect = list(filter(lambda item: item[0] <= max_rules, train_res_vect))

    if fixed_params['objective'] == 'regress':
        test_res_vect = list(filter(lambda item: item[0] > 0, test_res_vect))
        train_res_vect = list(filter(lambda item: item[0] > 0, train_res_vect))
    test_res_vect.sort(key=lambda item: item[0])
    train_res_vect.sort(key=lambda item: item[0])


    x, y = zip(*test_res_vect)
    x2,y2 = zip(*train_res_vect)

    x, x2 = list(x), list(x2)
    y, y2 = list(y), list(y2)

    xy_dict,xy_dict2 = dict(), dict()
    for i in range(len(x)):
        x_key = str(x[i])
        if x_key in xy_dict:
            xy_dict[x_key].append(y[i])
        else:
            xy_dict[x_key] = [y[i]]

        x_key = str(x2[i])
        if x_key in xy_dict2:
            xy_dict2[x_key].append(y2[i])
        else:
            xy_dict2[x_key] = [y2[i]]

    x_keys = xy_dict.keys()
    x_keys2 = xy_dict2.keys()


    x, x2 = [], []
    y, y2 = [], []
    for key in x_keys:
        x.append(int(key))
        y.append(median(xy_dict[key]))

    for key in x_keys2:
        x2.append(int(key))
        y2.append(median(xy_dict2[key]))


    while x[-1] < max_rules:
        x.append(x[-1]+1)
        y.append(y[-1])

    while x2[-1] < max_rules:
        x2.append(x2[-1]+1)
        y2.append(y2[-1])





    return (x,y), (x2,y2), average_fit_time, average_rule_length, best_rulefit

def full_comparision(dataset_names, all_datasets, showplot = False, random_seed = None,
                     new_header = True, xerr=0, yerr=0.1, save_models=True):

    res_df = pd.DataFrame()

    returned_models = dict()
    figures = []

    for dataset_name in dataset_names:

        problem_type, downsample_size, max_rules, realkd_reg, max_col_attr, rulefit_reg,\
        test_size, repeats, pos_class, opt_max_rules = \
            all_datasets[dataset_name]

        (min_Cs, max_Cs, no_Cs) = rulefit_reg
        (greedy_reg, opt_reg) = realkd_reg

        rulefit_aucs_test = []
        rulefit_aucs_train = []
        realkd_greedy_aucs_test = []
        realkd_greedy_aucs_train = []
        realkd_optimal_aucs_test = []
        realkd_optimal_aucs_train = []
        seeds = []

        for rep in range(repeats):
            random_seed = generate_random_seed() if ((random_seed is None) or (repeats != 1)) else random_seed

            seeds.append(random_seed)

            rulefit_fixed = {'dataset_name': dataset_name,
                             'model_class': 'rulefit',
                             'objective': 'classify' if problem_type == 'c' else 'regress',
                             'metrics': ['roc_auc'] if problem_type == 'c' else ['r2'],
                             'downsample_size': downsample_size,
                             'seed': random_seed,
                             'test_size': test_size,
                             'pos_class': pos_class}

            realkd_fixed = {'dataset_name': dataset_name,
                            'model_class': 'realkd',
                            'objective': 'logistic' if problem_type == 'c' else 'squared',
                            'metrics': ['roc_auc'] if problem_type == 'c' else ['r2'],
                            'downsample_size': downsample_size,
                            'seed': random_seed,
                            'test_size': test_size,
                            'pos_class': pos_class}

            realkd_greedy_params = {'max_rules': max_rules, 'reg': greedy_reg,
                                    'max_col_attr': max_col_attr, 'method': 'greedy',
                                    'offset_rule': False}

            realkd_opt_params = {'max_rules': max_rules if opt_max_rules is None else opt_max_rules,
                                 'reg': opt_reg,
                                 'max_col_attr': max_col_attr, 'method': 'bestboundfirst',
                                 'offset_rule': False}
            Cs_lst = np.linspace(min_Cs, max_Cs, no_Cs, endpoint=True)
            Cs_lst = [np.array([Cs_lst[i]]) for i in range(len(Cs_lst))]
            rulefit_params_lst = [{'Cs': Cs, 'random_state': random_seed} for Cs in Cs_lst]

            (rulefit_x_test, rulefit_y_test), (rulefit_x_train, rulefit_y_train), rulefit_time,\
            rulefit_rule_length, best_rulefit = \
                rulefit_sweep(rulefit_fixed, rulefit_params_lst, max_rules)
            rulefit_auc_test = np.trapz(rulefit_y_test, rulefit_x_test) / (
                        max(rulefit_x_test) - min(rulefit_x_test))  # + int(problem_type == 'c'))
            rulefit_auc_train = np.trapz(rulefit_y_train, rulefit_x_train) / (
                        max(rulefit_x_train) - min(rulefit_x_train))  # + int(problem_type == 'c'))

            rulefit_aucs_test.append(rulefit_auc_test)
            rulefit_aucs_train.append(rulefit_auc_train)

            best_rulefit_rules = best_rulefit.get_rules()
            best_rulefit_rules = best_rulefit_rules[best_rulefit_rules.coef != 0].sort_values("support",
                                                                                              ascending=False)
            for idx, rule in best_rulefit_rules.iterrows():
                print(('+' if rule['coef'] > 0 else '') + str(rule['coef']) + ' if ' + rule['rule'])
            for i in range(len(rulefit_x_test)):
                print(rulefit_x_test[i], rulefit_y_test[i])
            print(rulefit_auc_test)

            greedy_exp = Experiment(**realkd_fixed, params=realkd_greedy_params)
            realkd_greedy_df, greedy_model = greedy_exp.run()
            greedy_rule_length = sum(greedy_exp.results['rule_lengths'])/len(greedy_exp.results['rule_lengths'])

            len_vs_perf_greedy_test = greedy_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_test'][int(problem_type == 'r'):]
            len_vs_perf_greedy_train = greedy_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_train'][int(problem_type == 'r'):]
            print(len_vs_perf_greedy_test)

            realkd_greedy_auc_test = np.trapz(len_vs_perf_greedy_test.values, len_vs_perf_greedy_test.index)/(max_rules - int(problem_type=='r'))
            realkd_greedy_auc_train = np.trapz(len_vs_perf_greedy_train.values, len_vs_perf_greedy_train.index)/(max_rules - int(problem_type=='r'))

            realkd_greedy_aucs_test.append(realkd_greedy_auc_test)
            realkd_greedy_aucs_train.append(realkd_greedy_auc_train)
            print(realkd_greedy_auc_test)

            opt_exp = Experiment(**realkd_fixed, params=realkd_opt_params)
            realkd_opt_df, opt_model = opt_exp.run()
            len_vs_perf_opt_test = opt_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf' + '_test'][
                                   int(problem_type == 'r'):]
            len_vs_perf_opt_train = opt_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf' + '_train'][
                                    int(problem_type == 'r'):]

            while len_vs_perf_opt_test.index[-1] < max_rules:
                len_vs_perf_opt_test.loc[len_vs_perf_opt_test.index[-1] + 1] = len_vs_perf_opt_test.values[-1]
                len_vs_perf_opt_train.loc[len_vs_perf_opt_train.index[-1] + 1] = len_vs_perf_opt_train.values[-1]

            print(len_vs_perf_opt_test)
            realkd_opt_auc_test = np.trapz(len_vs_perf_opt_test.values, len_vs_perf_opt_test.index) / (
                    max_rules - int(problem_type == 'r'))
            realkd_opt_auc_train = np.trapz(len_vs_perf_opt_train.values, len_vs_perf_opt_train.index) / (
                    max_rules - int(problem_type == 'r'))

            realkd_optimal_aucs_test.append(realkd_opt_auc_test)
            realkd_optimal_aucs_train.append(realkd_opt_auc_train)
            print(realkd_opt_auc_test)

            opt_rule_length = sum(opt_exp.results['rule_lengths']) / len(opt_exp.results['rule_lengths'])

            plt.errorbar(rulefit_x_test, rulefit_y_test, xerr=[xerr for x in rulefit_x_test],
                            yerr=[yerr for y in rulefit_y_test], fmt='+', ecolor='b', label='rulefit')
            plt.plot(rulefit_x_test, rulefit_y_test, 'b-')

            plt.errorbar(len_vs_perf_greedy_test.index, len_vs_perf_greedy_test.values,
                         xerr=[xerr for x in len_vs_perf_greedy_test],
                         yerr=[yerr for y in len_vs_perf_greedy_test],
                         fmt='+', ecolor='r', label='realkd_greedy')
            plt.plot(len_vs_perf_greedy_test.index, len_vs_perf_greedy_test.values, 'r-')

            plt.errorbar(len_vs_perf_opt_test.index, len_vs_perf_opt_test.values,
                         xerr=[xerr for x in len_vs_perf_opt_test],
                         yerr=[yerr for y in len_vs_perf_opt_test],
                         fmt='+', ecolor='g', label='realkd_opt')
            plt.plot(len_vs_perf_opt_test.index, len_vs_perf_opt_test.values, 'g-')
            plt.xlabel('number of rules')
            plt.ylabel('roc_auc' if problem_type == 'c' else 'r2')
            plt.title(dataset_name)
            plt.legend()


            if save_models:
                to_save = [(None, realkd_fixed), (opt_model, realkd_opt_params), (greedy_model, realkd_greedy_params),
                           (None, rulefit_fixed), (best_rulefit, {'params': rulefit_params_lst})]
                save_dir = save_three_models(dataset_name, random_seed, to_save)
                plt.savefig(os.path.join(save_dir, 'comparison_plot' + str(rep)))

            if showplot:
                plt.show()

            res_dict = {'dataset': dataset_name,
                        'problem_type': problem_type,
                        'no_feat': int(realkd_opt_df['no_feat']),
                        'no_rows': int(realkd_opt_df['no_rows']),
                        'max_rules': max_rules,
                        'rulefit_auc_test': rulefit_auc_test,
                        'rulefit_auc_train': rulefit_auc_train,
                        'rulefit_reg': rulefit_reg,
                        'rulefit_time': rulefit_time,
                        'rulefit_rule_length': rulefit_rule_length,
                        'greedy_auc_test': realkd_greedy_auc_test,
                        'greedy_auc_train': realkd_greedy_auc_train,
                        'greedy_reg': greedy_reg,
                        'greedy_time': greedy_exp.fit_time,
                        'greedy_rule_length': greedy_rule_length,
                        'opt_auc_test': realkd_opt_auc_test,
                        'opt_auc_train': realkd_opt_auc_train,
                        'opt_reg': opt_reg,
                        'opt_max_rules': opt_max_rules,
                        'opt_time': opt_exp.fit_time,
                        'opt_rule_length': opt_rule_length,
                        'random_seed': random_seed,
                        'model_dir': save_dir if save_models else None,
                        'rulefit_len_v_perf_test': [rulefit_x_test, rulefit_y_test],
                        'rulefit_len_v_perf_train': [rulefit_x_train, rulefit_y_train],
                        'greedy_len_v_perf_test': [len_vs_perf_greedy_test.index, len_vs_perf_greedy_test.values],
                        'greedy_len_v_perf_train': [len_vs_perf_greedy_train.index, len_vs_perf_greedy_train.values],
                        'opt_len_v_perf_test': [len_vs_perf_opt_test.index, len_vs_perf_opt_test.values],
                        'opt_len_v_perf_train': [len_vs_perf_opt_train.index, len_vs_perf_opt_train.values]

                        }

            fieldnames_all_runs = res_dict.keys()
            with open('new_results_table.csv', 'a') as f: #datasets_perf_all_runs
                all_res_csv_writer = DictWriter(f, fieldnames=fieldnames_all_runs)
                if new_header:
                    all_res_csv_writer.writeheader()
                    new_header = False
                all_res_csv_writer.writerow(res_dict)

        dataset_results = pd.Series({'dataset': dataset_name,
                                     'problem_type': problem_type,
                                     'no_feat': int(realkd_opt_df['no_feat']),
                                     'no_rows': int(realkd_opt_df['no_rows']),
                                     'rulefit_auc_test': median(rulefit_aucs_test),
                                     'rulefit_auc_train': median(rulefit_aucs_train),
                                     'rulefit_reg': rulefit_reg,
                                     'realkd_greedy_auc_test': median(realkd_greedy_aucs_test),
                                     'realkd_greedy_auc_train': median(realkd_greedy_aucs_train),
                                     'greedy_reg': greedy_reg,
                                     'realkd_optimal_auc_test': median(realkd_optimal_aucs_test),
                                     'realkd_optimal_auc_train': median(realkd_optimal_aucs_train),
                                     'optimal_reg': opt_reg,
                                     'random_seeds': seeds})

        print(dataset_results)

#        res_df = res_df.append(dataset_results, ignore_index=True)

        returned_models[dataset_name] = {'rulefit': best_rulefit, 'greedy': greedy_model, 'optimal': opt_model}

    '''
    fieldnames = res_df.columns
    with open('datasets_perf.csv', 'w') as f:
        csv_writer = DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        for idx, row in res_df.iterrows():
            csv_writer.writerow(row.to_dict())
    '''

    return returned_models

def cummulative_trapz(y, x, problem_type):
    return [np.trapz(y[:i], x[:i]) / (x[:i][-1] - x[0] + int(problem_type == 'c'))
            if i > 0 else np.trapz([y[0]], [x[0]])
            for i in range(len(x))]


def save_three_models(dataset_name, seed, to_save):
    res_dir = os.path.join(os.getcwd(), 'results', str(seed))    # random_seed

    try:
        os.makedirs(res_dir)
    except FileExistsError:
        pass

    with open(os.path.join(res_dir, dataset_name+'.txt'), 'a') as f:
        for model, specs in to_save:
            print(specs)
            for key in specs.keys():
                value = specs[key]
                f.write(key + ': ' + str(value) + '\n')
#            f.write(json.dumps(specs))

            if not (model is None):
                if len(list(specs.keys())) == 1:
                    rules = model.get_rules()
                    rules = rules[rules.coef != 0].sort_values("support", ascending=False)
                    for idx, rule in rules.iterrows():
                        f.write(('+' if rule['coef'] > 0 else '') + str(rule['coef']) + ' if ' + rule['rule'] + '\n')
                else:
                    rules = model.members
                    for rule in rules:
                        f.write(str(rule) + '\n')

    return res_dir

def unrestricted_models(dataset_name, random_seeds, dataset_params):

        res_df = pd.DataFrame(columns=['dataset_name', 'Cs', 'random_seed', 'no_rules', 'test', 'train'])

        problem_type, Cs_lst, test_size, pos_class = dataset_params

        for random_seed in random_seeds:

            rulefit_fixed = {'dataset_name': dataset_name,
                             'model_class': 'rulefit',
                             'objective': 'classify' if problem_type == 'c' else 'regress',
                             'metrics': ['roc_auc'] if problem_type == 'c' else ['r2'],
                             'downsample_size': None,
                             'seed': random_seed,
                             'test_size': test_size,
                             'pos_class': pos_class}

            rulefit_params_lst = [{'Cs': np.array([Cs]), 'random_state': random_seed} for Cs in Cs_lst]

            for params in rulefit_params_lst:
                print(dataset_name, params)
                rulefit_fixed['params'] = params

                exp = Experiment(**rulefit_fixed)
                exp.run()

                (no_rules, test_score) = exp.results[rulefit_fixed['metrics'][0]+'_test']
                (no_rules, train_score) = exp.results[rulefit_fixed['metrics'][0]+'_train']

                print(no_rules, train_score, test_score)

                res_dict = {'dataset_name': dataset_name,
                            'Cs': rulefit_fixed['params']['Cs'][0],
                            'random_seed': random_seed,
                            'no_rules': no_rules,
                            'test': test_score,
                            'train': train_score}

                res_df = res_df.append(res_dict, ignore_index=True)

        return res_df


def rules_vs_compute(dataset_name, max_rules, all_datasets, random_seed = None):
    problem_type, downsample_size, old_max_rules, (grd_reg, opt_reg), max_col_attr, rulefit_reg, \
    test_size, repeats, pos_class, opt_max_rules = all_datasets[dataset_name]


    if dataset_name in PREMADE_DATASET_LIST:
        data_prepper = prep_data_sklearn
    else:
        data_prepper = exp_data_prep

    data, target_name, random_seed = data_prepper(dataset_name, pos_class=pos_class, random_seed=random_seed)
    [x_train, y_train, x_test, y_test] = data

    loss = SquaredLoss if problem_type == 'r' else LogisticLoss

    model = GradientBoostingRuleEnsemble(max_rules=1, loss=loss, reg=opt_reg, max_col_attr=max_col_attr)
    start = time()
    model.fit(x_train, y_train, verbose=2)
    rules_time = [(model.max_rules, time() - start)]
    while model.max_rules < max_rules:
        model.max_rules+=1
        model.fit(x_train, y_train, verbose=2)
        rules_time.append((model.max_rules, time()-start))

    return rules_time


def regs_vs_compute(dataset_name, regs, max_rules, all_datasets, random_seed = None):
    problem_type, downsample_size, old_max_rules, (grd_reg, opt_reg), max_col_attr, rulefit_reg, \
    test_size, repeats, pos_class, opt_max_rules = all_datasets[dataset_name]


    if dataset_name in PREMADE_DATASET_LIST:
        data_prepper = prep_data_sklearn
    else:
        data_prepper = exp_data_prep

    data, target_name, random_seed = data_prepper(dataset_name, pos_class=pos_class, random_seed=random_seed)
    [x_train, y_train, x_test, y_test] = data

    loss = SquaredLoss if problem_type == 'r' else LogisticLoss

    regs_time = []
    for reg in regs:
        model = GradientBoostingRuleEnsemble(max_rules=1, loss=loss, reg=reg, max_col_attr=max_col_attr)
        start = time()
        model.fit(x_train, y_train, verbose=2)
        while model.max_rules < max_rules:
            model.max_rules+=1
            model.fit(x_train, y_train, verbose=2)
        regs_time.append((reg, time() - start))

    return regs_time