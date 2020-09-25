import os
import re
import random
import datetime

import numpy as np
import pandas as pd

from numpy import where
from copy import deepcopy
from time import time
from statistics import median, mean
from csv import DictWriter

from sklearn import datasets as ds
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score as skl_auc
from sklearn.metrics import r2_score as skl_r2

import rulefit
import xgboost as xgb
from orb.rules import AdditiveRuleEnsemble, ExponentialObjective, SquaredLossObjective
from realkd.rules import GradientBoostingRuleEnsemble, LogisticLoss, SquaredLoss
from realkd import evaluation

from numpy import exp, log2, log
from realkd.patch import RuleFit as RF
from realkd.datasets import noisy_parity

def prep_noisy_pairity(d=5, random_seed=None):

    n = 100 * 2 ** d

    x_train, y_train = noisy_parity(n, d, random_seed=random_seed)
    x_test, y_test = noisy_parity(2000, d, random_seed=random_seed+1)

    target = 'y'

    data = [x_train, y_train, x_test, y_test]

    return data, target, random_seed




def dataset_signature(dataset_name):
    data = {"advertising": ("Clicked_on_Ad", ["Ad_Topic_Line", "City", "Timestamp"], []),
            "avocado_prices": ("AveragePrice", ["Date", "group"], []),
            "cdc_physical_activity_obesity": ("Data_Value",
                                              ["Data_Value_Alt", "LocationAbbr", "Data_Value_Footnote_Symbol",
                                               "Data_Value_Footnote", "GeoLocation",
                                               "ClassID", "TopicID", "QuestionID", "DataValueTypeID", "LocationID",
                                               "StratificationCategory1",
                                               "Stratification1", "StratificationCategoryId1", "StratificationID1"],
                                              []),
            "gdp_vs_satisfaction": ("Satisfaction", ["Country"], []),
            "halloween_candy_ranking": (
            "winpercent", ['competitorname'], ['very bad', 'bad', 'average', 'good', 'very good']),
            "metacritic_games": ("user_score", ["GameID", "name", "players", "attribute", "release_date", "link", "rating", "user_positive", "user_neutral",	"user_negative"], []),
            "random_regression": (None, [], []),  # "" not appropriate as null input, potentiall swap to None
            "red_wine_quality": ("quality", [], []),
#            "suicide_rates": ("suicides/100k pop", ["suicides_no", "population", "HDI for year"], []),
            "suicide_rates_cleaned": ("suicides/100k pop", ["suicides_no", "population", "HDI for year"], []),
            "titanic": ("Survived", ['PassengerId', 'Name', 'Ticket', 'Cabin'], []),
            "us_minimum_wage_by_state": ("CPI.Average", ["Table_Data", "Footnote"], []),  # may be wrong target
            "used_cars": ("avgPrice", ['minPrice','maxPrice','sdPrice'], []), # count,km,year,powerPS,
            "wages_demographics": ("earn", [], []),
            "who_life_expectancy": ("Life expectancy", [], []),
            "world_happiness_indicator": ("Happiness_Score", ["Country", "Happiness_Rank"], []),
            "boston": ("MEDV", [], []),
            'tic-tac-toe': ('positive', [], []),
            'breast_cancer': ('malignant_or_benign', [], []),
            'iris': ('versicolor', [], []),
            'Demographics':  ('1    ANNUAL INCOME OF HOUSEHOLD (PERSONAL INCOME IF SINGLE)', ['idx'], []),
            'digits': ('target', [], []),
            'kr - vs - kp': ('1', [], []),
            'make_friedman1': ('y', [], []),
            'make_friedman2': ('y', [], []),
            'make_friedman3': ('y', [], []),
            'make_classification2': ('1', [], []),
            'make_classification3': ('1', [], []),
            'load_wine': ('1', [], []),
            'make_hastie_10_2': ('1', [], []),
            'noisy_pairity_': ('y', [], []),
            'noisy_pairity_1': ('y', [], []),
            'noisy_pairity_2': ('y', [], []),
            'noisy_pairity_3': ('y', [], []),
            'noisy_pairity_4': ('y', [], []),
            'noisy_pairity_5': ('y', [], []),
            'noisy_pairity_6': ('y', [], [])
            }

    dataset_info = data[dataset_name]
    target = dataset_info[0]
    without = dataset_info[1]
    return target, without

def prep_data_sklearn(dataset_name, test_size=0.2, model_class='realkd', downsample_size=None, norm_mean=False,
                      random_seed=None, pos_class=None):

    target_name, without = dataset_signature(dataset_name)

    if dataset_name == 'tic-tac-toe':
        bunch = ds.fetch_openml(dataset_name)
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df.rename(lambda s: s[:-7], axis='columns', inplace=True)
        df.replace(0, 'b', inplace=True)
        df.replace(1, 'o', inplace=True)
        df.replace(2, 'x', inplace=True)
        data_rf = pd.get_dummies(df)
        target = pd.Series(where(bunch.target == 'positive', 1, -1))
    elif dataset_name == 'kr - vs - kp':
        bunch = ds.fetch_openml(data_id=3)
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        data_rf = pd.get_dummies(df)
        target = pd.Series(where(bunch.target == 'won', 1, -1))
    elif dataset_name == 'breast_cancer':
        bunch = ds.load_breast_cancer()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        data_rf = pd.get_dummies(df)
        target = pd.Series(where(bunch.target == 1, 1, -1))
    elif dataset_name == 'iris':
        bunch = ds.load_iris()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        data_rf = pd.get_dummies(df)
        target = pd.Series(where(bunch.target == 1, 1, -1))
    elif dataset_name == 'make_friedman1':
        data, target = ds.make_friedman1(n_samples=2000, n_features=global_friedman_cols, noise=0.1, random_state=random_seed) # 1
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i+1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(target)
    elif dataset_name == 'make_friedman2':
        data, target = ds.make_friedman2(n_samples=2000, noise=0.1, random_state=random_seed) # 1
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i+1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(target)
    elif dataset_name == 'make_friedman3':
        data, target = ds.make_friedman3(n_samples=2000, noise=0.1, random_state=random_seed)
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i+1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(target)
    elif dataset_name == 'make_classification2':
        data, target = ds.make_classification(n_samples=2000, n_features=8, n_classes=2,
                                              hypercube=True, n_clusters_per_class=3,
                                              n_informative=3, n_redundant=3, n_repeated=0,
                                              random_state=random_seed)
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i+1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(where(target == 1, 1, -1))
    elif dataset_name == 'make_classification3':
        data, target = ds.make_classification(n_samples=2000, n_features=15, n_classes=3,
                                              hypercube=True, n_clusters_per_class=3,
                                              n_informative=5, n_redundant=5, n_repeated=0,
                                              random_state=random_seed)
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i + 1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(where(target == 1, 1, -1))
    elif dataset_name == 'load_wine':
        bunch = ds.load_wine()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        data_rf = pd.get_dummies(df)
        target = pd.Series(where(bunch.target == 1, 1, -1))
    elif dataset_name == 'make_hastie_10_2':
        data, target = ds.make_hastie_10_2(n_samples=12000, random_state=random_seed)
        no_cols = np.size(data, 1)
        col_names = ['X' + str(i+1) for i in range(no_cols)]
        data_rf = pd.DataFrame(data, columns=col_names)
        target = pd.Series(where(target == 1, 1, -1))
    elif dataset_name[:-1] == 'noisy_pairity_':
        d = int(dataset_name[-1])
        data, target_name, random_seed = prep_noisy_pairity(d=d, random_seed=random_seed)
        return data, target_name, random_seed

    x_train, x_test, y_train, y_test = train_test_split(data_rf, target, test_size=test_size, random_state=random_seed)

    if downsample_size != None:
        x_train[target_name] = y_train
        sampled_train = x_train.sample(n=min(downsample_size, len(y_train)), random_state=random_seed)
        x_train.reset_index(inplace=True, drop=True) # this may be unncessesary
        y_train = sampled_train[target_name]
        x_train = sampled_train.drop([target_name], axis='columns')

    if norm_mean:  # scikitlearn transformer.
        target_train_mean = sum(y_train) / len(y_train)
        y_train -= target_train_mean
        y_test -= target_train_mean

        y_train = [y_train, target_train_mean]
        y_test = [y_test, target_train_mean]


    data = [x_train, y_train, x_test, y_test]

    n = (len(y_train), len(y_test))

    return data, target_name, random_seed

def prep_data(dataset_name, target=None, without=[], model_class='xgboost', norm_mean=False,
              random_seed = None, test_size = 0.2, pos_class = None):
    # assumes all files have the same labels. ie: test/train pre split, or data from various years
    """
    take a dataset in a single file, splits X and Y, and splits the data into train and test compartments
    >>> fn1 = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> [X_train, Y_train, X_test, Y_test], target_name,n = prep_data(fn1, target, without)
    >>> list(X_train.columns)
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'file_name_test.csv', 'file_name_train.csv']


    >>> fn2 = "avocado_prices"
    >>> target = "AveragePrice"
    >>> without = ['Date', '4046', '4225', '4770']
    >>> [X_train, Y_train, X_test, Y_test], target_name,n = prep_data(fn2, target, without)
    >>> list(X_train.columns)
    ['Unnamed: 0', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'type_conventional', 'type_organic', 'region_Albany', 'region_Atlanta', 'region_BaltimoreWashington', 'region_Boise', 'region_Boston', 'region_BuffaloRochester', 'region_California', 'region_Charlotte', 'region_Chicago', 'region_CincinnatiDayton', 'region_Columbus', 'region_DallasFtWorth', 'region_Denver', 'region_Detroit', 'region_GrandRapids', 'region_GreatLakes', 'region_HarrisburgScranton', 'region_HartfordSpringfield', 'region_Houston', 'region_Indianapolis', 'region_Jacksonville', 'region_LasVegas', 'region_LosAngeles', 'region_Louisville', 'region_MiamiFtLauderdale', 'region_Midsouth', 'region_Nashville', 'region_NewOrleansMobile', 'region_NewYork', 'region_Northeast', 'region_NorthernNewEngland', 'region_Orlando', 'region_Philadelphia', 'region_PhoenixTucson', 'region_Pittsburgh', 'region_Plains', 'region_Portland', 'region_RaleighGreensboro', 'region_RichmondNorfolk', 'region_Roanoke', 'region_Sacramento', 'region_SanDiego', 'region_SanFrancisco', 'region_Seattle', 'region_SouthCarolina', 'region_SouthCentral', 'region_Southeast', 'region_Spokane', 'region_StLouis', 'region_Syracuse', 'region_Tampa', 'region_TotalUS', 'region_West', 'region_WestTexNewMexico']


    >>> fn3 = "random_regression"
    >>> [X_train, Y_train, X_test, Y_test], target_name,n  = prep_data(fn3)
    >>> list(X_train.columns)
    ['x', 'file_name_test.csv', 'file_name_train.csv']

    >>> dsname = "world_happiness_indicator"
    >>> target = "Happiness Rank"
    >>> data, target_name,n = [X_train, Y_train, X_test, Y_test], target_name, n = prep_data(dsname, target, test_size = 0.2, model_class = 'xgboost')
    >>> len(X_train)
    624
    >>> len(X_test)
    157

    >>> fn3 = "random_regression"
    >>> [X_train, Y_train, X_test, Y_test], target_name, n = prep_data(fn3)
    >>> list(X_train.columns)
    ['x', 'file_name_test.csv', 'file_name_train.csv']

     >>> fn1 = "titanic"
     >>> target = "Survived"
     >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
     >>> model_class = "rule_learner"
     >>> test_size = 0.2
     >>> [train, train_target, test,  test_target], target, n = data, target, n = prep_data(fn1, target, without, test_size, model_class)
     >>> train.shape
     (834, 8)
     >>> test.shape
     (209, 8)

    :param dataset_name:
    :param target:
    :param without:
    :param test_size:
    :param model_class:
    :return:
    """

    dataset_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets", dataset_name)

    first_file = True
    for root, dirs, files in os.walk(dataset_dir):
        for name in files:
            fn = os.path.join(root, name)
            fn_data = pd.read_csv(fn)
            no_files = len(next(os.walk(dataset_dir))[2])
            if first_file:
                df = pd.DataFrame(fn_data)
                for c in without:
                    df[c]
                df = df.drop(without, axis='columns')
                df = df.dropna()
                if no_files > 1:
                    df['file_name'] = name
                first_file = False
            else:
                df_supp = pd.DataFrame(fn_data)
                df_supp = df_supp.drop(without, axis='columns')
                df_supp = df_supp.dropna()
                if no_files > 1:
                    df_supp['file_name'] = name
                df = df.append(df_supp)

    if no_files > 1:
        df = df.drop('file_name', axis='columns')

    if target is None:
        back_one = -1 if no_files > 1 else 0
        Y = df.iloc[:, -1 + back_one]
        target = df.columns[len(df.columns) - 1 + back_one]
        X = df.drop(df.columns[len(df.columns) - 1 + back_one], axis=1)
    else:
        Y = df[target]
        X = df.drop([target], axis='columns')

    if not(pos_class is None):
        Y = pd.Series(where(Y == pos_class, 1, -1))

    if model_class in ['xgb.XGBRegressor', 'xgb.XGBClassifier', 'rulefit']: # seems that rulefit does not handle categorical data very well. getting dummies
        X = pd.get_dummies(X)

    regex = re.compile(r"[|]|<", re.IGNORECASE)

    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                 X.columns.values]

    train, test, train_target, test_target = train_test_split(X, Y, random_state=random_seed, test_size=test_size)

    n = (len(test), len(train))

    if norm_mean:  # scikitlearn transformer.
        target_train_mean = sum(train_target) / len(train_target)
        train_target -= target_train_mean
        test_target -= target_train_mean

        train_target = [train_target, target_train_mean]
        test_target = [test_target, target_train_mean]

    data = [train, train_target, test, test_target]

    return data, target, n, random_seed

def exp_data_prep(dataset_name, model_class = "orb", norm_mean = False, downsample_size = None,
                  random_seed = None, test_size = 0.2, pos_class=None):
    target_name, without = dataset_signature(dataset_name)

    data, target_name, n, random_seed = prep_data(dataset_name=dataset_name, target=target_name, without=without,
                                     test_size=test_size, model_class=model_class,
                                                  norm_mean=norm_mean, random_seed=random_seed, pos_class=pos_class)

    if norm_mean == False:
        [train, train_target, test, test_target] = data
    else:
        [train, [train_target, train_target_mean], test, [test_target, test_target_mean]] = data

    if downsample_size != None:
        train[target_name] = train_target
        sampled_train = train.sample(n=min(downsample_size, len(train_target)), random_state=random_seed)
        train.reset_index(inplace=True, drop=True) # this may be unncessesary
        train_target = sampled_train[target_name]
        train = sampled_train.drop([target_name], axis='columns')


    data = [train, train_target, test, test_target]

    return data, target_name, random_seed

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
                        'noisy_pairity_4', 'noisy_pairity_5', 'noisy_pairity_6']

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
                model.fit(train, train_target, verbose=True)
                model.max_rules += 1
#                print(model.members)
        else:
            start = time()
            model.fit(train, train_target)

        self.fit_time = time() - start

        for metric_name in self.metrics:
            if self.model_class_key == 'rulefit':
                rules = model.get_rules()
                useful_rules = rules[rules.coef != 0].sort_values("support", ascending=False)
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




def many_exp(dataset_name, reqs, seeds = [None]):
    res_df = pd.DataFrame()
    best = None

    for idx in range(len(seeds)):
        seed = seeds[idx]
        print('repeat ' + str(idx+1) + ' for ' + dataset_name)
        print(reqs)
        exp = Experiment(dataset_name=dataset_name, **reqs, seed=seed)
        res_dict, model = exp.run()

        for i in range(len(model.members)):
            ensemble_length_vs_perf_i = res_dict[reqs['metrics'][0] + '_length_vs_perf' + '_test'][i]

            if (best is None) or (ensemble_length_vs_perf_i > best[2]):
                best = (deepcopy(model), i, ensemble_length_vs_perf_i)

        res_df = res_df.append(res_dict, ignore_index=True)

    return res_df, best


def save_results(dataset_name, timestamp, setup, res_df, best):

    model_method = setup['model_class'] + "_" + setup['params']['method']

    res_dir = os.path.join(os.getcwd(), 'results', timestamp, dataset_name, model_method)

    try:
        os.makedirs(res_dir)
    except FileExistsError:
        pass

    with open(os.path.join(res_dir, 'exp_setup.txt'), 'w') as f:
        for key in setup.keys():
            f.write(key + ': ' + str(setup[key]) + '\n')

    res_df.to_csv(os.path.join(res_dir, 'exp_results.csv'), mode='w')

    best_model, no_rules, len_vs_perf = best
    with open(os.path.join(res_dir, 'best_model.txt'), 'w') as f:
        f.write('no_rules: ' + str(no_rules) + '\n')
        f.write('len_vs_perf: ' + str(len_vs_perf) + '\n')
        f.write('model: \n')
        for rule in best_model.members[:no_rules+1]:
            f.write(rule.__str__() + '\n')

    return res_dir

def average_and_plot(dataset_name, methods, metrics, res_df, res_dir, problem_type): # this will need updating when rulefit is added
    res_df = res_df.loc[res_df['dataset'] == dataset_name]

    methods_df = {method: pd.DataFrame() for method in methods}
    for idx, row in res_df.iterrows():
        method = row['model_params']['method']
        methods_df[method] = methods_df[method].append(row, ignore_index = True)

    fig = plt.figure(1)
    fig.clf()
    for method in methods:
        method_df = methods_df[method]
        len_vs_perf_dict = {}
        for metric in metrics:
            metric_data = method_df[metric+'_length_vs_perf']
            for idx, row in metric_data.iteritems():
                for no_rules, len_vs_perf in row.iteritems():
                    if no_rules in len_vs_perf_dict:
                        len_vs_perf_dict[no_rules].append(len_vs_perf)
                    else:
                        len_vs_perf_dict[no_rules] = [len_vs_perf]

        x = list(len_vs_perf_dict.keys())
        x = x[1:] if problem_type == 'r' else x
        y = [sum(len_vs_perf_dict[no_rules])/len(len_vs_perf_dict[no_rules]) for no_rules in x]
        print(method, metric)
        print(x)
        print(y)
        plt.plot(x, y, label='realkd_'+method+'_'+metric, marker='o')
    plt.legend()
    plt.title(dataset_name)
    plt.xlabel('no_rules')
    plt.ylabel('performance')
    fig.savefig(os.path.join(res_dir, dataset_name+' comparison_plot'))

#    plt.show()

def dataset_sweep(datasets, repeats, cut_type = 'qcut'):
    dataset_names = list(datasets.keys())
    methods = ['greedy', 'bestboundfirst']
    timestamp = get_timestamp()
    best_df = pd.DataFrame()


    for i in range(len(datasets)):
        best_Series = pd.Series()

        dataset_name = dataset_names[i]
        problem_type, downsample_size, max_rules, reg, max_col_attr = datasets[dataset_name]
        metric = ['r2'] if problem_type == 'r' else ['roc_auc'] # r2, r2_adj, rmse, r2_skl
        seeds = [generate_random_seed() for i in range(repeats)]
        all_results = pd.DataFrame()

        best_Series['dataset_name'] = dataset_name

        for method in methods:
            orb_setup = {"model_class": 'realkd',
                         "objective": 'logistic' if problem_type == 'c' else 'squared',
                         "metrics": metric,
                         "downsample_size": downsample_size,
                         "params": {'max_rules': max_rules, 'reg': reg,
                                    'max_col_attr': max_col_attr, 'method': method,
                                    'offset_rule': False, 'discretization': DISCRETISATION_MAP[cut_type]} # change discretisation here directly for now. parameterise it later
                         }

            res_df, best = many_exp(dataset_name, orb_setup, seeds=seeds)

            best_Series[method+'_no_rules'] = best[1]
            best_Series[method+'_'+metric[0]+'_'+'len_vs_perf'] = best[2]

            res_dir = save_results(dataset_name, timestamp, orb_setup, res_df, best)
            all_results = all_results.append(res_df, ignore_index=True)

        res_dir = os.path.join(os.getcwd(), 'results', timestamp, dataset_name)
        average_and_plot(dataset_name, methods, metric, all_results, res_dir, problem_type)

        best_df = best_df.append(best_Series, ignore_index=True)

    exp_res_dir = os.path.join(os.getcwd(), 'results', timestamp)

    return exp_res_dir, best_df


def rulefit_sweep(fixed_params, variable_lst, max_rules):
    test_res_vect = []
    train_res_vect = []
    best_rulefit = None
    best_rulefit_score = None

    fit_times = []
    max_fitted_rules = 0
    for params in variable_lst:
        print(params)
        fixed_params['params'] = params
        metric = fixed_params['metrics'][0]
        exp = Experiment(**fixed_params)
        res, model = exp.run()
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



    return (x,y), (x2,y2), average_fit_time, best_rulefit

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


            (rulefit_x_test, rulefit_y_test), (rulefit_x_train, rulefit_y_train), rulefit_time, best_rulefit =\
                rulefit_sweep(rulefit_fixed, rulefit_params_lst, max_rules)
            rulefit_auc_test = np.trapz(rulefit_y_test, rulefit_x_test) / (max(rulefit_x_test) - min(rulefit_x_test)) # + int(problem_type == 'c'))
            rulefit_auc_train = np.trapz(rulefit_y_train, rulefit_x_train) / (max(rulefit_x_train) - min(rulefit_x_train)) # + int(problem_type == 'c'))

            rulefit_aucs_test.append(rulefit_auc_test)
            rulefit_aucs_train.append(rulefit_auc_train)

            best_rulefit_rules = best_rulefit.get_rules()
            best_rulefit_rules = best_rulefit_rules[best_rulefit_rules.coef != 0].sort_values("support", ascending=False)
            for idx, rule in best_rulefit_rules.iterrows():
                print(('+' if rule['coef'] > 0 else '') + str(rule['coef']) + ' if ' + rule['rule'])
            for i in range(len(rulefit_x_test)):
                print(rulefit_x_test[i], rulefit_y_test[i])
            print(rulefit_aucs_test)


            greedy_exp = Experiment(**realkd_fixed, params=realkd_greedy_params)
            realkd_greedy_df, greedy_model = greedy_exp.run()

            len_vs_perf_greedy_test = greedy_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_test'][int(problem_type == 'r'):]
            len_vs_perf_greedy_train = greedy_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_train'][int(problem_type == 'r'):]
            print(len_vs_perf_greedy_test)

            realkd_greedy_auc_test = np.trapz(len_vs_perf_greedy_test.values, len_vs_perf_greedy_test.index)/max_rules
            realkd_greedy_auc_train = np.trapz(len_vs_perf_greedy_train.values, len_vs_perf_greedy_train.index)/max_rules

            realkd_greedy_aucs_test.append(realkd_greedy_auc_test)
            realkd_greedy_aucs_train.append(realkd_greedy_auc_train)
            print(realkd_greedy_auc_test)

            opt_exp = Experiment(**realkd_fixed, params=realkd_opt_params)
            realkd_opt_df, opt_model = opt_exp.run()
            len_vs_perf_opt_test = opt_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_test'][int(problem_type == 'r'):]
            len_vs_perf_opt_train = opt_exp.results[realkd_fixed['metrics'][0] + '_length_vs_perf'+'_train'][int(problem_type == 'r'):]

            while len_vs_perf_opt_test.index[-1] < max_rules:
                len_vs_perf_opt_test.loc[len_vs_perf_opt_test.index[-1]+1] = len_vs_perf_opt_test.values[-1]
                len_vs_perf_opt_train.loc[len_vs_perf_opt_train.index[-1]+1] = len_vs_perf_opt_train.values[-1]

            print(len_vs_perf_opt_test)
            realkd_opt_auc_test = np.trapz(len_vs_perf_opt_test.values, len_vs_perf_opt_test.index)/max_rules
            realkd_opt_auc_train = np.trapz(len_vs_perf_opt_train.values, len_vs_perf_opt_train.index)/max_rules

            realkd_optimal_aucs_test.append(realkd_opt_auc_test)
            realkd_optimal_aucs_train.append(realkd_opt_auc_train)
            print(realkd_opt_auc_test)

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
                        'greedy_auc_test': realkd_greedy_auc_test,
                        'greedy_auc_train': realkd_greedy_auc_train,
                        'greedy_reg': greedy_reg,
                        'greedy_time': greedy_exp.fit_time,
                        'opt_auc_test': realkd_opt_auc_test,
                        'opt_auc_train': realkd_opt_auc_train,
                        'opt_reg': opt_reg,
                        'opt_max_rules': opt_max_rules,
                        'opt_time': opt_exp.fit_time,
                        'random_seed': random_seed,
                        'model_dir': save_dir if save_models else None,
                        'cummulative_rulefit_auc_test': [rulefit_x_test, cummulative_trapz(rulefit_y_test, rulefit_x_test, problem_type)],
                        'cummulative_rulefit_auc_train': [rulefit_x_train, cummulative_trapz(rulefit_y_train, rulefit_x_train, problem_type)],
                        'cummulative_greedy_auc_test': [len_vs_perf_greedy_test.index,
                                                    cummulative_trapz(len_vs_perf_greedy_test.values,
                                                                      len_vs_perf_greedy_test.index, problem_type)],
                        'cummulative_greedy_auc_train': [len_vs_perf_greedy_train.index,
                                                    cummulative_trapz(len_vs_perf_greedy_train.values,
                                                                      len_vs_perf_greedy_train.index, problem_type)],
                        'cummulative_opt_auc_test': [len_vs_perf_opt_test.index,
                                                cummulative_trapz(len_vs_perf_opt_test.values,
                                                                  len_vs_perf_opt_test.index, problem_type)],
                        'cummulative_opt_auc_train': [len_vs_perf_opt_train.index,
                                                cummulative_trapz(len_vs_perf_opt_train.values,
                                                                  len_vs_perf_opt_train.index, problem_type)]
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

if __name__ == "__main__":

    wle_ = 4

    default_max_rules = 10
    default_greedy_reg = 10**-3
    default_opt_reg = 10
    default_rulefit_iter = 50
    default_reps = 5

    # problem_type, downsample_size, max_rules, realkd_reg, max_col_attr, rulefit_reg, test_size, repeats, pos_class, realkd_max_rules

    all_datasets = {'titanic': ('c', None, default_max_rules, (default_greedy_reg, 5),
                                {'Pclass': 4, 'Sex': 2, 'Age': 20, 'SibSp': 12, 'Parch': 16, 'Fare': 20, 'Embarked': 3},
                                (0.005, 0.025, default_rulefit_iter),
                                0.2, default_reps,
                                1,
                                None),
                    'world_happiness_indicator': ('r', None, default_max_rules, (default_greedy_reg, default_opt_reg), #(0, 30),
                                                  {'Region': 10, 'Economy(GDP_per_Capita)': 10, 'Family': 10,
                                                   'Health(Life_Expectancy)': 10, 'Freedom': 10,
                                                   'Trust(Government_Corruption)': 10, 'Generosity': 10,
                                                   'Dystopia_Residual': 10},
                                                  (2, 5, default_rulefit_iter),
                                                  0.2, default_reps,
                                                  None,
                                                  None),
                    'advertising': ('c', None, default_max_rules, (default_greedy_reg, 5), #(0, 1),
                                    {'Daily Time Spent on Site': 20, 'Age': 10, 'Area Income': 20,
                                     'Daily Internet Usage': 20, 'Male': 2, 'Country': 10},
                                    (0.005, 0.02, default_rulefit_iter),
                                    0.2, default_reps,
                                    1,
                                    None),
                    'used_cars': ('r', None, default_max_rules, (default_greedy_reg, default_opt_reg), # (0, 0),
                                  10,
                                  (0.00025, 0.0008, default_rulefit_iter),
                                  0.2, default_reps,
                                  None,
                                  None),
                    'boston': ('r', None, default_max_rules, (default_greedy_reg, 10), # (0, 100),
                               8,
                               (0.3, 0.6, default_rulefit_iter),
                               0.2, default_reps,
                               None,
                               None),
                    'halloween_candy_ranking': ('r', None, 12, (default_greedy_reg, 5), # (0, 0),
                                                {'chocolate': 2, 'fruity': 2, 'caramel': 2, 'peanutyalmondy': 2,
                                                 'nougat': 2, 'crispedricewafer': 2, 'hard': 2, 'bar': 2, 'pluribus': 2,
                                                 'sugarpercent': 20, 'pricepercent': 20},
                                                (0.1, 1, default_rulefit_iter),
                                                0.02, default_reps,
                                                None,
                                                None),
                    'tic-tac-toe': ('c', None, default_max_rules, (default_greedy_reg, 5),  # (0, 0),
                                    2,
                                    (0.005, 0.075, default_rulefit_iter),
                                    0.2, default_reps,
                                    'positive',
                                    None),
                    'breast_cancer': ('c', None, default_max_rules, (default_greedy_reg, 5), # (0, 25),
                                      6,
                                      (0.005, 0.025, default_rulefit_iter),
                                      0.2, default_reps,
                                      1,
                                      None),
                    'iris': ('c', None, default_max_rules, (default_greedy_reg, 5),  # (0, 0),
                             20,
                             (0.04, 0.075, default_rulefit_iter),
                             0.2, default_reps,
                             1,
                             None),
                    'red_wine_quality': ('r', None, default_max_rules, (default_greedy_reg, 5),  # (0, 10),
                                         6, # 6
                                         (5, 12.5, default_rulefit_iter),
                                         0.2, default_reps,
                                         None,
                                         None),
                    'who_life_expectancy': ('r', None, default_max_rules, (default_greedy_reg, 5), # (0, 0),
                                            {'Country':	2*wle_-2, 'Year': wle_, 'Status': 2, 'Adult Mortality': wle_,
                                             'infant deaths': wle_, 'Alcohol': wle_, 'percentage expenditure': wle_,
                                             'Hepatitis B': wle_, 'Measles': wle_, 'BMI': wle_,
                                             'under-five deaths': wle_, 'Polio': wle_, 'Total expenditure': wle_,
                                             'Diphtheria': wle_, 'HIV/AIDS': wle_, 'GDP': wle_, 'Population': wle_,
                                             'thinness 1-19 years': wle_, 'thinness 5-9 years': wle_,
                                             'Income composition of resources': wle_, 'Schooling': wle_},
                                            (0.3, 0.6, default_rulefit_iter),
                                            0.2, default_reps,
                                            None,
                                            None),
                    'Demographics': ('r', None, default_max_rules, (default_greedy_reg, 5), # (0, 0),
                                             {'2    SEX': 2, '3    MARITAL STATUS': 5, '4    AGE': 12,
                                              '5    EDUCATION': 10, '6    OCCUPATION': 9,
                                              '7    HOW LONG HAVE YOU LIVED IN THE SAN FRAN./OAKLAND/SAN JOSE AREA?': 5,
                                              '8    DUAL INCOMES (IF MARRIED)': 3, '9    PERSONS IN YOUR HOUSEHOLD': 16,
                                              '10    PERSONS IN HOUSEHOLD UNDER 18': 16, '11    HOUSEHOLDER STATUS': 3,
                                              '12    TYPE OF HOME': 5, '13    ETHNIC CLASSIFICATION': 8,
                                              '14    WHAT LANGUAGE IS SPOKEN MOST OFTEN IN YOUR HOME?': 3},
                                             (1, 3.5, default_rulefit_iter),
                                             0.2, default_reps,
                                             None,
                                             None),
                    'digits': ('c', None, default_max_rules, (default_greedy_reg, 5),
                               4,
                               (0.0005, 0.003, default_rulefit_iter),
                               0.2, default_reps,
                               3,
                               None),
                    'make_friedman1': ('r', None, default_max_rules, (default_greedy_reg, default_opt_reg),
                                       6,
                                       (0.6, 1.4, default_rulefit_iter),
                                       0.2, 4,
                                       None,
                                       None),
                    'make_friedman2': ('r', None, default_max_rules, (default_greedy_reg, default_greedy_reg),
                                       10,
                                       (0.006, 0.015, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None),
                    'make_friedman3': ('r', None, default_max_rules, (default_greedy_reg, default_greedy_reg),
                                       10,
                                       (10.5, 16, default_rulefit_iter),
                                       0.2, default_reps,
                                       None,
                                       None),
                    'make_classification2': ('c', None, default_max_rules, (default_greedy_reg, 5),
                                             8,
                                             (0.001, 0.01, default_rulefit_iter),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'make_classification3': ('c', None, default_max_rules, (default_greedy_reg, default_opt_reg),
                                             6,
                                             (0.001, 0.01, 3),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'load_wine': ('c', None, default_max_rules, (default_greedy_reg, 2),
                                             6,
                                             (0.022, 0.075, default_rulefit_iter),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'make_hastie_10_2': ('c', None, default_max_rules, (default_greedy_reg, default_opt_reg),
                                             10,
                                             (1, 10, 3),
                                             0.2, default_reps,
                                             1,
                                             None),
                    'kr - vs - kp': ('c', None, default_max_rules, (default_greedy_reg, 10*default_opt_reg),
                                     2,
                                     (0.002, 0.006, default_rulefit_iter),
                                     0.2, default_reps,
                                     'Won',
                                     None),
                    'noisy_pairity_': ['c', None, default_max_rules, (10, 10),
                                      8,
                                      (0.018, 0.024, default_rulefit_iter),
                                      0.2, default_reps,
                                      1,
                                      None],
                    'noisy_pairity_5': ['c', None, default_max_rules, (10, 10),
                                       8,
                                       (0.018, 0.024, default_rulefit_iter),
                                       0.2, default_reps,
                                       1,
                                       None],
                    'noisy_pairity_6': ['c', None, default_max_rules, (10, 10),
                                       8,
                                       (0.015, 0.035, default_rulefit_iter),
                                       0.2, default_reps,
                                       1,
                                       None]
                    }

    dataset_names = list(all_datasets.keys())
    showplot = False
    save_models = True
    random_seed = None
    new_header = True
    x_err = 0
    y_err = 0.01

    dataset_names = ['load_wine', 'make_friedman1', 'make_friedman2', 'make_friedman3',
            'used_cars', 'tic-tac-toe', 'titanic',
            'iris', 'make_classification2', 'who_life_expectancy', 'world_happiness_indicator', 'advertising',
            'breast_cancer', 'red_wine_quality', 'boston', 'digits', 'Demographics', 'kr - vs - kp']

    global_friedman_cols = 15
    full_comparision(dataset_names, all_datasets, random_seed=random_seed,
                     xerr=x_err, yerr=y_err, save_models=save_models,
                     new_header=new_header, showplot=showplot)
    '''
    for global_friedman_cols in range(5, 16):
        returned_models = full_comparision(dataset_names, all_datasets,
                                           random_seed=random_seed,
                                           xerr=x_err, yerr=y_err, save_models=save_models,
                                           new_header=new_header, showplot=showplot)

    
    results_df = pd.read_csv('table_of_final_results_tex.csv')
    diagonal = np.linspace(0, 1, 3)
    greedy = results_df['greedy_auc']
    optimal = results_df['opt_auc']
    plt.plot(diagonal, diagonal, linestyle='--')
    plt.scatter(optimal, greedy)
    plt.xlabel('optimal auc')
    plt.ylabel('greedy auc')
    plt.savefig('optimal vs greedy auc.pdf')
    plt.show()

    df = pd.read_csv('friedman1_res.csv', delimiter=',')
    ds = range(5, 16)
    data_extraction_metric = mean
    rf_test = []
    rf_train = []
    grd_test = []
    grd_train = []
    opt_test = []
    opt_train = []
    print(df['no_feat'].iloc[0])
    for d in ds:
        d_frame = df.loc[df['no_feat'] == d]
        rf_test.append(data_extraction_metric(d_frame['rulefit_aucs_test']))
        rf_train.append(data_extraction_metric(d_frame['rulefit_aucs_train']))
        grd_test.append(data_extraction_metric(d_frame['greedy_auc_test']))
        grd_train.append(data_extraction_metric(d_frame['greedy_auc_train']))
        opt_test.append(data_extraction_metric(d_frame['opt_auc_test']))
        opt_train.append(data_extraction_metric(d_frame['opt_auc_train']))

    plt.plot(ds, rf_test, marker='s', label=f'rulefit (regression): test')
    plt.plot(ds, rf_train, marker='s', label=f'rulefit (regression): train', linestyle='--' )
    plt.plot(ds, grd_test, marker='s', label=f'greedy (squared), rule boosting: test')
    plt.plot(ds, grd_train, marker='s', label=f'greedy (squared), rule boosting: train', linestyle='--')
    plt.plot(ds, opt_test, marker='s', label=f'optimal (squared), rule boosting: test')
    plt.plot(ds, opt_train, marker='s', label=f'optimal (squared), rule boosting: train', linestyle='--')

    plt.legend()
    plt.xlabel('# dimensions')
    plt.ylabel('AUC from R2')
    plt.savefig('perf.pdf')
    plt.show()
    
    '''

    #ToDO
    # https://web.stanford.edu/~hastie/ElemStatLearn/datasets/marketing.info.txt
    # https://web.stanford.edu/~hastie/ElemStatLearn/datasets/marketing.data


    # noisy pairity, change setup to work within ensemble size requirements
    # friedman variable cols
