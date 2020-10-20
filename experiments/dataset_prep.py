import os
import re

import numpy as np
import pandas as pd

from numpy import where
from sklearn import datasets as ds
from sklearn.model_selection import train_test_split

from realkd.datasets import noisy_parity

from dataset_info import *

def prep_noisy_pairity(d=5, random_seed=None):

    n = 100 * 2 ** d

    x_train, y_train = noisy_parity(n, d, random_seed=random_seed)
    x_test, y_test = noisy_parity(2000, d, random_seed=random_seed+1)

    target = 'y'

    data = [x_train, y_train, x_test, y_test]

    return data, target, random_seed

def prep_digits(pos_class = [5], neg_class = [2,3,6,8,9,0], random_seed = None):
    def make_binary(data, target, pos_classes, neg_classes):
        pos_filter = target.isin(pos_classes)
        neg_filter = target.isin(neg_classes)
        y = target.copy()
        y[pos_filter] = 1
        y[neg_filter] = -1
        y = y[pos_filter | neg_filter]
        return data[pos_filter | neg_filter], y

    target_name = 'target'

    digits_df = pd.read_csv(os.path.join(os.path.dirname(os.getcwd()), 'datasets', 'digits', 'digits.csv'))

    data, target = make_binary(digits_df.drop(columns=target_name), digits_df[target_name], pos_class, neg_class)

    return data, target


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
        global_friedman_cols =10
        data, target = ds.make_friedman1(n_samples=2000, n_features=10, noise=0.1, random_state=random_seed) # 1
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
    elif dataset_name == 'load_diabetes':
        bunch = ds.load_diabetes()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        data_rf = pd.get_dummies(df)
        target = pd.Series(bunch.target)
    elif dataset_name[:-1] == 'noisy_pairity_':
        d = int(dataset_name[-1])
        data, target_name, random_seed = prep_noisy_pairity(d=d, random_seed=random_seed)
        return data, target_name, random_seed
    elif dataset_name == 'digits5':
        data_rf, target = prep_digits()

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

    if model_class in ['xgb.XGBRegressor', 'xgb.XGBClassifier', 'rulefit']:
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