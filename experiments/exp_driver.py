import re
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from orb.rules import AdditiveRuleEnsemble, ExponentialObjective, SquaredLossObjective, Rule
from orb.search import Constraint, KeyValueProposition, Conjunction, Context
from ast import literal_eval

import datetime
import numpy as np

from xgb_scripts.xgb_functions import generate_xgb_model
from xgb_scripts.xgb_functions import count_nodes
from xgb_scripts.xgb_functions import save_xgb_model
from xgb_scripts.xgb_functions import save_xgb_trees
from xgb_scripts.xgb_functions import prediction_cost, explanation_cost, count_trees

import xgboost as xgb

import graphviz

from math import log2, ceil

from matplotlib import pyplot as plt
from copy import deepcopy
import sys
import random


from sklearn.datasets import load_boston


def prep_data(dataset_name, target=None, without=[], test_size=0.2, model_class='xgboost', norm_mean=False, random_seed = None):
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

    if target is None:
        back_one = -1 if no_files > 1 else 0
        Y = df.iloc[:, -1 + back_one]
        target = df.columns[len(df.columns) - 1 + back_one]
        X = df.drop(df.columns[len(df.columns) - 1 + back_one], axis=1)
    else:
        Y = df[target]
        X = df.drop([target], axis='columns')

    if model_class == 'xgboost':
        X = pd.get_dummies(X)

    regex = re.compile(r"[|]|<", re.IGNORECASE)

    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                 X.columns.values]

#    if random_seed == None:
#       train, test, train_target, test_target = train_test_split(X, Y, test_size=test_size)
#    else:
#        train, test, train_target, test_target = train_test_split(X, Y, random_state=random_seed, test_size=test_size)
    train, test, train_target, test_target = train_test_split(X, Y, random_state=random_seed, test_size=test_size)


    n = (len(test), len(train))

    if norm_mean:  # scikitlearn transformer
        target_train_mean = sum(train_target) / len(train_target)
        #        target_test_mean = sum(test_target)/len(test_target)

        train_target -= target_train_mean
        test_target -= target_train_mean

        train_target = [train_target, target_train_mean]
        test_target = [test_target, target_train_mean]

    data = [train, train_target, test, test_target]

    return data, target, n, random_seed


def read_results():
    res_df = pd.read_pickle("results/results.pkl")  # open old results
    res_df.to_csv('results.csv')
    pd.set_option('display.max_columns', 10)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res_df)
    res_df.to_pickle("results/results.pkl")  # save current results table


def dataset_signature(dataset_name):
    #    data = {dataset_name: (target, without)}
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
            "suicide_rates": ("suicides/100k pop", ["suicides_no", "population", "HDI for year"], []),
            "titanic": ("Survived", ['PassengerId', 'Name', 'Ticket', 'Cabin'], []),
            "us_minimum_wage_by_state": ("CPI.Average", ["Table_Data", "Footnote"], []),  # may be wrong target
            "used_cars": ("avgPrice", [], []),
            "wages_demographics": ("earn", [], []),
            "who_life_expectancy": ("Life expectancy ", [], []),
            "world_happiness_indicator": ("Happiness_Score", ["Country", "Happiness_Rank"], []),
            "boston": ("MEDV", [], []),
            }

    dataset_info = data[dataset_name]
    target = dataset_info[0]
    without = dataset_info[1]
    return target, without


def next_lex(max_vals, current_val):
    """
    >>> next_lex([1,2,3], [1,2])
    <class 'ValueError'>
    >>> next_lex([1,2,3], [1,2,2])
    [1, 2, 3]
    >>> next_lex([1,2,3], [1,2,3])
    [1, 2, 3]
    >>> next_lex([1,2,3], [1,1,3])
    [1, 2, 0]
    >>> next_lex([1,2,3], [0,2,3])
    [1, 0, 0]
    """
    incremented = False
    pos = len(max_vals) - 1
    if len(max_vals) != len(current_val):
        return ValueError
    if current_val == max_vals:
        return current_val
    while not incremented:
        if current_val[pos] < max_vals[pos]:
            current_val[pos] += 1
            incremented = True
        elif current_val[pos] == max_vals[pos]:
            current_val[pos] = 0
            pos -= 1
        else:
            return ValueError
    return current_val


def lex_succ(max_vals):
    """
    >>> lex_succ([1,1,2])
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2]]
    """
    current_val = [0] * len(max_vals)
    lex_succs = [current_val[:]]
    while current_val != max_vals:
        current_val = next_lex(max_vals, current_val)
        lex_succs.append(current_val[:])
    return lex_succs


def parameter_sweep(model_parameters):
    """
    >>> parameter_sweep([[1,2,3], ["a", "b", "c"]])
    [[1, 'a'], [1, 'b'], [1, 'c'], [2, 'a'], [2, 'b'], [2, 'c'], [3, 'a'], [3, 'b'], [3, 'c']]

    >>> k_vals = [1, 3, 10]
    >>> d_vals = [1, 3, 5]
    >>> ld_vals = [1, 10, 100]
    >>> model_parameters = [k_vals, d_vals, ld_vals]
    >>> parameter_sweep(model_parameters)
    [[1, 1, 1], [1, 1, 10], [1, 1, 100], [1, 3, 1], [1, 3, 10], [1, 3, 100], [1, 5, 1], [1, 5, 10], [1, 5, 100], [3, 1, 1], [3, 1, 10], [3, 1, 100], [3, 3, 1], [3, 3, 10], [3, 3, 100], [3, 5, 1], [3, 5, 10], [3, 5, 100], [10, 1, 1], [10, 1, 10], [10, 1, 100], [10, 3, 1], [10, 3, 10], [10, 3, 100], [10, 5, 1], [10, 5, 10], [10, 5, 100]]
    """
    max_vals = [len(model_parameters[i]) - 1 for i in range(len(model_parameters))]
    parameter_combinations = lex_succ(max_vals)
    sweep_options = [[model_parameters[i][parameter_combination[i]] for i in range(len(parameter_combination))] for
                     parameter_combination in parameter_combinations]
    return sweep_options


def get_timestamp():
    now = datetime.datetime.now()
    timestamp = datetime.datetime.timestamp(now)
    dt_object = str(datetime.datetime.fromtimestamp(timestamp))

    dt_object = dt_object.replace(" ", "_")
    dt_object = dt_object.replace(":", "_")
    dt_object = dt_object.replace("-", "_")
    return dt_object


def generate_interpretability_curve(results, x_name, y_name, title_name, dataset_name, experiment_timestamp, plotname):
    result_directory_curves = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp)
    try:
        os.makedirs(result_directory_curves)
    except FileExistsError:
        pass
    tree_dir = os.path.join(result_directory_curves, plotname+".pdf")

    plt.figure(0)

    for result in results:
        plt.plot(result[0], result[1], label = result[2])
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title_name)
    plt.legend()

    plt.savefig(tree_dir, dpi=300, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)

    print("\n\n\n\n curve plotted and saved \n\n\n\n\n")

    plt.show()


def exp_data_prep(dataset_name, model_class = "ensemble", norm_mean = False, downsample_size = None, random_seed = None):
    target_name, without = dataset_signature(dataset_name)

    data, target_name, n, random_seed = prep_data(dataset_name=dataset_name, target=target_name, without=without,
                                     test_size=0.2, model_class=model_class, norm_mean=norm_mean, random_seed=random_seed)

    if norm_mean == False:
        [train, train_target, test, test_target] = data
    else:
        [train, [train_target, train_target_mean], test, [test_target, test_target_mean]] = data

    (n_test, n_train) = n

    if downsample_size != None:

        train = train[:min(downsample_size, len(train))]
        train_target = train_target[:min(downsample_size, len(train_target))]
        test = test[:min(downsample_size, len(test))]
        test_target = test_target[:min(downsample_size, len(test_target))]
        n_train = len(train)
        n_test = len(test)

    data = [train, train_target, test, test_target]

    return data, target_name, random_seed

def save_exp_res(exp_res, timestamp, name="ensemble_results"):
    import datetime

    dataset_name, target_name, ensamble, train_scores, test_scores, model_complexities, feature_names_to_drop, n = exp_res

    rules = ensamble.members
    alpha_function = ensamble.alpha_function  # cant save alpha function at present in an easily useable way. for now assume using same alpha as always
    alphas = ensamble.alphas
    # still need to save, number of datapoints, plot

    try:
        save_dir = os.path.join(os.getcwd(), "results", dataset_name, timestamp)
        print(save_dir)
        os.makedirs(save_dir)
    except FileExistsError:
        print('already made:', save_dir)
    except:
        print('other error')

    with open(os.path.join(save_dir, name + '.txt'), 'w') as f:
        f.write('dataset_name_:_' + dataset_name + '\n')
        f.write('target_name_:_' + target_name + '\n')
        f.write('dropped_attributes_:_' + str(feature_names_to_drop) + '\n')
        #        f.write('alpha function: ' + str(alpha_function) + '\n')
        f.write('max_col_attr_:_' + str(ensamble.max_col_attr) + '\n')

        f.write('k_:_' + str(ensamble.k) + '\n')
        f.write('reg_:_' + str(ensamble.reg) + '\n')
        f.write('min_alpha_:_' + str(ensamble.min_alpha) + '\n')

        f.write('n_:_' + str(n) + '\n')
        f.write('train_scores_:_' + str(train_scores) + '\n')
        f.write('test_scores_:_' + str(test_scores) + '\n')
        f.write('model_complexities_:_' + str(model_complexities) + '\n')

        #        for i in range(len(rules)):
        #            f.write('rule ' + str(i) + ":" + str(rules[i].__repr__()) + '\n')
        f.write('rules_:_' + str([[rule.y, rule.q, rule.z] for rule in rules]) + '\n')

        #        for i in range(len(alphas)):
        #            f.write('alpha ' + str(i) + ":" + str(alphas[i]) + '\n')
        f.write('alphas_:_' + str(alphas) + '\n')
        result_df = pd.DataFrame()


def cast(val):
    if '.' in val:
        try:
            val = float(val)
        except ValueError:
            val = val
    else:
        try:
            val = int(val)
        except ValueError:
            val = val
    return val


def constraint_maker(string):
    v = 1
    k = 0

    ops = [("<=", Constraint.less_equals), (">=", Constraint.greater_equals), ("==", Constraint.equals),
           ("<", Constraint.less_than), (">", Constraint.greater_than)]
    for op, const in ops:
        if op in string:
            s = string.split(op)
            return const(s[v]), s[k]

def string_to_conjunction(ps):
    if not len(ps):
        return Conjunction(Context([], []).attributes)
    kvps = ps.split(' & ')
    propositions = []
    for kvp in kvps:
        const, key = constraint_maker(kvp)
        proposition = KeyValueProposition(key=key, constraint=const)
        propositions.append(proposition)
    q = Conjunction(propositions)
    return q

def load_result(dataset_name, timestamp):
    result_dir = os.path.join(os.getcwd(), "results", dataset_name, timestamp, "ensamble_data.txt")
    result_dict = dict()
    with open(result_dir, 'r') as f:
        for line in f:
            attribute = line.strip().split('_:_')
            result_dict[attribute[0]] = attribute[1]

    reg = float(result_dict['reg'])
    k = int(result_dict['k'])
    max_col_attr = int(result_dict["max_col_attr"])
    alpha_function = lambda k: max(min(1 - k ** 2 / reg, 1), 0)
    min_alpha = float(result_dict["min_alpha"])

    alphas = literal_eval(result_dict["alphas"])
    #    alphas = [float(alpha) for alpha in result_dict["alphas"][1:-1].split(',')]

    rules_string = result_dict['rules']
    rules_string_list = [item.strip() for item in rules_string[1:-1].split(',')]

    members = []
    for i in range(0, len(rules_string_list), 3):
        (y, ps, z) = rules_string_list[i][1:], rules_string_list[i + 1], rules_string_list[i + 2][:-2]
        q = string_to_conjunction(ps)
        rule = Rule(y=float(y), q=q, z=float(z), reg=reg, max_col_attr=max_col_attr, alpha=alphas[int(i // 3)])
        members.append(rule)

    train_scores = literal_eval(result_dict["train_scores"])
    test_scores = literal_eval(result_dict["test_scores"])
    model_complexities = literal_eval(result_dict["model_complexities"])

    #    train_scores = [float(train_score) for train_score in result_dict["train_scores"][1:-1].split(',')]
    #    test_scores = [float(test_score) for test_score in result_dict["test_scores"][1:-1].split(',')]
    #    model_complexities = [int(model_complexity) for model_complexity in result_dict["model_complexities"][1:-1].split(',')]

    ARE = AdditiveRuleEnsemble(members=members, reg=reg, k=k, max_col_attr=max_col_attr, alpha_function=alpha_function,
                               min_alpha=min_alpha, alphas=alphas)

    return ARE, train_scores, test_scores, model_complexities

def generate_random_seed(max_seed=2**32 - 1):
    return random.randrange(max_seed)


def generate_fit_model(data, target, model_class="ensemble", objective_function=SquaredLossObjective, model_params=[5, 50], alpha=0.8):

    [train, train_target, test, test_target] = data

    # table with experiment options instead

    if model_class=="ensemble":
        [k, reg] = model_params
        alpha_function = lambda k: alpha
        model = AdditiveRuleEnsemble(k=k, reg=reg, alpha_function=alpha_function, objective=objective_function)
        model.fit(data=train, labels=train_target)
    elif model_class=="forest":
        [k, d, reg] = model_params
        model_type = "Regressor" if type(objective_function)==type(SquaredLossObjective) else "Classifier"
        model, train_rmse, test_rmse = generate_xgb_model(data=data, k=k, d=d, reg=reg, model_type=model_type)
    elif model_class == "tree":
        return
    elif model_class=="rule_fit":
        return
    else:
        return

    return model

def exp(dataset_name, model_class = "ensemble", objective_function=SquaredLossObjective, model_params = [5, 50], norm_mean = False, downsample_size=None, random_seed=None, alpha=0.8):
    '''

    '''
    target, without = dataset_signature(dataset_name)
    data, target_name, random_seed = exp_data_prep(dataset_name, model_class=model_class,  norm_mean=norm_mean, downsample_size=downsample_size, random_seed=random_seed)
    [train, train_target, test, test_target] = data

    model = generate_fit_model(data, model_class=model_class, objective_function=objective_function, model_params=model_params)

    return

if __name__ == "__main__":
    res_dir = os.path.join(os.getcwd(), 'results.csv')

    repeats = 1
    max_complexity = 150
    downsample_size = None
    norm_mean = False
    max_k = 5
    reg = 50
    alpha = 0.8
    alpha_function = lambda k: alpha
    model_class = "ARE"
    objective_function = SquaredLossObjective
    max_d = 5


    exp()