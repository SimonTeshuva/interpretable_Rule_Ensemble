import os

import pandas as pd
import matplotlib.pyplot as plt
import sys

from datetime import datetime

from data_processing.data_preprocessing import prep_data as prep
from orb.rules import AdditiveRuleEnsemble

from xgb_scripts.xgb_functions import generate_xgb_model
from xgb_scripts.xgb_functions import count_nodes

def rules_rmse(rules, test, test_target, n_test):
    rules_prediction = [rules(test.iloc[i]) for i in range(n_test)]
    rmse = sum([(test_target.iloc[i] - rules_prediction[i])**2 for i in range(n_test)])**0.5
    return rmse


def read_results():
    res_df = pd.read_pickle("results/results.pkl") #open old results
    res_df.to_csv('results.csv')
    pd.set_option('display.max_columns', 10)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res_df)
    res_df.to_pickle("./results.pkl") #save current results table


def dataset_signature(dataset_name):
#    data = {dataset_name: (target, without)}
    data = {"advertising": ("Clicked on Ad", ["Ad Topic Line", "City", "Timestamp"], []),
            "avocado_prices": ("AveragePrice", ["Date", "4046",	"4225",	"4770"], []),
            "cdc_physical_activity_obesity": ("Data_Value", ["Data_Value_Alt", "LocationAbbr", "Data_Value_Footnote_Symbol", "Data_Value_Footnote", "GeoLocation",
                                                               "ClassID", "TopicID", "QuestionID", "DataValueTypeID", "LocationID", "StratificationCategory1",
                                                               "Stratification1", "StratificationCategoryId1", "StratificationID1"], []),
            "gdp_vs_satisfaction": ("Satisfaction", ["Country"], []),
            "halloween_candy_ranking": ("winpercent", ['competitorname'], ['very bad', 'bad', 'average', 'good','very good']),
            "metacritic_games": ("user_score", ["GameID",	"name",	"players",	"attribute",	"release_date",	"link"], []),
            "random_regression": (None, [], []), # "" not appropriate as null input, potentiall swap to None
            "red_wine_quality": ("quality", [], []),
            "suicide_rates": ("suicides/100k pop", ["suicides_no","population","HDI for year"], []),
            "titanic": ("Survived", ['PassengerId', 'Name', 'Ticket', 'Cabin'], []),
            "us_minimum_wage_by_state": ("CPI.Average", ["Table_Data", "Footnote"], []), # may be wrong target
            "used_cars": ("avgPrice", [], []),
            "wages_demographics": ("earn", [], []),
            "who_life_expectancy": ("Life expectancy ", [], []),
            "world_happiness_indicator": ("Happiness Score", ["Country", "Happiness Rank", "Standard Error"], []),
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
    pos = len(max_vals)-1
    if len(max_vals)!=len(current_val):
        return ValueError
    if current_val==max_vals:
        return current_val
    while not incremented:
        if current_val[pos] < max_vals[pos]:
            current_val[pos]+=1
            incremented=True
        elif current_val[pos] == max_vals[pos]:
            current_val[pos]=0
            pos-=1
        else:
            return ValueError
    return current_val

def lex_succ(max_vals):
    """
    >>> lex_succ([1,1,2])
    [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2]]
    """
    current_val = [0]*len(max_vals)
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
    max_vals = [len(model_parameters[i])-1 for i in range(len(model_parameters))]
    parameter_combinations =  lex_succ(max_vals)
    sweep_options = [[model_parameters[i][parameter_combination[i]] for i in range(len(parameter_combination))] for parameter_combination in parameter_combinations]
    return sweep_options

def unified_experiment(dataset, target=None, without = [], test_size = 0.2, clear_results = False, model_class = "xgboost", model_parameters = [[3], [3], [10]]):
    """
    >>> print("titanic test")
    titanic test
    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 3, 10]
    >>> d_vals = [1, 3, 5]
    >>> ld_vals = [1, 10, 100]
    >>> model_parameters = [k_vals, d_vals, ld_vals]
    >>> model_parameters = [[3], [3], [10]] # done for speed, usually remove
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class="xgboost", model_parameters=model_parameters)


    >>> print("titanic test")
    titanic test
    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 3, 10]
    >>> d_vals = [1, 3, 5]
    >>> ld_vals = [1, 10, 100]
    >>> model_parameters = [k_vals, d_vals, ld_vals]
    >>> model_parameters = [[3], [3], [10]] # done for speed, usually remove
    >>> model_class = "xgboost"
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class=model_class, model_parameters=model_parameters)

    >>> print("titanic test")
    titanic test
    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 4]
    >>> reg_vals = [10, 50]
    >>> model_parameters = [k_vals, reg_vals]
    >>> model_parameters = [[4], [50]] # done for speed, usually remove
    >>> model_class = "rule_ensamble"
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class=model_class, model_parameters=model_parameters)

    >>> print("titanic test")
    titanic test
    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 3]
    >>> reg_vals = [10, 50]
    >>> model_parameters = [k_vals, reg_vals]
    >>> model_class = "other"
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class=model_class, model_parameters=model_parameters)

    """
    if clear_results:
        res_df = pd.DataFrame(
            columns=['dataset', 'target', 'no_feat', 'no_rows', 'tr_RMSE', 'te_RMSE', 'model_class', 'model_complexity',
                     'k', 'd', 'l']) # fix
    else:
        res_df = pd.read_pickle("results/results.pkl")  # open old results

    data, target_name, n = prep(dataset, target, without)
    [train, train_target, test, test_target] = data
    (n_test, n_train) = n

    sweep_options = parameter_sweep(model_parameters)

    for sweep_option in sweep_options:
        if model_class == "xgboost":
            model, train_RMSE, test_RMSE = generate_xgb_model(data, *sweep_option, model_type="Regressor")
            rmse = test_RMSE
            model_complexity = count_nodes(model)
        elif model_class == "rule_ensamble":
            ensamble_complexity = lambda reg, k: k  # a placeholder function for model complexity

            rules = AdditiveRuleEnsemble(*sweep_option)
            rules.fit(train, train_target)
            rmse = rules_rmse(rules, test, test_target, n_test)
            model_complexity = ensamble_complexity(*sweep_option)
        else:
            return ValueError

        exp_res = {"dataset": dataset, "target": target, "no_feat": len(train.columns),
                       "no_rows": n_train+n_test, "rmse": rmse, 'model_class': model_class,
                        "model_complexity": model_complexity, "model_parameters": sweep_option}

        res_df = res_df.append(exp_res, ignore_index=True)

        res_df.to_pickle("./results.pkl", protocol=4)  # save current results table

        testing = False
        if testing == True:
            read_results()

        return exp_res


def exp_on_all_datasets(datasets, test_size=0.2, model_class="xgboost", model_parameters=[[3], [3], [10]]):
    first = True
    for fn in datasets:
        print(fn)
        target, without = dataset_signature(fn)
        unified_experiment(fn, target, without, test_size=test_size, clear_results=first, model_class=model_class, model_parameters=model_parameters)
        first = False


"""
def get_timestamp():
    now = datetime.now()
    timestamp = datetime.timestamp(now)
    dt_object = str(datetime.fromtimestamp(timestamp))

    dt_object = dt_object.replace(" ", "_")
    dt_object= dt_object.replace(":", "_")
    dt_object = dt_object.replace("-", "_")
    return dt_object


def generate_interpretability_curve(error, interpretability, x_name, y_name, title_name, dataset_name,
                                    experiment_timestamp, param_name):
    result_directory_curves = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp,
                                           param_name, "plots")
    try:
        os.makedirs(result_directory_curves)
    except FileExistsError:
        pass
    tree_dir = os.path.join(result_directory_curves, "interpretability_curve.pdf")

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

"""
if __name__ == "__main__":
    # split target into ranges.
    datasets = ["advertising", "avocado_prices", "cdc_physical_activity_obesity", "gdp_vs_satisfaction",
                 "halloween_candy_ranking","metacritic_games", "random_regression", "red_wine_quality",
                 "suicide_rates", "titanic", "us_minimum_wage_by_state", "used_cars","wages_demographics",
                 "who_life_expectancy", "world_happiness_indicator"]

    small_datasets = ["advertising", "halloween_candy_ranking", "random_regression", "red_wine_quality",
                      "titanic", "used_cars", "wages_demographics"] # small

    # "us_minimum_wage_by_state" causing problems. suspect its the columns with ... and (b) entries in them

    # "gdp_vs_satisfaction": raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    # xgboost.core.XGBoostError: [16:34:12] D:\Build\xgboost\xgboost-0.90.git\src\metric\rank_metric.cc:200: Check failed: !auc_error: AUC: the dataset only contains pos or neg samples

    debugging_datasets = ["advertising", "titanic"]
    k_vals = [1, 3, 10]
    d_vals = [1, 3, 5]
    ld_vals = [1, 10, 100]
#    exp_on_all_datasets(small_datasets)
    exp_on_all_datasets(small_datasets, k_vals, d_vals, ld_vals)
    read_results()

