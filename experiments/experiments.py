import os

import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from data_processing.data_preprocessing import prep_data
from data_processing.data_preprocessing import prep_data_rule_learner
from data_processing.data_preprocessing import prep_data_rule_learner_better
from data_processing.data_preprocessing import prep
from orb.rules import AdditiveRuleEnsemble

from xgb_scripts.xgb_functions import generate_xgb_model
from xgb_scripts.xgb_functions import count_nodes

def rules_rmse(rules, test, test_target, n_test):
    rules_prediction = [rules(test.iloc[i]) for i in range(n_test)]
    rmse = sum([(test_target.iloc[i] - rules_prediction[i])**2 for i in range(n_test)])**0.5
    return rmse

def rules_exp(dataset_name, target = None, without = [], k=4, reg=50):
    """
    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> rules, rules_accuracy = rules_exp(dataset_name, target, without)
    >>> rules_accuracy>0.7
    True

    >>> dataset_name = "metacritic_games"
    >>> target = "user_score"
    >>> without = ["GameID",	"name",	"players",	"attribute",	"release_date",	"link"]
    >>> rules, rules_accuracy = rules_exp(dataset_name, target, without)
    >>> rules_accuracy > 0.7
    True

    """

    train, test, train_target, test_target, n_test, n_train = prep_data_rule_learner_better(dataset_name, target, without)

    rules = AdditiveRuleEnsemble(k, reg)
    rules.fit(train, train_target)

    rmse = rules_rmse(rules, test, test_target, n_test)
    return rules, rmse


def exp(fn, target=None, without=[], model_class="xgboost", clear_results=False, k_vals = [3], d_vals = [3], ld_vals = [10]):
    """
    >>> fn = "halloween_candy_ranking"
    >>> target = "winpercent"
    >>> without = ['competitorname']
    >>> exp(fn, target, without)

    :param fn:
    :param target:
    :param without:
    :param path:
    :param clear_results:
    :return:
    """
    if clear_results:
        res_df = pd.DataFrame(columns=['dataset', 'target', 'no_feat', 'no_rows', 'tr_RMSE', 'te_RMSE', 'model_class', 'model_complexity', 'k', 'd', 'l'])
    else:
        res_df = pd.read_pickle("results/results.pkl")  # open old results

    [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn, target, without)
    data = [X_train, Y_train, X_test, Y_test]

    for k in k_vals:
        for d in d_vals:
            for ld in ld_vals:
                # alternative, don't do with if block, feed model class as function input rather than model class's name
                if model_class == "xgboost":
                    model, train_RMSE, test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)
                elif model_class == "rule_ensamble":
                    # swap for a call to rule ensamble searcher
                    model, train_RMSE, test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)
                else:
                    # swap for a call to some other learner
                    model, train_RMSE, test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)

                exp_res = {"dataset": fn, "target": target_name, "no_feat": len(X_train.columns),
                           "no_rows": Y_train.shape[0] + Y_test.shape[0], "tr_RMSE": train_RMSE, "te_RMSE": test_RMSE,
                           'model_class': model_class, "model_complexity": model_complexity, 'k': k, 'd': d, 'l': ld}
                res_df = res_df.append(exp_res, ignore_index=True)

    res_df.to_pickle("./results.pkl", protocol=4) #save current results table

    testing = False
    if testing == True:
        read_results()

    return exp_res


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


def exp_on_all_datasets(datasets, k_vals = [3], d_vals = [3], ld_vals = [10]):
    first = True
    for fn in datasets:
        print(fn)
        target, without = dataset_signature(fn)
        exp(fn, target, without, path="", model_class="xgboost", clear_results=first, k_vals=k_vals, d_vals=d_vals, ld_vals=ld_vals)
        first = False

def exp_on_all_datasets_rules(datasets, k_vals = [3], d_vals = [3], ld_vals = [10]):
    first = True
    for fn in datasets:
        print(fn)
        target, without = dataset_signature(fn)
        xgb_res = exp(fn, target, without, path="", model_class="xgboost", clear_results=first, k_vals=k_vals, d_vals=d_vals, ld_vals=ld_vals)
        rules_res = rules_exp(fn, target, without)
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

