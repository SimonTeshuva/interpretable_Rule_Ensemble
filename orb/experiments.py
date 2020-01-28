import os
import pandas as pd

from orb.data_preprocessing import prep_data
from orb.xgb_functions import generate_xgb_model
from orb.xgb_functions import count_nodes

def exp(fn, target=None, without=[], target_labels = [], path="", model_class = "xgboost", clear_results = False): # add model type in results
    """
    >>> fn = "halloween_candy_ranking"
    >>> target = "winpercent"
    >>> without = ['competitorname']
    >>> target_labels = ['very bad', 'bad', 'average', 'good','very good']
    >>> exp(fn, target, without)

    :param fn:
    :param target:
    :param without:
    :param path:
    :param clear_results:
    :return:
    """
    if clear_results:
        res_df = pd.DataFrame(columns=['dataset', 'target', 'no_feat', 'no_rows', 'tr_RMSE', 'te_RMSE', 'model_complexity'])
    else:
        res_df = pd.read_pickle("./results.pkl")  # open old results

    [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn, target, without, target_labels)
    data = [X_train, Y_train, X_test, Y_test]

    for k in [3,10]: #[3,10]
        for d in [1,3,5]: #[1,3,5]
            for ld in [1,10,100]: # [1,10,100]
                # alternative, don't do with if block, feed model class as function input rather than model class's name
                if model_class == "xgboost":
                    model, train_RMSE,test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)
                elif model_class == "rule_ensamble":
                    # swap for a call to rule ensamble searcher
                    model, train_RMSE,test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)
                else:
                    # swap for a call to some other learner
                    model, train_RMSE, test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)

                exp_res = {"dataset": fn, "target": target_name, "no_feat": len(X_train.columns),
                           "no_rows": Y_train.shape[0] + Y_test.shape[0], "tr_RMSE": train_RMSE, "te_RMSE": test_RMSE,
                           "model_complexity": model_complexity}
                res_df = res_df.append(exp_res, ignore_index=True)

    res_df.to_pickle("./results.pkl") #save current results table

    testing = True
    if testing == True:
        read_results()

def read_results():
    res_df = pd.read_pickle("./results.pkl") #open old results
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
            "random_regression": ("", [], []), # "" not appropriate as null input, potentiall swap to None
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
    target_labels = dataset_info[2]
    return target, without, target_labels

def exp_on_all_datasets(datasets):
    first = True
    for fn in datasets:
        target, without, target_labels = dataset_signature(fn)
        exp(fn, target, without, target_labels, path="", model_class="xgboost", clear_results=first)
        first = False


if __name__ == "__main__":
    # split target into ranges.
    datasets =  ["advertising", "avocado_prices", "cdc_physical_activity_obesity", "gdp_vs_satisfaction",
                 "halloween_candy_ranking","metacritic_games", "random_regression", "red_wine_quality",
                 "suicide_rates", "titanic", "us_minimum_wage_by_state", "used_cars","wages_demographics",
                 "who_life_expectancy", "world_happiness_indicator"]

    small_datasets = ["advertising", "halloween_candy_ranking", "random_regression", "red_wine_quality",
                "titanic", "us_minimum_wage_by_state", "used_cars", "wages_demographics"] # small

    # "gdp_vs_satisfaction": raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    # xgboost.core.XGBoostError: [16:34:12] D:\Build\xgboost\xgboost-0.90.git\src\metric\rank_metric.cc:200: Check failed: !auc_error: AUC: the dataset only contains pos or neg samples

    debugging_datasets = ["advertising", "titanic"]
    exp_on_all_datasets(debugging_datasets)
