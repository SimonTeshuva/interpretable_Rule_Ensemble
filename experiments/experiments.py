import re
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from orb.rules import AdditiveRuleEnsemble

from xgb_scripts.xgb_functions import generate_xgb_model
from xgb_scripts.xgb_functions import count_nodes

def prep_data(dataset_name, target = None, without = [], test_size = 0.2, model_class = 'xgboost'):
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
                df = df.drop(without, axis='columns')
                df = df.dropna()
                if no_files >1:
                    df['file_name'] = name
                first_file = False
            else:
                df_supp = pd.DataFrame(fn_data)
                df_supp = df_supp.drop(without, axis='columns')
                df_supp = df_supp.dropna()
                if no_files > 1:
                    df_supp['file_name'] = name
                df = df.append(df_supp)

    if target == None:
        back_one = -1 if no_files>1 else 0
        Y = df.iloc[:, -1+back_one]
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

    train, test, train_target, test_target = train_test_split(X, Y, test_size=test_size)

    data = [train, train_target, test, test_target]

    n = (len(test), len(train))

    return data, target, n

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

    data, target_name, n = prep_data(dataset, target, without)
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

