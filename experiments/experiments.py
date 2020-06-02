import re
import os

import pandas as pd

from sklearn.model_selection import train_test_split

from orb.rules import AdditiveRuleEnsemble, save_exp_res, load_result
import datetime

from xgb_scripts.xgb_functions import generate_xgb_model
from xgb_scripts.xgb_functions import count_nodes
import xgboost as xgb

import graphviz

from matplotlib import pyplot as plt

def prep_data(dataset_name, target = None, without = [], test_size = 0.2, model_class = 'xgboost', mean_norm = False):
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

    n = (len(test), len(train))

    if mean_norm: #scikitlearn transformer
        target_train_mean = sum(train_target)/len(train_target)
#        target_test_mean = sum(test_target)/len(test_target)

        train_target -= target_train_mean
        test_target -= target_train_mean

        train_target = [train_target, target_train_mean]
        test_target = [test_target, target_train_mean]

    data = [train, train_target, test, test_target]

    return data, target, n

def read_results():
    res_df = pd.read_pickle("results/results.pkl") #open old results
    res_df.to_csv('results.csv')
    pd.set_option('display.max_columns', 10)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(res_df)
    res_df.to_pickle("results/results.pkl") #save current results table


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
            "world_happiness_indicator": ("Happiness Score", ["Country", "Happiness Rank"], []),
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


def exp_on_all_datasets(datasets, test_size=0.2, model_info = ["xgboost", [[3], [3]], 10]):
    """
    >>> datasets = ["titanic", "gdp_vs_satisfaction"]
    >>> k_vals = [1, 3, 10]
    >>> d_vals = [1, 3, 5]
    >>> ld_vals = [1, 10, 100]
    >>> xgb_model_parameters = [k_vals, d_vals, ld_vals]
    >>> k_vals = [1, 4]
    >>> reg_vals = [10, 50]
    >>> rule_ensamble_model_parameters = [k_vals, reg_vals]
    >>> model_info = [["xgboost", xgb_model_parameters], ["rule_ensamble", rule_ensamble_model_parameters]]
    >>> exp_on_all_datasets(datasets, test_size=0.2, model_info=model_info)
    """
    first = True
    for fn in datasets:
        for model in model_info:
            model_class = model[0]
            model_parameters = model[1]
            print(fn, model_class, model_parameters)
            target, without = dataset_signature(fn)
            unified_experiment(fn, target, without, test_size=test_size, clear_results=first, model_class=model_class, model_parameters=model_parameters)
            first = False


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


def unified_experiment(dataset, target=None, without = [], test_size = 0.2, clear_results = False, model_class = "xgboost", model_parameters = [[3], [3], [10]], downsample_size = None, norm_mean = False):
    """
    >>> print("titanic test")
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 3, 10]
    >>> d_vals = [1, 3, 5]
    >>> ld_vals = [1, 10, 100]
    >>> model_parameters = [k_vals, d_vals, ld_vals]
    >>> model_parameters = [[3], [3], [10]] # done for speed, usually remove
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class="xgboost", model_parameters=model_parameters)

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

    >>> dataset_name = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> k_vals = [1, 4]
    >>> reg_vals = [10, 50]
    >>> model_parameters = [k_vals, reg_vals]
    >>> model_parameters = [[4], [50]] # done for speed, usually remove
    >>> model_class = "rule_ensamble"
    >>> exp_res = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class=model_class, model_parameters=model_parameters)

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
        res_df = pd.DataFrame(columns=["dataset", "target", "no_feat", "no_rows", "train_rmse", "test_rmse", 'model_class', "model_complexity", "model_parameters"])
    else:
        res_df = pd.read_pickle("results/results.pkl")  # open old results

    data, target_name, n = prep_data(dataset, target, without, mean_norm=norm_mean)
    if norm_mean:
        [train, [train_target, train_target_mean], test, [test_target, test_target_mean]] = data
    else:
        [train, train_target, test, test_target] = data
    (n_test, n_train) = n

    if downsample_size != None: # for speed. usually remove
        train=train[:downsample_size]
        test=test[:downsample_size]
        train_target=train_target[:downsample_size]
        test_target=test_target[:downsample_size]
        n_train=downsample_size
        n_test=downsample_size


    sweep_options = parameter_sweep(model_parameters)

    max_complexity_model = None
    max_complexity = 0
    num_trees = 0
    for sweep_option in sweep_options:
        # print(dataset_name, target, model_class, sweep_option)

        if model_class == "xgboost":
            model, train_RMSE, test_RMSE = generate_xgb_model(data, *sweep_option, model_type="Regressor", norm_mean=norm_mean)
            model_complexity = count_nodes(model)
            if model_complexity>max_complexity:
                max_complexity_model = model
                max_complexity = model_complexity
                num_trees = sweep_option[0]
        elif model_class == "rule_ensamble":
            ensamble_complexity = lambda reg, k: k  # a placeholder function for model complexity
            rules = AdditiveRuleEnsemble(*sweep_option)
            rules.fit(train, train_target)
            # rmse = rules_rmse(rules, test, test_target, n_test)
            model_complexity = ensamble_complexity(*sweep_option)
        else:
            return ValueError

        exp_res = {"dataset": dataset, "target": target, "no_feat": len(train.columns),
                       "no_rows": n_train+n_test, "train_rmse": train_RMSE, "test_rmse": test_RMSE,
                        'model_class': model_class, "model_complexity": model_complexity, "model_parameters": sweep_option}
        print(exp_res)

        res_df = res_df.append(exp_res, ignore_index=True)

        res_df.to_pickle("results/results.pkl", protocol=4)  # save current results table

    testing = False
    if testing == True:
        read_results()

    return res_df, max_complexity_model, num_trees


def exp(dataset_name = "avocado_prices", target_name = "AveragePrice", feature_names_to_drop = ["Date", "group"], model_class = 'rule_ensamble', model_parameters = (3, 50), alpha_function = lambda k: 1):
    data, target_name, n = prep_data(dataset_name=dataset_name, target=target_name, without=feature_names_to_drop, test_size=0.2, model_class=model_class)

    [train, train_target, test, test_target] = data
    (n_test, n_train) = n

    downsample_size = 500
    train = train[:downsample_size]
    train_target = train_target[:downsample_size]
    test = test[:downsample_size]
    test_target = test_target[:downsample_size]
    n_train = downsample_size
    n_test = downsample_size



    if model_class == "rule_ensamble":
        print('creating rule ensamble')
        rules = AdditiveRuleEnsemble(k=model_parameters[0], reg=model_parameters[1], alpha_function=alpha_function)
        rules.reset()
        print('ftting rule ensamble')

        train_scores = []
        test_scores = []
        model_complexities = []
        for i in range(rules.k):
            print('adding rule: ', len(rules.members)+1)
            rules.add_rule(data=train, labels=train_target)
            model_complexities.append(rules.model_complexity())
            train_scores.append(rules.score(data=train, labels = train_target, n=downsample_size))
            test_scores.append(rules.score(data=test, labels = test_target, n=downsample_size))

        print('printing rule ensamble')
        for rule in rules.members:
            print(rule)

    elif model_class == "xgboost":
        print(1)

    data = [train, train_target, test, test_target]

    dataset_size = downsample_size
    results = [dataset_name, target_name, rules, train_scores, test_scores, model_complexities, feature_names_to_drop, dataset_size]
    return results, data # change to yeild for each rule, add alpha parameter

def unified_experiment2(dataset_name="avocado_prices", target_name="AveragePrice", feature_names_to_drop=["Date", "group"], ensamble_parameters=[[2], [50], [lambda k: 1]], forest_parameters = [[1,2,3], [1,2,3], [50]], clear_results=True, downsample_size=None):
    rule_ensamble = "rule_ensamble"
    random_forest = "xgboost"

    if clear_results:
        res_df = pd.DataFrame(columns=["dataset", "target", "no_feat", "no_rows", "train_rmse", "test_rmse", 'model_class', "model_complexity", "model_parameters"])
    else:
        res_df = pd.read_pickle("results/results.pkl")  # open old results

    data, target_name, n = prep_data(dataset_name, target=target_name, without=feature_names_to_drop, test_size=0.2, model_class="rule_ensamble")

    [train, train_target, test, test_target] = data
    (n_test, n_train) = n

    if downsample_size != None:
        train = train[:downsample_size]
        train_target = train_target[:downsample_size]
        test = test[:downsample_size]
        test_target = test_target[:downsample_size]
        n_train = downsample_size
        n_test = downsample_size

    ensamble_options = parameter_sweep(ensamble_parameters)
    forest_options = parameter_sweep(forest_parameters)

    all_options = []
    for option in ensamble_options:
        option.append(rule_ensamble)
        all_options.append(option)
    for option in forest_options:
        option.append(random_forest)
        all_options.append(option)

    for option in all_options:
        if option[-1]==rule_ensamble:
            k, reg, alpha_function, model_class = option
            rules = AdditiveRuleEnsemble(k=k, reg=reg, alpha_function=alpha_function)
            rules.reset()
            print('ftting rule ensamble')
            for i in range(rules.k):
                print('adding rule: ', len(rules.members) + 1)
                rules.add_rule(data=train, labels=train_target)
                model_complexity, train_RMSE, test_RMSE = rules.model_complexity(), rules.score(data=train, labels=train_target, n=downsample_size), rules.score(data=test, labels=test_target, n=downsample_size)
                exp_res = {"dataset": dataset_name, "target": target_name, "no_feat": len(train.columns),
                           "no_rows": n_train + n_test, "train_rmse": train_RMSE, "test_rmse": test_RMSE,
                           'model_class': model_class, "model_complexity": model_complexity, "model_parameters": option}
                print(exp_res)
                res_df = res_df.append(exp_res, ignore_index=True)
        elif option[-1]==random_forest:
            k, depth, reg, model_class = option
            model, train_RMSE, test_RMSE = generate_xgb_model(data, *option[:-1], model_type="Regressor", norm_mean=False)
            model_complexity = count_nodes(model)
            exp_res = {"dataset": dataset_name, "target": target_name, "no_feat": len(train.columns),
                       "no_rows": n_train + n_test, "train_rmse": train_RMSE, "test_rmse": test_RMSE,
                       'model_class': model_class, "model_complexity": model_complexity, "model_parameters": option}
            print(exp_res)
            res_df = res_df.append(exp_res, ignore_index=True)
        else:
            continue



#        res_df.to_pickle("results/results.pkl", protocol=4)  # save current results table

    testing = False
    if testing == True:
        read_results()

if __name__ == "__main__":

    # unified_experiment2()

    model_class = "ensamble"
    repeats = 3

    dataset_name = "avocado_prices"
    target = "AveragePrice"
    without = ["Date", "group"]
    '''
    dataset_name ="halloween_candy_ranking"
    target = "winpercent"
    without = ['competitorname']

    dataset_name = "red_wine_quality"
    target = "quality"
    without = []
    '''

    # ensemble vs forest (1, 3, ...) max depth/#rules

    # opt est doesnt eem to update
    # scikitlearn with 1 tree as a comparision as well
    # look into documentation to see if data is being normalised/centralisation about 0 for all numerical features

    if model_class == "ensamble":
        max_k = 2
        reg = 50
        alpha_function = lambda k: max(min(1 - k ** 2 / reg, 1), 0)  # when alpha <0.5 code continues with alpha = 0.5 instead of stopping. investigate. Rules made with "Unamed" add to drop # change this to fixed time budget, or just a constant
        alpha_function = lambda k: 0.8
        train_score_av = [0]*max_k
        test_score_av = [0]*max_k
        model_complexities_av = [0]*max_k
        for i in range(repeats):
            res, data = exp(dataset_name=dataset_name, target_name=target, feature_names_to_drop=without, model_parameters=(max_k, reg), alpha_function=alpha_function)
            dataset_name, target_name, rules, train_scores, test_scores, model_complexities, feature_names_to_drop, dataset_size = res
            [train, train_target, test, test_target] = data
            train_score_av = [train_score_av[j] + train_scores[j]/repeats for j in range(len(train_score_av))]
            test_score_av = [test_score_av[j] + test_scores[j]/repeats for j in range(len(test_score_av))]
            model_complexities_av = [model_complexities_av[j] + model_complexities[j]/repeats for j in range(len(model_complexities_av))]

        save_exp_res(res)
#        ARE, train_scores_old, test_scores_old, model_complexities_old = load_result("avocado_prices", "2020-05-27 15:01:22.111692")
        plt.plot(model_complexities_av, train_score_av, label="train")
        plt.plot(model_complexities_av, test_score_av, label="test")
        plt.xlabel('ensamble complexity')
        plt.ylabel('error')
        plt.legend()
        plt.title("avocado prices")
        plt.show()
    elif model_class=="xgboost":
        k_vals = [1, 2, 3]
        d_vals = [1, 2, 3]
        max_depth = 5
        max_comp = 60

        k_vals = range(1,6)
        sweep_options = [(k, [i for i in range(1, max_depth) if k*2**i < max_comp]) for k in k_vals]

        ld_vals = [10]
        model_parameters = [k_vals, d_vals, ld_vals]
        norm_mean = True
        for option in sweep_options:
            k, d, ld = option[0], option[1], ld_vals
            model_parameters = [[k], d, ld]
            for i in range(repeats):
                exp_res, max_model, num_trees = unified_experiment(dataset_name, target=target, without=without, test_size=0.2, model_class=model_class, model_parameters=model_parameters, downsample_size=500, clear_results=True, norm_mean = norm_mean)

                if i == 0:
                    train_RMSEs = exp_res['train_rmse']
                    test_RMSEs = exp_res['test_rmse']
                    model_complexities  = exp_res["model_complexity"]
                else:
                    train_RMSEs = [train_RMSEs[i] + exp_res['train_rmse'][i] for i in range(len(train_RMSEs))]
                    test_RMSEs = [test_RMSEs[i] + exp_res['test_rmse'][i] for i in range(len(test_RMSEs))]
                    model_complexities = [model_complexities[i] + exp_res['model_complexity'][i] for i in range(len(model_complexities))]

            train_RMSEs = [train_RMSEs[i]/repeats for i in range(len(train_RMSEs))]
            test_RMSEs = [test_RMSEs[i]/repeats for i in range(len(test_RMSEs))]
            model_complexities = [model_complexities[i]/repeats for i in range(len(model_complexities))]

            train_data = [(model_complexities[i], train_RMSEs[i]) for i in range(len(model_complexities))]
            train_data.sort(key=lambda x: x[1])
            train_data.sort(key = lambda x: x[0])
            train_data_cleaned = [train_data[i] for i in range(len(train_data)) if i==0 or (train_data[i][0] != train_data[i-1][0])]

            test_data = [(model_complexities[i], test_RMSEs[i]) for i in range(len(model_complexities))]
            test_data.sort(key = lambda x: x[1])
            test_data.sort(key = lambda x: x[0])
            test_data_cleaned = [test_data[i] for i in range(len(test_data)) if i==0 or (test_data[i][0] != test_data[i-1][0])]

            print(train_data_cleaned)
            print(test_data_cleaned)
            plt.plot(*zip(*train_data_cleaned), label="train")
            plt.plot(*zip(*test_data_cleaned), label="test")
            plt.xlabel('forest complexity')
            plt.ylabel('error')
            plt.legend()
            plt.title("avocado prices")
            plt.show()


        for i in range(num_trees):
            tree = xgb.plot_tree(max_model, num_trees=i)
            plt.show()

    ARE, train_scores, test_scores, model_complexities = load_result("avocado_prices", "2020-05-30 22:25:07.727598")
    for rule in ARE.members:
        print(rule)

    '''

    ARE, train_scores, test_scores, model_complexities = load_result("avocado_prices", "2020-05-26 21:51:41.971961) # I am not reconstructing the default rule correctly

    print('before extra rule')
    for rule in ARE.members:
        print(rule)

    ARE.add_rule(data=train, labels=train_target)

    print('after extra rule')
    for rule in ARE.members:
        print(rule)
    '''

    '''
    #    alpha_function = lambda k: 1
    alphas = []
    for k in range(1, max_k):
        res = exp(model_parameters=(k, reg), alpha_function=alpha_function)
        dataset_name = res[0]
        target_name = res[1]
        rules.append(res.members[2][-1])
        alphas.append(res.alphas[3][-1])
        rmse_train.append(res[4])
        rmse_test.append(res[5])

    from collections import deque

    ensamble_complexity = deque([0])
    for rule in rules:
        rule_complexity = len(str(rule.__repr__()).split('&'))
        ensamble_complexity.append(ensamble_complexity[-1] + rule_complexity)
    ensamble_complexity.popleft()
    ensamble_complexity = [complexity - 1 for complexity in ensamble_complexity]
    plt.plot(ensamble_complexity, rmse_train)
    plt.xlabel('ensamble_size')
    plt.ylabel('rmse')
    plt.title('Ensamble Complexity vs RMSE for Avocado Prices')
    print(dataset_name, target_name)
    ar = ([(rules[i], alphas[i]) for i in range(len(rules))])
    for rule in ar:
        print(rule[1], rule[0])
    plt.show()
    
    '''
