import os
import re

import pandas as pd

from sklearn.model_selection import train_test_split


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

    if model_class in ['xgb.XGBRegressor', 'xgb.XGBClassifier', 'rulefit']: # seems that rulefit does not handle categorical data very well. getting dummies
        X = pd.get_dummies(X)

    regex = re.compile(r"[|]|<", re.IGNORECASE)

    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                 X.columns.values]

    train, test, train_target, test_target = train_test_split(X, Y, random_state=random_seed, test_size=test_size)

    if model_class == 'realkd': #realkd update coming to remove this need

        train.reset_index(inplace=True, drop=True)
        test.reset_index(inplace=True, drop=True)
        train_target.reset_index(inplace=True, drop=True)
        test_target.reset_index(inplace=True, drop=True)

    n = (len(test), len(train))

    if norm_mean:  # scikitlearn transformer
        target_train_mean = sum(train_target) / len(train_target)
        train_target -= target_train_mean
        test_target -= target_train_mean

        train_target = [train_target, target_train_mean]
        test_target = [test_target, target_train_mean]

    data = [train, train_target, test, test_target]

    return data, target, n, random_seed

def exp_data_prep(dataset_name, model_class = "orb", norm_mean = False, downsample_size = None, random_seed = None):
    target_name, without = dataset_signature(dataset_name)

    data, target_name, n, random_seed = prep_data(dataset_name=dataset_name, target=target_name, without=without,
                                     test_size=0.2, model_class=model_class, norm_mean=norm_mean, random_seed=random_seed)

    if norm_mean == False:
        [train, train_target, test, test_target] = data
    else:
        [train, [train_target, train_target_mean], test, [test_target, test_target_mean]] = data

    if downsample_size != None:
        train[target_name] = train_target
        sampled_train = train.sample(n=min(downsample_size, len(train_target)), random_state=random_seed)
        train_target = sampled_train[target_name]
        train = sampled_train.drop([target_name], axis='columns')

    data = [train, train_target, test, test_target]

    return data, target_name, random_seed
