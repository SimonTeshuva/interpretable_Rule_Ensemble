import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

def prep_data_single_file(dataset_name, target = None, without = [], test_size = 0.2, model_class = 'xgboost'):
    """
    take a dataset in a single file, splits X and Y, and splits the data into train and test compartments
    >>> fn1 = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> [X_train, Y_train, X_test, Y_test] = prep_data(fn1, target, without)
    >>> list(X_train.columns)
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']


    >>> fn2 = "avocado_prices"
    >>> target = "AveragePrice"
    >>> without = ['Date', '4046', '4225', '4770']
    >>> [X_train, Y_train, X_test, Y_test] = prep_data(fn2, target, without)
    >>> list(X_train.columns)
    ['Unnamed: 0', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'type_conventional', 'type_organic', 'region_Albany', 'region_Atlanta', 'region_BaltimoreWashington', 'region_Boise', 'region_Boston', 'region_BuffaloRochester', 'region_California', 'region_Charlotte', 'region_Chicago', 'region_CincinnatiDayton', 'region_Columbus', 'region_DallasFtWorth', 'region_Denver', 'region_Detroit', 'region_GrandRapids', 'region_GreatLakes', 'region_HarrisburgScranton', 'region_HartfordSpringfield', 'region_Houston', 'region_Indianapolis', 'region_Jacksonville', 'region_LasVegas', 'region_LosAngeles', 'region_Louisville', 'region_MiamiFtLauderdale', 'region_Midsouth', 'region_Nashville', 'region_NewOrleansMobile', 'region_NewYork', 'region_Northeast', 'region_NorthernNewEngland', 'region_Orlando', 'region_Philadelphia', 'region_PhoenixTucson', 'region_Pittsburgh', 'region_Plains', 'region_Portland', 'region_RaleighGreensboro', 'region_RichmondNorfolk', 'region_Roanoke', 'region_Sacramento', 'region_SanDiego', 'region_SanFrancisco', 'region_Seattle', 'region_SouthCarolina', 'region_SouthCentral', 'region_Southeast', 'region_Spokane', 'region_StLouis', 'region_Syracuse', 'region_Tampa', 'region_TotalUS', 'region_West', 'region_WestTexNewMexico']


    >>> fn3 = "random_regression"
    >>> [X_train, Y_train, X_test, Y_test] = prep_data(fn3)
    >>> list(X_train.columns)
    ['x']

    :param dataset_name:
    :param target:
    :param without:
    :param test_size:
    :param model_class:
    :return:
    """


    dataset_dir = os.path.join(os.path.dirname(os.getcwd()), "datasets", dataset_name)

    dataset = pd.read_csv(os.path.join(dataset_dir, dataset_name + ".csv"))

    df = pd.DataFrame(dataset)
    df = df.drop(without, axis='columns')
    df = df.dropna()

    if target == None:
        Y = df.iloc[:, -1]
        X = df.drop(df.columns[len(df.columns)-1], axis=1)
    else:
        Y = df[target]
        X = df.drop([target], axis='columns')

    if model_class == 'xgboost':
        X = pd.get_dummies(X)

    regex = re.compile(r"[|]|<", re.IGNORECASE)

    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                 X.columns.values]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    data = [X_train, Y_train, X_test, Y_test]

    # need to move target to final column
    return data

def prep_data(dataset_name, target = None, without = [], target_q_cuts = [], test_size = 0.2, model_class = 'xgboost'):
    # assumes all files have the same labels. ie: test/train pre split, or data from various years
    """
    take a dataset in a single file, splits X and Y, and splits the data into train and test compartments
    >>> fn1 = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn1, target, without)
    >>> list(X_train.columns)
    ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female', 'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']


    >>> fn2 = "avocado_prices"
    >>> target = "AveragePrice"
    >>> without = ['Date', '4046', '4225', '4770']
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn2, target, without)
    >>> list(X_train.columns)
    ['Unnamed: 0', 'Total Volume', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'type_conventional', 'type_organic', 'region_Albany', 'region_Atlanta', 'region_BaltimoreWashington', 'region_Boise', 'region_Boston', 'region_BuffaloRochester', 'region_California', 'region_Charlotte', 'region_Chicago', 'region_CincinnatiDayton', 'region_Columbus', 'region_DallasFtWorth', 'region_Denver', 'region_Detroit', 'region_GrandRapids', 'region_GreatLakes', 'region_HarrisburgScranton', 'region_HartfordSpringfield', 'region_Houston', 'region_Indianapolis', 'region_Jacksonville', 'region_LasVegas', 'region_LosAngeles', 'region_Louisville', 'region_MiamiFtLauderdale', 'region_Midsouth', 'region_Nashville', 'region_NewOrleansMobile', 'region_NewYork', 'region_Northeast', 'region_NorthernNewEngland', 'region_Orlando', 'region_Philadelphia', 'region_PhoenixTucson', 'region_Pittsburgh', 'region_Plains', 'region_Portland', 'region_RaleighGreensboro', 'region_RichmondNorfolk', 'region_Roanoke', 'region_Sacramento', 'region_SanDiego', 'region_SanFrancisco', 'region_Seattle', 'region_SouthCarolina', 'region_SouthCentral', 'region_Southeast', 'region_Spokane', 'region_StLouis', 'region_Syracuse', 'region_Tampa', 'region_TotalUS', 'region_West', 'region_WestTexNewMexico']


    >>> fn3 = "random_regression"
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn3)
    >>> list(X_train.columns)
    ['x', 'file_name_test.csv', 'file_name_train.csv']

    >>> dsname = "world_happiness_indicator"
    >>> target = "Happiness Rank"
    >>> data, target_name = [X_train, Y_train, X_test, Y_test], target_name = prep_data(dsname, target, test_size = 0.2, model_class = 'xgboost')
    >>> len(X_train)
    624
    >>> len(X_test)
    157

    >>> fn3 = "random_regression"
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn3)
    >>> list(X_train.columns)
    ['x', 'file_name_test.csv', 'file_name_train.csv']

    >>> dsname = "halloween_candy_ranking"
    >>> target = "winpercent"
    >>> without = ['competitorname']
    >>> target_q_cuts  = ['very bad', 'bad', 'average', 'good','very good']
    >>> data, target_name = [X_train, Y_train, X_test, Y_test], target_name = prep_data(dsname, target, without, target_q_cuts , test_size = 0.2, model_class = 'xgboost')
    >>> len(X_train)
    68
    >>> len(X_test)
    17
    >>> list(X_train.columns)
    ['chocolate', 'fruity', 'caramel', 'peanutyalmondy', 'nougat', 'crispedricewafer', 'hard', 'bar', 'pluribus', 'sugarpercent', 'pricepercent']



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

    if target_q_cuts :
        Y = pd.qcut(Y, len(target_q_cuts), target_q_cuts)

    if model_class == 'xgboost':
        X = pd.get_dummies(X)

    regex = re.compile(r"[|]|<", re.IGNORECASE)

    X.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in
                 X.columns.values]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    data = [X_train, Y_train, X_test, Y_test], target

    return data


