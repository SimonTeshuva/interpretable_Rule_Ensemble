import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from orb.data_preprocessing import prep_data

def generate_xgb_model(k, d, ld, data, model_type="Regressor"):
    """
    >>> fn1 = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(fn1, target, without)
    >>> data = [X_train, Y_train, X_test, Y_test]
    >>> model, train_rmse, test_rmse = generate_xgb_model(3, 3, 3, data, model_type="Regressor")
    >>> train_rmse < 0.5 and train_rmse >0.4
    True

    >>> dsname = "halloween_candy_ranking"
    >>> target = "winpercent"
    >>> without = ['competitorname']
    >>> [X_train, Y_train, X_test, Y_test], target_name = prep_data(dsname, target, without)
    >>> data = [X_train, Y_train, X_test, Y_test]
    >>> model, train_rmse, test_rmse = generate_xgb_model(3, 3, 3, data, model_type="Regressor")



    :param k:
    :param d:
    :param ld:
    :param data:
    :param model_type:
    :return:
    """

    if model_type == "Classifier":
        model = xgb.XGBClassifier(max_depth=d, n_estimators=k, verbosity = 0, objective='reg:squarederror',reg_lambda=ld)
    elif model_type == "Regressor":
        model = xgb.XGBRegressor(max_depth=d, n_estimators=k, verbosity = 0, objective='reg:squarederror',reg_lambda=ld)

    [X_train, Y_train, X_test, Y_test] = data
    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=[(X_test, Y_test)], verbose=False)


    preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Y_train, preds))

    preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, preds))

    return model, train_rmse, test_rmse



def count_trees(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    return len(trees)

def count_nodes(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len(tree) for tree in trees])
    return nodes

