import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from re import search
import os
import graphviz
from matplotlib import pyplot as plt

def generate_xgb_model(data, k, d, ld, model_type="Regressor", norm_mean = False): # norm_mean redundant. go through code to see what needs to be removed
    """
    >>> fn1 = "titanic"
    >>> target = "Survived"
    >>> without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    >>> data, target_name, n  = prep_data(fn1, target, without)
    >>> model, train_rmse, test_rmse = generate_xgb_model(data, 3, 3, 3,  model_type="Regressor")

    >>> dsname = "halloween_candy_ranking"
    >>> target = "winpercent"
    >>> without = ['competitorname']
    >>> data, target_name, n  = prep_data(dsname, target, without)
    >>> model, train_rmse, test_rmse = generate_xgb_model(data, 3, 3, 3, model_type="Regressor")

    :param k:
    :param d:
    :param ld:
    :param data:
    :param model_type:
    :return:
    """

    [X_train, Y_train, X_test, Y_test] = data

    if model_type == "Classifier":
        model = xgb.XGBClassifier(max_depth=d, n_estimators=k, verbosity=0, objective='binary:logistic', reg_lambda=ld)
        eval_metric = "logloss"
    elif model_type == "Regressor":
        model = xgb.XGBRegressor(max_depth=d, n_estimators=k, verbosity=0, objective='reg:squarederror', reg_lambda=ld)
        eval_metric = "squarederror"

    model.fit(X_train, Y_train, early_stopping_rounds=10, eval_set=[(X_test, Y_test)], verbose=False)


  #  model = xgb.XGBRegressor(max_depth=d, n_estimators=k, verbosity=0, objective='reg:squarederror', reg_lambda=ld)
   # model.fit(X_train, Y_train, early_stopping_rounds=10, eval_set=[(X_test, Y_test)], verbose=False)



    preds = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(Y_train, preds))
    preds = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, preds))


    return model, train_rmse, test_rmse



def count_trees(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    return len(trees)

def count_nodes(xgb_model): # old model complexities were wrong??
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len([node for node in tree if node.find('leaf') == -1] ) for tree in trees])
    return nodes

def count_nodes_and_leaves(xgb_model): # old model complexities were wrong??
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    nodes = sum([len([node for node in tree]) for tree in trees])
    return nodes


def prediction_cost(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    forest_cost = 0
    leaves = 0
    for tree in trees:
        for node in tree:
            if node.find('leaf') != -1:
                leaf_cost = node.count('\t')+1
                leaves+=1
                forest_cost+=leaf_cost
    return forest_cost/leaves # change to average

def explanation_cost(xgb_model):
    trees = xgb_model.get_booster().get_dump()
    trees = [tree.split('\n') for tree in trees]
    forest_cost = 0
    for tree in trees:
        for node in tree:
            if node.find('leaf') != -1:
                node_depth = node.count('\t')
                forest_cost += node_depth+1
    return forest_cost


def save_xgb_model(model, dataset_name, experiment_timestamp):
    result_directory_models = os.path.join(os.getcwd(), "experiments", "results", dataset_name, experiment_timestamp)
    try:
        os.makedirs(result_directory_models)
    except FileExistsError:
        pass
    model.save_model(os.path.join(result_directory_models, "model"))

    with open(os.path.join(result_directory_models, "text_dump.txt"), 'w') as f:
        trees = model.get_booster().get_dump()
        for tree in trees:
            f.write(tree)

#    for tree in trees:
#        print(tree)

    return result_directory_models

def save_xgb_trees(model, dataset_name, experiment_timestamp):
    result_directory_trees = os.path.join(os.getcwd(), "experiments", "results", dataset_name, experiment_timestamp)
    try:
        os.makedirs(result_directory_trees)
    except FileExistsError:
        pass
    estimators = count_trees(model)
    for i in range(count_trees(model)):
        tree_dir = os.path.join(result_directory_trees, "tree " + str(i) + " of " + str(estimators)+".pdf")
        tree = xgb.plot_tree(model, num_trees=i)
        plt.savefig(tree_dir, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
        plt.close()
