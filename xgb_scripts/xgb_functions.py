import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def generate_xgb_model(data, k, d, ld, model_type="Regressor", norm_mean = False):
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
    if norm_mean:
        [X_train, [Y_train, Y_train_mean], X_test, [Y_test, Y_test_mean]] = data
    else:
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


    if norm_mean:
        preds = model.predict(X_train)
#        preds += Y_train_mean
#        Y_train += Y_train_mean
        train_rmse = np.sqrt(mean_squared_error(Y_train, preds))

        preds = model.predict(X_test)
#        preds += Y_test_mean
#        Y_test += Y_test_mean
        test_rmse = np.sqrt(mean_squared_error(Y_test, preds))
    else:
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

"""
def save_xgb_model(model, dataset_name, param_name, iteration_number, experiment_timestamp):
    result_directory_models = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp, param_name, "models")
    try:
        os.makedirs(result_directory_models)
    except FileExistsError:
        pass
    model.save_model(os.path.join(result_directory_models, str(iteration_number) + param_name))

def save_xgb_trees(model, dataset_name, param_name, iteration_number, experiment_timestamp):
    result_directory_trees = os.path.join(os.getcwd(), "Experiments", "results", dataset_name, experiment_timestamp, param_name, "trees")
    try:
        os.makedirs(result_directory_trees)
    except FileExistsError:
        pass
    estimators = count_trees(model)
    for i in range(count_trees(model)):
        tree_dir = os.path.join(result_directory_trees, str(estimators) + "estimators" + str(i)+".pdf")
        tree = xgb.plot_tree(model, num_trees=i)
        plt.savefig(tree_dir, dpi=1000, facecolor='w', edgecolor='w',
                    orientation='portrait', papertype=None, format=None,
                    transparent=False, bbox_inches=None, pad_inches=0.1,
                    frameon=None, metadata=None)
"""