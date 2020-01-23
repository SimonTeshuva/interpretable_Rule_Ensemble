import os
import pandas as pd

from orb.data_preprocessing import prep_data
from orb.xgb_functions import generate_xgb_model
from orb.xgb_functions import count_nodes

def exp(fn, target=None, without=[], path="", model_class = "xgboost", clear_results = False):
    """

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

    [X_train, Y_train, X_test, Y_test] = prep_data(fn, target, without)
    data = [X_train, Y_train, X_test, Y_test]

    for k in [3, 10]:
        for d in [1,3,5]:
            for ld in [1, 10, 100]:
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
                    model, train_RMSE,test_RMSE = generate_xgb_model(k, d, ld, data)
                    model_complexity = count_nodes(model)

                exp_res = {"dataset": fn, "target": target, "no_feat": len(X_train.columns),
                           "no_rows": Y_train.shape[0] + Y_test.shape[0], "tr_RMSE": train_RMSE, "te_RMSE": test_RMSE,
                           "model_complexity": model_complexity}
                res_df = res_df.append(exp_res, ignore_index=True)

    res_df.to_pickle("./results.pkl") #save current results table

    testing = True
    if testing == True:
        read_results()

def read_results():
    res_df = pd.read_pickle("./results.pkl") #open old results
    print("Dataframe Contents ", res_df, sep='\n')
    res_df.to_pickle("./results.pkl") #save current results table


if __name__ == "__main__":
    dataset_name = "titanic"
    target = "Survived"
    without = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    exp(dataset_name, target, without, clear_results=False)
