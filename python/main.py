
'''
def print_fucntion_names():
    current_path = os.getcwd()
    count = 1
    for file in os.listdir(current_path):
        if file != "main.py":
            print(str(count)+": " + "".join(file.split(".")[:-1]))
    print(current_path)

def import_script(pmName):
    pm = __import__(pmName)

def menu():

    return

'''

from additional_scripts import *
from functions import *
from datetime import datetime
import os
import sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    [X_train, Y_train, X_test, Y_test] = prepare_titanic()
    data = [X_train, Y_train, X_test, Y_test]
#    model = functions.generate_model(3, 10, X_train, Y_train, X_test, Y_test)
#    sweep_estimators(data, min_estimators = 1, max_estimators = 50)
#    generate_interpretability_curve(model, data, sweep_estimators, 1, 10, "nodes by estimators", "error", "estimators sweep")
    results = run_experiment("titanic", "test.csv", "train.csv", sweep_estimators, [1,20])
