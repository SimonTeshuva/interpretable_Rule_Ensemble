import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier
from orb.rules import AdditiveRuleEnsemble

imputer = SimpleImputer(strategy="most_frequent")
encoder = OneHotEncoder(handle_unknown='ignore')

target_name = 'Survived'
feature_names_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']

train = pd.read_csv('datasets/titanic2/train.csv')
test = pd.read_csv('datasets/titanic2/test.csv')
n_train = len(train)
n_test = len(test)

train_target = train[target_name]
test_target = test[target_name]

train.drop(columns=feature_names_to_drop + [target_name], inplace=True)
test.drop(columns=feature_names_to_drop + [target_name], inplace=True)

train_xgb = imputer.fit_transform(train)
test_xgb = imputer.transform(test)

train_xgb = encoder.fit_transform(train_xgb)
test_xgb = encoder.transform(test_xgb)

print("Fitting tree ensemble...")
trees = XGBClassifier(n_estimators=4, reg_lambda=10, gamma=0.1, verbosity=2)
trees.fit(train_xgb, train_target)

trees_prediction = trees.predict(test_xgb)
trees_residuals = test_target - trees_prediction
trees_accuracy = 1-sum(abs(trees_residuals))/len(trees_prediction)

print("Fitting rule ensemble...")
rules = AdditiveRuleEnsemble(k=4, reg=50)
rules.fit(train, train_target)

rules_prediction = [1 if rules(test.iloc[i])>0.5 else 0 for i in range(n_test)]
rules_residuals = [test_target[i] - rules_prediction[i] for i in range(n_test)]
rules_accuracy = 1 - sum(abs(r) for r in rules_residuals)/n_test
