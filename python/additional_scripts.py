# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 11:56:10 2019

@author: simon
"""
import numpy as np
import pandas as pd
import os

def prepare_titanic():
    titanic_dir = "\\".join(os.getcwd().split("\\")[:-2]) + "\\Datasets\\titanic"
        
    train = pd.read_csv(titanic_dir + "\\train.csv")
    train['label'] = 'train'
    test = pd.read_csv(titanic_dir + "\\test.csv")
    test['label'] = 'test'

    survival = pd.read_csv(titanic_dir + "\\gender_submission.csv")

    # Any results you write to the current directory are saved as output.

    # create training data frame
    train_df = pd.DataFrame(train)
    train_df = train_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis='columns')
    #train_df = train_df.apply (pd.to_numeric, errors='coerce')
    train_df = train_df.dropna()
    train_df = train_df.drop(['label'], axis='columns')
    #train_df = train_df.reset_index(drop=True)
    train_df.head()

    # creating test data fram

    test_df = pd.DataFrame(test)

    survival_df = pd.DataFrame(survival)

    test_df['Survived'] = survival_df['Survived']

    # survival_df = survival_df.drop(['PassengerId'], axis = 'columns')
    # test_df['Survived'] = survival_df

    test_df = test_df.drop(['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'], axis='columns')
    test_df = test_df.dropna()
    test_df = test_df.drop(['label'], axis='columns')

    # X, Y train
    Y_train = train_df['Survived']
    X_train = train_df.drop(['Survived'], axis='columns')
    X_train  = pd.get_dummies(X_train)

    #imputer = SimpleImputer()
    #X_train = imputer.fit_transform(X_train)

    X_train.head()

    # X, Y test
    Y_test = test_df['Survived']
    X_test = test_df.drop(['Survived'], axis='columns')
    X_test = pd.get_dummies(X_test)
    
    data = [X_train, Y_train, X_test, Y_test]

    return data