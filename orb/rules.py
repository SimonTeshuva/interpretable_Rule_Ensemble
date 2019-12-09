"""
Provides basic classes and functions for representing rules and the
propositions they are based on.

Rules and propositions are callable Boolean functions that interact
well with Pandas. For example:

    
>>> import pandas as pd
>>> data = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
...       "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
...       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
...       "population": [200.4, 143.5, 1252, 1357, 52.98] }
>>> frame = pd.DataFrame(data)
>>> frame.iloc[0, 0]
'Brazil'

>>> p1 = equals("country", "Russia")
>>> str(p1)
'country=Russia'
>>> frame.loc[p1]
  country capital  area  population
1  Russia  Moscow  17.1       143.5

>>> p2 = less_than('population', 210)
>>> p3 = greater_than('area', 10)
>>> q = Conjunction([p2, p3])
>>> str(q)
'area>10 & population<210'
>>> frame.query(str(q))
  country capital  area  population
1  Russia  Moscow  17.1       143.5
"""

from statistics import median
import numpy as np
import pandas as pd
from orb.branchbound import *

class Rule_Ensamble:
    def __init__(self, df, target_col, numeric_metric):
        self.df = df
        self.target_col = target_col
        self.numeric_metric = numeric_metric
        self.propositions = develop_proposition_set(df, target_col, numeric_metric)
        rules_as_propositions = branch_and_bound(propositions, rule = ([], []), rules = [[True]])
        self.rules = Conjunction_of_ruleset(rules_as_propositions)            
        
    def __call__(self, df):
        return self.cond(df)

    def __str__(self):
        return self.string
    
    def __setdf__(self, dataframe):
        self.df = dataframe
    
    def __settargetcol__(self, targetcol):
        self.target_col = targetcol
        
    def __setnumericmetric__(self, metric):
        self.numeric_metric = metric
    
    def evaluate_propositions(self):
        self.propositions = develop_proposition_set(self.df, self.target_col, self.numeric_metric)

    def evaluate_rules(self):
        rules_as_propositions = branch_and_bound(self.propositions, rule = ([], []), rules = [[True]])
        self.rules = Conjunction_of_ruleset(rules_as_propositions)            

class Proposition:

    def __init__(self, cond, string):
        self.cond = cond
        self.string = string
        self.__call__ = cond

    def __call__(self, df):
        return self.cond(df)

    def __str__(self):
        return self.string


class Conjunction:

    def __init__(self, props):
        self.props = props
        self.string = str.join(" & ", sorted(map(str, props)))

    def __call__(self, df):
        print(str(map(lambda p: p(df), self.props)))
        return all(map(lambda p: p(df), self.props))

    def __str__(self):
        return self.string


def equals(column, value):
    
    def cond(df):
        return df[column] == value

    return Proposition(cond, column+'='+str(value))


def less_than(column, value):

    def cond(df):
        return df[column] < value

    return Proposition(cond, column+'<'+str(value))


def less_than_equal(column, value):

    def cond(df):
        return df[column] <= value

    return Proposition(cond, column+'<='+str(value))

def greater_than(column, value):

    def cond(df):
        return df[column] < value

    return Proposition(cond, column+'>'+str(value))

def greater_than_equal(column, value):

    def cond(df):
        return df[column] <= value

    return Proposition(cond, column+'>='+str(value))

def develop_proposition_set(dataframe, target_col, metrics_for_numeric = [median]):
        props = {}
        for col in dataframe.columns:
            if col != target_col:
                col_type = type(dataframe[str(col)][0])
                try:
                    col_is_num = (float == type(float(dataframe[col][0])))
                except TypeError:
                    col_is_num = False
                prop_type = col
                props[prop_type] = []
                if col_type == str:
                    for item in dataframe[col]:
                        prop = equals(col, item)
                        props[prop_type].append(prop)
                elif col_type == bool:
                        prop = equals(col, True)
                        props[prop_type].append(prop)
                        prop = equals(col, False)
                        props[prop_type].append(prop)
                elif col_is_num:
#                elif col_type in [int, float, np.float64, np.float32, np.int64, np.int32, np.uint8]:
                    for metric_for_numeric in metrics_for_numeric:
                        value = metric_for_numeric(sorted(list(dataframe[col])))
                        numerical_props = [less_than, less_than_equal, greater_than, greater_than_equal]
                        for numerical_prop in numerical_props:
                            prop = numerical_prop(col, value)
                            props[prop_type].append(prop)
                else:
                    print(col_type)
        return props


def Conjunction_of_ruleset(rules_as_propositions):            
    rules = []
    for rule_as_propositions in rules_as_propositions:
        if rule_as_propositions[0] is True:
            rule = True
        elif type(rule_as_propositions) is not list or len(rule_as_propositions) == 1:
            rule = rule_as_propositions[0]
        else:                
            rule = Conjunction(rule_as_propositions)
        rules.append(rule)
    return rules

                    
if __name__ == '__main__':
#    import doctest
#    doctest.testmod()
    data = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
            "capital": ["Brasilia", "Moscow", "New Delhi", "Beijing", "Pretoria"],
            "area": [8.516, 17.10, 3.286, 9.597, 1.221],
            "population": [200.4, 143.5, 1252, 1357, 52.98] }
    frame = pd.DataFrame(data)
    
    conj = Conjunction([equals('capital', 'Moscow'), equals('capital', 'Beiging')])
    print(conj.string)

