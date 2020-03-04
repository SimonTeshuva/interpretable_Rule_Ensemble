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

import pandas as pd

from orb.search import Conjunction
from orb.search import KeyValueProposition
from orb.search import Constraint
from orb.search import SquaredLossObjective


class Rule:
    """
    Represents a rule of the form "r(x) = y if q(x) else z"
    for some binary query function q.

    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[0]
    Name        Braund, Mr. Owen Harris
    Sex                            male
    Survived                          0
    Name: 0, dtype: object
    >>> titanic[['Name', 'Sex', 'Survived']].iloc[1]
    Name        Cumings, Mrs. John Bradley (Florence Briggs Th...
    Sex                                                    female
    Survived                                                    1
    Name: 1, dtype: object

    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> r = Rule(female, 1.0, 0.0)
    >>> r(titanic.iloc[0]), r(titanic.iloc[1])
    (0.0, 1.0)
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> opt = Rule()
    >>> opt.fit(titanic, target)
    >>> opt
    """

    def __init__(self, q = lambda x: True, y=0.0, z=0.0, reg=0.0, max_col_attr=10):
        self.q = q
        self.y = y
        self.z = z
        self.reg = reg
        self.max_col_attr = max_col_attr

    def __call__(self, x):
        return self.y if self.q(x) else self.z

    def __repr__(self):
        return f'{self.y:+10.4f} if {self.q}'

    def fit(self, data, target):
        """
        :param data:
        :param target:
        :return:
        """
        obj = SquaredLossObjective(data, target, reg=self.reg)
        self.q = obj.search(max_col_attr=self.max_col_attr)
        self.y = obj.opt_value((i for i in range(len(data)) if self.q(data.iloc[i])))


class AdditiveRuleEnsemble:
    """
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> model = AdditiveRuleEnsemble(reg=50, k=4)
    >>> model.fit(titanic, target)
    >>> model
       +0.6873 if Sex==female
       +0.2570 if Fare>=10.5 & Pclass<=2
       -0.2584 if Embarked==S & Fare>=7.8542 & Pclass>=3 & Sex==female
       +0.1324 if Pclass>=3 & Sex==male & SibSp<=1.0
    """

    def __init__(self, members=[], reg=0, k=3, max_col_attr=10):
        self.reg = reg
        self.members = members
        self.k = k
        self.max_col_attr = max_col_attr

    def __call__(self, x):
        return sum(r(x) for r in self.members)

    def __repr__(self):
        return str.join("\n", (str(r) for r in self.members))

    def fit(self, data, labels):
        self.members = []
        while len(self.members) < self.k:
            res = [labels.iloc[i] - self(data.iloc[i]) for i in range(len(data))]
            r = Rule(reg=self.reg)
            r.fit(data, res)
            self.members += [r]

