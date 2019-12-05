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


def greater_than(column, value):

    def cond(df):
        return df[column] < value

    return Proposition(cond, column+'>'+str(value))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
