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

>>> p1 = KeyValueProposition("country", Constraint.equals("Russia"))
>>> str(p1)
'country==Russia'
>>> frame.loc[p1]
  country capital  area  population
1  Russia  Moscow  17.1       143.5

>>> p2 = KeyValueProposition('population', Constraint.less(210))
>>> p3 = KeyValueProposition('area', Constraint.greater(10))
>>> q = Conjunction([p2, p3])
>>> str(q)
'area>10 & population<210'
>>> frame.query(str(q))
  country capital  area  population
1  Russia  Moscow  17.1       143.5
"""

import pandas as pd
import os
from math import exp

# from rulefit import RuleFit
# use r^2

# from realkd.search import Conjunction
# from realkd.search import KeyValueProposition
# from realkd.search import Constraint
# from realkd.search import SquaredLossObjective

from orb.search import Conjunction
from orb.search import KeyValueProposition
from orb.search import Constraint
# from orb.search import SquaredLossObjective
from orb.search import Context
from orb.search import DfWrapper, cov_mean_bound


class SquaredLossObjective:
    """
    Rule boosting objective function for squared loss.

    >>> titanic = pd.read_csv("../datasets/titanic/train.csv")
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
    >>> obj = SquaredLossObjective(titanic, titanic['Survived'])
    >>> female = Conjunction([KeyValueProposition('Sex', Constraint.equals('female'))])
    >>> first_class = Conjunction([KeyValueProposition('Pclass', Constraint.less_equals(1))])
    >>> obj(female)
    0.19404590848327577
    >>> reg_obj = SquaredLossObjective(titanic.drop(columns=['Survived']), titanic['Survived'], reg=2)
    >>> reg_obj(female)
    0.19342988972618597
    >>> reg_obj(first_class)
    0.09566220318908493
    >>> reg_obj._mean(female)
    0.7420382165605095
    >>> reg_obj._mean(first_class)
    0.6296296296296297
    >>> reg_obj.search()
    Sex==female
    """

    def __init__(self, data, target, scores=None, reg=0):
        """
        :param data:
        :param target: _series_ of residuals values of matching dimension
        :param reg:
        """
        self.m = len(data)
        self.data = DfWrapper(data) if isinstance(data, pd.DataFrame) else data
        scores = scores or [0] * self.m
        self.res = [target.iloc[i] - scores[i] for i in range(self.m)]
        self.reg = reg

    def _f(self, count, mean):
        return self._reg_term(count) * count / self.m * pow(mean, 2)

    def _reg_term(self, c):
        return 1 / (1 + self.reg / (2 * c))

    def _count(self, q):  # almost code duplication: Impact
        return sum(1 for _ in filter(q, self.data))

    def _mean(self, q):  # code duplication: Impact
        s, c = 0.0, 0.0
        for i in range(self.m):
            if q(self.data[i]):
                s += self.res[i]
                c += 1
        return s / c

    #
    def search(self, max_col_attr=10, alpha=1):
        # here we need the function in list of row indices; can we save some of these conversions?
        def f(rows):
            c = len(rows)
            if c == 0:
                return 0.0
            m = sum(self.res[i] for i in rows) / c
            return self._f(c, m)

        g = cov_mean_bound(self.res, lambda c, m: self._f(c, m))

        ctx = Context.from_df(self.data.df, max_col_attr=max_col_attr)
        return ctx.search(f, g, alpha)

    def opt_value(self, rows):
        s, c = 0.0, 0
        for i in rows:
            s += self.res[i]
            c += 1

        return s / (self.reg / 2 + c) if (c > 0 or self.reg > 0) else 0.0

    def __call__(self, q):
        c = self._count(q)
        m = self._mean(q)
        return self._f(c, m)


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
    +0.7420 if Sex==female
    """

    # max_col attribute to change number of propositions
    def __init__(self, objective=SquaredLossObjective, q=lambda x: True, y=0.0, z=0.0, reg=0.0, max_col_attr=10,
                 alpha=1):
        self.q = q
        self.y = y
        self.z = z
        self.reg = reg
        self.max_col_attr = max_col_attr
        self.alpha = alpha
        self.objective = objective

    def __call__(self, x):
        return self.y if self.q(x) else self.z

    def __repr__(self):
        return f'{self.y:+10.4f} if {self.q}'

    def fit(self, data, target, scores=None):
        """
        attempting to find the best rule for over X/Y (data/target). improving over prior scores

        :param data:
        :param target:
        :param scores: prior scores
        :return:

        """
        obj = self.objective(data, target, reg=self.reg,
                             scores=scores)  # scores not compatible w/ SLO <-- change to follow same convention.
        # create residuals within init. modify implementation for that
        self.q = obj.search(max_col_attr=self.max_col_attr, alpha=self.alpha)
        self.y = obj.opt_value((i for i in range(len(data)) if self.q(data.iloc[i])))



def exp_loss(y, score):
    return exp(-y * score)


class ExponentialObjective:
    """
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> female = KeyValueProposition('Sex', Constraint.equals('female'))
    >>> obj = ExponentialObjective(titanic.drop(columns=['Survived']), titanic.Survived, pos_class=1)
    >>> obj.opt_value(female)
    0.4840764331210191
    >>> obj(female)
    0.04129047016520477
    >>> q = obj.search()
    >>> obj.opt_value(q)
    >>> q
    """

    def __init__(self, data, target, pos_class=1.0, reg=0.0, scores=None):
        """
        :param data:
        :param target:
        :param reg:
        :param scores: proper scoring function l(y, s) = l(-y, -s)
        """
        # cast target to list??

        self.data = data
        self.target = [1 if target.loc[i] == pos_class else -1 for i in target.index] #iloc?
#        self.target = [1 if target[i] == pos_class else -1 for i in target.index] #iloc?
        self.reg = reg
        self.scores = scores or [0.0] * len(target)

        # not making the right target

    def _gradient_summary(self, rows):
        sum_g, sum_h = 0.0, 0.0
        for i in rows:
            loss = exp(-self.target[i] * self.scores[i])
            sum_g += -self.target[i] * loss
            sum_h += loss
        return sum_g, sum_h

    def _f(self, rows):
        if len(rows) == 0:
            return 0.0
        sum_g, sum_h = self._gradient_summary(rows)
        return sum_g ** 2 / (2 * len(self.data) * (self.reg + sum_h))

    def _g(self, rows):
        pos_rows = [i for i in rows if self.target[i] == 1]
        neg_rows = [i for i in rows if self.target[i] == -1]
        pos_val = self._f(pos_rows)
        neg_val = self._f(neg_rows)
        return max(pos_val, neg_val)

    def __call__(self, q):
        """
        g_i = -y*exp(-y*score)
        h_i = exp(-y*score)
        :param q:
        :return:
        """
        sum_g, sum_h = self._gradient_summary(i for i in range(len(self.data)) if q(self.data.iloc[i]))
        return sum_g ** 2 / (2 * len(self.data) * (self.reg + sum_h))

    def opt_value(self, rows):
        #        sum_g, sum_h = self._gradient_summary(i for i in range(len(self.data)) if q(self.data.iloc[i]))

        sum_g, sum_h = self._gradient_summary(rows)
        return -sum_g / (self.reg + sum_h)

    def search(self, max_col_attr=10, alpha=1):
        ctx = Context.from_df(self.data, max_col_attr=max_col_attr)
        return ctx.search(self._f, self._g, alpha)
        # # here we need the function in list of row indices; can we save some of these conversions?
        # def f(rows):
        #     c = len(rows)
        #     if c == 0:
        #         return 0.0
        #     m = sum(self.target[i] for i in rows) / c
        #     return self._f(c, m)
        #
        # g = cov_mean_bound(self.target, lambda c, m: self._f(c, m))
        #

        # return ctx.search(f, g)


class AdditiveRuleEnsemble:
    """
    >>> advertising = pd.read_csv('../datasets/advertising/advertising.csv')
    >>> target = advertising['Clicked_on_Ad']
    >>> advertising.drop(columns=["Ad_Topic_Line", "City", "Timestamp", 'Clicked_on_Ad'], inplace=True)
    >>> ad_class_model = AdditiveRuleEnsemble(reg=50, k=4, objective=ExponentialObjective)
    >>> ad_class_model.fit(advertising, target)
    >>> ad_class_model
       -0.0000 if
       -0.8720 if Age<=44.0 & Area Income>=43644.412000000004 & Daily Internet Usage>=163.44 & Daily Time Spent on Site>=62.26
       +0.8173 if Daily Internet Usage<=198.94800000000004 & Daily Time Spent on Site<=72.952
       +0.7543 if Age>=28.0 & Area Income<=66625.602 & Daily Internet Usage<=224.836 & Daily Time Spent on Site<=79.982
    >>> ad_class_model.score()
    >>> advertising = pd.read_csv('../datasets/advertising/advertising.csv')
    >>> target = advertising['Clicked_on_Ad']
    >>> advertising.drop(columns=["Ad_Topic_Line", "City", "Timestamp", 'Clicked_on_Ad'], inplace=True)
    >>> ad_class_model = AdditiveRuleEnsemble(reg=50, k=4)
    >>> ad_class_model.fit(advertising, target)
    >>> ad_class_model
        +0.4878 if
        -0.4474 if Age<=44.0 & Area Income>=43644.412000000004 & Daily Internet Usage>=163.44 & Daily Time Spent on Site>=62.26
        +0.4367 if Daily Internet Usage<=198.94800000000004 & Daily Time Spent on Site<=72.952
        -0.2583 if Area Income>=43644.412000000004 & Daily Internet Usage<=198.94800000000004 & Daily Internet Usage>=163.44 & Daily Time Spent on Site<=72.952 & Daily Time Spent on Site>=47.23
    >>> ad_class_model.score()
    >>> titanic = pd.read_csv('../datasets/titanic/train.csv')
    >>> target = titanic.Survived
    >>> titanic.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], inplace=True)
    >>> model = AdditiveRuleEnsemble(reg=50, k=4)
    >>> model.fit(titanic, target)
    >>> model
       +0.3734 if
       +0.5001 if Pclass<=2 & Sex==female
       -0.2301 if Fare<=39.6875 & Parch<=1.0 & Pclass>=2 & Sex==male
       +0.2417 if Age<=25.0 & Fare<=39.6875 & Fare>=7.8542 & Parch>=1.0 & SibSp<=1.0
    >>> class_model = AdditiveRuleEnsemble(reg=50, k=4, objective=ExponentialObjective)
    >>> class_model.fit(titanic, target)
    >>> class_model
       -0.2200 if
       +0.7501 if Pclass<=2 & Sex==female
       -0.5277 if Pclass>=2 & Sex==male
       +0.4007 if Sex==female & SibSp<=1.0

    """

    def __init__(self, members=[], reg=0, k=3, max_rules = None, max_col_attr=10, alpha_function=lambda k: 1, min_alpha=0.5, alphas=[],
                 objective=SquaredLossObjective):
        self.reg = reg
        self.members = members[:]
        self.k = k
        self.max_col_attr = max_col_attr
        self.alpha_function = alpha_function
        self.min_alpha = min_alpha
        self.alphas = alphas
        self.objective = objective
        self.max_rules = self.k if max_rules == None else max_rules

    def __call__(self, x): # look into swapping to Series and numpy
        return sum(r(x) for r in self.members)

    def __repr__(self):
        return str.join("\n", (str(r) for r in self.members))

    def model_complexity(self):
        model_complexity = 0
        for rule in self.members:
            rule_complexity = len(str(rule.__repr__()).split('&'))
            model_complexity += rule_complexity
        return model_complexity + len(self.members)  # counting prediction value of rule in complexity

    # have two methods, add first rule or other rule.
    # alternatively, create new ensable each time where we load an ensamble from data and .fit the new one.
    # ^ immutable datatypes --> less bug prone

    def add_rule(self, data, labels):
        scores = [self(data.iloc[i]) for i in range(len(data))]
        alpha = self.alpha_function(len(self.members))
        r = Rule(reg=self.reg, alpha=alpha, objective=self.objective)
        if not len(self.members):  # need to generalise this better --< by default, the model should make the default rule
            # if query selects all data, do thing, else obj val
            obj = self.objective(data, labels, reg=self.reg, scores=scores)
            r.y = obj.opt_value(range(len(labels)))
            r.q = Conjunction(Context([], []).attributes)
        else:
            # before conjunction sort propositions inversly proportional to probability
            r.fit(data, labels, scores)  # g terms and h terms as columns are actually needed in a more general sense. g terms are residuals, h terms are a constant

        queries = [self.members[i].q for i in range(len(self.members))]
        duplicate_rule=False
        if r.q in queries:
            for q,index in queries:
                if r.q == q:
                    self.members[index].y += r.y
                    break
        else:
            self.members += [r]
            self.alphas.append(alpha)

#        if not len(r.q) and len(self.members) >= 1: # this should hold for any rule, not just the rule. should also let us know which rule this triggered
            # if r.q in self.members, self.members[index].y+= r.y
            # need to avoid infinite loops
            # need to keep track of rules fitted and ensemble members seperately.
            # have separate k and max_k

            '''
            [[1, 1,1],
             [1, 1,1],
             [1, 1,1],
             [1, 1,1]] <-- this causes infinite loop?
            '''

    def fit(self, data, labels, warmstart = False, verbose=True): #warmstart would somehow increase self.k but start with the existing rules
        self.max_rules += self.k if warmstart == True else 0
        rules_created = len(self.members)
        while len(self.members) < self.k and rules_created<self.max_rules: # continue until self.k many new rules have been attempted

            scores = [self(data.iloc[i]) for i in range(len(data))]
            alpha = self.alpha_function(len(self.members)) # alpha function is just a constant funciton at the moment, but can be expanded later
            r = Rule(reg=self.reg, alpha=alpha, objective=self.objective)
            if not len(self.members): # creating the default rule as the first rule. Normalised mean as r.y and empty rule as r.q
                obj = self.objective(data, labels, reg=self.reg, scores=scores)
                r.y = obj.opt_value(range(len(labels)))
                r.q = Conjunction(Context([], []).attributes)
            else:
                r.fit(data, labels,scores) # in all other cases, fit a rule to the residuals

            queries = [self.members[i].q for i in range(len(self.members))]
            if r.q in queries: # if the recently created query exists in the ensemble already, do not create a new query. just update the target value
                for q, index in queries:
                    if r.q == q:
                        self.members[index].y += r.y
                        break
            else: # otherwise, create a new rule
                self.members += [r]
                self.alphas.append(alpha)

            rules_created += 1 # either way, a new rule has been created. incrementing to avoid infinite loops in degenerate datasets

            # self.add_rule(data, labels)
            if verbose:
                print(self.members[-1])

    def score(self, data, labels):  # only works when target is in column 1, only uses squared loss as metric of accuracy. add score functionality to loss functions
        n = len(labels) # make sure downsampling is done correctly with sklearn
        rules_prediction = [self(data.iloc[i]) for i in range(n)]

        rmse = (sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) / (len(labels))) ** 0.5 # swap to a call to the loss function
        return rmse # make this comparable with rule_fit package
        # maybe use accuracy as metric for score in classification experiments
        # sklearn.auc, makes use of predict - also an option

    def predict(self, data):
        # returns predictions as Series/numpy array, mostly a wrapper around self.__call__(data)
        pass

    def predict_proba(self, Xnew): # look into this
        pass




if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)


