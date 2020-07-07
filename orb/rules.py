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
from ast import literal_eval
from math import exp

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
        self.data = data
        self.target = [1 if target[i] == pos_class else -1 for i in range(len(target))]
        self.reg = reg
        self.scores = scores or [0.0] * len(target)

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

    def __init__(self, members=[], reg=0, k=3, max_col_attr=10, alpha_function=lambda k: 1, min_alpha=0.5, alphas=[],
                 objective=SquaredLossObjective):
        self.reg = reg
        self.members = members[:]
        self.k = k
        self.max_col_attr = max_col_attr
        self.alpha_function = alpha_function
        self.min_alpha = min_alpha
        self.alphas = alphas
        self.objective = objective

    def reset(self):
        self.members = []
        self.alphas = []

    def __call__(self, x):
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
        #        r = Rule(reg=self.reg, alpha=alpha, objective_function = (g,h))

        if not len(self.members):  # need to generalise this better
            # if query selects all data, do thing, else obj val
            obj = self.objective(data, labels, reg=self.reg)
            r.y = obj.opt_value(range(len(labels)))
            r.q = Conjunction(Context([], []).attributes)
        #            r.y = sum([scores[i] - 0 for i in range(len(scores))]) / (0.5 * self.reg + len(scores))
        else:
            # before conjunction sort propositions inversly proportional to probability
            r.fit(data, labels,
                  scores)  # g terms and h terms as columns are actually needed in a more general sense. g terms are residuals, h terms are a constant

        if not len(r.q) and len(self.members) >= 1:
            self.members[0].y += r.y
        else:
            self.members += [r]
            self.alphas.append(alpha)

    def fit(self, data, labels):
        while len(self.members) < self.k:
            self.add_rule(data, labels)

    #            print(self.members[-1])

    def score(self, data, labels, n=None):  # only works when target is in column 1
        if n == None:
            n = len(labels)
        rules_prediction = [self(data.iloc[i]) for i in range(n)]
        # rmse = sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) ** 0.5
        # rme = sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) ** 0.5
        rmse = (sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) / (len(labels))) ** 0.5
        return rmse

    def save_ensamble(self, exp_res):  # this or the other version is redundant
        import datetime

        dataset_name, target_name, ensamble, rmse_train, rmse_test, feature_names_to_drop = exp_res
        rules = ensamble.members
        alpha_function = ensamble.alpha_function  # cant save alpha function at present in an easily useable way. for now assume using same alpha as always
        alphas = ensamble.alphas
        # still need to save residulas, number of datapoints, plot
        model_complexity = ensamble.model_complexity()

        try:
            save_dir = os.path.join(os.getcwd(), "results", dataset_name, str(datetime.datetime.now()))
            print(save_dir)
            os.makedirs(save_dir)
        except FileExistsError:
            print('already made:', save_dir)
        except:
            print('other error')

        with open(os.path.join(save_dir, 'ensamble_data.txt'), 'w') as f:
            f.write('dataset_name_:_' + dataset_name + '\n')
            f.write('target_name_:_' + target_name + '\n')
            f.write('dropped_attributes_:_' + str(feature_names_to_drop) + '\n')
            #        f.write('alpha function: ' + str(alpha_function) + '\n')
            f.write('max_col_attr_:_' + str(ensamble.max_col_attr) + '\n')

            f.write('k_:_' + str(ensamble.k) + '\n')
            f.write('reg_:_' + str(ensamble.reg) + '\n')

            f.write('train_rmse_:_' + str(rmse_train) + '\n')
            f.write('test_rmse_:_' + str(rmse_test) + '\n')
            f.write('model_complexity_:_' + str(model_complexity) + '\n')
            f.write('min_alpha_:_' + str(ensamble.min_alpha) + '\n')

            #        for i in range(len(rules)):
            #            f.write('rule ' + str(i) + ":" + str(rules[i].__repr__()) + '\n')
            f.write('rules_:_' + str([[rule.y, rule.q, rule.z] for rule in rules]) + '\n')

            #        for i in range(len(alphas)):
            #            f.write('alpha ' + str(i) + ":" + str(alphas[i]) + '\n')
            f.write('alphas_:_' + str(alphas) + '\n')

        return save_dir


def save_exp_res(exp_res, timestamp, name="ensemble_results"):
    import datetime

    dataset_name, target_name, ensamble, train_scores, test_scores, model_complexities, feature_names_to_drop, n = exp_res

    rules = ensamble.members
    alpha_function = ensamble.alpha_function  # cant save alpha function at present in an easily useable way. for now assume using same alpha as always
    alphas = ensamble.alphas
    # still need to save, number of datapoints, plot

    try:
        save_dir = os.path.join(os.getcwd(), "results", dataset_name, timestamp)
        print(save_dir)
        os.makedirs(save_dir)
    except FileExistsError:
        print('already made:', save_dir)
    except:
        print('other error')

    with open(os.path.join(save_dir, name + '.txt'), 'w') as f:
        f.write('dataset_name_:_' + dataset_name + '\n')
        f.write('target_name_:_' + target_name + '\n')
        f.write('dropped_attributes_:_' + str(feature_names_to_drop) + '\n')
        #        f.write('alpha function: ' + str(alpha_function) + '\n')
        f.write('max_col_attr_:_' + str(ensamble.max_col_attr) + '\n')

        f.write('k_:_' + str(ensamble.k) + '\n')
        f.write('reg_:_' + str(ensamble.reg) + '\n')
        f.write('min_alpha_:_' + str(ensamble.min_alpha) + '\n')

        f.write('n_:_' + str(n) + '\n')
        f.write('train_scores_:_' + str(train_scores) + '\n')
        f.write('test_scores_:_' + str(test_scores) + '\n')
        f.write('model_complexities_:_' + str(model_complexities) + '\n')

        #        for i in range(len(rules)):
        #            f.write('rule ' + str(i) + ":" + str(rules[i].__repr__()) + '\n')
        f.write('rules_:_' + str([[rule.y, rule.q, rule.z] for rule in rules]) + '\n')

        #        for i in range(len(alphas)):
        #            f.write('alpha ' + str(i) + ":" + str(alphas[i]) + '\n')
        f.write('alphas_:_' + str(alphas) + '\n')
        result_df = pd.DataFrame()


def cast(val):
    if '.' in val:
        try:
            val = float(val)
        except ValueError:
            val = val
    else:
        try:
            val = int(val)
        except ValueError:
            val = val
    return val


def constraint_maker(string):
    v = 1
    k = 0

    ops = [("<=", Constraint.less_equals), (">=", Constraint.greater_equals), ("==", Constraint.equals),
           ("<", Constraint.less_than), (">", Constraint.greater_than)]
    for op, const in ops:
        if op in string:
            s = string.split(op)
            return const(s[v]), s[k]

    '''
    if "<=" in string:
        s = string.split("<=")
        return Constraint.less_equals(cast(s[v])), s[k]
    elif ">=" in string:
        s = string.split(">=")
        return Constraint.greater_equals(cast(s[v])), s[k]
    elif "<" in string:
        s = string.split("<")
        return Constraint.less_than(cast(s[v])), s[k]
    elif ">" in string:
        s = string.split("<")
        return Constraint.greater_than(cast(s[v])), s[k]
    elif "!=" in string:
        s = string.split("!=")
        return Constraint.not_equals(cast(s[v])), s[k]
    elif "==" in string:
        s = string.split("==")
        return Constraint.equals(cast(s[v])), s[k]
    '''


def string_to_conjunction(ps):
    if not len(ps):
        return Conjunction(Context([], []).attributes)
    kvps = ps.split(' & ')
    propositions = []
    for kvp in kvps:
        const, key = constraint_maker(kvp)
        proposition = KeyValueProposition(key=key, constraint=const)
        propositions.append(proposition)
    q = Conjunction(propositions)
    return q


def load_result(dataset_name, timestamp):
    result_dir = os.path.join(os.getcwd(), "results", dataset_name, timestamp, "ensamble_data.txt")
    result_dict = dict()
    with open(result_dir, 'r') as f:
        for line in f:
            attribute = line.strip().split('_:_')
            result_dict[attribute[0]] = attribute[1]

    reg = float(result_dict['reg'])
    k = int(result_dict['k'])
    max_col_attr = int(result_dict["max_col_attr"])
    alpha_function = lambda k: max(min(1 - k ** 2 / reg, 1), 0)
    min_alpha = float(result_dict["min_alpha"])

    alphas = literal_eval(result_dict["alphas"])
    #    alphas = [float(alpha) for alpha in result_dict["alphas"][1:-1].split(',')]

    rules_string = result_dict['rules']
    rules_string_list = [item.strip() for item in rules_string[1:-1].split(',')]

    members = []
    for i in range(0, len(rules_string_list), 3):
        (y, ps, z) = rules_string_list[i][1:], rules_string_list[i + 1], rules_string_list[i + 2][:-2]
        q = string_to_conjunction(ps)
        rule = Rule(y=float(y), q=q, z=float(z), reg=reg, max_col_attr=max_col_attr, alpha=alphas[int(i // 3)])
        members.append(rule)

    train_scores = literal_eval(result_dict["train_scores"])
    test_scores = literal_eval(result_dict["test_scores"])
    model_complexities = literal_eval(result_dict["model_complexities"])

    #    train_scores = [float(train_score) for train_score in result_dict["train_scores"][1:-1].split(',')]
    #    test_scores = [float(test_score) for test_score in result_dict["test_scores"][1:-1].split(',')]
    #    model_complexities = [int(model_complexity) for model_complexity in result_dict["model_complexities"][1:-1].split(',')]

    ARE = AdditiveRuleEnsemble(members=members, reg=reg, k=k, max_col_attr=max_col_attr, alpha_function=alpha_function,
                               min_alpha=min_alpha, alphas=alphas)

    return ARE, train_scores, test_scores, model_complexities


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)


