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

#from realkd.search import Conjunction
#from realkd.search import KeyValueProposition
#from realkd.search import Constraint
#from realkd.search import SquaredLossObjective

from orb.search import Conjunction
from orb.search import KeyValueProposition
from orb.search import Constraint
from orb.search import SquaredLossObjective
from orb.search import Context


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
    def __init__(self, q = lambda x: True, y=0.0, z=0.0, reg=0.0, max_col_attr=10, alpha = 1):
        self.q = q
        self.y = y
        self.z = z
        self.reg = reg
        self.max_col_attr = max_col_attr
        self.alpha = alpha

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
        self.q = obj.search(max_col_attr=self.max_col_attr, alpha = self.alpha)
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

    def __init__(self, members=[], reg=0, k=3, max_col_attr=10, alpha_function = lambda k: 1, min_alpha = 0.5, alphas = []):
        self.reg = reg
        self.members = members
        self.k = k
        self.max_col_attr = max_col_attr
        self.alpha_function = alpha_function
        self.min_alpha = min_alpha
        self.alphas = alphas

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
        return model_complexity-1

    # have two methods, add first rule or other rule.
    # alternatively, create new ensable each time where we load an ensamble from data and .fit the new one.
    # ^ immutable datatypes --> less bug prone

    def add_rule(self, data, labels):
        res = [labels.iloc[i] - self(data.iloc[i]) for i in range(len(data))]
        alpha = self.alpha_function(len(self.members))
        r = Rule(reg=self.reg, alpha=alpha)
#        r = Rule(reg=self.reg, alpha=alpha, objective_function = (g,h))

        if not len(self.members):  # need to generalise this better
            r.y = sum([res[i] - 0 for i in range(len(res))]) / (0.5 * self.reg + len(res))
            r.q = Conjunction(Context([], []).attributes)
        else:
            r.fit(data, res) # g terms and h terms as columns are actually needed in a more general sense. g terms are residuals, h terms are a constant

        if not len(r.q) and len(self.members) == 1:
            self.members[0].y += r.y
        else:
            self.members += [r]
            self.alphas.append(alpha)

    def fit(self, data, labels):
        self.members = []

        while len(self.members) < self.k:
            self.add_rule(data, labels)


    def score(self, data, labels, n=None):  # only works when target is in column 1
        if n==None:
            n=len(labels)
        rules_prediction = [self(data.iloc[i]) for i in range(n)]
        #rmse = sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) ** 0.5
        # rme = sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)]) ** 0.5
        rmse = (sum([(labels.iloc[i] - rules_prediction[i]) ** 2 for i in range(n)])/(len(labels))) ** 0.5
        return rmse

    def save_ensamble(self, exp_res): # this or the other version is redundant
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




def save_exp_res(exp_res):
    import datetime

    dataset_name, target_name, ensamble, train_scores, test_scores, model_complexities, feature_names_to_drop, n = exp_res

    rules = ensamble.members
    alpha_function = ensamble.alpha_function # cant save alpha function at present in an easily useable way. for now assume using same alpha as always
    alphas = ensamble.alphas
    # still need to save, number of datapoints, plot

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

    ops = [("<=", Constraint.less_equals), (">=", Constraint.greater_equals), ("==", Constraint.equals), ("<", Constraint.less_than), (">", Constraint.greater_than)]
    for op,const in ops:
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
        (y, ps, z) = rules_string_list[i][1:], rules_string_list[i+1], rules_string_list[i+2][:-2]
        q = string_to_conjunction(ps)
        rule = Rule(y=float(y), q=q, z=float(z), reg=reg, max_col_attr=max_col_attr, alpha=alphas[int(i // 3)])
        members.append(rule)

    train_scores = literal_eval(result_dict["train_scores"])
    test_scores = literal_eval(result_dict["test_scores"])
    model_complexities = literal_eval(result_dict["model_complexities"])

#    train_scores = [float(train_score) for train_score in result_dict["train_scores"][1:-1].split(',')]
#    test_scores = [float(test_score) for test_score in result_dict["test_scores"][1:-1].split(',')]
#    model_complexities = [int(model_complexity) for model_complexity in result_dict["model_complexities"][1:-1].split(',')]


    ARE = AdditiveRuleEnsemble(members=members, reg=reg, k=k, max_col_attr=max_col_attr, alpha_function=alpha_function, min_alpha=min_alpha, alphas=alphas)

    return ARE, train_scores, test_scores, model_complexities


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose = True)
    

