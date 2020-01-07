from collections import deque
from sortedcontainers import SortedSet
from operator import attrgetter
from math import inf
import pandas as pd


class Node:

    def __init__(self, gen, clo, ext, idx, val, bnd):
        self.generator = gen
        self.closure = clo
        self.extension = ext
        self.gen_index = idx
        self.val = val
        self.val_bound = bnd

    def __repr__(self):
        return f'N({self.generator}, {self.closure}, {self.val:.5g}, {self.val_bound:.5g}, {list(self.extension)})'


value = attrgetter("val")


class Context:
    """
    Formal context, i.e., a binary relation between a set of objects and a set of attributes.
    """

    @staticmethod
    def from_tab(table):
        """
        Converts an input table where each row represents an object into
        a formal context (which uses column-based representation).

        Uses Boolean interpretation of table values to determine attribute
        presence for an object.

        For instance:

        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> list(ctx.extension([0, 2]))
        [1, 2]

        :param table:
        :return:
        """
        m = len(table)
        n = len(table[0])
        extents = [SortedSet([i for i in range(m) if table[i][j]]) for j in range(n)]
        return Context(list(range(n)), list(range(m)), extents)

    @staticmethod
    def from_df(df, max_col_attr=None):
        """
        Generates formal context from pandas dataframe by applying interordinal scaling to numerical data columns
        and for object columns creating one attribute per value.

        For interordinal scaling a maximum number of attributes per column can be specified. If required, threshold
        values are then selected quantile-based.

        The restriction should also be implemented for object columns in the future (by merging small categories
        into disjunctive propositions).

        The generated attributes correspond to pandas-compatible query strings. For example:

        >>> titanic_df = pd.read_csv("../datasets/titanic/train.csv")
        >>> titanic_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
        >>> titanic_ctx = Context.from_df(titanic_df, max_col_attr=6)
        >>> titanic_ctx.m
        891
        >>> titanic_ctx.attributes
        ['Survived<=0', 'Survived>=1', 'Pclass<=1', 'Pclass<=2', 'Pclass>=2', 'Pclass>=3', 'Sex=="male"', 'Sex=="female"', 'Age<=23.0', 'Age>=23.0', 'Age<=34.0', 'Age>=34.0', 'Age<=80.0', 'Age>=80.0', 'SibSp<=8.0', 'SibSp>=8.0', 'Parch<=6.0', 'Parch>=6.0', 'Fare<=8.6625', 'Fare>=8.6625', 'Fare<=26.0', 'Fare>=26.0', 'Fare<=512.3292', 'Fare>=512.3292', 'Embarked=="S"', 'Embarked=="C"', 'Embarked=="Q"', 'Embarked=="nan"']
        >>> titanic_ctx.n
        28

        :param df: pandas dataframe to be converted to formal context
        :param max_col_attr: maximum number of attributes generated per column
        :return: context representing dataframe
        """

        attributes = []
        extents = []
        for c in df:
            if df[c].dtype.kind in 'uif':
                vals = df[c].unique()
                reduced = False
                if max_col_attr and len(vals)*2 > max_col_attr:
                    _, vals = pd.qcut(df[c], q=max_col_attr // 2, retbins=True, duplicates='drop')
                    vals = vals[1:]
                    reduced = True
                vals = sorted(vals)
                for i, v in enumerate(vals):
                    if reduced or i < len(vals) - 1:
                        attributes += [f'{c}<={v}']
                        le = df[c] <= v
                        extents += [SortedSet([i for i in range(len(df[c])) if le[i]])]
                    if reduced or i > 0:
                        attributes += [f'{c}>={v}']
                        ge = df[c] >= v
                        extents += [SortedSet([i for i in range(len(df[c])) if ge[i]])]

            if df[c].dtype.kind in 'O':
                for v in df[c].unique():
                    attributes += [f'{c}=="{v}"']
                    eq = df[c] == v
                    extents += [SortedSet([i for i in range(len(df[c])) if eq[i]])]

        return Context(attributes, list(df.index), extents)

    def __init__(self, attributes, objects, extents):
        self.attributes = attributes
        self.objects = objects
        self.extents = extents
        self.n = len(attributes)
        self.m = len(objects)

    def extension(self, intent):
        """
        :param intent: attributes describing a set of objects
        :return: indices of objects that have all attributes in intent in common
        """
        if not intent:
            return range(len(self.objects))

        result = SortedSet.intersection(*map(lambda i: self.extents[i], intent))

        return result

    def bfs(self, f, g):
        """
        A first example with trivial objective and bounding function is as follows. In this example
        the optimal extension is the empty extension, which is generated via the
        the lexicographically smallest and shortest generator [0, 3].
        >>> table = [[0, 1, 0, 1],
        ...          [1, 1, 1, 0],
        ...          [1, 0, 1, 0],
        ...          [0, 1, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> search = ctx.bfs(lambda e: -len(e), lambda e: 1)
        >>> max(search, key=value)
        N([0, 3], [0, 1, 2, 3], 0, 1, [])

        ---> This looks like a bug! Why isn't the closure [0, 1, 2, 3]

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).
        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = ctx.bfs(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3])
        N([0], [0, 2], 0.5, 0.5, [1, 2])

        Finally, here is a complex example taken from the UdS seminar on subgroup discovery.
        >>> table = [[1, 1, 1, 1, 0],
        ...          [1, 1, 0, 0, 0],
        ...          [1, 0, 1, 0, 0],
        ...          [0, 1, 1, 1, 1],
        ...          [0, 0, 1, 1, 1],
        ...          [1, 1, 0, 0, 1]]
        >>> ctx = Context.from_tab(table)
        >>> labels = [1, 0, 1, 0, 0, 0]
        >>> f = impact(labels)
        >>> g = cov_incr_mean_bound(labels, impact_count_mean(labels))
        >>> search = ctx.bfs(f, g)
        >>> for n in search:
        ...     print(n)
        N([], [], 0, inf, [0, 1, 2, 3, 4, 5])
        N([0], [0], 0.11111, 0.22222, [0, 1, 2, 5])
        N([1], [1], -0.055556, 0.11111, [0, 1, 3, 5])
        N([2], [2, 3], 0.11111, 0.22222, [0, 2, 3, 4])
        N([3], [3], 0, 0.11111, [0, 3, 4])
        N([0, 2], [0, 2], 0.22222, 0.22222, [0, 2])

        :param f: objective function
        :param g: bounding function satisfying that g(I) >= max {f(J): J >= I}
        """
        boundary = deque()
        full = self.extension([])
        root_val = f(full)
        root = Node([], [], full, -1, root_val, inf)
        opt = root
        yield root
        boundary.append((range(self.n), root))

        def refinement(node, i):
            if i in node.closure:
                return None

            generator = node.generator + [i]
            extension = node.extension & self.extents[i]

            val = f(extension)
            bound = g(extension)

            if bound < opt.val:
                return None

            closure = []
            for j in range(0, i):
                if j in node.closure:
                    closure.append(j)
                    continue

                if extension >= self.extents[j]:
                    return None

            closure.append(i)

            for j in range(i + 1, self.n):
                if extension >= self.extents[j]:
                    closure.append(j)

            return Node(generator, closure, extension, i, val, bound)

        while boundary:
            ops, current = boundary.popleft()
            children = []
            for a in ops:
                child = refinement(current, a)
                if child:
                    opt = max(opt, child, key=value)
                    yield child
                    children += [child]
            filtered = list(filter(lambda c: c.val_bound > opt.val, children))
            ops = []
            for child in reversed(filtered):
                boundary.append((ops, child))
                ops = [child.gen_index] + ops


def cov_squared_dev(labels):
    n = len(labels)
    global_mean = sum(labels) / n

    def f(count, mean):
        return count/n * pow(mean - global_mean, 2)

    return f


def impact_count_mean(labels):
    """
    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> f(5, 0.4) # 1/2 * (2/5-1/5) = 1/6
    0.1
    """
    n = len(labels)
    m0 = sum(labels)/n

    def f(c, m):
        return c/n * (m - m0)

    return f


def impact(labels):
    """
    Compiles objective function for extension I defined by
    f(I) = len(I)/n (mean_I(l)-mean(l)) for some set of labels l of size n.

    >>> labels = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> f = impact(labels)
    >>> f([0, 1, 2, 3, 4]) # 0.5 * (0.4 - 0.2)
    0.1
    >>> f(range(len(labels)))
    0.0
    """
    g = impact_count_mean(labels)

    def f(extension):
        if len(extension) == 0:
            return -inf
        m = sum((labels[i] for i in extension))/len(extension)
        return g(len(extension), m)

    return f


def squared_loss_obj(labels):
    """
    Builds objective function that maps index set to product
    of relative size of index set times squared difference
    of mean label value described by index set to overall
    mean label value. For instance:

    >>> labels = [-4, -2, -1, -1, 0, 1, 10, 21]
    >>> sum(labels)/len(labels)
    3.0
    >>> obj = squared_loss_obj(labels)
    >>> obj([4, 5, 6, 7])  # local avg 8, relative size 1/2
    12.5

    :param labels: y-values
    :return: f(I) = |I|/n * (mean(y)-mean_I(y))^2
    """

    f = cov_squared_dev(labels)

    def label(i): return labels[i]

    def obj(extent):
        k = len(extent)
        local_mean = sum(map(label, extent)) / k

        return f(k, local_mean)

    return obj


def cov_incr_mean_bound(labels, f):
    """
    >>> labels = [1, 1, 1, 0, 1, 0, 1, 0, 0, 0]
    >>> f = impact_count_mean(labels)
    >>> g = cov_incr_mean_bound(labels, f)
    >>> g(range(len(labels)))
    0.25
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound


def cov_mean_bound(labels, f):
    """
    >>> labels = [-13, -2, -1, -1, 0, 1, 19, 21]
    >>> f = cov_squared_dev(labels)
    >>> obj = squared_loss_obj(labels)
    >>> obj(range(6,8))
    72.25
    >>> f(2, 20.0)
    72.25
    >>> bound = cov_mean_bound(labels, f)
    >>> bound(range(len(labels)))  # local avg 8, relative size 1/2
    72.25

    :param labels:
    :param f: any function that can be re-written as the maximum f(c, m)=max(g(c,m), h(c,m)) over functions g and h
              where g is monotonically increasing in its first and second argument (count and mean)
              and h is monotonically increasing in its first argument and monotonically decreasing in its second
              argument
    :return: bounding function that returns for any set of indices I, the maximum f-value over subsets J <= I
             where f is evaluated as f(|J|, mean(labels; J))
    """

    def label(i): return labels[i]

    def bound(extent):
        ordered = sorted(extent, key=label)
        k = len(ordered)
        opt = -inf

        s = 0
        for i in range(k):
            s += labels[ordered[-i-1]]
            opt = max(opt, f(i+1, s/(i+1)))

        s = 0
        for i in range(k):
            s += labels[ordered[i]]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound

