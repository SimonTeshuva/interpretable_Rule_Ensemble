from collections import deque, namedtuple
from sortedcontainers import SortedSet
from operator import attrgetter
from math import inf


def merge_intersection(a, b):
    """
    Forms the intersection of two sorted iterables in linear
    time. For instance:

    >>> a = [2, 40, 234, 349, 1001]
    >>> b = [-10, 2, 349, 500, 645, 702, 829]
    >>> merge_intersection(a, b)
    [2, 349]

    :param a: sorted iterable of mutually comparable elements
    :param b: sorted iterable of mutually comparable elements
    :return: sorted list of elements common to a and b
    """
    res = []

    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            res += [a[i]]
            i, j = i + 1, j + 1
        elif a[i] <= b[j]:
            i = i + 1
        elif a[i] >= b[j]:
            j = j + 1

    return res


Node = namedtuple('Node', ['generator', 'closure', 'extension', 'gen_index', 'val', 'val_bound'])
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
        Node(generator=[0, 3], closure=[0, 2, 3], extension=SortedSet([]), gen_index=3, val=0, val_bound=1)

        Let's use more realistic objective and bounding functions based on values associated with each
        object (row in the table).
        >>> values = [-1, 1, 1, -1]
        >>> f = lambda e: sum((values[i] for i in e))/4
        >>> g = lambda e: sum((values[i] for i in e if values[i]==1))/4
        >>> search = ctx.bfs(f, g)
        >>> for n in search:
        ...     print(n)
        Node(generator=[], closure=[], extension=range(0, 4), gen_index=-1, val=0.0, val_bound=inf)
        Node(generator=[0], closure=[0, 2], extension=SortedSet([1, 2]), gen_index=0, val=0.5, val_bound=0.5)

        :param f: objective function
        :param g: bounding function satisfying that g(I) >= max {f(J): J >= I}
        """
        n = len(self.attributes)
        boundary = deque()
        full = self.extension([])
        root = Node([], [], full, -1, f(full), inf)
        opt = root
        boundary.append(root)

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

            for j in range(i + 1, n):
                if extension >= self.extents[j]:
                    closure.append(j)

            return Node(generator, closure, extension, i, val, bound)

        while boundary:
            current = boundary.popleft()
            yield current
            for a in range(current.gen_index + 1, n):
                child = refinement(current, a)
                if child:
                    opt = max(opt, child, key=value)
                    boundary.append(child)


def cov_squared_dev(labels):
    n = len(labels)
    global_mean = sum(labels) / n

    def f(count, mean):
        return count/n * pow(mean - global_mean, 2)

    return f


def squared_loss_obj(labels):
    """
    Builds objective function that maps index to product
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

