from collections import deque, namedtuple
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


class Context:
    """
    Formal context, i.e., a binary relation between a set of objects and a set of attributes.
    """

    def __init__(self, attributes, objects, extents):
        self.attributes = attributes
        self.objects = objects
        self.extents = extents

    def extension(self, intent):
        """
        :param intent: attributes describing a set of objects
        :return: indices of objects that have all attributes in intent in common
        """
        if not intent:
            return range(len(self.objects))

        result = self.extents[intent[0]][:]
        for i in intent[1:]:
            result = merge_intersection(result, self.extents[i])

        return result


def ctx_from_tab(table):
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
    >>> ctx = ctx_from_tab(table)
    >>> list(ctx.extension([0, 2]))
    [1, 2]

    :param table:
    :return:
    """
    m = len(table)
    n = len(table[0])
    extents = [[i for i in range(m) if table[i][j]] for j in range(n)]
    return Context(list(range(n)), list(range(m)), extents)


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
    >>> labels = [-4, -2, -1, -1, 0, 1, 10, 21]
    >>> f = cov_squared_dev(labels)
    >>> obj = squared_loss_obj(labels)
    >>> obj(range(6,8))
    39.0625
    >>> f(2, 15.5)
    39.0625
    >>> bound = cov_mean_bound(labels, f)
    >>> bound(range(len(labels)))  # local avg 8, relative size 1/2
    12.5

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
        s = 0
        opt = -inf

        for i in range(k):
            s += ordered[-i-1]
            opt = max(opt, f(i+1, s/(i+1)))

        for i in range(k):
            s += ordered[i]
            opt = max(opt, f(i+1, s/(i+1)))

        return opt

    return bound


Node = namedtuple('Node', ['generator', 'closure', 'gen_index', 'val', 'val_bound'])
value = attrgetter("val")


def bfs(obj, bound, context):
    n = len(context.attributes)
    boundary = deque()
    root = Node([], obj([]), None, inf, obj(context.all))
    opt = root
    boundary.append(root)

    def refinement(node, i):
        if i in node.closure:
            return None

        generator = node.generator + [i]
        extension = merge_intersection(node.extension, context.extension(i))

        val = obj(extension)
        val_bound = bound(extension)

        if val_bound < opt.val:
            return None

        closure = []
        for j in range(0, i):
            if j in node.closure:
                closure.append(j)
                continue

            if extension >= context.extension(j):
                return None

        for j in range(i + 1, n):
            if extension >= context.extension(j):
                closure.append(j)

        return Node(generator, closure, i, val, val_bound)

    while boundary:
        current = boundary.popleft()
        for a in range(current.gen_index + 1, n):
            child = refinement(current, a)
            if child:
                opt = max(opt, child, key=value)
                boundary.append(child)

    return opt
