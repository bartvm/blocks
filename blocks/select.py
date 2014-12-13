import logging
import re
from collections import OrderedDict

import six

from blocks.bricks import Brick
from blocks.utils import dict_union

logger = logging.getLogger(__name__)


class Path(object):
    """Encapsulates a path in a hierarchy of bricks.

    Currently the only allowed elements of paths are names of the bricks
    and names of parameters. The latter can only be put in the end of the
    path. It is planned to support regular expressions in some way later.

    Attributes
    ----------
    nodes : tuple
        The tuple containing path nodes.

    Parameters
    ----------
    nodes : list or tuple of path nodes
        The nodes of the path.

    """

    SEPARATOR = "/"
    PARAM_SEPARATOR = "."
    SEPARATOR_RE = re.compile("([{}{}])".format(SEPARATOR, PARAM_SEPARATOR))

    class BrickName(str):
        def part(self):
            return Path.SEPARATOR + self

    class ParamName(str):
        def part(self):
            return Path.PARAM_SEPARATOR + self

    def __init__(self, nodes):
        assert isinstance(nodes, (list, tuple))
        self.nodes = tuple(nodes)

    def __str__(self):
        return "".join([node.part() for node in self.nodes])

    def __add__(self, other):
        return Path(self.nodes + other.nodes)

    def __eq__(self, other):
        return self.nodes == other.nodes

    def __hash__(self):
        return hash(self.nodes)

    @staticmethod
    def parse(string):
        """Constructs a path from its string representation.

        Parameters
        ----------
        string : str
            String representation of the path.

        .. todo::

            More error checking.

        """
        elements = Path.SEPARATOR_RE.split(string)[1:]
        separators = elements[::2]
        parts = elements[1::2]
        assert len(elements) == 2 * len(separators) == 2 * len(parts)

        nodes = []
        for separator, part in zip(separators, parts):
            if separator == Path.SEPARATOR:
                nodes.append(Path.BrickName(part))
            elif separator == Path.PARAM_SEPARATOR:
                nodes.append(Path.ParamName(part))
            else:
                # This should not happen if SEPARATOR_RE is a correct regexp
                raise ValueError("Wrong separator {}".format(separator))

        return Path(nodes)


class Selector(object):
    """Selection of elements of a hierarchy of bricks.

    Parameters
    ----------
    bricks : list of Bricks
        The bricks of the selection.

    """

    def __init__(self, bricks):
        if isinstance(bricks, Brick):
            bricks = [bricks]
        self.bricks = bricks

    def select(self, path):
        """Select a subset of current selection matching the path given.

        Parameters
        ----------
        path : :class:`Path` or str
            The path for the desired selection. If a string is given
            it is parsed into a path.

        .. warning::

            Current implementation is very inefficient (theoretical
            complexity is :math:`O(n^3)`, where :math:`n` is the number
            of bricks in the hierarchy). It can be sped up easily.

        Returns
        -------
        Depending on the path given, one of the following:

        * A :class:`Selector` with desired bricks.
        * A list of shared Theano variables.

        """
        if isinstance(path, six.string_types):
            path = Path.parse(path)

        current_bricks = [None]
        for node in path.nodes:
            next_bricks = []
            if isinstance(node, Path.ParamName):
                return list(Selector(current_bricks).get_params(node).values())
            if isinstance(node, Path.BrickName):
                for brick in current_bricks:
                    children = brick.children if brick else self.bricks
                    matching_bricks = [child for child in children
                                       if child.name == node]
                    for match in matching_bricks:
                        if match not in next_bricks:
                            next_bricks.append(match)
            current_bricks = next_bricks
        return Selector(current_bricks)

    def get_params(self, param_name=None):
        """Returns parameters the selected bricks and their ancestors.

        .. todo::

           The parameters could be functions of shared variables (e.g. when
           using the transpose of a weight matrix, or when the shared
           variables have been collected into a single large one), and this
           method should probably return the underlying shared variables.

        Parameters
        ----------
        param_name : :class:`Path.ParamName`
            If given, only parameters with the name `param_name` are returned.

        Returns
        -------
        params : OrderedDict
            A dictionary of (`path`, `param`) pairs, where `path` is the string
            representation of the part to the parameter, `param` is the
            parameter.

        """
        def recursion(brick):
            # TODO path logic should be separate
            result = [(Path([Path.BrickName(brick.name),
                             Path.ParamName(param.name)]),
                       param)
                      for param in brick.params
                      if param_name is None or param.name == param_name]
            result = OrderedDict(result)
            for child in brick.children:
                for path, param in recursion(child).items():
                    new_path = Path([Path.BrickName(brick.name)]) + path
                    result[new_path] = param
            return result
        result = dict_union(*[recursion(brick)
                            for brick in self.bricks])
        return OrderedDict((str(key), value) for key, value in result.items())
