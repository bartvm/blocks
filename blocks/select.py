import logging
import re
from collections import OrderedDict

from picklable_itertools.extras import equizip
import six

from blocks.bricks.base import Brick
from blocks.utils import dict_union

logger = logging.getLogger(__name__)

name_collision_error_message = """

The '{}' name appears more than once. Make sure that all bricks' children \
have different names and that user-defined shared variables have unique names.
"""


class Path(object):
    """Encapsulates a path in a hierarchy of bricks.

    Currently the only allowed elements of paths are names of the bricks
    and names of parameters. The latter can only be put in the end of the
    path. It is planned to support regular expressions in some way later.

    Parameters
    ----------
    nodes : list or tuple of path nodes
        The nodes of the path.

    Attributes
    ----------
    nodes : tuple
        The tuple containing path nodes.

    """
    separator = "/"
    param_separator = "."
    separator_re = re.compile("([{}{}])".format(separator, param_separator))

    class BrickName(str):

        def part(self):
            return Path.separator + self

    class ParamName(str):

        def part(self):
            return Path.param_separator + self

    def __init__(self, nodes):
        if not isinstance(nodes, (list, tuple)):
            raise ValueError
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

        .. todo::

            More error checking.

        Parameters
        ----------
        string : str
            String representation of the path.

        """
        elements = Path.separator_re.split(string)[1:]
        separators = elements[::2]
        parts = elements[1::2]
        if not len(elements) == 2 * len(separators) == 2 * len(parts):
            raise ValueError

        nodes = []
        for separator, part in equizip(separators, parts):
            if separator == Path.separator:
                nodes.append(Path.BrickName(part))
            elif Path.param_separator == Path.param_separator:
                nodes.append(Path.ParamName(part))
            else:
                # This can not if separator_re is a correct regexp
                raise ValueError("Wrong separator {}".format(separator))

        return Path(nodes)


class Selector(object):
    """Selection of elements of a hierarchy of bricks.

    Parameters
    ----------
    bricks : list of :class:`.Brick`
        The bricks of the selection.

    """
    def __init__(self, bricks):
        if isinstance(bricks, Brick):
            bricks = [bricks]
        self.bricks = bricks

    def select(self, path):
        """Select a subset of current selection matching the path given.

        .. warning::

            Current implementation is very inefficient (theoretical
            complexity is :math:`O(n^3)`, where :math:`n` is the number
            of bricks in the hierarchy). It can be sped up easily.

        Parameters
        ----------
        path : :class:`Path` or str
            The path for the desired selection. If a string is given
            it is parsed into a path.

        Returns
        -------
        Depending on the path given, one of the following:

        * :class:`Selector` with desired bricks.
        * list of :class:`~tensor.SharedTensorVariable`.

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

        Parameters
        ----------
        param_name : :class:`Path.ParamName`
            If given, only parameters with the name `param_name` are
            returned.

        Returns
        -------
        params : OrderedDict
            A dictionary of (`path`, `param`) pairs, where `path` is the
            string representation of the part to the parameter, `param` is
            the parameter.

        """
        def recursion(brick):
            # TODO path logic should be separate
            result = [(Path([Path.BrickName(brick.name),
                             Path.ParamName(param.name)]),
                       param)
                      for param in brick.params
                      if not param_name or param.name == param_name]
            result = OrderedDict(result)
            for child in brick.children:
                for path, param in recursion(child).items():
                    new_path = Path([Path.BrickName(brick.name)]) + path
                    if new_path in result:
                        raise ValueError(
                            "Name collision encountered while retrieving " +
                            "parameters." +
                            name_collision_error_message.format(new_path))
                    result[new_path] = param
            return result
        result = dict_union(*[recursion(brick)
                            for brick in self.bricks])
        return OrderedDict((str(key), value) for key, value in result.items())
