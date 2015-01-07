"""
The select module offers means to select sets of bricks and to filter
lists of variables, application calls and the like by their attributes,
brick ownership etc.

Selecting bricks
----------------

Bricks are selected using expressions similar to XPath formed of:
- the child selector '/'
- the all children selector '//'
- the name matcher (a valid python identifier)

Guys comment do we want this:
- the regexp name matcher in square brackets

- the attribute existence matcher @attribute
- the attribute value matcher @attribute==value
"""

import logging
import re
import operator

import warnings

logger = logging.getLogger(__name__)


class SelectorParseException(Exception):
    """
    Exception raised when the selector expression can't be parsed.
    """


class AbstractSelector(object):
    @staticmethod
    def partial_parse_path(path):
        """
        Check whether the be begining of the path can be interpreted
        as this selector. If it can, return the number of characters
        this type of selector matches and the selector.
        If it cannot, return 0, None.
        """
        return 0, None

    def _match(self, brick):
        "Helper for enumerate, do not use directly"
        return False

    def enumerate(self, bricks):
        for brick in bricks:
            if self._match(brick):
                yield brick

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            if self._match(brick):
                yield path, brick

    def enumerate_paths(self, bricks, path=''):
        if isinstance(bricks, dict):
            return self._enumerate_paths(bricks)
        else:
            paths_to_bricks = {"{}{}".format(path, b.name): b for b in bricks}
            return self._enumerate_paths(paths_to_bricks)


class ChildSelector(AbstractSelector):
    """
    The single '/' selector - enumerate children

    Parses:
    /
    """

    CHILDREN_FIELD = 'children'

    @staticmethod
    def partial_parse_path(path):
        if path.startswith('/'):
            if path.startswith('//'):
                warnings.warn("The AllDescendantsSelector (//) selector should"
                              "be matched before the Child (/) selector")
                return 0, None
            return 1, ChildSelector()
        else:
            return 0, None

    def __init__(self, **kwargs):
        super(ChildSelector, self).__init__(**kwargs)

    def enumerate(self, bricks):
        for brick in bricks:
            for c in getattr(brick, ChildSelector.CHILDREN_FIELD, []):
                yield c

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            for c in getattr(brick, ChildSelector.CHILDREN_FIELD, []):
                yield ("{}/{}".format(path, c.name), c)


class FieldSelector(AbstractSelector):
    """
    The single '.field' selector - enumerate contents of .field

    Parses:
    .field
    """

    FIELD_IDENTIFIER_RE = re.compile('^\.([^\d\W]\w*)')

    @staticmethod
    def partial_parse_path(path):
        m = FieldSelector.FIELD_IDENTIFIER_RE.match(path)
        if m:
            return m.end(), FieldSelector(m.group(1))
        return 0, None

    def __init__(self, field_name, **kwargs):
        super(FieldSelector, self).__init__(**kwargs)
        self.field_name = field_name

    def enumerate(self, bricks):
        for brick in bricks:
            for c in getattr(brick, self.field_name, []):
                yield c

    def _enumerate_paths(self, paths_to_bricks):
        for path, brick in paths_to_bricks.iteritems():
            for c in getattr(brick, self.field_name, []):
                yield ('{}.{}[{}]'.format(path,
                                          self.field_name,
                                          c.name),
                       c)


class AllDescendantsSelector(AbstractSelector):
    """
    The double '//' selector yields a brick and all its descendants.

    Parses:
    //
    """

    @staticmethod
    def partial_parse_path(path):
        if path.startswith('//'):
            return 2, AllDescendantsSelector()
        else:
            return 0, None

    def __init__(self, **kwargs):
        super(AllDescendantsSelector, self).__init__(**kwargs)

    def enumerate(self, bricks):
        def list_subbricks(brick):
            yield brick
            for child in brick.children:
                for c in list_subbricks(child):
                    yield c

        for brick in bricks:
            for r in list_subbricks(brick):
                yield r

    def _enumerate_paths(self, paths_to_bricks):
        def list_subbricks(path, brick):
            yield path, brick
            for child in brick.children:
                for c in list_subbricks("{}/{}".format(path, child.name),
                                        child):
                    yield c

        for path, brick in paths_to_bricks.iteritems():
            for r in list_subbricks(path, brick):
                yield r


class RegexpSelector(AbstractSelector):
    """
    The [name_regexp] selector matches names matching exactly the given
    regexp.

    Parses:
    -------
    [regular_expression]
    """
    @staticmethod
    def partial_parse_path(path):
        if not path.startswith('['):
            return 0, None
        num_paren = 1
        for offset, char in enumerate(path[1:]):
            if char == '[':
                num_paren += 1
            elif char == ']':
                num_paren -= 1
                if num_paren == 0:
                    break
        pattern_end = offset + 1
        if path[pattern_end] != ']':
            raise SelectorParseException("couldn't parse regexp selector "
                                         "starting with: {}".format(path))
        return pattern_end + 1, RegexpSelector(path[1:pattern_end])

    def __init__(self, name_re, **kwargs):
        super(RegexpSelector, self).__init__(**kwargs)
        self.name_re = re.compile('^{}$'.format(name_re))

    def _match(self, brick):
        return self.name_re.match(brick.name) is not None


class NotFound():
    pass


def recurrent_getattr(obj, fields):
    """Recursively get fields of obj.

    """
    for f in fields:
        if obj is NotFound:
            break
        obj = getattr(obj, f, NotFound)
    return obj


def recurrent_setattr(obj, fields, value):
    """Recursively set fields of obj.

    """
    for f in fields[::-1]:
        obj = getattr(obj, f)
    return setattr(obj, fields[-1], value)


class AttributeMatchSelector(AbstractSelector):
    """
    Realize matches by tests on attributes (by default on the name).

    Parses:
    name
    \@attr?
    \@attr == value
    \@attr != value
    """
    _operator_dict = {'==': operator.eq,
                      '!=': operator.ne,
                      }

    PYTHON_IDENTIFIER_RE = re.compile('^[^\d\W]\w*')
    ATTR_CHECK_RE = re.compile('^@([^\d\W]\w*'  # @ single identifier
                               '(?:\.[^\d\W]\w*)*)'  # dotted subfields
                               '(?:\?|'  # a ? or..
                               '(?:\s*({})\s*([^\d\W]\w*))'  # op & identifier
                               ')'  #
                               .format('|'.join(_operator_dict.keys())))

    @staticmethod
    def partial_parse_path(path):
        # 1. try a direct match on name
        m = AttributeMatchSelector.PYTHON_IDENTIFIER_RE.match(path)
        if m:
            return m.end(), AttributeMatchSelector('name',
                                                   operator.eq,
                                                   m.group(0))
        # 2. try a match on attribute value
        m = AttributeMatchSelector.ATTR_CHECK_RE.match(path)
        if m:
            attr_name, op, attr_val = m.groups()
            if op is not None:
                op = AttributeMatchSelector._operator_dict[op]
            else:
                op = operator.is_not
                attr_val = NotFound
            return m.end(), AttributeMatchSelector(attr_name,
                                                   op,
                                                   attr_val)
        return 0, None

    def __init__(self, attr_name, operator, attr_val, **kwargs):
        super(AttributeMatchSelector, self).__init__(**kwargs)
        self.attr_names = attr_name
        if isinstance(self.attr_names, str):
            self.attr_names = self.attr_names.split('.')
        self.attr_val = attr_val
        self.operator = operator

    def _match(self, brick):
        val = recurrent_getattr(brick, self.attr_names)
        return self.operator(val, self.attr_val)


class TerminatorSelector(AbstractSelector):
    """
    A dummy selector class which raises whtn the path is non-empty.
    It is used as a canary at the end of the Selector.ParseOrder list.
    """

    @staticmethod
    def partial_parse_path(path):
        raise SelectorParseException()


# Non-parsable selectors
class Filter(AbstractSelector):
    def __init__(self, filter_fun, **kwargs):
        super(Filter, self).__init__(**kwargs)
        self.filter_fun = filter_fun

    def _match(self, brick):
        return self.filter_fun(brick)


def _op_in(v, s):
    return v in s


def in_(set_or_attr_name, value_set=None):
    """Select based on set membership.

    Two variants are supported:
    1. in_(value_set) selects bricks that are in `value_set`
    2. in_(attr_name, value_set) selects bricks whose attribute (or sub-
       attributes) are in the `value_set`.

    Parameters
    ----------
    attr_name : string
        The name of the attribute, or dot-delimited chain of attribute names.
    value_set : set
        Set of values in which membership is evaluated.

    """
    if value_set:
        return AttributeMatchSelector(set_or_attr_name,
                                      _op_in,
                                      value_set)
    else:
        return Filter(lambda brick: brick in set_or_attr_name)


class BrickSelection(set):
    """A set of selected bricks. Provides conveninece methods on sets.

    """

    def enumerate(self, selector):
        if not isinstance(selector, AbstractSelector):
            selector = Selector(selector)
        return BrickSelection(selector.enuerate(self))

    def get_params(self):
        """Select all parameters.

        Shorthand for:
            Selector('//.params').enumerate_paths(self)

        """
        return Selector('//.params').enumerate_paths(self)

    def select(self, path):
        """Compatibility with old interface select method

        It differs from the :meth:`enumerate` method in the way forward
        slash is treated: select ignores it, while enumerate will select
        children.

        """
        if path.startswith('/') and not path.startswith('//'):
            path = path[1:]
        return self.enumerate(path)

    def setattr(self, name, value):
        fields = name.split('.')
        for o in self:
            recurrent_setattr(o, fields, value)


class Selector(AbstractSelector):
    ParseOrder = [AllDescendantsSelector, ChildSelector,
                  FieldSelector,
                  RegexpSelector,
                  AttributeMatchSelector,
                  TerminatorSelector]

    def __init__(self, path_or_selector_list, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.selectors = []
        if not isinstance(path_or_selector_list, list):
            selector_list = [path_or_selector_list]
        for path_or_sel in selector_list:
            if isinstance(path_or_sel, AbstractSelector):
                self.selectors.append(path_or_sel)
            else:
                assert isinstance(path_or_sel, str)
                path = path_or_sel
                while path:
                    for selector_klass in Selector.ParseOrder:
                        offset, sel = selector_klass.partial_parse_path(path)
                        if offset:
                            path = path[offset:]
                            self.selectors.append(sel)
                            break

    def enumerate(self, bricks):
        for sel in self.selectors:
            bricks = set(sel.enumerate(bricks))
        return BrickSelection(bricks)

    def _enumerate_paths(self, paths_to_bricks):
        for sel in self.selectors:
            paths_to_bricks = dict(sel.enumerate_paths(paths_to_bricks,
                                                       None))
        return paths_to_bricks
