from inspect import isclass
import re

from blocks.bricks.base import ApplicationCall, Brick
from blocks.roles import has_roles


def get_annotation(var, cls):
    """A helper function to retrieve an annotation of a particular type.

    Notes
    -----
    This function returns the first annotation of a particular type. If
    there are multiple--there shouldn't be--it will ignore them.

    """
    for annotation in getattr(var.tag, 'annotations', []):
        if isinstance(annotation, cls):
            return annotation


def get_brick(var):
    """Retrieves the brick that created this variable.

    See :func:`get_annotation`.

    """
    return get_annotation(var, Brick)


def get_application_call(var):
    """Retrieves the application call that created this variable.

    See :func:`get_annotation`.

    """
    return get_annotation(var, ApplicationCall)


class VariableFilter(object):
    """Filters Theano variables based on a range of criteria.

    Parameters
    ----------
    roles : list of :class:`.VariableRole` instances, optional
        Matches any variable which has one of the roles given.
    bricks : list of :class:`.Brick` classes or instances, optional
        Matches any variable whose brick is either one of the given
        bricks, or whose brick is of given classes.
    each_role : bool, optional
        If ``True``, the variable needs to have all given roles.  If
        ``False``, a variable matching any of the roles given will be
        returned. ``False`` by default.
    name : str, optional
        A regular expression for the variable name. The Blocks name (i.e.
        `x.tag.name`) is used.
    application : :class:`.Application` instance
        Matches a variable that was produced by the application given.

    Notes
    -----
    Note that only auxiliary variables, parameters, inputs and outputs are
    tagged with the brick that created them. Other Theano variables that
    were created in the process of applying a brick will be filtered out.

    Note that technically speaking, bricks are able to have non-shared
    variables as parameters. For example, we can use the transpose of
    another weight matrix as the parameter of a particular brick. This
    means that in some unusual cases, filtering by the :const:`PARAMETER`
    role alone will not be enough to retrieve all trainable parameters in
    your model; you will need to filter out the shared variables from these
    (using e.g. :func:`is_shared_variable`).

    Examples
    --------
    >>> from blocks.bricks import MLP, Linear, Sigmoid, Identity, BIAS
    >>> mlp = MLP(activations=[Identity(), Sigmoid()], dims=[20, 10, 20])
    >>> from theano import tensor
    >>> x = tensor.matrix()
    >>> y_hat = mlp.apply(x)
    >>> from blocks.graph import ComputationGraph
    >>> cg = ComputationGraph(y_hat)
    >>> from blocks.filter import VariableFilter
    >>> var_filter = VariableFilter(roles=[BIAS],
    ...                             bricks=[mlp.linear_transformations[0]])
    >>> var_filter(cg.variables)
    [b]

    """
    def __init__(self, roles=None, bricks=None, each_role=False, name=None,
                 application=None):
        self.roles = roles
        self.bricks = bricks
        self.each_role = each_role
        self.name = name
        self.application = application

    def __call__(self, variables):
        """Filter the given variables.

        Parameters
        ----------
        variables : list of :class:`~tensor.TensorVariable`

        """
        if self.roles:
            filtered_variables = []
            variables = [var for var in variables
                         if has_roles(var, self.roles, self.each_role)]
        if self.bricks is not None:
            filtered_variables = []
            for var in variables:
                var_brick = get_brick(var)
                if var_brick is None:
                    continue
                for brick in self.bricks:
                    if isclass(brick) and isinstance(var_brick, brick):
                        filtered_variables.append(var)
                        break
                    elif var_brick is brick:
                        filtered_variables.append(var)
                        break
            variables = filtered_variables
        if self.name:
            variables = [var for var in variables
                         if hasattr(var.tag, 'name') and
                         re.match(self.name, var.tag.name)]
        if self.application:
            variables = [var for var in variables
                         if get_application_call(var) and
                         get_application_call(var).application ==
                         self.application]
        return variables
