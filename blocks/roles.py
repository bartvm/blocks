def add_role(var, role):
    r"""Add a role to a given Theano variable.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        The variable to assign the new role to.
    role : :class:`.VariableRole` instance

    Notes
    -----
    Some roles are subroles of others (e.g. :const:`WEIGHT` is a subrole
    of :const:`PARAMETER`). This function will not add a role if a more
    specific role has already been added. If you need to replace a role
    with a parent role (e.g. replace :const:`WEIGHT` with
    :const:`PARAMETER`) you must do so manually.

    Examples
    --------
    >>> from theano import tensor
    >>> W = tensor.matrix()
    >>> from blocks.roles import PARAMETER, WEIGHT
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    PARAMETER
    >>> add_role(W, WEIGHT)
    >>> print(*W.tag.roles)
    WEIGHT
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    WEIGHT

    """
    roles = getattr(var.tag, 'roles', [])
    roles = [old_role for old_role in roles
             if not isinstance(role, old_role.__class__)]
    if not any(isinstance(old_role, role.__class__) for old_role in roles):
        roles += [role]
    var.tag.roles = roles


def has_roles(var, roles, match_all=False):
    r"""Test if a variable has given roles taking subroles into account.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        Variable being queried.
    roles : an iterable of :class:`.VariableRole` instances.
    match_all : bool, optional
        If ``True``, checks if the variable has all given roles.
        If ``False``, any of the roles is sufficient.
        ``False`` by default.

    """
    var_roles = getattr(var.tag, 'roles', [])
    matches = (any(isinstance(var_role, role.__class__) for
                   var_role in var_roles) for role in roles)
    return all(matches) if match_all else any(matches)


class VariableRole(object):
    """Base class for all variable roles."""
    def __eq__(self, other):
        return self.__class__ == other.__class__

    def __repr__(self):
        return self.__class__.__name__[:-4].upper()


class InputRole(VariableRole):
    pass

#: The input of a :class:`.Brick`
INPUT = InputRole()


class OutputRole(VariableRole):
    pass

#: The output of a :class:`.Brick`
OUTPUT = OutputRole()


class CostRole(VariableRole):
    pass

#: A scalar cost that can be used to train or regularize
COST = CostRole()


class ParameterRole(VariableRole):
    pass

#: A parameter of the model
PARAMETER = ParameterRole()


class AuxiliaryRole(VariableRole):
    pass

#: Variables added to the graph as annotations
AUXILIARY = AuxiliaryRole()


class WeightRole(ParameterRole):
    pass

#: The weight matrices of linear transformations
WEIGHT = WeightRole()


class BiasRole(ParameterRole):
    pass

#: Biases of linear transformations
BIAS = BiasRole()


class FilterRole(WeightRole):
    pass

#: The filters (kernels) of a convolution operation
FILTER = FilterRole()


class DropoutRole(VariableRole):
    pass

#: Inputs with applied dropout
DROPOUT = DropoutRole()
