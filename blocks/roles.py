def add_role(var, role):
    r"""Add a role to a given Theano variable.

    Parameters
    ----------
    var : :class:`~tensor.TensorVariable`
        The variable to assign the new role to.
    role : :class:`.VariableRole` instance

    Notes
    -----
    Some roles are subroles of others (e.g. :const:`WEIGHTS` is a subrole
    of :const:`PARAMETER`). This function will not add a role if a more
    specific role has already been added. If you need to replace a role
    with a parent role (e.g. replace :const:`WEIGHTS` with
    :const:`PARAMETER`) you must do so manually.

    Examples
    --------
    >>> from theano import tensor
    >>> W = tensor.matrix()
    >>> from blocks.roles import PARAMETER, WEIGHTS
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    PARAMETER
    >>> add_role(W, WEIGHTS)
    >>> print(*W.tag.roles)
    WEIGHTS
    >>> add_role(W, PARAMETER)
    >>> print(*W.tag.roles)
    WEIGHTS

    """
    roles = getattr(var.tag, 'roles', [])
    roles = [old_role for old_role in roles
             if not isinstance(role, old_role.__class__)]
    if not any(isinstance(old_role, role.__class__) for old_role in roles):
        roles += [role]
    var.tag.roles = roles


class VariableRole(object):
    """Base class for all variable roles."""
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


class WeightsRole(ParameterRole):
    pass

#: The weight matrices of linear transformations
WEIGHTS = WeightsRole()


class BiasesRole(ParameterRole):
    pass

#: Biases of linear transformations
BIASES = BiasesRole()


class FiltersRole(WeightsRole):
    pass

#: The filters (kernels) of a convolution operation
FILTERS = FiltersRole()
