from collections import OrderedDict
from ..roles import add_role, AUXILIARY


def add_annotation(var, annotation):
    annotations = getattr(var.tag, 'annotations', [])
    if any(old_annotation.__class__ == annotation.__class__
           for old_annotation in annotations):
        raise ValueError
    else:
        var.tag.annotations = annotations + [annotation]


class Annotation(object):
    """Annotations on Theano variables in a graph.

    In Blocks annotations are automatically attached to variables created
    using bricks. One form of annotation is that many variables are
    assigned a role (see :class:`.VariableRole`). A second form of
    annotation comes in the form of attaching a :class:`Annotation`
    instance to the variable's ``tag`` attribute, with auxiliary variables
    and/or updates.

    For example, we might be interested in the mean activation of certain
    application of a :class:`.Linear` brick. The variable representing the
    mean activation is attached as an auxiliary variable to the annotations
    of the input and output variables of this brick. Using the
    :class:`ComputationGraph` class (the
    :attr:`~ComputationGraph.variables`,
    :attr:`~ComputationGraph.auxiliary_variables`, etc.  attributes in
    particular) we can retrieve these Theano variables to pass on to the
    monitor, use as a regularizer, etc.

    In most cases, annotations are added on a brick level (e.g. each brick
    will assign the weight norm of its weights as an auxiliary value) or on
    an application level (e.g. each time a brick is applied, its mean
    activation will become an auxiliary variable). However, you can also
    add annotations manually, by setting the ``annotation`` value of a
    variable's ``tag`` field.

    Examples
    --------
    >>> from theano import tensor
    >>> x = tensor.vector()
    >>> annotation = Annotation()
    >>> annotation.add_auxiliary_variable(x + 1, name='x_plus_1')
    >>> add_annotation(x, annotation)
    >>> y = x ** 2
    >>> from blocks.graph import ComputationGraph
    >>> cg = ComputationGraph([y])
    >>> cg.auxiliary_variables
    [x_plus_1]

    """
    def __init__(self):
        self.auxiliary_variables = []
        self.updates = OrderedDict()

    def add_auxiliary_variable(self, variable, roles=None, name=None):
        """Attach an auxiliary variable to the graph.

        Auxiliary variables are Theano variables that are not part of a
        brick's output, but can be useful nonetheless e.g. as a regularizer
        or to monitor during training progress.

        Parameters
        ----------
        variable : :class:`~tensor.TensorVariable`
            The variable you want to add.
        roles : list of :class:`.VariableRole` instances, optional
            The roles of this variable. The :const:`.AUXILIARY`
            role will automatically be added. Other options are
            :const:`.COST`, :const:`.WEIGHT`, etc.
        name : str, optional
            Name to give to the variable. If the variable already has a
            name it will be overwritten.

        Examples
        --------
        >>> from blocks.bricks.base import application, Brick
        >>> from blocks.roles import COST
        >>> from blocks.utils import shared_floatx_nans
        >>> class Foo(Brick):
        ...     def _allocate(self):
        ...         W = shared_floatx_nans((10, 10))
        ...         self.add_auxiliary_variable(W.mean(), name='mean_W')
        ...     @application
        ...     def apply(self, x, application_call):
        ...         application_call.add_auxiliary_variable(
        ...             x - 1, name='x_minus_1')
        ...         application_call.add_auxiliary_variable(
        ...             x.mean(), roles=[COST], name='mean_x')
        ...         return x + 1
        >>> from theano import tensor
        >>> x = tensor.vector()
        >>> y = Foo().apply(x)
        >>> from blocks.graph import ComputationGraph
        >>> cg = ComputationGraph([y])
        >>> from blocks.filter import VariableFilter
        >>> var_filter = VariableFilter(roles=[AUXILIARY])
        >>> var_filter(cg.variables)  # doctest: +SKIP
        {x_minus_1, mean_W, mean_x}
        >>> var_filter = VariableFilter(roles=[COST])
        >>> var_filter(cg.variables)  # doctest: +SKIP
        {mean_x}

        """
        add_annotation(variable, self)
        if name is not None:
            variable.name = name
            variable.tag.name = name
        add_role(variable, AUXILIARY)
        if roles is not None:
            for role in roles:
                add_role(variable, role)
        self.auxiliary_variables.append(variable)
