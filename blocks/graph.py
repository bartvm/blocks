"""Annotated computation graph management."""
import logging
from collections import OrderedDict
from itertools import chain

import theano
from theano import Variable
from theano.gof import graph
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.scan_module.scan_op import Scan
from toolz import unique

from blocks import config
from blocks.roles import add_role, has_roles, AUXILIARY, PARAMETER
from blocks.utils import (is_graph_input, is_shared_variable, dict_union,
                          shared_like)

logger = logging.getLogger(__name__)


class ComputationGraph(object):
    r"""Encapsulates a managed Theano computation graph.

    This implies that it not only contains the variables required to
    compute the given outputs, but also all the auxiliary variables and
    updates that were attached to these variables through the annotation
    system.

    All variables are presented in topologically sorted order according to
    the apply nodes that they are an input to.

    Parameters
    ----------
    outputs : (list of) :class:`~tensor.TensorVariable`
        The output(s) of the computation graph.

    Attributes
    ----------
    inputs : list of :class:`~tensor.TensorVariable`
        The inputs of the computation graph. This does not include shared
        variables and constants.
    shared_variables : list of :class:`~tensor.TensorSharedVariable`
        All the shared variables in the graph.
    parameters : list of :class:`~tensor.TensorSharedVariable`
        All the shared variables which have the :const:`.PARAMETER` role.
    outputs : list of :class:`~tensor.TensorVariable`
        The outputs of the computations graph (as passed to the
        constructor).
    auxiliary_variables : list of :class:`~tensor.TensorVariable`
        All variables which have the :const:`.AUXILIARY` role.
    intermediary_variables : list of :class:`~tensor.TensorVariable`
        Any variable that is not part of :attr:`inputs` or :attr:`outputs`.
    variables : list of :class:`~tensor.TensorVariable`
        All variables (including auxiliary) in the managed graph.
    scans : list of :class:`~theano.scan_module.scan_op.Scan`
        All Scan ops used in this computation graph.
    scan_variables : list of :class:`~tensor.TensorVariable`
        All variables of the inner graphs of Scan ops.
    updates : :class:`~tensor.TensorSharedVariable` updates
        All the updates found attached to the annotations.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()
        self._has_inputs = {}

    def __iter__(self):
        return iter(self.variables)

    @property
    def inputs(self):
        """Inputs to the graph, excluding constants and shared variables."""
        return [var for var in self.variables if is_graph_input(var)]

    @property
    def intermediary_variables(self):
        return [var for var in self.variables if
                var not in self.inputs and
                var not in self.outputs]

    @property
    def shared_variables(self):
        return [var for var in self.variables if is_shared_variable(var)]

    @property
    def parameters(self):
        return [var for var in self.shared_variables
                if has_roles(var, [PARAMETER])]

    @property
    def auxiliary_variables(self):
        return [var for var in self.variables if has_roles(var, [AUXILIARY])]

    @property
    def scan_variables(self):
        """Variables of Scan ops."""
        return list(chain(*[g.variables for g in self._scan_graphs]))

    def _get_variables(self):
        """Collect variables, updates and auxiliary variables.

        In addition collects all :class:`.Scan` ops and recurses in the
        respective inner Theano graphs.

        """
        updates = OrderedDict()

        shared_outputs = [o for o in self.outputs if is_shared_variable(o)]
        usual_outputs = [o for o in self.outputs if not is_shared_variable(o)]
        variables = shared_outputs

        if usual_outputs:
            # Sort apply nodes topologically, get variables and remove
            # duplicates
            inputs = graph.inputs(self.outputs)
            sorted_apply_nodes = graph.io_toposort(inputs, usual_outputs)
            self.scans = list(unique([node.op for node in sorted_apply_nodes
                                     if isinstance(node.op, Scan)]))
            self._scan_graphs = [ComputationGraph(scan.outputs)
                                 for scan in self.scans]

            seen = set()
            main_vars = [var for var in list(chain(
                *[apply_node.inputs for apply_node in sorted_apply_nodes]))
                if not (var in seen or seen.add(var))] + self.outputs

            # While preserving order add auxiliary variables, and collect
            # updates
            seen = set()
            # Intermediate variables could be auxiliary
            seen_avs = set(main_vars)
            variables = []
            for var in main_vars:
                variables.append(var)
                for annotation in getattr(var.tag, 'annotations', []):
                    if annotation not in seen:
                        seen.add(annotation)
                        new_avs = [
                            av for av in annotation.auxiliary_variables
                            if not (av in seen_avs or seen_avs.add(av))]
                        variables.extend(new_avs)
                        updates = dict_union(updates, annotation.updates)

        self.variables = variables
        self.updates = updates

    def dict_of_inputs(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.inputs}

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        Parameters
        ----------
        replacements : dict
            The mapping from variables to be replaced to the corresponding
            substitutes.

        """
        return ComputationGraph(theano.clone(self.outputs,
                                             replace=replacements))

    def get_theano_function(self, additional_updates=None):
        """Create Theano function from the graph contained."""
        updates = self.updates
        if additional_updates:
            updates = dict_union(updates, OrderedDict(additional_updates))
        return theano.function(self.inputs, self.outputs, updates=updates)

    def get_snapshot(self, data):
        """Evaluate all role-carrying Theano variables on given data.

        Parameters
        ----------
        data : dict of (data source, data) pairs
            Data for input variables. The sources should match with the
            names of the input variables.

        Returns
        -------
        Dictionary of (variable, variable value on given data) pairs.

        """
        role_variables = [var for var in self.variables
                          if hasattr(var.tag, "roles") and
                          not is_shared_variable(var)]
        value_holders = [shared_like(var) for var in role_variables]
        function = self.get_theano_function(zip(value_holders, role_variables))
        function(*(data[input_.name] for input_ in self.inputs))
        return OrderedDict([(var, value_holder.get_value(borrow=True))
                            for var, value_holder in zip(role_variables,
                                                         value_holders)])

    def has_inputs(self, variable):
        """Check if a variable depends on input variables.

        Returns
        -------
        bool
            ``True`` if the given variable depends on input variables,
            ``False`` otherwise.

        """
        if variable not in self._has_inputs:
            self._has_inputs[variable] = False
            if is_graph_input(variable):
                self._has_inputs[variable] = True
            elif getattr(variable, 'owner', None):
                for dependancy in variable.owner.inputs:
                    if self.has_inputs(dependancy):
                        self._has_inputs[variable] = True
        return self._has_inputs[variable]


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
            :const:`.COST`, :const:`.WEIGHTS`, etc.
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
        >>> from blocks.filter import VariableFilter
        >>> cg = ComputationGraph([y])
        >>> var_filter = VariableFilter(roles=[AUXILIARY])
        >>> var_filter(cg.variables) # doctest: +SKIP
        {x_minus_1, mean_W, mean_x}
        >>> var_filter = VariableFilter(roles=[COST])
        >>> var_filter(cg.variables) # doctest: +SKIP
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


def apply_noise(computation_graph, variables, level, seed=None):
    """Add Gaussian noise to certain variable of a computation graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph.
    variables : :class:`~tensor.TensorVariable`
        Variables to add noise to.
    level : float
        Noise level.
    seed : int, optional
        The seed with which
        :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams` is initialized,
        is set to 1 by default.

    """
    if not seed:
        seed = config.default_seed
    rng = MRG_RandomStreams(seed)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return computation_graph.replace(replace)
