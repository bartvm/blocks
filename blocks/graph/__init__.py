"""Annotated computation graph management."""
import logging
from collections import OrderedDict
from itertools import chain
import warnings

import numpy
import theano
from picklable_itertools.extras import equizip
from theano import Variable
from theano.gof import graph
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.scan_module.scan_op import Scan
from toolz import unique

from ..config import config
from ..roles import (add_role, has_roles, AUXILIARY, PARAMETER, DROPOUT,
                     COLLECTED, COLLECTOR)
from ..utils import (is_graph_input, is_shared_variable, dict_union,
                     shared_floatx_zeros, shared_like)
from .annotations import add_annotation, Annotation  # noqa
from .bn import batch_normalization, apply_batch_normalization  # noqa
from .bn import get_batch_normalization_updates  # noqa

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
        self.outputs = list(outputs)
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
                                     if isinstance(node.op, Scan)],
                                     key=lambda op: id(op)))
            self._scan_graphs = [ComputationGraph(scan.outputs)
                                 for scan in self.scans]

            seen = set()
            main_vars = (
                [var for var in list(chain(
                    *[apply_node.inputs for apply_node in sorted_apply_nodes]))
                 if not (var in seen or seen.add(var))] +
                [var for var in self.outputs if var not in seen])

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

        Examples
        --------
        >>> import theano
        >>> from theano import tensor, function
        >>> x = tensor.scalar('x')
        >>> y = x + 2
        >>> z = y + 3
        >>> a = z + 5

        Let's suppose we have dependent replacements like

        >>> replacements = {y: x * 2, z: y * 3}
        >>> cg = ComputationGraph([a])
        >>> theano.pprint(a)  # doctest: +NORMALIZE_WHITESPACE
        '(((x + TensorConstant{2}) + TensorConstant{3}) +
        TensorConstant{5})'
        >>> cg_new = cg.replace(replacements)
        >>> theano.pprint(
        ...     cg_new.outputs[0])  # doctest: +NORMALIZE_WHITESPACE
        '(((x * TensorConstant{2}) * TensorConstant{3}) +
        TensorConstant{5})'

        First two sums turned into multiplications

        >>> float(function(cg_new.inputs, cg_new.outputs)(3.)[0])
        23.0

        """
        # Due to theano specifics we have to make one replacement in time
        replacements = OrderedDict(replacements)

        outputs_cur = self.outputs

        # `replacements` with previous replacements applied. We have to track
        # variables in the new graph corresponding to original replacements.
        replacement_keys_cur = []
        replacement_vals_cur = []
        # Sort `replacements` in topological order
        # variables in self.variables are in topological order
        remaining_replacements = replacements.copy()
        for variable in self.variables:
            if variable in replacements:
                if has_roles(variable, [AUXILIARY]):
                    warnings.warn(
                        "replace method was asked to replace a variable ({}) "
                        "that is an auxiliary variable.".format(variable))
                replacement_keys_cur.append(variable)
                # self.variables should not contain duplicates,
                # otherwise pop() may fail.
                replacement_vals_cur.append(
                    remaining_replacements.pop(variable))

        # if remaining_replacements is not empty
        if remaining_replacements:
            warnings.warn(
                "replace method was asked to replace a variable(s) ({}) "
                "that is not a part of the computational "
                "graph.".format(str(remaining_replacements.keys())))

        # Replace step-by-step in topological order
        while replacement_keys_cur:
            replace_what = replacement_keys_cur[0]
            replace_by = replacement_vals_cur[0]
            # We also want to make changes in future replacements
            outputs_new = theano.clone(
                outputs_cur + replacement_keys_cur[1:] +
                replacement_vals_cur[1:],
                replace={replace_what: replace_by})
            # Reconstruct outputs, keys, and values
            outputs_cur = outputs_new[:len(outputs_cur)]
            replacement_keys_cur = outputs_new[len(outputs_cur):
                                               len(outputs_cur) +
                                               len(replacement_keys_cur) - 1]
            replacement_vals_cur = outputs_new[len(outputs_cur) +
                                               len(replacement_keys_cur):]

        return ComputationGraph(outputs_cur)

    def get_theano_function(self, additional_updates=None, **kwargs):
        r"""Create Theano function from the graph contained.

        Parameters
        ----------
        \*\*kwargs : dict
            Keyword arguments to theano.function.
            Useful for specifying compilation modes or profiling.

        """
        updates = self.updates
        if additional_updates:
            updates = dict_union(updates, OrderedDict(additional_updates))
        return theano.function(self.inputs, self.outputs, updates=updates,
                               **kwargs)

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
        function = self.get_theano_function(equizip(value_holders,
                                                    role_variables))
        function(*(data[input_.name] for input_ in self.inputs))
        return OrderedDict([(var, value_holder.get_value(borrow=True))
                            for var, value_holder in equizip(role_variables,
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


def collect_parameters(computation_graph, parameters):
    """Replace parameters with a single shared variable.

    This can be useful if you need to calculate the full Hessian of a
    computational graph. It replaces parameters with slices of a single
    large vectors like

    >>> from blocks.utils import shared_floatx
    >>> W1 = shared_floatx(numpy.random.rand(10, 10))
    >>> W2 = shared_floatx(numpy.random.rand(10, 10))
    >>> all_parameters = shared_floatx(numpy.concatenate(
    ...     [W1.get_value().flatten(), W2.get_value().flatten()]))
    >>> W1 = all_parameters[:W1.size]
    >>> W2 = all_parameters[W1.size:]

    Parameters
    ----------
    computation_graph : :class:`ComputationGraph` instance
        The managed Theano graph in which to collect parameters.
    parameters : list of Theano shared variables
        The parameters whose values should be collected.

    Returns
    -------
    ComputationGraph instance
        A new Theano graph which has all the given parameters collected
        into a single large shared variable.

    Notes
    -----
    Note that this replacement makes the training of the model
    significantly slower because of the large amount of Theano's
    ``set_subtensor`` calls needed to train the model.

    Examples
    --------
    >>> from blocks.bricks import MLP, Logistic
    >>> from blocks.bricks.cost import SquaredError
    >>> from theano import tensor
    >>> x = tensor.matrix()
    >>> mlp = MLP(activations=[Logistic(), Logistic()],
    ...           dims=[784, 100, 784])
    >>> cost = SquaredError().apply(x, mlp.apply(x))
    >>> cg = ComputationGraph(cost)
    >>> new_cg = collect_parameters(cg, cg.shared_variables)

    The new graph only has a single shared variable. This variable receives
    the :const:`COLLECTOR` role.

    >>> new_cg.shared_variables
    [collected_parameters]

    The bricks' variables have been replaced with reshaped segments of this
    single shared variable. These replacements are given the
    :const:`.COLLECTED` role.

    >>> from blocks.filter import VariableFilter
    >>> from blocks.roles import PARAMETER
    >>> var_filter = VariableFilter(roles=[COLLECTED])
    >>> var_filter(new_cg.variables)  # doctest: +SKIP
    [Reshape{1}.0, Reshape{1}.0, Reshape{2}.0, Reshape{2}.0]

    """
    parameter_values, parameter_sizes, parameter_shapes = [], [], []
    for parameter in parameters:
        parameter_values.append(parameter.get_value(borrow=True))
        parameter_sizes.append(parameter_values[-1].size)
        parameter_shapes.append(parameter_values[-1].shape)

    new_parameters = shared_floatx_zeros(sum(parameter_sizes))
    new_parameters.set_value(numpy.concatenate([value.flatten()
                             for value in parameter_values]))
    new_parameters.name = 'collected_parameters'
    add_role(new_parameters, COLLECTOR)

    replacements = {}
    for parameter, shape, i, j in zip(parameters, parameter_shapes,
                                      numpy.cumsum([0] + parameter_sizes[:-1]),
                                      numpy.cumsum(parameter_sizes)):
        new_parameter = new_parameters[i:j].reshape(shape)
        new_parameter.replacement_of = parameter
        add_role(new_parameter, COLLECTED)
        replacements[parameter] = new_parameter
    return computation_graph.replace(replacements)


def apply_dropout(computation_graph, variables, drop_prob, rng=None,
                  seed=None, custom_divisor=None):
    """Apply dropout to specified variables in a graph.

    Parameters
    ----------
    computation_graph : instance of :class:`ComputationGraph`
        The computation graph.
    variables : list of :class:`~tensor.TensorVariable`
        Variables to be dropped out.
    drop_prob : float
        Probability of dropping out. If you want to apply the dropout
        with different probabilities for different layers, call it
        several times.
    rng : :class:`~theano.sandbox.rng_mrg.MRG_RandomStreams`
        Random number generator.
    seed : int
        Random seed to be used if `rng` was not specified.
    custom_divisor : float or None, optional
        Divide dropped variables by a given scalar value. If `None`,
        (default) dropped variables will be divided by `(1 - drop_prob)`
        which is equivalent to scaling by `(1 - drop_prob)` at test
        time as recommended in [DROPOUT]_.

    Returns
    -------
    dropped_computation_graph : instance of :class:`ComputationGraph`
        A new computation graph with dropout applied to the specified
        variables. In order to train with, or monitor, the outputs
        of the original computation graph with dropout applies, use
        the variables contained in `dropped_computation_graph.outputs`.

    Notes
    -----
    For more information, see [DROPOUT]_.

    .. [DROPOUT] Hinton et al. *Improving neural networks by preventing
       co-adaptation of feature detectors*, arXiv:1207.0580.

    Examples
    --------
    >>> import numpy
    >>> from theano import tensor, function
    >>> from blocks.bricks import MLP, Identity
    >>> from blocks.filter import VariableFilter
    >>> from blocks.initialization import Constant
    >>> from blocks.roles import INPUT
    >>> linear = MLP([Identity(), Identity()], [2, 10, 2],
    ...              weights_init=Constant(1), biases_init=Constant(2))
    >>> x = tensor.matrix('x')
    >>> y = linear.apply(x)
    >>> cg = ComputationGraph(y)

    We are going to drop out all the input variables

    >>> inputs = VariableFilter(roles=[INPUT])(cg.variables)

    Here we apply dropout with default setting to our computation graph

    >>> cg_dropout = apply_dropout(cg, inputs, 0.5)

    Dropped out variables have role `DROPOUT` and are tagged with
    `replacement_of` tag. Let's filter these variables and check if they
    have the links to original ones.

    >>> dropped_out = VariableFilter(roles=[DROPOUT])(cg_dropout.variables)
    >>> inputs_referenced = [var.tag.replacement_of for var in dropped_out]
    >>> set(inputs) == set(inputs_referenced)
    True

    Compiling theano functions to forward propagate in original and dropped
    out graphs

    >>> fprop = function(cg.inputs, cg.outputs[0])
    >>> fprop_dropout = function(cg_dropout.inputs, cg_dropout.outputs[0])

    Initialize an MLP and apply these functions

    >>> linear.initialize()
    >>> fprop(numpy.ones((3, 2),
    ...       dtype=theano.config.floatX))  # doctest:+ELLIPSIS
    array([[ 42.,  42.],
           [ 42.,  42.],
           [ 42.,  42.]]...
    >>> fprop_dropout(numpy.ones((3, 2),
    ...               dtype=theano.config.floatX))  # doctest:+ELLIPSIS
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0.,  0.]]...

    And after the second run answer is different

    >>> fprop_dropout(numpy.ones((3, 2),
    ...               dtype=theano.config.floatX))  # doctest:+ELLIPSIS
    array([[   0.,   52.],
           [ 100.,    0.],
           [   0.,    0.]]...

    """
    if not rng and not seed:
        seed = config.default_seed
    if not rng:
        rng = MRG_RandomStreams(seed)
    if custom_divisor is None:
        divisor = (1 - drop_prob)
    else:
        divisor = custom_divisor
    replacements = [(var, var *
                     rng.binomial(var.shape, p=1 - drop_prob,
                                  dtype=theano.config.floatX) /
                     divisor)
                    for var in variables]
    for variable, replacement in replacements:
        add_role(replacement, DROPOUT)
        replacement.tag.replacement_of = variable

    return computation_graph.replace(replacements)
