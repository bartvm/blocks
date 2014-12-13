"""Annotated computation graph management."""

import logging

import numpy
import theano
from theano import Variable
from theano.scalar import ScalarConstant
from theano.tensor import TensorConstant
from theano.tensor.sharedvar import SharedVariable
from theano.sandbox.rng_mrg import MRG_RandomStreams

from blocks.utils import shared_floatx

logger = logging.getLogger(__name__)


class ComputationGraph(object):
    """Encapsulates a managed Theano computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph.
    outputs : list of Theano variables
        The outputs of the computations graph.
    cost : Theano scalar
        The scalar variable representing the cost of the model to be
        minimized.

    Parameters
    ----------
    outputs : list of Theano variables
        The outputs of the computation graph, potentially with particular
        graph modifications applied.
    cost : Theano scalar
        The cost, potentially with graph modifications applied.
    outputs_and_cost : list of Theano variables
        The list of outputs with the cost appended.

    """
    def __init__(self, outputs, cost=None, seed=1):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self.cost = cost
        self.rng = MRG_RandomStreams(seed)
        self._get_variables()

    @property
    def outputs_and_cost(self):
        if self.cost is None:
            return self.outputs
        return self.outputs + [self.cost]

    @outputs_and_cost.setter
    def outputs_and_cost(self, value):
        if self.cost is None:
            self.outputs = value
        else:
            self.outputs = value[:-1]
            self.cost = value[-1]

    def _get_variables(self):
        def recursion(current):
            self.variables.add(current)
            if current.owner is not None:
                owner = current.owner
                if owner not in self.applies:
                    if hasattr(owner.tag, 'updates'):
                        logger.debug("updates of {}".format(owner))
                        self.updates.extend(owner.tag.updates.items())
                    self.applies.add(owner)

                for inp in owner.inputs:
                    if inp not in self.variables:
                        recursion(inp)

        def is_input(variable):
            return (variable.owner is None and
                    not isinstance(variable, SharedVariable) and
                    not isinstance(variable, TensorConstant) and
                    not isinstance(variable, ScalarConstant))

        self.variables = set()
        self.applies = set()
        self.updates = []
        for output in self.outputs_and_cost:
            recursion(output)
        self.inputs = [v for v in self.variables if is_input(v)]

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        .. todo::

            Either implement more efficiently, or make the whole
            ComputationGraph immutable and return a new one here.

        """
        self.outputs_and_cost = theano.clone(self.outputs_and_cost,
                                             replace=replacements)
        self._get_variables()

    def get_theano_function(self):
        """Create Theano function for the outputs (not the cost)."""
        return theano.function(self.inputs, self.outputs,
                               updates=self.updates)

    def collect_parameters(self, params):
        """Replace parameters with a single shared variable.

        This can be useful if you need to calculate the full Hessian of a
        computationl graph. It replaces parameters with slices of a single
        large vectors like

        >>> W1 = theano.shared(numpy.random.rand(10, 10))
        >>> W2 = theano.shared(numpy.random.rand(10, 10))
        >>> all_params = theano.shared(numpy.concatenate(
        ...     [W1.get_value().flatten(), W2.get_value().flatten()]))
        >>> W1 = all_params[:W1.size]
        >>> W2 = all_params[W1.size:]

        Parameters
        ----------
        params : list of Theano shared variables
            The parameters whose values should be collected

        Returns
        -------
        new_params : Theano shared variable
            A Theano shared variable (vector) which contains the values of
            all the parameters that have been replaced.

        Notes
        -----
        Note that this replacement makes the training of the model
        significantly slower because of the large amount of Theano's
        ``set_subtensor`` calls needed to train the model.

        """
        param_values, param_sizes, param_shapes = [], [], []
        for param in params:
            param_values.append(param.get_value())
            param_sizes.append(param_values[-1].size)
            param_shapes.append(param_values[-1].shape)

        new_params = shared_floatx(numpy.concatenate(
            [param_value.flatten() for param_value in param_values]))

        replacements = {}
        for param, shape, i, j in zip(params, param_shapes,
                                      numpy.cumsum([0] + param_sizes[:-1]),
                                      numpy.cumsum(param_sizes)):
            replacements[param] = new_params[i:j].reshape(shape)
        self.replace(replacements)
        return new_params


def apply_noise(cg, variables, level):
    """Add Gaussian noise to certain variables in the graph.

    Parameters
    ----------
    variables : Theano variables
        Variables to add noise to.
    level : float
        Noise level.

    """
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             cg.rng.normal(variable.shape, std=level))
    cg.replace(replace)
