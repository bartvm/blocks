"""Annonated computation graph management."""

import logging

import theano
from theano import Variable
from theano.scalar import ScalarConstant
from theano.tensor import TensorConstant
from theano.tensor.sharedvar import SharedVariable
from theano.sandbox.rng_mrg import MRG_RandomStreams


logger = logging.getLogger(__name__)


class ComputationGraph(object):
    """Encapsulates a managed Theano computation graph.

    Attributes
    ----------
    inputs : list of Theano variables
        The inputs of the computation graph.
    outputs : list of Theano variables
        The outputs of the computations graph.

    Parameters
    ----------
    outputs : list of Theano variables
        The outputs of the computation graph.

    """
    def __init__(self, outputs):
        if isinstance(outputs, Variable):
            outputs = [outputs]
        self.outputs = outputs
        self._get_variables()

    def _get_variables(self):
        self.variables = set()
        self.applies = set()
        self.application_calls = set()
        self.updates = []

        def recursion(current):
            self.variables.add(current)

            if hasattr(current.tag, 'application_call'):
                logger.debug("found application call of {}".format(current))
                application_call = current.tag.application_call
                if application_call not in self.application_calls:
                    self.application_calls.add(application_call)
                    for av in application_call.auxiliary_variables:
                        av.tag.application_call = current.tag.application_call
                    self.variables.update(application_call.auxiliary_variables)
                    self.updates.extend(application_call.updates)
            if current.owner:
                owner = current.owner
                if owner not in self.applies:
                    if hasattr(owner.tag, 'updates'):
                        logger.debug("found updates in application of {}"
                                     .format(owner))
                        self.updates.extend(owner.tag.updates.items())
                    self.applies.add(owner)
                for input_ in owner.inputs:
                    if input_ not in self.variables:
                        recursion(input_)

        def is_input(variable):
            return (not variable.owner
                    and not isinstance(variable, SharedVariable)
                    and not isinstance(variable, TensorConstant)
                    and not isinstance(variable, ScalarConstant))

        for output in self.outputs:
            if output not in self.variables:
                recursion(output)
        self.inputs = [v for v in self.variables if is_input(v)]

    def dict_of_inputs(self):
        """Return a mapping from an input name to the input."""
        return {var.name: var for var in self.inputs}

    def replace(self, replacements):
        """Replace certain variables in the computation graph.

        Parameters
        ----------
        replacement : dict
            The mapping from variables to be replaced to the corresponding
            substitutes.

        """
        return ComputationGraph(theano.clone(self.outputs,
                                             replace=replacements))

    def function(self):
        """Create Theano function from the graph contained."""
        return theano.function(self.inputs, self.outputs,
                               updates=self.updates)


def apply_noise(graph, variables, level, rng=None):
    """Add Gaussian noise to certain variable of a computation graph.

    Parameters
    ----------
    graph : instance of :class:`ComputationGraph`
        The computation graph.
    varibles : Theano variables
        Variables to add noise to.
    level : float
        Noise level.
    rng : Theano random stream
        The random stream to use.

    """
    if not rng:
        rng = MRG_RandomStreams(1)
    replace = {}
    for variable in variables:
        replace[variable] = (variable +
                             rng.normal(variable.shape, std=level))
    return graph.replace(replace)
