"""This part of the framework helps you monitor your model during training.

:class:`MonitorChannels` describe quantities that should be monitored. They
can either be Theano variables (e.g. the objective function, the weight
norms, etc.) or they can be callable functions (e.g. if you want to use
NumPy to calculate eigenvalues, or sample text from your language model).

.. todo::

   Monitored Theano variables might depend on inputs beyond those given to
   the model. There should be a way of providing these to the monitoring
   channels, collecting them, and compiling them as part of the Theano
   function that performs monitoring.

"""
import logging
from abc import ABCMeta, abstractmethod

from blocks.utils import shared_expression, update_instance

logger = logging.getLogger(__name__)


class MonitorChannel(object):
    """A channel to monitor

    Parameters
    ----------
    name : str
        The name of this monitor channel, should be unique in order to
        distinguish channels.

    """
    def __init__(self, name):
        update_instance(self, locals())


class VariableMonitorChannel(MonitorChannel):
    """Monitors a Theano variable, optionally with an aggregation scheme.

    .. warning::

       The VariableMonitorChannel should be instantiated by the
       :meth:`AggregationScheme.get_channel` method, not directly.

    Example usages are:

    * computing the mean of some value over examples, sequence lengths etc.
    * tracking a parameter of a model
    * monitoring a regularization penalty

    The VariableMonitorChannel maintains a set of Theano shared values
    called accumulators and specifies how they shoud be initialized, and
    updated with incremental calculations. Finally, it provides a Theano
    expression that reads the accumulators and computes the final value.

    Parameters
    ----------
    aggregation_scheme : :class:`AggregationScheme`
        The aggregation scheme that constructed this VariableMonitorChannel
    initialization_updates : list of Theano updates
        Updates that specify how to initialize shared variables of this
        VariableMonitorChannel. *Should only use shared variables and
        constants in the update expression.*
    accumulation_updates : list of Theano updates
        Updates that specify how a new batch of data gets processed
        by this VariableMonitorChannel. *Can refer to model inputs.*
    readout_expression : list of Theano variables
        Theano variable that computes the final value based on accumulated
        partial results. *Expression should only consist of shared
        variables and constants.*

    Attributes
    ----------
    All constructor parameters are accessible as attributes.

    """
    def __init__(self, aggregation_scheme, initialization_updates=None,
                 accumulation_updates=None, readout_expression=None, **kwargs):
        super(VariableMonitorChannel, self).__init__(**kwargs)
        if initialization_updates is None:
            initialization_updates = []
        if accumulation_updates is None:
            accumulation_updates = []
        update_instance(self, locals())


class FunctionMonitorChannel(MonitorChannel):
    """A function whose output should be monitored

    Parameters
    ----------
    function : callable
        A callable function, which takes a model, training dataset, and set
        of monitoring datasets as arguments.

    Notes
    -----
    The values returned by callable monitor channels can be any Python
    object.

    """
    def __init__(self, function, **kwargs):
        super(FunctionMonitorChannel, self).__init__(**kwargs)
        assert callable(function)
        update_instance(self, locals())


class AggregationScheme(object):
    """Specify how to incrementally evaluate a Theano variable on data.

    An AggregationScheme allocates :class:`VariableMonitorChannel`s that
    can incrementally compute the value of a Theano variable on a full
    dataset by aggregating partial results computed on multiple batches.

    The AggregationScheme should be attached via the tag
    `aggregation_scheme` to a Theano variable which computes the desired
    value on a single batch.

    Parameters
    ----------
    expression: Theano variable
        expression that computes the desired value on a single batch.

    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_channel(self):
        """Return a new VariableMonitorChannel for this variable."""
        pass


class Mean(AggregationScheme):
    """Aggregation scheme which computes the mean.

    Parameters
    ----------
    numerator : Theano variable
        The expression for the numerator e.g. the likelihood of a
        minibatch.
    denominator : Theano variable
        The expression for the denominator e.g. the size of the batch.

    """
    def __init__(self, numerator, denominator):
        update_instance(self, locals())

    def get_channel(self):
        numerator_accumulated = shared_expression(self.numerator)
        denominator_accumulated = shared_expression(self.denominator)
        initialization_updates = [(numerator_accumulated, 0.0),
                                  (denominator_accumulated, 0.0)]
        accumulation_updates = [(numerator_accumulated,
                                 numerator_accumulated + self.numerator),
                                (denominator_accumulated,
                                 denominator_accumulated + self.denominator)]
        return VariableMonitorChannel(
            aggregation_scheme=self,
            initialization_updates=initialization_updates,
            accumulation_updates=accumulation_updates,
            readout_expression=numerator_accumulated / denominator_accumulated)


def mean(numerator, denominator, name=None):
    """Mean of quantity (numerator) over a number (denominator) values."""
    expression = numerator / denominator
    expression.tag.aggregation_scheme = Mean(numerator, denominator)
    if name is not None:
        expression.name = name
    else:
        expression.name = 'mean{{}, {}}'.format(numerator.name,
                                                denominator.name)
    return expression


class ModelProperty(AggregationScheme):
    """AggregationScheme for values that don't depend on data."""
    def __init__(self, expression):
        self.expression = expression

    def get_channel(self):
        return VariableMonitorChannel(aggregation_scheme=self,
                                      readout_expression=self.expression)


def model_property(expression, name=None):
    """Copy the given expression and tag with aggregation scheme."""
    expression = expression.copy()
    if name is not None:
        expression.name = name
    expression.tag.aggregation_scheme = ModelProperty(expression)
    return expression
