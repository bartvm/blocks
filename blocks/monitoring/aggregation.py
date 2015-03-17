"""Evaluate Theano variables on auxiliary data and during training."""
import logging
from abc import ABCMeta, abstractmethod

from six import add_metaclass
from theano import tensor

from blocks.utils import shared_like

logger = logging.getLogger(__name__)


@add_metaclass(ABCMeta)
class AggregationScheme(object):
    """How to incrementally evaluate a Theano variable over minibatches.

    An AggregationScheme allocates :class:`Aggregator`s that can
    incrementally compute the value of a Theano variable on a full dataset
    by aggregating partial results computed on multiple batches.

    The AggregationScheme should be attached via the tag
    ``aggregation_scheme`` to a Theano variable which computes the desired
    value on a single batch.

    Parameters
    ----------
    variable: :class:`~tensor.TensorVariable`
        The variable that holds the desired value on a single batch.

    """
    @abstractmethod
    def get_aggregator(self):
        """Return a new Aggregator for this variable."""
        pass


class Aggregator(object):
    """An Aggregator incrementally evaluates a Theano variable on a dataset.

    .. warning::
        The Aggregators should never be created directly. Instead use the
        :meth:`AggregationScheme.get_aggregator` method.

    Example usages are:

    * compute the mean of some value over examples, sequence lengths etc.
    * track a parameter of a model
    * monitor a penalty

    The Aggregator maintains a set of Theano sharer values called
    accumulators and specifies how they should be initialized, and
    updated with incremental calculations. Finally, it
    provides a Theano variable that reads the accumulators
    and computes the final value.

    Parameters
    ----------
    aggregation_scheme : :class:`AggregationScheme`
        The aggregation scheme that constructed this Aggregator
    initialization_updates : list of Theano updates
        Updates that specify how to initialize shared variables of
        this Aggregator. *Can only refer to shared variables and
        constants.*
    accumulation_updates : list of Theano updates
        Updates that specify how a new batch of data gets processed
        by this Aggregator. *Can refer to model inputs.*
    readout_variable : :class:`~tensor.TensorVariable`
        Theano variable that holds the final value based on accumulated
        partial results. *readout_variable must only consist of shared
        variables and constants.*

    Attributes
    ----------
    All constructor parameters are accessible as attributes.

    """
    def __init__(self, aggregation_scheme, initialization_updates=None,
                 accumulation_updates=None, readout_variable=None):
        self.aggregation_scheme = aggregation_scheme
        self.readout_variable = readout_variable

        if initialization_updates is None:
            initialization_updates = []
        if accumulation_updates is None:
            accumulation_updates = []
        self.initialization_updates = initialization_updates
        self.accumulation_updates = accumulation_updates


class Mean(AggregationScheme):
    """Aggregation scheme which computes the mean.

    Parameters
    ----------
    numerator : :class:`~tensor.TensorVariable`
        Theano variable for the numerator e.g. the likelihood
    denominator : :class:`~tensor.TensorVariable`
        Theano variable for the denominator e.g. the batch size

    """
    def __init__(self, numerator, denominator):
        self.numerator = numerator
        self.denominator = denominator

    def get_aggregator(self):
        numerator_acc = shared_like(self.numerator)
        denominator_acc = shared_like(self.denominator)
        initialization_updates = [(numerator_acc, 0.0),
                                  (denominator_acc, 0.0)]
        accumulation_updates = [(numerator_acc,
                                 numerator_acc + self.numerator),
                                (denominator_acc,
                                 denominator_acc + self.denominator)]
        aggregator = Aggregator(aggregation_scheme=self,
                                initialization_updates=initialization_updates,
                                accumulation_updates=accumulation_updates,
                                readout_variable=(numerator_acc /
                                                  denominator_acc))
        return aggregator


def mean(numerator, denominator=1.0):
    """Mean of quantity (numerator) over a number (denominator) values."""
    variable = numerator / denominator
    variable.tag.aggregation_scheme = Mean(numerator, denominator)
    variable.name = numerator.name
    return variable


class _DataIndependent(AggregationScheme):
    """Dummy aggregation scheme for values that don't depend on data."""
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[],
                          accumulation_updates=[],
                          readout_variable=self.variable)


class TakeLast(AggregationScheme):
    """Aggregation scheme which remembers only the last value."""
    def __init__(self, variable):
        self.variable = variable

    def get_aggregator(self):
        self.storage = shared_like(self.variable)
        return Aggregator(aggregation_scheme=self,
                          initialization_updates=[
                              (self.storage, tensor.zeros_like(self.storage))],
                          accumulation_updates=[(self.storage, self.variable)],
                          readout_variable=self.storage)


@add_metaclass(ABCMeta)
class MonitoredQuantity(object):
    """The base class for monitored-quantities.

    To monitor a non-Theano quanity in Blocks you have to implement this
    interface for it. The initialize method initializes accumulators and
    the parameters needed to compute this quantity, accumulate method
    accumulates results for every batch, and finallly readout is called
    to get the accumulated resutls.

    See Also
    ------------
       :class:`~blocks.monitoring.evaluators.DatasetEvaluator`,
       :class:`~blocks.extensions.DataStreamMonitoring`

    Attributes
    ----------
    requires : list
        List of Theano variables needed to calculate this quantity.
    name : str
        The name of monitored quantity which appears in the log.

    """
    def __init__(self, requires=None, name=None):
        self.requires = requires
        self.name = name

    @abstractmethod
    def initialize(self):
        """Initialize accumulators for this monitored quantity."""
        pass

    @abstractmethod
    def accumulate(self):
        """Accumulate results for every batch."""
        pass

    @abstractmethod
    def readout(self):
        """Readout the accumulated results to capture the final result."""
        pass
