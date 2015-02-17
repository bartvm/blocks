from collections import OrderedDict
import logging

import theano
from theano import tensor

from blocks.utils import dict_subset
from blocks.monitoring.aggregation import _DataIndependent, Mean, TakeLast
from blocks.graph import ComputationGraph
from blocks.utils import reraise_as

logger = logging.getLogger()


class AggregationBuffer(object):
    """Intermediate results of aggregating values of Theano variables.

    Encapsulates aggregators for a list of Theano variables. Collects
    the respective updates and provides initialization and readout
    routines.


    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable`
        The variable names are used as record names in the logs. Hence, all
        the variable names must be different.
    use_take_last : bool
        When ``True``, the :class:`TakeLast` aggregation scheme is used
        instead of :class:`_DataIndependent` for those variables that
        do not require data to be computed.

    Attributes
    ----------
    initialization_updates : list of tuples
        Initialization updates of the aggregators.
    accumulation_updates : list of tuples
        Accumulation updates of the aggregators.
    readout_variables : dict
        A dictionary of record names to :class:`~tensor.TensorVariable`
        representing the aggregated values.
    input : list of :class:`~tensor.TensorVariable`
        The list of inputs needed for accumulation.
    input_names : list of str
        The name of the inputs needed for accumulation.

    """
    def __init__(self, variables, use_take_last=False):
        self.variables = variables
        self.use_take_last = use_take_last

        self.variable_names = [v.name for v in self.variables]
        if len(self.variable_names) < len(self.variables):
            raise ValueError("variables should have different names")
        self._computation_graph = ComputationGraph(self.variables)
        self.inputs = self._computation_graph.inputs
        self.input_names = [v.name for v in self.inputs]

        self._initialized = False
        self._create_aggregators()
        self._compile()

    def _create_aggregators(self):
        """Create aggregators and collect updates."""
        self.initialization_updates = []
        self.accumulation_updates = []
        self.readout_variables = OrderedDict()

        for v in self.variables:
            logger.debug('variable to evaluate: %s', v.name)
            if not hasattr(v.tag, 'aggregation_scheme'):
                if not self._computation_graph.has_inputs(v):
                    scheme = (TakeLast if self.use_take_last
                              else _DataIndependent)
                    logger.debug('Using %s aggregation scheme'
                                 ' for %s since it does not depend on'
                                 ' the data', scheme.__name__, v.name)
                    v.tag.aggregation_scheme = scheme(v)
                else:
                    if v.ndim == 0:
                        logger.debug('Using the default '
                                     ' (average over minibatches)'
                                     ' aggregation scheme for %s', v.name)
                        v.tag.aggregation_scheme = Mean(v, 1.0)
                    else:
                        # TODO: support averaging for multi-dim variables
                        logger.debug('Multidimensional variable:'
                                     ' using the TakeLast'
                                     ' aggregation scheme for %s', v.name)
                        v.tag.aggregation_scheme = TakeLast(v)

            aggregator = v.tag.aggregation_scheme.get_aggregator()
            self.initialization_updates.extend(
                aggregator.initialization_updates)
            self.accumulation_updates.extend(aggregator.accumulation_updates)
            self.readout_variables[v.name] = aggregator.readout_variable

    def _compile(self):
        """Compiles Theano functions.

        .. todo::

            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.

        """
        logger.debug("Compiling initialization and readout functions")
        if self.initialization_updates:
            self._initialize_fun = theano.function(
                [], [], updates=self.initialization_updates)
        else:
            self._initialize_fun = None

        # We need to call `as_tensor_variable` here
        # to avoid returning `CudaNdarray`s to the user, which
        # happens otherwise under some circumstances (see
        # https://groups.google.com/forum/#!topic/theano-users/H3vkDN-Shok)
        self._readout_fun = theano.function(
            [], [tensor.as_tensor_variable(v)
                 for v in self.readout_variables.values()])
        logger.debug("Initialization and readout functions compiled")

    def initialize_aggregators(self):
        """Initialize the aggregators."""
        self._initialized = True
        if self._initialize_fun is not None:
            self._initialize_fun()

    def get_aggregated_values(self):
        """Readout the aggregated values."""
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        ret_vals = self._readout_fun()
        return dict(zip(self.variable_names, ret_vals))


class DatasetEvaluator(object):
    """A DatasetEvaluator evaluates many Theano variables on a dataset.

    The DatasetEvaluator provides a do-it-all method, :meth:`evaluate`,
    which computes values of ``variables`` on a dataset.

    Alternatively, methods :meth:`initialize_aggregators`,
    :meth:`process_batch`, :meth:`get_aggregated_values` can be used with a
    custom loop over data.

    The values computed on subsets of the given dataset are aggregated
    using the :class:`AggregationScheme`s provided in the
    `aggregation_scheme` tags. If no tag is given, the value is **averaged
    over minibatches**. However, care is taken to ensure that variables
    which do not depend on data are not unnecessarily recomputed.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable`
        The variable names are used as record names in the logs. Hence, all
        the names must be different.

        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.

    """
    def __init__(self, variables):
        self.buffer_ = AggregationBuffer(variables)
        self._compile()

    def _compile(self):
        """Compiles Theano functions.

        .. todo::

            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.

        """
        if self.buffer_.accumulation_updates:
            self._accumulate_fun = theano.function(
                self.buffer_.inputs, [],
                updates=self.buffer_.accumulation_updates)
        else:
            self._accumulate_fun = None

    def initialize_aggregators(self):
        self.buffer_.initialize_aggregators()

    def process_batch(self, batch):
        try:
            batch = dict_subset(batch, self.buffer_.input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(self.buffer_.input_names))
        if self._accumulate_fun is not None:
            self._accumulate_fun(**batch)

    def get_aggregated_values(self):
        return self.buffer_.get_aggregated_values()

    def evaluate(self, data_stream):
        """Compute the variables over a data stream.

        Parameters
        ----------
        data_stream : instance of :class:`.DataStream`
            The data stream. Only the first epoch of data is used.

        Returns
        -------
        A mapping from record names to the values computed on the provided
        dataset.

        """
        self.initialize_aggregators()

        if self._accumulate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                self.process_batch(batch)
        else:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        return self.get_aggregated_values()
