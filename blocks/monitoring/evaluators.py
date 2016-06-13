from collections import OrderedDict, Counter
import logging

from picklable_itertools.extras import equizip
import theano
from theano import tensor

from blocks.utils import dict_subset
from blocks.monitoring.aggregation import (_DataIndependent, Mean,
                                           TakeLast, MonitoredQuantity)
from blocks.graph import ComputationGraph
from blocks.utils import reraise_as

logger = logging.getLogger(__name__)


def _validate_variable_names(variables):
    """Check for missing and duplicate variable names."""
    variable_names = [v.name for v in variables]
    name_counts = Counter(variable_names)
    if None in name_counts:
        none_names = [v for v in variables if v.name is None]
        raise ValueError('Variables must have names: {}'.format(none_names))

    if any(v > 1 for v in name_counts.values()):
        raise ValueError("Variables should have unique names."
                         " Duplicates: {}"
                         .format(', '.join(k for k, v in name_counts.items()
                                           if v > 1)))


class MonitoredQuantityBuffer(object):
    """Intermediate results of aggregating values of monitored-quantity.

    Aggregate results for a list of monitored-quantity for every
    single batch. Provides initialization and readout routines to
    initialize each quantity and capture its aggregated results.


    Parameters
    ----------
    quantities : list of :class:`MonitoredQuantity`
        The quantity names are used as record names in the logs. Hence, all
        the quantity names must be unique.

    Attributes
    ----------
    requires : list of :class:`~tensor.TensorVariable`
        Needed to calculate monitored-quantities.
    quantity_names : list of str
        Names of quantities.
    inputs : list of :class:`~tensor.TensorVariable`
        The list of inputs needed for variables in `requires`.

    """
    def __init__(self, quantities):
        self.quantities = quantities
        requires = []
        for quantity in quantities:
            requires += quantity.requires
        self.requires = list(set(requires))
        self._initialized = False

        self.quantity_names = [q.name for q in self.quantities]
        self._computation_graph = ComputationGraph(self.requires)
        self.inputs = self._computation_graph.inputs

    def initialize_quantities(self):
        """Initialize the quantities."""
        self._initialized = True
        for quantity in self.quantities:
            quantity.initialize()

    def get_aggregated_values(self):
        """Get the aggregated values."""
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        else:
            ret_vals = [q.get_aggregated_value() for q in self.quantities]
            return dict(zip(self.quantity_names, ret_vals))

    def aggregate_quantities(self, numerical_values):
        """Aggregate the results for every batch."""
        if not self._initialized:
            raise Exception("To readout you must first initialize, then"
                            "process batches!")
        else:
            for quantity in self.quantities:
                quantity.aggregate(
                    *[numerical_values[self.requires.index(requirement)]
                        for requirement in quantity.requires])


class AggregationBuffer(object):
    """Intermediate results of aggregating values of Theano variables.

    Encapsulates aggregators for a list of Theano variables. Collects
    the respective updates and provides initialization and readout
    routines.


    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable`
        The variable names are used as record names in the logs. Hence, all
        the variable names must be unique.
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
    inputs : list of :class:`~tensor.TensorVariable`
        The list of inputs needed for accumulation.

    """
    def __init__(self, variables, use_take_last=False):
        _validate_variable_names(variables)
        self.variables = variables
        self.variable_names = [v.name for v in self.variables]
        self.use_take_last = use_take_last
        self._computation_graph = ComputationGraph(self.variables)
        self.inputs = self._computation_graph.inputs

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
                    logger.debug('Using the default '
                                 ' (average over minibatches)'
                                 ' aggregation scheme for %s', v.name)
                    v.tag.aggregation_scheme = Mean(v, 1.0)

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
            raise Exception("To readout you must first initialize, then "
                            "process batches!")
        ret_vals = self._readout_fun()
        return OrderedDict(equizip(self.variable_names, ret_vals))


class DatasetEvaluator(object):
    """A DatasetEvaluator evaluates many Theano variables or other quantities.

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
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variable names are used as record names in the logs. Hence, all
        the names must be unique.

        Each variable can be tagged with an :class:`AggregationScheme` that
        specifies how the value can be computed for a data set by
        aggregating minibatches.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningfullway. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to:function:`~theano.scan` which
        might have returned shared variable updates.

    """
    def __init__(self, variables, updates=None):
        _validate_variable_names(variables)
        theano_variables = []
        monitored_quantities = []
        for variable in variables:
            if isinstance(variable, MonitoredQuantity):
                monitored_quantities.append(variable)
            else:
                theano_variables.append(variable)
        self.theano_variables = theano_variables
        self.monitored_quantities = monitored_quantities
        self.theano_buffer = AggregationBuffer(theano_variables)
        self.monitored_quantities_buffer = MonitoredQuantityBuffer(
            monitored_quantities)
        self.updates = updates
        self._compile()

    def _compile(self):
        """Compiles Theano functions.

        .. todo::

            The current compilation method does not account for updates
            attached to `ComputationGraph` elements. Compiling should
            be out-sourced to `ComputationGraph` to deal with it.

        """
        inputs = []
        outputs = []
        updates = None
        if self.theano_buffer.accumulation_updates:
            updates = OrderedDict()
            updates.update(self.theano_buffer.accumulation_updates)
            inputs += self.theano_buffer.inputs
        if self.updates:
            # Handle the case in which we dont have any theano variables
            # to evaluate but we do have MonitoredQuantity
            # that may require an update of their own
            if updates is None:
                updates = self.updates
            else:
                updates.update(self.updates)
        inputs += self.monitored_quantities_buffer.inputs
        outputs = self.monitored_quantities_buffer.requires

        if inputs != []:
            self.unique_inputs = list(set(inputs))
            self._aggregate_fun = theano.function(self.unique_inputs,
                                                  outputs,
                                                  updates=updates)
        else:
            self._aggregate_fun = None

    def initialize_aggregators(self):
        self.theano_buffer.initialize_aggregators()
        self.monitored_quantities_buffer.initialize_quantities()

    def process_batch(self, batch):
        try:
            input_names = [v.name for v in self.unique_inputs]
            batch = dict_subset(batch, input_names)
        except KeyError:
            reraise_as(
                "Not all data sources required for monitoring were"
                " provided. The list of required data sources:"
                " {}.".format(input_names))
        if self._aggregate_fun is not None:
            numerical_values = self._aggregate_fun(**batch)
            self.monitored_quantities_buffer.aggregate_quantities(
                numerical_values)

    def get_aggregated_values(self):
        values = self.theano_buffer.get_aggregated_values()
        values.update(
            self.monitored_quantities_buffer.get_aggregated_values())
        return values

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
        if self._aggregate_fun is not None:
            for batch in data_stream.get_epoch_iterator(as_dict=True):
                self.process_batch(batch)
        else:
            logger.debug(
                'Only data independent variables were given,'
                'will not iterate the over data!')

        return self.get_aggregated_values()
