"""Extensions for monitoring the training process."""
import logging

import theano

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.algorithms import UpdatesAlgorithm
from blocks.monitoring.aggregation import MonitoredQuantity, take_last
from blocks.monitoring.evaluators import (
    AggregationBuffer, MonitoredQuantityBuffer, DatasetEvaluator)

SEPARATOR = '_'
logger = logging.getLogger(__name__)


class MonitoringExtension(TrainingExtension):
    """A mixin with logic shared by monitoring extensions.

    Parameters
    ----------
    prefix : str, optional
        The prefix for the log records done by the extension.  It is
        prepended to the variable names with an underscore as a separator.
        If not given, no prefix is added to the names of the observed
        variables.
    suffix : str, optional
        The suffix for the log records done by the extension.  It is
        appended to the end of variable names with an underscore as a
        separator. If not given, no suffix is added the names of the
        observed variables.

    """
    SEPARATOR = SEPARATOR

    def __init__(self, prefix=None, suffix=None, **kwargs):
        super(MonitoringExtension, self).__init__(**kwargs)
        self.prefix = prefix
        self.suffix = suffix

    def _record_name(self, name):
        """The record name for a variable name."""
        if not isinstance(name, str):
            raise ValueError("record name must be a string")

        return self.SEPARATOR.join(
            [morpheme for morpheme in [self.prefix, name, self.suffix]
             if morpheme is not None])

    def record_name(self, variable):
        """The record name for a variable."""
        return self._record_name(variable.name)

    def add_records(self, log, record_tuples):
        """Helper function to add monitoring records to the log."""
        for name, value in record_tuples:
            if not name:
                raise ValueError("monitor variable without name")
            log.current_row[self._record_name(name)] = value


class DataStreamMonitoring(SimpleExtension, MonitoringExtension):
    """Monitors Theano variables and monitored-quantities on a data stream.

    By default monitoring is done before the first and after every epoch.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` and
        :class:`MonitoredQuantity`
        The variables to monitor. The variable names are used as record
        names in the logs.
    updates : list of tuples or :class:`~collections.OrderedDict` or None
        :class:`~tensor.TensorSharedVariable` updates to be performed
        during evaluation. This parameter is only for Theano variables.
        Be careful not to update any model parameters as this is not
        intended to alter your model in any meaningful way. A typical
        use case of this option arises when the theano function used
        for evaluation contains a call to :func:`~theano.scan` which
        might have returned shared variable updates.
    data_stream : instance of :class:`.DataStream`
        The data stream to monitor on. A data epoch is requested
        each time monitoring is done.

    """
    def __init__(self, variables, data_stream, updates=None, **kwargs):
        kwargs.setdefault("after_epoch", True)
        kwargs.setdefault("before_first_epoch", True)
        super(DataStreamMonitoring, self).__init__(**kwargs)
        self._evaluator = DatasetEvaluator(variables, updates)
        self.data_stream = data_stream

    def do(self, callback_name, *args):
        """Write the values of monitored variables to the log."""
        logger.info("Monitoring on auxiliary data started")
        value_dict = self._evaluator.evaluate(self.data_stream)
        self.add_records(self.main_loop.log, value_dict.items())
        logger.info("Monitoring on auxiliary data finished")


class TrainingDataMonitoring(SimpleExtension, MonitoringExtension):
    """Monitors values of Theano variables on training batches.

    Use this extension to monitor a quantity on every training batch
    cheaply. It integrates with the training algorithm in order to avoid
    recomputing same things several times. For instance, if you are
    training a network and you want to log the norm of the gradient on
    every batch, the backpropagation will only be done once.  By
    controlling the frequency with which the :meth:`do` method is called,
    you can aggregate the monitored variables, e.g. only log the gradient
    norm average over an epoch.

    Parameters
    ----------
    variables : list of :class:`~tensor.TensorVariable` or
                  :class:`~blocks.monitoring.aggregation.MonitoredQuantity`
        The variables or non-Theano quantities to monitor.
        The variable names are used as record names in the logs.

    Notes
    -----
    All the monitored variables are evaluated _before_ the parameter
    update.

    Requires the training algorithm to be an instance of
    :class:`.UpdatesAlgorithm`.

    """
    def __init__(self, variables, **kwargs):
        kwargs.setdefault("before_training", True)
        super(TrainingDataMonitoring, self).__init__(**kwargs)
        self.add_condition(['after_batch'], arguments=('just_aggregate',))

        self._non_variables = []
        self._variables = []
        for variable_or_not in variables:
            if isinstance(variable_or_not, theano.Variable):
                self._variables.append(variable_or_not)
            elif isinstance(variable_or_not, MonitoredQuantity):
                self._non_variables.append(variable_or_not)
            else:
                raise ValueError("can not monitor {}".format(variable_or_not))

        self._non_variables = MonitoredQuantityBuffer(self._non_variables)
        self._required_for_non_variables = AggregationBuffer(
            [take_last(v) for v in self._non_variables.requires])
        self._variables = AggregationBuffer(
            self._variables, use_take_last=True)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        """Initializes the buffer or commits the values to the log.

        What this method does depends on from what callback it is called
        and with which arguments.  When called within `before_training`, it
        initializes the aggregation buffer and instructs the training
        algorithm what additional computations should be carried at each
        step by adding corresponding updates to it. In most_other cases it
        writes aggregated values of the monitored variables to the log. An
        exception is when an argument `just_aggregate` is given: in this
        cases it updates the values of monitored non-Theano quantities, but
        does not write anything to the log.

        """
        data, args = self.parse_args(callback_name, args)
        if callback_name == 'before_training':
            if not isinstance(self.main_loop.algorithm,
                              UpdatesAlgorithm):
                raise ValueError
            self.main_loop.algorithm.add_updates(
                self._variables.accumulation_updates)
            self.main_loop.algorithm.add_updates(
                self._required_for_non_variables.accumulation_updates)
            self._variables.initialize_aggregators()
            self._required_for_non_variables.initialize_aggregators()
            self._non_variables.initialize_quantities()
        else:
            # When called first time at any iterations, update
            # monitored non-Theano quantities
            if (self.main_loop.status['iterations_done'] >
                    self._last_time_called):
                self._non_variables.aggregate_quantities(
                    list(self._required_for_non_variables
                         .get_aggregated_values().values()))
                self._required_for_non_variables.initialize_aggregators()
                self._last_time_called = (
                    self.main_loop.status['iterations_done'])
            # If only called to update non-Theano quantities,
            # do just that
            if args == ('just_aggregate',):
                return
            # Otherwise, also output current values of from the accumulators
            # to the log.
            self.add_records(
                self.main_loop.log,
                self._variables.get_aggregated_values().items())
            self._variables.initialize_aggregators()
            self.add_records(
                self.main_loop.log,
                self._non_variables.get_aggregated_values().items())
            self._non_variables.initialize_quantities()
