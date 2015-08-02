"""Extensions for monitoring the training process."""
import logging

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.algorithms import DifferentiableCostMinimizer
from blocks.monitoring.evaluators import AggregationBuffer, DatasetEvaluator

PREFIX_SEPARATOR = '_'
logger = logging.getLogger(__name__)


class MonitoringExtension(TrainingExtension):
    """A mixin with logic shared by monitoring extensions.

    Parameters
    ----------
    prefix : str, optional
        The prefix for the log records done by the extension.  It is
        appended to the variable names with an underscore as a separator.
        If not given, the names of the observed variables are used as is.

    """
    def __init__(self, prefix=None, **kwargs):
        super(MonitoringExtension, self).__init__(**kwargs)
        self.prefix = prefix

    def _record_name(self, name):
        """The record name for a variable name."""
        return self.prefix + PREFIX_SEPARATOR + name if self.prefix else name

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
    PREFIX_SEPARATOR = '_'

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
    variables : list of :class:`~tensor.TensorVariable`
        The variables to monitor. The variable names are used as record
        names in the logs.

    Notes
    -----
    All the monitored variables are evaluated _before_ the parameter
    update.

    Requires the training algorithm to be an instance of
    :class:`.DifferentiableCostMinimizer`.

    """
    def __init__(self, variables, **kwargs):
        kwargs.setdefault("before_training", True)
        super(TrainingDataMonitoring, self).__init__(**kwargs)
        self._buffer = AggregationBuffer(variables, use_take_last=True)
        self._last_time_called = -1

    def do(self, callback_name, *args):
        """Initializes the buffer or commits the values to the log.

        What this method does depends on from what callback it is called.
        When called within `before_training`, it initializes the
        aggregation buffer and instructs the training algorithm what
        additional computations should be carried at each step by adding
        corresponding updates to it. In all other cases it writes
        aggregated values of the monitored variables to the log.

        """
        if callback_name == 'before_training':
            if not isinstance(self.main_loop.algorithm,
                              DifferentiableCostMinimizer):
                raise ValueError
            self.main_loop.algorithm.add_updates(
                self._buffer.accumulation_updates)
            self._buffer.initialize_aggregators()
        else:
            if (self.main_loop.status['iterations_done'] ==
                    self._last_time_called):
                raise Exception("TrainingDataMonitoring.do should be invoked"
                                " no more than once per iteration")
            self._last_time_called = self.main_loop.status['iterations_done']
            self.add_records(self.main_loop.log,
                             self._buffer.get_aggregated_values().items())
            self._buffer.initialize_aggregators()
