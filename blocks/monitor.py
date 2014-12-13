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
from collections import defaultdict

import numpy
import theano
from theano import Variable

from blocks.utils import pack, update_instance

# TODO Set up logging properly for the entire framework
logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.INFO)


class MonitorChannel(object):
    """A channel to monitor

    Parameters
    ----------
    name : str
        The name of this monitor channel, should be unique in order to
        distinguish channels.
    value : Theano variable or callable
        Either a Theano variable, or a callable function, which takes a
        model, training dataset, and set of monitoring datasets as
        arguments.

    Notes
    -----
    The values returned by callable monitor channels can be any Python
    object.

    """
    def __init__(self, name, value):
        if isinstance(value, Variable):
            self.is_theano_var = True
        else:
            assert callable(value)
            self.is_theano_var = False
        update_instance(self, locals())

    def __call__(self, model, training_dataset, monitoring_datasets):
        """Call the monitoring channel function.

        .. todo::

           Passing the model and datasets slightly breaks the modularity of
           the framework. Maybe an alternative should be considered, or at
           least a way to make this optional.

        """
        assert not self.is_theano_var
        return self.value(model, training_dataset, monitoring_datasets)


class IntervalMonitorChannel(MonitorChannel):
    """A channel that is monitored after a fixed interval or epoch.

    Parameters
    ----------
    name : str
        The name of this monitor channel, should be unique in order to
        distinguish channels.
    value : Theano variable or callable
        Either a Theano variable, or a callable function, which takes a
        model, training dataset, and set of monitoring datasets as
        arguments.
    needs_data : bool, optional
        If ``True``, it means that this channel's value depends on the
        input to the model (e.g. for a cost this should be ``True``, for
        weight norms it should be ``False``). In this case, the value will
        be averaged over a series of batches. If not, the value will only
        be requested once. Is ``False`` by default.

    """
    def __init__(self, name, value, needs_data=False):
        if needs_data:
            assert isinstance(value, Variable)
        update_instance(self, locals())
        super(IntervalMonitorChannel, self).__init__(name, value)


class UpdateMonitorChannel(MonitorChannel):
    """A channel that is monitored with each update.

    Parameters
    ----------
    name : str
        The name of this monitor channel, should be unique in order to
        distinguish channels.
    value : Theano variable
        A Theano variable to monitor at each update.

    Notes
    -----
    A update monitoring channel is compiled together with the Theano
    function that performs the update for performance. As such, it can only
    monitor Theano variables.

    """
    def __init__(self, name, value, **kwargs):
        assert isinstance(value, Variable)
        super(UpdateMonitorChannel, self).__init__(name, value, **kwargs)


class Monitor(object):
    """A class for monitoring Models while they are being trained.

    Records a variety of things (the objective function, reconstruction
    errors, weight norms, gradients, etc.)

    There are three kinds of monitoring.

    The first on calculates statistics after each epoch, potentially using
    a set of monitoring datasets (usually your validation and test sets).

    The second type of monitoring happens at each update. These monitoring
    channels are compiled together with the update function of the training
    algorithm. This can be useful to calculate e.g. the likelihood of your
    training batches for models where calculating statistics on a
    validation set would slow training down too much.

    .. todo::

       The third one calculates after a fixed interval of updates,
       potentially also using monitoring datasets. This is useful for very
       large (or infinite) datasets, where we want to get intermediate
       progress reports.

    Attributes
    ----------
    num_examples_seen : int
        The number of examples seen by the model.
    num_updates_seen : int
        The number of updates (e.g. minibatches) seen by the model.
    num_epochs_seen : int
        The number of epochs (full passes over a finite dataset) seen by
        the model.

    Notes
    -----
    Before calling the monitor, set the :attr:`num_examples_seen`,
    :attr:`num_updates_seen` and :attr:`num_epochs_seen` attributes to
    their correct values.

    """
    def __init__(self):
        self.num_examples_seen = 0
        self.num_updates_seen = 0
        self.num_epochs_seen = 0
        self.channels = []

    def get_interval_theano_vars(self, needs_data=None):
        """Get the Theano monitoring variables.

        Parameters
        ----------
        needs_data : bool, optional
            If ``True``, returns the channels that need data and vice
            versa. By default both are returned.

        Returns
        -------
        channels : list of channels
            List of :class:`UpdateMonitorChannel` instances.

        """
        channels = [channel for channel in self.channels if
                    isinstance(channel, IntervalMonitorChannel) and
                    channel.is_theano_var]
        if needs_data is not None:
            channels = [channel for channel in channels if
                        channel.needs_data == needs_data]
        return channels

    def setup(self, inputs=None):
        """Compiles the Theano functions for monitoring.

        Also initializes the Pandas data frame.

        Parameters
        ----------
        inputs : list of Theano variables
            These will be inputs of the compiled Theano function.

        """
        if inputs is None:
            inputs = []
        self.num_inputs = len(inputs)
        data_channels = self.get_interval_theano_vars(True)
        no_data_channels = self.get_interval_theano_vars(False)
        self.monitor_with_data_func = theano.function(
            inputs, [channel.value for channel in data_channels],
            allow_input_downcast=True, on_unused_input='ignore')
        self.monitor_without_data_func = theano.function(
            [], [channel.value for channel in no_data_channels],
            allow_input_downcast=True, on_unused_input='ignore')
        self.data = defaultdict(dict)

    def get_update_theano_vars(self):
        """Return the arguments of Theano monitor variables.

        This returns the arguments and keyword arguments needed to compile
        a Theano function that returns values to be monitored after each
        update. These values can be combined with the updates to the model
        to compile a function that both monitors and updates the model
        using a single computational graph, which is more computationally
        efficient.

        Returns
        -------
        channels : list of channels
            List of :class:`UpdateMonitorChannel` instances.

        Notes
        -----
        The numerical values of these monitoring channels are expected to
        be provided as arguments (in the form of a dictionary) when calling
        the :meth:`__call__` method.

        """
        channels = [channel for channel in self.channels
                    if isinstance(channel, UpdateMonitorChannel)]
        return channels

    def monitor(self, update_monitoring_values=None, monitoring_datasets=[]):
        """Perform monitoring.

        If batch-level monitor values are given, only these are monitored.
        If they aren't given, then the normal interval monitoring procedure
        is executed i.e. monitoring datasets and such.

        Parameters
        ----------
        update_monitoring_values : dict of Theano variables, object pairs
            A dictionary with the channel values (Theano variables) as
            keys, and the numerical values as values.

        """
        # TODO Check whether num_examples_seen/num_updates seen has changed
        index = (self.num_epochs_seen, self.num_updates_seen,
                 self.num_examples_seen)

        # Update monitoring channels
        if update_monitoring_values is not None:
            for channel, value in update_monitoring_values.items():
                self.data['batch_' + channel.name][index] = value

        else:
            # Interval monitoring channels that need data
            data_channel_names = \
                [channel.name for channel in
                 self.get_interval_theano_vars(needs_data=True)]
            for dataset in monitoring_datasets:
                for channel_name in data_channel_names:
                    # Looping to avoid KeyError the first call
                    self.data[channel_name][index] = 0
                for i, batch in enumerate(dataset):
                    data = self.monitor_with_data_func(
                        *pack(batch[:self.num_inputs]))
                    for channel_name, value in zip(data_channel_names, data):
                        self.data[channel_name][index] += value
                for channel_name in data_channel_names:
                    self.data[channel_name][index] /= i + 1

            # Interval monitoring channels that do not need data
            no_data_channel_names = \
                [channel.name for channel in
                 self.get_interval_theano_vars(needs_data=False)]
            data = self.monitor_without_data_func()
            for channel_name, value in zip(no_data_channel_names, data):
                # TODO Figure out what this bug is about
                if isinstance(value, theano.sandbox.cuda.CudaNdarray):
                    value = numpy.asarray(value)
                self.data[channel_name][index] = value

        # TODO Prettify this
        # TODO Much too verbose for batch monitoring, how to control this?
        logger.info((index,
                     [(channel_name, self.data[channel_name].get(index))
                      for channel_name in self.data.keys()
                      if self.data[channel_name].get(index) is not None]))
