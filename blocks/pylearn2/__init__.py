"""Wrappers for training bricks with PyLearn2.

This module contains a set of wrappers that allows to outsource
training and monitoring to Pylearn2.

"""
import logging
from collections import OrderedDict

import theano
from theano import tensor
import pylearn2.costs.cost
import pylearn2.models
import pylearn2.train
import pylearn2.training_algorithms.learning_rule
import pylearn2.train_extensions
import pylearn2.space
from pylearn2.space import CompositeSpace
from pylearn2.utils import serial
from pylearn2.monitor import push_monitor

from blocks.select import BrickSelection
from blocks.utils import pack
from blocks.graph import ComputationGraph
from blocks.utils import shared_floatx, unpack

logger = logging.getLogger()


class Pylearn2Model(pylearn2.models.Model):
    supervised = False

    def __init__(self, brick, **kwargs):
        """Wraps a brick to support the Pylearn2 model interface.

        Parameters
        ----------
        brick : Brick
            The brick to wrap.

        """
        self.brick = brick
        super(Pylearn2Model, self).__init__(**kwargs)

    def get_params(self):
        return BrickSelection(self.brick).get_params().values()

    @staticmethod
    def load(path):
        """Loads a model from path.

        We need this wrapper to make the loaded monitor continuable
        (currently deserialized monitor is non-functional in PyLearn2).
        For this we had to create a new monitor and initialize with the
        data from the old one.

        Parameters
        ----------
        path : str
            The model path.

        """
        model = push_monitor(serial.load(path), "_delete_me",
                             transfer_experience=True, save_records=True)
        del model._delete_me
        return model


class Pylearn2Cost(pylearn2.costs.cost.Cost):
    """Wraps a Theano cost to support the PyLearn2 Cost interface.

    Parameters
    ----------
    cost : Theano variable
        The Theano variable corresponding to the end of the cost
        computation graph.

    Notes
    -----
    The inputs of the computation graph must have names compatible with
    names of the data sources. The is necessary in order to replace with
    with the ones given by PyLearn2.

    """
    def __init__(self, cost):
        self.cost = cost
        self.inputs = ComputationGraph(self.cost).dict_of_inputs()

    def expr(self, model, data, **kwargs):
        assert not model.supervised
        data = pack(data)
        data = [tensor.unbroadcast(var, *range(var.ndim))
                for var in data]
        return theano.clone(
            self.cost, replace=dict(zip(self.inputs.values(), data)))

    def get_gradients(self, model, data, **kwargs):
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.expr(model, data), p)
                           for p in model.get_params()]
        return OrderedDict(zip(model.get_params(), self._grads)), OrderedDict()

    def get_monitoring_channels(self, model, data, **kwargs):
        return OrderedDict()

    def get_data_specs(self, model):
        return model.data_specs


class SGDLearningRule(pylearn2.training_algorithms.learning_rule.LearningRule):
    """The default SGD learning rule.

    .. todo::

        Move this class to PyLearn2 and make it the default learning rule.

    """
    def get_updates(self, learning_rate, grads, lr_scalers):
        return {param:
                param - learning_rate * lr_scalers.get(param, 1.) * grad
                for param, grad in grads.items()}


class Pylearn2LearningRule(pylearn2.training_algorithms
                           .learning_rule.LearningRule):
    """Wraps a PyLearn2 learning rule to add per-update monitoring.

    Parameters
    ----------
    learning_rule : :class:`LearningRule`
        A PyLearn2 learning rule to wrap.
    monitor_values : dict of (name, Theano variable) pairs
        The values to monitor and their names.
    updates : OrderedDict
        Custom updates to perform when computing gradients.

    .. todo::
        `updates` are never used.

    """
    def __init__(self, learning_rule, monitor_values=None, updates=None):
        self.learning_rule = learning_rule
        self.values = []
        self.accumulators = []
        self._callback_called = False

        if monitor_values:
            for name, value in monitor_values.items():
                self.monitor_value(name, value)
        if not updates:
            updates = OrderedDict()
        self.updates = updates

    def monitor_value(self, name, value):
        """Add monitoring to be performed with gradient computation.

        Parameters
        ----------
        name : str
            The name of the value to be monitored.
        value : Theano variable
            The value to be monitored.

        """
        if self._callback_called:
            raise Exception("It is to add to monitoring to the {}:"
                            "a callback has been called".format(self.__name__))
        self.values.append(value)
        self.accumulators.append(shared_floatx(0, name=name))

    def add_channels_to_monitor(self, monitor, datasets):
        self.learning_rule.add_channels_to_monitor(monitor, datasets)
        for accumulator in self.accumulators:
            monitor.add_channel(accumulator.name, ipt=None, val=accumulator,
                                data_specs=(pylearn2.space.NullSpace(), ''),
                                dataset=datasets)
        self._callback_called = True

    def get_updates(self, learning_rate, grads, lr_scalers):
        """Wraps the respective method of the wrapped learning rule.

        Performs name-based input substitution for the monitored values.
        Currently very hacky: the inputs from the gradients are typically
        named `$ALGO[$SOURCE]` in PyLearn2, where `$ALGO` is the algorithm
        name and `$SOURCE` is a source name from the data specification.
        This convention is exploited to match them with the inputs of
        monitoring values, whose input names are expected to match source
        names.

        """
        updates = self.learning_rule.get_updates(learning_rate, grads,
                                                 lr_scalers)
        grad_inputs = ComputationGraph(list(grads.values())).dict_of_inputs()
        for value, accumulator in zip(self.values, self.accumulators):
            value_inputs = ComputationGraph(value).dict_of_inputs()
            replace_dict = dict()
            for name, input_ in value_inputs.items():
                # See docstring to see how it works
                grad_input = grad_inputs[unpack(
                    [n for n in grad_inputs
                     if n.endswith('[{}]'.format(name))],
                    singleton=True)]
                replace_dict[input_] = tensor.unbroadcast(
                    grad_input, *range(grad_input.ndim))
            updates[accumulator] = (
                accumulator + theano.clone(value, replace_dict))
        self._callback_called = True
        updates.update(self.updates)
        return updates


class DefaultExtension(pylearn2.train_extensions.TrainExtension):
    """This extension helps Pylearn2LearningRule do its job.

    The job of this extensions is to help the Pylearn2LearningRule in its
    monitoring duties. Due to impossibility of reseting the accumulators of
    monitored values, the gradient computation function simply adds values
    from new batches to the accumulators. At the end of each epoch the
    accumulator's value from the previous epoch should be subtracted and
    the difference should be divided over the number of batches to get an
    average for the last epoch. This is done in the `on_monitor` method.

    """
    def setup(self, model, dataset, algoritm):
        self._last_batches_seen = model.monitor.get_batches_seen()
        self._last_values = dict()

    def on_monitor(self, model, dataset, algorithm):
        learning_rule = algorithm.learning_rule
        if not learning_rule:
            return
        batches_seen = model.monitor.get_batches_seen()
        if (isinstance(learning_rule, Pylearn2LearningRule) and
                len(self._last_values)):
            for accum in learning_rule.accumulators:
                accum.set_value(
                    ((accum.get_value() - self._last_values[accum]) /
                     (batches_seen - self._last_batches_seen)).astype(
                         theano.config.floatX))
        for accum in learning_rule.accumulators:
            self._last_values[accum] = accum.get_value()
        batches_seen -= self._last_batches_seen


class Pylearn2Train(pylearn2.train.Train):
    """Convinience wrapper over the PyLearn2 main loop.

    Sets `model.data_specs` using `dataset.data_specs` and the names of the
    input variables.

    """
    def __init__(self, dataset, model, algorithm,
                 save_path=None, save_freq=0, extensions=None,
                 *args, **kwargs):
        # Set data_specs
        spaces, sources = dataset.data_specs
        if isinstance(spaces, CompositeSpace):
            spaces = spaces.components
        else:
            spaces = (spaces,)
            sources = (sources,)
        input_names = list(algorithm.cost.inputs.keys())
        spaces = [spaces[sources.index(source)] for source in input_names]
        if len(spaces) > 1:
            spaces = CompositeSpace(spaces)
            sources = input_names
        else:
            spaces = spaces[0]
            sources = input_names[0]
        model.data_specs = (spaces, sources)

        # Add default extensions
        if not extensions:
            extensions = list()
        extensions.append(DefaultExtension())

        super(Pylearn2Train, self).__init__(
            dataset, model, algorithm, save_path, save_freq, extensions,
            *args, **kwargs)

    def setup(self):
        """Make monitor persistency the default behaviour."""
        if hasattr(self.model, 'monitor'):
            # Cheat on monitor._sanity_check
            # TODO: raise a discussion about it
            for channel in self.model.monitor.channels.values():
                channel.prereqs = None
        super(Pylearn2Train, self).setup()
        self.model.monitor.on_channel_conflict = 'copy_history'
