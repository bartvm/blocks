from abc import ABCMeta, abstractmethod

from theano import tensor

from blocks.monitor import IntervalMonitorChannel, UpdateMonitorChannel
from blocks.utils import shared_floatx


class TrainingAlgorithm(object):
    """The interface for training algorithms."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_train_updates(self):
        """Return the updates to the model constituting a single update."""
        # TODO Should this return inputs as well? Does an algorithm ever
        # need inputs beyond the model inputs?
        pass

    def get_monitoring_channels(self):
        """Return monitoring channels.

        For example, the learning rate, momentum, gradient norms, etc.

        """
        return []


class SGD(TrainingAlgorithm):
    """Basic SGD training algorithm.

    .. todo::

       * Support learning rate scheduling.
       * Support momentum
       * Support passing of single parameter

    """
    def __init__(self, cost, parameters, learning_rate):
        self.cost = cost
        self.parameters = parameters
        self.learning_rate = shared_floatx(learning_rate, name='learning_rate')

    def get_train_updates(self):
        self.gradients = tensor.grad(self.cost, self.parameters)
        updates = dict((parameter, parameter - self.learning_rate * gradient)
                       for parameter, gradient in zip(self.parameters,
                                                      self.gradients))
        return updates

    def get_monitoring_channels(self):
        """Returns the gradient and learning rate as monitoring channels."""
        gradient_norm = tensor.sqrt(tensor.sum(tensor.sqr(tensor.concatenate(
            [gradient.flatten() for gradient in self.gradients]))))

        channels = []
        channels.append(UpdateMonitorChannel('gradient_norm', gradient_norm))
        channels.append(IntervalMonitorChannel('gradient_norm', gradient_norm,
                                               needs_data=True))
        channels.append(IntervalMonitorChannel('learning_rate',
                                               self.learning_rate))
        return channels
