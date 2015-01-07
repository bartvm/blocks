import logging
import numpy
from abc import ABCMeta, abstractmethod

import theano
from theano import tensor
from theano import Variable

from blocks.graph import ComputationGraph
from blocks.select import BrickSelection
from blocks.serialization import save_params, load_params

logger = logging.getLogger(__name__)


class GroundhogIterator(object):
    """A base class for Groundhog compatible iterator.

    Has mock implementations of all required methods except `next`.

    """
    __metaclass__ = ABCMeta

    def start(self, offset):
        pass

    @abstractmethod
    def next(self):
        pass

    @property
    def next_offset(self):
        return 0


class GroundhogModel(object):
    """Wraps a model into a Groundhog compatible interface."""
    def __init__(self, bricks, cost):
        if not isinstance(bricks, BrickSelection):
            bricks = BrickSelection(bricks)
        if isinstance(cost, Variable):
            cost = ComputationGraph(cost)
        self.bricks = bricks
        self.cost = cost

        self.properties = []
        self.updates = []

    @property
    def params(self):
        # Caching for speed
        if not hasattr(self, "_params"):
            self._params = self.bricks.get_params().values()
        return self._params

    @property
    def train_cost(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_train_cost"):
            self._train_cost = self.cost.outputs[0]
        return self._train_cost

    @property
    def param_grads(self):
        # Caching to simplify the computation graph
        if not hasattr(self, "_grads"):
            self._grads = [tensor.grad(self.train_cost, p)
                           for p in self.params]
        return self._grads

    @property
    def inputs(self):
        return self.cost.inputs

    @property
    def valid_costs(self):
        return ['cost'] + [name for name, var in self.properties]

    def validate(self, data):
        valid_names = self.valid_costs
        valid_vars = [self.train_cost] + [var for name, var in self.properties]

        sums = numpy.zeros((len(valid_vars),))
        num_batches = 0

        if not hasattr(self, "_valid_func"):
            self._valid_func = theano.function(self.inputs, valid_vars,
                                               updates=self.updates)
        for batch in data:
            sums += numpy.hstack(self._valid_func(
                *[batch[input_.name] for input_ in self.inputs]))
            num_batches += 1
        return zip(valid_names, sums / num_batches)

    @property
    def params_grad_scale(self):
        return [1.0] * len(self.params)

    @property
    def exclude_params_for_norm(self):
        return []

    def perturb(self, **kwargs):
        return kwargs

    def get_schedules(self):
        return []

    def save(self, path):
        save_params(self.bricks, path)

    def load(self, path):
        load_params(self.bricks, path)


class GroundhogState(object):
    """Good default values for groundhog state."""
    def __init__(self, prefix, batch_size, learning_rate, **kwargs):
        self.prefix = prefix
        self.bs = batch_size
        self.lr = learning_rate
        self.seed = 1

        # Early stopping
        self.cost_threshold = 10 ** 3  # effectively disables early stopping
        self.patience = 1
        self.divide_lr = False
        self.minerr = -1
        self.timeStop = 10 ** 9
        self.minlr = 0

        self.overwrite = True
        self.hookFreq = -1
        self.saveFreq = 30
        self.validFreq = 10 ** 9
        self.trainFreq = 1
        self.loopIters = 10 ** 6

        self.cutoff_rescale_length = 0.0

    def as_dict(self):
        return {key: value for key, value in self.__dict__.items()
                if not key.startswith('_')}
