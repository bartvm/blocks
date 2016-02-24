"""Bricks are parameterized Theano operations."""
from .base import application, Brick, lazy
from .bn import (BatchNormalization, SpatialBatchNormalization,
                 BatchNormalizedMLP)
from .interfaces import Activation, Feedforward, Initializable, Random
from .simple import (Linear, Bias, Maxout, LinearMaxout, Identity, Tanh,
                     Logistic, Softplus, Rectifier, Softmax,
                     NDimensionalSoftmax)
from .sequences import Sequence, FeedforwardSequence, MLP
from .wrappers import WithExtraDims

__all__ = ('application', 'Brick', 'lazy', 'BatchNormalization',
           'SpatialBatchNormalization', 'BatchNormalizedMLP',
           'Activation', 'Feedforward', 'Initializable', 'Random',
           'Linear', 'Bias', 'Maxout', 'LinearMaxout', 'Identity',
           'Tanh', 'Logistic', 'Softplus', 'Rectifier', 'Softmax',
           'NDimensionalSoftmax', 'Sequence', 'FeedforwardSequence',
           'MLP', 'WithExtraDims')
