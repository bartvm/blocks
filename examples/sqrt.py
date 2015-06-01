#!/usr/bin/env python
"""Super-basic example, mainly for testing purposes.

This script trains a tiny network to compute square root. It also
serves as an example of using dumping.

"""
import logging
import math
import numpy
from argparse import ArgumentParser

import theano
from theano import tensor

from blocks.algorithms import GradientDescent, Scale
from blocks.bricks import MLP, Tanh, Identity
from blocks.bricks.cost import SquaredError
from blocks.graph import ComputationGraph
from blocks.initialization import IsotropicGaussian, Constant
from blocks.model import Model
from fuel.datasets import IterableDataset
from fuel.transformers import Batch, Mapping
from fuel.schemes import ConstantScheme
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.main_loop import MainLoop


def _data_sqrt(data):
    return (math.sqrt(data[0]),)


def _array_tuple(data):
    return tuple((numpy.asarray(d, dtype=theano.config.floatX) for d in data))


def get_data_stream(iterable):
    dataset = IterableDataset({'numbers': iterable})
    data_stream = Mapping(dataset.get_example_stream(),
                          _data_sqrt, add_sources=('roots',))
    data_stream = Mapping(data_stream, _array_tuple)
    return Batch(data_stream, ConstantScheme(20))


def main(save_to, num_batches, continue_=False):
    mlp = MLP([Tanh(), Identity()], [1, 10, 1],
              weights_init=IsotropicGaussian(0.01),
              biases_init=Constant(0), seed=1)
    mlp.initialize()
    x = tensor.vector('numbers')
    y = tensor.vector('roots')
    cost = SquaredError().apply(y[:, None], mlp.apply(x[:, None]))
    cost.name = "cost"

    main_loop = MainLoop(
        GradientDescent(
            cost=cost, params=ComputationGraph(cost).parameters,
            step_rule=Scale(learning_rate=0.001)),
        get_data_stream(range(100)),
        model=Model(cost),
        extensions=[Timing(),
                    FinishAfter(after_n_batches=num_batches),
                    DataStreamMonitoring(
                        [cost], get_data_stream(range(100, 200)),
                        prefix="test"),
                    TrainingDataMonitoring([cost], after_epoch=True),
                    Checkpoint(save_to),
                    Printing()])
    main_loop.run()
    return main_loop

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser("An example of learning to take square root")
    parser.add_argument("--num-batches", type=int, default=1000,
                        help="Number of training batches to do.")
    parser.add_argument("save_to", default="sqrt", nargs="?",
                        help=("Destination to save the state of the training "
                              "process."))
    args = parser.parse_args()
    main(**vars(args))
