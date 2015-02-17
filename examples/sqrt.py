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
from blocks.initialization import IsotropicGaussian, Constant
from blocks.datasets import ContainerDataset
from blocks.datasets.streams import BatchDataStream, DataStreamMapping
from blocks.datasets.schemes import ConstantScheme
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.saveload import LoadFromDump, Dump
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.main_loop import MainLoop

floatX = theano.config.floatX


def get_data_stream(iterable):
    dataset = ContainerDataset({'numbers': iterable})
    data_stream = DataStreamMapping(dataset.get_default_stream(),
                                    lambda data: (math.sqrt(data[0]),),
                                    add_sources=('roots',))
    data_stream = DataStreamMapping(
        data_stream,
        lambda data: tuple((numpy.asarray(d, dtype=floatX) for d in data)))
    return BatchDataStream(data_stream, ConstantScheme(20))


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
        mlp,
        get_data_stream(range(100)),
        GradientDescent(
            cost=cost, step_rule=Scale(learning_rate=0.001)),
        extensions=([LoadFromDump(save_to)] if continue_ else []) +
        [Timing(),
            FinishAfter(after_n_batches=num_batches),
            DataStreamMonitoring(
                [cost], get_data_stream(range(100, 200)),
                prefix="test"),
            TrainingDataMonitoring([cost], after_every_epoch=True),
            Dump(save_to),
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
