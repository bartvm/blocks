#!/usr/bin/env python
"""Learn a Markov chain with an RNN and sample from it."""
from __future__ import print_function

import argparse
import logging
import pprint
import sys

from six.moves import cPickle
import numpy
import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.recurrent import GatedRecurrent
from blocks.bricks.sequence_generators import (
    SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from fuel.streams import DataStream
from fuel.schemes import ConstantScheme
from blocks.algorithms import GradientDescent, Scale
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.select import Selector
from examples.markov_chain.dataset import MarkovChainDataset

sys.setrecursionlimit(10000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def main(mode, save_path, steps, num_batches):
    num_states = MarkovChainDataset.num_states

    if mode == "train":
        # Experiment configuration
        rng = numpy.random.RandomState(1)
        batch_size = 50
        seq_len = 100
        dim = 10
        feedback_dim = 8

        # Build the bricks and initialize them
        transition = GatedRecurrent(name="transition", activation=Tanh(),
                                    dim=dim)
        generator = SequenceGenerator(
            Readout(readout_dim=num_states, source_names=["states"],
                    emitter=SoftmaxEmitter(name="emitter"),
                    feedback_brick=LookupFeedback(
                        num_states, feedback_dim, name='feedback'),
                    name="readout"),
            transition,
            weights_init=IsotropicGaussian(0.01), biases_init=Constant(0),
            name="generator")
        generator.push_initialization_config()
        transition.weights_init = Orthogonal()
        generator.initialize()

        # Give an idea of what's going on.
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in Selector(generator).get_params().items()],
                        width=120))
        logger.info("Markov chain entropy: {}".format(
            MarkovChainDataset.entropy))
        logger.info("Expected min error: {}".format(
            -MarkovChainDataset.entropy * seq_len))

        # Build the cost computation graph.
        x = tensor.lmatrix('data')
        cost = aggregation.mean(generator.cost(x[:, :]).sum(),
                                x.shape[1])
        cost.name = "sequence_log_likelihood"

        algorithm = GradientDescent(
            cost=cost, params=list(Selector(generator).get_params().values()),
            step_rule=Scale(0.001))
        main_loop = MainLoop(
            algorithm=algorithm,
            data_stream=DataStream(
                MarkovChainDataset(rng, seq_len),
                iteration_scheme=ConstantScheme(batch_size)),
            model=Model(cost),
            extensions=[FinishAfter(after_n_batches=num_batches),
                        TrainingDataMonitoring([cost], prefix="this_step",
                                               after_every_batch=True),
                        TrainingDataMonitoring([cost], prefix="average",
                                               every_n_batches=100),
                        Checkpoint(save_path, every_n_batches=500),
                        Printing(every_n_batches=100)])
        main_loop.run()
    elif mode == "sample":
        main_loop = cPickle.load(open(save_path, "rb"))
        generator = main_loop.model

        sample = ComputationGraph(generator.generate(
            n_steps=steps, batch_size=1, iterate=True)).get_theano_function()

        states, outputs, costs = [data[:, 0] for data in sample()]

        numpy.set_printoptions(precision=3, suppress=True)
        print("Generation cost:\n{}".format(costs.sum()))

        freqs = numpy.bincount(outputs).astype(floatX)
        freqs /= freqs.sum()
        print("Frequencies:\n {} vs {}".format(freqs,
                                               MarkovChainDataset.equilibrium))

        trans_freqs = numpy.zeros((num_states, num_states), dtype=floatX)
        for a, b in zip(outputs, outputs[1:]):
            trans_freqs[a, b] += 1
        trans_freqs /= trans_freqs.sum(axis=1)[:, None]
        print("Transition frequencies:\n{}\nvs\n{}".format(
            trans_freqs, MarkovChainDataset.trans_prob))
    else:
        assert False


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        "Case study of generating a Markov chain with RNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "mode", choices=["train", "sample"],
        help="The mode to run. Use `train` to train a new model"
             " and `sample` to sample a sequence generated by an"
             " existing one.")
    parser.add_argument(
        "save_path", default="chain",
        help="The path to save the training process.")
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Number of steps to samples.")
    parser.add_argument(
        "--num-batches", default=1000, type=int,
        help="Train on this many batches.")
    args = parser.parse_args()
    main(**vars(args))
