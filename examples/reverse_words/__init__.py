from __future__ import print_function
import logging
import pprint
import sys
import math
import dill
import numpy
import os

import theano
from six.moves import input
from theano import tensor

from blocks.bricks import Tanh, application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.datasets.streams import (
    DataStreamMapping, BatchDataStream, PaddingDataStream,
    DataStreamFilter)
from blocks.datasets.text import OneBillionWord, TextFile
from blocks.datasets.schemes import ConstantScheme
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import SerializeMainLoop, LoadFromDump
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.select import Selector
from blocks.filter import VariableFilter
from blocks.utils import named_copy, unpack, dict_union

sys.setrecursionlimit(100000)
floatX = theano.config.floatX
logger = logging.getLogger(__name__)

# Dictionaries
all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '<UNK>'] +
             [' ', '<S>', '</S>'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}


def reverse_words(sample):
    sentence = sample[0]
    result = []
    word_start = -1
    for i, code in enumerate(sentence):
        if code >= char2code[' ']:
            if word_start >= 0:
                result.extend(sentence[i - 1:word_start - 1:-1])
                word_start = -1
            result.append(code)
        else:
            if word_start == -1:
                word_start = i
    return (result,)


class Transition(SimpleRecurrent):
    def __init__(self, attended_dim, **kwargs):
        super(Transition, self).__init__(**kwargs)
        self.attended_dim = attended_dim

    @application(contexts=['attended', 'attended_mask'])
    def apply(self, *args, **kwargs):
        for context in Transition.apply.contexts:
            kwargs.pop(context)
        return super(Transition, self).apply(*args, **kwargs)

    @apply.delegate
    def apply_delegate(self):
        return super(Transition, self).apply

    def get_dim(self, name):
        if name == 'attended':
            return self.attended_dim
        if name == 'attended_mask':
            return 0
        return super(Transition, self).get_dim(name)


def main(mode, save_path, num_batches, from_dump, data_path=None):
    if mode == "train":
        # Experiment configuration
        dimension = 100
        readout_dimension = len(char2code)

        # Data processing pipeline
        dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=str.lower)
        if data_path:
            dataset = TextFile(data_path, **dataset_options)
        else:
            dataset = OneBillionWord("training", [99], **dataset_options)
        data_stream = DataStreamMapping(
            mapping=lambda data: tuple(array.T for array in data),
            data_stream=PaddingDataStream(
                BatchDataStream(
                    iteration_scheme=ConstantScheme(10),
                    data_stream=DataStreamMapping(
                        mapping=reverse_words,
                        add_sources=("targets",),
                        data_stream=DataStreamFilter(
                            predicate=lambda data: len(data[0]) <= 100,
                            data_stream=dataset
                            .get_default_stream())))))

        # Build the model
        chars = tensor.lmatrix("features")
        chars_mask = tensor.matrix("features_mask")
        targets = tensor.lmatrix("targets")
        targets_mask = tensor.matrix("targets_mask")

        encoder = Bidirectional(
            SimpleRecurrent(dim=dimension, activation=Tanh()),
            weights_init=Orthogonal())
        encoder.initialize()
        fork = Fork([name for name in encoder.prototype.apply.sequences
                     if name != 'mask'],
                    weights_init=IsotropicGaussian(0.1),
                    biases_init=Constant(0))
        fork.input_dim = dimension
        fork.output_dims = {name: dimension for name in fork.input_names}
        fork.initialize()
        lookup = LookupTable(readout_dimension, dimension,
                             weights_init=IsotropicGaussian(0.1))
        lookup.initialize()
        transition = Transition(
            activation=Tanh(),
            dim=dimension, attended_dim=2 * dimension, name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            match_dim=dimension, name="attention")
        readout = LinearReadout(
            readout_dim=readout_dimension, source_names=["states"],
            emitter=SoftmaxEmitter(name="emitter"),
            feedbacker=LookupFeedback(readout_dimension, dimension),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            weights_init=IsotropicGaussian(0.1), biases_init=Constant(0),
            name="generator")
        generator.push_initialization_config()
        transition.weights_init = Orthogonal()
        generator.initialize()
        bricks = [encoder, fork, lookup, generator]

        # Give an idea of what's going on
        params = Selector(bricks).get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        # Build the cost computation graph
        batch_cost = generator.cost(
            targets, targets_mask,
            attended=encoder.apply(
                **dict_union(
                    fork.apply(lookup.lookup(chars), return_dict=True),
                    mask=chars_mask)),
            attended_mask=chars_mask).sum()
        batch_size = named_copy(chars.shape[1], "batch_size")
        cost = aggregation.mean(batch_cost,  batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Fetch variables useful for debugging
        max_length = named_copy(chars.shape[0], "max_length")
        cost_per_character = named_copy(
            aggregation.mean(batch_cost, batch_size * max_length),
            "character_log_likelihood")
        cg = ComputationGraph(cost)
        energies = unpack(
            VariableFilter(application=readout.readout, name="output")(
                cg.variables),
            singleton=True)
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        (activations,) = VariableFilter(
            application=generator.transition.apply,
            name="states")(cg.variables)
        mean_activation = named_copy(abs(activations).mean(),
                                     "mean_activation")

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=cost, step_rule=CompositeRule([StepClipping(10.0),
                                                Scale(0.01)]))

        observables = [
            cost, min_energy, max_energy, mean_activation,
            batch_size, max_length, cost_per_character,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        average_monitoring = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)
        main_loop = MainLoop(
            model=bricks,
            data_stream=data_stream,
            algorithm=algorithm,
            extensions=([LoadFromDump(from_dump)] if from_dump else []) +
            [Timing(),
                TrainingDataMonitoring(observables, after_every_batch=True),
                average_monitoring,
                FinishAfter(after_n_batches=num_batches)
                .add_condition(
                    "after_batch",
                    lambda log:
                        math.isnan(log.current_row.total_gradient_norm)),
                Plot(os.path.basename(save_path),
                     [[average_monitoring.record_name(cost)],
                      [average_monitoring.record_name(cost_per_character)]],
                     every_n_batches=10),
                SerializeMainLoop(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                Printing(every_n_batches=1)])
        main_loop.run()
    elif mode == "test":
        with open(save_path, "rb") as source:
            encoder, fork, lookup, generator = dill.load(source)
        logger.info("Model is loaded")
        chars = tensor.lmatrix("features")
        generated = generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=encoder.apply(
                **dict_union(fork.apply(lookup.lookup(chars),
                             return_dict=True))),
            attended_mask=tensor.ones(chars.shape))
        sample_function = ComputationGraph(generated).get_theano_function()
        logging.info("Sampling function is compiled")

        while True:
            # Python 2-3 compatibility
            line = input("Enter a sentence\n")
            batch_size = int(input("Enter a number of samples\n"))
            encoded_input = [char2code.get(char, char2code["<UNK>"])
                             for char in line.lower().strip()]
            encoded_input = ([char2code['<S>']] + encoded_input +
                             [char2code['</S>']])
            print("Encoder input:", encoded_input)
            target = reverse_words((encoded_input,))[0]
            print("Target: ", target)
            states, samples, glimpses, weights, costs = sample_function(
                numpy.repeat(numpy.array(encoded_input)[:, None],
                             batch_size, axis=1))

            messages = []
            for i in range(samples.shape[1]):
                sample = list(samples[:, i])
                try:
                    true_length = sample.index(char2code['</S>']) + 1
                except ValueError:
                    true_length = len(sample)
                sample = sample[:true_length]
                cost = costs[:true_length, i].sum()
                message = "({})".format(cost)
                message += "".join(code2char[code] for code in sample)
                if sample == target:
                    message += " CORRECT!"
                messages.append((cost, message))
            messages.sort(key=lambda tuple_: -tuple_[0])
            for _, message in messages:
                print(message)
