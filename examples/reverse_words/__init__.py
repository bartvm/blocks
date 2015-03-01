from __future__ import print_function
from collections import OrderedDict
import logging
import pprint
import math
import numpy
import os
import operator

import theano
from six.moves import input
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (
    SequenceGenerator, LinearReadout, SoftmaxEmitter, LookupFeedback)
from blocks.config_parser import config
from blocks.graph import ComputationGraph
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.datasets import OneBillionWord, TextFile
from fuel.schemes import ConstantScheme
from blocks.dump import load_parameter_values
from blocks.algorithms import (GradientDescent, Scale,
                               StepClipping, CompositeRule)
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.model import Model
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import SerializeMainLoop
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter
from blocks.utils import named_copy, dict_union

from blocks.search import BeamSearch

config.recursion_limit = 100000
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


def _lower(s):
    return s.lower()


def _transpose(data):
    return tuple(array.T for array in data)


def _filter_long(data):
    return len(data[0]) <= 100


def _is_nan(log):
    return math.isnan(log.current_row.total_gradient_norm)


def main(mode, save_path, num_batches, data_path=None):
    # Experiment configuration
    dimension = 100
    readout_dimension = len(char2code)

    # Build bricks
    encoder = Bidirectional(
        SimpleRecurrent(dim=dimension, activation=Tanh()),
        weights_init=Orthogonal())
    fork = Fork([name for name in encoder.prototype.apply.sequences
                 if name != 'mask'],
                weights_init=IsotropicGaussian(0.1),
                biases_init=Constant(0))
    fork.input_dim = dimension
    fork.output_dims = {name: dimension for name in fork.input_names}
    lookup = LookupTable(readout_dimension, dimension,
                         weights_init=IsotropicGaussian(0.1))
    transition = SimpleRecurrent(
        activation=Tanh(),
        dim=dimension, name="transition")
    attention = SequenceContentAttention(
        state_names=transition.apply.states,
        sequence_dim=2 * dimension, match_dim=dimension, name="attention")
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

    if mode == "train":
        # Data processing pipeline
        dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)
        if data_path:
            dataset = TextFile(data_path, **dataset_options)
        else:
            dataset = OneBillionWord("training", [99], **dataset_options)
        data_stream = Mapping(
            mapping=_transpose,
            data_stream=Padding(
                Batch(
                    iteration_scheme=ConstantScheme(10),
                    data_stream=Mapping(
                        mapping=reverse_words,
                        add_sources=("targets",),
                        data_stream=Filter(
                            predicate=_filter_long,
                            data_stream=dataset
                            .get_example_stream())))))

        # Build the cost computation graph
        chars = tensor.lmatrix("features")
        chars_mask = tensor.matrix("features_mask")
        targets = tensor.lmatrix("targets")
        targets_mask = tensor.matrix("targets_mask")
        batch_cost = generator.cost(
            targets, targets_mask,
            attended=encoder.apply(
                **dict_union(
                    fork.apply(lookup.lookup(chars), as_dict=True),
                    mask=chars_mask)),
            attended_mask=chars_mask).sum()
        batch_size = named_copy(chars.shape[1], "batch_size")
        cost = aggregation.mean(batch_cost,  batch_size)
        cost.name = "sequence_log_likelihood"
        logger.info("Cost graph is built")

        # Give an idea of what's going on
        model = Model(cost)
        params = model.get_params()
        logger.info("Parameters:\n" +
                    pprint.pformat(
                        [(key, value.get_value().shape) for key, value
                         in params.items()],
                        width=120))

        # Initialize parameters
        for brick in model.get_top_bricks():
            brick.initialize()

        # Fetch variables useful for debugging
        max_length = named_copy(chars.shape[0], "max_length")
        cost_per_character = named_copy(
            aggregation.mean(batch_cost, batch_size * max_length),
            "character_log_likelihood")
        cg = ComputationGraph(cost)
        (energies,) = VariableFilter(
            application=readout.readout, name="output")(cg.variables)
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        (activations,) = VariableFilter(
            application=generator.transition.apply,
            name="states")(cg.variables)
        mean_activation = named_copy(abs(activations).mean(),
                                     "mean_activation")

        # Define the training algorithm.
        algorithm = GradientDescent(
            cost=cost, params=cg.parameters,
            step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

        # More variables for debugging
        observables = [
            cost, min_energy, max_energy, mean_activation,
            batch_size, max_length, cost_per_character,
            algorithm.total_step_norm, algorithm.total_gradient_norm]
        for name, param in params.items():
            observables.append(named_copy(
                param.norm(2), name + "_norm"))
            observables.append(named_copy(
                algorithm.gradients[param].norm(2), name + "_grad_norm"))

        # Construct the main loop and start training!
        average_monitoring = TrainingDataMonitoring(
            observables, prefix="average", every_n_batches=10)
        main_loop = MainLoop(
            model=model,
            data_stream=data_stream,
            algorithm=algorithm,
            extensions=[
                Timing(),
                TrainingDataMonitoring(observables, after_every_batch=True),
                average_monitoring,
                FinishAfter(after_n_batches=num_batches)
                .add_condition("after_batch", _is_nan),
                Plot(os.path.basename(save_path),
                     [[average_monitoring.record_name(cost)],
                      [average_monitoring.record_name(cost_per_character)]],
                     every_n_batches=10),
                SerializeMainLoop(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                Printing(every_n_batches=1)])
        main_loop.run()
    elif mode == "test":
        logger.info("Model is loaded")
        chars = tensor.lmatrix("features")
        generated = generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=encoder.apply(
                **dict_union(fork.apply(lookup.lookup(chars),
                             as_dict=True))),
            attended_mask=tensor.ones(chars.shape))
        model = Model(generated)
        model.set_param_values(load_parameter_values(save_path))
        sample_function = model.get_theano_function()
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
            messages.sort(key=operator.itemgetter(0), reverse=True)
            for _, message in messages:
                print(message)
    elif mode == 'beam_search':
        logger.info("Model is loaded")
        chars = tensor.lmatrix("features")
        chars_mask = tensor.matrix("features_mask")
        attended = encoder.apply(
            **dict_union(fork.apply(lookup.lookup(chars),
                                    as_dict=True)))
        generated = generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=attended,
            attended_mask=chars_mask)
        cg = ComputationGraph(generated)
        model = Model(generated)
        model.set_param_values(load_parameter_values(save_path))
        batch_size = 1
        beam_search = BeamSearch(10, generator, cg)
        beam_search.compile()

        while True:
            # Python 2-3 compatibility
            line = input("Enter a sentence\n")
            encoded_input = [char2code.get(char, char2code["<UNK>"])
                             for char in line.lower().strip()]
            encoded_input = ([char2code['<S>']] + encoded_input +
                             [char2code['</S>']])
            print("Encoder input:", encoded_input)
            target = reverse_words((encoded_input,))[0]
            print("Target: ", target)
            numpy_inputs = numpy.repeat(numpy.array(encoded_input)[:, None],
                                        batch_size, axis=1)
            outputs, masks, probs = beam_search.search(
                OrderedDict([('features', numpy_inputs),
                             ('features_mask',
                              numpy.ones_like(numpy_inputs))]),
                char2code['</S>'])

            messages = []
            for i in range(outputs.shape[1]):
                sample = list(outputs[:, i])
                try:
                    true_length = sample.index(char2code['</S>']) + 1
                except ValueError:
                    true_length = len(sample)
                sample = sample[:true_length]
                prob = probs[i]
                message = "({})".format(prob)
                message += "".join(code2char[code] for code in sample)
                if sample == target:
                    message += " CORRECT!"
                messages.append((prob, message))
            messages.sort(key=operator.itemgetter(0), reverse=True)
            for _, message in messages:
                print(message)
