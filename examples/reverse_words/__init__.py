from __future__ import print_function
import logging
import pprint
import math
import numpy
import os
import operator

import theano
from six.moves import input
from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
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
from blocks.extensions.plot import PlotManager, Plotter
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


class WordReverser(Initializable):
    """The top brick.

    It is often convenient to gather all bricks of the model under the
    roof of a single top brick.

    """
    def __init__(self, dimension, alphabet_size, **kwargs):
        super(WordReverser, self).__init__(**kwargs)
        encoder = Bidirectional(
            SimpleRecurrent(dim=dimension, activation=Tanh()))
        fork = Fork([name for name in encoder.prototype.apply.sequences
                    if name != 'mask'])
        fork.input_dim = dimension
        fork.output_dims = {name: dimension for name in fork.input_names}
        lookup = LookupTable(alphabet_size, dimension)
        transition = SimpleRecurrent(
            activation=Tanh(),
            dim=dimension, name="transition")
        attention = SequenceContentAttention(
            state_names=transition.apply.states,
            attended_dim=2 * dimension, match_dim=dimension, name="attention")
        readout = LinearReadout(
            readout_dim=alphabet_size,
            source_names=[transition.apply.states[0],
                          attention.take_glimpses.outputs[0]],
            emitter=SoftmaxEmitter(name="emitter"),
            feedback_brick=LookupFeedback(alphabet_size, dimension),
            name="readout")
        generator = SequenceGenerator(
            readout=readout, transition=transition, attention=attention,
            name="generator")

        self.lookup = lookup
        self.fork = fork
        self.encoder = encoder
        self.generator = generator
        self.children = [lookup, fork, encoder, generator]

    @application
    def cost(self, chars, chars_mask, targets, targets_mask):
        return self.generator.cost(
            targets, targets_mask,
            attended=self.encoder.apply(
                **dict_union(
                    self.fork.apply(self.lookup.apply(chars), as_dict=True),
                    mask=chars_mask)),
            attended_mask=chars_mask)

    @application
    def generate(self, chars):
        return self.generator.generate(
            n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
            attended=self.encoder.apply(
                **dict_union(self.fork.apply(self.lookup.apply(chars),
                             as_dict=True))),
            attended_mask=tensor.ones(chars.shape))


def main(mode, save_path, num_batches, data_path=None):
    reverser = WordReverser(100, len(char2code), name="reverser")

    if mode == "train":
        # Data processing pipeline
        dataset_options = dict(dictionary=char2code, level="character",
                               preprocess=_lower)
        if data_path:
            dataset = TextFile(data_path, **dataset_options)
        else:
            dataset = OneBillionWord("training", [99], **dataset_options)
        data_stream = dataset.get_example_stream()
        data_stream = Filter(data_stream, _filter_long)
        data_stream = Mapping(data_stream, reverse_words,
                              add_sources=("targets",))
        data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10))
        data_stream = Padding(data_stream)
        data_stream = Mapping(data_stream, _transpose)

        # Initialization settings
        reverser.weights_init = IsotropicGaussian(0.1)
        reverser.biases_init = Constant(0.0)
        reverser.push_initialization_config()
        reverser.encoder.weghts_init = Orthogonal()
        reverser.generator.transition.weights_init = Orthogonal()

        # Build the cost computation graph
        chars = tensor.lmatrix("features")
        chars_mask = tensor.matrix("features_mask")
        targets = tensor.lmatrix("targets")
        targets_mask = tensor.matrix("targets_mask")
        batch_cost = reverser.cost(
            chars, chars_mask, targets, targets_mask).sum()
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

        # Define the training algorithm.
        cg = ComputationGraph(cost)
        algorithm = GradientDescent(
            cost=cost, params=cg.parameters,
            step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

        # Fetch variables useful for debugging
        generator = reverser.generator
        (energies,) = VariableFilter(
            application=generator.readout.readout,
            name="output")(cg.variables)
        (activations,) = VariableFilter(
            application=generator.transition.apply,
            name=generator.transition.apply.states[0])(cg.variables)
        max_length = named_copy(chars.shape[0], "max_length")
        cost_per_character = named_copy(
            aggregation.mean(batch_cost, batch_size * max_length),
            "character_log_likelihood")
        min_energy = named_copy(energies.min(), "min_energy")
        max_energy = named_copy(energies.max(), "max_energy")
        mean_activation = named_copy(abs(activations).mean(),
                                     "mean_activation")
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
                # This shows a way to handle NaN emerging during
                # training: simply finish it.
                .add_condition("after_batch", _is_nan),
                PlotManager(os.path.basename(save_path),
                            Plotter(
                            [[average_monitoring.record_name(cost)],
                             [average_monitoring.record_name(
                                 cost_per_character)]]),
                            every_n_batches=10),
                # Saving the model and the log separately is convenient,
                # because loading the whole pickle takes quite some time.
                SerializeMainLoop(save_path, every_n_batches=500,
                                  save_separately=["model", "log"]),
                Printing(every_n_batches=1)])
        main_loop.run()
    elif mode == "sample" or mode == "beam_search":
        chars = tensor.lmatrix("input")
        generated = reverser.generate(chars)
        model = Model(generated)
        logger.info("Loading the model..")
        model.set_param_values(load_parameter_values(save_path))

        def generate(input_):
            """Generate output sequences for an input sequence.

            Incapsulates most of the difference between sampling and beam
            search.

            Returns
            -------
            outputs : list of lists
                Trimmed output sequences.
            costs : list
                The negative log-likelihood of generating the respective
                sequences.

            """
            if mode == "beam_search":
                samples, = VariableFilter(
                    bricks=[reverser.generator], name="outputs")(
                        ComputationGraph(generated[1]))
                # NOTE: this will recompile beam search functions
                # every time user presses Enter. Do not create
                # a new `BeamSearch` object every time if
                # speed is important for you.
                beam_search = BeamSearch(input_.shape[1], samples)
                outputs, costs = beam_search.search(
                    {chars: input_}, char2code['</S>'],
                    3 * input_.shape[0])
            else:
                _1, outputs, _2, _3, costs = (
                    model.get_theano_function()(input_))
                outputs = list(outputs.T)
                costs = list(costs.T)
                for i in range(len(outputs)):
                    outputs[i] = list(outputs[i])
                    try:
                        true_length = outputs[i].index(char2code['</S>']) + 1
                    except ValueError:
                        true_length = len(outputs[i])
                    outputs[i] = outputs[i][:true_length]
                    costs[i] = costs[i][:true_length].sum()
            return outputs, costs

        while True:
            line = input("Enter a sentence\n")
            message = ("Enter the number of samples\n" if mode == "sample"
                       else "Enter the beam size\n")
            batch_size = int(input(message))

            encoded_input = [char2code.get(char, char2code["<UNK>"])
                             for char in line.lower().strip()]
            encoded_input = ([char2code['<S>']] + encoded_input +
                             [char2code['</S>']])
            print("Encoder input:", encoded_input)
            target = reverse_words((encoded_input,))[0]
            print("Target: ", target)

            samples, costs = generate(
                numpy.repeat(numpy.array(encoded_input)[:, None],
                             batch_size, axis=1))
            messages = []
            for sample, cost in zip(samples, costs):
                message = "({})".format(cost)
                message += "".join(code2char[code] for code in sample)
                if sample == target:
                    message += " CORRECT!"
                messages.append((cost, message))
            messages.sort(key=operator.itemgetter(0), reverse=True)
            for _, message in messages:
                print(message)
