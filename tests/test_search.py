from collections import OrderedDict

import numpy

import theano
from theano import tensor

from blocks.bricks import Tanh
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.lookup import LookupTable
from blocks.bricks.parallel import Fork
from blocks.bricks.recurrent import Bidirectional, SimpleRecurrent
from blocks.bricks.sequence_generators import (SequenceGenerator,
                                               LinearReadout, SoftmaxEmitter,
                                               LookupFeedback)
from blocks.graph import ComputationGraph
from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.search import BeamSearch
from blocks.utils import dict_union

floatX = theano.config.floatX


def test_top_probs():
    """Test top probabilities."""
    a = numpy.array([[[3, 6, 4], [1, 2, 7]]])
    ind, maxs = BeamSearch._top_probs(a, 2)
    assert numpy.all(numpy.array(ind) == (numpy.array([[1, 0], [2, 1]])))
    assert numpy.all(maxs == [7, 6])


def test_beam_search():
    # Checks only dimension

    dimension = 15
    readout_dimension = 6

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

    beam_size = 9
    beam_search = BeamSearch(beam_size, generator, cg)
    beam_search.compile()
    sequence_length = 11
    outputs, masks, probs = beam_search.search(
        OrderedDict([('features', numpy.zeros((sequence_length, 1),
                                              dtype='int64')),
                     ('features_mask', numpy.ones((sequence_length, 1),
                                                  dtype=floatX))]),
        0)
    assert outputs.shape[0] <= 512
    assert masks.shape[0] <= 512
    assert outputs.shape[1] == beam_size
    assert masks.shape[1] == beam_size
    assert probs.shape[0] == beam_size
