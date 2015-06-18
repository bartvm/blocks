import numpy
from numpy.testing import assert_allclose

import theano
from theano import tensor

from blocks.bricks import Identity
from blocks.bricks.attention import (
    SequenceContentAttention, AttentionRecurrent)
from blocks.bricks.recurrent import SimpleRecurrent
from blocks.initialization import IsotropicGaussian, Constant
from blocks.graph import ComputationGraph
from blocks.select import Selector


def test_sequence_content_attention():
    # Disclaimer: only check dimensions, not values
    rng = numpy.random.RandomState([2014, 12, 2])

    seq_len = 5
    batch_size = 6
    state_dim = 2
    attended_dim = 3
    match_dim = 4

    attention = SequenceContentAttention(
        state_names=["states"], state_dims=[state_dim],
        attended_dim=attended_dim, match_dim=match_dim,
        weights_init=IsotropicGaussian(0.5),
        biases_init=Constant(0))
    attention.initialize()

    sequences = tensor.tensor3('sequences')
    states = tensor.matrix('states')
    mask = tensor.matrix('mask')
    glimpses, weights = attention.take_glimpses(
        sequences, attended_mask=mask, states=states)
    assert glimpses.ndim == 2
    assert weights.ndim == 2

    seq_values = numpy.zeros((seq_len, batch_size, attended_dim),
                             dtype=theano.config.floatX)
    states_values = numpy.zeros((batch_size, state_dim),
                                dtype=theano.config.floatX)
    mask_values = numpy.zeros((seq_len, batch_size),
                              dtype=theano.config.floatX)
    # randomly generate a sensible mask
    for sed_idx in range(batch_size):
        mask_values[:rng.randint(1, seq_len), sed_idx] = 1
    glimpses_values, weight_values = theano.function(
        [sequences, states, mask], [glimpses, weights])(
            seq_values, states_values, mask_values)
    assert glimpses_values.shape == (batch_size, attended_dim)
    assert weight_values.shape == (batch_size, seq_len)
    assert numpy.all(weight_values >= 0)
    assert numpy.all(weight_values <= 1)
    assert numpy.all(weight_values.sum(axis=1) == 1)
    assert numpy.all((weight_values.T == 0) == (mask_values == 0))


def test_attention_recurrent():
    rng = numpy.random.RandomState(1234)

    dim = 5
    batch_size = 4
    input_length = 20

    attended_dim = 10
    attended_length = 15

    wrapped = SimpleRecurrent(dim, Identity())
    attention = SequenceContentAttention(
        state_names=wrapped.apply.states,
        attended_dim=attended_dim, match_dim=attended_dim)
    recurrent = AttentionRecurrent(wrapped, attention, seed=1234)
    recurrent.weights_init = IsotropicGaussian(0.5)
    recurrent.biases_init = Constant(0)
    recurrent.initialize()

    attended = tensor.tensor3("attended")
    attended_mask = tensor.matrix("attended_mask")
    inputs = tensor.tensor3("inputs")
    inputs_mask = tensor.matrix("inputs_mask")
    outputs = recurrent.apply(
        inputs=inputs, mask=inputs_mask,
        attended=attended, attended_mask=attended_mask)
    states, glimpses, weights = outputs
    assert states.ndim == 3
    assert glimpses.ndim == 3
    assert weights.ndim == 3

    # For values.
    def rand(size):
        return rng.uniform(size=size).astype(theano.config.floatX)

    # For masks.
    def generate_mask(length, batch_size):
        mask = numpy.ones((length, batch_size), dtype=theano.config.floatX)
        # To make it look like read data
        for i in range(batch_size):
            mask[1 + rng.randint(0, length - 1):, i] = 0.0
        return mask

    input_vals = rand((input_length, batch_size, dim))
    input_mask_vals = generate_mask(input_length, batch_size)
    attended_vals = rand((attended_length, batch_size, attended_dim))
    attended_mask_vals = generate_mask(attended_length, batch_size)

    func = theano.function([inputs, inputs_mask, attended, attended_mask],
                           [states, glimpses, weights])
    states_vals, glimpses_vals, weight_vals = func(
        input_vals, input_mask_vals,
        attended_vals, attended_mask_vals)
    assert states_vals.shape == (input_length, batch_size, dim)
    assert glimpses_vals.shape == (input_length, batch_size, attended_dim)

    assert (len(ComputationGraph(outputs).shared_variables) ==
            len(Selector(recurrent).get_parameters()))

    # weights for not masked position must be zero
    assert numpy.all(weight_vals * (1 - attended_mask_vals.T) == 0)
    # weights for masked positions must be non-zero
    assert numpy.all(abs(weight_vals + (1 - attended_mask_vals.T)) > 1e-5)
    # weights from different steps should be noticeably different
    assert (abs(weight_vals[0] - weight_vals[1])).sum() > 1e-2
    # weights for all state after the last masked position should be same
    for i in range(batch_size):
        last = int(input_mask_vals[:, i].sum())
        for j in range(last, input_length):
            assert_allclose(weight_vals[last, i], weight_vals[j, i], 1e-5)

    # freeze sums
    assert_allclose(weight_vals.sum(), input_length * batch_size, 1e-5)
    assert_allclose(states_vals.sum(), 113.429, rtol=1e-5)
    assert_allclose(glimpses_vals.sum(), 415.901, rtol=1e-5)


def test_compute_weights_with_zero_mask():
    state_dim = 2
    attended_dim = 3
    match_dim = 4
    attended_length = 5
    batch_size = 6

    attention = SequenceContentAttention(
        state_names=["states"], state_dims=[state_dim],
        attended_dim=attended_dim, match_dim=match_dim,
        weights_init=IsotropicGaussian(0.5),
        biases_init=Constant(0))
    attention.initialize()

    energies = tensor.as_tensor_variable(
        numpy.random.rand(attended_length, batch_size))
    mask = tensor.as_tensor_variable(
        numpy.zeros((attended_length, batch_size)))
    weights = attention.compute_weights(energies, mask).eval()
    assert numpy.all(numpy.isfinite(weights))


def test_stable_attention_weights():
    state_dim = 2
    attended_dim = 3
    match_dim = 4
    attended_length = 5
    batch_size = 6

    attention = SequenceContentAttention(
        state_names=["states"], state_dims=[state_dim],
        attended_dim=attended_dim, match_dim=match_dim,
        weights_init=IsotropicGaussian(0.5),
        biases_init=Constant(0))
    attention.initialize()

    # Random high energies with mu=800, sigma=50
    energies_val = (
        50. * numpy.random.randn(attended_length, batch_size) + 800
        ).astype(theano.config.floatX)
    energies = tensor.as_tensor_variable(energies_val)
    mask = tensor.as_tensor_variable(
        numpy.ones((attended_length, batch_size)))
    weights = attention.compute_weights(energies, mask).eval()
    assert numpy.all(numpy.isfinite(weights))
